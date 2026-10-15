//! Exact integer author-field filtering of GPU-generated scatter records.

use prism_world_residency::{
    estimate_scatter_work, DensityBrush, DensityBrushOp, FieldScatterCacheKey, ScatterCacheKey,
    ScatterConfig, ScatterDensityField, ScatterFieldError, ScatterRegion, SCATTER_FIELD_VERSION,
    SCATTER_KERNEL_VERSION,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry};
use crate::scatter::{check, dispatch, validate_capacity};
use crate::{GpuContext, GpuScatter, GpuScatterError};

const PARAM_BYTES: u64 = 128;
const BRUSH_BYTES: u64 = 64;
const _: () = assert!(SCATTER_FIELD_VERSION == 1 && SCATTER_KERNEL_VERSION == 1);

/// Composite preflight failures. Returned errors enqueue no request writes and
/// preserve the previous field and source request. Device loss/OOM use wgpu's
/// normal error handling, not this recoverable preflight contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuScatterFieldError {
    /// Source kernel validation, allocation limits or missing request.
    Scatter(GpuScatterError),
    /// Field arithmetic or logical work limit.
    Field(ScatterFieldError),
    /// Entire recipe must fit the fixed brush allocation; never truncate.
    BrushCapacityExceeded {
        /// Supplied validated recipe brush count.
        required: usize,
        /// Maximum configured brush count.
        capacity: usize,
    },
    /// Fallible reservation of the reusable host brush-upload scratch failed.
    AllocationFailed,
}

impl std::fmt::Display for GpuScatterFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scatter(error) => write!(f, "GPU source scatter: {error}"),
            Self::Field(error) => write!(f, "author field: {error}"),
            Self::BrushCapacityExceeded { required, capacity } => {
                write!(f, "field needs {required} brushes, capacity {capacity}")
            }
            Self::AllocationFailed => f.write_str("brush upload scratch reservation failed"),
        }
    }
}

impl std::error::Error for GpuScatterFieldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scatter(error) => Some(error),
            Self::Field(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GpuScatterError> for GpuScatterFieldError {
    fn from(error: GpuScatterError) -> Self {
        Self::Scatter(error)
    }
}

impl From<ScatterFieldError> for GpuScatterFieldError {
    fn from(error: ScatterFieldError) -> Self {
        Self::Field(error)
    }
}

/// Fixed-capacity GPU source generation followed by integer density/species
/// sampling and compaction. No production CPU per-point loop or readback.
///
/// Uses exactly field revision 1's noise/warp, squared brush falloff, stable
/// composition, signed rounding and independent candidate rank. Global i64
/// millimetres never pass through floats; warp extends beyond the world edges
/// mathematically, wrapping only the virtual lattice hash coordinate.
///
/// Source records keep [`GpuScatter`]'s 64-byte ABI and *source* accepted flag.
/// The aligned 16-byte attribute record is four u32 words: UNORM16 density,
/// species, field-accepted flag (0/1), selection rank. Source-rejected slots
/// have all-zero attributes. Field-rejected source survivors still have valid
/// density/species/rank. Only slots in this object's compact/indirect prefix
/// are final accepted samples; never consume the source accepted flag alone.
///
/// Submit the current encoder before updating/encoding the next request on
/// this object. Sequential submissions need no host wait. Buffers and host
/// brush-upload scratch are allocated only at construction; queue/driver
/// internal upload storage is outside the owned-scratch allocation contract.
pub struct GpuScatterField {
    source: GpuScatter,
    queue: wgpu::Queue,
    max_brushes: usize,
    indexed: bool,
    max_groups: u32,
    allocated_bytes: u64,
    params: wgpu::Buffer,
    brushes: wgpu::Buffer,
    attributes: wgpu::Buffer,
    compact: wgpu::Buffer,
    indirect: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    brush_scratch: Vec<[u32; 16]>,
    request: Option<Request>,
}

struct Request {
    words: [u32; 32],
    groups: [u32; 2],
    key: FieldScatterCacheKey,
}

impl GpuScatterField {
    /// Checks composite limits before GPU allocation. Owned logical buffers:
    /// `88*max(capacity,1) + 64*max(max_brushes,1) + 352` bytes, including the
    /// source kernel's resources. Not physical VRAM, metadata or RSS.
    /// Host scratch separately reserves `max_brushes` 64-byte brush records.
    pub fn new(
        ctx: &GpuContext,
        capacity: usize,
        max_brushes: usize,
        vertex_count: u32,
        max_buffer_bytes: u64,
    ) -> Result<Self, GpuScatterFieldError> {
        Self::create(
            ctx,
            capacity,
            max_brushes,
            vertex_count,
            max_buffer_bytes,
            false,
        )
    }

    /// Opt-in canonical-order AABB hierarchy. Keeps the same five storage
    /// bindings and exact field semantics as [`Self::new`]. For `B` brushes,
    /// the fixed upload/storage capacity adds `64*(2*B-1)` bytes if `B>0`.
    /// Each query has no scratch or sorting; spatially incoherent IDs and
    /// overlapping bounds can make this slower than the linear path.
    pub fn new_indexed(
        ctx: &GpuContext,
        capacity: usize,
        max_brushes: usize,
        vertex_count: u32,
        max_buffer_bytes: u64,
    ) -> Result<Self, GpuScatterFieldError> {
        Self::create(
            ctx,
            capacity,
            max_brushes,
            vertex_count,
            max_buffer_bytes,
            true,
        )
    }

    fn create(
        ctx: &GpuContext,
        capacity: usize,
        max_brushes: usize,
        vertex_count: u32,
        max_buffer_bytes: u64,
        indexed: bool,
    ) -> Result<Self, GpuScatterFieldError> {
        let device = ctx.device();
        let limits = device.limits();
        let bytes = if indexed {
            validate_with_index(capacity, max_brushes, max_buffer_bytes, &limits, true)?
        } else {
            validate(capacity, max_brushes, max_buffer_bytes, &limits)?
        };
        let records = brush_records(max_brushes, indexed)?;
        let mut brush_scratch = Vec::new();
        brush_scratch
            .try_reserve_exact(records as usize)
            .map_err(|_| GpuScatterFieldError::AllocationFailed)?;
        let source = GpuScatter::new(ctx, capacity, vertex_count, max_buffer_bytes)?;
        let storage = |label, size, upload| {
            let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            if upload {
                usage |= wgpu::BufferUsages::COPY_DST;
            }
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let brushes = storage(
            "world scatter author brushes",
            BRUSH_BYTES * u64::from(records.max(1)),
            true,
        );
        let attributes = storage(
            "world scatter field attributes",
            16 * capacity.max(1) as u64,
            false,
        );
        let compact = storage(
            "world scatter field compact",
            4 * capacity.max(1) as u64,
            false,
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world scatter author field request"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("world scatter field indirect"),
            contents: bytemuck::cast_slice(&[vertex_count, 0_u32, 0, 0]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world scatter field layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world scatter field pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world scatter author field integer shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scatter_field.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("world scatter field compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("field_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world scatter field bindings"),
            layout: &layout,
            entries: &[
                buffer_binding(0, &params),
                buffer_binding(1, source.records()),
                buffer_binding(2, &brushes),
                buffer_binding(3, &attributes),
                buffer_binding(4, &compact),
                buffer_binding(5, &indirect),
            ],
        });
        Ok(Self {
            source,
            queue: ctx.queue().clone(),
            max_brushes,
            indexed,
            max_groups: limits.max_compute_workgroups_per_dimension,
            allocated_bytes: bytes,
            params,
            brushes,
            attributes,
            compact,
            indirect,
            pipeline,
            bind_group,
            brush_scratch,
            request: None,
        })
    }

    /// Validates source/field work and complete brush capacity before any write.
    /// No CPU sample generation. Source changes upload 192 bytes, field uniform
    /// changes 128 bytes, changed brush lists 64 bytes per supplied brush plus
    /// 64 bytes per hierarchy node in indexed mode. Equal
    /// recipes/requests upload nothing; recipe comparison is worst-case O(B),
    /// not O(1). Source edits never require retransmitting unchanged brushes.
    pub fn write_request(
        &mut self,
        config: ScatterConfig,
        region: ScatterRegion,
        max_candidate_evaluations: u64,
        field: &ScatterDensityField,
        max_field_work: u64,
    ) -> Result<(), GpuScatterFieldError> {
        let (count, groups) = self.preflight_request(
            config,
            region,
            max_candidate_evaluations,
            field,
            max_field_work,
        )?;
        let mut words = [0_u32; 32];
        let fc = field.config();
        words[0] = fc.seed as u32;
        words[1] = (fc.seed >> 32) as u32;
        words[2] = u32::from(fc.base_density);
        words[3] = fc.base_species;
        if let Some(noise) = fc.noise {
            words[4] = u32::from(noise.amplitude != 0);
            words[5] = noise.cell_size_mm;
            words[6] = u32::from(noise.amplitude);
            words[7] = noise.warp_cell_size_mm;
            words[8] = noise.warp_amplitude_mm;
        }
        words[9] = field.brushes().len() as u32;
        words[10] = count;
        words[11] = groups[0];
        words[12] = config.seed as u32;
        words[13] = (config.seed >> 32) as u32;
        words[14] = config.version;
        words[15] = config.grid_spacing_mm;
        words[16] = config.salt as u32;
        words[17] = (config.salt >> 32) as u32;
        words[18] = SCATTER_KERNEL_VERSION;
        if self.indexed {
            words[19] = words[9];
            // The complete capacity was validated at construction/preflight.
            words[20] = crate::scatter_brush_index::node_count(field.brushes().len())?;
        }
        let key = FieldScatterCacheKey {
            scatter: ScatterCacheKey {
                kernel_version: SCATTER_KERNEL_VERSION,
                config,
                region,
            },
            field_version: SCATTER_FIELD_VERSION,
            field: field.clone(),
        };
        // No recoverable validation below this point can fail. All source
        // validation above ran before field/source uploads were enqueued.
        self.source
            .write_request(config, region, max_candidate_evaluations)?;
        if self.request.as_ref().is_none_or(|old| old.words != words) {
            self.queue
                .write_buffer(&self.params, 0, bytemuck::cast_slice(&words));
        }
        if self
            .request
            .as_ref()
            .is_none_or(|old| old.key.field.brushes() != field.brushes())
        {
            self.brush_scratch.clear();
            self.brush_scratch
                .extend(field.brushes().iter().map(brush_words));
            if self.indexed {
                crate::scatter_brush_index::append_nodes(field.brushes(), &mut self.brush_scratch);
            }
            if !self.brush_scratch.is_empty() {
                self.queue.write_buffer(
                    &self.brushes,
                    0,
                    bytemuck::cast_slice(&self.brush_scratch),
                );
            }
        }
        self.request = Some(Request { words, groups, key });
        Ok(())
    }

    /// Pure composite preflight for sibling stages; never uploads or mutates.
    pub(super) fn validate_request(
        &self,
        config: ScatterConfig,
        region: ScatterRegion,
        max_candidate_evaluations: u64,
        field: &ScatterDensityField,
        max_field_work: u64,
    ) -> Result<(), GpuScatterFieldError> {
        self.preflight_request(
            config,
            region,
            max_candidate_evaluations,
            field,
            max_field_work,
        )?;
        Ok(())
    }

    // Single pure admission path for validation and upload. Keep every
    // recoverable check here, before either stage can enqueue a queue write.
    fn preflight_request(
        &self,
        config: ScatterConfig,
        region: ScatterRegion,
        max_candidate_evaluations: u64,
        field: &ScatterDensityField,
        max_field_work: u64,
    ) -> Result<(u32, [u32; 2]), GpuScatterFieldError> {
        self.source
            .validate_request(config, region, max_candidate_evaluations)?;
        let sample_work = self.max_sample_work(field)?;
        let estimate = estimate_scatter_work(config, region).map_err(GpuScatterError::from)?;
        let required = estimate
            .primary_candidates
            .checked_mul(sample_work)
            .ok_or(ScatterFieldError::ArithmeticOverflow)?;
        if required > max_field_work {
            return Err(ScatterFieldError::WorkBudgetExceeded {
                required,
                limit: max_field_work,
            }
            .into());
        }
        let count = u32::try_from(estimate.primary_candidates)
            .map_err(|_| GpuScatterError::ArithmeticOverflow)?;
        let groups = dispatch(estimate.primary_candidates, self.max_groups)?;
        Ok((count, groups))
    }

    /// Whether this object uses the optional canonical-order brush hierarchy.
    pub fn brush_index_enabled(&self) -> bool {
        self.indexed
    }

    /// Conservative logical work per primary candidate, including every
    /// possible hierarchy visit. Checks brush capacity without mutating or
    /// uploading. Multiply by the primary candidate count for admission.
    pub fn max_sample_work(
        &self,
        field: &ScatterDensityField,
    ) -> Result<u64, GpuScatterFieldError> {
        if field.brushes().len() > self.max_brushes {
            return Err(GpuScatterFieldError::BrushCapacityExceeded {
                required: field.brushes().len(),
                capacity: self.max_brushes,
            });
        }
        let nodes = if self.indexed {
            crate::scatter_brush_index::node_count(field.brushes().len())?
        } else {
            0
        };
        field
            .work_per_sample()
            .checked_add(u64::from(nodes))
            .ok_or_else(|| ScatterFieldError::ArithmeticOverflow.into())
    }

    /// Records source generation then field filtering/compaction. No owned
    /// resource allocation, upload, submit, wait, mapping or readback.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), GpuScatterFieldError> {
        let request = self
            .request
            .as_ref()
            .ok_or(GpuScatterError::RequestNotUploaded)?;
        self.source.encode(encoder)?;
        encoder.clear_buffer(&self.indirect, 4, Some(4));
        if request.words[10] != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("world scatter author field"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(request.groups[0], request.groups[1], 1);
        }
        Ok(())
    }

    /// Primary source-slot prefix; accepted count remains on the GPU.
    pub fn candidate_count(&self) -> u32 {
        self.source.candidate_count()
    }
    /// Complete immutable source/field cache descriptor, not a digest identity.
    pub fn cache_key(&self) -> Option<&FieldScatterCacheKey> {
        self.request.as_ref().map(|r| &r.key)
    }
    /// Original source records, before field filtering. See struct-level ABI.
    pub fn records(&self) -> &wgpu::Buffer {
        self.source.records()
    }
    /// Source-slot-aligned density/species/accepted/rank u32 records.
    pub fn attributes(&self) -> &wgpu::Buffer {
        &self.attributes
    }
    /// Final accepted source slots, unique and in unspecified order.
    pub fn compact_indices(&self) -> &wgpu::Buffer {
        &self.compact
    }
    /// Final `[vertex_count, field_accepted_count, 0, 0]` indirect draw arguments.
    pub fn indirect_args(&self) -> &wgpu::Buffer {
        &self.indirect
    }
    /// Sum of all owned logical GPU buffers, including source resources.
    pub fn allocated_buffer_bytes(&self) -> u64 {
        self.allocated_bytes
    }
    /// Actual reusable host upload scratch payload capacity, including hierarchy
    /// records in indexed mode; excludes Vec metadata and allocator overhead.
    pub fn brush_scratch_bytes(&self) -> usize {
        self.brush_scratch.capacity() * 64
    }

    /// Fixed primary source-slot capacity.
    pub fn capacity(&self) -> usize {
        self.source.capacity()
    }

    pub(super) fn device_identity(&self) -> &std::sync::Arc<()> {
        self.source.device_identity()
    }

    pub(super) fn draw_args(&self) -> [u32; 3] {
        self.source.draw_args()
    }

    pub(super) fn request_generation(&self) -> u64 {
        self.source.request_generation()
    }
}

pub(crate) fn brush_words(brush: &DensityBrush) -> [u32; 16] {
    let mut words = [0; 16];
    words[0] = brush.id as u32;
    words[1] = (brush.id >> 32) as u32;
    for (axis, position) in brush.center_mm.into_iter().enumerate() {
        words[2 + axis * 2] = position as u32;
        words[3 + axis * 2] = (position as u64 >> 32) as u32;
    }
    words[8] = brush.inner_radius_mm;
    words[9] = brush.outer_radius_mm;
    words[10] = u32::from(brush.opacity);
    (words[11], words[12]) = match brush.density {
        DensityBrushOp::Replace(value) => (0, u32::from(value)),
        DensityBrushOp::Add(delta) => (1, delta as u32),
    };
    words[13] = u32::from(brush.species.is_some());
    words[14] = brush.species.unwrap_or(0);
    words
}

pub(crate) fn validate(
    capacity: usize,
    brushes: usize,
    budget: u64,
    limits: &wgpu::Limits,
) -> Result<u64, GpuScatterFieldError> {
    validate_with_index(capacity, brushes, budget, limits, false)
}

fn brush_records(brushes: usize, indexed: bool) -> Result<u32, GpuScatterError> {
    let count = u32::try_from(brushes).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let nodes = if indexed {
        crate::scatter_brush_index::node_count(brushes)?
    } else {
        0
    };
    count
        .checked_add(nodes)
        .ok_or(GpuScatterError::ArithmeticOverflow)
}

pub(crate) fn validate_with_index(
    capacity: usize,
    brushes: usize,
    budget: u64,
    limits: &wgpu::Limits,
    indexed: bool,
) -> Result<u64, GpuScatterFieldError> {
    let source_bytes = validate_capacity(capacity, budget, limits)?;
    let brush_count = brush_records(brushes, indexed)?;
    let brush_bytes = BRUSH_BYTES * u64::from(brush_count.max(1));
    let bytes = source_bytes + 20 * capacity.max(1) as u64 + brush_bytes + PARAM_BYTES + 16;
    if bytes > budget {
        return Err(GpuScatterError::BufferBudgetExceeded {
            required: bytes,
            limit: budget,
        }
        .into());
    }
    check(
        "max_storage_buffer_binding_size",
        brush_bytes,
        limits.max_storage_buffer_binding_size,
    )?;
    check("max_buffer_size", brush_bytes, limits.max_buffer_size)?;
    check(
        "max_storage_buffers_per_shader_stage",
        5,
        u64::from(limits.max_storage_buffers_per_shader_stage),
    )?;
    check(
        "max_bindings_per_bind_group",
        6,
        u64::from(limits.max_bindings_per_bind_group),
    )?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_budget_covers_source_field_brush_and_zero_sentinels() {
        let limits = wgpu::Limits::default();
        for (capacity, brushes) in [(0, 0), (1, 1), (129, 17), (10_000, 32)] {
            let expected = 88 * capacity.max(1) as u64 + 64 * brushes.max(1) as u64 + 352;
            assert_eq!(validate(capacity, brushes, expected, &limits), Ok(expected));
            assert!(matches!(
                validate(capacity, brushes, expected - 1, &limits),
                Err(GpuScatterFieldError::Scatter(
                    GpuScatterError::BufferBudgetExceeded { .. }
                ))
            ));
        }
        let mut small = limits.clone();
        small.max_storage_buffers_per_shader_stage = 4;
        assert!(matches!(
            validate(1, 1, 4096, &small),
            Err(GpuScatterFieldError::Scatter(
                GpuScatterError::DeviceLimit {
                    limit: "max_storage_buffers_per_shader_stage",
                    ..
                }
            ))
        ));
        small = limits;
        small.max_storage_buffer_binding_size = 64;
        assert!(matches!(
            validate(1, 2, 4096, &small),
            Err(GpuScatterFieldError::Scatter(
                GpuScatterError::DeviceLimit {
                    limit: "max_storage_buffer_binding_size",
                    ..
                }
            ))
        ));
    }

    #[test]
    fn indexed_composite_budget_and_address_space_include_every_node() {
        let limits = wgpu::Limits::default();
        for (capacity, brushes) in [(0, 0), (1, 1), (129, 17), (10_000, 32)] {
            let nodes = crate::scatter_brush_index::node_count(brushes).unwrap();
            let records = (brushes as u64 + u64::from(nodes)).max(1);
            let expected = 88 * capacity.max(1) as u64 + 64 * records + 352;
            assert_eq!(
                validate_with_index(capacity, brushes, expected, &limits, true),
                Ok(expected)
            );
            assert!(matches!(
                validate_with_index(capacity, brushes, expected - 1, &limits, true),
                Err(GpuScatterFieldError::Scatter(
                    GpuScatterError::BufferBudgetExceeded { .. }
                ))
            ));
        }
        // Combined brush/node addressing must fit, even if node count alone fits.
        let brushes = (u32::MAX as usize / 3) + 1;
        assert!(crate::scatter_brush_index::node_count(brushes).is_ok());
        assert!(matches!(
            brush_records(brushes, true),
            Err(GpuScatterError::ArithmeticOverflow)
        ));
        let mut small = limits;
        small.max_storage_buffer_binding_size = 127;
        assert!(matches!(
            validate_with_index(1, 1, 4096, &small, true),
            Err(GpuScatterFieldError::Scatter(
                GpuScatterError::DeviceLimit {
                    limit: "max_storage_buffer_binding_size",
                    ..
                }
            ))
        ));
    }

    #[test]
    fn brush_upload_abi_preserves_extreme_coordinates_ids_and_signed_delta() {
        let brush = DensityBrush {
            id: u64::MAX,
            center_mm: [i64::MIN, i64::MAX, -1],
            inner_radius_mm: 17,
            outer_radius_mm: u32::MAX,
            opacity: 32768,
            density: DensityBrushOp::Add(-65535),
            species: Some(u32::MAX),
        };
        let words = brush_words(&brush);
        assert_eq!(
            words,
            [
                u32::MAX,
                u32::MAX,
                0,
                0x8000_0000,
                u32::MAX,
                0x7fff_ffff,
                u32::MAX,
                u32::MAX,
                17,
                u32::MAX,
                32768,
                1,
                (-65535_i32) as u32,
                1,
                u32::MAX,
                0
            ]
        );
        assert_eq!(
            brush_words(&DensityBrush {
                density: DensityBrushOp::Replace(65535),
                species: None,
                ..brush
            })[11..16],
            [0, 65535, 0, 0, 0]
        );
    }
}
