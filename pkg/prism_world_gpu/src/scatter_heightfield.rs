//! Exact heightfield selection over GPU source/author-field records.

use prism_world_residency::{
    estimate_scatter_work, ScatterConfig, ScatterDensityField, ScatterFieldError, ScatterRegion,
};
use prism_world_terrain::{
    HeightfieldQueryError, HeightfieldScatterBudget, HeightfieldScatterCacheKey,
    HeightfieldScatterError, HeightfieldScatterFilters, PlanarHeightfieldSurface,
    HEIGHTFIELD_SCATTER_FILTER_VERSION, HEIGHTFIELD_SURFACE_VERSION,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry};
use crate::scatter::{check, dispatch};
use crate::{GpuContext, GpuScatterError, GpuScatterField, GpuScatterFieldError};

const PARAM_BYTES: u64 = 128;
const _: () = assert!(HEIGHTFIELD_SURFACE_VERSION == 1 && HEIGHTFIELD_SCATTER_FILTER_VERSION == 1);

/// Recoverable preflight errors enqueue no writes and retain the old request.
/// Device loss and OOM follow wgpu's separate error handling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuScatterHeightfieldError {
    /// Source/author-field validation or device/buffer limit.
    Field(GpuScatterFieldError),
    /// Exact terrain filters, coverage or conservative sample-read budget.
    Terrain(HeightfieldScatterError),
    /// The entire immutable snapshot must fit; samples are never truncated.
    SurfaceCapacityExceeded {
        /// Actual source sample count.
        required: usize,
        /// Fixed configured sample capacity.
        capacity: usize,
    },
    /// Fallible reservation of the reusable i32 height upload scratch failed.
    AllocationFailed,
}

impl std::fmt::Display for GpuScatterHeightfieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Field(error) => write!(f, "GPU source/author field: {error}"),
            Self::Terrain(error) => write!(f, "GPU terrain selection: {error}"),
            Self::SurfaceCapacityExceeded { required, capacity } => {
                write!(
                    f,
                    "heightfield needs {required} samples, capacity {capacity}"
                )
            }
            Self::AllocationFailed => f.write_str("height upload scratch reservation failed"),
        }
    }
}

impl std::error::Error for GpuScatterHeightfieldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Field(error) => Some(error),
            Self::Terrain(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GpuScatterFieldError> for GpuScatterHeightfieldError {
    fn from(error: GpuScatterFieldError) -> Self {
        Self::Field(error)
    }
}

impl From<GpuScatterError> for GpuScatterHeightfieldError {
    fn from(error: GpuScatterError) -> Self {
        Self::Field(error.into())
    }
}

impl From<HeightfieldScatterError> for GpuScatterHeightfieldError {
    fn from(error: HeightfieldScatterError) -> Self {
        Self::Terrain(error)
    }
}

/// Fixed-capacity integer source generation, author field, then exact terrain
/// thinning. No production per-point CPU evaluation, projection or readback.
///
/// Positions/keys remain unmodified; source minimum-distance and arbitrary XYZ
/// request-partition invariance are retained. Six storage bindings implement
/// signed wide arithmetic with core u32 limbs, without float world positions
/// or shader-i64. Each field survivor probes the real four-corner height data.
///
/// Source records keep their original 64-byte ABI and source accepted flag.
/// This object's 16-byte attributes contain density/species/final accepted/rank
/// per primary slot. Only its own compact/indirect output is final selection;
/// consuming source or intermediate field accepted flags bypasses terrain.
/// Compact order is unspecified; only the indirect-count prefix is defined.
///
/// Submit the current encoder before updating the next request. Sequential
/// submissions need no host wait. Encode allocates/uploads/maps nothing owned;
/// driver internal storage, retained Arc snapshots and metadata are separate.
pub struct GpuScatterHeightfield {
    field: GpuScatterField,
    queue: wgpu::Queue,
    max_surface_samples: usize,
    max_groups: u32,
    allocated_bytes: u64,
    params: wgpu::Buffer,
    heights: wgpu::Buffer,
    attributes: wgpu::Buffer,
    compact: wgpu::Buffer,
    indirect: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    height_scratch: Vec<i32>,
    request: Option<Request>,
}

struct Request {
    words: [u32; 32],
    groups: [u32; 2],
    key: HeightfieldScatterCacheKey,
}

impl GpuScatterHeightfield {
    /// Validates the complete resource set before any allocation. Owned logical
    /// buffers total `108*max(capacity,1) + 64*max(max_brushes,1) +
    /// 4*max(max_surface_samples,1) + 496` bytes. Encoded i16 samples upload as
    /// i32, using separate preallocated host scratch (four bytes per capacity).
    /// The shared immutable source snapshot is retained but not duplicated.
    pub fn new(
        ctx: &GpuContext,
        capacity: usize,
        max_brushes: usize,
        max_surface_samples: usize,
        vertex_count: u32,
        max_buffer_bytes: u64,
    ) -> Result<Self, GpuScatterHeightfieldError> {
        Self::create(
            ctx,
            capacity,
            max_brushes,
            max_surface_samples,
            vertex_count,
            max_buffer_bytes,
            false,
        )
    }

    /// Same terrain pipeline with the opt-in canonical-order brush hierarchy.
    /// Adds `64*(2*max_brushes-1)` bytes for nonzero brush capacity, in the
    /// existing field binding. Terrain work/semantics are unchanged.
    pub fn new_indexed(
        ctx: &GpuContext,
        capacity: usize,
        max_brushes: usize,
        max_surface_samples: usize,
        vertex_count: u32,
        max_buffer_bytes: u64,
    ) -> Result<Self, GpuScatterHeightfieldError> {
        Self::create(
            ctx,
            capacity,
            max_brushes,
            max_surface_samples,
            vertex_count,
            max_buffer_bytes,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create(
        ctx: &GpuContext,
        capacity: usize,
        max_brushes: usize,
        max_surface_samples: usize,
        vertex_count: u32,
        max_buffer_bytes: u64,
        indexed: bool,
    ) -> Result<Self, GpuScatterHeightfieldError> {
        let device = ctx.device();
        let limits = device.limits();
        let allocated_bytes = if indexed {
            validate_with_index(
                capacity,
                max_brushes,
                max_surface_samples,
                max_buffer_bytes,
                &limits,
                true,
            )?
        } else {
            validate(
                capacity,
                max_brushes,
                max_surface_samples,
                max_buffer_bytes,
                &limits,
            )?
        };
        let mut height_scratch = Vec::new();
        height_scratch
            .try_reserve_exact(max_surface_samples)
            .map_err(|_| GpuScatterHeightfieldError::AllocationFailed)?;
        let field = if indexed {
            GpuScatterField::new_indexed(
                ctx,
                capacity,
                max_brushes,
                vertex_count,
                max_buffer_bytes,
            )?
        } else {
            GpuScatterField::new(ctx, capacity, max_brushes, vertex_count, max_buffer_bytes)?
        };
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
        let heights = storage(
            "world scatter height samples",
            4 * max_surface_samples.max(1) as u64,
            true,
        );
        let attributes = storage(
            "world scatter terrain attributes",
            16 * capacity.max(1) as u64,
            false,
        );
        let compact = storage(
            "world scatter terrain compact",
            4 * capacity.max(1) as u64,
            false,
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world scatter terrain request"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("world scatter terrain indirect"),
            contents: bytemuck::cast_slice(&[vertex_count, 0_u32, 0, 0]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world scatter terrain layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world scatter terrain pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world scatter exact integer terrain shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/scatter_heightfield.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("world scatter terrain compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("terrain_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world scatter terrain bindings"),
            layout: &layout,
            entries: &[
                buffer_binding(0, &params),
                buffer_binding(1, field.records()),
                buffer_binding(2, field.attributes()),
                buffer_binding(3, &heights),
                buffer_binding(4, &attributes),
                buffer_binding(5, &compact),
                buffer_binding(6, &indirect),
            ],
        });
        Ok(Self {
            field,
            queue: ctx.queue().clone(),
            max_surface_samples,
            max_groups: limits.max_compute_workgroups_per_dimension,
            allocated_bytes,
            params,
            heights,
            attributes,
            compact,
            indirect,
            pipeline,
            bind_group,
            height_scratch,
            request: None,
        })
    }

    /// Admits the full snapshot, filters, XZ coverage and conservative
    /// `4*primary_count` sample reads before source/field writes. Then the
    /// composed field validates all its own limits before any write. Every
    /// recoverable failure preserves the complete previous composite request.
    ///
    /// Uploads changed source/field data as documented by `GpuScatterField`,
    /// plus changed terrain uniform (128 bytes) and changed sample payload
    /// (4 bytes per encoded sample). Changes only to bounds/filters/embedding/
    /// height scale never retransmit equal samples. Structural comparisons are
    /// worst-case O(terrain samples + brushes), not constant-time identity.
    #[allow(clippy::too_many_arguments)]
    pub fn write_request(
        &mut self,
        config: ScatterConfig,
        region: ScatterRegion,
        max_candidate_evaluations: u64,
        field: &ScatterDensityField,
        max_field_work: u64,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldScatterFilters,
        surface_budget: HeightfieldScatterBudget,
    ) -> Result<(), GpuScatterHeightfieldError> {
        let (words, groups) = self.preflight_request(
            config,
            region,
            max_candidate_evaluations,
            field,
            max_field_work,
            surface,
            filters,
            surface_budget,
        )?;
        // Field preflight itself queues nothing until all its checks pass.
        // No fallible operation follows this call: the scratch was reserved.
        self.field.write_request(
            config,
            region,
            max_candidate_evaluations,
            field,
            max_field_work,
        )?;
        if self.request.as_ref().is_none_or(|old| old.words != words) {
            self.queue
                .write_buffer(&self.params, 0, bytemuck::cast_slice(&words));
        }
        if self
            .request
            .as_ref()
            .is_none_or(|old| old.key.surface.tile().samples() != surface.tile().samples())
        {
            self.height_scratch.clear();
            self.height_scratch
                .extend(surface.tile().samples().iter().copied().map(i32::from));
            self.queue
                .write_buffer(&self.heights, 0, bytemuck::cast_slice(&self.height_scratch));
        }
        self.request = Some(Request {
            words,
            groups,
            key: HeightfieldScatterCacheKey {
                source: self
                    .field
                    .cache_key()
                    .expect("successfully uploaded field request")
                    .clone(),
                surface_version: HEIGHTFIELD_SURFACE_VERSION,
                filter_version: HEIGHTFIELD_SCATTER_FILTER_VERSION,
                surface: surface.clone(),
                filters,
            },
        });
        Ok(())
    }

    /// Pure full-chain preflight for sibling stages; never uploads or mutates.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate_request(
        &self,
        config: ScatterConfig,
        region: ScatterRegion,
        max_candidate_evaluations: u64,
        field: &ScatterDensityField,
        max_field_work: u64,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldScatterFilters,
        surface_budget: HeightfieldScatterBudget,
    ) -> Result<(), GpuScatterHeightfieldError> {
        self.preflight_request(
            config,
            region,
            max_candidate_evaluations,
            field,
            max_field_work,
            surface,
            filters,
            surface_budget,
        )?;
        Ok(())
    }

    // Same complete preflight for both validation and upload; no request state
    // is touched until the entire source/field/surface chain has been admitted.
    #[allow(clippy::too_many_arguments)]
    fn preflight_request(
        &self,
        config: ScatterConfig,
        region: ScatterRegion,
        max_candidate_evaluations: u64,
        field: &ScatterDensityField,
        max_field_work: u64,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldScatterFilters,
        surface_budget: HeightfieldScatterBudget,
    ) -> Result<([u32; 32], [u32; 2]), GpuScatterHeightfieldError> {
        let count = validate_terrain(
            config,
            region,
            surface,
            filters,
            surface_budget,
            self.max_surface_samples,
        )?;
        let groups = dispatch(count, self.max_groups)?;
        let count = u32::try_from(count).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
        self.field.validate_request(
            config,
            region,
            max_candidate_evaluations,
            field,
            max_field_work,
        )?;
        Ok((request_words(count, groups[0], surface, filters), groups))
    }

    /// Records all three stages and clears only the final count before its
    /// compact pass. No owned allocation/upload/submit/wait/readback occurs.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuScatterHeightfieldError> {
        let request = self
            .request
            .as_ref()
            .ok_or(GpuScatterError::RequestNotUploaded)?;
        self.field.encode(encoder)?;
        encoder.clear_buffer(&self.indirect, 4, Some(4));
        if request.words[0] != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("world scatter exact heightfield filter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(request.groups[0], request.groups[1], 1);
        }
        Ok(())
    }

    /// Primary source-slot prefix; accepted count stays on the GPU.
    pub fn candidate_count(&self) -> u32 {
        self.field.candidate_count()
    }
    /// Full structural identity including the retained immutable terrain.
    pub fn cache_key(&self) -> Option<&HeightfieldScatterCacheKey> {
        self.request.as_ref().map(|r| &r.key)
    }
    /// Original, unmodified 64-byte source records (not final accepted flags).
    pub fn records(&self) -> &wgpu::Buffer {
        self.field.records()
    }
    /// Source-slot-aligned density/species/final-accepted/rank records.
    pub fn attributes(&self) -> &wgpu::Buffer {
        &self.attributes
    }
    /// Unique final accepted slots in unspecified order; count is indirect[1].
    pub fn compact_indices(&self) -> &wgpu::Buffer {
        &self.compact
    }
    /// Final `[vertex_count, terrain_accepted_count, 0, 0]` draw arguments.
    pub fn indirect_args(&self) -> &wgpu::Buffer {
        &self.indirect
    }
    /// All owned logical GPU buffers, including source and field resources.
    pub fn allocated_buffer_bytes(&self) -> u64 {
        self.allocated_bytes
    }
    /// Actual reusable height-upload scratch capacity, not allocator/RSS usage.
    pub fn height_scratch_bytes(&self) -> usize {
        self.height_scratch.capacity() * size_of::<i32>()
    }
    /// Actual reusable brush/node-upload scratch capacity.
    pub fn brush_scratch_bytes(&self) -> usize {
        self.field.brush_scratch_bytes()
    }

    /// Whether the author-field stage uses the optional brush hierarchy.
    pub fn brush_index_enabled(&self) -> bool {
        self.field.brush_index_enabled()
    }

    /// Conservative author-field work per candidate, including hierarchy visits.
    pub fn max_sample_work(
        &self,
        field: &ScatterDensityField,
    ) -> Result<u64, GpuScatterFieldError> {
        self.field.max_sample_work(field)
    }

    /// Fixed primary source-slot capacity.
    pub fn capacity(&self) -> usize {
        self.field.capacity()
    }

    pub(super) fn device_identity(&self) -> &std::sync::Arc<()> {
        self.field.device_identity()
    }

    pub(super) fn draw_args(&self) -> [u32; 3] {
        self.field.draw_args()
    }

    pub(super) fn request_generation(&self) -> u64 {
        self.field.request_generation()
    }
}

fn validate(
    capacity: usize,
    brushes: usize,
    samples: usize,
    budget: u64,
    limits: &wgpu::Limits,
) -> Result<u64, GpuScatterHeightfieldError> {
    validate_with_index(capacity, brushes, samples, budget, limits, false)
}

fn validate_with_index(
    capacity: usize,
    brushes: usize,
    samples: usize,
    budget: u64,
    limits: &wgpu::Limits,
    indexed: bool,
) -> Result<u64, GpuScatterHeightfieldError> {
    let field_bytes = if indexed {
        crate::scatter_field::validate_with_index(capacity, brushes, budget, limits, true)?
    } else {
        crate::scatter_field::validate(capacity, brushes, budget, limits)?
    };
    let sample_count = u32::try_from(samples).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let height_bytes = 4 * u64::from(sample_count.max(1));
    let bytes = field_bytes + 20 * capacity.max(1) as u64 + height_bytes + PARAM_BYTES + 16;
    if bytes > budget {
        return Err(GpuScatterError::BufferBudgetExceeded {
            required: bytes,
            limit: budget,
        }
        .into());
    }
    check(
        "max_storage_buffer_binding_size",
        height_bytes,
        limits.max_storage_buffer_binding_size,
    )?;
    check("max_buffer_size", height_bytes, limits.max_buffer_size)?;
    check(
        "max_storage_buffers_per_shader_stage",
        6,
        u64::from(limits.max_storage_buffers_per_shader_stage),
    )?;
    check(
        "max_bindings_per_bind_group",
        7,
        u64::from(limits.max_bindings_per_bind_group),
    )?;
    Ok(bytes)
}

fn validate_terrain(
    config: ScatterConfig,
    region: ScatterRegion,
    surface: &PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
    budget: HeightfieldScatterBudget,
    capacity: usize,
) -> Result<u64, GpuScatterHeightfieldError> {
    if filters.surface_height_mm[0] > filters.surface_height_mm[1]
        || filters.clearance_mm[0] > filters.clearance_mm[1]
        || filters.max_slope_run == 0
    {
        return Err(HeightfieldScatterError::InvalidFilters.into());
    }
    let work = estimate_scatter_work(config, region)
        .map_err(|error| HeightfieldScatterError::Source(ScatterFieldError::from(error)))?;
    let required = surface.tile().samples().len();
    if required > capacity {
        return Err(GpuScatterHeightfieldError::SurfaceCapacityExceeded { required, capacity });
    }
    let empty = region
        .min_mm
        .into_iter()
        .zip(region.max_mm)
        .any(|(min, max)| min == max);
    if !empty {
        let min = surface.min_xz_mm();
        let max = surface.max_xz_mm();
        if region.min_mm[0] < min[0]
            || region.min_mm[2] < min[1]
            || region.max_mm[0] > max[0]
            || region.max_mm[2] > max[1]
        {
            return Err(
                HeightfieldScatterError::Surface(HeightfieldQueryError::OutsideSurface).into(),
            );
        }
    }
    let reads = work
        .primary_candidates
        .checked_mul(4)
        .ok_or(HeightfieldScatterError::SurfaceWorkOverflow)?;
    if reads > budget.max_sample_reads {
        return Err(HeightfieldScatterError::SurfaceWorkBudgetExceeded {
            required: reads,
            limit: budget.max_sample_reads,
        }
        .into());
    }
    Ok(work.primary_candidates)
}

fn request_words(
    count: u32,
    groups_x: u32,
    surface: &PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
) -> [u32; 32] {
    let embedding = surface.embedding();
    let spec = surface.tile().spec();
    let mut words = [0; 32];
    words[..4].copy_from_slice(&[
        count,
        groups_x,
        spec.samples_per_edge(),
        embedding.sample_spacing_mm,
    ]);
    for (index, value) in embedding
        .sample_origin_mm
        .into_iter()
        .chain([spec.height_scale_mm()])
        .chain(filters.surface_height_mm)
        .chain(filters.clearance_mm)
        .enumerate()
    {
        words[4 + index * 2] = value as u32;
        words[5 + index * 2] = (value as u64 >> 32) as u32;
    }
    words[20] = filters.max_slope_rise;
    words[21] = filters.max_slope_run;
    words
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_world_terrain::{
        BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec,
        HeightfieldTile, PlanarHeightfieldEmbedding, TileKey,
    };
    use std::sync::Arc;

    struct Flat;
    impl HeightSampler for Flat {
        type Error = core::convert::Infallible;
        fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            Ok(HeightSampleValue::from_millimetres(0))
        }
    }

    fn surface() -> PlanarHeightfieldSurface {
        let tile = HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveY),
            HeightfieldSpec::new(2, i64::MAX).unwrap(),
            &Flat,
            BakeLimits {
                max_samples: 4,
                max_bytes: 8,
            },
        )
        .unwrap();
        PlanarHeightfieldSurface::new(
            Arc::new(tile),
            PlanarHeightfieldEmbedding {
                sample_origin_mm: [i64::MIN, i64::MAX, i64::MIN],
                sample_spacing_mm: u32::MAX,
            },
        )
        .unwrap()
    }

    fn filters() -> HeightfieldScatterFilters {
        HeightfieldScatterFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            clearance_mm: [-1_234_567_890_123, i64::MAX],
            max_slope_rise: u32::MAX,
            max_slope_run: u32::MAX,
        }
    }

    fn config() -> ScatterConfig {
        ScatterConfig {
            seed: 0,
            version: 1,
            salt: 0,
            grid_spacing_mm: 1,
            min_distance_mm: 0,
        }
    }

    #[test]
    fn request_abi_preserves_signed_extremes_scale_ranges_and_reserved_zeroes() {
        let words = request_words(129, 3, &surface(), filters());
        assert_eq!(&words[..4], &[129, 3, 2, u32::MAX]);
        for (index, expected) in [
            i64::MIN,
            i64::MAX,
            i64::MIN,
            i64::MAX,
            i64::MIN,
            i64::MAX,
            -1_234_567_890_123,
            i64::MAX,
        ]
        .into_iter()
        .enumerate()
        {
            let actual = u64::from(words[4 + 2 * index]) | (u64::from(words[5 + 2 * index]) << 32);
            assert_eq!(actual as i64, expected);
        }
        assert_eq!(&words[20..22], &[u32::MAX; 2]);
        assert_eq!(&words[22..], &[0; 10]);
    }

    #[test]
    fn indexed_resource_preflight_includes_the_full_field_hierarchy() {
        let limits = wgpu::Limits::default();
        for (capacity, brushes, samples) in [(0, 0, 0), (1, 1, 4), (129, 17, 289)] {
            let nodes = u64::from(crate::scatter_brush_index::node_count(brushes).unwrap());
            let records = (brushes as u64 + nodes).max(1);
            let expected =
                108 * capacity.max(1) as u64 + 64 * records + 4 * samples.max(1) as u64 + 496;
            assert_eq!(
                validate_with_index(capacity, brushes, samples, expected, &limits, true),
                Ok(expected)
            );
            assert!(matches!(
                validate_with_index(capacity, brushes, samples, expected - 1, &limits, true),
                Err(GpuScatterHeightfieldError::Field(
                    GpuScatterFieldError::Scatter(GpuScatterError::BufferBudgetExceeded { .. })
                ))
            ));
        }
    }

    #[test]
    fn terrain_preflight_checks_full_coverage_empty_requests_and_read_overflow() {
        let surface = surface();
        let bounds = ScatterRegion::new([i64::MIN; 3], [i64::MIN + 1; 3]).unwrap();
        assert_eq!(
            validate_terrain(
                config(),
                bounds,
                &surface,
                filters(),
                HeightfieldScatterBudget {
                    max_sample_reads: 4
                },
                4
            ),
            Ok(1)
        );
        assert_eq!(
            validate_terrain(
                config(),
                bounds,
                &surface,
                filters(),
                HeightfieldScatterBudget {
                    max_sample_reads: 3
                },
                4
            ),
            Err(GpuScatterHeightfieldError::Terrain(
                HeightfieldScatterError::SurfaceWorkBudgetExceeded {
                    required: 4,
                    limit: 3
                }
            ))
        );
        assert_eq!(
            validate_terrain(
                config(),
                bounds,
                &surface,
                filters(),
                HeightfieldScatterBudget {
                    max_sample_reads: 4
                },
                3
            ),
            Err(GpuScatterHeightfieldError::SurfaceCapacityExceeded {
                required: 4,
                capacity: 3
            })
        );
        let outside = ScatterRegion::new([0; 3], [1; 3]).unwrap();
        assert_eq!(
            validate_terrain(
                config(),
                outside,
                &surface,
                filters(),
                HeightfieldScatterBudget {
                    max_sample_reads: 4
                },
                4
            ),
            Err(GpuScatterHeightfieldError::Terrain(
                HeightfieldScatterError::Surface(HeightfieldQueryError::OutsideSurface)
            ))
        );
        for axis in 0..3 {
            let mut max = [1; 3];
            max[axis] = 0;
            assert_eq!(
                validate_terrain(
                    config(),
                    ScatterRegion::new([0; 3], max).unwrap(),
                    &surface,
                    filters(),
                    HeightfieldScatterBudget {
                        max_sample_reads: 0
                    },
                    4
                ),
                Ok(0)
            );
        }
        let bounds = ScatterRegion::new([i64::MIN; 3], [i64::MIN + (1 << 21); 3]).unwrap();
        assert_eq!(
            validate_terrain(
                config(),
                bounds,
                &surface,
                filters(),
                HeightfieldScatterBudget {
                    max_sample_reads: u64::MAX
                },
                4
            ),
            Err(GpuScatterHeightfieldError::Terrain(
                HeightfieldScatterError::SurfaceWorkOverflow
            ))
        );
    }

    #[test]
    fn full_resource_preflight_covers_all_stages_and_zero_sentinels() {
        let limits = wgpu::Limits::default();
        for (capacity, brushes, samples) in
            [(0, 0, 0), (1, 1, 4), (129, 17, 1089), (100_000, 32, 4225)]
        {
            let expected = 108 * capacity.max(1) as u64
                + 64 * brushes.max(1) as u64
                + 4 * samples.max(1) as u64
                + 496;
            assert_eq!(
                validate(capacity, brushes, samples, expected, &limits),
                Ok(expected)
            );
            assert!(matches!(
                validate(capacity, brushes, samples, expected - 1, &limits),
                Err(GpuScatterHeightfieldError::Field(
                    GpuScatterFieldError::Scatter(GpuScatterError::BufferBudgetExceeded { .. })
                ))
            ));
        }
        let mut small = limits.clone();
        small.max_storage_buffers_per_shader_stage = 5;
        assert!(matches!(
            validate(1, 0, 4, 10_000, &small),
            Err(GpuScatterHeightfieldError::Field(
                GpuScatterFieldError::Scatter(GpuScatterError::DeviceLimit {
                    limit: "max_storage_buffers_per_shader_stage",
                    ..
                })
            ))
        ));
        small = limits.clone();
        small.max_bindings_per_bind_group = 6;
        assert!(matches!(
            validate(1, 0, 4, 10_000, &small),
            Err(GpuScatterHeightfieldError::Field(
                GpuScatterFieldError::Scatter(GpuScatterError::DeviceLimit {
                    limit: "max_bindings_per_bind_group",
                    ..
                })
            ))
        ));
        small = limits;
        small.max_storage_buffer_binding_size = 64;
        assert!(matches!(
            validate(1, 0, 17, 10_000, &small),
            Err(GpuScatterHeightfieldError::Field(
                GpuScatterFieldError::Scatter(GpuScatterError::DeviceLimit {
                    limit: "max_storage_buffer_binding_size",
                    ..
                })
            ))
        ));
        if usize::BITS > 32 {
            assert!(matches!(
                validate(1, 0, u32::MAX as usize + 1, u64::MAX, &small),
                Err(GpuScatterHeightfieldError::Field(
                    GpuScatterFieldError::Scatter(GpuScatterError::ArithmeticOverflow)
                ))
            ));
        }
    }
}
