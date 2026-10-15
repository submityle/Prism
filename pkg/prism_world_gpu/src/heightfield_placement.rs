//! Independent two-dimensional placement, exact projection, then author field.

use prism_world_residency::{ScatterDensityField, SCATTER_FIELD_VERSION, SCATTER_KERNEL_VERSION};
use prism_world_terrain::{
    estimate_heightfield_placement_work, HeightfieldPlacementCacheKey, HeightfieldPlacementConfig,
    HeightfieldPlacementError, HeightfieldPlacementFilters, HeightfieldPlacementRegion,
    HeightfieldQueryError, PlanarHeightfieldSurface, HEIGHTFIELD_PLACEMENT_VERSION,
    HEIGHTFIELD_SURFACE_VERSION,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry};
use crate::scatter::{check, dispatch, validate_capacity};
use crate::{GpuContext, GpuScatterError};

const _: () = assert!(
    HEIGHTFIELD_PLACEMENT_VERSION == 1
        && HEIGHTFIELD_SURFACE_VERSION == 1
        && SCATTER_KERNEL_VERSION == 1
        && SCATTER_FIELD_VERSION == 1
);

/// Conservative logical admission; not GPU instruction counts or a time bound.
/// Output capacity is fixed at construction and must hold every primary slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuHeightfieldPlacementWorkBudget {
    /// Primary and repeated halo probes, including potentially invalid probes.
    pub max_candidate_evaluations: u64,
    /// Four terrain reads per primary, before any rejection.
    pub max_sample_reads: u64,
    /// Primary count times the linear author-field work per sample.
    pub max_field_work: u64,
}

/// Recoverable host preflight failures enqueue no writes and retain the prior
/// request. Device/OOM errors follow wgpu; projection overflow is GPU status.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuHeightfieldPlacementError {
    /// GPU capacity, device limits, arithmetic or missing uploaded request.
    Gpu(GpuScatterError),
    /// CPU contract inputs, coverage or logical work budget.
    Placement(HeightfieldPlacementError),
    /// The full author recipe must fit, without truncation.
    BrushCapacityExceeded {
        /// Required immutable brush count.
        required: usize,
        /// Fixed allocated count.
        capacity: usize,
    },
    /// The entire heightfield snapshot must fit, without truncation.
    SurfaceCapacityExceeded {
        /// Required height sample count.
        required: usize,
        /// Fixed allocated sample count.
        capacity: usize,
    },
    /// Fallible construction-time host upload scratch reservation failed.
    AllocationFailed,
}

impl std::fmt::Display for GpuHeightfieldPlacementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "GPU placement: {e}"),
            Self::Placement(e) => write!(f, "placement contract: {e}"),
            Self::BrushCapacityExceeded { required, capacity } => {
                write!(f, "placement needs {required} brushes, capacity {capacity}")
            }
            Self::SurfaceCapacityExceeded { required, capacity } => {
                write!(
                    f,
                    "placement needs {required} height samples, capacity {capacity}"
                )
            }
            Self::AllocationFailed => f.write_str("placement upload scratch reservation failed"),
        }
    }
}

impl std::error::Error for GpuHeightfieldPlacementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Placement(e) => Some(e),
            _ => None,
        }
    }
}

impl From<GpuScatterError> for GpuHeightfieldPlacementError {
    fn from(e: GpuScatterError) -> Self {
        Self::Gpu(e)
    }
}
impl From<HeightfieldPlacementError> for GpuHeightfieldPlacementError {
    fn from(e: HeightfieldPlacementError) -> Self {
        Self::Placement(e)
    }
}

/// Fixed-capacity GPU twin of CPU planar heightfield placement revision 1.
///
/// Generates canonical grid-Y-zero XZ candidates, horizontal Matérn-II thinning,
/// exact triangle height/normal, whole (height + offset) rounding, unoffset
/// surface filters, then the existing integer author field at final XYZ.
/// No production per-point CPU sampling, survivor upload or readback is used.
/// Six storage bindings, core u32 limbs and no optional GPU features suffice.
///
/// Records are 64 bytes: grid XYZ i64 pairs (words 0..5), final XYZ i64 pairs
/// (6..11), pre-field terrain accepted (12), horizontal priority u64 (13..14),
/// zero (15). Full identity is placement version plus canonical source key,
/// including the request seed/version/salt/spacing; raw source Y is NOT rebuilt
/// geometry. This is a separate placement stream, not a `GpuScatter` upstream.
/// Samples are 64 bytes: signed i128 unoffset height numerator, then signed i128
/// normal X/Y/Z, least-significant u32 limb first; denominator is terrain spacing.
/// Samples are zero for source rejects; valid even for terrain-filter rejects.
/// Attributes are density/species/final accepted/author rank, four u32 per slot.
/// Final unique compact indices have unspecified group order, live only up to
/// indirect[1]; indirect = [vertex_count, accepted_count, 0, 0].
///
/// `status()[0]` is the count of projection overflows, checked BEFORE filters.
/// Nonzero invalidates the WHOLE request: finalization clears indirect count,
/// every live record acceptance and attribute acceptance. Positions/samples/
/// compact tails in that case are scratch, never a partial successful result.
/// A consumer that needs an error reason may explicitly read status; drawing
/// indirectly safely produces zero instances without a CPU wait or readback.
///
/// Submit each encoder before writing the next request. Separate sequential
/// submissions require no wait; encoding multiple requests before submission
/// on the same object is unsupported. Encode allocates/uploads/maps nothing
/// owned; driver allocations and retained immutable Arc snapshots are separate.
pub struct GpuHeightfieldPlacement {
    device_identity: std::sync::Arc<()>,
    queue: wgpu::Queue,
    capacity: usize,
    vertex_count: u32,
    request_generation: u64,
    max_brushes: usize,
    max_surface_samples: usize,
    max_groups: u32,
    allocated_bytes: u64,
    source_params: wgpu::Buffer,
    project_params: wgpu::Buffer,
    field_params: wgpu::Buffer,
    records: wgpu::Buffer,
    samples: wgpu::Buffer,
    attributes: wgpu::Buffer,
    compact: wgpu::Buffer,
    indirect: wgpu::Buffer,
    status: wgpu::Buffer,
    heights: wgpu::Buffer,
    brushes: wgpu::Buffer,
    source_pipeline: wgpu::ComputePipeline,
    project_pipeline: wgpu::ComputePipeline,
    field_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    source_bind: wgpu::BindGroup,
    project_bind: wgpu::BindGroup,
    field_bind: wgpu::BindGroup,
    height_scratch: Vec<i32>,
    brush_scratch: Vec<[u32; 16]>,
    request: Option<Request>,
}

struct Request {
    source: [u32; 48],
    project: [u32; 32],
    field: [u32; 32],
    groups: [u32; 2],
    key: HeightfieldPlacementCacheKey,
}

impl GpuHeightfieldPlacement {
    /// Checks the complete resource set before allocating. Logical GPU buffers
    /// total `148*max(capacity,1)+64*max(brushes,1)+4*max(samples,1)+468` bytes.
    /// Separate reusable host scratch has actual Vec capacities charged by the
    /// accessors; retained snapshots/allocator overhead are not buffer bytes.
    pub fn new(
        ctx: &GpuContext,
        capacity: usize,
        max_brushes: usize,
        max_surface_samples: usize,
        vertex_count: u32,
        max_buffer_bytes: u64,
    ) -> Result<Self, GpuHeightfieldPlacementError> {
        let device = ctx.device();
        let limits = device.limits();
        let allocated_bytes = validate(
            capacity,
            max_brushes,
            max_surface_samples,
            max_buffer_bytes,
            &limits,
        )?;
        let mut height_scratch = Vec::new();
        let mut brush_scratch = Vec::new();
        height_scratch
            .try_reserve_exact(max_surface_samples)
            .map_err(|_| GpuHeightfieldPlacementError::AllocationFailed)?;
        brush_scratch
            .try_reserve_exact(max_brushes)
            .map_err(|_| GpuHeightfieldPlacementError::AllocationFailed)?;
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let upload = wgpu::BufferUsages::COPY_DST;
        let uniform = wgpu::BufferUsages::UNIFORM | upload;
        let source_params = buffer("placement source params", 192, uniform);
        let project_params = buffer("placement projection params", 128, uniform);
        let field_params = buffer("placement field params", 128, uniform);
        let slots = capacity.max(1) as u64;
        let records = buffer("placement geometry", 64 * slots, storage);
        let samples = buffer("placement exact surface samples", 64 * slots, storage);
        let attributes = buffer("placement author attributes", 16 * slots, storage);
        let compact = buffer("placement compact indices", 4 * slots, storage);
        let heights = buffer(
            "placement heights",
            4 * max_surface_samples.max(1) as u64,
            storage | upload,
        );
        let brushes = buffer(
            "placement brushes",
            64 * max_brushes.max(1) as u64,
            storage | upload,
        );
        let status = buffer("placement projection status", 4, storage | upload);
        let indirect = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("placement final indirect"),
            contents: bytemuck::cast_slice(&[vertex_count, 0, 0, 0]),
            usage: storage | upload | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::INDIRECT,
        });
        let source_layout = layout(
            device,
            "placement source layout",
            &[uniform_entry(0), storage_entry(1, false)],
        );
        let project_layout = layout(
            device,
            "placement projection layout",
            &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        );
        let field_layout = layout(
            device,
            "placement field layout",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        );
        let source_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("placement XZ kernel"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/heightfield_placement_source.wgsl").into(),
            ),
        });
        let project_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("placement exact projection kernel"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/heightfield_placement_project.wgsl").into(),
            ),
        });
        let field_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("placement author kernel"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scatter_field.wgsl").into()),
        });
        let source_pipeline = pipeline(device, &source_layout, &source_shader, "placement_main");
        let project_pipeline = pipeline(device, &project_layout, &project_shader, "project_main");
        let finalize_pipeline = pipeline(device, &project_layout, &project_shader, "finalize_main");
        let field_pipeline = pipeline(device, &field_layout, &field_shader, "field_main");
        let bind = |label, layout, entries: &[wgpu::BindGroupEntry<'_>]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries,
            })
        };
        let source_bind = bind(
            "placement source bindings",
            &source_layout,
            &[
                buffer_binding(0, &source_params),
                buffer_binding(1, &records),
            ],
        );
        let project_bind = bind(
            "placement project bindings",
            &project_layout,
            &[
                buffer_binding(0, &project_params),
                buffer_binding(1, &records),
                buffer_binding(2, &heights),
                buffer_binding(3, &samples),
                buffer_binding(4, &status),
                buffer_binding(5, &indirect),
                buffer_binding(6, &attributes),
            ],
        );
        let field_bind = bind(
            "placement field bindings",
            &field_layout,
            &[
                buffer_binding(0, &field_params),
                buffer_binding(1, &records),
                buffer_binding(2, &brushes),
                buffer_binding(3, &attributes),
                buffer_binding(4, &compact),
                buffer_binding(5, &indirect),
            ],
        );
        Ok(Self {
            device_identity: ctx.device_identity().clone(),
            queue: ctx.queue().clone(),
            capacity,
            vertex_count,
            request_generation: 0,
            max_brushes,
            max_surface_samples,
            max_groups: limits.max_compute_workgroups_per_dimension,
            allocated_bytes,
            source_params,
            project_params,
            field_params,
            records,
            samples,
            attributes,
            compact,
            indirect,
            status,
            heights,
            brushes,
            source_pipeline,
            project_pipeline,
            field_pipeline,
            finalize_pipeline,
            source_bind,
            project_bind,
            field_bind,
            height_scratch,
            brush_scratch,
            request: None,
        })
    }

    /// Pure complete validation precedes every upload. Unchanged recipes enqueue
    /// no uploads; snapshot equality is structural, potentially O(samples+brushes).
    /// Changed heights upload four bytes per sample, brushes 64 bytes per brush,
    /// and changed uniforms 192/128/128 bytes. No candidate data is CPU-generated.
    #[allow(clippy::too_many_arguments)]
    pub fn write_request(
        &mut self,
        config: HeightfieldPlacementConfig,
        region: HeightfieldPlacementRegion,
        field: &ScatterDensityField,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldPlacementFilters,
        budget: GpuHeightfieldPlacementWorkBudget,
    ) -> Result<(), GpuHeightfieldPlacementError> {
        let next = prepare(
            config,
            region,
            field,
            surface,
            filters,
            budget,
            [self.capacity, self.max_brushes, self.max_surface_samples],
            self.max_groups,
        )?;
        let next_generation =
            self.request_generation
                .checked_add(1)
                .ok_or(GpuHeightfieldPlacementError::Gpu(
                    GpuScatterError::ArithmeticOverflow,
                ))?;
        let old = self.request.as_ref();
        if old.is_none_or(|r| r.source != next.source) {
            self.queue
                .write_buffer(&self.source_params, 0, bytemuck::cast_slice(&next.source));
        }
        if old.is_none_or(|r| r.project != next.project) {
            self.queue
                .write_buffer(&self.project_params, 0, bytemuck::cast_slice(&next.project));
        }
        if old.is_none_or(|r| r.field != next.field) {
            self.queue
                .write_buffer(&self.field_params, 0, bytemuck::cast_slice(&next.field));
        }
        if old.is_none_or(|r| r.key.surface.tile().samples() != surface.tile().samples()) {
            self.height_scratch.clear();
            self.height_scratch
                .extend(surface.tile().samples().iter().copied().map(i32::from));
            self.queue
                .write_buffer(&self.heights, 0, bytemuck::cast_slice(&self.height_scratch));
        }
        if old.is_none_or(|r| r.key.field.brushes() != field.brushes()) {
            self.brush_scratch.clear();
            self.brush_scratch.extend(
                field
                    .brushes()
                    .iter()
                    .map(crate::scatter_field::brush_words),
            );
            if !self.brush_scratch.is_empty() {
                self.queue.write_buffer(
                    &self.brushes,
                    0,
                    bytemuck::cast_slice(&self.brush_scratch),
                );
            }
        }
        self.request = Some(next);
        self.request_generation = next_generation;
        Ok(())
    }

    /// Records four ordered compute passes and clears transient GPU state.
    /// No owned allocation, queue upload, submit, wait, map or readback.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuHeightfieldPlacementError> {
        let r = self
            .request
            .as_ref()
            .ok_or(GpuScatterError::RequestNotUploaded)?;
        encoder.clear_buffer(&self.indirect, 4, Some(4));
        encoder.clear_buffer(&self.status, 0, None);
        if r.source[11] != 0 {
            for (pipeline, bind, label) in [
                (&self.source_pipeline, &self.source_bind, "placement XZ"),
                (
                    &self.project_pipeline,
                    &self.project_bind,
                    "placement projection",
                ),
                (
                    &self.field_pipeline,
                    &self.field_bind,
                    "placement author field",
                ),
                (
                    &self.finalize_pipeline,
                    &self.project_bind,
                    "placement fail-closed finalize",
                ),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(r.groups[0], r.groups[1], 1);
            }
        }
        Ok(())
    }

    /// Primary prefix, not the GPU accepted count.
    pub fn candidate_count(&self) -> u32 {
        self.request.as_ref().map_or(0, |r| r.source[11])
    }
    /// Full structural immutable semantic key, identical to the CPU contract.
    pub fn cache_key(&self) -> Option<&HeightfieldPlacementCacheKey> {
        self.request.as_ref().map(|r| &r.key)
    }
    /// Independent placement geometry ABI described on the type.
    pub fn records(&self) -> &wgpu::Buffer {
        &self.records
    }
    /// Exact unoffset height and upward normal per primary slot.
    pub fn surface_samples(&self) -> &wgpu::Buffer {
        &self.samples
    }
    /// Final density/species/accepted/rank per primary slot.
    pub fn attributes(&self) -> &wgpu::Buffer {
        &self.attributes
    }
    /// Live final source-slot prefix determined by indirect[1].
    pub fn compact_indices(&self) -> &wgpu::Buffer {
        &self.compact
    }
    /// Final draw arguments, zero instances on any projection overflow.
    pub fn indirect_args(&self) -> &wgpu::Buffer {
        &self.indirect
    }
    /// One u32 count of projection overflows, reset by every encode.
    pub fn status(&self) -> &wgpu::Buffer {
        &self.status
    }
    /// All owned logical GPU buffers, excluding driver overhead.
    pub fn allocated_buffer_bytes(&self) -> u64 {
        self.allocated_bytes
    }
    /// Actual host i32 scratch payload capacity, excluding metadata.
    pub fn height_scratch_bytes(&self) -> usize {
        self.height_scratch.capacity() * 4
    }
    /// Actual host brush scratch payload capacity, excluding metadata.
    pub fn brush_scratch_bytes(&self) -> usize {
        self.brush_scratch.capacity() * 64
    }
    /// Fixed primary-slot capacity; never an accepted-point truncation limit.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn device_identity(&self) -> &std::sync::Arc<()> {
        &self.device_identity
    }

    pub(super) fn draw_args(&self) -> [u32; 3] {
        [self.vertex_count, 0, 0]
    }

    pub(super) fn request_generation(&self) -> u64 {
        self.request_generation
    }

    pub(super) fn validate_request(
        &self,
        config: HeightfieldPlacementConfig,
        region: HeightfieldPlacementRegion,
        field: &ScatterDensityField,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldPlacementFilters,
        budget: GpuHeightfieldPlacementWorkBudget,
    ) -> Result<(), GpuHeightfieldPlacementError> {
        prepare(
            config,
            region,
            field,
            surface,
            filters,
            budget,
            [self.capacity, self.max_brushes, self.max_surface_samples],
            self.max_groups,
        )?;
        Ok(())
    }
}

fn layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn validate(
    capacity: usize,
    brushes: usize,
    samples: usize,
    budget: u64,
    limits: &wgpu::Limits,
) -> Result<u64, GpuHeightfieldPlacementError> {
    validate_capacity(capacity, u64::MAX, limits)?;
    let b = u32::try_from(brushes).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let h = u32::try_from(samples).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let brush_bytes = 64 * u64::from(b.max(1));
    let height_bytes = 4 * u64::from(h.max(1));
    let bytes = 148 * capacity.max(1) as u64 + brush_bytes + height_bytes + 468;
    if bytes > budget {
        return Err(GpuScatterError::BufferBudgetExceeded {
            required: bytes,
            limit: budget,
        }
        .into());
    }
    for size in [brush_bytes, height_bytes] {
        check(
            "max_storage_buffer_binding_size",
            size,
            limits.max_storage_buffer_binding_size,
        )?;
        check("max_buffer_size", size, limits.max_buffer_size)?;
    }
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

fn put64(words: &mut [u32], index: usize, value: u64) {
    words[index] = value as u32;
    words[index + 1] = (value >> 32) as u32;
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    config: HeightfieldPlacementConfig,
    region: HeightfieldPlacementRegion,
    field: &ScatterDensityField,
    surface: &PlanarHeightfieldSurface,
    filters: HeightfieldPlacementFilters,
    budget: GpuHeightfieldPlacementWorkBudget,
    capacities: [usize; 3],
    max_groups: u32,
) -> Result<Request, GpuHeightfieldPlacementError> {
    if filters.surface_height_mm[0] > filters.surface_height_mm[1] || filters.max_slope_run == 0 {
        return Err(HeightfieldPlacementError::InvalidFilters.into());
    }
    let work = estimate_heightfield_placement_work(config, region, field)?;
    if work.primary_candidates != 0
        && (0..2).any(|a| {
            region.min_xz_mm[a] < surface.min_xz_mm()[a]
                || region.max_xz_mm[a] > surface.max_xz_mm()[a]
        })
    {
        return Err(
            HeightfieldPlacementError::Surface(HeightfieldQueryError::OutsideSurface).into(),
        );
    }
    if work.max_candidate_evaluations > budget.max_candidate_evaluations {
        return Err(HeightfieldPlacementError::CandidateBudgetExceeded {
            required: work.max_candidate_evaluations,
            limit: budget.max_candidate_evaluations,
        }
        .into());
    }
    if work.max_sample_reads > budget.max_sample_reads {
        return Err(HeightfieldPlacementError::SurfaceBudgetExceeded {
            required: work.max_sample_reads,
            limit: budget.max_sample_reads,
        }
        .into());
    }
    if work.max_field_work > budget.max_field_work {
        return Err(HeightfieldPlacementError::FieldBudgetExceeded {
            required: work.max_field_work,
            limit: budget.max_field_work,
        }
        .into());
    }
    if work.primary_candidates > capacities[0] as u64 {
        return Err(GpuScatterError::CapacityExceeded {
            required: work.primary_candidates,
            capacity: capacities[0],
        }
        .into());
    }
    if field.brushes().len() > capacities[1] {
        return Err(GpuHeightfieldPlacementError::BrushCapacityExceeded {
            required: field.brushes().len(),
            capacity: capacities[1],
        });
    }
    if surface.tile().samples().len() > capacities[2] {
        return Err(GpuHeightfieldPlacementError::SurfaceCapacityExceeded {
            required: surface.tile().samples().len(),
            capacity: capacities[2],
        });
    }
    let count =
        u32::try_from(work.primary_candidates).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let groups = dispatch(work.primary_candidates, max_groups)?;
    let mut source = [0; 48];
    put64(&mut source, 0, config.seed);
    source[2] = config.version;
    source[3] = config.grid_spacing_mm;
    put64(&mut source, 4, config.salt);
    source[6] = config.min_distance_mm;
    source[7] = config.min_distance_mm.div_ceil(config.grid_spacing_mm);
    source[11] = count;
    let spacing = i64::from(config.grid_spacing_mm);
    for (a, axis) in [0, 2].into_iter().enumerate() {
        put64(&mut source, 18 + axis * 2, region.min_xz_mm[a] as u64);
        put64(&mut source, 24 + axis * 2, region.max_xz_mm[a] as u64);
        if count != 0 {
            let min = region.min_xz_mm[a].div_euclid(spacing);
            let max = (region.max_xz_mm[a] - 1).div_euclid(spacing);
            source[8 + axis] = u32::try_from(i128::from(max) - i128::from(min) + 1)
                .map_err(|_| GpuScatterError::ArithmeticOverflow)?;
            put64(&mut source, 12 + axis * 2, min as u64);
        }
    }
    put64(&mut source, 30, i64::MIN.div_euclid(spacing) as u64);
    put64(&mut source, 32, i64::MAX.div_euclid(spacing) as u64);
    source[34] = i64::MIN.rem_euclid(spacing) as u32;
    source[35] = i64::MAX.rem_euclid(spacing) as u32;
    source[36] = groups[0];
    let embed = surface.embedding();
    let spec = surface.tile().spec();
    let mut project = [0; 32];
    project[..4].copy_from_slice(&[
        count,
        groups[0],
        spec.samples_per_edge(),
        embed.sample_spacing_mm,
    ]);
    for (i, value) in embed
        .sample_origin_mm
        .into_iter()
        .chain([spec.height_scale_mm()])
        .chain(filters.surface_height_mm)
        .chain([config.vertical_offset_mm])
        .enumerate()
    {
        put64(&mut project, 4 + i * 2, value as u64);
    }
    project[20] = filters.max_slope_rise;
    project[21] = filters.max_slope_run;
    let fc = field.config();
    let mut fw = [0; 32];
    put64(&mut fw, 0, fc.seed);
    fw[2] = u32::from(fc.base_density);
    fw[3] = fc.base_species;
    if let Some(noise) = fc.noise {
        fw[4] = u32::from(noise.amplitude != 0);
        fw[5] = noise.cell_size_mm;
        fw[6] = u32::from(noise.amplitude);
        fw[7] = noise.warp_cell_size_mm;
        fw[8] = noise.warp_amplitude_mm;
    }
    fw[9] =
        u32::try_from(field.brushes().len()).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    fw[10] = count;
    fw[11] = groups[0];
    put64(&mut fw, 12, config.seed);
    fw[14] = config.version;
    fw[15] = config.grid_spacing_mm;
    put64(&mut fw, 16, config.salt);
    fw[18] = SCATTER_KERNEL_VERSION;
    Ok(Request {
        source,
        project,
        field: fw,
        groups,
        key: HeightfieldPlacementCacheKey {
            placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
            source_kernel_version: SCATTER_KERNEL_VERSION,
            surface_version: HEIGHTFIELD_SURFACE_VERSION,
            field_version: SCATTER_FIELD_VERSION,
            config,
            region,
            filters,
            surface: surface.clone(),
            field: field.clone(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: 6,
            ..wgpu::Limits::downlevel_defaults()
        }
    }

    #[test]
    fn exact_combined_buffer_accounting_and_zero_slot_minima() {
        assert_eq!(validate(0, 0, 0, u64::MAX, &limits()), Ok(684));
        let bytes = 148 * 91 + 64 * 5 + 4 * 81 + 468;
        assert_eq!(validate(91, 5, 81, bytes, &limits()), Ok(bytes));
        assert!(matches!(
            validate(91, 5, 81, bytes - 1, &limits()),
            Err(GpuHeightfieldPlacementError::Gpu(
                GpuScatterError::BufferBudgetExceeded { .. }
            ))
        ));
    }

    #[test]
    fn limits_cover_each_stage_before_construction() {
        let mut l = limits();
        l.max_storage_buffers_per_shader_stage = 5;
        assert!(validate(1, 0, 4, u64::MAX, &l).is_err());
        let mut l = limits();
        l.max_storage_buffer_binding_size = 256;
        assert!(validate(1, 5, 4, u64::MAX, &l).is_err());
        assert!(validate(5, 0, 4, u64::MAX, &l).is_err());
        assert!(validate(1, 0, 65, u64::MAX, &l).is_err());
        let mut l = limits();
        l.max_bindings_per_bind_group = 6;
        assert!(validate(1, 0, 4, u64::MAX, &l).is_err());
    }
}
