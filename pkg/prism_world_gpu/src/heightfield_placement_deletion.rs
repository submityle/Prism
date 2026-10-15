//! Fixed-resource full-key deletion after the exact two-dimensional placement chain.
//!
//! Placement identity is 56 bytes: placement revision plus the complete source
//! candidate key. This ABI is intentionally independent from the 52-byte
//! three-dimensional scatter deletion ABI. The wrapper owns its placement
//! producer, so the deletion pass can only consume the sealed final compact and
//! indirect output of that producer.

use prism_world_residency::{ScatterDensityField, SCATTER_KERNEL_VERSION};
use prism_world_terrain::{
    estimate_heightfield_placement_work, HeightfieldPlacementCacheKey, HeightfieldPlacementConfig,
    HeightfieldPlacementDeletionSet, HeightfieldPlacementError, HeightfieldPlacementFilters,
    HeightfieldPlacementRegion, PlanarHeightfieldSurface, HEIGHTFIELD_PLACEMENT_DELETION_VERSION,
    HEIGHTFIELD_PLACEMENT_VERSION,
};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry};
use crate::scatter::{check, dispatch};
use crate::{GpuContext, GpuHeightfieldPlacement, GpuHeightfieldPlacementError};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

const PARAM_BYTES: u64 = 128;
const KEY_BYTES: u64 = 56;

/// Revision of the placement deletion GPU algorithm.
pub const GPU_HEIGHTFIELD_PLACEMENT_DELETION_VERSION: u32 = 1;
/// Fourteen scalar u32 words: 56-byte placement key ABI.
pub const GPU_HEIGHTFIELD_PLACEMENT_DELETION_KEY_ABI_VERSION: u32 = 1;

const _: () = assert!(
    HEIGHTFIELD_PLACEMENT_DELETION_VERSION == 1
        && HEIGHTFIELD_PLACEMENT_VERSION == 1
        && SCATTER_KERNEL_VERSION == 1
);

/// Fixed GPU and reusable host scratch capacity for placement deletions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuHeightfieldPlacementDeletionCapacity {
    pub max_keys: usize,
    pub max_buffer_bytes: u64,
    /// Key packing scratch payload, including spare Vec capacity.
    pub max_scratch_bytes: usize,
}

/// Conservative deletion admission independent of the unknown survivor count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuHeightfieldPlacementDeletionWorkBudget {
    pub max_input_slots: u64,
    pub max_keys: usize,
    pub max_key_comparisons: u64,
}

/// Uploads caused by the latest successful deletion request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuHeightfieldPlacementDeletionUploadStats {
    pub uniform_bytes: u64,
    pub key_bytes: u64,
}

/// Complete structural cache dependency for a filtered placement stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuHeightfieldPlacementDeletionCacheKey {
    pub upstream: HeightfieldPlacementCacheKey,
    pub deletions: HeightfieldPlacementDeletionSet,
    pub deletion_version: u32,
    pub gpu_deletion_version: u32,
    pub key_abi_version: u32,
    /// [vertex_count, first_vertex, first_instance], excluding count.
    pub draw_args: [u32; 3],
}

/// Recoverable host-side preflight failure. A failed request retains the prior
/// placement request, deletion set and upload statistics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuHeightfieldPlacementDeletionError {
    Placement(GpuHeightfieldPlacementError),
    DeviceMismatch,
    RequestNotUploaded,
    UpstreamChanged,
    ArithmeticOverflow,
    KeyCapacityExceeded { required: usize, capacity: usize },
    InputBudgetExceeded { required: u64, limit: u64 },
    KeyBudgetExceeded { required: usize, limit: usize },
    ComparisonBudgetExceeded { required: u64, limit: u64 },
    ScratchBudgetExceeded { required: usize, limit: usize },
    AllocationFailed,
}

impl std::fmt::Display for GpuHeightfieldPlacementDeletionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placement(error) => write!(f, "GPU heightfield placement: {error}"),
            Self::DeviceMismatch => f.write_str("placement belongs to a different logical device"),
            Self::RequestNotUploaded => {
                f.write_str("upload a placement deletion request before encoding")
            }
            Self::UpstreamChanged => {
                f.write_str("placement upstream no longer matches the deletion request")
            }
            Self::ArithmeticOverflow => f.write_str("GPU placement deletion arithmetic overflow"),
            Self::KeyCapacityExceeded { required, capacity } => {
                write!(
                    f,
                    "placement deletions need {required} keys, capacity {capacity}"
                )
            }
            Self::InputBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "placement deletion visits {required} slots, budget {limit}"
                )
            }
            Self::KeyBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "placement deletion contains {required} keys, budget {limit}"
                )
            }
            Self::ComparisonBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "placement deletion needs {required} full-key comparisons, budget {limit}"
                )
            }
            Self::ScratchBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "placement deletion scratch needs {required} bytes, budget {limit}"
                )
            }
            Self::AllocationFailed => {
                f.write_str("placement deletion key scratch reservation failed")
            }
        }
    }
}

impl std::error::Error for GpuHeightfieldPlacementDeletionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Placement(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GpuHeightfieldPlacementError> for GpuHeightfieldPlacementDeletionError {
    fn from(error: GpuHeightfieldPlacementError) -> Self {
        Self::Placement(error)
    }
}

impl From<HeightfieldPlacementError> for GpuHeightfieldPlacementDeletionError {
    fn from(error: HeightfieldPlacementError) -> Self {
        Self::Placement(error.into())
    }
}

/// An owned exact placement producer followed by 56-byte full-key deletion.
///
/// Upstream records, samples and author attributes are never rewritten. Only
/// this object's compact prefix and indirect count are the final filtered
/// selection. Submit each encoded request before writing another request.
pub struct GpuHeightfieldPlacementDeletions {
    placement: GpuHeightfieldPlacement,
    queue: wgpu::Queue,
    max_keys: usize,
    max_groups: u32,
    allocated_bytes: u64,
    deletion_bytes: u64,
    params: wgpu::Buffer,
    keys: wgpu::Buffer,
    compact: wgpu::Buffer,
    indirect: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    key_scratch: Vec<[u32; 14]>,
    request: Option<Request>,
    upload_stats: GpuHeightfieldPlacementDeletionUploadStats,
}

struct Request {
    words: [u32; 32],
    groups: [u32; 2],
    upstream_generation: u64,
    key: GpuHeightfieldPlacementDeletionCacheKey,
}

impl GpuHeightfieldPlacementDeletions {
    /// Takes ownership of one placement producer on the same logical device.
    /// New logical buffers cost `56*D + 4*C + 144` bytes, with one physical
    /// slot allocated for zero capacities.
    pub fn new(
        ctx: &GpuContext,
        placement: GpuHeightfieldPlacement,
        capacity: GpuHeightfieldPlacementDeletionCapacity,
    ) -> Result<Self, GpuHeightfieldPlacementDeletionError> {
        if !std::sync::Arc::ptr_eq(placement.device_identity(), ctx.device_identity()) {
            return Err(GpuHeightfieldPlacementDeletionError::DeviceMismatch);
        }
        let limits = ctx.device().limits();
        let (deletion_bytes, allocated_bytes) = validate_capacity(
            placement.capacity(),
            placement.allocated_buffer_bytes(),
            capacity,
            &limits,
        )?;
        let mut key_scratch = Vec::<[u32; 14]>::new();
        key_scratch
            .try_reserve_exact(capacity.max_keys)
            .map_err(|_| GpuHeightfieldPlacementDeletionError::AllocationFailed)?;
        validate_scratch(key_scratch.capacity(), capacity.max_scratch_bytes)?;

        let device = ctx.device();
        let keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("heightfield placement deletion full keys"),
            size: KEY_BYTES * capacity.max_keys.max(1) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compact = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("heightfield placement deletion compact"),
            size: 4 * placement.capacity().max(1) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("heightfield placement deletion request"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw = placement.draw_args();
        let indirect = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("heightfield placement deletion indirect"),
            contents: bytemuck::cast_slice(&[draw[0], 0, draw[1], draw[2]]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("heightfield placement deletion layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("heightfield placement deletion pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("heightfield placement deletion shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/heightfield_placement_deletion.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("heightfield placement deletion compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("heightfield_placement_deletion_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("heightfield placement deletion bindings"),
            layout: &layout,
            entries: &[
                buffer_binding(0, &params),
                buffer_binding(1, placement.records()),
                buffer_binding(2, placement.compact_indices()),
                buffer_binding(3, placement.indirect_args()),
                buffer_binding(4, &keys),
                buffer_binding(5, &compact),
                buffer_binding(6, &indirect),
            ],
        });
        Ok(Self {
            placement,
            queue: ctx.queue().clone(),
            max_keys: capacity.max_keys,
            max_groups: limits.max_compute_workgroups_per_dimension,
            allocated_bytes,
            deletion_bytes,
            params,
            keys,
            compact,
            indirect,
            pipeline,
            bind_group,
            key_scratch,
            request: None,
            upload_stats: Default::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn validate_request(
        &self,
        config: HeightfieldPlacementConfig,
        region: HeightfieldPlacementRegion,
        field: &ScatterDensityField,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldPlacementFilters,
        placement_budget: crate::GpuHeightfieldPlacementWorkBudget,
        deletions: &HeightfieldPlacementDeletionSet,
        budget: GpuHeightfieldPlacementDeletionWorkBudget,
    ) -> Result<(), GpuHeightfieldPlacementDeletionError> {
        self.prepare(
            config,
            region,
            field,
            surface,
            filters,
            placement_budget,
            deletions,
            budget,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare(
        &self,
        config: HeightfieldPlacementConfig,
        region: HeightfieldPlacementRegion,
        field: &ScatterDensityField,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldPlacementFilters,
        placement_budget: crate::GpuHeightfieldPlacementWorkBudget,
        deletions: &HeightfieldPlacementDeletionSet,
        budget: GpuHeightfieldPlacementDeletionWorkBudget,
    ) -> Result<([u32; 32], [u32; 2]), GpuHeightfieldPlacementDeletionError> {
        self.placement.validate_request(
            config,
            region,
            field,
            surface,
            filters,
            placement_budget,
        )?;
        let work = estimate_heightfield_placement_work(config, region, field)?;
        let groups = validate_work(
            work.primary_candidates,
            deletions.keys().len(),
            self.max_keys,
            budget,
            self.max_groups,
        )?;
        let key = placement_key(config, region, field, surface, filters);
        let mut words = [0; 32];
        words[0] = u32::try_from(work.primary_candidates)
            .map_err(|_| GpuHeightfieldPlacementDeletionError::ArithmeticOverflow)?;
        words[1] = u32::try_from(deletions.keys().len())
            .map_err(|_| GpuHeightfieldPlacementDeletionError::ArithmeticOverflow)?;
        words[2] = groups[0];
        words[3] = groups[1];
        words[4] = HEIGHTFIELD_PLACEMENT_VERSION;
        words[5] = SCATTER_KERNEL_VERSION;
        words[6] = key.config.seed as u32;
        words[7] = (key.config.seed >> 32) as u32;
        words[8] = key.config.version;
        words[9] = key.config.salt as u32;
        words[10] = (key.config.salt >> 32) as u32;
        words[11] = key.config.grid_spacing_mm;
        Ok((words, groups))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_request(
        &mut self,
        config: HeightfieldPlacementConfig,
        region: HeightfieldPlacementRegion,
        field: &ScatterDensityField,
        surface: &PlanarHeightfieldSurface,
        filters: HeightfieldPlacementFilters,
        placement_budget: crate::GpuHeightfieldPlacementWorkBudget,
        deletions: &HeightfieldPlacementDeletionSet,
        budget: GpuHeightfieldPlacementDeletionWorkBudget,
    ) -> Result<(), GpuHeightfieldPlacementDeletionError> {
        let (words, groups) = self.prepare(
            config,
            region,
            field,
            surface,
            filters,
            placement_budget,
            deletions,
            budget,
        )?;
        self.placement
            .write_request(config, region, field, surface, filters, placement_budget)?;
        let upstream = self
            .placement
            .cache_key()
            .cloned()
            .expect("successful placement write has a cache key");
        let mut stats = GpuHeightfieldPlacementDeletionUploadStats::default();
        if self.request.as_ref().is_none_or(|old| old.words != words) {
            self.queue
                .write_buffer(&self.params, 0, bytemuck::cast_slice(&words));
            stats.uniform_bytes = PARAM_BYTES;
        }
        if self
            .request
            .as_ref()
            .is_none_or(|old| old.key.deletions != *deletions)
        {
            self.key_scratch.clear();
            self.key_scratch
                .extend(deletions.keys().iter().map(key_words));
            if !self.key_scratch.is_empty() {
                self.queue
                    .write_buffer(&self.keys, 0, bytemuck::cast_slice(&self.key_scratch));
                stats.key_bytes = KEY_BYTES * self.key_scratch.len() as u64;
            }
        }
        self.request = Some(Request {
            words,
            groups,
            upstream_generation: self.placement.request_generation(),
            key: GpuHeightfieldPlacementDeletionCacheKey {
                upstream,
                deletions: deletions.clone(),
                deletion_version: HEIGHTFIELD_PLACEMENT_DELETION_VERSION,
                gpu_deletion_version: GPU_HEIGHTFIELD_PLACEMENT_DELETION_VERSION,
                key_abi_version: GPU_HEIGHTFIELD_PLACEMENT_DELETION_KEY_ABI_VERSION,
                draw_args: self.placement.draw_args(),
            },
        });
        self.upload_stats = stats;
        Ok(())
    }

    /// Records the placement chain, copies its draw arguments, clears this
    /// count, then removes exact full-key matches from the final placement set.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuHeightfieldPlacementDeletionError> {
        let request = self
            .request
            .as_ref()
            .ok_or(GpuHeightfieldPlacementDeletionError::RequestNotUploaded)?;
        if request.upstream_generation != self.placement.request_generation()
            || request.words[0] != self.placement.candidate_count()
            || request.key.draw_args != self.placement.draw_args()
        {
            return Err(GpuHeightfieldPlacementDeletionError::UpstreamChanged);
        }
        self.placement.encode(encoder)?;
        encoder.copy_buffer_to_buffer(self.placement.indirect_args(), 0, &self.indirect, 0, 16);
        encoder.clear_buffer(&self.indirect, 4, Some(4));
        if request.words[0] != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("heightfield placement full-key deletions"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(request.groups[0], request.groups[1], 1);
        }
        Ok(())
    }

    pub fn records(&self) -> &wgpu::Buffer {
        self.placement.records()
    }
    pub fn surface_samples(&self) -> &wgpu::Buffer {
        self.placement.surface_samples()
    }
    pub fn attributes(&self) -> &wgpu::Buffer {
        self.placement.attributes()
    }
    pub fn compact_indices(&self) -> &wgpu::Buffer {
        &self.compact
    }
    pub fn indirect_args(&self) -> &wgpu::Buffer {
        &self.indirect
    }
    pub fn upstream_compact_indices(&self) -> &wgpu::Buffer {
        self.placement.compact_indices()
    }
    pub fn candidate_count(&self) -> u32 {
        self.request.as_ref().map_or(0, |request| request.words[0])
    }
    pub fn cache_key(&self) -> Option<&GpuHeightfieldPlacementDeletionCacheKey> {
        self.request.as_ref().map(|request| &request.key)
    }
    pub fn deletions(&self) -> Option<&HeightfieldPlacementDeletionSet> {
        self.cache_key().map(|key| &key.deletions)
    }
    pub fn allocated_buffer_bytes(&self) -> u64 {
        self.allocated_bytes
    }
    pub fn deletion_buffer_bytes(&self) -> u64 {
        self.deletion_bytes
    }
    pub fn key_scratch_bytes(&self) -> usize {
        self.key_scratch.capacity() * KEY_BYTES as usize
    }
    pub fn upload_stats(&self) -> GpuHeightfieldPlacementDeletionUploadStats {
        self.upload_stats
    }
}

fn placement_key(
    config: HeightfieldPlacementConfig,
    region: HeightfieldPlacementRegion,
    field: &ScatterDensityField,
    surface: &PlanarHeightfieldSurface,
    filters: HeightfieldPlacementFilters,
) -> HeightfieldPlacementCacheKey {
    HeightfieldPlacementCacheKey {
        placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
        source_kernel_version: SCATTER_KERNEL_VERSION,
        surface_version: prism_world_terrain::HEIGHTFIELD_SURFACE_VERSION,
        field_version: prism_world_residency::SCATTER_FIELD_VERSION,
        config,
        region,
        filters,
        surface: surface.clone(),
        field: field.clone(),
    }
}

fn key_words(key: &prism_world_terrain::HeightfieldPlacementKey) -> [u32; 14] {
    let source = key.source;
    [
        key.placement_version,
        source.kernel_version,
        source.seed as u32,
        (source.seed >> 32) as u32,
        source.version,
        source.salt as u32,
        (source.salt >> 32) as u32,
        source.grid_spacing_mm,
        source.grid.x as u32,
        (source.grid.x as u64 >> 32) as u32,
        source.grid.y as u32,
        (source.grid.y as u64 >> 32) as u32,
        source.grid.z as u32,
        (source.grid.z as u64 >> 32) as u32,
    ]
}

fn validate_capacity(
    capacity: usize,
    upstream_bytes: u64,
    request: GpuHeightfieldPlacementDeletionCapacity,
    limits: &wgpu::Limits,
) -> Result<(u64, u64), GpuHeightfieldPlacementDeletionError> {
    let c = u32::try_from(capacity)
        .map_err(|_| GpuHeightfieldPlacementDeletionError::ArithmeticOverflow)?;
    let d = u32::try_from(request.max_keys)
        .map_err(|_| GpuHeightfieldPlacementDeletionError::ArithmeticOverflow)?;
    let deletion = KEY_BYTES
        .checked_mul(u64::from(d.max(1)))
        .and_then(|value| value.checked_add(4 * u64::from(c.max(1))))
        .and_then(|value| value.checked_add(PARAM_BYTES + 16))
        .ok_or(GpuHeightfieldPlacementDeletionError::ArithmeticOverflow)?;
    let total = upstream_bytes
        .checked_add(deletion)
        .ok_or(GpuHeightfieldPlacementDeletionError::ArithmeticOverflow)?;
    if total > request.max_buffer_bytes {
        return Err(GpuHeightfieldPlacementDeletionError::Placement(
            crate::GpuScatterError::BufferBudgetExceeded {
                required: total,
                limit: request.max_buffer_bytes,
            }
            .into(),
        ));
    }
    gpu_check(
        "max_storage_buffer_binding_size",
        KEY_BYTES * u64::from(d.max(1)),
        limits.max_storage_buffer_binding_size,
    )?;
    gpu_check(
        "max_storage_buffer_binding_size",
        4 * u64::from(c.max(1)),
        limits.max_storage_buffer_binding_size,
    )?;
    gpu_check(
        "max_storage_buffer_binding_size",
        16,
        limits.max_storage_buffer_binding_size,
    )?;
    gpu_check(
        "max_buffer_size",
        KEY_BYTES * u64::from(d.max(1)),
        limits.max_buffer_size,
    )?;
    gpu_check(
        "max_buffer_size",
        4 * u64::from(c.max(1)),
        limits.max_buffer_size,
    )?;
    gpu_check("max_buffer_size", PARAM_BYTES, limits.max_buffer_size)?;
    gpu_check("max_buffer_size", 16, limits.max_buffer_size)?;
    gpu_check(
        "max_uniform_buffer_binding_size",
        PARAM_BYTES,
        limits.max_uniform_buffer_binding_size,
    )?;
    gpu_check(
        "max_storage_buffers_per_shader_stage",
        6,
        u64::from(limits.max_storage_buffers_per_shader_stage),
    )?;
    gpu_check(
        "max_uniform_buffers_per_shader_stage",
        1,
        u64::from(limits.max_uniform_buffers_per_shader_stage),
    )?;
    gpu_check("max_bind_groups", 1, u64::from(limits.max_bind_groups))?;
    gpu_check(
        "max_bindings_per_bind_group",
        7,
        u64::from(limits.max_bindings_per_bind_group),
    )?;
    gpu_check(
        "max_compute_invocations_per_workgroup",
        64,
        u64::from(limits.max_compute_invocations_per_workgroup),
    )?;
    gpu_check(
        "max_compute_workgroup_size_x",
        64,
        u64::from(limits.max_compute_workgroup_size_x),
    )?;
    gpu_check(
        "max_compute_workgroup_storage_size",
        260,
        u64::from(limits.max_compute_workgroup_storage_size),
    )?;
    dispatch(u64::from(c), limits.max_compute_workgroups_per_dimension)
        .map_err(|error| GpuHeightfieldPlacementDeletionError::Placement(error.into()))?;
    Ok((deletion, total))
}

fn gpu_check(
    name: &'static str,
    required: u64,
    available: u64,
) -> Result<(), GpuHeightfieldPlacementDeletionError> {
    check(name, required, available)
        .map_err(|error| GpuHeightfieldPlacementDeletionError::Placement(error.into()))
}

fn validate_scratch(
    capacity: usize,
    limit: usize,
) -> Result<(), GpuHeightfieldPlacementDeletionError> {
    let required = capacity
        .checked_mul(KEY_BYTES as usize)
        .ok_or(GpuHeightfieldPlacementDeletionError::ArithmeticOverflow)?;
    if required > limit {
        return Err(
            GpuHeightfieldPlacementDeletionError::ScratchBudgetExceeded { required, limit },
        );
    }
    Ok(())
}

fn validate_work(
    n: u64,
    d: usize,
    capacity: usize,
    budget: GpuHeightfieldPlacementDeletionWorkBudget,
    max_groups: u32,
) -> Result<[u32; 2], GpuHeightfieldPlacementDeletionError> {
    if d > capacity {
        return Err(GpuHeightfieldPlacementDeletionError::KeyCapacityExceeded {
            required: d,
            capacity,
        });
    }
    if n > budget.max_input_slots {
        return Err(GpuHeightfieldPlacementDeletionError::InputBudgetExceeded {
            required: n,
            limit: budget.max_input_slots,
        });
    }
    if d > budget.max_keys {
        return Err(GpuHeightfieldPlacementDeletionError::KeyBudgetExceeded {
            required: d,
            limit: budget.max_keys,
        });
    }
    let d32 =
        u32::try_from(d).map_err(|_| GpuHeightfieldPlacementDeletionError::ArithmeticOverflow)?;
    let steps = u64::from(u32::BITS - d32.leading_zeros());
    let required = n
        .checked_mul(steps)
        .ok_or(GpuHeightfieldPlacementDeletionError::ArithmeticOverflow)?;
    if required > budget.max_key_comparisons {
        return Err(
            GpuHeightfieldPlacementDeletionError::ComparisonBudgetExceeded {
                required,
                limit: budget.max_key_comparisons,
            },
        );
    }
    dispatch(n, max_groups)
        .map_err(|error| GpuHeightfieldPlacementDeletionError::Placement(error.into()))
}
