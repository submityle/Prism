//! Fixed-resource full-key deletion after the planetary placement kernel.
//!
//! The deletion pass consumes only the sealed compact and indirect output of
//! [`GpuPlanetPlacement`]. It never rewrites planetary records or height
//! samples; only its own compact prefix and indirect instance count are live.

use prism_world_terrain::{
    PlanetPlacementConfig, PlanetPlacementDeletionSet, PlanetPlacementError, PlanetPlacementRegion,
    PLANET_PLACEMENT_DELETION_VERSION, PLANET_PLACEMENT_VERSION,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry};
use crate::scatter::{check, dispatch};
use crate::{
    GpuContext, GpuPlanetPlacement, GpuPlanetPlacementError,
    GpuPlanetPlacementRegistrationSnapshot, GpuPlanetPlacementWorkBudget,
};

const PARAM_BYTES: u64 = 64;
const KEY_BYTES: u64 = 48;
const WORKGROUP_SIZE: u64 = 64;

/// Revision of the planetary placement deletion GPU algorithm.
pub const GPU_PLANET_PLACEMENT_DELETION_VERSION: u32 = 1;
/// Twelve scalar `u32` words: 48-byte complete planetary placement key ABI.
pub const GPU_PLANET_PLACEMENT_DELETION_KEY_ABI_VERSION: u32 = 1;

const _: () = assert!(PLANET_PLACEMENT_DELETION_VERSION == 1 && PLANET_PLACEMENT_VERSION == 1);

/// Fixed GPU and reusable host scratch capacity for planetary deletions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPlanetPlacementDeletionCapacity {
    /// Maximum number of complete deletion keys.
    pub max_keys: usize,
    /// Maximum bytes owned by the upstream and deletion resources.
    pub max_buffer_bytes: u64,
    /// Maximum reserved host key-packing payload.
    pub max_scratch_bytes: usize,
}

/// Conservative admission independent of the survivor count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPlanetPlacementDeletionWorkBudget {
    /// Maximum candidate slots inspected by the deletion pass.
    pub max_input_slots: u64,
    /// Maximum supplied deletion keys.
    pub max_keys: usize,
    /// Maximum full-key comparisons in binary searches.
    pub max_key_comparisons: u64,
}

/// Bytes uploaded by the latest successful request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuPlanetPlacementDeletionUploadStats {
    pub uniform_bytes: u64,
    pub key_bytes: u64,
}

/// One native planetary-deletion buffer and its logical RHI import size.
///
/// Cloning this value clones a handle only; it does not allocate or copy GPU
/// memory. The importing bridge must use the producer's original device.
#[derive(Clone)]
pub struct GpuPlanetPlacementDeletionBufferSnapshot {
    buffer: wgpu::Buffer,
    bytes: u64,
}

impl GpuPlanetPlacementDeletionBufferSnapshot {
    /// Native buffer owned by the deletion producer.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Exact logical byte size for graph import and binding validation.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Planetary full-key deletion compute pipeline, bindings, and dispatch.
#[derive(Clone)]
pub struct GpuPlanetPlacementDeletionStageSnapshot {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch: [u32; 3],
}

impl GpuPlanetPlacementDeletionStageSnapshot {
    /// Native deletion pipeline.
    #[must_use]
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Native deletion bind group compatible with `pipeline`.
    #[must_use]
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Dispatch dimensions for the uploaded request.
    #[must_use]
    pub const fn dispatch(&self) -> [u32; 3] {
        self.dispatch
    }
}

/// Complete native registration surface for one planetary deletion request.
///
/// `upstream` contains the six placement buffers/stages. `stage` is absent
/// exactly when the request has no candidate slots; callers must still reset
/// the deletion indirect instance count after copying the upstream arguments.
/// This snapshot neither submits work nor retains resources through completion.
#[derive(Clone)]
pub struct GpuPlanetPlacementDeletionRegistrationSnapshot {
    /// Generation of the exact placement request consumed by deletion.
    pub request_generation: u64,
    /// Logical candidate slots to filter.
    pub candidate_count: u32,
    /// Number of sorted full keys uploaded for this request.
    pub key_count: u32,
    /// Complete upstream placement registration surface.
    pub upstream: GpuPlanetPlacementRegistrationSnapshot,
    /// Request uniform buffer.
    pub params: GpuPlanetPlacementDeletionBufferSnapshot,
    /// Allocated full-key storage buffer; only `key_count * 48` bytes are live.
    pub keys: GpuPlanetPlacementDeletionBufferSnapshot,
    /// Filtered compact source-slot output.
    pub compact: GpuPlanetPlacementDeletionBufferSnapshot,
    /// Filtered four-word indirect arguments.
    pub indirect: GpuPlanetPlacementDeletionBufferSnapshot,
    /// Full-key deletion stage; absent for an empty candidate stream.
    pub stage: Option<GpuPlanetPlacementDeletionStageSnapshot>,
}

/// Complete structural cache dependency for filtered planetary placement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuPlanetPlacementDeletionCacheKey {
    pub upstream: crate::GpuPlanetPlacementCacheKey,
    pub deletions: PlanetPlacementDeletionSet,
    pub deletion_version: u32,
    pub gpu_deletion_version: u32,
    pub key_abi_version: u32,
    /// `[vertex_count, first_vertex, first_instance]`, excluding count.
    pub draw_args: [u32; 3],
}

/// Recoverable preflight and request-state failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuPlanetPlacementDeletionError {
    Placement(GpuPlanetPlacementError),
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

impl std::fmt::Display for GpuPlanetPlacementDeletionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placement(error) => write!(f, "GPU planetary placement: {error}"),
            Self::DeviceMismatch => f.write_str("placement belongs to a different logical device"),
            Self::RequestNotUploaded => {
                f.write_str("upload a planetary deletion request before encoding")
            }
            Self::UpstreamChanged => {
                f.write_str("planetary placement upstream no longer matches the deletion request")
            }
            Self::ArithmeticOverflow => f.write_str("GPU planetary deletion arithmetic overflow"),
            Self::KeyCapacityExceeded { required, capacity } => write!(
                f,
                "planetary deletions need {required} keys, capacity {capacity}"
            ),
            Self::InputBudgetExceeded { required, limit } => write!(
                f,
                "planetary deletion visits {required} slots, budget {limit}"
            ),
            Self::KeyBudgetExceeded { required, limit } => write!(
                f,
                "planetary deletion contains {required} keys, budget {limit}"
            ),
            Self::ComparisonBudgetExceeded { required, limit } => write!(
                f,
                "planetary deletion needs {required} full-key comparisons, budget {limit}"
            ),
            Self::ScratchBudgetExceeded { required, limit } => write!(
                f,
                "planetary deletion scratch needs {required} bytes, budget {limit}"
            ),
            Self::AllocationFailed => {
                f.write_str("planetary deletion key scratch reservation failed")
            }
        }
    }
}

impl std::error::Error for GpuPlanetPlacementDeletionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Placement(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GpuPlanetPlacementError> for GpuPlanetPlacementDeletionError {
    fn from(error: GpuPlanetPlacementError) -> Self {
        Self::Placement(error)
    }
}

impl From<PlanetPlacementError> for GpuPlanetPlacementDeletionError {
    fn from(error: PlanetPlacementError) -> Self {
        Self::Placement(error.into())
    }
}

/// An owned planetary placement producer followed by complete-key deletion.
pub struct GpuPlanetPlacementDeletions {
    placement: GpuPlanetPlacement,
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
    key_scratch: Vec<[u32; 12]>,
    request: Option<Request>,
    upload_stats: GpuPlanetPlacementDeletionUploadStats,
}

struct Request {
    words: [u32; 16],
    groups: [u32; 2],
    upstream_generation: u64,
    key: GpuPlanetPlacementDeletionCacheKey,
}

impl GpuPlanetPlacementDeletions {
    /// Takes ownership of a planetary placement producer on the same device.
    pub fn new(
        ctx: &GpuContext,
        placement: GpuPlanetPlacement,
        capacity: GpuPlanetPlacementDeletionCapacity,
    ) -> Result<Self, GpuPlanetPlacementDeletionError> {
        if !std::sync::Arc::ptr_eq(placement.device_identity(), ctx.device_identity()) {
            return Err(GpuPlanetPlacementDeletionError::DeviceMismatch);
        }
        let limits = ctx.device().limits();
        let (deletion_bytes, allocated_bytes) = validate_capacity(
            placement.capacity(),
            placement.allocated_buffer_bytes(),
            capacity,
            &limits,
        )?;
        let mut key_scratch = Vec::<[u32; 12]>::new();
        key_scratch
            .try_reserve_exact(capacity.max_keys)
            .map_err(|_| GpuPlanetPlacementDeletionError::AllocationFailed)?;
        validate_scratch(key_scratch.capacity(), capacity.max_scratch_bytes)?;

        let device = ctx.device();
        let keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet placement deletion full keys"),
            size: KEY_BYTES * capacity.max_keys.max(1) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compact = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet placement deletion compact"),
            size: 4 * placement.capacity().max(1) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet placement deletion request"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw = placement.draw_args();
        let indirect = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("planet placement deletion indirect"),
            contents: bytemuck::cast_slice(&[draw[0], 0_u32, draw[1], draw[2]]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet placement deletion layout"),
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
            label: Some("planet placement deletion pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet placement deletion shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/planet_placement_deletion.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet placement deletion compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("planet_placement_deletion_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet placement deletion bindings"),
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

    pub fn validate_request(
        &self,
        config: PlanetPlacementConfig,
        region: PlanetPlacementRegion,
        tile: &std::sync::Arc<prism_world_terrain::HeightfieldTile>,
        placement_budget: GpuPlanetPlacementWorkBudget,
        deletions: &PlanetPlacementDeletionSet,
        budget: GpuPlanetPlacementDeletionWorkBudget,
    ) -> Result<(), GpuPlanetPlacementDeletionError> {
        self.prepare(config, region, tile, placement_budget, deletions, budget)
            .map(|_| ())
    }

    fn prepare(
        &self,
        config: PlanetPlacementConfig,
        region: PlanetPlacementRegion,
        tile: &std::sync::Arc<prism_world_terrain::HeightfieldTile>,
        placement_budget: GpuPlanetPlacementWorkBudget,
        deletions: &PlanetPlacementDeletionSet,
        budget: GpuPlanetPlacementDeletionWorkBudget,
    ) -> Result<([u32; 16], [u32; 2]), GpuPlanetPlacementDeletionError> {
        self.placement
            .validate_request(config, region, tile, placement_budget)?;
        let work = prism_world_terrain::estimate_planet_placement_work(region, tile)?;
        let groups = validate_work(
            work,
            deletions.keys().len(),
            self.max_keys,
            budget,
            self.max_groups,
        )?;
        let mut words = [0_u32; 16];
        words[0] =
            u32::try_from(work).map_err(|_| GpuPlanetPlacementDeletionError::ArithmeticOverflow)?;
        words[1] = u32::try_from(deletions.keys().len())
            .map_err(|_| GpuPlanetPlacementDeletionError::ArithmeticOverflow)?;
        words[2] = groups[0];
        words[3] = groups[1];
        words[4] = PLANET_PLACEMENT_VERSION;
        words[5] = config.seed as u32;
        words[6] = (config.seed >> 32) as u32;
        words[7] = config.version;
        words[8] = config.salt as u32;
        words[9] = (config.salt >> 32) as u32;
        Ok((words, groups))
    }

    pub fn write_request(
        &mut self,
        config: PlanetPlacementConfig,
        region: PlanetPlacementRegion,
        tile: &std::sync::Arc<prism_world_terrain::HeightfieldTile>,
        placement_budget: GpuPlanetPlacementWorkBudget,
        deletions: &PlanetPlacementDeletionSet,
        budget: GpuPlanetPlacementDeletionWorkBudget,
    ) -> Result<(), GpuPlanetPlacementDeletionError> {
        let (words, groups) =
            self.prepare(config, region, tile, placement_budget, deletions, budget)?;
        self.placement
            .write_request(config, region, tile, placement_budget)?;
        let upstream = self
            .placement
            .cache_key()
            .cloned()
            .expect("successful planetary placement write has a cache key");
        let mut stats = GpuPlanetPlacementDeletionUploadStats::default();
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
            key: GpuPlanetPlacementDeletionCacheKey {
                upstream,
                deletions: deletions.clone(),
                deletion_version: PLANET_PLACEMENT_DELETION_VERSION,
                gpu_deletion_version: GPU_PLANET_PLACEMENT_DELETION_VERSION,
                key_abi_version: GPU_PLANET_PLACEMENT_DELETION_KEY_ABI_VERSION,
                draw_args: self.placement.draw_args(),
            },
        });
        self.upload_stats = stats;
        Ok(())
    }

    /// Encodes planetary placement, copies its final count, then filters exact keys.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuPlanetPlacementDeletionError> {
        let request = self
            .request
            .as_ref()
            .ok_or(GpuPlanetPlacementDeletionError::RequestNotUploaded)?;
        if request.upstream_generation != self.placement.request_generation()
            || request.words[0] != self.placement.candidate_count()
            || request.key.draw_args != self.placement.draw_args()
        {
            return Err(GpuPlanetPlacementDeletionError::UpstreamChanged);
        }
        self.placement.encode(encoder)?;
        encoder.copy_buffer_to_buffer(self.placement.indirect_args(), 0, &self.indirect, 0, 16);
        encoder.clear_buffer(&self.indirect, 4, Some(4));
        if request.words[0] != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet placement full-key deletions"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(request.groups[0], request.groups[1], 1);
        }
        Ok(())
    }

    /// Clones the complete planetary placement/deletion RHI registration
    /// surface for the current request without submitting or copying buffers.
    ///
    /// The caller must register every buffer and compute stage on this
    /// producer's original device, declare all accesses to its render graph,
    /// preserve the signed-i128 upstream record ABI, and retain registrations
    /// until submitted work completes. The snapshot does not itself provide
    /// those lifecycle guarantees.
    pub fn registration_snapshot(
        &self,
    ) -> Result<GpuPlanetPlacementDeletionRegistrationSnapshot, GpuPlanetPlacementDeletionError>
    {
        let request = self
            .request
            .as_ref()
            .ok_or(GpuPlanetPlacementDeletionError::RequestNotUploaded)?;
        if request.upstream_generation != self.placement.request_generation()
            || request.words[0] != self.placement.candidate_count()
            || request.key.draw_args != self.placement.draw_args()
        {
            return Err(GpuPlanetPlacementDeletionError::UpstreamChanged);
        }
        let upstream = self.placement.registration_snapshot()?;
        let buffer = |buffer: &wgpu::Buffer, bytes| GpuPlanetPlacementDeletionBufferSnapshot {
            buffer: buffer.clone(),
            bytes,
        };
        let stage = (request.words[0] != 0).then(|| GpuPlanetPlacementDeletionStageSnapshot {
            pipeline: self.pipeline.clone(),
            bind_group: self.bind_group.clone(),
            dispatch: [request.groups[0], request.groups[1], 1],
        });
        Ok(GpuPlanetPlacementDeletionRegistrationSnapshot {
            request_generation: request.upstream_generation,
            candidate_count: request.words[0],
            key_count: request.words[1],
            upstream,
            params: buffer(&self.params, PARAM_BYTES),
            keys: buffer(&self.keys, KEY_BYTES * self.max_keys.max(1) as u64),
            compact: buffer(&self.compact, 4 * self.placement.capacity().max(1) as u64),
            indirect: buffer(&self.indirect, 16),
            stage,
        })
    }

    pub fn records(&self) -> &wgpu::Buffer {
        self.placement.records()
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
    pub fn cache_key(&self) -> Option<&GpuPlanetPlacementDeletionCacheKey> {
        self.request.as_ref().map(|request| &request.key)
    }
    pub fn deletions(&self) -> Option<&PlanetPlacementDeletionSet> {
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
    pub fn upload_stats(&self) -> GpuPlanetPlacementDeletionUploadStats {
        self.upload_stats
    }

    #[cfg(feature = "placement_rhi")]
    pub(super) fn device_identity(&self) -> &std::sync::Arc<()> {
        self.placement.device_identity()
    }
}

fn key_words(key: &prism_world_terrain::PlanetPlacementKey) -> [u32; 12] {
    [
        key.placement_version,
        key.seed as u32,
        (key.seed >> 32) as u32,
        key.version,
        key.salt as u32,
        (key.salt >> 32) as u32,
        u32::from(key.tile.face().index()),
        u32::from(key.tile.level()),
        key.tile.x(),
        key.tile.y(),
        key.cell[0],
        key.cell[1],
    ]
}

fn validate_capacity(
    capacity: usize,
    upstream_bytes: u64,
    request: GpuPlanetPlacementDeletionCapacity,
    limits: &wgpu::Limits,
) -> Result<(u64, u64), GpuPlanetPlacementDeletionError> {
    let c =
        u32::try_from(capacity).map_err(|_| GpuPlanetPlacementDeletionError::ArithmeticOverflow)?;
    let d = u32::try_from(request.max_keys)
        .map_err(|_| GpuPlanetPlacementDeletionError::ArithmeticOverflow)?;
    let deletion = KEY_BYTES
        .checked_mul(u64::from(d.max(1)))
        .and_then(|value| value.checked_add(4 * u64::from(c.max(1))))
        .and_then(|value| value.checked_add(PARAM_BYTES + 16))
        .ok_or(GpuPlanetPlacementDeletionError::ArithmeticOverflow)?;
    let total = upstream_bytes
        .checked_add(deletion)
        .ok_or(GpuPlanetPlacementDeletionError::ArithmeticOverflow)?;
    if total > request.max_buffer_bytes {
        return Err(GpuPlanetPlacementDeletionError::Placement(
            GpuPlanetPlacementError::Gpu(crate::GpuScatterError::BufferBudgetExceeded {
                required: total,
                limit: request.max_buffer_bytes,
            }),
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
        WORKGROUP_SIZE,
        u64::from(limits.max_compute_invocations_per_workgroup),
    )?;
    gpu_check(
        "max_compute_workgroup_size_x",
        WORKGROUP_SIZE,
        u64::from(limits.max_compute_workgroup_size_x),
    )?;
    gpu_check(
        "max_compute_workgroup_storage_size",
        260,
        u64::from(limits.max_compute_workgroup_storage_size),
    )?;
    dispatch(u64::from(c), limits.max_compute_workgroups_per_dimension).map_err(|error| {
        GpuPlanetPlacementDeletionError::Placement(GpuPlanetPlacementError::Gpu(error))
    })?;
    Ok((deletion, total))
}

fn gpu_check(
    name: &'static str,
    required: u64,
    available: u64,
) -> Result<(), GpuPlanetPlacementDeletionError> {
    check(name, required, available).map_err(|error| {
        GpuPlanetPlacementDeletionError::Placement(GpuPlanetPlacementError::Gpu(error))
    })
}

fn validate_scratch(capacity: usize, limit: usize) -> Result<(), GpuPlanetPlacementDeletionError> {
    let required = capacity
        .checked_mul(KEY_BYTES as usize)
        .ok_or(GpuPlanetPlacementDeletionError::ArithmeticOverflow)?;
    if required > limit {
        return Err(GpuPlanetPlacementDeletionError::ScratchBudgetExceeded { required, limit });
    }
    Ok(())
}

fn validate_work(
    n: u64,
    d: usize,
    capacity: usize,
    budget: GpuPlanetPlacementDeletionWorkBudget,
    max_groups: u32,
) -> Result<[u32; 2], GpuPlanetPlacementDeletionError> {
    if d > capacity {
        return Err(GpuPlanetPlacementDeletionError::KeyCapacityExceeded {
            required: d,
            capacity,
        });
    }
    if n > budget.max_input_slots {
        return Err(GpuPlanetPlacementDeletionError::InputBudgetExceeded {
            required: n,
            limit: budget.max_input_slots,
        });
    }
    if d > budget.max_keys {
        return Err(GpuPlanetPlacementDeletionError::KeyBudgetExceeded {
            required: d,
            limit: budget.max_keys,
        });
    }
    let d32 = u32::try_from(d).map_err(|_| GpuPlanetPlacementDeletionError::ArithmeticOverflow)?;
    let steps = u64::from(u32::BITS - d32.leading_zeros());
    let required = n
        .checked_mul(steps)
        .ok_or(GpuPlanetPlacementDeletionError::ArithmeticOverflow)?;
    if required > budget.max_key_comparisons {
        return Err(GpuPlanetPlacementDeletionError::ComparisonBudgetExceeded {
            required,
            limit: budget.max_key_comparisons,
        });
    }
    dispatch(n, max_groups).map_err(|error| {
        GpuPlanetPlacementDeletionError::Placement(GpuPlanetPlacementError::Gpu(error))
    })
}

#[cfg(test)]
#[path = "planet_placement_deletion_tests.rs"]
mod tests;
