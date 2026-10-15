//! Fixed-resource full-key deletions after a sealed, owned GPU scatter chain.

use prism_world_residency::{
    estimate_scatter_work, FieldScatterCacheKey, ScatterCacheKey, ScatterCandidateKey,
    ScatterConfig, ScatterDeletionSet, ScatterDensityField, ScatterRegion,
    SCATTER_DELETION_VERSION, SCATTER_KERNEL_VERSION,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry};
use crate::scatter::{check, dispatch};
use crate::{GpuContext, GpuScatter, GpuScatterError, GpuScatterField, GpuScatterFieldError};

#[cfg(feature = "terrain_scatter")]
use crate::{GpuScatterHeightfield, GpuScatterHeightfieldError};
#[cfg(feature = "terrain_scatter")]
use prism_world_terrain::{
    HeightfieldScatterBudget, HeightfieldScatterCacheKey, HeightfieldScatterFilters,
    PlanarHeightfieldSurface,
};

const PARAM_BYTES: u64 = 128;
const KEY_BYTES: u64 = 52;
/// Revision of the sorted full-key GPU deletion algorithm.
pub const GPU_SCATTER_DELETION_VERSION: u32 = 1;
/// Thirteen scalar u32 words per key, explicitly packed, never Rust memory.
pub const GPU_SCATTER_DELETION_KEY_ABI_VERSION: u32 = 1;
const _: () = assert!(SCATTER_DELETION_VERSION == 1 && SCATTER_KERNEL_VERSION == 1);

/// Source request; work limits are admission policy, not cache identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuScatterSourceRequest {
    pub config: ScatterConfig,
    pub region: ScatterRegion,
    pub max_candidate_evaluations: u64,
}

/// Borrowed author-field request, retaining no second brush payload.
#[derive(Clone, Copy, Debug)]
pub struct GpuScatterFieldRequest<'a> {
    pub source: GpuScatterSourceRequest,
    pub field: &'a ScatterDensityField,
    pub max_field_work: u64,
}

/// Borrowed heightfield request; the successful cache shares immutable snapshots.
#[cfg(feature = "terrain_scatter")]
#[derive(Clone, Copy, Debug)]
pub struct GpuScatterHeightfieldRequest<'a> {
    pub field: GpuScatterFieldRequest<'a>,
    pub surface: &'a PlanarHeightfieldSurface,
    pub filters: HeightfieldScatterFilters,
    pub surface_budget: HeightfieldScatterBudget,
}

/// Fixed deletion resources. GPU bytes include the supplied owned upstream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuScatterDeletionCapacity {
    pub max_keys: usize,
    pub max_buffer_bytes: u64,
    /// Only the new key-packing Vec payload, including spare capacity.
    pub max_scratch_bytes: usize,
}

/// Conservative admission bounds, independent of unknown GPU survivor counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuScatterDeletionWorkBudget {
    /// Primary slots N, not the GPU final survivor count M.
    pub max_input_slots: u64,
    pub max_keys: usize,
    /// N * ceil(log2(D+1)) complete structural comparisons. D=0 costs zero
    /// comparisons but still needs the separately budgeted traversal/compaction.
    pub max_key_comparisons: u64,
}

/// Deletion-stage uploads in the latest successful write_request, not upstream
/// uploads, driver staging, persistent memory, or GPU execution time.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuScatterDeletionUploadStats {
    pub uniform_bytes: u64,
    pub key_bytes: u64,
}

/// Complete semantic dependency; equality is structural, not a digest, pointer,
/// revision-only check, or proof that a GPU result has already been generated.
/// Request budgets and physical buffer identity deliberately do not belong here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuScatterDeletionCacheKey<K> {
    pub upstream: K,
    /// Includes keys outside this request and in other generation namespaces.
    pub deletions: ScatterDeletionSet,
    pub deletion_version: u32,
    pub gpu_deletion_version: u32,
    pub key_abi_version: u32,
    /// [vertex_count, first_vertex, first_instance], excluding the GPU count.
    pub draw_args: [u32; 3],
}

/// Recoverable failures preserve the previous uploaded request and upload stats.
/// Device loss/OOM and arbitrary external writes through aliased raw GPU handles
/// are outside this managed API's preflight contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuScatterDeletionError {
    Scatter(GpuScatterError),
    Field(GpuScatterFieldError),
    #[cfg(feature = "terrain_scatter")]
    Heightfield(GpuScatterHeightfieldError),
    DeviceMismatch,
    RequestNotUploaded,
    UpstreamChanged,
    ArithmeticOverflow,
    KeyCapacityExceeded {
        required: usize,
        capacity: usize,
    },
    InputBudgetExceeded {
        required: u64,
        limit: u64,
    },
    KeyBudgetExceeded {
        required: usize,
        limit: usize,
    },
    ComparisonBudgetExceeded {
        required: u64,
        limit: u64,
    },
    ScratchBudgetExceeded {
        required: usize,
        limit: usize,
    },
    AllocationFailed,
}

impl std::fmt::Display for GpuScatterDeletionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scatter(e) => write!(f, "GPU scatter: {e}"),
            Self::Field(e) => write!(f, "GPU author field: {e}"),
            #[cfg(feature = "terrain_scatter")]
            Self::Heightfield(e) => write!(f, "GPU heightfield: {e}"),
            Self::DeviceMismatch => f.write_str("upstream belongs to a different logical device"),
            Self::RequestNotUploaded => f.write_str("upload a deletion request before encoding"),
            Self::UpstreamChanged => f.write_str("upstream no longer matches the deletion request"),
            Self::ArithmeticOverflow => f.write_str("GPU deletion arithmetic overflow"),
            Self::KeyCapacityExceeded { required, capacity } => {
                write!(f, "deletions need {required} keys, capacity {capacity}")
            }
            Self::InputBudgetExceeded { required, limit } => {
                write!(f, "deletions visit {required} slots, budget {limit}")
            }
            Self::KeyBudgetExceeded { required, limit } => {
                write!(f, "deletions contain {required} keys, budget {limit}")
            }
            Self::ComparisonBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "deletions need at most {required} full-key comparisons, budget {limit}"
                )
            }
            Self::ScratchBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "key scratch needs {required} capacity bytes, budget {limit}"
                )
            }
            Self::AllocationFailed => f.write_str("key upload scratch reservation failed"),
        }
    }
}

impl std::error::Error for GpuScatterDeletionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scatter(e) => Some(e),
            Self::Field(e) => Some(e),
            #[cfg(feature = "terrain_scatter")]
            Self::Heightfield(e) => Some(e),
            _ => None,
        }
    }
}

impl From<GpuScatterError> for GpuScatterDeletionError {
    fn from(error: GpuScatterError) -> Self {
        Self::Scatter(error)
    }
}
impl From<GpuScatterFieldError> for GpuScatterDeletionError {
    fn from(error: GpuScatterFieldError) -> Self {
        Self::Field(error)
    }
}
#[cfg(feature = "terrain_scatter")]
impl From<GpuScatterHeightfieldError> for GpuScatterDeletionError {
    fn from(error: GpuScatterHeightfieldError) -> Self {
        Self::Heightfield(error)
    }
}

// This private supertrait supplies access without exposing an upstream reference
// from the owning wrapper. There is no user-implementable/raw-buffer producer.
mod sealed {
    use super::*;

    pub trait Access: Sized {
        fn device_identity(&self) -> &std::sync::Arc<()>;
        fn capacity(&self) -> usize;
        fn allocated_bytes(&self) -> u64;
        fn draw_args(&self) -> [u32; 3];
        fn generation(&self) -> u64;
        fn candidate_count(&self) -> u32;
        fn records(&self) -> &wgpu::Buffer;
        fn attributes(&self) -> Option<&wgpu::Buffer>;
        fn compact(&self) -> &wgpu::Buffer;
        fn indirect(&self) -> &wgpu::Buffer;
        fn encode(&self, encoder: &mut wgpu::CommandEncoder)
            -> Result<(), GpuScatterDeletionError>;
        fn source_request(
            request: &<Self as GpuScatterDeletionUpstream>::Request<'_>,
        ) -> GpuScatterSourceRequest
        where
            Self: GpuScatterDeletionUpstream;
        fn validate(
            &self,
            request: &<Self as GpuScatterDeletionUpstream>::Request<'_>,
        ) -> Result<(), GpuScatterDeletionError>
        where
            Self: GpuScatterDeletionUpstream;
        fn write(
            &mut self,
            request: &<Self as GpuScatterDeletionUpstream>::Request<'_>,
        ) -> Result<(), GpuScatterDeletionError>
        where
            Self: GpuScatterDeletionUpstream;
        fn snapshot(&self) -> Option<<Self as GpuScatterDeletionUpstream>::CacheKey>
        where
            Self: GpuScatterDeletionUpstream;
    }
}

/// Sealed producer contract implemented only for source, field, and heightfield.
/// Move an existing producer into `GpuScatterDeletions::new`; it is never exposed
/// mutably (or returned as a separately encodable upstream).
#[allow(private_bounds)]
pub trait GpuScatterDeletionUpstream: sealed::Access {
    type Request<'a>;
    type CacheKey: Clone + std::fmt::Debug + Eq;
}

impl GpuScatterDeletionUpstream for GpuScatter {
    type Request<'a> = GpuScatterSourceRequest;
    type CacheKey = ScatterCacheKey;
}
impl GpuScatterDeletionUpstream for GpuScatterField {
    type Request<'a> = GpuScatterFieldRequest<'a>;
    type CacheKey = FieldScatterCacheKey;
}
#[cfg(feature = "terrain_scatter")]
impl GpuScatterDeletionUpstream for GpuScatterHeightfield {
    type Request<'a> = GpuScatterHeightfieldRequest<'a>;
    type CacheKey = HeightfieldScatterCacheKey;
}

// All buffer access is inseparable from the concrete, sealed owner. Binding
// identity cannot be spoofed with an equal semantic cache key from another set.
macro_rules! common_access {
    () => {
        fn device_identity(&self) -> &std::sync::Arc<()> {
            self.device_identity()
        }
        fn capacity(&self) -> usize {
            self.capacity()
        }
        fn allocated_bytes(&self) -> u64 {
            self.allocated_buffer_bytes()
        }
        fn draw_args(&self) -> [u32; 3] {
            self.draw_args()
        }
        fn generation(&self) -> u64 {
            self.request_generation()
        }
        fn candidate_count(&self) -> u32 {
            self.candidate_count()
        }
        fn records(&self) -> &wgpu::Buffer {
            self.records()
        }
        fn compact(&self) -> &wgpu::Buffer {
            self.compact_indices()
        }
        fn indirect(&self) -> &wgpu::Buffer {
            self.indirect_args()
        }
        fn encode(
            &self,
            encoder: &mut wgpu::CommandEncoder,
        ) -> Result<(), GpuScatterDeletionError> {
            self.encode(encoder).map_err(Into::into)
        }
    };
}

impl sealed::Access for GpuScatter {
    common_access!();
    fn attributes(&self) -> Option<&wgpu::Buffer> {
        None
    }
    fn source_request(request: &GpuScatterSourceRequest) -> GpuScatterSourceRequest {
        *request
    }
    fn validate(&self, request: &GpuScatterSourceRequest) -> Result<(), GpuScatterDeletionError> {
        self.validate_request(
            request.config,
            request.region,
            request.max_candidate_evaluations,
        )?;
        Ok(())
    }
    fn write(&mut self, request: &GpuScatterSourceRequest) -> Result<(), GpuScatterDeletionError> {
        self.write_request(
            request.config,
            request.region,
            request.max_candidate_evaluations,
        )?;
        Ok(())
    }
    fn snapshot(&self) -> Option<ScatterCacheKey> {
        self.cache_key()
    }
}
impl sealed::Access for GpuScatterField {
    common_access!();
    fn attributes(&self) -> Option<&wgpu::Buffer> {
        Some(self.attributes())
    }
    fn source_request(request: &GpuScatterFieldRequest<'_>) -> GpuScatterSourceRequest {
        request.source
    }
    fn validate(
        &self,
        request: &GpuScatterFieldRequest<'_>,
    ) -> Result<(), GpuScatterDeletionError> {
        self.validate_request(
            request.source.config,
            request.source.region,
            request.source.max_candidate_evaluations,
            request.field,
            request.max_field_work,
        )?;
        Ok(())
    }
    fn write(
        &mut self,
        request: &GpuScatterFieldRequest<'_>,
    ) -> Result<(), GpuScatterDeletionError> {
        self.write_request(
            request.source.config,
            request.source.region,
            request.source.max_candidate_evaluations,
            request.field,
            request.max_field_work,
        )?;
        Ok(())
    }
    fn snapshot(&self) -> Option<FieldScatterCacheKey> {
        self.cache_key().cloned()
    }
}
#[cfg(feature = "terrain_scatter")]
impl sealed::Access for GpuScatterHeightfield {
    common_access!();
    fn attributes(&self) -> Option<&wgpu::Buffer> {
        Some(self.attributes())
    }
    fn source_request(request: &GpuScatterHeightfieldRequest<'_>) -> GpuScatterSourceRequest {
        request.field.source
    }
    fn validate(
        &self,
        request: &GpuScatterHeightfieldRequest<'_>,
    ) -> Result<(), GpuScatterDeletionError> {
        let source = request.field.source;
        self.validate_request(
            source.config,
            source.region,
            source.max_candidate_evaluations,
            request.field.field,
            request.field.max_field_work,
            request.surface,
            request.filters,
            request.surface_budget,
        )?;
        Ok(())
    }
    fn write(
        &mut self,
        request: &GpuScatterHeightfieldRequest<'_>,
    ) -> Result<(), GpuScatterDeletionError> {
        let source = request.field.source;
        self.write_request(
            source.config,
            source.region,
            source.max_candidate_evaluations,
            request.field.field,
            request.field.max_field_work,
            request.surface,
            request.filters,
            request.surface_budget,
        )?;
        Ok(())
    }
    fn snapshot(&self) -> Option<HeightfieldScatterCacheKey> {
        self.cache_key().cloned()
    }
}

struct Request<K> {
    words: [u32; 32],
    groups: [u32; 2],
    upstream_generation: u64,
    key: GpuScatterDeletionCacheKey<K>,
}

/// An owned source/field/heightfield chain followed by sorted full-key deletion.
/// Bindings are fixed at construction, and no mutable upstream escape exists.
/// Final compact entries index the ORIGINAL records and optional attributes;
/// source/attribute accepted flags are NOT deletion-filtered. Only this compact
/// prefix and indirect count represent the final selection. Order is unspecified.
///
/// Submit each encoded request before another write_request/encode on this
/// object. Queue writes run before submission, not at encoder positions. This
/// is a caller contract, not machine-verified submission tracking. Sequential
/// submissions need no host wait. External raw-handle writes are unsupported.
pub struct GpuScatterDeletions<U: GpuScatterDeletionUpstream> {
    upstream: U,
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
    key_scratch: Vec<[u32; 13]>,
    request: Option<Request<U::CacheKey>>,
    upload_stats: GpuScatterDeletionUploadStats,
}

impl<U: GpuScatterDeletionUpstream> GpuScatterDeletions<U> {
    /// Takes ownership of one producer on the same logical device. New logical
    /// GPU buffers cost `52*max(Dcap,1)+4*max(C,1)+144` bytes; C is inherited.
    /// Host scratch is reserved once and its actual capacity is budget-checked.
    pub fn new(
        ctx: &GpuContext,
        upstream: U,
        capacity: GpuScatterDeletionCapacity,
    ) -> Result<Self, GpuScatterDeletionError> {
        validate_device_identity(upstream.device_identity(), ctx.device_identity())?;
        let device = ctx.device();
        let limits = device.limits();
        let (deletion_bytes, allocated_bytes) = validate_capacity(
            upstream.capacity(),
            upstream.allocated_bytes(),
            capacity,
            &limits,
        )?;
        let mut key_scratch = Vec::<[u32; 13]>::new();
        key_scratch
            .try_reserve_exact(capacity.max_keys)
            .map_err(|_| GpuScatterDeletionError::AllocationFailed)?;
        validate_scratch(key_scratch.capacity(), capacity.max_scratch_bytes)?;
        let keys = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world scatter deletion full keys"),
            size: KEY_BYTES * capacity.max_keys.max(1) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compact = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world scatter deletion compact"),
            size: 4 * upstream.capacity().max(1) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world scatter deletion request"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw = upstream.draw_args();
        let indirect = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("world scatter deletion indirect"),
            contents: bytemuck::cast_slice(&[draw[0], 0, draw[1], draw[2]]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world scatter deletion layout"),
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
            label: Some("world scatter deletion pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world scatter full-key deletion shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scatter_deletion.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("world scatter deletion compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_deletion_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world scatter deletion bindings"),
            layout: &layout,
            entries: &[
                buffer_binding(0, &params),
                buffer_binding(1, upstream.records()),
                buffer_binding(2, upstream.compact()),
                buffer_binding(3, upstream.indirect()),
                buffer_binding(4, &keys),
                buffer_binding(5, &compact),
                buffer_binding(6, &indirect),
            ],
        });
        Ok(Self {
            upstream,
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

    /// Pure full-chain admission: no state changes, writes, or owned allocations.
    pub fn validate_request(
        &self,
        upstream: &U::Request<'_>,
        deletions: &ScatterDeletionSet,
        budget: GpuScatterDeletionWorkBudget,
    ) -> Result<(), GpuScatterDeletionError> {
        self.prepare(upstream, deletions, budget)?;
        Ok(())
    }

    fn prepare(
        &self,
        upstream: &U::Request<'_>,
        deletions: &ScatterDeletionSet,
        budget: GpuScatterDeletionWorkBudget,
    ) -> Result<([u32; 32], [u32; 2]), GpuScatterDeletionError> {
        self.upstream.validate(upstream)?;
        let source = U::source_request(upstream);
        let work =
            estimate_scatter_work(source.config, source.region).map_err(GpuScatterError::from)?;
        let groups = validate_work(
            work.primary_candidates,
            deletions.keys().len(),
            self.max_keys,
            budget,
            self.max_groups,
        )?;
        let mut words = [0; 32];
        words[0] = u32::try_from(work.primary_candidates)
            .map_err(|_| GpuScatterDeletionError::ArithmeticOverflow)?;
        words[1] = u32::try_from(deletions.keys().len())
            .map_err(|_| GpuScatterDeletionError::ArithmeticOverflow)?;
        words[2] = groups[0];
        words[4] = SCATTER_KERNEL_VERSION;
        words[5] = source.config.seed as u32;
        words[6] = (source.config.seed >> 32) as u32;
        words[7] = source.config.version;
        words[8] = source.config.salt as u32;
        words[9] = (source.config.salt >> 32) as u32;
        words[10] = source.config.grid_spacing_mm;
        Ok((words, groups))
    }

    /// Preflights the ENTIRE chain before any write. Changed keys are packed into
    /// fixed scratch, never a fresh Vec. Equality/upload work can be O(D+B+S).
    /// A returned error retains the old upstream, deletion request and stats.
    pub fn write_request(
        &mut self,
        upstream: &U::Request<'_>,
        deletions: &ScatterDeletionSet,
        budget: GpuScatterDeletionWorkBudget,
    ) -> Result<(), GpuScatterDeletionError> {
        let (words, groups) = self.prepare(upstream, deletions, budget)?;
        // The sealed upstream repeats its deterministic pure checks before its
        // first write, then has no recoverable failure path. No fallible work
        // follows a successful upstream commit here either.
        self.upstream.write(upstream)?;
        let mut stats = GpuScatterDeletionUploadStats::default();
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
            upstream_generation: self.upstream.generation(),
            key: GpuScatterDeletionCacheKey {
                upstream: self
                    .upstream
                    .snapshot()
                    .expect("sealed upstream committed a request"),
                deletions: deletions.clone(),
                deletion_version: SCATTER_DELETION_VERSION,
                gpu_deletion_version: GPU_SCATTER_DELETION_VERSION,
                key_abi_version: GPU_SCATTER_DELETION_KEY_ABI_VERSION,
                draw_args: self.upstream.draw_args(),
            },
        });
        self.upload_stats = stats;
        Ok(())
    }

    /// Checks the owned producer generation/count before ANY commands, then
    /// records upstream -> copy draw arguments -> clear own count -> deletions.
    /// No owned resource allocation, CPU key scan, upload, submit, wait or readback.
    /// wgpu/driver command-recording internals are not an allocator guarantee.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuScatterDeletionError> {
        let request = self
            .request
            .as_ref()
            .ok_or(GpuScatterDeletionError::RequestNotUploaded)?;
        validate_generation(
            request.upstream_generation,
            self.upstream.generation(),
            request.words[0],
            self.upstream.candidate_count(),
            request.key.draw_args,
            self.upstream.draw_args(),
        )?;
        self.upstream.encode(encoder)?;
        encoder.copy_buffer_to_buffer(self.upstream.indirect(), 0, &self.indirect, 0, 16);
        encoder.clear_buffer(&self.indirect, 4, Some(4));
        if request.words[0] != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("world scatter full-key deletions"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(request.groups[0], request.groups[1], 1);
        }
        Ok(())
    }

    pub fn records(&self) -> &wgpu::Buffer {
        self.upstream.records()
    }
    /// Unmodified upstream attributes; their accepted flag precedes deletion.
    pub fn attributes(&self) -> Option<&wgpu::Buffer> {
        self.upstream.attributes()
    }
    pub fn compact_indices(&self) -> &wgpu::Buffer {
        &self.compact
    }
    pub fn indirect_args(&self) -> &wgpu::Buffer {
        &self.indirect
    }
    /// Primary slots, NOT the final GPU accepted count. Zero before first request.
    pub fn candidate_count(&self) -> u32 {
        self.request.as_ref().map_or(0, |r| r.words[0])
    }
    pub fn cache_key(&self) -> Option<&GpuScatterDeletionCacheKey<U::CacheKey>> {
        self.request.as_ref().map(|r| &r.key)
    }
    pub fn upstream_cache_key(&self) -> Option<&U::CacheKey> {
        self.cache_key().map(|k| &k.upstream)
    }
    pub fn deletions(&self) -> Option<&ScatterDeletionSet> {
        self.cache_key().map(|k| &k.deletions)
    }
    /// Whole-chain logical buffer payload, not physical VRAM/RSS.
    pub fn allocated_buffer_bytes(&self) -> u64 {
        self.allocated_bytes
    }
    pub fn deletion_buffer_bytes(&self) -> u64 {
        self.deletion_bytes
    }
    pub fn key_scratch_bytes(&self) -> usize {
        self.key_scratch.capacity() * KEY_BYTES as usize
    }
    pub fn upload_stats(&self) -> GpuScatterDeletionUploadStats {
        self.upload_stats
    }
}

fn validate_device_identity(
    upstream: &std::sync::Arc<()>,
    context: &std::sync::Arc<()>,
) -> Result<(), GpuScatterDeletionError> {
    if !std::sync::Arc::ptr_eq(upstream, context) {
        return Err(GpuScatterDeletionError::DeviceMismatch);
    }
    Ok(())
}

fn validate_generation(
    expected: u64,
    actual: u64,
    count: u32,
    actual_count: u32,
    draw: [u32; 3],
    actual_draw: [u32; 3],
) -> Result<(), GpuScatterDeletionError> {
    if expected == 0 || expected != actual || count != actual_count || draw != actual_draw {
        return Err(GpuScatterDeletionError::UpstreamChanged);
    }
    Ok(())
}

fn validate_work(
    n: u64,
    d: usize,
    capacity: usize,
    budget: GpuScatterDeletionWorkBudget,
    max_groups: u32,
) -> Result<[u32; 2], GpuScatterDeletionError> {
    if d > capacity {
        return Err(GpuScatterDeletionError::KeyCapacityExceeded {
            required: d,
            capacity,
        });
    }
    if n > budget.max_input_slots {
        return Err(GpuScatterDeletionError::InputBudgetExceeded {
            required: n,
            limit: budget.max_input_slots,
        });
    }
    if d > budget.max_keys {
        return Err(GpuScatterDeletionError::KeyBudgetExceeded {
            required: d,
            limit: budget.max_keys,
        });
    }
    let d = u32::try_from(d).map_err(|_| GpuScatterDeletionError::ArithmeticOverflow)?;
    // floor(log2(D))+1 == ceil(log2(D+1)), without overflowing D+1.
    let steps = u64::from(u32::BITS - d.leading_zeros());
    let required = n
        .checked_mul(steps)
        .ok_or(GpuScatterDeletionError::ArithmeticOverflow)?;
    if required > budget.max_key_comparisons {
        return Err(GpuScatterDeletionError::ComparisonBudgetExceeded {
            required,
            limit: budget.max_key_comparisons,
        });
    }
    u32::try_from(n).map_err(|_| GpuScatterDeletionError::ArithmeticOverflow)?;
    Ok(dispatch(n, max_groups)?)
}

fn validate_scratch(capacity: usize, limit: usize) -> Result<(), GpuScatterDeletionError> {
    let required = capacity
        .checked_mul(KEY_BYTES as usize)
        .ok_or(GpuScatterDeletionError::ArithmeticOverflow)?;
    if required > limit {
        return Err(GpuScatterDeletionError::ScratchBudgetExceeded { required, limit });
    }
    Ok(())
}

fn validate_capacity(
    c: usize,
    upstream_bytes: u64,
    capacity: GpuScatterDeletionCapacity,
    limits: &wgpu::Limits,
) -> Result<(u64, u64), GpuScatterDeletionError> {
    let n = u32::try_from(c).map_err(|_| GpuScatterDeletionError::ArithmeticOverflow)?;
    let d = u32::try_from(capacity.max_keys)
        .map_err(|_| GpuScatterDeletionError::ArithmeticOverflow)?;
    validate_scratch(capacity.max_keys, capacity.max_scratch_bytes)?;
    let key_bytes = KEY_BYTES * u64::from(d.max(1));
    let compact_bytes = 4 * u64::from(n.max(1));
    let own = key_bytes
        .checked_add(compact_bytes)
        .and_then(|v| v.checked_add(PARAM_BYTES + 16))
        .ok_or(GpuScatterDeletionError::ArithmeticOverflow)?;
    let total = own
        .checked_add(upstream_bytes)
        .ok_or(GpuScatterDeletionError::ArithmeticOverflow)?;
    if total > capacity.max_buffer_bytes {
        return Err(GpuScatterError::BufferBudgetExceeded {
            required: total,
            limit: capacity.max_buffer_bytes,
        }
        .into());
    }
    check(
        "max_storage_buffer_binding_size",
        key_bytes.max(compact_bytes).max(16),
        limits.max_storage_buffer_binding_size,
    )?;
    check(
        "max_buffer_size",
        key_bytes.max(compact_bytes).max(PARAM_BYTES),
        limits.max_buffer_size,
    )?;
    check(
        "max_uniform_buffer_binding_size",
        PARAM_BYTES,
        limits.max_uniform_buffer_binding_size,
    )?;
    check(
        "max_storage_buffers_per_shader_stage",
        6,
        u64::from(limits.max_storage_buffers_per_shader_stage),
    )?;
    check(
        "max_uniform_buffers_per_shader_stage",
        1,
        u64::from(limits.max_uniform_buffers_per_shader_stage),
    )?;
    check("max_bind_groups", 1, u64::from(limits.max_bind_groups))?;
    check(
        "max_bindings_per_bind_group",
        7,
        u64::from(limits.max_bindings_per_bind_group),
    )?;
    check(
        "max_compute_invocations_per_workgroup",
        64,
        u64::from(limits.max_compute_invocations_per_workgroup),
    )?;
    check(
        "max_compute_workgroup_size_x",
        64,
        u64::from(limits.max_compute_workgroup_size_x),
    )?;
    check(
        "max_compute_workgroup_storage_size",
        260,
        u64::from(limits.max_compute_workgroup_storage_size),
    )?;
    dispatch(u64::from(n), limits.max_compute_workgroups_per_dimension)?;
    Ok((own, total))
}

fn key_words(key: &ScatterCandidateKey) -> [u32; 13] {
    [
        key.kernel_version,
        key.seed as u32,
        (key.seed >> 32) as u32,
        key.version,
        key.salt as u32,
        (key.salt >> 32) as u32,
        key.grid_spacing_mm,
        key.grid.x as u32,
        (key.grid.x as u64 >> 32) as u32,
        key.grid.y as u32,
        (key.grid.y as u64 >> 32) as u32,
        key.grid.z as u32,
        (key.grid.z as u64 >> 32) as u32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_world_residency::{ScatterDeletionBudget, SectorCoord};
    use std::{cmp::Ordering, sync::Arc};

    fn unlimited_work() -> GpuScatterDeletionWorkBudget {
        GpuScatterDeletionWorkBudget {
            max_input_slots: u64::MAX,
            max_keys: usize::MAX,
            max_key_comparisons: u64::MAX,
        }
    }

    fn capacity(keys: usize) -> GpuScatterDeletionCapacity {
        GpuScatterDeletionCapacity {
            max_keys: keys,
            max_buffer_bytes: u64::MAX,
            max_scratch_bytes: usize::MAX,
        }
    }

    fn key(x: i64) -> ScatterCandidateKey {
        ScatterCandidateKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            seed: 0,
            version: 0,
            salt: 0,
            grid_spacing_mm: 1,
            grid: SectorCoord::new(x, 0, 0),
        }
    }

    #[test]
    fn logical_device_identity_accepts_clones_not_independent_tokens() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<GpuContext>();
        let first = Arc::new(());
        let cloned = first.clone();
        let independent = Arc::new(());
        assert_eq!(validate_device_identity(&first, &cloned), Ok(()));
        // All payloads are equal, just like colliding backend-local IDs can be.
        assert_eq!(first, independent);
        assert_eq!(
            validate_device_identity(&first, &independent),
            Err(GpuScatterDeletionError::DeviceMismatch)
        );
        drop(first);
        assert_eq!(validate_device_identity(&cloned, &cloned), Ok(()));
        assert_eq!(
            validate_device_identity(&cloned, &independent),
            Err(GpuScatterDeletionError::DeviceMismatch)
        );
    }

    #[test]
    fn capacity_accounts_for_sentinels_and_the_entire_chain() {
        let limits = wgpu::Limits::default();
        assert_eq!(
            validate_capacity(0, 276, capacity(0), &limits),
            Ok((200, 476))
        );
        assert_eq!(
            validate_capacity(10, 888, capacity(3), &limits),
            Ok((340, 1228))
        );
        let mut cap = capacity(3);
        cap.max_buffer_bytes = 1227;
        assert_eq!(
            validate_capacity(10, 888, cap, &limits),
            Err(GpuScatterDeletionError::Scatter(
                GpuScatterError::BufferBudgetExceeded {
                    required: 1228,
                    limit: 1227,
                }
            ))
        );
        cap.max_buffer_bytes = 1228;
        assert_eq!(validate_capacity(10, 888, cap, &limits), Ok((340, 1228)));
        assert_eq!(
            validate_capacity(0, u64::MAX, capacity(0), &limits),
            Err(GpuScatterDeletionError::ArithmeticOverflow)
        );
    }

    #[test]
    fn scratch_checks_payload_capacity_not_only_uploaded_length() {
        assert_eq!(validate_scratch(0, 0), Ok(()));
        assert_eq!(validate_scratch(3, 156), Ok(()));
        assert_eq!(
            validate_scratch(4, 156),
            Err(GpuScatterDeletionError::ScratchBudgetExceeded {
                required: 208,
                limit: 156
            })
        );
        assert_eq!(
            validate_scratch(usize::MAX, usize::MAX),
            Err(GpuScatterDeletionError::ArithmeticOverflow)
        );
        let mut cap = capacity(3);
        cap.max_scratch_bytes = 155;
        assert_eq!(
            validate_capacity(10, 0, cap, &wgpu::Limits::default()),
            Err(GpuScatterDeletionError::ScratchBudgetExceeded {
                required: 156,
                limit: 155
            })
        );
    }

    #[test]
    fn capacity_preflights_binding_and_workgroup_limits() {
        for (name, limits, required, available) in [
            (
                "max_storage_buffers_per_shader_stage",
                wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 5,
                    ..Default::default()
                },
                6,
                5,
            ),
            (
                "max_bindings_per_bind_group",
                wgpu::Limits {
                    max_bindings_per_bind_group: 6,
                    ..Default::default()
                },
                7,
                6,
            ),
            (
                "max_compute_workgroup_storage_size",
                wgpu::Limits {
                    max_compute_workgroup_storage_size: 259,
                    ..Default::default()
                },
                260,
                259,
            ),
            (
                "max_compute_invocations_per_workgroup",
                wgpu::Limits {
                    max_compute_invocations_per_workgroup: 63,
                    ..Default::default()
                },
                64,
                63,
            ),
        ] {
            assert_eq!(
                validate_capacity(1, 0, capacity(1), &limits),
                Err(GpuScatterDeletionError::Scatter(
                    GpuScatterError::DeviceLimit {
                        limit: name,
                        required,
                        available,
                    }
                ))
            );
        }
    }

    #[test]
    fn admission_has_independent_slot_key_and_capacity_limits() {
        assert_eq!(validate_work(0, 0, 0, unlimited_work(), 0), Ok([0, 0]));
        assert_eq!(
            validate_work(0, 1, 0, unlimited_work(), 65535),
            Err(GpuScatterDeletionError::KeyCapacityExceeded {
                required: 1,
                capacity: 0
            })
        );
        let mut budget = unlimited_work();
        budget.max_input_slots = 63;
        assert_eq!(
            validate_work(64, 0, 0, budget, 65535),
            Err(GpuScatterDeletionError::InputBudgetExceeded {
                required: 64,
                limit: 63
            })
        );
        budget = unlimited_work();
        budget.max_keys = 2;
        assert_eq!(
            validate_work(0, 3, 3, budget, 65535),
            Err(GpuScatterDeletionError::KeyBudgetExceeded {
                required: 3,
                limit: 2
            })
        );
        budget.max_key_comparisons = 0;
        assert_eq!(validate_work(65, 0, 0, budget, 65535), Ok([2, 1]));
    }

    #[test]
    fn comparison_budget_uses_checked_conservative_binary_search_bound() {
        for (d, steps) in [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 3),
            (7, 3),
            (8, 4),
            (255, 8),
            (256, 9),
        ] {
            let required = 100 * steps;
            let mut budget = unlimited_work();
            budget.max_key_comparisons = required;
            assert_eq!(validate_work(100, d, d, budget, 65535), Ok([2, 1]));
            if required != 0 {
                budget.max_key_comparisons -= 1;
                assert_eq!(
                    validate_work(100, d, d, budget, 65535),
                    Err(GpuScatterDeletionError::ComparisonBudgetExceeded {
                        required,
                        limit: required - 1
                    })
                );
            }
        }
        let mut budget = unlimited_work();
        budget.max_key_comparisons = 31;
        assert_eq!(
            validate_work(1, u32::MAX as usize, usize::MAX, budget, 65535),
            Err(GpuScatterDeletionError::ComparisonBudgetExceeded {
                required: 32,
                limit: 31
            })
        );
        assert_eq!(
            validate_work(u64::MAX, 2, 2, unlimited_work(), 65535),
            Err(GpuScatterDeletionError::ArithmeticOverflow)
        );
    }

    #[test]
    fn count_abi_and_padded_dispatch_cannot_wrap() {
        assert_eq!(
            validate_work(u64::from(u32::MAX) + 1, 0, 0, unlimited_work(), 65535),
            Err(GpuScatterDeletionError::ArithmeticOverflow)
        );
        assert_eq!(
            validate_work(u64::from(u32::MAX), 0, 0, unlimited_work(), 65535),
            Err(GpuScatterDeletionError::Scatter(
                GpuScatterError::ArithmeticOverflow
            ))
        );
        assert_eq!(
            validate_work(u64::from(u32::MAX), 0, 0, unlimited_work(), u32::MAX),
            Ok([1 << 26, 1])
        );
        #[cfg(target_pointer_width = "64")]
        {
            let oversized = u32::MAX as usize + 1;
            assert_eq!(
                validate_work(0, oversized, oversized, unlimited_work(), 65535),
                Err(GpuScatterDeletionError::ArithmeticOverflow)
            );
            assert_eq!(
                validate_capacity(oversized, 0, capacity(0), &wgpu::Limits::default()),
                Err(GpuScatterDeletionError::ArithmeticOverflow)
            );
            assert_eq!(
                validate_capacity(0, 0, capacity(oversized), &wgpu::Limits::default()),
                Err(GpuScatterDeletionError::ArithmeticOverflow)
            );
        }
    }

    #[test]
    fn encode_guard_rejects_missing_epoch_mutation_count_or_draw_changes() {
        let draw = [3, 0, 0];
        assert_eq!(validate_generation(1, 1, 10, 10, draw, draw), Ok(()));
        assert_eq!(
            validate_generation(u64::MAX, u64::MAX, 0, 0, draw, draw),
            Ok(())
        );
        for (expected, actual, count, actual_count, actual_draw) in [
            (0, 0, 10, 10, draw),
            (1, 2, 10, 10, draw),
            (2, 1, 10, 10, draw),
            (1, 1, 10, 9, draw),
            (1, 1, 10, 10, [4, 0, 0]),
            (1, 1, 10, 10, [3, 1, 0]),
            (1, 1, 10, 10, [3, 0, 1]),
        ] {
            assert_eq!(
                validate_generation(expected, actual, count, actual_count, draw, actual_draw),
                Err(GpuScatterDeletionError::UpstreamChanged)
            );
        }
    }

    // Host model of the WGSL compare policy, not a claim of shader execution.
    fn packed_cmp(a: &[u32; 13], b: &[u32; 13]) -> Ordering {
        for i in [0, 2, 1, 3, 5, 4, 6, 8, 7, 10, 9, 12, 11] {
            let mask = if matches!(i, 8 | 10 | 12) {
                0x8000_0000
            } else {
                0
            };
            let result = (a[i] ^ mask).cmp(&(b[i] ^ mask));
            if result != Ordering::Equal {
                return result;
            }
        }
        Ordering::Equal
    }

    #[test]
    fn packing_is_explicit_and_preserves_all_signed_coordinate_bits() {
        let k = ScatterCandidateKey {
            kernel_version: 1,
            seed: 0x0123_4567_89ab_cdef,
            version: 7,
            salt: 0xfedc_ba98_7654_3210,
            grid_spacing_mm: 99,
            grid: SectorCoord::new(i64::MIN, -1, i64::MAX),
        };
        assert_eq!(
            key_words(&k),
            [
                1,
                0x89ab_cdef,
                0x0123_4567,
                7,
                0x7654_3210,
                0xfedc_ba98,
                99,
                0,
                0x8000_0000,
                u32::MAX,
                u32::MAX,
                u32::MAX,
                0x7fff_ffff
            ]
        );
        assert_eq!(std::mem::size_of::<[u32; 13]>(), KEY_BYTES as usize);
    }

    #[test]
    fn packed_comparator_matches_rust_ord_in_every_key_field() {
        let base = key(0);
        let mut keys = vec![base];
        for value in [0, 1, u32::MAX] {
            keys.push(ScatterCandidateKey {
                kernel_version: value,
                ..base
            });
            keys.push(ScatterCandidateKey {
                version: value,
                ..base
            });
            keys.push(ScatterCandidateKey {
                grid_spacing_mm: value,
                ..base
            });
        }
        for value in [0, 1, u64::from(u32::MAX), 1 << 32, 1 << 63, u64::MAX] {
            keys.push(ScatterCandidateKey {
                seed: value,
                ..base
            });
            keys.push(ScatterCandidateKey {
                salt: value,
                ..base
            });
        }
        for value in [i64::MIN, -(1 << 32), -1, 0, 1, 1 << 32, i64::MAX] {
            for axis in 0..3 {
                let mut xyz = [0; 3];
                xyz[axis] = value;
                keys.push(ScatterCandidateKey {
                    grid: SectorCoord::new(xyz[0], xyz[1], xyz[2]),
                    ..base
                });
            }
        }
        // Also exercise precedence when several fields differ simultaneously.
        for i in 0..64_u64 {
            keys.push(ScatterCandidateKey {
                kernel_version: (i % 3) as u32,
                seed: i.rotate_left(32),
                version: (63 - i) as u32,
                salt: (!i).rotate_left(33),
                grid_spacing_mm: i as u32,
                grid: SectorCoord::new(i as i64 - 32, !(i as i64), i64::MIN + i as i64),
            });
        }
        for a in &keys {
            for b in &keys {
                assert_eq!(
                    packed_cmp(&key_words(a), &key_words(b)),
                    a.cmp(b),
                    "{a:?} vs {b:?}"
                );
            }
        }
    }

    fn search_model(keys: &[[u32; 13]], target: &[u32; 13]) -> (bool, u32) {
        let (mut low, mut high, mut comparisons) = (0, keys.len(), 0);
        while low < high {
            let mid = low + (high - low) / 2;
            comparisons += 1;
            match packed_cmp(target, &keys[mid]) {
                Ordering::Equal => return (true, comparisons),
                Ordering::Less => high = mid,
                Ordering::Greater => low = mid + 1,
            }
        }
        (false, comparisons)
    }

    #[test]
    fn binary_search_model_obeys_work_bound_for_hits_and_misses() {
        for d in 0..=256_u32 {
            let keys: Vec<_> = (0..d).map(|x| key_words(&key(i64::from(x) * 2))).collect();
            let bound = u32::BITS - d.leading_zeros();
            for x in -1..=i64::from(d) * 2 + 1 {
                let (found, comparisons) = search_model(&keys, &key_words(&key(x)));
                assert_eq!(found, x >= 0 && x < i64::from(d) * 2 && x % 2 == 0);
                assert!(
                    comparisons <= bound,
                    "D={d} x={x} comparisons={comparisons} bound={bound}"
                );
            }
        }
    }

    #[test]
    fn cache_keeps_full_deletions_and_all_structural_dependencies() {
        let upstream = ScatterCacheKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            config: ScatterConfig {
                seed: 0,
                version: 0,
                salt: 0,
                grid_spacing_mm: 1,
                min_distance_mm: 0,
            },
            region: ScatterRegion {
                min_mm: [0; 3],
                max_mm: [1; 3],
            },
        };
        let absent = ScatterCandidateKey {
            seed: 99,
            ..key(100)
        };
        let deletions = ScatterDeletionSet::new(
            vec![key(0), absent],
            ScatterDeletionBudget {
                max_keys: 2,
                max_key_bytes: 2 * std::mem::size_of::<ScatterCandidateKey>(),
            },
        )
        .unwrap();
        let original = GpuScatterDeletionCacheKey {
            upstream,
            deletions,
            deletion_version: SCATTER_DELETION_VERSION,
            gpu_deletion_version: GPU_SCATTER_DELETION_VERSION,
            key_abi_version: GPU_SCATTER_DELETION_KEY_ABI_VERSION,
            draw_args: [3, 0, 0],
        };
        assert_eq!(original.deletions.keys(), &[key(0), absent]);
        assert_eq!(original, original.clone());
        let mut changed = original.clone();
        changed.deletions = ScatterDeletionSet::default();
        assert_ne!(original, changed);
        for field in 0..15 {
            let mut changed = original.clone();
            match field {
                0 => changed.upstream.kernel_version += 1,
                1 => changed.upstream.config.seed += 1,
                2 => changed.upstream.config.version += 1,
                3 => changed.upstream.config.salt += 1,
                4 => changed.upstream.config.grid_spacing_mm += 1,
                5 => changed.upstream.config.min_distance_mm += 1,
                6 => changed.upstream.region.min_mm[0] -= 1,
                7 => changed.upstream.region.max_mm[2] += 1,
                8 => changed.deletion_version += 1,
                9 => changed.gpu_deletion_version += 1,
                10 => changed.key_abi_version += 1,
                11 => changed.draw_args[0] += 1,
                12 => changed.draw_args[1] += 1,
                13 => changed.draw_args[2] += 1,
                14 => {
                    changed.deletions = ScatterDeletionSet::new(
                        vec![key(0)],
                        ScatterDeletionBudget {
                            max_keys: 1,
                            max_key_bytes: std::mem::size_of::<ScatterCandidateKey>(),
                        },
                    )
                    .unwrap()
                }
                _ => unreachable!(),
            }
            assert_ne!(original, changed, "dependency {field}");
        }
    }
}
