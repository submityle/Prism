//! Fixed-capacity GPU twin of the integer, rebuildable residency scatter kernel.

use prism_world_residency::{
    estimate_scatter_work, ScatterCacheKey, ScatterConfig, ScatterError, ScatterRegion,
    SCATTER_KERNEL_VERSION,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry};
use crate::GpuContext;

const WORKGROUP_SIZE: u32 = 64;
const PARAM_BYTES: u64 = 192;
const RECORD_BYTES: u64 = 64;
// A residency algorithm revision must trigger an explicit GPU twin review,
// never silently produce a new cache key with the old shader's samples.
const _: () = assert!(SCATTER_KERNEL_VERSION == 1);

/// Preflight failures; no request upload occurs on any returned error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuScatterError {
    /// Invalid CPU scatter inputs or unrepresentable worst-case work.
    Scatter(ScatterError),
    /// The candidate count, dispatch or byte size cannot fit the GPU ABI.
    ArithmeticOverflow,
    /// All primary candidates need slots, even when most will be rejected.
    CapacityExceeded { required: u64, capacity: usize },
    /// Includes all owned logical buffers, not physical VRAM or driver metadata.
    BufferBudgetExceeded { required: u64, limit: u64 },
    /// Upload a validated request before recording GPU generation.
    RequestNotUploaded,
    /// The logical device cannot support this allocation or dispatch.
    DeviceLimit {
        limit: &'static str,
        required: u64,
        available: u64,
    },
}

impl std::fmt::Display for GpuScatterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scatter(error) => write!(f, "scatter preflight: {error}"),
            Self::ArithmeticOverflow => f.write_str("GPU scatter arithmetic overflow"),
            Self::CapacityExceeded { required, capacity } => {
                write!(
                    f,
                    "scatter needs {required} candidate slots, capacity {capacity}"
                )
            }
            Self::BufferBudgetExceeded { required, limit } => {
                write!(f, "scatter buffers need {required} bytes, budget {limit}")
            }
            Self::RequestNotUploaded => f.write_str("upload a scatter request before encoding"),
            Self::DeviceLimit {
                limit,
                required,
                available,
            } => {
                write!(
                    f,
                    "{limit}: requires {required}, device supports {available}"
                )
            }
        }
    }
}

impl std::error::Error for GpuScatterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scatter(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ScatterError> for GpuScatterError {
    fn from(error: ScatterError) -> Self {
        Self::Scatter(error)
    }
}

/// Reusable, integer-only GPU generation with no production readback.
///
/// Reuses `prism_world_residency`'s generation inputs, complete cache key and
/// preflight estimate. WGSL emulates signed/unsigned 64-bit integer arithmetic
/// using pairs of `u32`; no shader-i64 or floating-point conversion is needed.
/// Accepted grid keys and millimetre positions exactly match the CPU kernel.
///
/// Every intersecting grid voxel has a deterministic source slot in X/Y/Z
/// lexicographic order (Z fastest). Each 64-byte record contains 16 little-endian
/// words: signed grid XYZ as low/high pairs (0..6), signed position XYZ (6..12),
/// accepted flag (12), unsigned priority low/high (13..15), zero padding (15).
/// Rejected records must not be consumed as samples. Positions of candidates
/// outside the representable world are zero. Full structural candidate identity
/// is grid plus the common generation namespace in [`Self::cache_key`], never
/// the compact index or the priority hash.
///
/// Compact entries are unique source slots in unspecified order. The first
/// indirect `instanceCount` entries are valid; downstream compute/draw may read
/// positions without copying them to the CPU. Projection, surface filtering and
/// renderer material/mesh choice belong to consumers, not this generic kernel.
///
/// Submit each encoded request before another upload/encode on this object.
/// Queue uploads execute before the next submission, not at encoder positions.
/// Sequential submissions require no host wait. Returned validation errors
/// preserve the last uploaded request; device loss/OOM follow normal wgpu error
/// handling and are not represented as recoverable preflight errors here.
pub struct GpuScatter {
    device_identity: std::sync::Arc<()>,
    queue: wgpu::Queue,
    capacity: usize,
    vertex_count: u32,
    request_generation: u64,
    allocated_bytes: u64,
    max_groups: u32,
    params: wgpu::Buffer,
    records: wgpu::Buffer,
    compact: wgpu::Buffer,
    indirect: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    request: Option<Request>,
}

struct Request {
    words: [u32; 48],
    groups: [u32; 2],
    key: ScatterCacheKey,
}

impl GpuScatter {
    /// Allocates buffers, layout and pipeline once, after checking every limit.
    /// Zero capacity uses one unaddressed physical slot. Buffer budget is
    /// `68*max(capacity,1)+208` bytes; uniforms are always 192 bytes, and indirect
    /// arguments 16 bytes. This is not a physical-memory/RSS measurement.
    pub fn new(
        ctx: &GpuContext,
        capacity: usize,
        vertex_count: u32,
        max_buffer_bytes: u64,
    ) -> Result<Self, GpuScatterError> {
        let device = ctx.device();
        let limits = device.limits();
        let bytes = validate_capacity(capacity, max_buffer_bytes, &limits)?;
        let slots = capacity.max(1) as u64;
        let storage = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let records = storage("world scatter candidate records", slots * RECORD_BYTES);
        let compact = storage("world scatter accepted source slots", slots * 4);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world scatter request"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let indirect = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("world scatter indirect arguments"),
            contents: bytemuck::cast_slice(&[vertex_count, 0_u32, 0, 0]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world scatter layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world scatter pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world scatter integer shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scatter.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("world scatter pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world scatter bind group"),
            layout: &layout,
            entries: &[
                buffer_binding(0, &params),
                buffer_binding(1, &records),
                buffer_binding(2, &compact),
                buffer_binding(3, &indirect),
            ],
        });
        Ok(Self {
            device_identity: ctx.device_identity().clone(),
            queue: ctx.queue().clone(),
            capacity,
            vertex_count,
            request_generation: 0,
            allocated_bytes: bytes,
            max_groups: limits.max_compute_workgroups_per_dimension,
            params,
            records,
            compact,
            indirect,
            pipeline,
            bind_group,
            request: None,
        })
    }

    /// O(1) integer preflight then one 192-byte uniform upload when changed.
    /// Work budget bounds all primary plus halo candidate evaluations, before
    /// accepting a request. Capacity bounds *primary slots*, not just survivors;
    /// no budget-dependent output prefix or hidden per-instance host scan exists.
    pub fn write_request(
        &mut self,
        config: ScatterConfig,
        region: ScatterRegion,
        max_candidate_evaluations: u64,
    ) -> Result<(), GpuScatterError> {
        let (request, generation) = prepare_next(
            config,
            region,
            max_candidate_evaluations,
            self.capacity,
            self.max_groups,
            self.request_generation,
        )?;
        if self
            .request
            .as_ref()
            .is_none_or(|previous| previous.words != request.words)
        {
            self.queue
                .write_buffer(&self.params, 0, bytemuck::cast_slice(&request.words));
        }
        self.request = Some(request);
        self.request_generation = generation;
        Ok(())
    }

    /// Sibling GPU stages can validate an entire composite request before any
    /// stage enqueues a queue write. This does not change the active request.
    pub(super) fn validate_request(
        &self,
        config: ScatterConfig,
        region: ScatterRegion,
        max_candidate_evaluations: u64,
    ) -> Result<(), GpuScatterError> {
        prepare_next(
            config,
            region,
            max_candidate_evaluations,
            self.capacity,
            self.max_groups,
            self.request_generation,
        )?;
        Ok(())
    }

    /// Clears the count and records generation/compaction. No allocation, upload,
    /// submit, CPU sample generation, wait, mapping or readback is performed.
    /// Tail storage beyond the current candidate/count prefixes remains stale.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), GpuScatterError> {
        let request = self
            .request
            .as_ref()
            .ok_or(GpuScatterError::RequestNotUploaded)?;
        encoder.clear_buffer(&self.indirect, 4, Some(4));
        if request.words[11] != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("world scatter generation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(request.groups[0], request.groups[1], 1);
        }
        Ok(())
    }

    /// Active primary voxel prefix, not the accepted count (which stays on GPU).
    pub fn candidate_count(&self) -> u32 {
        self.request.as_ref().map_or(0, |r| r.words[11])
    }
    /// Complete reconstruction/invalidation key for the uploaded request.
    pub fn cache_key(&self) -> Option<ScatterCacheKey> {
        self.request.as_ref().map(|r| r.key)
    }
    /// Fixed 64-byte source-slot records. See the struct-level ABI.
    pub fn records(&self) -> &wgpu::Buffer {
        &self.records
    }
    /// GPU-compacted accepted source-slot indices; order is unspecified.
    pub fn compact_indices(&self) -> &wgpu::Buffer {
        &self.compact
    }
    /// Non-indexed `[vertex_count, accepted_count, 0, 0]` GPU draw arguments.
    pub fn indirect_args(&self) -> &wgpu::Buffer {
        &self.indirect
    }
    /// Sum of owned logical buffer sizes; excludes wgpu/driver metadata.
    pub fn allocated_buffer_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Fixed primary source-slot capacity, independent of the active request.
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
}

pub(super) fn check(
    limit: &'static str,
    required: u64,
    available: u64,
) -> Result<(), GpuScatterError> {
    if required > available {
        Err(GpuScatterError::DeviceLimit {
            limit,
            required,
            available,
        })
    } else {
        Ok(())
    }
}

pub(super) fn dispatch(count: u64, max_groups: u32) -> Result<[u32; 2], GpuScatterError> {
    if count == 0 {
        return Ok([0, 0]);
    }
    if max_groups == 0 {
        return Err(GpuScatterError::DeviceLimit {
            limit: "max_compute_workgroups_per_dimension",
            required: 1,
            available: 0,
        });
    }
    let groups = count.div_ceil(u64::from(WORKGROUP_SIZE));
    let x = groups.min(u64::from(max_groups));
    let y = groups.div_ceil(x);
    check(
        "max_compute_workgroups_per_dimension",
        y,
        u64::from(max_groups),
    )?;
    // The shader flattens the padded rectangle in u32, including tail lanes.
    // Exactly 2^32 invocations is safe (last index is u32::MAX); any more would
    // wrap tail indices into the live prefix even if count itself fits u32.
    let invocations = x
        .checked_mul(y)
        .and_then(|groups| groups.checked_mul(u64::from(WORKGROUP_SIZE)))
        .ok_or(GpuScatterError::ArithmeticOverflow)?;
    if invocations > u64::from(u32::MAX) + 1 {
        return Err(GpuScatterError::ArithmeticOverflow);
    }
    Ok([x as u32, y as u32])
}

pub(super) fn validate_capacity(
    capacity: usize,
    budget: u64,
    limits: &wgpu::Limits,
) -> Result<u64, GpuScatterError> {
    let count = u32::try_from(capacity).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let slots = u64::from(count.max(1));
    let records = slots * RECORD_BYTES;
    let bytes = slots * 68 + PARAM_BYTES + 16;
    if bytes > budget {
        return Err(GpuScatterError::BufferBudgetExceeded {
            required: bytes,
            limit: budget,
        });
    }
    check(
        "max_storage_buffer_binding_size",
        records,
        limits.max_storage_buffer_binding_size,
    )?;
    check(
        "max_buffer_size",
        records.max(PARAM_BYTES),
        limits.max_buffer_size,
    )?;
    check(
        "max_uniform_buffer_binding_size",
        PARAM_BYTES,
        limits.max_uniform_buffer_binding_size,
    )?;
    check(
        "max_storage_buffers_per_shader_stage",
        3,
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
        4,
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
    // 64 scan words plus one shared reservation base.
    check(
        "max_compute_workgroup_storage_size",
        260,
        u64::from(limits.max_compute_workgroup_storage_size),
    )?;
    dispatch(
        u64::from(count),
        limits.max_compute_workgroups_per_dimension,
    )?;
    Ok(bytes)
}

fn put64(words: &mut [u32; 48], index: usize, value: u64) {
    words[index] = value as u32;
    words[index + 1] = (value >> 32) as u32;
}

// Validation and upload share exactly the same checks and error ordering.
// Computing the next epoch is pure: only a successful write commits it.
fn prepare_next(
    config: ScatterConfig,
    region: ScatterRegion,
    max_candidate_evaluations: u64,
    capacity: usize,
    max_groups: u32,
    generation: u64,
) -> Result<(Request, u64), GpuScatterError> {
    let request = prepare(
        config,
        region,
        max_candidate_evaluations,
        capacity,
        max_groups,
    )?;
    let next = generation
        .checked_add(1)
        .ok_or(GpuScatterError::ArithmeticOverflow)?;
    Ok((request, next))
}

fn prepare(
    config: ScatterConfig,
    region: ScatterRegion,
    budget: u64,
    capacity: usize,
    max_groups: u32,
) -> Result<Request, GpuScatterError> {
    let estimate = estimate_scatter_work(config, region)?;
    if estimate.max_candidate_evaluations > budget {
        return Err(ScatterError::CandidateBudgetExceeded {
            required: estimate.max_candidate_evaluations,
            limit: budget,
        }
        .into());
    }
    if estimate.primary_candidates > capacity as u64 {
        return Err(GpuScatterError::CapacityExceeded {
            required: estimate.primary_candidates,
            capacity,
        });
    }
    let count = u32::try_from(estimate.primary_candidates)
        .map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let groups = dispatch(u64::from(count), max_groups)?;
    let mut words = [0; 48];
    put64(&mut words, 0, config.seed);
    words[2] = config.version;
    words[3] = config.grid_spacing_mm;
    put64(&mut words, 4, config.salt);
    words[6] = config.min_distance_mm;
    words[7] = config.min_distance_mm.div_ceil(config.grid_spacing_mm);
    words[11] = count;
    let spacing = i64::from(config.grid_spacing_mm);
    for axis in 0..3 {
        put64(&mut words, 18 + axis * 2, region.min_mm[axis] as u64);
        put64(&mut words, 24 + axis * 2, region.max_mm[axis] as u64);
        if count != 0 {
            let min_grid = region.min_mm[axis].div_euclid(spacing);
            let max_grid = (region.max_mm[axis] - 1).div_euclid(spacing);
            let length = i128::from(max_grid) - i128::from(min_grid) + 1;
            words[8 + axis] =
                u32::try_from(length).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
            put64(&mut words, 12 + axis * 2, min_grid as u64);
        }
    }
    put64(&mut words, 30, i64::MIN.div_euclid(spacing) as u64);
    put64(&mut words, 32, i64::MAX.div_euclid(spacing) as u64);
    words[34] = i64::MIN.rem_euclid(spacing) as u32;
    words[35] = i64::MAX.rem_euclid(spacing) as u32;
    words[36] = groups[0];
    Ok(Request {
        words,
        groups,
        key: ScatterCacheKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            config,
            region,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ScatterConfig {
        ScatterConfig {
            seed: u64::MAX,
            version: 19,
            salt: 1 << 48,
            grid_spacing_mm: 17,
            min_distance_mm: 0,
        }
    }

    #[test]
    fn pure_preflight_advances_epoch_without_wrapping_or_changing_semantic_key() {
        let region = ScatterRegion::new([0; 3], [17; 3]).unwrap();
        let (first, next) = prepare_next(config(), region, 1, 1, 65535, 0).unwrap();
        assert_eq!(next, 1);
        let (same, next) = prepare_next(config(), region, 1, 1, 65535, 1).unwrap();
        assert_eq!(next, 2);
        assert_eq!(first.key, same.key);
        assert_eq!(first.words, same.words);
        let (_, last) = prepare_next(config(), region, 1, 1, 65535, u64::MAX - 1).unwrap();
        assert_eq!(last, u64::MAX);
        assert!(matches!(
            prepare_next(config(), region, 1, 1, 65535, u64::MAX),
            Err(GpuScatterError::ArithmeticOverflow)
        ));
        assert!(matches!(
            prepare_next(config(), region, 0, 1, 65535, 0),
            Err(GpuScatterError::Scatter(
                ScatterError::CandidateBudgetExceeded { .. }
            ))
        ));
    }

    #[test]
    fn exact_preflight_and_two_dimensional_dispatch() {
        let region = ScatterRegion::new([-100, -1, -5], [100, 1, 10]).unwrap();
        let request = prepare(config(), region, 1000, 1000, 2).unwrap();
        assert_eq!(request.words[8..12], [12, 2, 2, 48]);
        assert_eq!(request.groups, [1, 1]);
        assert_eq!(dispatch(257, 3).unwrap(), [3, 2]);
        assert!(dispatch(577, 3).is_err());
        assert!(dispatch(1, 0).is_err());
        assert_eq!(dispatch(0, 0).unwrap(), [0, 0]);
        assert_eq!(
            dispatch(u64::from(u32::MAX), 65_536).unwrap(),
            [65_536, 1024]
        );
        assert!(matches!(
            dispatch(u64::from(u32::MAX), 65_535),
            Err(GpuScatterError::ArithmeticOverflow)
        ));
    }

    #[test]
    fn empty_inverted_and_budgeted_requests() {
        let region = ScatterRegion::new([0; 3], [17; 3]).unwrap();
        assert!(matches!(
            prepare(config(), region, 0, 1000, 65535),
            Err(GpuScatterError::Scatter(
                ScatterError::CandidateBudgetExceeded { .. }
            ))
        ));
        assert!(matches!(
            prepare(config(), region, 1000, 0, 65535),
            Err(GpuScatterError::CapacityExceeded { .. })
        ));
        assert!(prepare(
            config(),
            ScatterRegion {
                min_mm: [1; 3],
                max_mm: [0; 3]
            },
            1000,
            1000,
            65535
        )
        .is_err());
        let empty = ScatterRegion::new([i64::MIN; 3], [i64::MIN; 3]).unwrap();
        assert_eq!(prepare(config(), empty, 0, 0, 65535).unwrap().words[11], 0);
        let mut invalid = config();
        invalid.grid_spacing_mm = 0;
        assert!(prepare(invalid, empty, 0, 0, 65535).is_err());
    }

    #[test]
    fn buffer_budgets_and_device_limits_precede_allocation() {
        let limits = wgpu::Limits::downlevel_defaults();
        assert_eq!(validate_capacity(0, 276, &limits).unwrap(), 276);
        assert_eq!(validate_capacity(129, u64::MAX, &limits).unwrap(), 8980);
        assert!(matches!(
            validate_capacity(1, 275, &limits),
            Err(GpuScatterError::BufferBudgetExceeded { .. })
        ));
        assert!(matches!(
            validate_capacity(usize::MAX, u64::MAX, &limits),
            Err(GpuScatterError::ArithmeticOverflow)
        ));
        let mut small = limits.clone();
        small.max_storage_buffers_per_shader_stage = 2;
        assert!(validate_capacity(1, u64::MAX, &small).is_err());
        small = limits.clone();
        small.max_uniform_buffer_binding_size = 191;
        assert!(validate_capacity(1, u64::MAX, &small).is_err());
        small = limits;
        small.max_storage_buffer_binding_size = 128;
        assert!(validate_capacity(3, u64::MAX, &small).is_err());
    }
}
