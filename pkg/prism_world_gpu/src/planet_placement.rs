//! Fixed-capacity GPU placement on a cube-sphere heightfield tile.
//!
//! This module owns a GPU candidate stream and does not submit or implicitly
//! wait for work. The CPU validates the complete request and uploads immutable
//! height samples as signed i128 limbs. The shader uses core WGSL `f32` only
//! for sphere projection and rendered positions; exact recipe inputs stay intact.

use prism_world_terrain::{
    estimate_planet_placement_work, HeightfieldTile, PlanetPlacementConfig, PlanetPlacementError,
    PlanetPlacementRegion,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry};
use crate::scatter::{check, dispatch};
use crate::{GpuContext, GpuScatterError};

const PARAM_BYTES: u64 = 192;
const RECORD_BYTES: u64 = 80;
const _: () = assert!(prism_world_terrain::PLANET_PLACEMENT_VERSION == 1);

/// Bounded GPU work for one planetary placement request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPlanetPlacementWorkBudget {
    /// Maximum cell candidates enumerated by the shader.
    pub max_candidate_evaluations: u64,
    /// Conservative four height reads per candidate, including threshold rejects.
    pub max_sample_reads: u64,
}

/// Stable cache identity for a GPU planetary placement request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuPlanetPlacementCacheKey {
    /// Complete CPU recipe.
    pub config: PlanetPlacementConfig,
    /// Owned tile and cell region.
    pub region: PlanetPlacementRegion,
    /// Heightfield revision and samples used by the request.
    pub tile: std::sync::Arc<HeightfieldTile>,
}

/// Recoverable validation and upload failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuPlanetPlacementError {
    /// Invalid CPU planetary placement contract.
    Placement(PlanetPlacementError),
    /// Shared GPU capacity, arithmetic, dispatch, or request errors.
    Gpu(GpuScatterError),
    /// The complete tile snapshot must fit the fixed allocation.
    SurfaceCapacityExceeded { required: usize, capacity: usize },
    /// Conservative height-read work exceeds the request budget.
    SampleBudgetExceeded { required: u64, limit: u64 },
    /// Host upload scratch reservation failed.
    AllocationFailed,
}

impl std::fmt::Display for GpuPlanetPlacementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placement(error) => write!(f, "planet placement contract: {error}"),
            Self::Gpu(error) => write!(f, "planet placement GPU: {error}"),
            Self::SurfaceCapacityExceeded { required, capacity } => {
                write!(
                    f,
                    "planet tile needs {required} samples, capacity {capacity}"
                )
            }
            Self::SampleBudgetExceeded { required, limit } => {
                write!(f, "planet needs {required} height reads, budget {limit}")
            }
            Self::AllocationFailed => {
                f.write_str("planet height upload scratch reservation failed")
            }
        }
    }
}

impl std::error::Error for GpuPlanetPlacementError {}

impl From<PlanetPlacementError> for GpuPlanetPlacementError {
    fn from(error: PlanetPlacementError) -> Self {
        Self::Placement(error)
    }
}

impl From<GpuScatterError> for GpuPlanetPlacementError {
    fn from(error: GpuScatterError) -> Self {
        Self::Gpu(error)
    }
}

/// GPU planetary placement stream with CPU-compatible full-key threshold ranks.
///
/// Geometry is approximate f32 world millimetres, NOT CPU i64 position parity.
/// Each 80-byte record contains tile face/level/x/y (0..4), cell u/v (4..6),
/// position f32 bits (6..9), outward normal f32 bits (9..12), acceptance (12),
/// rank u16 in u32 (13), exact signed i128 bilinear height numerator (14..18,
/// low limb first, denominator four), invalid marker (18), and reserved zero
/// (19). Geometry is not integer parity.
/// Identity is the tile/cell plus version/seed/salt in the full cache key.
/// Compact order is unspecified; only indirect[1] entries are live.
/// Invalid radial distance or coordinate in any threshold survivor invalidates
/// the whole request: finalize clears every acceptance and the indirect count.
/// Submit each encoder before uploading the next request on this object.
/// Preflight errors enqueue no writes; device/OOM errors follow wgpu.
pub struct GpuPlanetPlacement {
    device_identity: std::sync::Arc<()>,
    queue: wgpu::Queue,
    capacity: usize,
    vertex_count: u32,
    max_surface_samples: usize,
    max_groups: u32,
    allocated_bytes: u64,
    params: wgpu::Buffer,
    records: wgpu::Buffer,
    heights: wgpu::Buffer,
    compact: wgpu::Buffer,
    indirect: wgpu::Buffer,
    status: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    height_scratch: Vec<[u32; 4]>,
    request: Option<Request>,
    request_generation: u64,
}

struct Request {
    words: [u32; 48],
    groups: [u32; 2],
    candidate_count: u32,
    key: GpuPlanetPlacementCacheKey,
}

impl GpuPlanetPlacement {
    /// Creates all fixed resources after complete device and byte-budget checks.
    /// Logical bytes: `84*max(capacity,1)+16*max(samples,1)+216`.
    /// Separate reusable host scratch and retained Arc snapshots are excluded.
    pub fn new(
        ctx: &GpuContext,
        capacity: usize,
        max_surface_samples: usize,
        vertex_count: u32,
        max_buffer_bytes: u64,
    ) -> Result<Self, GpuPlanetPlacementError> {
        let limits = ctx.device().limits();
        let allocated_bytes =
            validate_capacity(capacity, max_surface_samples, max_buffer_bytes, &limits)?;
        let mut height_scratch = Vec::new();
        height_scratch
            .try_reserve_exact(max_surface_samples)
            .map_err(|_| GpuPlanetPlacementError::AllocationFailed)?;
        let device = ctx.device();
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet placement params"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let records = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet placement records"),
            size: RECORD_BYTES * capacity.max(1) as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let heights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet placement height samples"),
            size: 16 * max_surface_samples.max(1) as u64,
            usage: storage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compact = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet placement compact indices"),
            size: 4 * capacity.max(1) as u64,
            usage: storage,
            mapped_at_creation: false,
        });
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("planet placement accepted count and failures"),
            size: 8,
            usage: storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let indirect = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("planet placement indirect args"),
            contents: bytemuck::cast_slice(&[vertex_count, 0_u32, 0, 0]),
            usage: storage
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet placement layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet placement pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet placement shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/planet_placement.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet placement compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let finalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet placement finalize"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("finalize_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet placement bindings"),
            layout: &layout,
            entries: &[
                buffer_binding(0, &params),
                buffer_binding(1, &records),
                buffer_binding(2, &heights),
                buffer_binding(3, &compact),
                buffer_binding(4, &status),
                buffer_binding(5, &indirect),
            ],
        });
        Ok(Self {
            device_identity: ctx.device_identity().clone(),
            queue: ctx.queue().clone(),
            capacity,
            vertex_count,
            max_surface_samples,
            max_groups: limits.max_compute_workgroups_per_dimension,
            allocated_bytes,
            params,
            records,
            heights,
            compact,
            indirect,
            status,
            pipeline,
            finalize_pipeline,
            bind_group,
            height_scratch,
            request: None,
            request_generation: 0,
        })
    }

    /// Validates the whole request before changing any uploaded state.
    pub fn write_request(
        &mut self,
        config: PlanetPlacementConfig,
        region: PlanetPlacementRegion,
        tile: &std::sync::Arc<HeightfieldTile>,
        budget: GpuPlanetPlacementWorkBudget,
    ) -> Result<(), GpuPlanetPlacementError> {
        let next = prepare(
            config,
            region,
            tile,
            budget,
            self.capacity,
            self.max_surface_samples,
            self.max_groups,
        )?;
        let generation =
            self.request_generation
                .checked_add(1)
                .ok_or(GpuPlanetPlacementError::Gpu(
                    GpuScatterError::ArithmeticOverflow,
                ))?;
        let old = self.request.as_ref();
        if old.is_none_or(|request| request.words != next.words) {
            self.queue
                .write_buffer(&self.params, 0, bytemuck::cast_slice(&next.words));
        }
        if old.is_none_or(|request| {
            request.key.tile.spec().height_scale_mm() != tile.spec().height_scale_mm()
                || request.key.tile.samples() != tile.samples()
        }) {
            self.height_scratch.clear();
            let scale = i128::from(tile.spec().height_scale_mm());
            self.height_scratch.extend(
                tile.samples()
                    .iter()
                    .map(|&sample| i128_words(i128::from(sample) * scale)),
            );
            self.queue
                .write_buffer(&self.heights, 0, bytemuck::cast_slice(&self.height_scratch));
        }
        self.request = Some(next);
        self.request_generation = generation;
        Ok(())
    }

    /// Encodes placement and final indirect-count passes without submission or readback.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuPlanetPlacementError> {
        let request = self
            .request
            .as_ref()
            .ok_or(GpuScatterError::RequestNotUploaded)?;
        encoder.clear_buffer(&self.indirect, 4, Some(4));
        encoder.clear_buffer(&self.status, 0, None);
        if request.candidate_count != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet placement candidates"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(request.groups[0], request.groups[1], 1);
            drop(pass);
            let mut finalize = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("planet placement indirect finalize"),
                timestamp_writes: None,
            });
            finalize.set_pipeline(&self.finalize_pipeline);
            finalize.set_bind_group(0, &self.bind_group, &[]);
            finalize.dispatch_workgroups(1, 1, 1);
        }
        Ok(())
    }

    /// Candidate-aligned 80-byte records; acceptance gates geometry consumption.
    pub fn records(&self) -> &wgpu::Buffer {
        &self.records
    }
    /// Unspecified-order source slots; live prefix length is indirect[1].
    pub fn compact_indices(&self) -> &wgpu::Buffer {
        &self.compact
    }
    /// Draw arguments, with zero instances if any projected candidate fails.
    pub fn indirect_args(&self) -> &wgpu::Buffer {
        &self.indirect
    }
    /// Two u32 words: provisional accepted count and projection failure count.
    /// On failure the first word is diagnostic only; use indirect for drawing.
    pub fn status(&self) -> &wgpu::Buffer {
        &self.status
    }
    /// Copies the two status words and indirect instance count to a readback buffer.
    /// The destination must provide at least 24 bytes; offsets are status at 0 and
    /// indirect arguments at 8.
    pub fn encode_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        destination: &wgpu::Buffer,
    ) -> Result<(), GpuPlanetPlacementError> {
        if destination.size() < 24 {
            return Err(GpuPlanetPlacementError::Gpu(
                GpuScatterError::BufferBudgetExceeded {
                    required: 24,
                    limit: destination.size(),
                },
            ));
        }
        encoder.copy_buffer_to_buffer(&self.status, 0, destination, 0, 8);
        encoder.copy_buffer_to_buffer(&self.indirect, 0, destination, 8, 16);
        Ok(())
    }
    /// Active primary-slot prefix, independent of GPU acceptance.
    pub fn candidate_count(&self) -> u32 {
        self.request.as_ref().map_or(0, |r| r.candidate_count)
    }
    /// Fixed candidate capacity, never a survivor truncation limit.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Owned logical GPU buffer bytes, excluding driver overhead.
    pub fn allocated_buffer_bytes(&self) -> u64 {
        self.allocated_bytes
    }
    /// Actual reserved host height-upload payload bytes.
    pub fn height_scratch_bytes(&self) -> usize {
        self.height_scratch.capacity() * 16
    }
    /// Structural recipe identity, including the immutable full terrain snapshot.
    pub fn cache_key(&self) -> Option<&GpuPlanetPlacementCacheKey> {
        self.request.as_ref().map(|r| &r.key)
    }
    /// Nonwrapping upload epoch; failed writes leave it unchanged.
    pub fn request_generation(&self) -> u64 {
        self.request_generation
    }

    pub(super) fn device_identity(&self) -> &std::sync::Arc<()> {
        &self.device_identity
    }

    pub(super) fn draw_args(&self) -> [u32; 3] {
        [self.vertex_count, 0, 0]
    }

    pub(super) fn validate_request(
        &self,
        config: PlanetPlacementConfig,
        region: PlanetPlacementRegion,
        tile: &std::sync::Arc<HeightfieldTile>,
        budget: GpuPlanetPlacementWorkBudget,
    ) -> Result<(), GpuPlanetPlacementError> {
        prepare(
            config,
            region,
            tile,
            budget,
            self.capacity,
            self.max_surface_samples,
            self.max_groups,
        )
        .map(|_| ())
    }
}

fn put64(words: &mut [u32; 48], index: usize, value: u64) {
    words[index] = value as u32;
    words[index + 1] = (value >> 32) as u32;
}

fn i128_words(value: i128) -> [u32; 4] {
    let value = value as u128;
    [
        value as u32,
        (value >> 32) as u32,
        (value >> 64) as u32,
        (value >> 96) as u32,
    ]
}

fn prepare(
    config: PlanetPlacementConfig,
    region: PlanetPlacementRegion,
    tile: &std::sync::Arc<HeightfieldTile>,
    budget: GpuPlanetPlacementWorkBudget,
    capacity: usize,
    max_surface_samples: usize,
    max_groups: u32,
) -> Result<Request, GpuPlanetPlacementError> {
    if !(1..=1_i64 << 52).contains(&config.radius_mm) {
        return Err(PlanetPlacementError::InvalidConfig.into());
    }
    let work = estimate_planet_placement_work(region, tile)?;
    if work > budget.max_candidate_evaluations {
        return Err(PlanetPlacementError::CandidateBudgetExceeded {
            required: work,
            limit: budget.max_candidate_evaluations,
        }
        .into());
    }
    let capacity_u64 = u64::try_from(capacity).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    if work > capacity_u64 {
        return Err(GpuScatterError::CapacityExceeded {
            required: work,
            capacity,
        }
        .into());
    }
    let reads = work
        .checked_mul(4)
        .ok_or(GpuScatterError::ArithmeticOverflow)?;
    if reads > budget.max_sample_reads {
        return Err(GpuPlanetPlacementError::SampleBudgetExceeded {
            required: reads,
            limit: budget.max_sample_reads,
        });
    }
    if tile.samples().len() > max_surface_samples {
        return Err(GpuPlanetPlacementError::SurfaceCapacityExceeded {
            required: tile.samples().len(),
            capacity: max_surface_samples,
        });
    }
    let groups = dispatch(work, max_groups)?;
    let count = u32::try_from(work).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let mut words = [0_u32; 48];
    words[0] = count;
    words[1] = groups[0];
    words[2] = tile.spec().samples_per_edge();
    words[3] = u32::from(region.tile.face().index());
    words[4] = u32::from(region.tile.level());
    words[5] = region.tile.x();
    words[6] = region.tile.y();
    words[7] = u32::from(config.acceptance_threshold);
    put64(&mut words, 8, config.seed);
    words[10] = config.version;
    put64(&mut words, 11, config.salt);
    words[13] = groups[1];
    words[14] = region.max_cell[0] - region.min_cell[0];
    words[15] = region.max_cell[1] - region.min_cell[1];
    words[16] = region.min_cell[0];
    words[17] = region.min_cell[1];
    let radial_base = i128::from(config.radius_mm)
        .checked_add(i128::from(config.vertical_offset_mm))
        .and_then(|value| value.checked_mul(4))
        .ok_or(PlanetPlacementError::ArithmeticOverflow)?;
    words[20..24].copy_from_slice(&i128_words(radial_base));
    for (axis, origin) in config.origin_mm.into_iter().enumerate() {
        words[24 + 4 * axis..28 + 4 * axis].copy_from_slice(&i128_words(i128::from(origin)));
    }
    Ok(Request {
        words,
        groups,
        candidate_count: count,
        key: GpuPlanetPlacementCacheKey {
            config,
            region,
            tile: tile.clone(),
        },
    })
}

fn validate_capacity(
    capacity: usize,
    samples: usize,
    budget: u64,
    limits: &wgpu::Limits,
) -> Result<u64, GpuPlanetPlacementError> {
    crate::scatter::validate_capacity(capacity, u64::MAX, limits)?;
    let count = u32::try_from(samples).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let slots = u64::try_from(capacity.max(1)).map_err(|_| GpuScatterError::ArithmeticOverflow)?;
    let sample_slots = u64::from(count.max(1));
    let records_bytes = RECORD_BYTES
        .checked_mul(slots)
        .ok_or(GpuScatterError::ArithmeticOverflow)?;
    let heights_bytes = 16_u64
        .checked_mul(sample_slots)
        .ok_or(GpuScatterError::ArithmeticOverflow)?;
    let compact_bytes = 4_u64
        .checked_mul(slots)
        .ok_or(GpuScatterError::ArithmeticOverflow)?;
    let bytes = records_bytes
        .checked_add(heights_bytes)
        .and_then(|value| value.checked_add(PARAM_BYTES))
        .and_then(|value| value.checked_add(compact_bytes))
        .and_then(|value| value.checked_add(8 + 16))
        .ok_or(GpuScatterError::ArithmeticOverflow)?;
    if bytes > budget {
        return Err(GpuScatterError::BufferBudgetExceeded {
            required: bytes,
            limit: budget,
        }
        .into());
    }
    check(
        "max_storage_buffer_binding_size",
        heights_bytes,
        limits.max_storage_buffer_binding_size,
    )?;
    for size in [heights_bytes, records_bytes, compact_bytes, 8, 16] {
        check("max_buffer_size", size, limits.max_buffer_size)?;
        check(
            "max_storage_buffer_binding_size",
            size,
            limits.max_storage_buffer_binding_size,
        )?;
    }
    check(
        "max_uniform_buffer_binding_size",
        PARAM_BYTES,
        u64::from(limits.max_uniform_buffer_binding_size),
    )?;
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
#[path = "planet_placement_tests.rs"]
mod tests;
