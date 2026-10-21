//! Fixed-capacity GPU lookup for sparse signed-distance bricks.
//!
//! The CPU owns the sparse grid and remains the authoritative source. This
//! module uploads a sorted, immutable-for-the-dispatch brick table and packed
//! signed samples to a bounded GPU pool. Queries return an explicit value or
//! `Unknown`; an absent brick is never interpreted as empty space.

use std::{
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};
use prism_world_residency::{
    BrickCoord, SdfClipmapConfig, SdfClipmapError, SdfToroidalClipmap, SparseSdfBrickGrid,
};

use crate::gpu::{buffer_binding, storage_entry, uniform_entry, WORKGROUP_SIZE};
use crate::GpuContext;

/// Revision of the sorted sparse-brick lookup algorithm.
pub const GPU_SDF_VERSION: u32 = 1;
/// Revision of the key/meta/sample/query/result buffer ABI.
pub const GPU_SDF_ABI_VERSION: u32 = 1;
/// Revision of the fixed-slot toroidal SDF update and lookup algorithm.
pub const GPU_SDF_TOROIDAL_VERSION: u32 = 1;
/// Revision of the toroidal SDF fixed-slot buffer ABI.
pub const GPU_SDF_TOROIDAL_ABI_VERSION: u32 = 1;

const PARAM_BYTES: u64 = 16;
const RECORD_BYTES: u64 = 16;

/// One integer GPU lookup request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSdfQuery {
    /// Integer world-voxel coordinate.
    pub voxel: [i32; 3],
    /// Clipmap mip level.
    pub mip: u8,
}

/// Result of one GPU lookup. Missing bricks are explicit unknowns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuSdfQueryResult {
    /// The requested brick was resident and contained this quantized sample.
    Value(i16),
    /// The requested brick was not resident.
    Unknown,
}

/// Host wall-clock evidence for one lookup dispatch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuSdfTelemetry {
    /// Number of uploaded sparse bricks active for the dispatch.
    pub brick_count: usize,
    /// Number of queries dispatched.
    pub query_count: usize,
    /// Bytes written for the brick keys, metadata, and packed samples.
    pub brick_upload_bytes: u64,
    /// Bytes written for the query buffer.
    pub query_upload_bytes: u64,
    /// Bytes copied into the mapped result buffer.
    pub readback_bytes: u64,
    /// Number of one-dimensional workgroups submitted.
    pub dispatch_workgroups: u32,
    /// Time from queue submission through completion.
    pub dispatch_wall_time: Duration,
    /// Time spent mapping and copying the result buffer.
    pub readback_wall_time: Duration,
}

/// GPU outputs paired with their dispatch telemetry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuSdfEvaluation {
    /// Results in the same order as the submitted queries.
    pub results: Vec<GpuSdfQueryResult>,
    /// Resource and host timing evidence.
    pub telemetry: GpuSdfTelemetry,
}

/// Recoverable validation failures detected before GPU writes are queued.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuSdfError {
    /// A host count or byte-size calculation overflowed.
    ArithmeticOverflow,
    /// The fixed geometry parameters are outside the supported contract.
    InvalidConfiguration { brick_size: u8, mip_levels: u8 },
    /// A fallible host-side packing allocation failed.
    AllocationFailed,
    /// The grid geometry differs from the fixed GPU pool geometry.
    GridMismatch {
        /// Expected brick edge length.
        expected_brick_size: u8,
        /// Actual brick edge length.
        actual_brick_size: u8,
        /// Expected mip count.
        expected_mip_levels: u8,
        /// Actual mip count.
        actual_mip_levels: u8,
    },
    /// The grid contains more bricks than the fixed pool can hold.
    BrickCapacityExceeded { required: usize, capacity: usize },
    /// CPU clipmap planning rejected the requested center update.
    Clipmap(SdfClipmapError),
    /// A requested brick update is outside the current toroidal window.
    BrickOutsideClipmap { index: usize },
    /// A brick occurs more than once in one staged update.
    DuplicateBrickUpdate { index: usize },
    /// The staged ring upload exceeds its configured logical byte budget.
    UploadBudgetExceeded { required: u64, budget: u64 },
    /// A brick upload requires an initialized clipmap center.
    ClipmapUninitialized,
    /// The query batch exceeds the fixed query pool.
    QueryCapacityExceeded { required: usize, capacity: usize },
    /// A query batch was encoded before `write_queries`.
    QueriesNotUploaded,
    /// A query names a mip outside the fixed grid geometry.
    InvalidQueryMip { index: usize, mip: u8 },
    /// The lookup and context belong to different logical devices.
    DeviceMismatch,
    /// The requested resource or dispatch exceeds a device limit.
    DeviceLimit {
        /// Name of the limit.
        limit: &'static str,
        /// Required value.
        required: u64,
        /// Available value.
        available: u64,
    },
    /// A result status or sample could not be represented by the ABI.
    InvalidGpuResult { index: usize },
}

impl std::fmt::Display for GpuSdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArithmeticOverflow => f.write_str("GPU SDF arithmetic overflow"),
            Self::InvalidConfiguration {
                brick_size,
                mip_levels,
            } => write!(
                f,
                "invalid GPU SDF configuration: brick size {brick_size}, mip levels {mip_levels}"
            ),
            Self::AllocationFailed => f.write_str("GPU SDF host packing allocation failed"),
            Self::GridMismatch {
                expected_brick_size,
                actual_brick_size,
                expected_mip_levels,
                actual_mip_levels,
            } => write!(
                f,
                "GPU SDF grid mismatch: expected brick {expected_brick_size}/mips {expected_mip_levels}, got brick {actual_brick_size}/mips {actual_mip_levels}"
            ),
            Self::BrickCapacityExceeded { required, capacity } => {
                write!(f, "GPU SDF needs {required} bricks, capacity {capacity}")
            }
            Self::Clipmap(error) => write!(f, "GPU SDF clipmap update failed: {error}"),
            Self::BrickOutsideClipmap { index } => {
                write!(f, "GPU SDF brick update {index} is outside the clipmap")
            }
            Self::DuplicateBrickUpdate { index } => {
                write!(f, "GPU SDF brick update {index} duplicates a toroidal slot")
            }
            Self::UploadBudgetExceeded { required, budget } => write!(
                f,
                "GPU SDF clipmap upload requires {required} bytes, budget is {budget} bytes"
            ),
            Self::ClipmapUninitialized => {
                f.write_str("initialize the GPU SDF clipmap center before updating bricks")
            }
            Self::QueryCapacityExceeded { required, capacity } => {
                write!(f, "GPU SDF needs {required} queries, capacity {capacity}")
            }
            Self::QueriesNotUploaded => f.write_str("upload GPU SDF queries before encoding"),
            Self::InvalidQueryMip { index, mip } => {
                write!(f, "GPU SDF query {index} names invalid mip {mip}")
            }
            Self::DeviceMismatch => f.write_str("GPU SDF resource belongs to another device"),
            Self::DeviceLimit {
                limit,
                required,
                available,
            } => write!(f, "{limit}: requires {required}, device supports {available}"),
            Self::InvalidGpuResult { index } => write!(f, "invalid GPU SDF result at {index}"),
        }
    }
}

impl std::error::Error for GpuSdfError {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSdfParams {
    brick_size: u32,
    brick_count: u32,
    query_count: u32,
    sample_words_per_brick: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSdfToroidalParams {
    brick_size: u32,
    side: u32,
    mip_levels: u32,
    query_count: u32,
}

#[derive(Clone, Copy)]
struct StagedToroidalBrick {
    slot: usize,
    key_offset_bytes: u64,
    sample_offset_bytes: u64,
    coord: BrickCoord,
    input_index: usize,
    sample_offset_words: Option<usize>,
}

/// Fixed-capacity sparse-brick lookup resources.
pub struct GpuSdfLookup {
    queue: wgpu::Queue,
    device_identity: Arc<()>,
    max_bricks: usize,
    max_queries: usize,
    brick_size: u8,
    mip_levels: u8,
    sample_words_per_brick: usize,
    brick_count: u32,
    query_count: u32,
    params: wgpu::Buffer,
    keys: wgpu::Buffer,
    metadata: wgpu::Buffer,
    samples: wgpu::Buffer,
    queries: wgpu::Buffer,
    results: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    queries_uploaded: bool,
    brick_upload_bytes: u64,
    query_upload_bytes: u64,
}

impl GpuSdfLookup {
    /// Allocates a bounded sorted-brick table and query/result buffers.
    ///
    /// The pool uses five storage bindings: keys, metadata, packed samples,
    /// queries, and results. A caller that uses downlevel limits must request
    /// at least five storage buffers per shader stage.
    pub fn new(
        context: &GpuContext,
        max_bricks: usize,
        max_queries: usize,
        brick_size: u8,
        mip_levels: u8,
    ) -> Result<Self, GpuSdfError> {
        if !(1..=32).contains(&brick_size) || mip_levels == 0 {
            return Err(GpuSdfError::InvalidConfiguration {
                brick_size,
                mip_levels,
            });
        }
        u32::try_from(max_bricks).map_err(|_| GpuSdfError::ArithmeticOverflow)?;
        u32::try_from(max_queries).map_err(|_| GpuSdfError::ArithmeticOverflow)?;
        let edge = usize::from(brick_size);
        let samples_per_brick = edge
            .checked_mul(edge)
            .and_then(|value| value.checked_mul(edge))
            .ok_or(GpuSdfError::ArithmeticOverflow)?;
        let sample_words_per_brick = samples_per_brick.div_ceil(2);
        let physical_bricks = max_bricks.max(1);
        let physical_queries = max_queries.max(1);
        let key_bytes = bytes_for(physical_bricks, RECORD_BYTES)?;
        let meta_bytes = bytes_for(physical_bricks, RECORD_BYTES)?;
        let sample_bytes = bytes_for(physical_bricks, sample_words_per_brick as u64 * 4)?;
        let query_bytes = bytes_for(physical_queries, RECORD_BYTES)?;
        let result_bytes = bytes_for(physical_queries, RECORD_BYTES)?;
        validate_limits(
            &context.device().limits(),
            key_bytes,
            meta_bytes,
            sample_bytes,
            query_bytes,
            result_bytes,
            max_queries,
        )?;

        let device = context.device();
        let storage = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | usage,
                mapped_at_creation: false,
            })
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU SDF parameters"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let keys = storage(
            "GPU SDF brick keys",
            key_bytes,
            wgpu::BufferUsages::COPY_DST,
        );
        let metadata = storage(
            "GPU SDF brick metadata",
            meta_bytes,
            wgpu::BufferUsages::COPY_DST,
        );
        let samples = storage(
            "GPU SDF packed samples",
            sample_bytes,
            wgpu::BufferUsages::COPY_DST,
        );
        let queries = storage("GPU SDF queries", query_bytes, wgpu::BufferUsages::COPY_DST);
        let results = storage(
            "GPU SDF results",
            result_bytes,
            wgpu::BufferUsages::COPY_SRC,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU SDF lookup shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sdf_lookup.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU SDF lookup layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU SDF lookup pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GPU SDF lookup pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU SDF lookup bind group"),
            layout: &layout,
            entries: &[
                buffer_binding(0, &params),
                buffer_binding(1, &keys),
                buffer_binding(2, &metadata),
                buffer_binding(3, &samples),
                buffer_binding(4, &queries),
                buffer_binding(5, &results),
            ],
        });
        Ok(Self {
            queue: context.queue().clone(),
            device_identity: context.device_identity().clone(),
            max_bricks,
            max_queries,
            brick_size,
            mip_levels,
            sample_words_per_brick,
            brick_count: 0,
            query_count: 0,
            params,
            keys,
            metadata,
            samples,
            queries,
            results,
            pipeline,
            bind_group,
            queries_uploaded: false,
            brick_upload_bytes: 0,
            query_upload_bytes: 0,
        })
    }

    /// Uploads a complete CPU sparse grid after all capacity and packing checks.
    ///
    /// Coordinates are consumed in the grid's deterministic `BrickCoord` order
    /// and the shader performs a binary search. No write is queued when the grid
    /// geometry or host packing fails.
    pub fn upload_grid(&mut self, grid: &SparseSdfBrickGrid) -> Result<(), GpuSdfError> {
        if grid.brick_size() != self.brick_size || grid.mip_levels() != self.mip_levels {
            return Err(GpuSdfError::GridMismatch {
                expected_brick_size: self.brick_size,
                actual_brick_size: grid.brick_size(),
                expected_mip_levels: self.mip_levels,
                actual_mip_levels: grid.mip_levels(),
            });
        }
        let coords = grid.allocated_bricks();
        if coords.len() > self.max_bricks {
            return Err(GpuSdfError::BrickCapacityExceeded {
                required: coords.len(),
                capacity: self.max_bricks,
            });
        }
        let mut keys = Vec::new();
        let mut metadata = Vec::new();
        let mut samples = Vec::new();
        keys.try_reserve(coords.len())
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        metadata
            .try_reserve(coords.len())
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        samples
            .try_reserve(
                coords
                    .len()
                    .checked_mul(self.sample_words_per_brick)
                    .ok_or(GpuSdfError::ArithmeticOverflow)?,
            )
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        for (index, coord) in coords.iter().copied().enumerate() {
            let brick = grid
                .brick_samples(coord)
                .map_err(|_| GpuSdfError::GridMismatch {
                    expected_brick_size: self.brick_size,
                    actual_brick_size: grid.brick_size(),
                    expected_mip_levels: self.mip_levels,
                    actual_mip_levels: grid.mip_levels(),
                })?
                .ok_or(GpuSdfError::ArithmeticOverflow)?;
            if brick.len() != grid.sample_count() {
                return Err(GpuSdfError::ArithmeticOverflow);
            }
            keys.push([coord.x, coord.y, coord.z, i32::from(coord.mip)]);
            let base = index
                .checked_mul(self.sample_words_per_brick)
                .ok_or(GpuSdfError::ArithmeticOverflow)?;
            metadata.push([
                u32::try_from(base).map_err(|_| GpuSdfError::ArithmeticOverflow)?,
                u32::try_from(brick.len()).map_err(|_| GpuSdfError::ArithmeticOverflow)?,
                0,
                0,
            ]);
            for pair in brick.chunks(2) {
                let low = u32::from(u16::from_le_bytes(pair[0].to_le_bytes()));
                let high = pair
                    .get(1)
                    .map(|sample| u32::from(u16::from_le_bytes(sample.to_le_bytes())) << 16)
                    .unwrap_or(0);
                samples.push(low | high);
            }
        }
        if !keys.is_empty() {
            self.queue
                .write_buffer(&self.keys, 0, bytemuck::cast_slice(&keys));
            self.queue
                .write_buffer(&self.metadata, 0, bytemuck::cast_slice(&metadata));
            self.queue
                .write_buffer(&self.samples, 0, bytemuck::cast_slice(&samples));
        }
        self.brick_count =
            u32::try_from(coords.len()).map_err(|_| GpuSdfError::ArithmeticOverflow)?;
        self.write_params();
        self.brick_upload_bytes = (keys.len() * RECORD_BYTES as usize
            + metadata.len() * RECORD_BYTES as usize
            + samples.len() * 4) as u64;
        Ok(())
    }

    /// Uploads a bounded query batch. A failed validation preserves the prior
    /// query count and does not queue a write.
    pub fn write_queries(&mut self, queries: &[GpuSdfQuery]) -> Result<(), GpuSdfError> {
        if queries.len() > self.max_queries {
            return Err(GpuSdfError::QueryCapacityExceeded {
                required: queries.len(),
                capacity: self.max_queries,
            });
        }
        let mut packed = Vec::new();
        packed
            .try_reserve(queries.len())
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        for (index, query) in queries.iter().enumerate() {
            if query.mip >= self.mip_levels {
                return Err(GpuSdfError::InvalidQueryMip {
                    index,
                    mip: query.mip,
                });
            }
            packed.push([
                query.voxel[0],
                query.voxel[1],
                query.voxel[2],
                i32::from(query.mip),
            ]);
        }
        if !packed.is_empty() {
            self.queue
                .write_buffer(&self.queries, 0, bytemuck::cast_slice(&packed));
        }
        self.query_count =
            u32::try_from(queries.len()).map_err(|_| GpuSdfError::ArithmeticOverflow)?;
        self.queries_uploaded = true;
        self.write_params();
        self.query_upload_bytes = (packed.len() * RECORD_BYTES as usize) as u64;
        Ok(())
    }

    /// Encodes only the lookup dispatch into the caller's command encoder.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), GpuSdfError> {
        if self.query_count != 0 && !self.queries_uploaded {
            return Err(GpuSdfError::QueriesNotUploaded);
        }
        if self.query_count == 0 {
            return Ok(());
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU SDF lookup pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.query_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        Ok(())
    }

    /// Submits, waits, and reads a query batch for a strict parity harness.
    pub fn evaluate_with_readback(
        &mut self,
        context: &GpuContext,
        queries: &[GpuSdfQuery],
    ) -> Result<GpuSdfEvaluation, GpuSdfError> {
        if !Arc::ptr_eq(&self.device_identity, context.device_identity()) {
            return Err(GpuSdfError::DeviceMismatch);
        }
        self.write_queries(queries)?;
        let readback_bytes = bytes_for(queries.len().max(1), RECORD_BYTES)?;
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU SDF result readback"),
            size: readback_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GPU SDF lookup encoder"),
                });
        self.encode(&mut encoder)?;
        if !queries.is_empty() {
            encoder.copy_buffer_to_buffer(
                &self.results,
                0,
                &staging,
                0,
                queries.len() as u64 * RECORD_BYTES,
            );
        }
        let dispatch_start = Instant::now();
        context.queue().submit([encoder.finish()]);
        context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| GpuSdfError::DeviceMismatch)?;
        let dispatch_wall_time = dispatch_start.elapsed();

        let readback_start = Instant::now();
        let (sender, receiver) = mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| GpuSdfError::DeviceMismatch)?;
        receiver
            .recv()
            .map_err(|_| GpuSdfError::DeviceMismatch)?
            .map_err(|_| GpuSdfError::DeviceMismatch)?;
        let bytes = staging
            .get_mapped_range(..)
            .map_err(|_| GpuSdfError::DeviceMismatch)?
            .to_vec();
        staging.unmap();
        let mut results = Vec::new();
        results
            .try_reserve(queries.len())
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        for (index, record) in bytes
            .chunks_exact(RECORD_BYTES as usize)
            .enumerate()
            .take(queries.len())
        {
            let status = u32::from_le_bytes(record[0..4].try_into().unwrap());
            let value = i32::from_le_bytes(record[4..8].try_into().unwrap());
            results.push(match status {
                0 if value == 0 => GpuSdfQueryResult::Unknown,
                1 if (i16::MIN as i32..=i16::MAX as i32).contains(&value) => {
                    GpuSdfQueryResult::Value(value as i16)
                }
                _ => return Err(GpuSdfError::InvalidGpuResult { index }),
            });
        }
        Ok(GpuSdfEvaluation {
            results,
            telemetry: GpuSdfTelemetry {
                brick_count: self.brick_count as usize,
                query_count: queries.len(),
                brick_upload_bytes: self.brick_upload_bytes,
                query_upload_bytes: self.query_upload_bytes,
                readback_bytes: queries.len() as u64 * RECORD_BYTES,
                dispatch_workgroups: queries.len().div_ceil(WORKGROUP_SIZE as usize) as u32,
                dispatch_wall_time,
                readback_wall_time: readback_start.elapsed(),
            },
        })
    }

    /// Number of currently uploaded bricks.
    #[must_use]
    pub fn brick_count(&self) -> usize {
        self.brick_count as usize
    }

    /// Number of active queries.
    #[must_use]
    pub fn query_count(&self) -> usize {
        self.query_count as usize
    }

    /// Logical owned buffer bytes, excluding driver allocations and staging.
    #[must_use]
    pub fn allocated_buffer_bytes(&self) -> u64 {
        self.params.size()
            + self.keys.size()
            + self.metadata.size()
            + self.samples.size()
            + self.queries.size()
            + self.results.size()
    }

    /// Storage buffer containing `[status, value, 0, 0]` result records.
    ///
    /// Only the first `query_count()` records are valid after `encode` has
    /// executed in the caller's submission.
    #[must_use]
    pub fn results(&self) -> &wgpu::Buffer {
        &self.results
    }

    fn write_params(&self) {
        let params = GpuSdfParams {
            brick_size: u32::from(self.brick_size),
            brick_count: self.brick_count,
            query_count: self.query_count,
            sample_words_per_brick: self.sample_words_per_brick as u32,
        };
        self.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }
}

struct StagedToroidalUpdate {
    slots: Vec<StagedToroidalBrick>,
    keys: Vec<[i32; 4]>,
    metadata: Vec<[u32; 4]>,
    samples: Vec<u32>,
    upload_bytes: u64,
}

/// Fixed-slot GPU SDF ring driven by [`SdfToroidalClipmap`] deltas.
///
/// Unlike [`GpuSdfLookup`], this resource uses direct modulo slot addressing,
/// compares the complete key to reject stale aliases, and uploads only changed
/// slots. The CPU sparse grid remains authoritative; missing bricks invalidate
/// their slot metadata and therefore query as `Unknown`.
pub struct GpuSdfToroidalLookup {
    queue: wgpu::Queue,
    planner: SdfToroidalClipmap,
    max_queries: usize,
    max_upload_bytes: u64,
    brick_size: u8,
    side: u32,
    mip_levels: u8,
    slot_count: usize,
    sample_words_per_brick: usize,
    active_bricks: usize,
    valid_slots: Vec<bool>,
    query_count: u32,
    params: wgpu::Buffer,
    keys: wgpu::Buffer,
    metadata: wgpu::Buffer,
    samples: wgpu::Buffer,
    queries: wgpu::Buffer,
    results: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    queries_uploaded: bool,
    brick_upload_bytes: u64,
    query_upload_bytes: u64,
}

impl GpuSdfToroidalLookup {
    /// Allocates fixed per-mip toroidal slots and bounded query resources.
    ///
    /// `max_upload_bytes` bounds the physical key, metadata and resident sample
    /// bytes written by one center roll or explicit dirty-brick update. The
    /// clipmap's own config separately bounds its logical coordinate delta.
    pub fn new(
        context: &GpuContext,
        config: SdfClipmapConfig,
        brick_size: u8,
        max_queries: usize,
        max_upload_bytes: u64,
    ) -> Result<Self, GpuSdfError> {
        if !(1..=32).contains(&brick_size) {
            return Err(GpuSdfError::InvalidConfiguration {
                brick_size,
                mip_levels: config.mip_levels,
            });
        }
        let planner = SdfToroidalClipmap::new(config).map_err(GpuSdfError::Clipmap)?;
        let side = u64::from(config.ring_radius)
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(GpuSdfError::ArithmeticOverflow)?;
        let side = u32::try_from(side).map_err(|_| GpuSdfError::ArithmeticOverflow)?;
        let slot_count = config.layout_cell_count().map_err(GpuSdfError::Clipmap)?;
        u32::try_from(slot_count).map_err(|_| GpuSdfError::ArithmeticOverflow)?;
        u32::try_from(max_queries).map_err(|_| GpuSdfError::ArithmeticOverflow)?;

        let edge = usize::from(brick_size);
        let samples_per_brick = edge
            .checked_mul(edge)
            .and_then(|value| value.checked_mul(edge))
            .ok_or(GpuSdfError::ArithmeticOverflow)?;
        let sample_words_per_brick = samples_per_brick.div_ceil(2);
        let physical_slots = slot_count.max(1);
        let physical_queries = max_queries.max(1);
        let key_bytes = bytes_for(physical_slots, RECORD_BYTES)?;
        let metadata_bytes = bytes_for(physical_slots, RECORD_BYTES)?;
        let sample_stride = u64::try_from(sample_words_per_brick)
            .map_err(|_| GpuSdfError::ArithmeticOverflow)?
            .checked_mul(4)
            .ok_or(GpuSdfError::ArithmeticOverflow)?;
        let sample_bytes = bytes_for(physical_slots, sample_stride)?;
        let query_bytes = bytes_for(physical_queries, RECORD_BYTES)?;
        let result_bytes = bytes_for(physical_queries, RECORD_BYTES)?;
        validate_limits(
            &context.device().limits(),
            key_bytes,
            metadata_bytes,
            sample_bytes,
            query_bytes,
            result_bytes,
            max_queries,
        )?;

        let mut valid_slots = Vec::new();
        valid_slots
            .try_reserve_exact(slot_count)
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        valid_slots.resize(slot_count, false);

        let device = context.device();
        let storage = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | usage,
                mapped_at_creation: false,
            })
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU SDF toroidal parameters"),
            size: PARAM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let keys = storage(
            "GPU SDF toroidal brick keys",
            key_bytes,
            wgpu::BufferUsages::COPY_DST,
        );
        let metadata = storage(
            "GPU SDF toroidal brick metadata",
            metadata_bytes,
            wgpu::BufferUsages::COPY_DST,
        );
        let samples = storage(
            "GPU SDF toroidal packed samples",
            sample_bytes,
            wgpu::BufferUsages::COPY_DST,
        );
        let queries = storage(
            "GPU SDF toroidal queries",
            query_bytes,
            wgpu::BufferUsages::COPY_DST,
        );
        let results = storage(
            "GPU SDF toroidal results",
            result_bytes,
            wgpu::BufferUsages::COPY_SRC,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU SDF toroidal lookup shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/sdf_toroidal_lookup.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU SDF toroidal lookup layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU SDF toroidal lookup pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GPU SDF toroidal lookup pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU SDF toroidal lookup bind group"),
            layout: &layout,
            entries: &[
                buffer_binding(0, &params),
                buffer_binding(1, &keys),
                buffer_binding(2, &metadata),
                buffer_binding(3, &samples),
                buffer_binding(4, &queries),
                buffer_binding(5, &results),
            ],
        });
        let lookup = Self {
            queue: context.queue().clone(),
            planner,
            max_queries,
            max_upload_bytes,
            brick_size,
            side,
            mip_levels: config.mip_levels,
            slot_count,
            sample_words_per_brick,
            active_bricks: 0,
            valid_slots,
            query_count: 0,
            params,
            keys,
            metadata,
            samples,
            queries,
            results,
            pipeline,
            bind_group,
            queries_uploaded: false,
            brick_upload_bytes: 0,
            query_upload_bytes: 0,
        };
        lookup.write_params();
        Ok(lookup)
    }

    /// Rolls the toroidal window and stages only newly entering slot contents.
    ///
    /// All geometry, work, byte budgets and host allocations are checked
    /// before queue writes; a returned error leaves both the CPU center and GPU
    /// slot contents unchanged. Entering coordinates include every slot whose
    /// full key changed; absent source bricks invalidate metadata only.
    pub fn update_center(
        &mut self,
        grid: &SparseSdfBrickGrid,
        center: BrickCoord,
    ) -> Result<u64, GpuSdfError> {
        self.validate_grid(grid)?;
        let mut next_planner = self.planner.try_clone().map_err(GpuSdfError::Clipmap)?;
        let delta = next_planner
            .update_center(center)
            .map_err(GpuSdfError::Clipmap)?;
        let coord_count = delta.layers().iter().try_fold(0_usize, |total, layer| {
            total
                .checked_add(layer.len())
                .ok_or(GpuSdfError::ArithmeticOverflow)
        })?;
        let mut coords = Vec::new();
        coords
            .try_reserve_exact(coord_count)
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        coords.extend(delta.layers().iter().flatten().copied());

        let staged = self.prepare_bricks(grid, &coords, &next_planner)?;
        let written = staged.upload_bytes;
        self.apply_staged(staged);
        self.planner = next_planner;
        Ok(written)
    }

    /// Uploads edited brick contents for the currently resident toroidal ring.
    ///
    /// Duplicate or out-of-window keys, grid mismatch, byte-budget failure and
    /// allocation failure are rejected before any queue write.
    pub fn upload_bricks(
        &mut self,
        grid: &SparseSdfBrickGrid,
        coords: &[BrickCoord],
    ) -> Result<u64, GpuSdfError> {
        self.validate_grid(grid)?;
        if self.planner.center().is_none() {
            return Err(GpuSdfError::ClipmapUninitialized);
        }
        let staged = self.prepare_bricks(grid, coords, &self.planner)?;
        let written = staged.upload_bytes;
        self.apply_staged(staged);
        Ok(written)
    }

    fn validate_grid(&self, grid: &SparseSdfBrickGrid) -> Result<(), GpuSdfError> {
        if grid.brick_size() != self.brick_size || grid.mip_levels() != self.mip_levels {
            return Err(GpuSdfError::GridMismatch {
                expected_brick_size: self.brick_size,
                actual_brick_size: grid.brick_size(),
                expected_mip_levels: self.mip_levels,
                actual_mip_levels: grid.mip_levels(),
            });
        }
        Ok(())
    }

    fn prepare_bricks(
        &self,
        grid: &SparseSdfBrickGrid,
        coords: &[BrickCoord],
        planner: &SdfToroidalClipmap,
    ) -> Result<StagedToroidalUpdate, GpuSdfError> {
        if coords.len() > self.slot_count {
            return Err(GpuSdfError::BrickCapacityExceeded {
                required: coords.len(),
                capacity: self.slot_count,
            });
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(coords.len())
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        for (index, coord) in coords.iter().copied().enumerate() {
            let slot = planner
                .slot_index(coord)
                .ok_or(GpuSdfError::BrickOutsideClipmap { index })?;
            let key_offset_bytes = bytes_for(slot, RECORD_BYTES)?;
            let sample_stride = u64::try_from(self.sample_words_per_brick)
                .map_err(|_| GpuSdfError::ArithmeticOverflow)?
                .checked_mul(4)
                .ok_or(GpuSdfError::ArithmeticOverflow)?;
            let sample_offset_bytes = bytes_for(slot, sample_stride)?;
            slots.push(StagedToroidalBrick {
                slot,
                key_offset_bytes,
                sample_offset_bytes,
                coord,
                input_index: index,
                sample_offset_words: None,
            });
        }
        slots.sort_unstable_by_key(|entry| entry.slot);
        for pair in slots.windows(2) {
            if pair[0].slot == pair[1].slot {
                return Err(GpuSdfError::DuplicateBrickUpdate {
                    index: pair[1].input_index,
                });
            }
        }

        let mut present_count = 0_usize;
        for entry in &mut slots {
            if let Some(samples) = grid
                .brick_samples(entry.coord)
                .map_err(|_| GpuSdfError::ArithmeticOverflow)?
            {
                if samples.len() != grid.sample_count() {
                    return Err(GpuSdfError::ArithmeticOverflow);
                }
                entry.sample_offset_words = Some(
                    present_count
                        .checked_mul(self.sample_words_per_brick)
                        .ok_or(GpuSdfError::ArithmeticOverflow)?,
                );
                present_count = present_count
                    .checked_add(1)
                    .ok_or(GpuSdfError::ArithmeticOverflow)?;
            }
        }

        let record_bytes = u64::try_from(slots.len())
            .map_err(|_| GpuSdfError::ArithmeticOverflow)?
            .checked_mul(2 * RECORD_BYTES)
            .ok_or(GpuSdfError::ArithmeticOverflow)?;
        let sample_stride = u64::try_from(self.sample_words_per_brick)
            .map_err(|_| GpuSdfError::ArithmeticOverflow)?
            .checked_mul(4)
            .ok_or(GpuSdfError::ArithmeticOverflow)?;
        let sample_bytes = bytes_for(present_count, sample_stride)?;
        let upload_bytes = record_bytes
            .checked_add(sample_bytes)
            .ok_or(GpuSdfError::ArithmeticOverflow)?;
        if upload_bytes > self.max_upload_bytes {
            return Err(GpuSdfError::UploadBudgetExceeded {
                required: upload_bytes,
                budget: self.max_upload_bytes,
            });
        }

        let mut keys = Vec::new();
        let mut metadata = Vec::new();
        let sample_word_count = present_count
            .checked_mul(self.sample_words_per_brick)
            .ok_or(GpuSdfError::ArithmeticOverflow)?;
        let mut samples = Vec::new();
        keys.try_reserve_exact(slots.len())
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        metadata
            .try_reserve_exact(slots.len())
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        samples
            .try_reserve_exact(sample_word_count)
            .map_err(|_| GpuSdfError::AllocationFailed)?;

        for entry in &slots {
            keys.push([
                entry.coord.x,
                entry.coord.y,
                entry.coord.z,
                i32::from(entry.coord.mip),
            ]);
            let base = entry
                .slot
                .checked_mul(self.sample_words_per_brick)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(GpuSdfError::ArithmeticOverflow)?;
            if entry.sample_offset_words.is_some() {
                metadata.push([
                    base,
                    u32::try_from(grid.sample_count())
                        .map_err(|_| GpuSdfError::ArithmeticOverflow)?,
                    0,
                    0,
                ]);
                let brick = grid
                    .brick_samples(entry.coord)
                    .map_err(|_| GpuSdfError::ArithmeticOverflow)?
                    .ok_or(GpuSdfError::ArithmeticOverflow)?;
                for pair in brick.chunks(2) {
                    let low = u32::from(u16::from_le_bytes(pair[0].to_le_bytes()));
                    let high = pair
                        .get(1)
                        .map(|sample| u32::from(u16::from_le_bytes(sample.to_le_bytes())) << 16)
                        .unwrap_or(0);
                    samples.push(low | high);
                }
            } else {
                metadata.push([base, 0, 0, 0]);
            }
        }
        debug_assert_eq!(samples.len(), sample_word_count);
        Ok(StagedToroidalUpdate {
            slots,
            keys,
            metadata,
            samples,
            upload_bytes,
        })
    }

    fn apply_staged(&mut self, staged: StagedToroidalUpdate) {
        let mut start = 0_usize;
        while start < staged.slots.len() {
            let mut end = start + 1;
            while end < staged.slots.len()
                && staged.slots[end - 1].slot.checked_add(1) == Some(staged.slots[end].slot)
            {
                end += 1;
            }
            let byte_start = staged.slots[start].key_offset_bytes;
            self.queue.write_buffer(
                &self.keys,
                byte_start,
                bytemuck::cast_slice(&staged.keys[start..end]),
            );
            self.queue.write_buffer(
                &self.metadata,
                byte_start,
                bytemuck::cast_slice(&staged.metadata[start..end]),
            );

            let mut cursor = start;
            while cursor < end {
                let Some(sample_start) = staged.slots[cursor].sample_offset_words else {
                    cursor += 1;
                    continue;
                };
                let mut sample_end = cursor + 1;
                while sample_end < end
                    && staged.slots[sample_end].sample_offset_words
                        == staged.slots[sample_end - 1]
                            .sample_offset_words
                            .map(|offset| offset + self.sample_words_per_brick)
                    && staged.slots[sample_end - 1].slot.checked_add(1)
                        == Some(staged.slots[sample_end].slot)
                {
                    sample_end += 1;
                }
                let sample_word_end = staged.slots[sample_end - 1]
                    .sample_offset_words
                    .expect("sample range begins with a resident slot")
                    + self.sample_words_per_brick;
                self.queue.write_buffer(
                    &self.samples,
                    staged.slots[cursor].sample_offset_bytes,
                    bytemuck::cast_slice(&staged.samples[sample_start..sample_word_end]),
                );
                cursor = sample_end;
            }

            for entry in &staged.slots[start..end] {
                let is_present = entry.sample_offset_words.is_some();
                let was_present = self.valid_slots[entry.slot];
                match (was_present, is_present) {
                    (false, true) => self.active_bricks += 1,
                    (true, false) => self.active_bricks -= 1,
                    _ => {}
                }
                self.valid_slots[entry.slot] = is_present;
            }
            start = end;
        }
        self.brick_upload_bytes = staged.upload_bytes;
    }

    /// Uploads a bounded query batch without submitting work.
    pub fn write_queries(&mut self, queries: &[GpuSdfQuery]) -> Result<(), GpuSdfError> {
        if queries.len() > self.max_queries {
            return Err(GpuSdfError::QueryCapacityExceeded {
                required: queries.len(),
                capacity: self.max_queries,
            });
        }
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(queries.len())
            .map_err(|_| GpuSdfError::AllocationFailed)?;
        for (index, query) in queries.iter().enumerate() {
            if query.mip >= self.mip_levels {
                return Err(GpuSdfError::InvalidQueryMip {
                    index,
                    mip: query.mip,
                });
            }
            packed.push([
                query.voxel[0],
                query.voxel[1],
                query.voxel[2],
                i32::from(query.mip),
            ]);
        }
        if !packed.is_empty() {
            self.queue
                .write_buffer(&self.queries, 0, bytemuck::cast_slice(&packed));
        }
        self.query_count =
            u32::try_from(queries.len()).map_err(|_| GpuSdfError::ArithmeticOverflow)?;
        self.queries_uploaded = true;
        self.write_params();
        self.query_upload_bytes = (packed.len() * RECORD_BYTES as usize) as u64;
        Ok(())
    }

    /// Encodes only the toroidal lookup dispatch into the caller's encoder.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) -> Result<(), GpuSdfError> {
        if self.query_count != 0 && !self.queries_uploaded {
            return Err(GpuSdfError::QueriesNotUploaded);
        }
        if self.query_count == 0 {
            return Ok(());
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GPU SDF toroidal lookup pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.query_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        Ok(())
    }

    /// Returns the number of currently valid ring slots.
    #[must_use]
    pub fn brick_count(&self) -> usize {
        self.active_bricks
    }

    /// Returns the fixed number of physical slots across all mips.
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Returns the last physical upload byte count.
    #[must_use]
    pub const fn last_upload_bytes(&self) -> u64 {
        self.brick_upload_bytes
    }

    /// Returns the last query upload byte count.
    #[must_use]
    pub const fn last_query_upload_bytes(&self) -> u64 {
        self.query_upload_bytes
    }

    /// Returns the CPU toroidal planner's committed center.
    #[must_use]
    pub const fn center(&self) -> Option<BrickCoord> {
        self.planner.center()
    }

    /// Logical owned buffer bytes, excluding driver allocations and staging.
    #[must_use]
    pub fn allocated_buffer_bytes(&self) -> u64 {
        self.params.size()
            + self.keys.size()
            + self.metadata.size()
            + self.samples.size()
            + self.queries.size()
            + self.results.size()
    }

    /// Storage buffer containing `[status, value, 0, 0]` result records.
    #[must_use]
    pub fn results(&self) -> &wgpu::Buffer {
        &self.results
    }

    fn write_params(&self) {
        let params = GpuSdfToroidalParams {
            brick_size: u32::from(self.brick_size),
            side: self.side,
            mip_levels: u32::from(self.mip_levels),
            query_count: self.query_count,
        };
        self.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }
}

fn bytes_for(count: usize, stride: u64) -> Result<u64, GpuSdfError> {
    u64::try_from(count)
        .map_err(|_| GpuSdfError::ArithmeticOverflow)?
        .checked_mul(stride)
        .ok_or(GpuSdfError::ArithmeticOverflow)
}

fn require_limit(limit: &'static str, required: u64, available: u64) -> Result<(), GpuSdfError> {
    if required > available {
        Err(GpuSdfError::DeviceLimit {
            limit,
            required,
            available,
        })
    } else {
        Ok(())
    }
}

fn validate_limits(
    limits: &wgpu::Limits,
    key_bytes: u64,
    meta_bytes: u64,
    sample_bytes: u64,
    query_bytes: u64,
    result_bytes: u64,
    max_queries: usize,
) -> Result<(), GpuSdfError> {
    for (name, bytes) in [
        ("keys max_buffer_size", key_bytes),
        ("metadata max_buffer_size", meta_bytes),
        ("samples max_buffer_size", sample_bytes),
        ("queries max_buffer_size", query_bytes),
        ("results max_buffer_size", result_bytes),
    ] {
        require_limit(name, bytes, limits.max_buffer_size)?;
        require_limit(
            "max_storage_buffer_binding_size",
            bytes,
            limits.max_storage_buffer_binding_size,
        )?;
    }
    require_limit(
        "max_uniform_buffer_binding_size",
        PARAM_BYTES,
        limits.max_uniform_buffer_binding_size,
    )?;
    require_limit(
        "max_storage_buffers_per_shader_stage",
        5,
        u64::from(limits.max_storage_buffers_per_shader_stage),
    )?;
    require_limit(
        "max_uniform_buffers_per_shader_stage",
        1,
        u64::from(limits.max_uniform_buffers_per_shader_stage),
    )?;
    require_limit(
        "max_bindings_per_bind_group",
        6,
        u64::from(limits.max_bindings_per_bind_group),
    )?;
    require_limit("max_bind_groups", 1, u64::from(limits.max_bind_groups))?;
    require_limit(
        "max_compute_workgroup_size_x",
        u64::from(WORKGROUP_SIZE),
        u64::from(limits.max_compute_workgroup_size_x),
    )?;
    require_limit(
        "max_compute_invocations_per_workgroup",
        u64::from(WORKGROUP_SIZE),
        u64::from(limits.max_compute_invocations_per_workgroup),
    )?;
    require_limit(
        "max_compute_workgroups_per_dimension",
        u64::try_from(max_queries.div_ceil(WORKGROUP_SIZE as usize).max(1))
            .map_err(|_| GpuSdfError::ArithmeticOverflow)?,
        u64::from(limits.max_compute_workgroups_per_dimension),
    )
}
