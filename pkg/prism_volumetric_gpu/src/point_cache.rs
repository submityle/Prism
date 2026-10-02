//! `wgpu` compute twin of the baked point-cache per-point time-interpolation
//! sample ([`point_cache`](prism_render_architecture::particle::point_cache),
//! particle design §8.3).
//!
//! The `CPU` golden
//! [`PointCache`](prism_render_architecture::particle::point_cache::PointCache)
//! records `F` frames of `N` points frame-major, with parallel `position`,
//! `velocity` and optional `scalar` channels, and replays them by mapping a
//! playback time to a fractional frame at the cache `FPS` and blending the two
//! bracketing frames under a
//! [`PlaybackMode`](prism_render_architecture::particle::point_cache::PlaybackMode)
//! boundary policy (`Clamp`, `Loop`, `PingPong`). [`GpuPointCache`] is the
//! on-device twin for that stateless, per-point query: one thread resolves one
//! `(point, time, mode)` sample against one shared frame buffer, so a passing
//! real-device parity test is direct evidence the ported kernel reproduces the
//! same interpolated values the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Only the per-point time-interpolation accessors are twinned: the frame
//! location
//! ([`PointCache::frame_pair`](prism_render_architecture::particle::point_cache::PointCache::frame_pair)),
//! the signed-frame boundary resolve (`resolve_frame`), the frame-major
//! `linear_index`, and the linear blend of the two bracketing frames feeding
//! [`PointCache::position_at`](prism_render_architecture::particle::point_cache::PointCache::position_at),
//! [`PointCache::velocity_at`](prism_render_architecture::particle::point_cache::PointCache::velocity_at)
//! and
//! [`PointCache::scalar_at`](prism_render_architecture::particle::point_cache::PointCache::scalar_at).
//! The `RNG`-driven point pick of `emit_point` and the variable-length
//! `sample_frame` / bounds helpers are deliberately **not** twinned here.
//!
//! # Mode codes
//!
//! The reference
//! [`PlaybackMode`](prism_render_architecture::particle::point_cache::PlaybackMode)
//! is carried into the kernel as a `u32`: [`POINT_CACHE_CLAMP`] is `0`,
//! [`POINT_CACHE_LOOP`] is `1` and [`POINT_CACHE_PINGPONG`] is `2`. These
//! classification codes are integer and compared with `==`.
//!
//! # Correctness model
//!
//! The frame location and boundary resolve are integer and `floor`-only, so the
//! chosen frame pair is reproduced exactly for queries clear of a frame
//! boundary. The channel blend threads through one subtract, one multiply and
//! one add, so the interpolated `position`, `velocity` and `scalar` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`) on every continuous channel while asserting an exact
//! `==` on the integer `has_scalar` flag.
//!
//! # Degenerate inputs
//!
//! A single-frame cache (`F == 1`) resolves every signed frame to `0` under all
//! three modes, so playback holds that frame; a single-point cache (`N == 1`)
//! clamps every point index to `0`. Negative and past-the-end times are
//! resolved by the boundary policy exactly as the reference does. A cache with
//! no scalar channel reports `has_scalar == 0` and a zero scalar. An empty query
//! batch short-circuits on the host with no dispatch, since a storage buffer
//! cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `floor`, `abs`,
//! `min`, `clamp`, `+ - * /`, signed/unsigned index arithmetic and the integer
//! `%` operator (truncated remainder, matching Rust) — with no `sin`, `cos`,
//! `exp`, `log`, `pow`, `tan`, no inverse trigonometry, no `sqrt` and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. It has no loop, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::point_cache::PointCache;
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, Queue,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
const WORKGROUP_SIZE: u32 = 64;

/// Mode code for a `Clamp` boundary hold, matching
/// [`PlaybackMode::Clamp`](prism_render_architecture::particle::point_cache::PlaybackMode::Clamp).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
pub const POINT_CACHE_CLAMP: u32 = 0;

/// Mode code for a `Loop` boundary wrap, matching
/// [`PlaybackMode::Loop`](prism_render_architecture::particle::point_cache::PlaybackMode::Loop).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
pub const POINT_CACHE_LOOP: u32 = 1;

/// Mode code for a `PingPong` boundary reflection, matching
/// [`PlaybackMode::PingPong`](prism_render_architecture::particle::point_cache::PlaybackMode::PingPong).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
pub const POINT_CACHE_PINGPONG: u32 = 2;

/// The portable core-`WGSL` point-cache sample kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden
/// [`PointCache::position_at`](prism_render_architecture::particle::point_cache::PointCache::position_at)
/// / `velocity_at` / `scalar_at` path; see the module documentation for the
/// algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
const POINT_CACHE_WGSL: &str = r#"
// Point-cache sample twin: one shared read-only frame buffer, one thread per
// query. Each thread reproduces the per-point time-interpolated sample. It
// mirrors the CPU golden `particle::point_cache` position_at / velocity_at /
// scalar_at path, uses only the portable core-WGSL subset (floor/abs/min/clamp,
// + - * / and integer index/remainder arithmetic), needs no transcendental call
// and takes no optional feature, so it runs unmodified on Metal, Vulkan and
// DX12. It has no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::point_cache；无第三方
// 引擎源码或衍生代码。

// Mode codes mirroring the reference `PlaybackMode`.
const MODE_CLAMP: u32 = 0u;
const MODE_LOOP: u32 = 1u;
const MODE_PINGPONG: u32 = 2u;

struct Params {
    // Number of recorded frames in the shared cache.
    frame_count: u32,
    // Number of points recorded in every frame.
    point_count: u32,
    // Recording rate in frames per second.
    fps: f32,
    // Whether the shared scalar buffer carries a real channel (1) or is a
    // single unused pad lane (0).
    has_scalar: u32,
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Playback time in seconds.
    time: f32,
    // Boundary-policy mode code (MODE_CLAMP .. MODE_PINGPONG).
    mode: u32,
    // Point identity to sample.
    point: u32,
    pad0: u32,
}

struct Sample {
    // Interpolated world position of the chosen point; a pad lane follows.
    position: vec3<f32>,
    pad0: f32,
    // Interpolated velocity of the chosen point; a pad lane follows.
    velocity: vec3<f32>,
    pad1: f32,
    // Interpolated scalar channel, or zero when the cache stores no scalar.
    scalar: f32,
    // 1 when the cache carries a scalar channel, 0 otherwise.
    has_scalar: u32,
    pad2: f32,
    pad3: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> scalars: array<f32>;
@group(0) @binding(4) var<storage, read> queries: array<Query>;
@group(0) @binding(5) var<storage, read_write> results: array<Sample>;

// Resolves a signed frame index into a valid [0, frame_count) frame under a
// boundary policy, mirroring the reference `resolve_frame`. `frame_count` is
// assumed non-zero (guaranteed by the host). The integer `%` operator is a
// truncated remainder in WGSL, matching Rust, so the double-mod positive
// remainder idiom is reproduced directly.
fn resolve_frame(i: i32, frame_count: u32, mode: u32) -> u32 {
    let f = i32(frame_count);
    if (mode == MODE_CLAMP) {
        return u32(clamp(i, 0, f - 1));
    }
    if (mode == MODE_LOOP) {
        return u32(((i % f) + f) % f);
    }
    // MODE_PINGPONG.
    if (frame_count <= 1u) {
        return 0u;
    }
    let period = 2 * (f - 1);
    let m = abs(((i % period) + period) % period);
    if (m < f) {
        return u32(m);
    }
    return u32(period - m);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // frame_pair: locate the fractional frame and its two bracketing frames.
    let t = q.time * params.fps;
    let base = floor(t);
    let frac = t - base;
    let i0 = i32(base);
    let f0 = resolve_frame(i0, params.frame_count, q.mode);
    let f1 = resolve_frame(i0 + 1, params.frame_count, q.mode);

    // linear_index: frame-major, with the point clamped into range.
    let p = min(q.point, params.point_count - 1u);
    let idx0 = f0 * params.point_count + p;
    let idx1 = f1 * params.point_count + p;

    let pos0 = positions[idx0].xyz;
    let pos1 = positions[idx1].xyz;
    let vel0 = velocities[idx0].xyz;
    let vel1 = velocities[idx1].xyz;

    var out: Sample;
    // lerp: a + (b - a) * frac, matching `lerp_vec3` / `lerp_scalar`.
    out.position = pos0 + (pos1 - pos0) * frac;
    out.pad0 = 0.0;
    out.velocity = vel0 + (vel1 - vel0) * frac;
    out.pad1 = 0.0;
    if (params.has_scalar == 0u) {
        out.scalar = 0.0;
        out.has_scalar = 0u;
    } else {
        let s0 = scalars[idx0];
        let s1 = scalars[idx1];
        out.scalar = s0 + (s1 - s0) * frac;
        out.has_scalar = 1u;
    }
    out.pad2 = 0.0;
    out.pad3 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the shared cache header
/// (`frame_count`, `point_count`, `fps`, `has_scalar`) plus the query `count`
/// and three pad words, filling a `32`-byte, `16`-byte-aligned uniform struct
/// matching `Params` in [`POINT_CACHE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of recorded frames in the shared cache.
    frame_count: u32,
    /// Number of points recorded in every frame.
    point_count: u32,
    /// Recording rate in frames per second.
    fps: f32,
    /// `1` when the scalar buffer carries a real channel, `0` otherwise.
    has_scalar: u32,
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A single `vec3` channel lane padded to a `16`-byte `std430` slot, so the host
/// buffer stride matches the `WGSL` `array<vec4<f32>>` the kernel reads.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVec3Lane {
    /// The three stored components.
    xyz: [f32; 3],
    /// Padding lane filling the `16`-byte slot.
    pad: f32,
}

impl GpuVec3Lane {
    /// Packs a [`Vec3`] into its padded `std430` lane.
    fn new(v: Vec3) -> GpuVec3Lane {
        GpuVec3Lane {
            xyz: [v.x, v.y, v.z],
            pad: 0.0,
        }
    }
}

/// `repr(C)` `std430` layout of one result: a `vec3` position slot, a `vec3`
/// velocity slot, then the scalar value and its integer `has_scalar` flag with
/// two pad lanes — `48` bytes matching the `WGSL` `Sample` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    /// Interpolated position.
    position: [f32; 3],
    /// Padding lane after the position.
    pad0: f32,
    /// Interpolated velocity.
    velocity: [f32; 3],
    /// Padding lane after the velocity.
    pad1: f32,
    /// Interpolated scalar value (zero when no scalar channel).
    scalar: f32,
    /// `1` when a scalar channel is present, `0` otherwise.
    has_scalar: u32,
    /// Padding lane.
    pad2: f32,
    /// Padding lane.
    pad3: f32,
}

/// One query for the point-cache twin against the shared uploaded frames: the
/// playback `time` in seconds, the boundary-policy `mode` code and the `point`
/// identity to sample. The `repr(C)` layout matches the `std430` `WGSL` `Query`
/// struct (four words, `16` bytes), so it uploads directly.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct PointCacheQuery {
    /// Playback time in seconds fed to
    /// [`PointCache::frame_pair`](prism_render_architecture::particle::point_cache::PointCache::frame_pair).
    pub time: f32,
    /// Mode code ([`POINT_CACHE_CLAMP`], [`POINT_CACHE_LOOP`] or
    /// [`POINT_CACHE_PINGPONG`]), matching the reference
    /// [`PlaybackMode`](prism_render_architecture::particle::point_cache::PlaybackMode).
    pub mode: u32,
    /// Point identity to sample.
    pub point: u32,
    /// Padding word filling the `16`-byte `std430` slot.
    pub pad0: u32,
}

impl PointCacheQuery {
    /// Builds a query from the playback `time`, the `mode` code and the `point`
    /// identity.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
    #[must_use]
    pub const fn new(time: f32, mode: u32, point: u32) -> PointCacheQuery {
        PointCacheQuery {
            time,
            mode,
            point,
            pad0: 0,
        }
    }
}

/// The interpolated sample the point-cache twin yields for one query: the
/// blended `position` and `velocity`, the blended `scalar`, and a `has_scalar`
/// flag reporting whether the cache carried a scalar channel.
///
/// This mirrors the reference accessor trio: `scalar` is meaningful only when
/// `has_scalar` is `true`, matching the `None` the reference
/// [`PointCache::scalar_at`](prism_render_architecture::particle::point_cache::PointCache::scalar_at)
/// returns for a scalar-less cache.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointCacheSample {
    /// Interpolated world position of the sampled point.
    pub position: Vec3,
    /// Interpolated velocity of the sampled point.
    pub velocity: Vec3,
    /// Interpolated scalar value; meaningful only when `has_scalar` is `true`.
    pub scalar: f32,
    /// Whether the shared cache carried a scalar channel.
    pub has_scalar: bool,
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A compiled, reusable point-cache sample compute pipeline, twinning the
/// per-point time-interpolation accessors of the `CPU` golden
/// [`PointCache`](prism_render_architecture::particle::point_cache::PointCache).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
pub struct GpuPointCache {
    /// Logical device, cloned from the acquiring [`GpuContext`] so a dispatch
    /// needs no borrowed context.
    device: Device,
    /// Submission queue, cloned from the acquiring [`GpuContext`].
    queue: Queue,
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPointCache {
    /// Compiles the point-cache sample kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPointCache {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_point_cache"),
            source: ShaderSource::Wgsl(POINT_CACHE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_point_cache_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_point_cache_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_point_cache_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPointCache {
            device: device.clone(),
            queue: ctx.queue().clone(),
            module,
            layout,
            pipeline,
        }
    }

    /// Samples every query in `queries` against one shared `cache` and returns
    /// one [`PointCacheSample`] per input, in order.
    ///
    /// The shared frame buffers are reconstructed from `cache` in frame-major
    /// order through its public accessors, so every query reads the same
    /// baked layout the reference addresses. Each returned sample matches the
    /// reference
    /// [`PointCache::position_at`](prism_render_architecture::particle::point_cache::PointCache::position_at)
    /// / `velocity_at` / `scalar_at` to within the tolerance documented on this
    /// module, with an exact `has_scalar` flag. An empty `queries` batch returns
    /// an empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::point_cache`。
    #[must_use]
    pub fn sample(&self, queries: &[PointCacheQuery], cache: &PointCache) -> Vec<PointCacheSample> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = &self.device;

        let frame_count = cache.frame_count();
        let point_count = cache.point_count();
        let has_scalar = cache.has_scalar_channel();

        // Reconstruct the shared frame-major channel buffers through the public
        // accessors so the kernel reads exactly the reference layout.
        let lane_count = (frame_count as usize) * (point_count as usize);
        let mut positions: Vec<GpuVec3Lane> = Vec::with_capacity(lane_count);
        let mut velocities: Vec<GpuVec3Lane> = Vec::with_capacity(lane_count);
        let mut scalars: Vec<f32> = Vec::new();
        if has_scalar {
            scalars.reserve(lane_count);
        }
        for frame in 0..frame_count {
            for point in 0..point_count {
                positions.push(GpuVec3Lane::new(cache.position(frame, point)));
                velocities.push(GpuVec3Lane::new(cache.velocity(frame, point)));
                if let Some(s) = cache.scalar(frame, point) {
                    scalars.push(s);
                }
            }
        }
        // A storage buffer cannot be zero-sized; a scalar-less cache uploads a
        // single unused zero lane that no `has_scalar == 1` path ever reads.
        if scalars.is_empty() {
            scalars.push(0.0);
        }

        let params = GpuParams {
            frame_count,
            point_count,
            fps: cache.fps(),
            has_scalar: u32::from(has_scalar),
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_cache_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_cache_positions"),
            contents: bytemuck::cast_slice(&positions),
            usage: BufferUsages::STORAGE,
        });
        let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_cache_velocities"),
            contents: bytemuck::cast_slice(&velocities),
            usage: BufferUsages::STORAGE,
        });
        let scalars_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_cache_scalars"),
            contents: bytemuck::cast_slice(&scalars),
            usage: BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_point_cache_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuSample>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_point_cache_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_point_cache_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: positions_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: velocities_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: scalars_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_point_cache_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_point_cache_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_point_cache_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        self.queue.submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("device poll should complete the submitted work");
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuSample>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter()
            .map(|s| PointCacheSample {
                position: Vec3::new(s.position[0], s.position[1], s.position[2]),
                velocity: Vec3::new(s.velocity[0], s.velocity[1], s.velocity[2]),
                scalar: s.scalar,
                has_scalar: s.has_scalar != 0,
            })
            .collect()
    }
}
