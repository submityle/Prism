//! `wgpu` compute twin of the per-particle clip-space screen motion vector
//! ([`screen_motion_vector`](prism_render_architecture::particle::motion_vectors::screen_motion_vector),
//! design §21, "屏幕运动矢量").
//!
//! Temporal upsamplers (`TAA`, `TSR`, `DLSS`-style reprojection) reproject the
//! previous frame onto the current one by following each shaded fragment's
//! screen-space displacement. For a particle that displacement is obtained by
//! projecting its current and previous world positions through the current and
//! previous (jittered) view-projection matrices, applying the perspective
//! divide, removing the baked per-frame jitter, mapping each
//! normalized-device-coordinate (`NDC`) position to `UV` under the framebuffer
//! convention, and taking the `UV` difference.
//!
//! The `CPU` golden
//! [`particle::motion_vectors`](prism_render_architecture::particle::motion_vectors)
//! owns that math per particle; [`GpuMotionVectors`] is the on-device twin that
//! runs one thread per particle and returns the same per-lane
//! [`Option<ScreenMotionVector>`](prism_render_architecture::particle::motion_vectors::ScreenMotionVector)
//! that the batch form
//! [`screen_motion_vectors`](prism_render_architecture::particle::motion_vectors::screen_motion_vectors)
//! produces. A passing real-device parity test is therefore direct evidence the
//! ported kernel reprojects the same particles the reference does, not merely
//! that its shader compiles.
//!
//! Only the per-element, embarrassingly-parallel *compute* path is ported. The
//! scalar *decision* layers that live alongside it in the reference —
//! `flipbook` stability, the reactive-mask history weight, the order-independent
//! transparency (`OIT`) contract, and the `fp16` / `snorm16` bandwidth encodings
//! — are per-effect branch logic, not per-particle arithmetic, and stay on the
//! `CPU`.
//!
//! # Step-for-step parity
//!
//! The kernel mirrors the reference exactly: the same column-major
//! (`cols[col][row]`) homogeneous transform
//! [`Mat4::transform_point`](prism_render_architecture::particle::motion_vectors::Mat4::transform_point),
//! the same [`EPS_W`](prism_render_architecture::particle::motion_vectors::EPS_W)
//! visibility floor (a point with `w <= EPS_W` is at or behind the camera plane
//! and yields no vector), the same perspective divide `1 / w`, the same
//! post-divide jitter subtraction, the same
//! [`NdcConvention`](prism_render_architecture::particle::motion_vectors::NdcConvention)
//! `NDC`-to-`UV` mapping (`u = x * 0.5 + 0.5`; `v = -y * 0.5 + 0.5` for the
//! top-left `Y`-down framebuffer, `v = y * 0.5 + 0.5` for the bottom-left
//! `Y`-up one), and the same `cur_uv - prev_uv` difference. A particle marked
//! as spawned this frame (no valid history) produces no vector, exactly as the
//! reference returns [`None`].
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — multiply, add,
//! subtract, a single reciprocal divide and unsigned-integer compares — with no
//! `sin`, `cos`, `exp`, `log`, `pow` or optional device feature, so it runs
//! unmodified on Metal, Vulkan and DX12. There is no transcendental call at all
//! on this path (not even `sqrt`), matching the reference, and the only divide
//! is the perspective `1 / w`, which is reached only after the `EPS_W` guard
//! proves `w` is strictly positive.
//!
//! # Correctness model
//!
//! Each lane is a fixed, non-reorderable sequence of multiplies and adds (the
//! `4x4` transform, the divide, the jitter subtraction and the `UV` map), so
//! `CPU` and `GPU` evaluate the same closed form in the same order. They are
//! not bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test asserts a tolerance (`abs_diff <= 1e-5` or
//! `rel_diff <= 1e-5`) tight enough to catch a genuinely wrong port (a
//! transposed matrix read, a dropped jitter term, a flipped `V` axis, a missing
//! `w` guard) yet loose enough to admit legal fused multiply-add contraction
//! across the four terms of each transformed component.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard clip-space reprojection screen motion vector (the
//! velocity output `TAA` / `TSR` / `DLSS` upsamplers consume, mirrored at the
//! algorithm level) plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::motion_vectors::{
    CameraMotionState, Mat4, NdcConvention, PrevParticleState, ScreenMotionVector, Vec2,
};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` screen-motion-vector kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` golden
/// [`screen_motion_vector`](prism_render_architecture::particle::motion_vectors::screen_motion_vector)
/// exactly; see the module documentation for the algorithm.
const MOTION_VECTORS_WGSL: &str = r#"
// Screen-motion-vector twin: one thread per particle projects the current and
// previous world positions through the current and previous view-projection
// matrices, applies the perspective divide, removes the baked per-frame jitter,
// maps NDC to UV under the framebuffer convention, and writes the UV
// difference. A spawned particle (no valid history) or a point at/behind the
// camera plane writes an invalid result, matching the reference `Option::None`.
// It mirrors the CPU golden `particle::motion_vectors::screen_motion_vector`,
// uses only the portable core-WGSL subset (+ - * / and unsigned-integer
// compares, no transcendental), and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard clip-space reprojection screen motion vector; no Unreal
// Engine source or derived code.

struct Params {
    // Current (jittered) view-projection matrix, column-major.
    cur_view_proj: mat4x4<f32>,
    // Previous (jittered) view-projection matrix, column-major.
    prev_view_proj: mat4x4<f32>,
    // Current-frame NDC-space jitter baked into `cur_view_proj`.
    cur_jitter: vec2<f32>,
    // Previous-frame NDC-space jitter baked into `prev_view_proj`.
    prev_jitter: vec2<f32>,
    // Framebuffer NDC-to-UV convention: 0 = top-left Y-down, 1 = bottom-left
    // Y-up.
    convention: u32,
    // Number of valid particles in `particles`.
    count: u32,
    // Padding to a 16-byte boundary.
    pad0: u32,
    pad1: u32,
}

// One particle's input. 32-byte std430 stride: the current world position plus
// the history-valid flag, then the previous world position plus a pad word,
// matching the host `GpuParticle`.
struct Particle {
    cur_x: f32,
    cur_y: f32,
    cur_z: f32,
    prev_valid: u32,
    prev_x: f32,
    prev_y: f32,
    prev_z: f32,
    pad: u32,
}

// One particle's output. 16-byte std430 stride: the UV delta, a validity flag
// (0 = no vector, matching `Option::None`; 1 = valid), and a pad word, matching
// the host `GpuMotion`.
struct Motion {
    uv_x: f32,
    uv_y: f32,
    valid: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> results: array<Motion>;

// Minimum homogeneous `w` for which the perspective divide is well defined; a
// point with `w <= EPS_W` is at or behind the camera plane. Matches the
// reference `EPS_W`.
const EPS_W: f32 = 1.0e-6;

// Convention tags, matching the host mapping of `NdcConvention`.
const TOP_LEFT_Y_DOWN: u32 = 0u;

// Transforms a world-space point (homogeneous `(x, y, z, 1)`) into clip space,
// mirroring the reference `Mat4::transform_point`. The matrix is column-major
// (`m[col][row]`), so this reproduces the exact term order
// `m[0][k]*x + m[1][k]*y + m[2][k]*z + m[3][k]` the reference uses.
fn transform_point(m: mat4x4<f32>, p: vec3<f32>) -> vec4<f32> {
    let cx = m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0];
    let cy = m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1];
    let cz = m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2];
    let cw = m[0][3] * p.x + m[1][3] * p.y + m[2][3] * p.z + m[3][3];
    return vec4<f32>(cx, cy, cz, cw);
}

// Maps an unjittered NDC XY (components in -1..=1) to UV (components in 0..=1)
// under the convention tag, mirroring the reference `NdcConvention::ndc_to_uv`.
fn ndc_to_uv(ndc: vec2<f32>, convention: u32) -> vec2<f32> {
    let u = ndc.x * 0.5 + 0.5;
    var v = ndc.y * 0.5 + 0.5;
    if (convention == TOP_LEFT_Y_DOWN) {
        v = -ndc.y * 0.5 + 0.5;
    }
    return vec2<f32>(u, v);
}

@compute @workgroup_size(64)
fn motion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let particle = particles[idx];

    var out: Motion;
    out.uv_x = 0.0;
    out.uv_y = 0.0;
    out.valid = 0u;
    out.pad = 0u;

    // No valid history (spawned this frame): no motion vector.
    if (particle.prev_valid == 0u) {
        results[idx] = out;
        return;
    }

    let cur_world = vec3<f32>(particle.cur_x, particle.cur_y, particle.cur_z);
    let prev_world = vec3<f32>(particle.prev_x, particle.prev_y, particle.prev_z);

    let cur_clip = transform_point(params.cur_view_proj, cur_world);
    let prev_clip = transform_point(params.prev_view_proj, prev_world);

    // Both projected points must sit in front of the camera plane for the
    // perspective divide to be well defined, matching the reference guard.
    if (cur_clip.w <= EPS_W || prev_clip.w <= EPS_W) {
        results[idx] = out;
        return;
    }

    let cur_inv_w = 1.0 / cur_clip.w;
    let prev_inv_w = 1.0 / prev_clip.w;
    let cur_ndc = vec2<f32>(cur_clip.x * cur_inv_w, cur_clip.y * cur_inv_w);
    let prev_ndc = vec2<f32>(prev_clip.x * prev_inv_w, prev_clip.y * prev_inv_w);

    let cur_unjittered = cur_ndc - params.cur_jitter;
    let prev_unjittered = prev_ndc - params.prev_jitter;

    let cur_uv = ndc_to_uv(cur_unjittered, params.convention);
    let prev_uv = ndc_to_uv(prev_unjittered, params.convention);

    let uv_delta = cur_uv - prev_uv;
    out.uv_x = uv_delta.x;
    out.uv_y = uv_delta.y;
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// One particle's reprojection inputs: its current world-space position and the
/// previous-frame tracking state (previous position plus a validity flag).
///
/// Mirrors the `(cur_world, prev)` pair the reference
/// [`screen_motion_vector`](prism_render_architecture::particle::motion_vectors::screen_motion_vector)
/// consumes. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32`
/// positions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionVectorQuery {
    /// The particle's current-frame world-space position.
    pub cur_world: Vec3,
    /// The previous-frame tracking state (position and history-valid flag).
    pub prev: PrevParticleState,
}

/// Uniform parameters for one motion-vector dispatch. `repr(C)` layout matching
/// `Params` in [`MOTION_VECTORS_WGSL`]: two column-major `4x4` matrices (`16`
/// floats each), two `NDC`-space jitter pairs, then the convention tag, the
/// particle count and two pad words — `160` bytes, each field at the `std140`
/// uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Current view-projection matrix, column-major (`cols[col][row]`) flattened.
    cur_view_proj: [f32; 16],
    /// Previous view-projection matrix, column-major flattened.
    prev_view_proj: [f32; 16],
    /// Current-frame `NDC`-space jitter.
    cur_jitter: [f32; 2],
    /// Previous-frame `NDC`-space jitter.
    prev_jitter: [f32; 2],
    /// Convention tag: `0` top-left `Y`-down, `1` bottom-left `Y`-up.
    convention: u32,
    /// Number of particles in the dispatch.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One particle's inputs as uploaded. `32`-byte `std430` stride matching
/// `Particle` in the shader: the current position plus the history-valid flag,
/// then the previous position plus a pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParticle {
    /// Current world `x`.
    cur_x: f32,
    /// Current world `y`.
    cur_y: f32,
    /// Current world `z`.
    cur_z: f32,
    /// `1` when the previous-frame state is valid history, `0` otherwise.
    prev_valid: u32,
    /// Previous world `x`.
    prev_x: f32,
    /// Previous world `y`.
    prev_y: f32,
    /// Previous world `z`.
    prev_z: f32,
    /// Padding word.
    pad: u32,
}

/// One particle's result as read back. `16`-byte `std430` stride matching
/// `Motion` in the shader: the `UV` delta, a validity flag and a pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMotion {
    /// `UV` delta `x` (`current_uv.x - previous_uv.x`).
    uv_x: f32,
    /// `UV` delta `y` (`current_uv.y - previous_uv.y`).
    uv_y: f32,
    /// `1` when a motion vector was produced, `0` when the reference returns
    /// [`None`] (spawned particle or a point at/behind the camera plane).
    valid: u32,
    /// Padding word.
    pad: u32,
}

/// A compiled, reusable screen-motion-vector pipeline.
pub struct GpuMotionVectors {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMotionVectors {
    /// Compiles the screen-motion-vector kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMotionVectors {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_motion_vectors"),
            source: ShaderSource::Wgsl(MOTION_VECTORS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_motion_vectors_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_motion_vectors_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_motion_vectors_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("motion"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMotionVectors {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the screen-space motion vector for every particle in `queries`
    /// against the shared `camera`, returning one [`Option<ScreenMotionVector>`]
    /// per particle in input order.
    ///
    /// The returned result for query `q` equals
    /// [`screen_motion_vector`](prism_render_architecture::particle::motion_vectors::screen_motion_vector)`(camera, q.cur_world, q.prev)`
    /// to within the tolerance documented on this module: [`Some`] with the same
    /// `UV` delta when the reference produces a vector, [`None`] when it does
    /// (a spawned particle or a projected point at or behind the camera plane).
    /// An empty `queries` slice yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        camera: CameraMotionState,
        queries: &[MotionVectorQuery],
    ) -> Vec<Option<ScreenMotionVector>> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = Params {
            cur_view_proj: flatten(camera.cur_view_proj),
            prev_view_proj: flatten(camera.prev_view_proj),
            cur_jitter: [camera.cur_jitter.x, camera.cur_jitter.y],
            prev_jitter: [camera.prev_jitter.x, camera.prev_jitter.y],
            convention: convention_tag(camera.convention),
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
        };

        let particles: Vec<GpuParticle> = queries
            .iter()
            .map(|q| GpuParticle {
                cur_x: q.cur_world.x,
                cur_y: q.cur_world.y,
                cur_z: q.cur_world.z,
                prev_valid: u32::from(q.prev.valid),
                prev_x: q.prev.world_position.x,
                prev_y: q.prev.world_position.y,
                prev_z: q.prev.world_position.z,
                pad: 0,
            })
            .collect();

        let out_bytes = (queries.len() as u64) * (size_of::<GpuMotion>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_vectors_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let particles_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_motion_vectors_particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_vectors_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_motion_vectors_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_motion_vectors_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: particles_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_motion_vectors_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_motion_vectors_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per particle, in workgroups of 64 (the kernel's size).
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_results = bytemuck::cast_slice::<u8, GpuMotion>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
            .into_iter()
            .map(|m| {
                if m.valid == 0 {
                    None
                } else {
                    Some(ScreenMotionVector {
                        uv_delta: Vec2::new(m.uv_x, m.uv_y),
                    })
                }
            })
            .collect()
    }
}

/// Flattens a column-major [`Mat4`] (`cols[col][row]`) into the `16`-float
/// column-major array a `WGSL` `mat4x4<f32>` expects.
fn flatten(m: Mat4) -> [f32; 16] {
    let mut out = [0.0_f32; 16];
    for (col, column) in m.cols.iter().enumerate() {
        for (row, value) in column.iter().enumerate() {
            // Column-major: column `col`, row `row` lives at `col * 4 + row`.
            out[col * 4 + row] = *value;
        }
    }
    out
}

/// Maps an [`NdcConvention`] to the `u32` tag the kernel branches on.
fn convention_tag(convention: NdcConvention) -> u32 {
    match convention {
        // Keep in sync with `TOP_LEFT_Y_DOWN` in the kernel.
        NdcConvention::TopLeftYDown => 0,
        NdcConvention::BottomLeftYUp => 1,
    }
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
