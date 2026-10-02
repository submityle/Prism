//! Real-machine GPU coverage of the artist-painted per-vertex constraint kernels
//! in `cloth_painted.wesl`: the `painted_anim_drive`, `painted_clamp_max_distance`,
//! `painted_backstop` and `painted_blend_to_skin` entries.
//!
//! CPU golden: `prism_render_architecture::cloth::painted` -- the four stateless
//! per-vertex projections that steer a garment toward its skinned animation pose
//! (`drive_toward_anim`, `clamp_max_distance`, `apply_painted_backstop` and
//! `blend_to_skin`). Until now those kernels were only proven to compile under
//! `naga`; this module dispatches each on a real Metal device, reads the mutated
//! positions (and, for the anim-drive pass, velocities) back, and asserts
//! value-for-value parity with the golden.
//!
//! ## Why these cases match the golden within float32 rounding
//!
//! Every pass reads only vertex `i`, its skinned anchor `i` and its painted
//! weights `i`, and writes only vertex `i`, so there is no cross-vertex write
//! hazard: the golden iterates in order purely for determinism, and since no
//! vertex reads the updated position of a neighbour the per-vertex dispatch
//! reproduces the golden bit-for-bit regardless of order. The kernels re-clamp
//! the raw authored weights exactly as `PaintedConstraint::clamped` does and
//! reuse the same `1.0 / sqrt(len_sq)` normalize, so the only slack is a few ULPs
//! of fused-multiply-add contraction, well inside `PARITY_EPS`.
//!
//! ## The whole-pass anim-drive gate lives host-side
//!
//! `drive_toward_anim` returns before its loop when the pass is disabled, the
//! master gain is non-positive, or `dt` is non-positive. That whole-pass gate is
//! reproduced on the host: when gated we do not dispatch, matching the golden
//! no-op without feeding a `1.0 / dt` division a zero `dt`. When enabled we upload
//! the sanitized gain and the precomputed `inv_dt` so the on-device follow
//! fraction and velocity update share the exact golden scalars.
//!
//! Device acquisition is best-effort: a headless host with no wgpu adapter returns
//! `None` from `try_compute_device`, so each on-device test prints a skip note and
//! passes. A device-free compile test still guards that the shader parses.

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::asset::PaintedConstraint;
use prism_render_architecture::cloth::painted::{
    apply_painted_backstop, blend_to_skin, clamp_max_distance, drive_toward_anim, AnimDriveParams,
    SkinnedAnchor,
};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::GpuClothPaintedParams;
use super::gpu_test_support::{
    compile_painted_wgsl, find_entry_point, storage_from_slice, try_compute_device, PARITY_EPS,
};

/// A free particle (unit inverse mass) at `position`.
fn free(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 1.0)
}

/// A pinned particle (zero inverse mass) at `position`.
fn pinned(position: Vec3) -> ClothParticle {
    ClothParticle::new(position, 0.0)
}

/// Host upload layout for the position buffer: `xyz` = world position, `w` =
/// inverse mass, matching the shader (which reads `w` only to skip pinned).
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// Host upload layout for the velocity buffer: `xyz` = velocity, `w` unused.
fn upload_velocities(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z, 0.0])
        .collect()
}

/// Packs anchor positions into `vec4` rows (`xyz` used, `w` padded).
fn upload_anchor_positions(anchors: &[SkinnedAnchor]) -> Vec<[f32; 4]> {
    anchors
        .iter()
        .map(|a| [a.position.x, a.position.y, a.position.z, 0.0])
        .collect()
}

/// Packs anchor normals into `vec4` rows (`xyz` used, `w` padded).
fn upload_anchor_normals(anchors: &[SkinnedAnchor]) -> Vec<[f32; 4]> {
    anchors
        .iter()
        .map(|a| [a.normal.x, a.normal.y, a.normal.z, 0.0])
        .collect()
}

/// Packs the raw authored painted weights (`x` = `max_distance`, `y` =
/// `backstop`, `z` = `blend_weight`, `w` = `anim_drive`) so the kernel re-clamps
/// them exactly
/// like `PaintedConstraint::clamped`.
fn upload_weights(painted: &[PaintedConstraint]) -> Vec<[f32; 4]> {
    painted
        .iter()
        .map(|w| [w.max_distance, w.backstop, w.blend_weight, w.anim_drive])
        .collect()
}

/// The group-0 layout shared by all four painted passes: `painted_positions`
/// (rw), `painted_velocities` (rw), `painted_anchor_positions` (ro),
/// `painted_anchor_normals` (ro), `painted_weights` (ro) and `painted_params`
/// (uniform), matching `@binding(0..5)` in the WESL.
fn painted_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
    let storage = |binding: u32, read_only: bool| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("cloth_painted_parity_group0"),
        entries: &[
            storage(0, false),
            storage(1, false),
            storage(2, true),
            storage(3, true),
            storage(4, true),
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Runs one painted pass on a real device and reads the mutated position and
/// velocity buffers back (`xyz` of each `vec4` is the value under test).
#[expect(
    clippy::too_many_arguments,
    reason = "并行 parity harness 需显式传入每个绑定缓冲，避免引入一次性聚合结构体"
)]
fn run_painted_pass_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    positions: &[[f32; 4]],
    velocities: &[[f32; 4]],
    anchor_positions: &[[f32; 4]],
    anchor_normals: &[[f32; 4]],
    weights: &[[f32; 4]],
    params: GpuClothPaintedParams,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let layout = painted_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_painted_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_painted_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("cloth_painted_pass_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_painted_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_painted_velocities"),
        contents: bytemuck::cast_slice(velocities),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let anchor_positions_buf = storage_from_slice(
        device,
        "cloth_painted_anchor_positions",
        anchor_positions,
        [0.0f32; 4],
    );
    let anchor_normals_buf = storage_from_slice(
        device,
        "cloth_painted_anchor_normals",
        anchor_normals,
        [0.0f32; 4],
    );
    let weights_buf = storage_from_slice(device, "cloth_painted_weights", weights, [0.0f32; 4]);
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_painted_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_painted_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: velocities_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: anchor_positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: anchor_normals_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: weights_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let positions_bytes = size_of_val(positions) as u64;
    let velocities_bytes = size_of_val(velocities) as u64;
    let positions_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_painted_positions_stage"),
        size: positions_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let velocities_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_painted_velocities_stage"),
        size: velocities_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_painted_parity_encoder"),
    });
    let groups = params.vertex_count.div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_painted_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &positions_stage, 0, positions_bytes);
    encoder.copy_buffer_to_buffer(&velocities_buf, 0, &velocities_stage, 0, velocities_bytes);
    queue.submit([encoder.finish()]);

    positions_stage.slice(..).map_async(MapMode::Read, |_| {});
    velocities_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted painted work");

    let positions_view = positions_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped position readback range should be available after poll");
    let positions_out: Vec<[f32; 4]> =
        bytemuck::cast_slice::<u8, [f32; 4]>(&positions_view).to_vec();
    drop(positions_view);
    positions_stage.unmap();

    let velocities_view = velocities_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped velocity readback range should be available after poll");
    let velocities_out: Vec<[f32; 4]> =
        bytemuck::cast_slice::<u8, [f32; 4]>(&velocities_view).to_vec();
    drop(velocities_view);
    velocities_stage.unmap();

    (positions_out, velocities_out)
}

/// Which painted pass is under test, carrying the extra scalars for the
/// anim-drive pass.
#[derive(Clone, Copy)]
enum PaintedPass {
    AnimDrive { params: AnimDriveParams, dt: f32 },
    ClampMaxDistance,
    Backstop,
    BlendToSkin,
}

impl PaintedPass {
    /// The WESL entry-point substring for this pass.
    fn entry_needle(self) -> &'static str {
        match self {
            PaintedPass::AnimDrive { .. } => "painted_anim_drive",
            PaintedPass::ClampMaxDistance => "painted_clamp_max_distance",
            PaintedPass::Backstop => "painted_backstop",
            PaintedPass::BlendToSkin => "painted_blend_to_skin",
        }
    }

    /// Runs the CPU golden for this pass in place over `particles`.
    fn run_golden(
        self,
        particles: &mut [ClothParticle],
        anchors: &[SkinnedAnchor],
        painted: &[PaintedConstraint],
    ) {
        match self {
            PaintedPass::AnimDrive { params, dt } => {
                drive_toward_anim(particles, anchors, painted, params, dt);
            }
            PaintedPass::ClampMaxDistance => clamp_max_distance(particles, anchors, painted),
            PaintedPass::Backstop => apply_painted_backstop(particles, anchors, painted),
            PaintedPass::BlendToSkin => blend_to_skin(particles, anchors, painted),
        }
    }

    /// Whether the whole-pass gate of `drive_toward_anim` fires (a host-side
    /// no-op that skips the dispatch entirely).
    fn anim_drive_gated(self) -> bool {
        match self {
            PaintedPass::AnimDrive { params, dt } => {
                let params = params.sanitized();
                !params.enabled || params.gain <= 0.0 || dt <= 0.0
            }
            _ => false,
        }
    }
}

/// Runs the CPU golden and the real-machine twin over the same fixture, then
/// asserts every vertex position (and, for the anim-drive pass, velocity) agrees
/// within `PARITY_EPS`.
fn assert_painted_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    particles: &[ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
    pass: PaintedPass,
) {
    let mut golden = particles.to_vec();
    pass.run_golden(&mut golden, anchors, painted);

    // Effective vertex count mirrors the golden `zip`: the shortest of the three
    // per-vertex arrays bounds both the dispatch and the guard.
    let vertex_count = particles.len().min(anchors.len()).min(painted.len()) as u32;

    let positions = upload_positions(particles);
    let velocities = upload_velocities(particles);
    let anchor_positions = upload_anchor_positions(anchors);
    let anchor_normals = upload_anchor_normals(anchors);
    let weights = upload_weights(painted);

    let (positions_out, velocities_out) = if pass.anim_drive_gated() {
        // Whole-pass gate: the golden returned before its loop, so the device
        // twin must not dispatch either -- the buffers stay exactly as uploaded.
        (positions.clone(), velocities.clone())
    } else {
        let (anim_gain, inv_dt) = match pass {
            PaintedPass::AnimDrive { params, dt } => (params.sanitized().gain, 1.0 / dt),
            _ => (0.0, 0.0),
        };
        let params = GpuClothPaintedParams {
            anim_gain,
            inv_dt,
            vertex_count,
            pad0: 0,
        };
        let entry = find_entry_point(wgsl, pass.entry_needle());
        run_painted_pass_on_gpu(
            device,
            queue,
            wgsl,
            &entry,
            &positions,
            &velocities,
            &anchor_positions,
            &anchor_normals,
            &weights,
            params,
        )
    };

    assert_eq!(positions_out.len(), golden.len());
    for (i, cpu) in golden.iter().enumerate() {
        let gpu = positions_out[i];
        assert!(
            (gpu[0] - cpu.position.x).abs() <= PARITY_EPS
                && (gpu[1] - cpu.position.y).abs() <= PARITY_EPS
                && (gpu[2] - cpu.position.z).abs() <= PARITY_EPS,
            "vertex {i} position: gpu {gpu:?} vs cpu {:?}",
            cpu.position,
        );
    }

    if matches!(pass, PaintedPass::AnimDrive { .. }) {
        for (i, cpu) in golden.iter().enumerate() {
            let gpu = velocities_out[i];
            assert!(
                (gpu[0] - cpu.velocity.x).abs() <= PARITY_EPS
                    && (gpu[1] - cpu.velocity.y).abs() <= PARITY_EPS
                    && (gpu[2] - cpu.velocity.z).abs() <= PARITY_EPS,
                "vertex {i} velocity: gpu {gpu:?} vs cpu {:?}",
                cpu.velocity,
            );
        }
    }
}

/// The shader must compile and expose all four painted entry points; this guard
/// stays green on headless hosts with no GPU.
#[test]
fn painted_wesl_compiles() {
    let wgsl = compile_painted_wgsl();
    let _ = find_entry_point(&wgsl, "painted_anim_drive");
    let _ = find_entry_point(&wgsl, "painted_clamp_max_distance");
    let _ = find_entry_point(&wgsl, "painted_backstop");
    let _ = find_entry_point(&wgsl, "painted_blend_to_skin");
}

/// The anim-drive pass pulls each free vertex a painted fraction toward its
/// anchor and advances velocity by the imposed displacement over `dt`; a pinned
/// vertex and a zero-anim-drive vertex are left untouched, exactly like the
/// golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn anim_drive_pulls_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("anim_drive_pulls_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_painted_wgsl();
    let particles = [
        free(Vec3::new(2.0, 0.0, 0.0)),
        free(Vec3::new(0.0, 4.0, 0.0)),
        pinned(Vec3::new(-3.0, 0.0, 0.0)),
    ];
    let anchors = [
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
    ];
    let painted = [
        PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 1.0),
        PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 0.0),
        PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 1.0),
    ];
    assert_painted_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &anchors,
        &painted,
        PaintedPass::AnimDrive {
            params: AnimDriveParams {
                gain: 0.5,
                enabled: true,
            },
            dt: 0.5,
        },
    );
}

/// A disabled anim-drive pass is a whole-pass no-op; the host skips the dispatch
/// and the buffers must match the golden (also a no-op) untouched.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn anim_drive_gate_skips_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("anim_drive_gate_skips_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_painted_wgsl();
    let particles = [free(Vec3::new(2.0, 0.0, 0.0))];
    let anchors = [SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))];
    let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 1.0)];
    assert_painted_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &anchors,
        &painted,
        PaintedPass::AnimDrive {
            params: AnimDriveParams {
                gain: 0.75,
                enabled: false,
            },
            dt: 0.5,
        },
    );
}

/// The clamp pass projects an over-limit vertex back onto its max-distance
/// sphere, leaves an in-range vertex alone, and skips both an uncapped (`+inf`)
/// and a `NaN` (also uncapped after `clamped`) vertex -- matching the golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn clamp_caps_drift_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("clamp_caps_drift_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_painted_wgsl();
    let particles = [
        free(Vec3::new(3.0, 0.0, 0.0)),
        free(Vec3::new(0.5, 0.0, 0.0)),
        free(Vec3::new(9.0, 0.0, 0.0)),
        free(Vec3::new(9.0, 0.0, 0.0)),
    ];
    let anchors = [
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
    ];
    let painted = [
        PaintedConstraint::new(1.0, 0.0, 1.0, 0.0),
        PaintedConstraint::new(1.0, 0.0, 1.0, 0.0),
        PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 0.0),
        PaintedConstraint::new(f32::NAN, 0.0, 1.0, 0.0),
    ];
    assert_painted_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &anchors,
        &painted,
        PaintedPass::ClampMaxDistance,
    );
}

/// The backstop pass pushes a vertex inside the cushion sphere out onto its
/// surface, handles the degenerate centre case along the anchor normal, and skips
/// a vertex with a zero anchor normal -- all matching the golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn backstop_pushes_out_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("backstop_pushes_out_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_painted_wgsl();
    let particles = [
        free(Vec3::new(0.0, -0.5, 0.0)),
        free(Vec3::new(0.0, -1.0, 0.0)),
        free(Vec3::new(0.0, -0.5, 0.0)),
    ];
    let anchors = [
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SkinnedAnchor::new(Vec3::ZERO, Vec3::ZERO),
    ];
    let painted = [
        PaintedConstraint::new(f32::INFINITY, 1.0, 1.0, 0.0),
        PaintedConstraint::new(f32::INFINITY, 1.0, 1.0, 0.0),
        PaintedConstraint::new(f32::INFINITY, 1.0, 1.0, 0.0),
    ];
    assert_painted_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &anchors,
        &painted,
        PaintedPass::Backstop,
    );
}

/// The blend pass mixes each simulated position toward its skinned anchor by the
/// painted `blend_weight`; the GPU must land on the exact golden mix.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn blend_mixes_to_skin_like_golden() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("blend_mixes_to_skin_like_golden: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_painted_wgsl();
    let particles = [
        free(Vec3::new(4.0, 0.0, 0.0)),
        free(Vec3::new(0.0, 8.0, 0.0)),
    ];
    let anchors = [
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SkinnedAnchor::new(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
    ];
    let painted = [
        PaintedConstraint::new(f32::INFINITY, 0.0, 0.25, 0.0),
        PaintedConstraint::new(f32::INFINITY, 0.0, 0.0, 0.0),
    ];
    assert_painted_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &anchors,
        &painted,
        PaintedPass::BlendToSkin,
    );
}

/// A pinned vertex (zero inverse mass) must be skipped by every pass; a free
/// vertex in the same batch still gets projected, proving the per-vertex pin
/// guard is per-vertex on the GPU exactly as in the golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn pinned_vertices_untouched() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("pinned_vertices_untouched: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_painted_wgsl();
    let particles = [
        pinned(Vec3::new(5.0, 0.0, 0.0)),
        free(Vec3::new(4.0, 0.0, 0.0)),
    ];
    let anchors = [
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
    ];
    let painted = [
        PaintedConstraint::new(1.0, 0.0, 0.5, 1.0),
        PaintedConstraint::new(1.0, 0.0, 0.5, 1.0),
    ];
    for pass in [
        PaintedPass::AnimDrive {
            params: AnimDriveParams {
                gain: 1.0,
                enabled: true,
            },
            dt: 0.5,
        },
        PaintedPass::ClampMaxDistance,
        PaintedPass::Backstop,
        PaintedPass::BlendToSkin,
    ] {
        assert_painted_parity(&device, &queue, &wgsl, &particles, &anchors, &painted, pass);
    }
}
