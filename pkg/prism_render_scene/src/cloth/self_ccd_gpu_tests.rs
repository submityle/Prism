//! Real-machine GPU coverage of the three own-slot continuous self-collision
//! (self-CCD) kernels in `cloth_self_ccd.wesl`: `cloth_ccd_hash_build`,
//! `cloth_ccd_resolve` and `cloth_ccd_apply`.
//!
//! CPU golden: `prism_render_architecture::cloth::self_ccd::resolve_self_ccd`, the
//! swept spatial-hash resolver that snaps a tunnelling cloth-vs-cloth pair back
//! to its time-of-impact (TOI) contact instead of letting a fast layer pass
//! clean through another in a single step (design section 6.3). Until now the
//! shader was only proven to compile under `naga` (via `cloth::shader_tests`); this
//! module dispatches the full build -> resolve -> apply chain on a real Metal
//! device, reads positions AND velocities back, and asserts value-for-value
//! parity with the golden.
//!
//! ## Why these cases can match the golden within float32 rounding
//!
//! The golden resolves unique candidate pairs in `BTreeSet` order in place
//! (Gauss-Seidel), while the GPU is own-slot Jacobi: each particle writes only
//! its own half of a contact into `ccd_pos_delta` / `ccd_vel_delta` and a final
//! apply pass folds the deltas in. For a SINGLE contacting pair the two schemes
//! are identical (there is no earlier pair for a later one to observe), so the
//! only slack is a few ULPs between the host sqrt and the device sqrt, well
//! inside `PARITY_EPS`.
//!
//! Every fixture also keeps each particle inside exactly ONE spatial-hash cell
//! (a `cell_size` large relative to the frame motion and thickness). That keeps
//! the per-particle `ccd_particle_next` linked list valid -- a swept box that
//! overlapped several cells would need a per-cell entry list, out of scope for
//! this parity twin -- and matches the single-cell regime that
//! `cloth_ccd_min_shared_cell` canonicalises. Multi-cell swept boxes remain the
//! CPU golden's domain and are exercised in its own unit tests.
//!
//! Device acquisition is best-effort: a headless host with no wgpu adapter
//! returns `None` from `try_compute_device`, so the test prints a skip note and
//! passes, keeping the suite green on any machine.

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_architecture::cloth::self_ccd::{resolve_self_ccd, SelfCcdParams};
use prism_render_architecture::cloth::{ClothParticle, Vec3};

use super::abi::{GpuClothCcdParams, GpuClothHashCell};
use super::gpu_test_support::{
    compile_self_ccd_wgsl, find_entry_point, try_compute_device, PARITY_EPS,
};

/// Linked-list terminator, matching `CLOTH_CCD_SENTINEL` in `cloth_self_ccd.wesl`.
const SENTINEL: u32 = 0xffff_ffffu32;

/// Hash-table bucket count used by every fixture; a prime keeps the small cell
/// set collision-free. Matches the modulus fed into the uniform.
const TABLE_SIZE: u32 = 97;

/// Builds a movable particle (`inverse_mass` = 1) with an explicit velocity.
///
/// The stored velocity is only preserved when no contact fires; on a hit the
/// golden overwrites it with the TOI-recovered value, so the fixtures seed a
/// recognisable non-zero velocity to prove the no-op path leaves it untouched.
fn moving(position: Vec3, velocity: Vec3) -> ClothParticle {
    ClothParticle {
        position,
        velocity,
        inverse_mass: 1.0,
    }
}

/// Builds a pinned particle (`inverse_mass` = 0) that must never be written.
fn pinned(position: Vec3) -> ClothParticle {
    ClothParticle::pinned(position)
}

/// host upload layout for `ccd_positions`: xyz = position, w = inverse mass
/// (<= 0 = pinned), matching the shader's convention.
fn upload_positions(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
        .collect()
}

/// host upload layout for `ccd_prev_positions`: xyz = frame-start position, w
/// unused.
fn upload_prev(prev: &[Vec3]) -> Vec<[f32; 4]> {
    prev.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect()
}

/// host upload layout for `ccd_velocities`: xyz = velocity, w unused.
fn upload_velocities(particles: &[ClothParticle]) -> Vec<[f32; 4]> {
    particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z, 0.0])
        .collect()
}

/// The group-0 layout for the self-CCD kernels: `ccd_positions` (rw),
/// `ccd_prev_positions` (ro), `ccd_velocities` (rw), `ccd_cell_table` (rw),
/// `ccd_particle_next` (rw), `ccd_pos_delta` (rw), `ccd_vel_delta` (rw) and
/// `ccd_params` (uniform), matching @binding(0..7) in the WESL.
fn ccd_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
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
        label: Some("cloth_ccd_parity_group0"),
        entries: &[
            storage(0, false),
            storage(1, true),
            storage(2, false),
            storage(3, false),
            storage(4, false),
            storage(5, false),
            storage(6, false),
            BindGroupLayoutEntry {
                binding: 7,
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

/// The three self-CCD entry-point symbol names as they appear in the compiled
/// WESL output (WESL may prefix module-local names, so they are located by
/// substring rather than a fixed symbol).
struct CcdEntryPoints {
    build: String,
    resolve: String,
    apply: String,
}

fn ccd_entry_points(wgsl: &str) -> CcdEntryPoints {
    CcdEntryPoints {
        build: find_entry_point(wgsl, "cloth_ccd_hash_build"),
        resolve: find_entry_point(wgsl, "cloth_ccd_resolve"),
        apply: find_entry_point(wgsl, "cloth_ccd_apply"),
    }
}

/// Runs build -> resolve -> apply on a real device and reads both the final
/// positions and velocities back (xyzw each; position w = inverse mass).
#[expect(
    clippy::too_many_arguments,
    reason = "parity replay 需要几何、前一帧位置、速度与网格参数全部显式传入，聚成结构体反而分散阅读"
)]
fn run_ccd_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entries: &CcdEntryPoints,
    positions: &[[f32; 4]],
    prev_positions: &[[f32; 4]],
    velocities: &[[f32; 4]],
    params: GpuClothCcdParams,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let count = positions.len();
    let layout = ccd_bind_group_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("cloth_ccd_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("cloth_ccd_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let make_pipeline = |label: &str, entry: &str| {
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    let build_pipeline = make_pipeline("cloth_ccd_build_pipeline", &entries.build);
    let resolve_pipeline = make_pipeline("cloth_ccd_resolve_pipeline", &entries.resolve);
    let apply_pipeline = make_pipeline("cloth_ccd_apply_pipeline", &entries.apply);

    let cell_init = vec![
        GpuClothHashCell {
            head: SENTINEL,
            count: 0,
        };
        params.table_size as usize
    ];
    let next_init = vec![SENTINEL; count];
    let delta_init = vec![[0.0f32; 4]; count];

    let positions_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_positions"),
        contents: bytemuck::cast_slice(positions),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let prev_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_prev_positions"),
        contents: bytemuck::cast_slice(prev_positions),
        usage: BufferUsages::STORAGE,
    });
    let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_velocities"),
        contents: bytemuck::cast_slice(velocities),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let cell_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_cell_table"),
        contents: bytemuck::cast_slice(&cell_init),
        usage: BufferUsages::STORAGE,
    });
    let next_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_particle_next"),
        contents: bytemuck::cast_slice(&next_init),
        usage: BufferUsages::STORAGE,
    });
    let pos_delta_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_pos_delta"),
        contents: bytemuck::cast_slice(&delta_init),
        usage: BufferUsages::STORAGE,
    });
    let vel_delta_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_vel_delta"),
        contents: bytemuck::cast_slice(&delta_init),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("cloth_ccd_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("cloth_ccd_parity_bind"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: prev_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: velocities_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: cell_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: next_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: pos_delta_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: vel_delta_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let stage_bytes = size_of_val(positions) as u64;
    let pos_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_ccd_positions_stage"),
        size: stage_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let vel_stage = device.create_buffer(&BufferDescriptor {
        label: Some("cloth_ccd_velocities_stage"),
        size: stage_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloth_ccd_parity_encoder"),
    });
    let groups = (count as u32).div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_ccd_build_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&build_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_ccd_resolve_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&resolve_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloth_ccd_apply_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&apply_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &pos_stage, 0, stage_bytes);
    encoder.copy_buffer_to_buffer(&velocities_buf, 0, &vel_stage, 0, stage_bytes);
    queue.submit([encoder.finish()]);

    pos_stage.slice(..).map_async(MapMode::Read, |_| {});
    vel_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted self-CCD work");

    let pos_view = pos_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped position readback range should be available after poll");
    let positions_out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&pos_view).to_vec();
    drop(pos_view);
    pos_stage.unmap();

    let vel_view = vel_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped velocity readback range should be available after poll");
    let velocities_out: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&vel_view).to_vec();
    drop(vel_view);
    vel_stage.unmap();

    (positions_out, velocities_out)
}

/// Runs the CPU golden and the real-machine three-pass twin over the same
/// single-contact, single-cell fixture, then asserts every particle's position
/// AND velocity agree within `PARITY_EPS` and that the inverse mass is untouched.
///
/// params is sanitised once up front so the host uniform and the golden's
/// internal sanitise (`resolve_self_ccd`'s first line) see identical scalars.
fn assert_ccd_parity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    particles: &[ClothParticle],
    prev: &[Vec3],
    params: SelfCcdParams,
    dt: f32,
) {
    let params = params.sanitized();

    let mut golden = particles.to_vec();
    resolve_self_ccd(&mut golden, prev, params, dt);

    let inv_dt = if dt.abs() <= 1.0e-12 { 0.0 } else { 1.0 / dt };
    let gpu_params = GpuClothCcdParams {
        particle_count: particles.len() as u32,
        table_size: TABLE_SIZE,
        cell_size: params.cell_size,
        thickness: params.thickness,
        restitution: params.restitution,
        inv_dt,
        _pad0: 0,
        _pad1: 0,
    };

    let positions = upload_positions(particles);
    let prev_positions = upload_prev(prev);
    let velocities = upload_velocities(particles);
    let entry_points = ccd_entry_points(wgsl);
    let (pos_out, vel_out) = run_ccd_on_gpu(
        device,
        queue,
        wgsl,
        &entry_points,
        &positions,
        &prev_positions,
        &velocities,
        gpu_params,
    );

    assert_eq!(pos_out.len(), golden.len());
    assert_eq!(vel_out.len(), golden.len());
    for (i, cpu) in golden.iter().enumerate() {
        let gpu_pos = pos_out[i];
        let gpu_vel = vel_out[i];
        assert!(
            (gpu_pos[0] - cpu.position.x).abs() <= PARITY_EPS
                && (gpu_pos[1] - cpu.position.y).abs() <= PARITY_EPS
                && (gpu_pos[2] - cpu.position.z).abs() <= PARITY_EPS,
            "particle {i} position: gpu [{}, {}, {}] vs cpu [{}, {}, {}]",
            gpu_pos[0],
            gpu_pos[1],
            gpu_pos[2],
            cpu.position.x,
            cpu.position.y,
            cpu.position.z,
        );
        assert!(
            (gpu_vel[0] - cpu.velocity.x).abs() <= PARITY_EPS
                && (gpu_vel[1] - cpu.velocity.y).abs() <= PARITY_EPS
                && (gpu_vel[2] - cpu.velocity.z).abs() <= PARITY_EPS,
            "particle {i} velocity: gpu [{}, {}, {}] vs cpu [{}, {}, {}]",
            gpu_vel[0],
            gpu_vel[1],
            gpu_vel[2],
            cpu.velocity.x,
            cpu.velocity.y,
            cpu.velocity.z,
        );
        assert!(
            (gpu_pos[3] - particles[i].inverse_mass).abs() <= f32::EPSILON,
            "particle {i}: inverse mass mutated {} -> {}",
            particles[i].inverse_mass,
            gpu_pos[3],
        );
    }
}

/// Enabled sweep with a large single-cell grid and no bounce, keeping every
/// fixture in one hash cell (see the module docs for why that matters).
fn ccd_params(thickness: f32, restitution: f32) -> SelfCcdParams {
    SelfCcdParams {
        cell_size: 10.0,
        thickness,
        restitution,
        enabled: true,
    }
}

/// Two free particles swap sides along x in a single step: the discrete tier
/// would let them tunnel clean through, but the swept TOI snaps each back to the
/// contact instant. Their inbound closing velocity is fully absorbed
/// (restitution 0), so both end at rest at the TOI positions.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn head_on_tunnel_snaps_to_the_toi_contact() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("head_on_tunnel_snaps_to_the_toi_contact: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_self_ccd_wgsl();
    let prev = [Vec3::new(1.0, 5.0, 5.0), Vec3::new(9.0, 5.0, 5.0)];
    let particles = [
        moving(Vec3::new(9.0, 5.0, 5.0), Vec3::new(8.0, 0.0, 0.0)),
        moving(Vec3::new(1.0, 5.0, 5.0), Vec3::new(-8.0, 0.0, 0.0)),
    ];
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        ccd_params(0.5, 0.0),
        1.0,
    );
}

/// A restitution of 1 mirrors the inbound normal velocity instead of cancelling
/// it: the two swapping layers rebound at their approach speed. The impulse math
/// (a factor of 1 + restitution) is what the GPU must reproduce bit-for-bit.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn restitution_rebounds_the_closing_velocity() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("restitution_rebounds_the_closing_velocity: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_self_ccd_wgsl();
    let prev = [Vec3::new(1.0, 5.0, 5.0), Vec3::new(9.0, 5.0, 5.0)];
    let particles = [
        moving(Vec3::new(9.0, 5.0, 5.0), Vec3::new(8.0, 0.0, 0.0)),
        moving(Vec3::new(1.0, 5.0, 5.0), Vec3::new(-8.0, 0.0, 0.0)),
    ];
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        ccd_params(0.5, 1.0),
        1.0,
    );
}

/// A free particle sweeps into a pinned partner: the pinned particle is never
/// written (its apply pass early-outs on `inverse_mass` <= 0) and the free one
/// takes the full TOI snap and impulse, exactly as the golden's `is_pinned` guard
/// dictates.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn pinned_partner_stays_put_and_only_the_free_particle_moves() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!(
            "pinned_partner_stays_put_and_only_the_free_particle_moves: no wgpu adapter, skipping"
        );
        return;
    };
    let wgsl = compile_self_ccd_wgsl();
    let prev = [Vec3::new(1.0, 5.0, 5.0), Vec3::new(5.0, 5.0, 5.0)];
    let particles = [
        moving(Vec3::new(9.0, 5.0, 5.0), Vec3::new(8.0, 0.0, 0.0)),
        pinned(Vec3::new(5.0, 5.0, 5.0)),
    ];
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        ccd_params(0.5, 0.0),
        1.0,
    );
}

/// Two particles that never come within thickness produce no TOI: positions and
/// the seeded velocities are both left untouched. This guards the no-contact
/// path (empty deltas) and proves the velocity buffer is only rewritten on a
/// genuine hit.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "无合适 wgpu 设备的主机上，跳过提示需要进入测试日志"
)]
fn separated_pair_is_a_no_op() {
    let Some((device, queue)) = try_compute_device() else {
        eprintln!("separated_pair_is_a_no_op: no wgpu adapter, skipping");
        return;
    };
    let wgsl = compile_self_ccd_wgsl();
    // Both sit inside cell (0,0,0) but three units apart on y and moving only
    // along x, so the swept separation never reaches thickness 0.5.
    let prev = [Vec3::new(1.0, 2.0, 5.0), Vec3::new(1.0, 8.0, 5.0)];
    let particles = [
        moving(Vec3::new(9.0, 2.0, 5.0), Vec3::new(8.0, 0.0, 0.0)),
        moving(Vec3::new(9.0, 8.0, 5.0), Vec3::new(8.0, 0.0, 0.0)),
    ];
    assert_ccd_parity(
        &device,
        &queue,
        &wgsl,
        &particles,
        &prev,
        ccd_params(0.5, 0.0),
        1.0,
    );
}
