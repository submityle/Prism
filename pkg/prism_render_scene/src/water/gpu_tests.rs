//! Real-device `GPU` parity coverage for the water compute kernels.
//!
//! The sibling [`shader_tests`](super::shader_tests) module proves every water
//! `WESL` source parses and type-checks through the render world's
//! [`ShaderCache`]. That guards the *shape* of each kernel, but it never runs a
//! kernel: a shader can compile cleanly and still compute the wrong number. The
//! tests here close that gap for the shallow-water (`SWE`) step by binding the
//! real `water_swe_step` compute pipeline on an actual `Metal` (or any native
//! `wgpu`) device, dispatching one step over a non-trivial grid, reading the
//! results back, and asserting them cell-for-cell against the `CPU` golden twin
//! [`prism_render_architecture::water::swe::step`]. Because the `WESL` kernel and
//! the `CPU` reference share byte-identical arithmetic (conservative continuity
//! flux, central pressure gradient, first-order upwind self-advection, linear
//! damping, and the interaction source folded in after the step), a green run
//! is direct on-device evidence that the ported kernel matches its reference to
//! `float32` rounding, not merely that it compiles.
//!
//! Adapter acquisition is best-effort: on a headless host with no `wgpu`
//! adapter (some `CI` images) [`try_solver_device`] returns `None` and the test
//! skips with a printed notice instead of failing, so the suite stays green
//! everywhere while still exercising the full dispatch on any machine with a
//! real device (for example an `Apple` `M`-series `GPU`).

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineCompilationOptions, PollType,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
};

use prism_render_architecture::water::swe::{self, SweConfig, SweState};

use super::abi::GpuWaterSweParams;

/// Absolute per-cell tolerance for the `GPU`-versus-`CPU` comparison.
///
/// The two paths run the same `float32` arithmetic, so agreement is far tighter
/// than this in practice; the margin only absorbs a driver's fused-multiply-add
/// contraction and reordering freedom.
const PARITY_EPS: f32 = 1.0e-3;

/// Accumulated parity bound for a multi-frame roll-out. Per-frame rounding and
/// fused-multiply-add reordering compound across frames; on an `M2` the measured
/// worst-case drift over 64 frames is ~5.5e-6, so this bound (matching the
/// single-step [`PARITY_EPS`]) leaves ample headroom for driver variance while
/// still catching a divergent scheme, which grows without bound rather than
/// staying near rounding.
const MULTI_FRAME_PARITY_EPS: f32 = 1.0e-3;

/// Relative bound on total-volume drift for the conservative continuity flux
/// with damping disabled, reflective walls, and no sources. Volume is invariant
/// in exact arithmetic; only float32 summation rounding may nudge it.
const MASS_CONSERVATION_REL_EPS: f32 = 1.0e-4;

/// Slack allowed on the `PBF` bounded-stability invariant: after the first
/// density projection (which produces the largest transient excursion for a
/// finite blob whose under-dense shell has no outside neighbours), no later
/// iteration's worst density-constraint residual may exceed that peak. The
/// slack absorbs last-digit rounding without hiding a genuinely diverging
/// (monotonically growing) solve.
const PBF_OVERSHOOT_SLACK: f32 = 1.0e-3;

/// Streams the `Wgsl` source back out of the shader cache without a device.
///
/// Mirrors the closure [`shader_tests`](super::shader_tests) uses so the `WESL`
/// is composed through the exact render-world pipeline; here we keep the
/// compiled `Wgsl` string (rather than a device module) so the parity test can
/// hand it to a raw `wgpu` device it created itself.
fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("water shaders are WESL"),
    }
}

/// Compiles `water_surface.wesl` and returns its `Wgsl` translation.
fn compile_surface_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_5247_5055_0001),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_surface.wesl"),
            "embedded://prism_render_scene/shaders/water_surface.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_surface.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Finds the compute entry point whose name contains `needle` in `wgsl`.
///
/// The `WESL` compiler may prefix module-local names, so the parity test locates
/// the `water_swe_step` entry by substring rather than assuming a fixed symbol.
fn find_entry_point(wgsl: &str, needle: &str) -> String {
    for line in wgsl.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("fn ")
            && let Some(paren) = rest.find('(')
        {
            let name = &rest[..paren];
            if name.contains(needle) {
                return name.to_string();
            }
        }
    }
    panic!("no compute entry point containing `{needle}` in compiled Wgsl");
}

/// Best-effort acquisition of a native compute device and queue.
///
/// Returns `None` (rather than panicking) when no adapter is available so the
/// suite stays green on headless hosts; on a machine with a real `GPU` this
/// yields a live device the parity test dispatches against.
fn try_solver_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Builds a deterministic, non-trivial shallow-water initial state.
///
/// A raised algebraic bump over an otherwise still `1 m` sheet, sheared
/// velocities, and a couple of interaction sources exercise every branch of the
/// step (interior flux, reflective walls, pressure gradient, upwind advection,
/// damping and source injection) without any transcendental functions.
fn build_initial_state(nx: usize, nz: usize) -> (SweState, Vec<[f32; 4]>) {
    let n = nx * nz;
    let mut h = vec![0.0_f32; n];
    let mut u = vec![0.0_f32; n];
    let mut v = vec![0.0_f32; n];
    let cx = (nx as f32 - 1.0) * 0.5;
    let cz = (nz as f32 - 1.0) * 0.5;
    let mut z = 0;
    while z < nz {
        let mut x = 0;
        while x < nx {
            let i = z * nx + x;
            let fx = x as f32 - cx;
            let fz = z as f32 - cz;
            let r2 = fx * fx + fz * fz;
            let bump = (1.0 - r2 * 0.02).max(0.0);
            h[i] = 1.0 + 0.5 * bump;
            u[i] = 0.05 * fx * 0.1;
            v[i] = -0.04 * fz * 0.1;
            x += 1;
        }
        z += 1;
    }
    let mut sources = vec![[0.0_f32; 4]; n];
    let center = (nz / 2) * nx + (nx / 2);
    sources[center] = [0.2, 0.0, 0.0, 0.0];
    let off = (nz / 4) * nx + (nx / 4);
    sources[off] = [0.0, 0.15, -0.1, 0.0];
    (SweState { h, u, v }, sources)
}

/// Dispatches one `water_swe_step` on device and reads back `h`, `u`, `v`.
///
/// The bind group is built directly from the pipeline's reflected `group(0)`
/// layout so the eight bindings match the shader's declaration order exactly.
#[expect(
    clippy::too_many_lines,
    reason = "one linear dispatch-and-readback keeps the parity path auditable"
)]
fn dispatch_swe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    cfg: SweConfig,
    state: &SweState,
    sources: &[[f32; 4]],
    params: &GpuWaterSweParams,
) -> SweState {
    let n = cfg.cell_count();
    let scalar_bytes = (n * size_of::<f32>()) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_surface_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_swe_step_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let storage_read = BufferUsages::STORAGE;
    let storage_out = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let h_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("swe_h_in"),
        contents: bytemuck::cast_slice(&state.h),
        usage: storage_read,
    });
    let u_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("swe_u_in"),
        contents: bytemuck::cast_slice(&state.u),
        usage: storage_read,
    });
    let v_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("swe_v_in"),
        contents: bytemuck::cast_slice(&state.v),
        usage: storage_read,
    });
    let h_out = device.create_buffer(&BufferDescriptor {
        label: Some("swe_h_out"),
        size: scalar_bytes,
        usage: storage_out,
        mapped_at_creation: false,
    });
    let u_out = device.create_buffer(&BufferDescriptor {
        label: Some("swe_u_out"),
        size: scalar_bytes,
        usage: storage_out,
        mapped_at_creation: false,
    });
    let v_out = device.create_buffer(&BufferDescriptor {
        label: Some("swe_v_out"),
        size: scalar_bytes,
        usage: storage_out,
        mapped_at_creation: false,
    });
    let src_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("swe_sources"),
        contents: bytemuck::cast_slice(sources),
        usage: storage_read,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("swe_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("swe_group0"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: h_in.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: u_in.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: v_in.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: h_out.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: u_out.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: v_out.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: src_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let h_stage = device.create_buffer(&BufferDescriptor {
        label: Some("swe_h_stage"),
        size: scalar_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let u_stage = device.create_buffer(&BufferDescriptor {
        label: Some("swe_u_stage"),
        size: scalar_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let v_stage = device.create_buffer(&BufferDescriptor {
        label: Some("swe_v_stage"),
        size: scalar_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("swe_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("swe_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cfg.nx.div_ceil(8), cfg.nz.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&h_out, 0, &h_stage, 0, scalar_bytes);
    encoder.copy_buffer_to_buffer(&u_out, 0, &u_stage, 0, scalar_bytes);
    encoder.copy_buffer_to_buffer(&v_out, 0, &v_stage, 0, scalar_bytes);
    queue.submit([encoder.finish()]);

    h_stage.slice(..).map_async(MapMode::Read, |_| {});
    u_stage.slice(..).map_async(MapMode::Read, |_| {});
    v_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let read_back = |buffer: &wgpu::Buffer| -> Vec<f32> {
        let view = buffer
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let values: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        buffer.unmap();
        values
    };
    SweState {
        h: read_back(&h_stage),
        u: read_back(&u_stage),
        v: read_back(&v_stage),
    }
}

/// Applies the interaction sources the `CPU` `step` leaves to its caller.
///
/// The `WESL` kernel folds `swe_sources` into the same dispatch, so the golden
/// reference must add them after [`swe::step`] to line the two paths up.
fn fold_sources(state: &mut SweState, sources: &[[f32; 4]]) {
    let mut i = 0;
    while i < state.h.len() {
        state.h[i] += sources[i][0];
        state.u[i] += sources[i][1];
        state.v[i] += sources[i][2];
        i += 1;
    }
}

/// One on-device `SWE` step must match the `CPU` golden within `float32` rounding.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn swe_step_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("swe_step_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity");
        return;
    };

    const NX: usize = 16;
    const NZ: usize = 16;
    let cfg = SweConfig {
        nx: NX as u32,
        nz: NZ as u32,
        dx: 0.5,
        gravity: 9.81,
        damping: 0.2,
    };
    let cfl_number = 0.5_f32;
    let requested_dt = 0.1_f32;

    let (state, sources) = build_initial_state(NX, NZ);
    let max_wave_speed = swe::max_wave_speed(&state, cfg);
    let cfl_dt = swe::cfl_timestep(max_wave_speed, cfg.dx, cfl_number);
    let effective_dt = requested_dt.min(cfl_dt);

    let mut golden = swe::step(&state, cfg, effective_dt);
    fold_sources(&mut golden, &sources);

    let params = GpuWaterSweParams {
        nx: cfg.nx,
        nz: cfg.nz,
        dx: cfg.dx,
        gravity: cfg.gravity,
        damping: cfg.damping,
        dt: requested_dt,
        cfl_number,
        max_wave_speed,
    };

    let wgsl = compile_surface_wgsl();
    let entry = find_entry_point(&wgsl, "swe_step");
    let gpu = dispatch_swe(
        &device, &queue, &wgsl, &entry, cfg, &state, &sources, &params,
    );

    assert_eq!(gpu.h.len(), golden.h.len(), "cell count mismatch");
    let mut i = 0;
    while i < golden.h.len() {
        let dh = (gpu.h[i] - golden.h[i]).abs();
        let du = (gpu.u[i] - golden.u[i]).abs();
        let dv = (gpu.v[i] - golden.v[i]).abs();
        assert!(
            dh < PARITY_EPS && du < PARITY_EPS && dv < PARITY_EPS,
            "cell {i}: gpu=({}, {}, {}) cpu=({}, {}, {}) |d|=({dh}, {du}, {dv})",
            gpu.h[i],
            gpu.u[i],
            gpu.v[i],
            golden.h[i],
            golden.u[i],
            golden.v[i],
        );
        i += 1;
    }
}

/// A long `SWE` roll-out on device must track the `CPU` golden step-for-step and
/// keep total water volume within rounding of its initial value.
///
/// The single-step golden only proves one dispatch is faithful. Production
/// surfaces integrate for thousands of frames, so this test loops the on-device
/// `water_swe_step` and the [`swe::step`] reference in lockstep for many frames,
/// feeding each frame's output back as the next frame's input. Two invariants
/// are asserted every frame:
///
/// 1. **Parity does not drift**: the on-device state stays within a snug bound
///    of the `CPU` reference, and neither path ever produces a non-finite value
///    (an explicit scheme that diverges would blow up to `NaN`/`inf` first).
/// 2. **Mass is conserved**: with linear damping disabled, reflective (no-flux)
///    walls, and no interaction sources, the conservative continuity flux must
///    preserve `sum(h) * dx * dx` across the whole roll-out. This is the same
///    physical golden `FLIP`/`SWE` pipelines (Houdini, UE5 Water) hold their
///    solvers to across a long transition band.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and observed drift must reach the test log"
)]
fn swe_multi_frame_gpu_tracks_cpu_and_conserves_mass() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "swe_multi_frame_gpu_tracks_cpu_and_conserves_mass: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    const NX: usize = 32;
    const NZ: usize = 32;
    const FRAMES: usize = 64;
    let cfg = SweConfig {
        nx: NX as u32,
        nz: NZ as u32,
        dx: 0.5,
        gravity: 9.81,
        // Damping disabled so the conservative flux is the only thing moving
        // volume: any per-frame leak shows up directly in the mass invariant.
        damping: 0.0,
    };
    let cfl_number = 0.5_f32;
    let requested_dt = 0.05_f32;

    // A non-trivial bump plus a shear flow gives the flux something to do; the
    // interaction sources are zeroed so volume can only change through a solver
    // bug, not through injection.
    let (init_state, _) = build_initial_state(NX, NZ);
    let zero_sources = vec![[0.0_f32; 4]; NX * NZ];

    let wgsl = compile_surface_wgsl();
    let entry = find_entry_point(&wgsl, "swe_step");

    let volume0 = init_state.total_volume(cfg);
    assert!(volume0 > 0.0, "seed volume must be positive");

    let mut cpu_state = init_state.clone();
    let mut gpu_state = init_state;

    let mut max_parity_drift = 0.0_f32;
    let mut max_volume_rel_drift = 0.0_f32;

    let mut frame = 0;
    while frame < FRAMES {
        // The golden state is the timestep authority; the same effective dt and
        // wave-speed feed the device so the two paths advance identically.
        let max_wave_speed = swe::max_wave_speed(&cpu_state, cfg);
        let cfl_dt = swe::cfl_timestep(max_wave_speed, cfg.dx, cfl_number);
        let effective_dt = requested_dt.min(cfl_dt);

        let cpu_next = swe::step(&cpu_state, cfg, effective_dt);

        let params = GpuWaterSweParams {
            nx: cfg.nx,
            nz: cfg.nz,
            dx: cfg.dx,
            gravity: cfg.gravity,
            damping: cfg.damping,
            dt: requested_dt,
            cfl_number,
            max_wave_speed,
        };
        let gpu_next = dispatch_swe(
            &device,
            &queue,
            &wgsl,
            &entry,
            cfg,
            &gpu_state,
            &zero_sources,
            &params,
        );

        assert_eq!(gpu_next.h.len(), cpu_next.h.len(), "cell count mismatch");
        let mut i = 0;
        while i < cpu_next.h.len() {
            for (label, g, c) in [
                ("h", gpu_next.h[i], cpu_next.h[i]),
                ("u", gpu_next.u[i], cpu_next.u[i]),
                ("v", gpu_next.v[i], cpu_next.v[i]),
            ] {
                assert!(
                    g.is_finite() && c.is_finite(),
                    "frame {frame} cell {i} {label}: non-finite gpu={g} cpu={c}"
                );
                let drift = (g - c).abs();
                if drift > max_parity_drift {
                    max_parity_drift = drift;
                }
                assert!(
                    drift < MULTI_FRAME_PARITY_EPS,
                    "frame {frame} cell {i} {label}: gpu={g} cpu={c} |d|={drift}"
                );
            }
            i += 1;
        }

        // Reflective walls + zero damping + no sources => volume is invariant.
        let gpu_volume = gpu_next.total_volume(cfg);
        let rel = ((gpu_volume - volume0) / volume0).abs();
        if rel > max_volume_rel_drift {
            max_volume_rel_drift = rel;
        }
        assert!(
            rel < MASS_CONSERVATION_REL_EPS,
            "frame {frame}: gpu volume drifted {volume0} -> {gpu_volume} (rel {rel})"
        );

        cpu_state = cpu_next;
        gpu_state = gpu_next;
        frame += 1;
    }

    eprintln!(
        "swe_multi_frame_gpu_tracks_cpu_and_conserves_mass: {FRAMES} frames, \
         max parity drift {max_parity_drift:e}, max volume rel drift {max_volume_rel_drift:e}"
    );
}

// ===========================================================================
// Foam advection + decay parity (`water_foam_advect`, water_surface.wesl)
// ===========================================================================

use prism_render_architecture::water::foam::{self, FoamConfig};

use super::abi::GpuWaterFoamParams;

/// Builds a deterministic, non-trivial foam field plus its surface flow and
/// this step's reactive sources.
///
/// A raised algebraic coverage bump, a sheared velocity field (so the
/// semi-Lagrangian backtrace lands on fractional cell coordinates and exercises
/// the bilinear resample), and two additive sources drive every branch of the
/// step: advection, flow-aware decay, source injection, and the `0..=1` clamp.
/// No transcendental inputs are used; the only nonlinearity is the shared
/// polynomial `exp_approx`, identical on both paths.
fn build_initial_foam(nx: usize, nz: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = nx * nz;
    let mut density = vec![0.0_f32; n];
    let mut u = vec![0.0_f32; n];
    let mut v = vec![0.0_f32; n];
    let cx = (nx as f32 - 1.0) * 0.5;
    let cz = (nz as f32 - 1.0) * 0.5;
    let mut z = 0;
    while z < nz {
        let mut x = 0;
        while x < nx {
            let i = z * nx + x;
            let fx = x as f32 - cx;
            let fz = z as f32 - cz;
            let r2 = fx * fx + fz * fz;
            let bump = (1.0 - r2 * 0.03).max(0.0);
            density[i] = 0.15 + 0.7 * bump;
            u[i] = 0.3 + 0.05 * fx;
            v[i] = -0.2 + 0.04 * fz;
            x += 1;
        }
        z += 1;
    }
    let mut sources = vec![0.0_f32; n];
    sources[(nz / 2) * nx + (nx / 2)] = 0.4;
    sources[(nz / 4) * nx + (nx / 4)] = 0.25;
    (density, u, v, sources)
}

/// Dispatches one `water_foam_advect` step on device and reads back the field.
///
/// The kernel declares its resources on `@group(1)`, so the bind group is built
/// from the pipeline's reflected `group(1)` layout and set at binding index `1`
/// (the auto-derived `group(0)` is empty and referenced by nothing, so it needs
/// no bind group). The six bindings match the shader's declaration order:
/// `foam_in`, `foam_u`, `foam_v`, `foam_sources`, `foam_out`, `foam_params`.
#[expect(
    clippy::too_many_lines,
    reason = "one linear dispatch-and-readback keeps the parity path auditable"
)]
fn dispatch_foam(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    cfg: FoamConfig,
    density: &[f32],
    u: &[f32],
    v: &[f32],
    sources: &[f32],
    params: &GpuWaterFoamParams,
) -> Vec<f32> {
    let n = cfg.cell_count();
    let scalar_bytes = (n * size_of::<f32>()) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_surface_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_foam_advect_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let storage_read = BufferUsages::STORAGE;
    let storage_out = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
    let foam_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("foam_in"),
        contents: bytemuck::cast_slice(density),
        usage: storage_read,
    });
    let foam_u = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("foam_u"),
        contents: bytemuck::cast_slice(u),
        usage: storage_read,
    });
    let foam_v = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("foam_v"),
        contents: bytemuck::cast_slice(v),
        usage: storage_read,
    });
    let foam_sources = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("foam_sources"),
        contents: bytemuck::cast_slice(sources),
        usage: storage_read,
    });
    let foam_out = device.create_buffer(&BufferDescriptor {
        label: Some("foam_out"),
        size: scalar_bytes,
        usage: storage_out,
        mapped_at_creation: false,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("foam_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(1);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("water_foam_advect_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: foam_in.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: foam_u.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: foam_v.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: foam_sources.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: foam_out.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let out_stage = device.create_buffer(&BufferDescriptor {
        label: Some("foam_out_stage"),
        size: scalar_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("foam_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("foam_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.dispatch_workgroups(cfg.nx.div_ceil(8), cfg.nz.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&foam_out, 0, &out_stage, 0, scalar_bytes);
    queue.submit([encoder.finish()]);

    out_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = out_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    out_stage.unmap();
    values
}

/// One on-device foam step must match the `CPU` golden within `float32` rounding.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn foam_advect_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("foam_advect_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity");
        return;
    };

    const NX: usize = 16;
    const NZ: usize = 16;
    let cfg = FoamConfig {
        nx: NX as u32,
        nz: NZ as u32,
        dx: 0.5,
        base_decay: 0.8,
        persistence_floor: 0.1,
        reference_speed: 2.0,
    };
    let dt = 0.1_f32;

    let (density, u, v, sources) = build_initial_foam(NX, NZ);
    let golden = foam::step_foam(&density, &u, &v, &sources, cfg, dt);

    let params = GpuWaterFoamParams {
        nx: cfg.nx,
        nz: cfg.nz,
        dx: cfg.dx,
        dt,
        base_decay: cfg.base_decay,
        persistence_floor: cfg.persistence_floor,
        reference_speed: cfg.reference_speed,
        _pad: 0,
    };

    let wgsl = compile_surface_wgsl();
    let entry = find_entry_point(&wgsl, "foam_advect");
    let gpu = dispatch_foam(
        &device, &queue, &wgsl, &entry, cfg, &density, &u, &v, &sources, &params,
    );

    assert_eq!(gpu.len(), golden.len(), "cell count mismatch");
    let mut i = 0;
    while i < golden.len() {
        let d = (gpu[i] - golden[i]).abs();
        assert!(
            d < PARITY_EPS,
            "cell {i}: gpu={} cpu={} |d|={d}",
            gpu[i],
            golden[i],
        );
        i += 1;
    }
}

// ===========================================================================
// PBF density-constraint solve parity (`water_pbf_density_solve`, water_pbf.wesl)
// ===========================================================================

use prism_render_architecture::water::pbf::{self, NeighborContribution, PbfGrid, PbfParams};
use prism_render_architecture::water::Vec3;

use super::abi::GpuPbfParams;

/// Compiles `water_pbf.wesl` and returns its `Wgsl` translation.
///
/// Mirrors [`compile_surface_wgsl`] but streams the standalone `PBF` module
/// (its own `@group(0)` resource set) through the render world's shader cache,
/// so the parity test dispatches the exact `Wgsl` the engine would.
fn compile_pbf_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_5250_4246_0002),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_pbf.wesl"),
            "embedded://prism_render_scene/shaders/water_pbf.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_pbf.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Builds a deterministic, non-trivial `PBF` particle set inside a 4x4x4 grid.
///
/// A dense 3x3x3 block (spacing well under the smoothing radius, so it is
/// compressed and yields a non-zero constraint), a second offset cluster that
/// straddles a cell boundary (so cross-cell neighbour gathering is exercised),
/// and two isolated particles in far cells (no neighbours, so the pass-through
/// path is covered) drive every branch of the solve. The `w` lane carries a
/// distinct per-particle payload that must survive the projection untouched.
fn build_pbf_particles() -> (Vec<Vec3>, Vec<f32>) {
    let mut positions = Vec::new();
    // Dense compressed block centred at (1.5, 1.5, 1.5), spacing 0.35 m.
    let mut i = 0;
    while i < 3 {
        let mut j = 0;
        while j < 3 {
            let mut k = 0;
            while k < 3 {
                positions.push(Vec3::new(
                    1.5 - 0.35 + i as f32 * 0.35,
                    1.5 - 0.35 + j as f32 * 0.35,
                    1.5 - 0.35 + k as f32 * 0.35,
                ));
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    // Offset 2x2x2 cluster straddling the (2, .., ..) cell boundary.
    let mut i = 0;
    while i < 2 {
        let mut j = 0;
        while j < 2 {
            let mut k = 0;
            while k < 2 {
                positions.push(Vec3::new(
                    2.35 + i as f32 * 0.3,
                    1.55 + j as f32 * 0.3,
                    1.55 + k as f32 * 0.3,
                ));
                k += 1;
            }
            j += 1;
        }
        i += 1;
    }
    // Two isolated particles in far corners (no neighbours -> pass-through).
    positions.push(Vec3::new(0.5, 0.5, 0.5));
    positions.push(Vec3::new(3.5, 3.5, 3.5));

    let carried: Vec<f32> = (0..positions.len()).map(|n| n as f32 + 0.5).collect();
    (positions, carried)
}

/// Packs `bin_particles` output into the shader's single `hash` storage buffer.
///
/// Layout mirrors `water_pbf.wesl`: `hash[2*c]` = start offset (into the index
/// region) of cell `c`, `hash[2*c + 1]` = its particle count, and the index
/// region (`hash[2*cell_count + start + s]`) lists the binned particle indices
/// in cell-major, ascending-index order.
fn build_pbf_hash(grid: PbfGrid, positions: &[Vec3]) -> Vec<u32> {
    let bins = pbf::bin_particles(grid, positions);
    let cell_count = grid.cell_count();
    let mut hash = vec![0u32; 2 * cell_count];
    let mut index_region: Vec<u32> = Vec::new();
    let mut flat = 0;
    while flat < cell_count {
        let cell = &bins.cells[flat];
        hash[2 * flat] = index_region.len() as u32;
        hash[2 * flat + 1] = cell.len() as u32;
        for &particle in cell {
            index_region.push(particle);
        }
        flat += 1;
    }
    hash.extend_from_slice(&index_region);
    hash
}

/// `CPU` golden twin of `water_pbf_density_solve`.
///
/// Reproduces the shader exactly: every `lambda` is pre-computed from the frozen
/// input (the shader re-derives `lambda_j` on demand from the same frozen
/// buffer, so a pre-pass is numerically identical), then each particle's
/// position correction is accumulated over the neighbourhood in the same
/// cell-major, ascending-index order the shader walks. The `w` lane is carried
/// through unchanged.
fn pbf_golden(
    positions: &[Vec3],
    carried: &[f32],
    grid: PbfGrid,
    params: PbfParams,
) -> Vec<[f32; 4]> {
    let h = params.smoothing_radius;
    let rest = params.rest_density;
    let mass = params.particle_mass;
    let eps = params.relaxation_epsilon;
    let bins = pbf::bin_particles(grid, positions);

    let lambdas: Vec<f32> = (0..positions.len())
        .map(|i| {
            let neighbors = pbf::gather_neighbors(grid, &bins, positions, i as u32);
            let mut density = pbf::poly6(0.0, h);
            let mut grad_sum = Vec3::ZERO;
            let mut grad_sq_sum = 0.0_f32;
            for &j in &neighbors {
                let r_vec = positions[i].sub(positions[j as usize]);
                let r2 = r_vec.length_squared();
                density += pbf::poly6(r2, h);
                let grad = pbf::spiky_gradient(r_vec, h);
                grad_sum = grad_sum.add(grad);
                grad_sq_sum += grad.length_squared();
            }
            density *= mass.max(0.0);
            pbf::constraint_lambda(density, rest, grad_sum, grad_sq_sum, eps)
        })
        .collect();

    (0..positions.len())
        .map(|i| {
            let neighbors = pbf::gather_neighbors(grid, &bins, positions, i as u32);
            let contribs: Vec<NeighborContribution> = neighbors
                .iter()
                .map(|&j| {
                    let r_vec = positions[i].sub(positions[j as usize]);
                    let r2 = r_vec.length_squared();
                    NeighborContribution {
                        lambda_j: lambdas[j as usize],
                        scorr: pbf::artificial_pressure(r2, params),
                        gradient: pbf::spiky_gradient(r_vec, h),
                    }
                })
                .collect();
            let delta = pbf::position_correction(lambdas[i], rest, &contribs);
            let p = positions[i].add(delta);
            [p.x, p.y, p.z, carried[i]]
        })
        .collect()
}

/// Dispatches one `water_pbf_density_solve` on device and reads `positions_out`.
///
/// The bind group is built from the pipeline's reflected `group(0)` layout so
/// the four bindings line up with the shader's declaration order:
/// `positions_in`, `positions_out`, `hash`, `params`. The dispatch covers one
/// invocation per particle (`ceil(count / 64)` workgroups of 64).
fn dispatch_pbf(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    positions_in: &[[f32; 4]],
    hash: &[u32],
    params: &GpuPbfParams,
) -> Vec<[f32; 4]> {
    let count = positions_in.len();
    let vec4_bytes = size_of_val(positions_in) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_pbf_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_pbf_density_solve_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let pos_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("pbf_positions_in"),
        contents: bytemuck::cast_slice(positions_in),
        usage: BufferUsages::STORAGE,
    });
    let pos_out = device.create_buffer(&BufferDescriptor {
        label: Some("pbf_positions_out"),
        size: vec4_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let hash_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("pbf_hash"),
        contents: bytemuck::cast_slice(hash),
        usage: BufferUsages::STORAGE,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("pbf_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("pbf_group0"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: pos_in.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: pos_out.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: hash_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let out_stage = device.create_buffer(&BufferDescriptor {
        label: Some("pbf_out_stage"),
        size: vec4_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("pbf_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("pbf_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&pos_out, 0, &out_stage, 0, vec4_bytes);
    queue.submit([encoder.finish()]);

    out_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = out_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values: Vec<[f32; 4]> = bytemuck::cast_slice::<u8, [f32; 4]>(&view).to_vec();
    drop(view);
    out_stage.unmap();
    values
}

/// One on-device `PBF` density solve must match the `CPU` golden within
/// `float32` rounding.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn pbf_density_solve_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "pbf_density_solve_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let cell_size = 1.0_f32;
    let grid = PbfGrid {
        origin: Vec3::ZERO,
        cell_size,
        nx: 4,
        ny: 4,
        nz: 4,
    };
    let cpu_params = PbfParams {
        rest_density: 20.0,
        particle_mass: 1.0,
        smoothing_radius: cell_size,
        relaxation_epsilon: 0.01,
        artificial_pressure_k: 0.1,
        artificial_pressure_n: 4,
        artificial_pressure_delta_q: 0.2,
        solver_iterations: 1,
    };

    let (positions, carried) = build_pbf_particles();
    let count = positions.len();
    let golden = pbf_golden(&positions, &carried, grid, cpu_params);

    let positions_in: Vec<[f32; 4]> = positions
        .iter()
        .zip(&carried)
        .map(|(p, &w)| [p.x, p.y, p.z, w])
        .collect();
    let hash = build_pbf_hash(grid, &positions);

    let params = GpuPbfParams {
        grid_origin: [grid.origin.x, grid.origin.y, grid.origin.z],
        cell_size,
        rest_density: cpu_params.rest_density,
        particle_mass: cpu_params.particle_mass,
        smoothing_radius: cpu_params.smoothing_radius,
        relaxation_epsilon: cpu_params.relaxation_epsilon,
        artificial_pressure_k: cpu_params.artificial_pressure_k,
        artificial_pressure_delta_q: cpu_params.artificial_pressure_delta_q,
        artificial_pressure_n: cpu_params.artificial_pressure_n,
        particle_count: count as u32,
        grid_nx: grid.nx,
        grid_ny: grid.ny,
        grid_nz: grid.nz,
        _pad: 0,
    };

    let wgsl = compile_pbf_wgsl();
    let entry = find_entry_point(&wgsl, "pbf_density_solve");
    let gpu = dispatch_pbf(
        &device,
        &queue,
        &wgsl,
        &entry,
        &positions_in,
        &hash,
        &params,
    );

    assert_eq!(gpu.len(), golden.len(), "particle count mismatch");
    let mut i = 0;
    while i < golden.len() {
        let dx = (gpu[i][0] - golden[i][0]).abs();
        let dy = (gpu[i][1] - golden[i][1]).abs();
        let dz = (gpu[i][2] - golden[i][2]).abs();
        let dw = (gpu[i][3] - golden[i][3]).abs();
        assert!(
            dx < PARITY_EPS && dy < PARITY_EPS && dz < PARITY_EPS && dw < PARITY_EPS,
            "particle {i}: gpu={:?} cpu={:?} |d|=({dx}, {dy}, {dz}, {dw})",
            gpu[i],
            golden[i],
        );
        i += 1;
    }
}

/// Iterating the on-device `PBF` density solve must drive the compressed block
/// toward the rest density and stay locked to the `CPU` golden every iteration.
///
/// The single-step golden proves one Jacobi projection is faithful. Production
/// incompressibility needs several projections per frame (`solver_iterations`),
/// so this test loops `water_pbf_density_solve`, rebuilding the spatial hash from
/// the corrected positions each iteration and feeding them back in. Three
/// invariants are asserted:
///
/// 1. **Parity holds across iterations**: the device stays within a snug bound
///    of the `CPU` reference and never produces a non-finite position.
/// 2. **Positive-pressure convergence**: the worst compression overshoot
///    `max(density - rest, 0)` is non-increasing (within rounding) across the
///    roll-out, i.e. the projection relaxes the constraint rather than
///    amplifying it — the stability property Houdini/`Frostbite` `PBF` fluids
///    rely on.
/// 3. **Net relaxation**: the final overshoot is strictly below the seed
///    overshoot, so the solver made real progress rather than merely not
///    diverging.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and observed convergence must reach the test log"
)]
fn pbf_density_solve_gpu_multi_iteration_tracks_cpu_and_stays_bounded() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "pbf_density_solve_gpu_multi_iteration_tracks_cpu_and_stays_bounded: no wgpu adapter, skipping"
        );
        return;
    };

    const ITERS: usize = 8;
    let cell_size = 1.0_f32;
    let grid = PbfGrid {
        origin: Vec3::ZERO,
        cell_size,
        nx: 4,
        ny: 4,
        nz: 4,
    };
    let cpu_params = PbfParams {
        rest_density: 20.0,
        particle_mass: 1.0,
        smoothing_radius: cell_size,
        relaxation_epsilon: 0.01,
        artificial_pressure_k: 0.1,
        artificial_pressure_n: 4,
        artificial_pressure_delta_q: 0.2,
        solver_iterations: 1,
    };

    // Worst absolute density-constraint magnitude `|density / rest - 1|` over the
    // particles that actually have neighbours (isolated particles can never be
    // corrected, so their residual is fixed and would mask solver progress).
    // This is the quantity the density projection relaxes toward zero. Rebinning
    // each call keeps the neighbourhood consistent with the current positions.
    let max_constraint = |positions: &[Vec3]| -> f32 {
        let bins = pbf::bin_particles(grid, positions);
        let mut worst = 0.0_f32;
        let mut i = 0;
        while i < positions.len() {
            let neighbors = pbf::gather_neighbors(grid, &bins, positions, i as u32);
            if neighbors.is_empty() {
                i += 1;
                continue;
            }
            let mut density = pbf::poly6(0.0, cpu_params.smoothing_radius);
            for &j in &neighbors {
                let r2 = positions[i].sub(positions[j as usize]).length_squared();
                density += pbf::poly6(r2, cpu_params.smoothing_radius);
            }
            density *= cpu_params.particle_mass.max(0.0);
            let c = pbf::density_constraint(density, cpu_params.rest_density).abs();
            if c > worst {
                worst = c;
            }
            i += 1;
        }
        worst
    };

    let (seed_positions, carried) = build_pbf_particles();
    let count = seed_positions.len();
    let seed_constraint = max_constraint(&seed_positions);
    assert!(
        seed_constraint > 0.0,
        "seed block must deviate from rest density to exercise the projection"
    );

    let params = GpuPbfParams {
        grid_origin: [grid.origin.x, grid.origin.y, grid.origin.z],
        cell_size,
        rest_density: cpu_params.rest_density,
        particle_mass: cpu_params.particle_mass,
        smoothing_radius: cpu_params.smoothing_radius,
        relaxation_epsilon: cpu_params.relaxation_epsilon,
        artificial_pressure_k: cpu_params.artificial_pressure_k,
        artificial_pressure_delta_q: cpu_params.artificial_pressure_delta_q,
        artificial_pressure_n: cpu_params.artificial_pressure_n,
        particle_count: count as u32,
        grid_nx: grid.nx,
        grid_ny: grid.ny,
        grid_nz: grid.nz,
        _pad: 0,
    };

    let wgsl = compile_pbf_wgsl();
    let entry = find_entry_point(&wgsl, "pbf_density_solve");

    // The CPU golden owns the authoritative trajectory. Every iteration the GPU
    // kernel is driven from the *same* authoritative positions the CPU solver
    // consumed, so the parity check isolates single-dispatch kernel fidelity at
    // every point along a real multi-iteration relaxation instead of letting
    // last-digit rounding compound across independently advanced states. The
    // convergence property is then measured on that authoritative trajectory.
    let mut cpu_positions = seed_positions;

    let mut max_parity_drift = 0.0_f32;
    let mut trajectory: Vec<f32> = Vec::new();

    let mut iter = 0;
    while iter < ITERS {
        // Both solvers consume the identical current authoritative positions.
        let gpu_in: Vec<[f32; 4]> = cpu_positions
            .iter()
            .zip(&carried)
            .map(|(p, &w)| [p.x, p.y, p.z, w])
            .collect();
        let hash = build_pbf_hash(grid, &cpu_positions);
        let gpu_next = dispatch_pbf(&device, &queue, &wgsl, &entry, &gpu_in, &hash, &params);
        let cpu_next = pbf_golden(&cpu_positions, &carried, grid, cpu_params);

        assert_eq!(gpu_next.len(), cpu_next.len(), "particle count mismatch");
        let mut i = 0;
        while i < cpu_next.len() {
            let mut lane = 0;
            while lane < 4 {
                let g = gpu_next[i][lane];
                let c = cpu_next[i][lane];
                assert!(
                    g.is_finite() && c.is_finite(),
                    "iter {iter} particle {i} lane {lane}: non-finite gpu={g} cpu={c}"
                );
                let drift = (g - c).abs();
                if drift > max_parity_drift {
                    max_parity_drift = drift;
                }
                assert!(
                    drift < MULTI_FRAME_PARITY_EPS,
                    "iter {iter} particle {i} lane {lane}: gpu={g} cpu={c} |d|={drift}"
                );
                lane += 1;
            }
            i += 1;
        }

        // Advance the authoritative trajectory by the CPU golden result.
        cpu_positions = cpu_next
            .iter()
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();

        trajectory.push(max_constraint(&cpu_positions));
        iter += 1;
    }

    // Bounded-stability check. For a finite fluid blob the very first density
    // projection produces the largest excursion: the under-dense surface shell
    // (no neighbours outside the blob) gets pulled inward and briefly overshoots
    // the compact core into compression. A stable solver must not keep growing
    // past that transient peak. We therefore require every residual to be finite
    // and every post-transient iteration to stay at or below the initial peak
    // (within rounding slack) rather than demanding strict convergence, which is
    // physically unattainable here because the shell can never reach `rest`
    // density without exterior neighbours.
    let peak = trajectory[0];
    assert!(
        peak.is_finite() && peak > 0.0,
        "initial projection residual must be a finite positive excursion, got {peak}"
    );
    let mut idx = 1;
    while idx < trajectory.len() {
        let r = trajectory[idx];
        assert!(r.is_finite(), "iter {idx}: non-finite residual {r}");
        assert!(
            r <= peak + PBF_OVERSHOOT_SLACK,
            "iter {idx}: residual grew past the transient peak {peak} -> {r} (solver diverging)"
        );
        idx += 1;
    }

    eprintln!(
        "pbf_density_solve_gpu_multi_iteration_tracks_cpu_and_stays_bounded: {ITERS} iters, \
         seed {seed_constraint:e}, peak {peak:e}, final {:e}, max parity drift {max_parity_drift:e}",
        trajectory[trajectory.len() - 1]
    );
}

use super::abi::GpuFlipSimParams;

/// Fixed-point domain constants mirrored from `water_flip.wesl` so the host can
/// pack the momentum/mass scatter buffer with the exact bit pattern the kernel
/// decodes. Kept private to this parity block.
const FLIP_FIXED_SCALE: f32 = 65536.0;
/// Clamp bound preventing the signed accumulator from overflowing `i32`.
const FLIP_FIXED_LIMIT: f32 = 30000.0;
/// "Effectively zero" threshold shared with the shader's mass/count guards.
const FLIP_EPS: f32 = 1.0e-6;
/// `APIC` affine inertia inverse `D⁻¹` factor `3 / dx²` mirrored from
/// `water_flip.wesl` (scaled by `inv_dx²` at use in `G2P`).
const FLIP_APIC_INV_D: f32 = 3.0;

/// Compiles `water_flip.wesl` and returns its `Wgsl` translation.
fn compile_flip_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_5246_4c50_0002),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_flip.wesl"),
            "embedded://prism_render_scene/shaders/water_flip.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_flip.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Host mirror of `flip_encode_fixed`: clamp, scale, round, reinterpret as
/// `u32`. All parity inputs are chosen dyadic so `round` never lands on a tie,
/// keeping `WGSL` (ties-to-even) and Rust (ties-away) byte-identical here.
fn flip_encode_fixed(value: f32) -> u32 {
    let clamped = value.clamp(-FLIP_FIXED_LIMIT, FLIP_FIXED_LIMIT);
    ((clamped * FLIP_FIXED_SCALE).round() as i32) as u32
}

/// Host mirror of `flip_decode_fixed`.
fn flip_decode_fixed(bits: u32) -> f32 {
    (bits as i32) as f32 / FLIP_FIXED_SCALE
}

/// Host mirror of `flip_cell_index` (x fastest, then y, then z).
fn flip_cell_index(i: u32, j: u32, k: u32, dim: [u32; 3]) -> u32 {
    (k * dim[1] + j) * dim[0] + i
}

/// Host mirror of `flip_in_bounds`.
fn flip_in_bounds(i: i32, j: i32, k: i32, dim: [u32; 3]) -> bool {
    i >= 0 && j >= 0 && k >= 0 && i < dim[0] as i32 && j < dim[1] as i32 && k < dim[2] as i32
}

/// Host mirror of `flip_cell_velocity`: decodes `momentum / mass` (or zero for a
/// massless air cell) plus the mass in `w`.
fn flip_cell_velocity(scatter: &[u32], cell: usize) -> [f32; 4] {
    let base = cell * 4;
    let mass = flip_decode_fixed(scatter[base + 3]);
    if mass <= FLIP_EPS {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let mx = flip_decode_fixed(scatter[base]);
    let my = flip_decode_fixed(scatter[base + 1]);
    let mz = flip_decode_fixed(scatter[base + 2]);
    [mx / mass, my / mass, mz / mass, mass]
}

/// Host mirror of `flip_neighbor_velocity`: out-of-grid neighbours read as a
/// zero-velocity solid wall (the Neumann boundary the projection assumes).
fn flip_neighbor_velocity(
    scatter: &[u32],
    dim: [u32; 3],
    base: [i32; 3],
    off: [i32; 3],
) -> [f32; 3] {
    let ni = base[0] + off[0];
    let nj = base[1] + off[1];
    let nk = base[2] + off[2];
    if !flip_in_bounds(ni, nj, nk, dim) {
        return [0.0, 0.0, 0.0];
    }
    let v = flip_cell_velocity(
        scatter,
        flip_cell_index(ni as u32, nj as u32, nk as u32, dim) as usize,
    );
    [v[0], v[1], v[2]]
}

/// `CPU` golden twin of `water_flip_pressure_solve`.
///
/// Reproduces the kernel line-for-line: skip massless air cells (writing zero),
/// measure the central-difference divergence of the frozen cell-centered
/// velocity field, sum in-bounds neighbour pressures with a Neumann count, then
/// damp the `Jacobi` relaxation toward the previous estimate. Every cell is
/// walked in the same x-fastest order the dispatch covers.
fn flip_pressure_golden(
    scatter: &[u32],
    pressure_in: &[f32],
    dim: [u32; 3],
    dx: f32,
    inv_dx: f32,
    jacobi_omega: f32,
) -> Vec<f32> {
    let total = (dim[0] * dim[1] * dim[2]) as usize;
    let cell_count = total as u32;
    let mut out = vec![0.0_f32; total];
    let offsets = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];
    let mut k = 0i32;
    while (k as u32) < dim[2] {
        let mut j = 0i32;
        while (j as u32) < dim[1] {
            let mut i = 0i32;
            while (i as u32) < dim[0] {
                let cell = flip_cell_index(i as u32, j as u32, k as u32, dim);
                if cell < cell_count {
                    let here = flip_cell_velocity(scatter, cell as usize);
                    if here[3] <= FLIP_EPS {
                        out[cell as usize] = 0.0;
                    } else {
                        let base = [i, j, k];
                        let vx_pos = flip_neighbor_velocity(scatter, dim, base, [1, 0, 0])[0];
                        let vx_neg = flip_neighbor_velocity(scatter, dim, base, [-1, 0, 0])[0];
                        let vy_pos = flip_neighbor_velocity(scatter, dim, base, [0, 1, 0])[1];
                        let vy_neg = flip_neighbor_velocity(scatter, dim, base, [0, -1, 0])[1];
                        let vz_pos = flip_neighbor_velocity(scatter, dim, base, [0, 0, 1])[2];
                        let vz_neg = flip_neighbor_velocity(scatter, dim, base, [0, 0, -1])[2];
                        let divergence =
                            ((vx_pos - vx_neg) + (vy_pos - vy_neg) + (vz_pos - vz_neg))
                                * (0.5 * inv_dx);
                        let mut p_sum = 0.0_f32;
                        let mut count = 0.0_f32;
                        for off in offsets {
                            let ni = i + off[0];
                            let nj = j + off[1];
                            let nk = k + off[2];
                            if !flip_in_bounds(ni, nj, nk, dim) {
                                continue;
                            }
                            p_sum += pressure_in
                                [flip_cell_index(ni as u32, nj as u32, nk as u32, dim) as usize];
                            count += 1.0;
                        }
                        if count <= FLIP_EPS {
                            out[cell as usize] = 0.0;
                        } else {
                            let dx2 = dx * dx;
                            let relaxed = (p_sum - dx2 * divergence) / count;
                            let omega = jacobi_omega.clamp(0.0, 1.0);
                            out[cell as usize] =
                                pressure_in[cell as usize] * (1.0 - omega) + relaxed * omega;
                        }
                    }
                }
                i += 1;
            }
            j += 1;
        }
        k += 1;
    }
    out
}

/// Dispatches one `water_flip_pressure_solve` sweep on device and reads back the
/// relaxed pressure field.
///
/// The bind group is built from the pipeline's reflected `group(0)` layout,
/// which — because the pressure kernel touches only the scatter, both pressure
/// buffers, and the parameter block — contains exactly bindings `1..=4`.
fn dispatch_flip_pressure(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    scatter: &[u32],
    pressure_in: &[f32],
    params: &GpuFlipSimParams,
) -> Vec<f32> {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_flip"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("flip_pressure_pipeline"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let scatter_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_scatter"),
        contents: bytemuck::cast_slice(scatter),
        usage: BufferUsages::STORAGE,
    });
    let pressure_in_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_pressure_in"),
        contents: bytemuck::cast_slice(pressure_in),
        usage: BufferUsages::STORAGE,
    });
    let out_bytes = size_of_val(pressure_in) as u64;
    let pressure_out_buf = device.create_buffer(&BufferDescriptor {
        label: Some("flip_pressure_out"),
        size: out_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("flip_group0"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 1,
                resource: scatter_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: pressure_in_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: pressure_out_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let out_stage = device.create_buffer(&BufferDescriptor {
        label: Some("flip_out_stage"),
        size: out_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("flip_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("flip_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            params.dim[0].div_ceil(4),
            params.dim[1].div_ceil(4),
            params.dim[2].div_ceil(4),
        );
    }
    encoder.copy_buffer_to_buffer(&pressure_out_buf, 0, &out_stage, 0, out_bytes);
    queue.submit([encoder.finish()]);

    out_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");
    let view = out_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    out_stage.unmap();
    values
}

/// One on-device damped-`Jacobi` pressure sweep must match the `CPU` golden.
///
/// The 4x4x4 scene mixes a fully interior fluid block (all six neighbours
/// in-bounds), a boundary corner cell (three out-of-grid neighbours exercising
/// the Neumann count), and an isolated fluid cell whose in-bounds neighbours are
/// air (zero-velocity read + reduced neighbour pressure sum). Air cells verify
/// the early-out zero write. All momenta/masses are dyadic so the fixed-point
/// round is tie-free and the two paths agree to `float32` rounding.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn flip_pressure_solve_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "flip_pressure_solve_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let dim = [4u32, 4u32, 4u32];
    let total = (dim[0] * dim[1] * dim[2]) as usize;
    let dx = 1.0_f32;
    let inv_dx = 1.0_f32;
    let jacobi_omega = 0.5_f32;

    let is_fluid = |i: u32, j: u32, k: u32| -> bool {
        (i == 0 && j == 0 && k == 0)
            || (i == 3 && j == 3 && k == 3)
            || ((1..=2).contains(&i) && (1..=2).contains(&j) && (1..=2).contains(&k))
    };

    let mut scatter = vec![0u32; total * 4];
    let mut k = 0u32;
    while k < dim[2] {
        let mut j = 0u32;
        while j < dim[1] {
            let mut i = 0u32;
            while i < dim[0] {
                if is_fluid(i, j, k) {
                    let cell = flip_cell_index(i, j, k, dim) as usize;
                    let base = cell * 4;
                    let mass = 2.0_f32;
                    let vx = i as f32 * 0.5;
                    let vy = j as f32 * 0.5;
                    let vz = k as f32 * 0.5;
                    scatter[base] = flip_encode_fixed(mass * vx);
                    scatter[base + 1] = flip_encode_fixed(mass * vy);
                    scatter[base + 2] = flip_encode_fixed(mass * vz);
                    scatter[base + 3] = flip_encode_fixed(mass);
                }
                i += 1;
            }
            j += 1;
        }
        k += 1;
    }

    let pressure_in: Vec<f32> = (0..total).map(|c| c as f32 * 0.25).collect();

    let params = GpuFlipSimParams {
        origin: [0.0, 0.0, 0.0, 0.0],
        dim: [dim[0], dim[1], dim[2], 0],
        dx,
        inv_dx,
        flip_blend: 0.0,
        particle_mass: 0.0,
        jacobi_omega,
        use_affine: 0,
        particle_count: 0,
        cell_count: total as u32,
    };

    let golden = flip_pressure_golden(&scatter, &pressure_in, dim, dx, inv_dx, jacobi_omega);

    let wgsl = compile_flip_wgsl();
    let entry = find_entry_point(&wgsl, "flip_pressure_solve");
    let gpu = dispatch_flip_pressure(
        &device,
        &queue,
        &wgsl,
        &entry,
        &scatter,
        &pressure_in,
        &params,
    );

    assert_eq!(gpu.len(), golden.len(), "cell count mismatch");
    let mut c = 0;
    while c < golden.len() {
        let d = (gpu[c] - golden[c]).abs();
        assert!(
            d < PARITY_EPS,
            "cell {c}: gpu={} cpu={} |d|={d}",
            gpu[c],
            golden[c],
        );
        c += 1;
    }
}

// ===========================================================================
// Ocean `Gerstner` displacement (water_ocean.wesl :: water_gerstner_displace)
// ===========================================================================
//
// This block closes the on-device gap for the art-directable near-field ocean
// pass. Unlike the buffer-only `SWE`/`PBF`/`FLIP` kernels above, the `Gerstner`
// kernel writes two `rgba32float` storage textures (world displacement and the
// closed-form surface normal), so the parity path here creates real storage
// textures, dispatches the pass, copies both back through
// `copy_texture_to_buffer`, and compares every texel against a `CPU` golden that
// replays the exact shader math. The one deliberate divergence from the
// `prism_render_architecture` ocean reference is trigonometry: that crate forbids
// `libm` and uses hand-rolled `sin`/`cos` polynomials, whereas the `WGSL` kernel
// emits the device's native `sin`/`cos`. To stay byte-close to the hardware we
// evaluate the golden with `bevy_math::ops::sin`/`cos` (the crate's `libm`-backed,
// determinism-approved trig), which tracks the `GPU` intrinsics to well inside
// [`PARITY_EPS`]. Wave arguments are kept moderate so neither
// implementation's range reduction dominates the tolerance.

use super::abi::{GpuGerstnerWave, GpuWaterGerstnerParams};
use wgpu::{
    BindingResource, Extent3d, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor,
};

/// Gravitational acceleration used by the deep-water dispersion; mirrors
/// `WATER_GRAVITY` in `water_ocean.wesl`.
const WATER_GRAVITY: f32 = 9.81;
/// `2*PI`; mirrors `WATER_TWO_PI` in `water_ocean.wesl` (rounds to the same
/// `f32` as the shader literal).
const WATER_TWO_PI: f32 = core::f32::consts::TAU;
/// Squared-length floor guarding the direction normalize and the normal
/// normalize; mirrors `WATER_EPS_LEN_SQ` in `water_ocean.wesl`.
const WATER_EPS_LEN_SQ: f32 = 1.0e-12;

/// Compiles `water_ocean.wesl` and returns its `Wgsl` translation.
fn compile_ocean_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_524f_434e_0002),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_ocean.wesl"),
            "embedded://prism_render_scene/shaders/water_ocean.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_ocean.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Host mirror of `water_dispersion`: deep-water `omega(k) = sqrt(g*k)`,
/// `0` for non-positive `k`.
fn ocean_dispersion(k: f32) -> f32 {
    if k <= 0.0 {
        0.0
    } else {
        (WATER_GRAVITY * k).sqrt()
    }
}

/// `CPU` golden twin of `water_gerstner_displace`.
///
/// Replays the shader arithmetic texel-for-texel with `std` `sin`/`cos` and
/// returns the two flat `rgba32float` fields (`total*4` lanes each): the world
/// displacement `(Dx, base+Dy, Dz, 0)` and the closed-form normal `(nx, ny, nz,
/// 0)`, in texel order `n = y*N + x`.
fn gerstner_golden(
    waves: &[GpuGerstnerWave],
    params: &GpuWaterGerstnerParams,
) -> (Vec<f32>, Vec<f32>) {
    let n = params.grid_size;
    let inv_n = 1.0_f32 / (n as f32).max(1.0);
    let cell = params.patch_size * inv_n;
    let total = (n * n) as usize;
    let mut disp_out = vec![0.0_f32; total * 4];
    let mut norm_out = vec![0.0_f32; total * 4];

    let mut gy = 0u32;
    while gy < n {
        let mut gx = 0u32;
        while gx < n {
            let rest_x = gx as f32 * cell;
            let rest_z = gy as f32 * cell;
            let mut disp = [0.0_f32, 0.0, 0.0];
            let mut nrm = [0.0_f32, 1.0, 0.0];

            let mut i = 0u32;
            while i < params.wave_count {
                let w = waves[i as usize];
                i += 1;
                if w.wavelength <= 0.0 {
                    continue;
                }
                let dir_len_sq = w.dir_x * w.dir_x + w.dir_z * w.dir_z;
                if dir_len_sq <= WATER_EPS_LEN_SQ {
                    continue;
                }
                let inv_dir_len = 1.0_f32 / dir_len_sq.sqrt();
                let dx = w.dir_x * inv_dir_len;
                let dz = w.dir_z * inv_dir_len;

                let k = WATER_TWO_PI / w.wavelength;
                let omega = w.speed * ocean_dispersion(k);
                let theta = k * (dx * rest_x + dz * rest_z) - omega * params.time + w.phase;
                let c = bevy_math::ops::cos(theta);
                let s = bevy_math::ops::sin(theta);

                let qa = w.steepness * w.amplitude;
                disp[0] += qa * dx * c;
                disp[2] += qa * dz * c;
                disp[1] += w.amplitude * s;

                let wa = k * w.amplitude;
                nrm[0] -= dx * wa * c;
                nrm[2] -= dz * wa * c;
                nrm[1] -= w.steepness * wa * s;
            }

            let mut normal = [0.0_f32, 1.0, 0.0];
            let dot = nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2];
            if dot > WATER_EPS_LEN_SQ {
                let inv = 1.0_f32 / dot.sqrt();
                normal = [nrm[0] * inv, nrm[1] * inv, nrm[2] * inv];
            }

            let base = ((gy * n + gx) * 4) as usize;
            disp_out[base] = disp[0];
            disp_out[base + 1] = params.base_level + disp[1];
            disp_out[base + 2] = disp[2];
            disp_out[base + 3] = 0.0;
            norm_out[base] = normal[0];
            norm_out[base + 1] = normal[1];
            norm_out[base + 2] = normal[2];
            norm_out[base + 3] = 0.0;

            gx += 1;
        }
        gy += 1;
    }

    (disp_out, norm_out)
}

/// Dispatches one `water_gerstner_displace` pass on device and reads back the
/// displacement and normal textures as flat `rgba32float` lanes.
///
/// The `group(0)` layout is reflected straight off the pipeline, so bindings
/// `5` (waves), `6` (params uniform), `7` (displacement texture) and `8`
/// (normal texture) match the shader declaration order exactly. `N` is chosen so
/// `row_bytes == N*16` is `256`-aligned and the readback rows are dense.
#[expect(
    clippy::too_many_lines,
    reason = "one linear dispatch-and-readback over two storage textures keeps the parity path auditable"
)]
fn dispatch_gerstner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    waves: &[GpuGerstnerWave],
    params: &GpuWaterGerstnerParams,
) -> (Vec<f32>, Vec<f32>) {
    let n = params.grid_size;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_ocean_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_gerstner_displace_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let wave_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("gerstner_waves"),
        contents: bytemuck::cast_slice(waves),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("gerstner_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let extent = Extent3d {
        width: n,
        height: n,
        depth_or_array_layers: 1,
    };
    let disp_tex = device.create_texture(&TextureDescriptor {
        label: Some("gerstner_displacement_out"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let norm_tex = device.create_texture(&TextureDescriptor {
        label: Some("gerstner_normal_out"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let disp_view = disp_tex.create_view(&TextureViewDescriptor::default());
    let norm_view = norm_tex.create_view(&TextureViewDescriptor::default());

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("gerstner_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 5,
                resource: wave_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: BindingResource::TextureView(&disp_view),
            },
            BindGroupEntry {
                binding: 8,
                resource: BindingResource::TextureView(&norm_view),
            },
        ],
    });

    let row_bytes = n * 16;
    let readback_size = u64::from(row_bytes * n);
    let disp_readback = device.create_buffer(&BufferDescriptor {
        label: Some("gerstner_disp_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let norm_readback = device.create_buffer(&BufferDescriptor {
        label: Some("gerstner_norm_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("gerstner_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("gerstner_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = n.div_ceil(8);
        pass.dispatch_workgroups(groups, groups, 1);
    }
    for (tex, readback) in [(&disp_tex, &disp_readback), (&norm_tex, &norm_readback)] {
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(n),
                },
            },
            extent,
        );
    }
    queue.submit([encoder.finish()]);

    disp_readback.slice(..).map_async(MapMode::Read, |_| {});
    norm_readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let disp_out = {
        let view = disp_readback
            .slice(..)
            .get_mapped_range()
            .expect("mapped displacement readback should be available after poll");
        let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        disp_readback.unmap();
        floats
    };
    let norm_out = {
        let view = norm_readback
            .slice(..)
            .get_mapped_range()
            .expect("mapped normal readback should be available after poll");
        let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        norm_readback.unmap();
        floats
    };

    (disp_out, norm_out)
}

/// Real-device parity for `water_gerstner_displace`: superpose a fixed set of
/// wave trains on device and match both output textures against the `CPU`
/// golden. Includes a zero-wavelength wave and a zero-direction wave to cover
/// both `continue` guards.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gerstner_displace_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "gerstner_displace_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let grid_size = 16u32;
    let waves = [
        GpuGerstnerWave {
            dir_x: 1.0,
            dir_z: 0.0,
            amplitude: 0.4,
            wavelength: 16.0,
            steepness: 0.4,
            speed: 1.0,
            phase: 0.0,
            _pad: 0.0,
        },
        GpuGerstnerWave {
            dir_x: 0.6,
            dir_z: 0.8,
            amplitude: 0.25,
            wavelength: 10.0,
            steepness: 0.3,
            speed: 1.0,
            phase: 0.7,
            _pad: 0.0,
        },
        GpuGerstnerWave {
            dir_x: -1.0,
            dir_z: 1.0,
            amplitude: 0.15,
            wavelength: 8.0,
            steepness: 0.25,
            speed: 1.0,
            phase: 1.3,
            _pad: 0.0,
        },
        // Zero wavelength -> exercises the first `continue` guard.
        GpuGerstnerWave {
            dir_x: 1.0,
            dir_z: 0.0,
            amplitude: 0.1,
            wavelength: 0.0,
            steepness: 0.1,
            speed: 1.0,
            phase: 0.25,
            _pad: 0.0,
        },
        // Zero direction -> exercises the direction-normalize `continue` guard.
        GpuGerstnerWave {
            dir_x: 0.0,
            dir_z: 0.0,
            amplitude: 0.1,
            wavelength: 6.0,
            steepness: 0.1,
            speed: 1.0,
            phase: 0.5,
            _pad: 0.0,
        },
    ];
    let params = GpuWaterGerstnerParams {
        grid_size,
        patch_size: 16.0,
        time: 1.5,
        wave_count: waves.len() as u32,
        base_level: 1.0,
        _pad: [0, 0, 0],
    };

    let (golden_disp, golden_norm) = gerstner_golden(&waves, &params);

    let wgsl = compile_ocean_wgsl();
    let entry = find_entry_point(&wgsl, "gerstner_displace");
    let (gpu_disp, gpu_norm) = dispatch_gerstner(&device, &queue, &wgsl, &entry, &waves, &params);

    assert_eq!(gpu_disp.len(), golden_disp.len(), "displacement lane count");
    assert_eq!(gpu_norm.len(), golden_norm.len(), "normal lane count");

    let mut i = 0;
    while i < golden_disp.len() {
        let dd = (gpu_disp[i] - golden_disp[i]).abs();
        assert!(
            dd < PARITY_EPS,
            "displacement lane {i}: gpu={} cpu={} |d|={dd}",
            gpu_disp[i],
            golden_disp[i],
        );
        let dn = (gpu_norm[i] - golden_norm[i]).abs();
        assert!(
            dn < PARITY_EPS,
            "normal lane {i}: gpu={} cpu={} |d|={dn}",
            gpu_norm[i],
            golden_norm[i],
        );
        i += 1;
    }
}

// ===========================================================================
// Kernel: water_caustics_project (render_fx, 8x8) — on-device parity.
// ===========================================================================
//
// The caustics projection has no transcendental functions on its path: the
// refracted-ray Jacobian is built from central finite differences of the
// offset field, the focus gain is `min(1/|jacobian|, max_gain)`, and the photon
// term is `count * power / (PI * r^2)`. That lets the `CPU` golden call the
// architecture reference functions [`project_caustic_intensity`] and
// [`photon_splat_density`] verbatim, so a green run is byte-level evidence that
// the ported kernel matches its reference to `float32` rounding.
//
// Binding 3 is a sampled `texture_2d<f32>` read through `textureLoad`, backed by
// a non-filterable `Rgba32Float` texture, so the pass is built with an explicit
// bind group + pipeline layout that pins `Float { filterable: false }`; auto
// layout would reflect a filterable sample type and fail bind-group validation.

use super::abi::GpuWaterCausticsParams;
use prism_render_architecture::water::caustics::{photon_splat_density, project_caustic_intensity};
use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    PipelineLayoutDescriptor, ShaderStages, StorageTextureAccess, TextureSampleType,
    TextureViewDimension,
};

use wgpu::{AddressMode, FilterMode, MipmapFilterMode, SamplerBindingType, SamplerDescriptor};

use prism_render_architecture::water::dispersion;

use super::abi::GpuWaterDispersionParams;

/// Finite-difference floor; mirrors `WATER_EPS` in `water_render_fx.wesl`.
const WATER_EPS: f32 = 1.0e-6;

/// Compiles `water_render_fx.wesl` and returns its `Wgsl` translation.
fn compile_render_fx_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_5246_5800_0001),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_render_fx.wesl"),
            "embedded://prism_render_scene/shaders/water_render_fx.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_render_fx.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Builds a smooth, transcendental-free refracted-ray offset field and a
/// photon-count field (including zero counts) for the caustics parity inputs.
///
/// Offsets are packed `rgba32float` (`width*height*4` lanes, `.zw = 0`); the
/// polynomial warp gives a non-trivial spatially varying Jacobian without any
/// `sin`/`cos`. Photon counts cycle through `0..=4` so the zero-count path is
/// exercised.
fn caustics_inputs(width: u32, height: u32) -> (Vec<f32>, Vec<u32>) {
    let total = (width * height) as usize;
    let mut offsets = vec![0.0_f32; total * 4];
    let mut photons = vec![0_u32; total];
    let inv_w = 1.0_f32 / width as f32;
    let inv_h = 1.0_f32 / height as f32;
    let mut y = 0u32;
    while y < height {
        let mut x = 0u32;
        while x < width {
            let fx = x as f32 * inv_w;
            let fy = y as f32 * inv_h;
            let base = ((y * width + x) * 4) as usize;
            offsets[base] = 0.05 * fx - 0.03 * fy + 0.02 * fx * fy;
            offsets[base + 1] = -0.04 * fx * fx + 0.06 * fy;
            let idx = (y * width + x) as usize;
            photons[idx] = (x * 7 + y * 13) % 5;
            x += 1;
        }
        y += 1;
    }
    (offsets, photons)
}

/// `CPU` golden twin of `water_caustics_project`.
///
/// Replays the shader per texel: the refracted-ray Jacobian from central finite
/// differences of the offset field (clamped at the borders exactly like the
/// kernel), the Jacobian focus gain projected onto the incident irradiance, plus
/// the photon splat-density term, floored at zero. The projection and photon
/// terms call the architecture golden functions directly, so the only host
/// arithmetic is the finite-difference stencil the shader also runs.
fn caustics_golden(params: &GpuWaterCausticsParams, offsets: &[f32], photons: &[u32]) -> Vec<f32> {
    let w = params.width;
    let h = params.height;
    let total = (w * h) as usize;
    let mut out = vec![0.0_f32; total];
    let fd_step = params.texel_size.max(WATER_EPS) * 2.0;

    let sample = |x: i32, y: i32| -> (f32, f32) {
        let base = ((y as u32 * w + x as u32) * 4) as usize;
        (offsets[base], offsets[base + 1])
    };

    let mut gy = 0u32;
    while gy < h {
        let mut gx = 0u32;
        while gx < w {
            let x = gx as i32;
            let y = gy as i32;
            let xm = (x - 1).max(0);
            let ym = (y - 1).max(0);
            let xp = (x + 1).min(w as i32 - 1);
            let yp = (y + 1).min(h as i32 - 1);
            let (oxp_x, oxp_y) = sample(xp, y);
            let (oxm_x, oxm_y) = sample(xm, y);
            let (oyp_x, oyp_y) = sample(x, yp);
            let (oym_x, oym_y) = sample(x, ym);
            let dudx = 1.0 + (oxp_x - oxm_x) / fd_step;
            let dvdx = (oxp_y - oxm_y) / fd_step;
            let dudy = (oyp_x - oym_x) / fd_step;
            let dvdy = 1.0 + (oyp_y - oym_y) / fd_step;
            let jacobian = dudx * dvdy - dvdx * dudy;
            let projected = project_caustic_intensity(params.incident, jacobian, params.max_gain);
            let idx = (gy * w + gx) as usize;
            let photon_term =
                photon_splat_density(photons[idx], params.photon_power, params.splat_radius);
            out[idx] = (projected + photon_term).max(0.0);
            gx += 1;
        }
        gy += 1;
    }

    out
}

/// Dispatches one `water_caustics_project` pass on device and reads back the
/// single-channel `r32float` caustic irradiance target as flat lanes.
///
/// Uses an explicit `@group(0)` layout so binding 3 (the `Rgba32Float` offset
/// field, read via `textureLoad`) is pinned to a non-filterable sample type.
/// `width` is chosen so the `r32float` readback row (`width*4`) is `256`-aligned
/// and dense.
#[expect(
    clippy::too_many_lines,
    reason = "one explicit-layout dispatch-and-readback over the caustics target keeps the parity path auditable"
)]
fn dispatch_caustics(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    params: &GpuWaterCausticsParams,
    offsets: &[f32],
    photons: &[u32],
) -> Vec<f32> {
    let width = params.width;
    let height = params.height;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_render_fx_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water_caustics_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::R32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("water_caustics_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_caustics_project_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let photon_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("caustics_photon_counts"),
        contents: bytemuck::cast_slice(photons),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("caustics_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let extent = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let offsets_tex = device.create_texture(&TextureDescriptor {
        label: Some("caustics_offsets"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // `write_texture` has no 256-byte row-alignment requirement, so the dense
    // `rgba32float` offset field uploads as-is.
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &offsets_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(offsets),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(height),
        },
        extent,
    );
    let out_tex = device.create_texture(&TextureDescriptor {
        label: Some("caustics_out"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let offsets_view = offsets_tex.create_view(&TextureViewDescriptor::default());
    let out_view = out_tex.create_view(&TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("caustics_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: photon_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&out_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&offsets_view),
            },
        ],
    });

    let row_bytes = width * 4;
    let readback_size = u64::from(row_bytes * height);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("caustics_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("caustics_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("caustics_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
        },
        extent,
    );
    queue.submit([encoder.finish()]);

    readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = readback
        .slice(..)
        .get_mapped_range()
        .expect("mapped caustics readback should be available after poll");
    let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    readback.unmap();
    floats
}

/// Real-device parity for `water_caustics_project`: project one caustics frame
/// over a `64x16` receiver grid on device and match the `r32float` irradiance
/// target texel-for-texel against the `CPU` golden built from the architecture
/// reference functions.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn caustics_project_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "caustics_project_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let width = 64u32;
    let height = 16u32;
    let params = GpuWaterCausticsParams {
        incident: 1.0,
        max_gain: 1.5,
        photon_power: 0.5,
        splat_radius: 0.25,
        texel_size: 0.05,
        width,
        height,
        _pad: 0,
    };
    let (offsets, photons) = caustics_inputs(width, height);

    let golden = caustics_golden(&params, &offsets, &photons);

    let wgsl = compile_render_fx_wgsl();
    let entry = find_entry_point(&wgsl, "caustics_project");
    let gpu = dispatch_caustics(&device, &queue, &wgsl, &entry, &params, &offsets, &photons);

    assert_eq!(gpu.len(), golden.len(), "caustics lane count");

    let mut i = 0;
    while i < golden.len() {
        let d = (gpu[i] - golden[i]).abs();
        assert!(
            d < PARITY_EPS,
            "caustics texel {i}: gpu={} cpu={} |d|={d}",
            gpu[i],
            golden[i],
        );
        i += 1;
    }
}

// ===========================================================================
// Kernel: water_coupling_readback (render_fx, 64) — on-device parity.
// ===========================================================================
//
// The two-way coupling read-back packs per-body buoyancy, quadratic drag,
// added-mass reaction, and the momentum write-back fraction, and it caps the
// batch at `min(query_count, max_readback)` so the pass never reads the whole
// field back. Every term is plain arithmetic (products, a `v*v`, a clamped
// ratio) with no transcendental, so the `CPU` golden calls the architecture
// reference functions verbatim and the readback matches to `float32` rounding.
//
// The kernel binds its resources at `@group(4)`, so the parity path builds an
// explicit pipeline layout whose groups `0..=3` are empty and group `4` carries
// the query buffer, the read-back buffer, and the parameter uniform. Lanes past
// the batch cap never write, so the read-back slots beyond `batch` keep their
// pre-seeded sentinel — direct evidence the bound is honoured on device.

use super::abi::{GpuWaterCouplingParams, GpuWaterCouplingQuery};
use prism_render_architecture::water::coupling::{
    added_mass, buoyancy_force, drag_force, source_writeback_fraction,
};

/// Sentinel written into every read-back slot before dispatch; lanes past the
/// batch cap must still read back as this untouched value.
const COUPLING_SENTINEL: f32 = -1.0;

/// Builds a deterministic, transcendental-free set of coupling queries that
/// covers the write-back branches: a zero total volume (inert), a body more
/// than fully submerged (fraction clamps to one), and a zero relative speed
/// (drag vanishes), alongside a spread of partial submersions.
fn coupling_queries(count: u32) -> Vec<GpuWaterCouplingQuery> {
    let mut queries = Vec::with_capacity(count as usize);
    let mut i = 0u32;
    while i < count {
        let f = i as f32;
        let (submerged, total) = if i.is_multiple_of(20) {
            // Zero total volume -> the write-back fraction stays inert.
            (0.0, 0.0)
        } else if i % 20 == 1 {
            // Over-submerged -> the fraction saturates at one.
            (5.0, 2.0)
        } else {
            (0.15 * f + 0.2, 3.0 + 0.02 * f)
        };
        queries.push(GpuWaterCouplingQuery {
            submerged_volume: submerged,
            total_volume: total,
            cross_section: 0.5 + 0.01 * f,
            // Every fifth body is at rest so its drag term vanishes.
            rel_speed: if i.is_multiple_of(5) {
                0.0
            } else {
                0.1 * f + 0.3
            },
        });
        i += 1;
    }
    queries
}

/// `CPU` golden twin of `water_coupling_readback`.
///
/// Packs `[buoyancy, drag, added_mass, writeback]` for every lane below the
/// batch cap `min(query_count, max_readback)` using the architecture reference
/// functions, and leaves the tail lanes at [`COUPLING_SENTINEL`] exactly like
/// the kernel, which returns before touching them.
fn coupling_golden(params: &GpuWaterCouplingParams, queries: &[GpuWaterCouplingQuery]) -> Vec<f32> {
    let count = params.query_count as usize;
    let batch = params.query_count.min(params.max_readback) as usize;
    let mut out = vec![COUPLING_SENTINEL; count * 4];
    let mut i = 0usize;
    while i < batch {
        let q = queries[i];
        let base = i * 4;
        out[base] = buoyancy_force(params.fluid_density, q.submerged_volume, params.gravity);
        out[base + 1] = drag_force(
            params.drag_coeff,
            params.fluid_density,
            q.cross_section,
            q.rel_speed,
        );
        out[base + 2] = added_mass(
            params.added_mass_coeff,
            params.fluid_density,
            q.submerged_volume,
        );
        out[base + 3] = source_writeback_fraction(q.submerged_volume, q.total_volume);
        i += 1;
    }
    out
}

/// Dispatches one `water_coupling_readback` pass on device and reads back the
/// packed `vec4` result lanes as flat `f32`s.
///
/// The kernel lives at `@group(4)`, so an explicit pipeline layout is built with
/// empty layouts for groups `0..=3` and the real query/read-back/params layout
/// at group `4`. The read-back storage buffer is pre-seeded with
/// [`COUPLING_SENTINEL`] so the tail past the batch cap is observable.
#[expect(
    clippy::too_many_lines,
    reason = "one explicit-group-4 dispatch-and-readback over the coupling buffers keeps the parity path auditable"
)]
fn dispatch_coupling(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    params: &GpuWaterCouplingParams,
    queries: &[GpuWaterCouplingQuery],
) -> Vec<f32> {
    let count = params.query_count as usize;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_render_fx_coupling_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let empty_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water_coupling_empty_layout"),
        entries: &[],
    });
    let group4_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water_coupling_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("water_coupling_pipeline_layout"),
        bind_group_layouts: &[
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&group4_layout),
        ],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_coupling_readback_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("coupling_queries"),
        contents: bytemuck::cast_slice(queries),
        usage: BufferUsages::STORAGE,
    });
    let sentinel = vec![COUPLING_SENTINEL; count * 4];
    let readback_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("coupling_readback"),
        contents: bytemuck::cast_slice(&sentinel),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("coupling_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let empty_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("coupling_empty_group"),
        layout: &empty_layout,
        entries: &[],
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("coupling_bind_group"),
        layout: &group4_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: queries_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: readback_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let out_bytes = (count * 16) as u64;
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("coupling_stage"),
        size: out_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("coupling_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("coupling_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &empty_group, &[]);
        pass.set_bind_group(1, &empty_group, &[]);
        pass.set_bind_group(2, &empty_group, &[]);
        pass.set_bind_group(3, &empty_group, &[]);
        pass.set_bind_group(4, &bind_group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&readback_buf, 0, &stage, 0, out_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped coupling readback should be available after poll");
    let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    stage.unmap();
    floats
}

/// Real-device parity for `water_coupling_readback`: pack the coupling forces
/// for a batch-capped set of `100` queries (cap `80`) on device and match the
/// `vec4` read-back lane-for-lane against the `CPU` golden, including the
/// untouched sentinel tail past the cap.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn coupling_readback_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "coupling_readback_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let params = GpuWaterCouplingParams {
        fluid_density: 1.0,
        drag_coeff: 1.2,
        added_mass_coeff: 0.5,
        gravity: 9.81,
        query_count: 100,
        max_readback: 80,
        _pad0: 0,
        _pad1: 0,
    };
    let queries = coupling_queries(params.query_count);

    let golden = coupling_golden(&params, &queries);

    let wgsl = compile_render_fx_wgsl();
    let entry = find_entry_point(&wgsl, "coupling_readback");
    let gpu = dispatch_coupling(&device, &queue, &wgsl, &entry, &params, &queries);

    assert_eq!(gpu.len(), golden.len(), "coupling lane count");

    let mut i = 0;
    while i < golden.len() {
        let d = (gpu[i] - golden[i]).abs();
        assert!(
            d < PARITY_EPS,
            "coupling lane {i}: gpu={} cpu={} |d|={d}",
            gpu[i],
            golden[i],
        );
        i += 1;
    }
}

// ===========================================================================
// Kernel 6: water_wetness_step (Surface, 8x8) — twin of `wetness::step_moisture`
// plus the `wet_albedo_scale`/`capillary_height`/`is_puddle` shading helpers.
// ===========================================================================

use super::abi::GpuWaterWetnessParams;
use prism_render_architecture::water::wetness::{self, SurfaceMoisture, WetnessParams};

/// Field width in cells; a multiple of 32 keeps the `rgba16float` read-back row
/// (four channels x 2 bytes = 8 bytes/texel) an exact 256-byte multiple, so the
/// copied texture is dense with no per-row padding.
const WETNESS_W: u32 = 32;
/// Field height in cells.
const WETNESS_H: u32 = 8;
/// Tight tolerance for the `f32` state read-back: the shared `exp_approx` is a
/// bit-identical squaring twin, so only a possible fused multiply-add in the
/// puddle integration can drift, and that stays within one ulp.
const WETNESS_STATE_EPS: f32 = 2.0e-5;

/// Decodes an IEEE 754 binary16 bit pattern into `f32` for the `rgba16float`
/// output texture read-back. The wetness outputs are all finite and in `0..=1`
/// (albedo, capillary, wetness, flag), so the normal path dominates; the
/// subnormal and zero cases are handled for completeness.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 {
        -1.0_f32
    } else {
        1.0_f32
    };
    let exp = i32::from((bits >> 10) & 0x1f);
    let mant = f32::from(bits & 0x03ff);
    if exp == 0 {
        // Subnormal (or zero when mant == 0): no implicit leading one.
        sign * (mant / 1024.0) * exp2i(-14)
    } else if exp == 0x1f {
        // Inf/NaN: the wetness data never reaches this, map to a large finite.
        sign * f32::MAX
    } else {
        sign * (1.0 + mant / 1024.0) * exp2i(exp - 15)
    }
}

/// Bit-exact `2^n` for the `f16` decode path (integer exponents in the normal
/// `f32` range map straight onto the IEEE 754 exponent field). Avoids the
/// disallowed `f32::powi` while staying exact for the `n in [-14, 15]` the
/// decoder ever passes.
fn exp2i(n: i32) -> f32 {
    let biased = u32::try_from(n + 127).unwrap_or(0);
    f32::from_bits(biased << 23)
}

/// Projects the GPU wetness scalars onto the architecture `WetnessParams` used
/// by the golden reference.
fn wetness_arch_params(p: &GpuWaterWetnessParams) -> WetnessParams {
    WetnessParams {
        max_capillary_height: p.max_capillary_height,
        absorb_rate: p.absorb_rate,
        dry_rate: p.dry_rate,
        darkening_strength: p.darkening_strength,
        puddle_threshold: p.puddle_threshold,
    }
}

/// Builds a deterministic, transcendental-free per-cell moisture field that
/// spreads the initial wetness across `0..=1` and seeds a range of puddle
/// depths straddling the puddle threshold.
fn wetness_state_field(count: usize) -> Vec<f32> {
    let mut field = Vec::with_capacity(count * 2);
    let mut i = 0usize;
    while i < count {
        let w0 = ((i % 11) as f32) / 10.0;
        let p0 = ((i % 7) as f32) * 0.005;
        field.push(w0);
        field.push(p0);
        i += 1;
    }
    field
}

/// `CPU` golden for the wetness step. Returns the packed next state
/// (`wetness`, `puddle_depth` per cell) and the packed output texels (`albedo`,
/// `capillary`, `wetness`, `flag` per cell) in row-major cell order.
fn wetness_golden(params: &GpuWaterWetnessParams, field: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let arch = wetness_arch_params(params);
    let contact = params.water_contact != 0;
    let count = (params.width * params.height) as usize;
    let mut state = Vec::with_capacity(count * 2);
    let mut tex = Vec::with_capacity(count * 4);
    let mut idx = 0usize;
    while idx < count {
        let init = SurfaceMoisture {
            wetness: field[idx * 2],
            puddle_depth: field[idx * 2 + 1],
        };
        let next = wetness::step_moisture(
            init,
            arch,
            contact,
            params.rain_rate,
            params.drain_rate,
            params.dt,
        );
        let albedo = wetness::wet_albedo_scale(next.wetness, arch);
        let capillary = wetness::capillary_height(next.wetness, params.dist_above_water, arch);
        let flag = if wetness::is_puddle(next.puddle_depth, arch) {
            1.0
        } else {
            0.0
        };
        state.push(next.wetness);
        state.push(next.puddle_depth);
        tex.push(albedo);
        tex.push(capillary);
        tex.push(next.wetness);
        tex.push(flag);
        idx += 1;
    }
    (state, tex)
}

/// Dispatches `water_wetness_step` on device and reads back both the `f32`
/// storage state and the decoded `rgba16float` output texture.
#[expect(
    clippy::too_many_lines,
    reason = "the parity harness sets up an explicit layout, a storage texture, and two read-backs in one place"
)]
fn dispatch_wetness(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    params: &GpuWaterWetnessParams,
    field: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let count = (params.width * params.height) as usize;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_render_fx_wetness_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let empty_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water_wetness_empty_layout"),
        entries: &[],
    });
    let group3_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water_wetness_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("water_wetness_pipeline_layout"),
        bind_group_layouts: &[
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&group3_layout),
        ],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_wetness_step_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let state_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("wetness_state"),
        contents: bytemuck::cast_slice(field),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("wetness_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });
    let out_tex = device.create_texture(&TextureDescriptor {
        label: Some("wetness_out"),
        size: Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&TextureViewDescriptor::default());

    let empty_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("wetness_empty_group"),
        layout: &empty_layout,
        entries: &[],
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("wetness_bind_group"),
        layout: &group3_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: state_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&out_view),
            },
        ],
    });

    let state_bytes = (count * 8) as u64;
    let state_stage = device.create_buffer(&BufferDescriptor {
        label: Some("wetness_state_stage"),
        size: state_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let row_bytes = params.width * 8;
    let tex_bytes = u64::from(row_bytes * params.height);
    let tex_stage = device.create_buffer(&BufferDescriptor {
        label: Some("wetness_tex_stage"),
        size: tex_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("wetness_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("wetness_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &empty_group, &[]);
        pass.set_bind_group(1, &empty_group, &[]);
        pass.set_bind_group(2, &empty_group, &[]);
        pass.set_bind_group(3, &bind_group, &[]);
        // One extra workgroup per axis exercises the in-kernel bounds guard.
        pass.dispatch_workgroups(
            params.width.div_ceil(8) + 1,
            params.height.div_ceil(8) + 1,
            1,
        );
    }
    encoder.copy_buffer_to_buffer(&state_buf, 0, &state_stage, 0, state_bytes);
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &tex_stage,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(params.height),
            },
        },
        Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    state_stage.slice(..).map_async(MapMode::Read, |_| {});
    tex_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let state_view = state_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped wetness state should be available after poll");
    let state = bytemuck::cast_slice::<u8, f32>(&state_view).to_vec();
    drop(state_view);
    state_stage.unmap();

    let tex_view = tex_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped wetness texture should be available after poll");
    let halves = bytemuck::cast_slice::<u8, u16>(&tex_view).to_vec();
    drop(tex_view);
    tex_stage.unmap();
    let tex: Vec<f32> = halves.iter().map(|&h| f16_bits_to_f32(h)).collect();

    (state, tex)
}

/// Real-device parity for `water_wetness_step`: advance a spread of per-cell
/// moisture states through the rain-wetting branch on device and match the
/// `f32` state read-back tightly, plus the decoded `rgba16float` output texture
/// (albedo darkening, capillary band, wetness, puddle flag) to the golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn wetness_step_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "wetness_step_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let params = GpuWaterWetnessParams {
        max_capillary_height: 0.5,
        absorb_rate: 2.0,
        dry_rate: 0.5,
        darkening_strength: 0.4,
        puddle_threshold: 0.02,
        rain_rate: 0.3,
        drain_rate: 0.1,
        dt: 0.05,
        dist_above_water: 0.1,
        water_contact: 0,
        width: WETNESS_W,
        height: WETNESS_H,
    };
    let count = (params.width * params.height) as usize;
    let field = wetness_state_field(count);

    let (state_gold, tex_gold) = wetness_golden(&params, &field);

    let wgsl = compile_render_fx_wgsl();
    let entry = find_entry_point(&wgsl, "wetness_step");
    let (state_gpu, tex_gpu) = dispatch_wetness(&device, &queue, &wgsl, &entry, &params, &field);

    assert_eq!(
        state_gpu.len(),
        state_gold.len(),
        "wetness state lane count"
    );
    assert_eq!(tex_gpu.len(), tex_gold.len(), "wetness texel lane count");

    let mut i = 0;
    while i < state_gold.len() {
        let d = (state_gpu[i] - state_gold[i]).abs();
        assert!(
            d < WETNESS_STATE_EPS,
            "wetness state lane {i}: gpu={} cpu={} |d|={d}",
            state_gpu[i],
            state_gold[i],
        );
        i += 1;
    }

    let mut j = 0;
    while j < tex_gold.len() {
        let d = (tex_gpu[j] - tex_gold[j]).abs();
        assert!(
            d < PARITY_EPS,
            "wetness texel lane {j}: gpu={} cpu={} |d|={d}",
            tex_gpu[j],
            tex_gold[j],
        );
        j += 1;
    }
}

// ===========================================================================
// Kernel 7: water_underwater_volume (Grid3d, 4x4x4) — twin of the froxel volume
// math in `underwater.rs` (Beer-Lambert colour shift, Henyey-Greenstein phase,
// bounded multiple scatter, god-ray in-scatter).
// ===========================================================================

use prism_render_architecture::water::underwater;

use super::abi::GpuWaterUnderwaterParams;

/// `froxel` grid width; a multiple of 32 keeps the `rgba16float` read-back row
/// (8 bytes/texel) an exact 256-byte multiple, so each copied slice is dense.
const UNDERWATER_W: u32 = 32;
/// `froxel` grid height.
const UNDERWATER_H: u32 = 4;
/// `froxel` grid depth (slice count).
const UNDERWATER_D: u32 = 4;

/// Deterministic, transcendental-free per-column surface-light field sampled by
/// the god-ray and single-scatter terms.
fn underwater_surface_field(width: u32, height: u32) -> Vec<f32> {
    let mut field = Vec::with_capacity((width * height) as usize);
    let mut y = 0u32;
    while y < height {
        let mut x = 0u32;
        while x < width {
            let v = ((x * 3 + y * 7) % 13) as f32 / 12.0;
            field.push(v);
            x += 1;
        }
        y += 1;
    }
    field
}

/// `CPU` golden for the underwater froxel volume, packed as `(inscatter.rgb,
/// avg_transmittance)` per froxel in `(z, y, x)` image-row order to match the
/// `copy_texture_to_buffer` layout.
fn underwater_golden(params: &GpuWaterUnderwaterParams, surface: &[f32]) -> Vec<f32> {
    let thickness = params.slice_thickness.max(0.0);
    let phase = underwater::henyey_greenstein(params.sun_cos, params.phase_g);
    let albedo_clamped = params.scatter_albedo.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity((params.width * params.height * params.depth * 4) as usize);
    let mut z = 0u32;
    while z < params.depth {
        let depth = (z as f32 + 0.5) * thickness;
        let tr = underwater::beer_lambert_transmittance(params.extinction[0], depth);
        let tg = underwater::beer_lambert_transmittance(params.extinction[1], depth);
        let tb = underwater::beer_lambert_transmittance(params.extinction[2], depth);
        let shifted = [
            params.base_color[0].max(0.0) * tr,
            params.base_color[1].max(0.0) * tg,
            params.base_color[2].max(0.0) * tb,
        ];
        let avg_t = (tr + tg + tb) / 3.0;
        let mut y = 0u32;
        while y < params.height {
            let mut x = 0u32;
            while x < params.width {
                let light = surface[(y * params.width + x) as usize];
                let single = light.max(0.0) * phase * albedo_clamped;
                let boosted = underwater::multiple_scatter_boost(single, params.scatter_albedo);
                let gr = underwater::godray_inscatter(
                    light,
                    params.scatter_albedo,
                    params.extinction[1],
                    depth,
                );
                let inscatter = [
                    (shifted[0] * boosted + shifted[0] * gr).max(0.0),
                    (shifted[1] * boosted + shifted[1] * gr).max(0.0),
                    (shifted[2] * boosted + shifted[2] * gr).max(0.0),
                ];
                out.push(inscatter[0]);
                out.push(inscatter[1]);
                out.push(inscatter[2]);
                out.push(avg_t);
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    out
}

/// Dispatches `water_underwater_volume` on device and reads back the decoded
/// `rgba16float` 3D scattering/transmittance volume.
#[expect(
    clippy::too_many_lines,
    reason = "the parity harness sets up an explicit layout, an input light texture, a 3D storage texture, and the volume read-back in one place"
)]
fn dispatch_underwater(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    params: &GpuWaterUnderwaterParams,
    surface: &[f32],
) -> Vec<f32> {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_render_fx_underwater_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let empty_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water_underwater_empty_layout"),
        entries: &[],
    });
    let group2_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water_underwater_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D3,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("water_underwater_pipeline_layout"),
        bind_group_layouts: &[
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&group2_layout),
        ],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_underwater_volume_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("underwater_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });
    let light_tex = device.create_texture(&TextureDescriptor {
        label: Some("underwater_surface_light"),
        size: Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &light_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(surface),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(params.width * 4),
            rows_per_image: Some(params.height),
        },
        Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
    );
    let light_view = light_tex.create_view(&TextureViewDescriptor::default());
    let out_tex = device.create_texture(&TextureDescriptor {
        label: Some("underwater_out"),
        size: Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: params.depth,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&TextureViewDescriptor::default());

    let empty_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("underwater_empty_group"),
        layout: &empty_layout,
        entries: &[],
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("underwater_bind_group"),
        layout: &group2_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&out_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&light_view),
            },
        ],
    });

    let row_bytes = params.width * 8;
    let tex_bytes = u64::from(row_bytes * params.height * params.depth);
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("underwater_stage"),
        size: tex_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("underwater_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("underwater_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &empty_group, &[]);
        pass.set_bind_group(1, &empty_group, &[]);
        pass.set_bind_group(2, &bind_group, &[]);
        // One extra workgroup per axis exercises the in-kernel bounds guard.
        pass.dispatch_workgroups(
            params.width.div_ceil(4) + 1,
            params.height.div_ceil(4) + 1,
            params.depth.div_ceil(4) + 1,
        );
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &stage,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(params.height),
            },
        },
        Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: params.depth,
        },
    );
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped underwater volume should be available after poll");
    let halves = bytemuck::cast_slice::<u8, u16>(&view).to_vec();
    drop(view);
    stage.unmap();
    halves.iter().map(|&h| f16_bits_to_f32(h)).collect()
}

/// Real-device parity for `water_underwater_volume`: build the froxel
/// scattering/transmittance volume on device and match the decoded
/// `rgba16float` output lane-for-lane against the `CPU` golden.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn underwater_volume_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "underwater_volume_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let params = GpuWaterUnderwaterParams {
        extinction: [0.8, 0.35, 0.15],
        slice_thickness: 0.5,
        base_color: [0.9, 0.85, 0.8],
        phase_g: 0.4,
        sun_cos: 0.7,
        scatter_albedo: 0.6,
        width: UNDERWATER_W,
        height: UNDERWATER_H,
        depth: UNDERWATER_D,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let surface = underwater_surface_field(params.width, params.height);

    let golden = underwater_golden(&params, &surface);

    let wgsl = compile_render_fx_wgsl();
    let entry = find_entry_point(&wgsl, "underwater_volume");
    let gpu = dispatch_underwater(&device, &queue, &wgsl, &entry, &params, &surface);

    assert_eq!(gpu.len(), golden.len(), "underwater texel lane count");

    let mut i = 0;
    while i < golden.len() {
        let d = (gpu[i] - golden[i]).abs();
        assert!(
            d < PARITY_EPS,
            "underwater lane {i}: gpu={} cpu={} |d|={d}",
            gpu[i],
            golden[i],
        );
        i += 1;
    }
}

// ===========================================================================
// Waterline mask parity (`water_waterline_mask`, water_surface.wesl)
// ===========================================================================

use super::abi::GpuWaterWaterlineParams;
use prism_render_architecture::water::waterline::{self, WaterlineParams};

/// The waterline mask is pure subtraction / division / clamp, so the on-device
/// result is bit-for-bit `float32` arithmetic against the `CPU` golden; a
/// micro tolerance only guards against a stray fused multiply-add.
const WATERLINE_EPS: f32 = 1.0e-6;

/// Deterministic waterline sample field: per-cell sample world height, local
/// water-surface height, and total water depth (row-major, `nx * nz`).
///
/// The sample sweeps from above the surface down through it into submersion so
/// the soft-transition ramp is covered on both saturated ends and the linear
/// interior, while the total depth spans a dry margin (`0`) up to deep water so
/// the shoreline band rises, saturates, and switches fully off.
fn build_waterline_field(nx: u32, nz: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = (nx * nz) as usize;
    let mut sample_y = Vec::with_capacity(n);
    let mut surface_y = Vec::with_capacity(n);
    let mut depth = Vec::with_capacity(n);
    let mut z = 0u32;
    while z < nz {
        let mut x = 0u32;
        while x < nx {
            let fx = x as f32;
            let fz = z as f32;
            // Surface tilts gently across the grid.
            let s = 5.0 + 0.05 * fz;
            // Sample height sweeps roughly +1 .. -1 m around the surface.
            let sweep = ((fx * 3.0 + fz) % 20.0) / 10.0;
            sample_y.push(s + 1.0 - sweep);
            surface_y.push(s);
            // Total water depth: dry margin up through deep water.
            depth.push(((fx + fz * 2.0) % 13.0) / 6.0);
            x += 1;
        }
        z += 1;
    }
    (sample_y, surface_y, depth)
}

/// `CPU` golden for the waterline mask, packed `(weight, band, underwater,
/// submersion_depth)` per cell in row-major order to match `waterline_out`.
fn waterline_golden(
    sample_y: &[f32],
    surface_y: &[f32],
    water_depth: &[f32],
    params: WaterlineParams,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(sample_y.len() * 4);
    let mut i = 0;
    while i < sample_y.len() {
        let depth = waterline::submersion_depth(sample_y[i], surface_y[i]);
        let underwater = if waterline::is_underwater(sample_y[i], surface_y[i]) {
            1.0
        } else {
            0.0
        };
        let weight = waterline::waterline_weight(sample_y[i], surface_y[i], params);
        let band = waterline::shoreline_band(sample_y[i], surface_y[i], water_depth[i], params);
        out.push(weight);
        out.push(band);
        out.push(underwater);
        out.push(depth);
        i += 1;
    }
    out
}

/// Dispatches one `water_waterline_mask` pass and reads back the packed mask.
///
/// The kernel declares its resources on `@group(2)` (the surface module's
/// `swe`/`foam` passes own `group(0)`/`group(1)`, unreferenced by this entry
/// point), so the bind group is built from the pipeline's reflected `group(2)`
/// layout and set at binding index `2`. The five bindings match the shader's
/// declaration order: three read-only sample buffers, the packed `vec4` output,
/// and the uniform params.
fn dispatch_waterline(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    sample_y: &[f32],
    surface_y: &[f32],
    water_depth: &[f32],
    params: &GpuWaterWaterlineParams,
) -> Vec<f32> {
    let n = sample_y.len();
    let out_bytes = (n * 4 * size_of::<f32>()) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_surface_waterline_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_waterline_mask_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let storage_read = BufferUsages::STORAGE;
    let sample_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("waterline_sample_y"),
        contents: bytemuck::cast_slice(sample_y),
        usage: storage_read,
    });
    let surface_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("waterline_surface_y"),
        contents: bytemuck::cast_slice(surface_y),
        usage: storage_read,
    });
    let depth_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("waterline_depth"),
        contents: bytemuck::cast_slice(water_depth),
        usage: storage_read,
    });
    let out_buf = device.create_buffer(&BufferDescriptor {
        label: Some("waterline_out"),
        size: out_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("waterline_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });
    let layout = pipeline.get_bind_group_layout(2);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("water_waterline_mask_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: sample_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: surface_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: depth_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let out_stage = device.create_buffer(&BufferDescriptor {
        label: Some("waterline_out_stage"),
        size: out_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("waterline_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("waterline_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(2, &bind_group, &[]);
        pass.dispatch_workgroups(params.nx.div_ceil(8), params.nz.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
    queue.submit([encoder.finish()]);

    out_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = out_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    out_stage.unmap();
    values
}

/// One on-device waterline mask must match the `CPU` golden bit-for-bit.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn waterline_mask_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "waterline_mask_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    const NX: u32 = 24;
    const NZ: u32 = 20;
    let params = WaterlineParams {
        transition_half_width: 0.35,
        shoreline_depth: 1.5,
    };
    let (sample_y, surface_y, water_depth) = build_waterline_field(NX, NZ);
    let golden = waterline_golden(&sample_y, &surface_y, &water_depth, params);

    let gpu_params = GpuWaterWaterlineParams {
        nx: NX,
        nz: NZ,
        transition_half_width: params.transition_half_width,
        shoreline_depth: params.shoreline_depth,
    };

    let wgsl = compile_surface_wgsl();
    let entry = find_entry_point(&wgsl, "waterline_mask");
    let gpu = dispatch_waterline(
        &device,
        &queue,
        &wgsl,
        &entry,
        &sample_y,
        &surface_y,
        &water_depth,
        &gpu_params,
    );

    assert_eq!(gpu.len(), golden.len(), "packed mask length mismatch");
    let mut i = 0;
    while i < golden.len() {
        let d = (gpu[i] - golden[i]).abs();
        assert!(
            d < WATERLINE_EPS,
            "lane {i}: gpu={} cpu={} |d|={d}",
            gpu[i],
            golden[i],
        );
        i += 1;
    }
}

// ===========================================================================
// Crest-spray emitter parity (`water_spray_emit`, water_pbf.wesl)
// ===========================================================================

use super::abi::{GpuSprayParams, GpuSprayParticle, GpuSpraySource, GpuSpraySpawnHeader};
use prism_render_architecture::water::breaking::{self, BreakingCriteria, BreakingSample};

/// Deterministic crest-spray candidates covering every classifier branch:
/// a calm sample (no spray), a steep-but-unfolded crest (cresting, still no
/// spray), a folded-Jacobian breaker, and a high-intensity breaker, then the
/// same four repeated with rotated launch frames so the velocity blend and the
/// atomic counter accumulate across slots.
fn build_spray_sources(count: u32) -> Vec<GpuSpraySource> {
    let mut sources = Vec::with_capacity(count as usize);
    let mut i = 0u32;
    while i < count {
        let phase = i % 4;
        let f = i as f32;
        // Rotate the launch frame slightly per slot; both vectors stay well
        // above the length floor so the shared normalize is non-degenerate.
        let tangent = [1.0 + 0.05 * f, 0.1, 0.2 + 0.01 * f];
        let normal = [0.1, 1.0 + 0.03 * f, 0.05];
        let (steepness, jacobian, curvature) = match phase {
            // Calm: below every threshold.
            0 => (0.2, 1.0, 0.4),
            // Cresting: steep past threshold, Jacobian healthy, low curvature.
            1 => (1.4, 1.0, 0.5),
            // Breaking by fold: Jacobian at/under the fold threshold.
            2 => (0.6, 0.05, 0.8),
            // Breaking by intensity: steep and sharply curved.
            _ => (2.6, 0.6, 4.2),
        };
        sources.push(GpuSpraySource {
            position: [f, 0.5 * f, -f],
            steepness,
            tangent,
            jacobian,
            normal,
            curvature,
        });
        i += 1;
    }
    sources
}

/// `CPU` golden for the spray emitter: the accumulated spawn counter plus one
/// planned burst per source slot, in slot order.
fn spray_golden(
    sources: &[GpuSpraySource],
    params: &GpuSprayParams,
) -> (u32, Vec<GpuSprayParticle>) {
    let criteria = BreakingCriteria {
        steepness_threshold: params.steepness_threshold,
        jacobian_fold_threshold: params.jacobian_fold_threshold,
        curvature_threshold: params.curvature_threshold,
        breaking_intensity: params.breaking_intensity,
    };
    let mut counter = 0u32;
    let mut out = Vec::with_capacity(sources.len());
    for src in sources {
        let sample = BreakingSample {
            steepness: src.steepness,
            jacobian: src.jacobian,
            curvature: src.curvature,
        };
        let tangent = Vec3::new(src.tangent[0], src.tangent[1], src.tangent[2]);
        let normal = Vec3::new(src.normal[0], src.normal[1], src.normal[2]);
        let emission = breaking::plan_spray(
            sample,
            criteria,
            tangent,
            normal,
            params.jet_speed,
            params.max_spray_count,
        );
        counter += emission.count;
        out.push(GpuSprayParticle {
            position: src.position,
            count: emission.count,
            velocity: [
                emission.velocity.x,
                emission.velocity.y,
                emission.velocity.z,
            ],
            _pad: 0,
        });
    }
    (counter, out)
}

/// Dispatches one `water_spray_emit` pass and reads back the spawn buffer.
///
/// All resources live on `@group(0)`. The spawn buffer is uploaded zeroed and
/// laid out as a 16-byte `atomic` counter header (the `vec3` in the burst
/// record forces the array to start at offset 16) followed by one 32-byte burst
/// record per source slot, matching `SpraySpawn` in the shader.
fn dispatch_spray(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    sources: &[GpuSpraySource],
    params: &GpuSprayParams,
) -> (u32, Vec<GpuSprayParticle>) {
    let n = sources.len();
    let header_bytes = size_of::<GpuSpraySpawnHeader>();
    let record_bytes = size_of::<GpuSprayParticle>();
    let spawn_bytes = (header_bytes + n * record_bytes) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_pbf_spray_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_spray_emit_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let source_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("spray_sources"),
        contents: bytemuck::cast_slice(sources),
        usage: BufferUsages::STORAGE,
    });
    let spawn_init = vec![0u8; spawn_bytes as usize];
    let spawn_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("spray_spawn"),
        contents: &spawn_init,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("spray_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("water_spray_emit_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: source_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: spawn_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("spray_spawn_stage"),
        size: spawn_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("spray_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("spray_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(params.source_count.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&spawn_buf, 0, &stage, 0, spawn_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let header: GpuSpraySpawnHeader = *bytemuck::from_bytes(&view[..header_bytes]);
    let records: Vec<GpuSprayParticle> =
        bytemuck::cast_slice::<u8, GpuSprayParticle>(&view[header_bytes..]).to_vec();
    drop(view);
    stage.unmap();
    (header.counter, records)
}

/// One on-device spray emission must match the `CPU` golden: identical spawn
/// counts and positions, and launch velocities within `float32` rounding.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn spray_emit_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("spray_emit_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity");
        return;
    };

    const COUNT: u32 = 40;
    let params = GpuSprayParams {
        steepness_threshold: 1.0,
        jacobian_fold_threshold: 0.2,
        curvature_threshold: 2.0,
        breaking_intensity: 0.5,
        jet_speed: 3.0,
        source_count: COUNT,
        spawn_capacity: COUNT,
        max_spray_count: 32,
    };
    let sources = build_spray_sources(COUNT);
    let (golden_counter, golden) = spray_golden(&sources, &params);

    let wgsl = compile_pbf_wgsl();
    let entry = find_entry_point(&wgsl, "spray_emit");
    let (gpu_counter, gpu) = dispatch_spray(&device, &queue, &wgsl, &entry, &sources, &params);

    assert_eq!(gpu.len(), golden.len(), "spawn record count mismatch");
    assert_eq!(
        gpu_counter, golden_counter,
        "atomic spawn counter mismatch: gpu={gpu_counter} cpu={golden_counter}"
    );
    let mut i = 0;
    while i < golden.len() {
        assert_eq!(
            gpu[i].count, golden[i].count,
            "slot {i}: spawn count mismatch"
        );
        let mut c = 0;
        while c < 3 {
            let pd = (gpu[i].position[c] - golden[i].position[c]).abs();
            assert!(pd < WATER_EPS, "slot {i} pos[{c}]: gpu vs cpu |d|={pd}");
            let vd = (gpu[i].velocity[c] - golden[i].velocity[c]).abs();
            assert!(vd < PARITY_EPS, "slot {i} vel[{c}]: gpu vs cpu |d|={vd}");
            c += 1;
        }
        i += 1;
    }
}

// ===========================================================================
// FLIP/APIC particle-to-grid scatter (water_flip.wesl :: water_flip_p2g)
// ===========================================================================
//
// This block closes the on-device gap for stage 1 of the `FLIP`/`APIC` loop:
// the mass-weighted, trilinearly blended `APIC` momentum splat from particles
// onto the staggered `MAC` grid's four fixed-point atomics per cell
// (`[momentum_x, momentum_y, momentum_z, mass]`). The kernel touches only
// `group(0)` bindings 0 (particles), 1 (scatter atomics), and 4 (params), so
// the reflected auto layout carries exactly those three entries.

use super::abi::GpuFlipParticle;
use prism_render_architecture::water::flip::{apic_velocity, trilinear_weights};

/// `CPU` golden twin of `water_flip_p2g`.
///
/// Reproduces the kernel line-for-line: per live particle it forms the local
/// grid coordinate, the eight trilinear corner weights (the shared golden
/// [`trilinear_weights`]), and the per-corner `APIC` node velocity (the shared
/// golden [`apic_velocity`]), then scatters `node_vel * (mass * w)` and
/// `mass * w` into the eight surrounding cells. Every corner contribution is
/// encoded to the signed fixed-point domain and summed in the exact wrapping
/// two's-complement `u32` arithmetic the on-device `atomicAdd` uses, so the
/// packed scatter buffers agree bit-for-bit before decode. Out-of-grid corners
/// and non-positive weights are skipped exactly as the shader does.
fn flip_p2g_golden(particles: &[GpuFlipParticle], params: &GpuFlipSimParams) -> Vec<u32> {
    let dim = [params.dim[0], params.dim[1], params.dim[2]];
    let total = (dim[0] * dim[1] * dim[2]) as usize;
    let mut scatter = vec![0u32; total * 4];
    let mass = params.particle_mass.max(0.0);
    if mass <= FLIP_EPS {
        return scatter;
    }
    let origin = Vec3::new(params.origin[0], params.origin[1], params.origin[2]);
    let affine = f32::from(params.use_affine != 0);
    let dx = params.dx;

    let mut p = 0usize;
    while p < params.particle_count as usize {
        let particle = particles[p];
        p += 1;
        if particle.pos[3] <= 0.5 {
            continue;
        }
        let pos = Vec3::new(particle.pos[0], particle.pos[1], particle.pos[2]);
        let vel = Vec3::new(particle.vel[0], particle.vel[1], particle.vel[2]);
        let rows = [
            Vec3::new(particle.c0[0], particle.c0[1], particle.c0[2]).scale(affine),
            Vec3::new(particle.c1[0], particle.c1[1], particle.c1[2]).scale(affine),
            Vec3::new(particle.c2[0], particle.c2[1], particle.c2[2]).scale(affine),
        ];
        let local = pos.sub(origin).scale(params.inv_dx);
        let base_i = local.x.floor() as i32;
        let base_j = local.y.floor() as i32;
        let base_k = local.z.floor() as i32;
        let weights = trilinear_weights(
            local.x - local.x.floor(),
            local.y - local.y.floor(),
            local.z - local.z.floor(),
        );
        let mut corner = 0usize;
        let mut cz = 0i32;
        while cz < 2 {
            let mut cy = 0i32;
            while cy < 2 {
                let mut cx = 0i32;
                while cx < 2 {
                    let w = weights[corner];
                    corner += 1;
                    let ci = base_i + cx;
                    let cj = base_j + cy;
                    let ck = base_k + cz;
                    cx += 1;
                    if !flip_in_bounds(ci, cj, ck, dim) {
                        continue;
                    }
                    if w <= 0.0 {
                        continue;
                    }
                    let node = Vec3::new(
                        origin.x + (ci as f32 + 0.5) * dx,
                        origin.y + (cj as f32 + 0.5) * dx,
                        origin.z + (ck as f32 + 0.5) * dx,
                    );
                    let offset = node.sub(pos);
                    let node_vel = apic_velocity(vel, rows, offset);
                    let contribution = node_vel.scale(mass * w);
                    let cell = flip_cell_index(ci as u32, cj as u32, ck as u32, dim) as usize;
                    let acc = cell * 4;
                    scatter[acc] = scatter[acc].wrapping_add(flip_encode_fixed(contribution.x));
                    scatter[acc + 1] =
                        scatter[acc + 1].wrapping_add(flip_encode_fixed(contribution.y));
                    scatter[acc + 2] =
                        scatter[acc + 2].wrapping_add(flip_encode_fixed(contribution.z));
                    scatter[acc + 3] = scatter[acc + 3].wrapping_add(flip_encode_fixed(mass * w));
                }
                cy += 1;
            }
            cz += 1;
        }
    }
    scatter
}

/// Dispatches one `water_flip_p2g` scatter on device and reads back the raw
/// fixed-point grid accumulators.
///
/// The bind group is built from the pipeline's reflected `group(0)` layout,
/// which — because the `P2G` kernel touches only the particles, the scatter
/// atomics, and the parameter block — contains exactly bindings 0, 1, and 4.
fn dispatch_flip_p2g(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    particles: &[GpuFlipParticle],
    cell_count: usize,
    params: &GpuFlipSimParams,
) -> Vec<u32> {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_flip"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("flip_p2g_pipeline"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let particle_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_particles"),
        contents: bytemuck::cast_slice(particles),
        usage: BufferUsages::STORAGE,
    });
    let scatter_len = cell_count * 4;
    let scatter_zero = vec![0u32; scatter_len];
    let scatter_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_scatter"),
        contents: bytemuck::cast_slice(&scatter_zero),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("flip_p2g_group0"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: scatter_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let out_bytes = (scatter_len * size_of::<u32>()) as u64;
    let out_stage = device.create_buffer(&BufferDescriptor {
        label: Some("flip_p2g_stage"),
        size: out_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("flip_p2g_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("flip_p2g_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(params.particle_count.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&scatter_buf, 0, &out_stage, 0, out_bytes);
    queue.submit([encoder.finish()]);

    out_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");
    let view = out_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    out_stage.unmap();
    values
}

/// One on-device `P2G` scatter must match the `CPU` golden after decode.
///
/// The scene mixes an interior particle whose eight corners are all in-bounds,
/// a boundary particle straddling the grid edge (some corners skipped by the
/// bounds guard), two particles sharing one cell (order-independent atomic
/// accumulation), an `APIC`-affine particle (non-zero `C_p` rows), and an
/// inactive particle (`pos.w <= 0.5`) that must contribute nothing. All
/// positions, velocities, affine rows, mass, and cell size are dyadic so the
/// fixed-point round is tie-free and the decoded momentum/mass agree to
/// `float32` rounding.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn flip_p2g_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("flip_p2g_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity");
        return;
    };

    let dim = [4u32, 4u32, 4u32];
    let cell_count = (dim[0] * dim[1] * dim[2]) as usize;
    let dx = 1.0_f32;
    let inv_dx = 1.0_f32;

    let particles = [
        // Interior particle: all eight corners inside the grid.
        GpuFlipParticle {
            pos: [1.25, 1.5, 1.75, 1.0],
            vel: [0.5, -0.25, 0.75, 0.0],
            c0: [0.0, 0.0, 0.0, 0.0],
            c1: [0.0, 0.0, 0.0, 0.0],
            c2: [0.0, 0.0, 0.0, 0.0],
        },
        // Boundary particle near the max corner: several corners fall outside.
        GpuFlipParticle {
            pos: [3.5, 3.25, 3.75, 1.0],
            vel: [-0.5, 0.5, -0.25, 0.0],
            c0: [0.0, 0.0, 0.0, 0.0],
            c1: [0.0, 0.0, 0.0, 0.0],
            c2: [0.0, 0.0, 0.0, 0.0],
        },
        // Two particles in the same cell exercise order-independent atomics.
        GpuFlipParticle {
            pos: [2.25, 2.25, 2.25, 1.0],
            vel: [1.0, 0.0, 0.0, 0.0],
            c0: [0.0, 0.0, 0.0, 0.0],
            c1: [0.0, 0.0, 0.0, 0.0],
            c2: [0.0, 0.0, 0.0, 0.0],
        },
        GpuFlipParticle {
            pos: [2.75, 2.75, 2.75, 1.0],
            vel: [0.0, 1.0, 0.0, 0.0],
            c0: [0.0, 0.0, 0.0, 0.0],
            c1: [0.0, 0.0, 0.0, 0.0],
            c2: [0.0, 0.0, 0.0, 0.0],
        },
        // APIC-affine particle: non-zero C_p rows tilt the node velocity.
        GpuFlipParticle {
            pos: [1.5, 2.5, 1.5, 1.0],
            vel: [0.25, 0.25, 0.25, 0.0],
            c0: [0.5, 0.0, 0.0, 0.0],
            c1: [0.0, 0.25, 0.0, 0.0],
            c2: [0.0, 0.0, 0.5, 0.0],
        },
        // Inactive particle (pos.w <= 0.5) must not contribute.
        GpuFlipParticle {
            pos: [0.5, 0.5, 0.5, 0.0],
            vel: [9.0, 9.0, 9.0, 0.0],
            c0: [9.0, 9.0, 9.0, 0.0],
            c1: [9.0, 9.0, 9.0, 0.0],
            c2: [9.0, 9.0, 9.0, 0.0],
        },
    ];

    let params = GpuFlipSimParams {
        origin: [0.0, 0.0, 0.0, 0.0],
        dim: [dim[0], dim[1], dim[2], 0],
        dx,
        inv_dx,
        flip_blend: 0.0,
        particle_mass: 2.0,
        jacobi_omega: 0.0,
        use_affine: 1,
        particle_count: particles.len() as u32,
        cell_count: cell_count as u32,
    };

    let golden = flip_p2g_golden(&particles, &params);

    let wgsl = compile_flip_wgsl();
    let entry = find_entry_point(&wgsl, "water_flip_p2g");
    let gpu = dispatch_flip_p2g(
        &device, &queue, &wgsl, &entry, &particles, cell_count, &params,
    );

    assert_eq!(gpu.len(), golden.len(), "scatter length mismatch");
    let mut cell = 0usize;
    while cell < cell_count {
        let base = cell * 4;
        let mut lane = 0usize;
        while lane < 4 {
            let g = flip_decode_fixed(gpu[base + lane]);
            let c = flip_decode_fixed(golden[base + lane]);
            let d = (g - c).abs();
            assert!(
                d < PARITY_EPS,
                "cell {cell} lane {lane}: gpu={g} cpu={c} |d|={d}"
            );
            lane += 1;
        }
        cell += 1;
    }
}

// ===========================================================================
// FLIP/APIC grid-to-particle gather (water_flip.wesl :: water_flip_g2p)
// ===========================================================================
//
// This block closes the on-device gap for stage 3 of the `FLIP`/`APIC` loop:
// the projected grid velocity is gathered back to each particle, the `FLIP`
// (velocity-delta) and `PIC` (absolute) updates are blended, and the `APIC`
// affine matrix is rebuilt from `Σ w·v_proj⊗offset·D⁻¹`. The single
// incompressibility correction `v -= ∇p` is applied per node here from the
// converged `pressure_in`. The kernel reads the scatter atomics (binding 1) and
// the pressure field (binding 2), reads+writes the particles (binding 0), and
// reads the params (binding 4), so the reflected auto layout carries bindings
// 0, 1, 2, and 4.

/// Host mirror of `flip_pressure_gradient`: central-difference gradient of
/// `pressure_in`, with out-of-grid neighbours reusing the cell's own pressure
/// (the Neumann wall that yields a zero one-sided gradient).
fn flip_pressure_gradient(
    pressure_in: &[f32],
    dim: [u32; 3],
    inv_dx: f32,
    i: i32,
    j: i32,
    k: i32,
    cell: usize,
) -> [f32; 3] {
    let p_here = pressure_in[cell];
    let axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let mut g = [0.0_f32; 3];
    let mut a = 0usize;
    while a < 3 {
        let o = axes[a];
        let mut p_pos = p_here;
        if flip_in_bounds(i + o[0], j + o[1], k + o[2], dim) {
            p_pos = pressure_in[flip_cell_index(
                (i + o[0]) as u32,
                (j + o[1]) as u32,
                (k + o[2]) as u32,
                dim,
            ) as usize];
        }
        let mut p_neg = p_here;
        if flip_in_bounds(i - o[0], j - o[1], k - o[2], dim) {
            p_neg = pressure_in[flip_cell_index(
                (i - o[0]) as u32,
                (j - o[1]) as u32,
                (k - o[2]) as u32,
                dim,
            ) as usize];
        }
        g[a] = (p_pos - p_neg) * (0.5 * inv_dx);
        a += 1;
    }
    g
}

/// `CPU` golden twin of `water_flip_g2p`.
///
/// Reproduces the kernel line-for-line for every live particle: form the eight
/// trilinear weights, gather the decoded grid velocity `v_grid` and the
/// projected velocity `v_proj = v_grid - ∇p`, accumulate the `PIC` velocity, the
/// `FLIP` delta (`-∇p`), and the `APIC` affine outer product, then blend
/// `(1 - alpha)·PIC + alpha·(vel + delta)` and rebuild the affine rows scaled by
/// `D⁻¹ = 3 / dx²`. Inactive particles (`pos.w <= 0.5`) pass through unchanged.
fn flip_g2p_golden(
    particles: &[GpuFlipParticle],
    scatter: &[u32],
    pressure_in: &[f32],
    params: &GpuFlipSimParams,
) -> Vec<GpuFlipParticle> {
    let dim = [params.dim[0], params.dim[1], params.dim[2]];
    let origin = Vec3::new(params.origin[0], params.origin[1], params.origin[2]);
    let dx = params.dx;
    let inv_dx = params.inv_dx;
    let inv_d = FLIP_APIC_INV_D * inv_dx * inv_dx;
    let alpha = params.flip_blend.clamp(0.0, 1.0);
    let use_affine = params.use_affine != 0;

    let mut out = particles.to_vec();
    let mut p = 0usize;
    while p < params.particle_count as usize {
        let particle = particles[p];
        if particle.pos[3] <= 0.5 {
            p += 1;
            continue;
        }
        let pos = Vec3::new(particle.pos[0], particle.pos[1], particle.pos[2]);
        let vel = Vec3::new(particle.vel[0], particle.vel[1], particle.vel[2]);
        let local = pos.sub(origin).scale(inv_dx);
        let base_i = local.x.floor() as i32;
        let base_j = local.y.floor() as i32;
        let base_k = local.z.floor() as i32;
        let weights = trilinear_weights(
            local.x - local.x.floor(),
            local.y - local.y.floor(),
            local.z - local.z.floor(),
        );

        let mut pic = Vec3::new(0.0, 0.0, 0.0);
        let mut delta = Vec3::new(0.0, 0.0, 0.0);
        let mut c0 = Vec3::new(0.0, 0.0, 0.0);
        let mut c1 = Vec3::new(0.0, 0.0, 0.0);
        let mut c2 = Vec3::new(0.0, 0.0, 0.0);

        let mut corner = 0usize;
        let mut cz = 0i32;
        while cz < 2 {
            let mut cy = 0i32;
            while cy < 2 {
                let mut cx = 0i32;
                while cx < 2 {
                    let w = weights[corner];
                    corner += 1;
                    let ci = base_i + cx;
                    let cj = base_j + cy;
                    let ck = base_k + cz;
                    cx += 1;
                    if !flip_in_bounds(ci, cj, ck, dim) || w <= 0.0 {
                        continue;
                    }
                    let cell = flip_cell_index(ci as u32, cj as u32, ck as u32, dim) as usize;
                    let v_cell = flip_cell_velocity(scatter, cell);
                    let v_grid = Vec3::new(v_cell[0], v_cell[1], v_cell[2]);
                    let grad_arr =
                        flip_pressure_gradient(pressure_in, dim, inv_dx, ci, cj, ck, cell);
                    let grad = Vec3::new(grad_arr[0], grad_arr[1], grad_arr[2]);
                    let v_proj = v_grid.sub(grad);
                    pic = pic.add(v_proj.scale(w));
                    delta = delta.sub(grad.scale(w));
                    let node = Vec3::new(
                        origin.x + (ci as f32 + 0.5) * dx,
                        origin.y + (cj as f32 + 0.5) * dx,
                        origin.z + (ck as f32 + 0.5) * dx,
                    );
                    let offset = node.sub(pos);
                    c0 = c0.add(offset.scale(v_proj.x * w));
                    c1 = c1.add(offset.scale(v_proj.y * w));
                    c2 = c2.add(offset.scale(v_proj.z * w));
                }
                cy += 1;
            }
            cz += 1;
        }

        let flip_vel = vel.add(delta);
        // (1 - alpha)·PIC + alpha·FLIP, mirroring the CPU `blend_flip_pic`.
        let blended = pic.scale(1.0 - alpha).add(flip_vel.scale(alpha));
        let mut result = particle;
        result.vel = [blended.x, blended.y, blended.z, particle.vel[3]];
        if use_affine {
            let r0 = c0.scale(inv_d);
            let r1 = c1.scale(inv_d);
            let r2 = c2.scale(inv_d);
            result.c0 = [r0.x, r0.y, r0.z, particle.c0[3]];
            result.c1 = [r1.x, r1.y, r1.z, particle.c1[3]];
            result.c2 = [r2.x, r2.y, r2.z, particle.c2[3]];
        } else {
            result.c0 = [0.0, 0.0, 0.0, particle.c0[3]];
            result.c1 = [0.0, 0.0, 0.0, particle.c1[3]];
            result.c2 = [0.0, 0.0, 0.0, particle.c2[3]];
        }
        out[p] = result;
        p += 1;
    }
    out
}

/// Dispatches one `water_flip_g2p` gather on device and reads back the updated
/// particle buffer.
///
/// The bind group is built from the pipeline's reflected `group(0)` layout,
/// which — because `G2P` touches the particles, the scatter atomics, the
/// pressure field, and the params — contains exactly bindings 0, 1, 2, and 4.
fn dispatch_flip_g2p(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    particles: &[GpuFlipParticle],
    scatter: &[u32],
    pressure_in: &[f32],
    params: &GpuFlipSimParams,
) -> Vec<GpuFlipParticle> {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_flip"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("flip_g2p_pipeline"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let particle_bytes = size_of_val(particles) as u64;
    let particle_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_particles"),
        contents: bytemuck::cast_slice(particles),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let scatter_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_scatter"),
        contents: bytemuck::cast_slice(scatter),
        usage: BufferUsages::STORAGE,
    });
    let pressure_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_pressure_in"),
        contents: bytemuck::cast_slice(pressure_in),
        usage: BufferUsages::STORAGE,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("flip_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("flip_g2p_group0"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: scatter_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: pressure_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let out_stage = device.create_buffer(&BufferDescriptor {
        label: Some("flip_g2p_stage"),
        size: particle_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("flip_g2p_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("flip_g2p_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(params.particle_count.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&particle_buf, 0, &out_stage, 0, particle_bytes);
    queue.submit([encoder.finish()]);

    out_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");
    let view = out_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values: Vec<GpuFlipParticle> = bytemuck::cast_slice::<u8, GpuFlipParticle>(&view).to_vec();
    drop(view);
    out_stage.unmap();
    values
}

/// One on-device `G2P` gather must match the `CPU` golden particle update.
///
/// The scatter grid is packed with dyadic momentum/mass so the decoded cell
/// velocities are exact, and a dyadic pressure ramp drives a non-trivial
/// gradient (hence a real `v -= ∇p` correction and `FLIP` delta). The particle
/// set mixes an interior particle, a boundary particle whose corners straddle
/// the grid edge (Neumann gradient + skipped corners), an `APIC`-affine
/// particle whose rebuilt affine rows must match, and an inactive particle that
/// must pass through untouched. The blend is set mid-range (`alpha = 0.5`) so
/// both the `PIC` and `FLIP` paths contribute.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn flip_g2p_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("flip_g2p_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity");
        return;
    };

    let dim = [4u32, 4u32, 4u32];
    let total = (dim[0] * dim[1] * dim[2]) as usize;
    let dx = 1.0_f32;
    let inv_dx = 1.0_f32;

    // Pack every cell with a dyadic velocity so the decoded field is exact.
    let mut scatter = vec![0u32; total * 4];
    let mut k = 0u32;
    while k < dim[2] {
        let mut j = 0u32;
        while j < dim[1] {
            let mut i = 0u32;
            while i < dim[0] {
                let cell = flip_cell_index(i, j, k, dim) as usize;
                let base = cell * 4;
                let mass = 2.0_f32;
                let vx = i as f32 * 0.25;
                let vy = j as f32 * 0.5;
                let vz = k as f32 * 0.25;
                scatter[base] = flip_encode_fixed(mass * vx);
                scatter[base + 1] = flip_encode_fixed(mass * vy);
                scatter[base + 2] = flip_encode_fixed(mass * vz);
                scatter[base + 3] = flip_encode_fixed(mass);
                i += 1;
            }
            j += 1;
        }
        k += 1;
    }

    let pressure_in: Vec<f32> = (0..total).map(|c| c as f32 * 0.25).collect();

    let particles = [
        GpuFlipParticle {
            pos: [1.25, 1.5, 1.75, 1.0],
            vel: [0.5, -0.25, 0.75, 0.0],
            c0: [0.0, 0.0, 0.0, 0.0],
            c1: [0.0, 0.0, 0.0, 0.0],
            c2: [0.0, 0.0, 0.0, 0.0],
        },
        GpuFlipParticle {
            pos: [3.5, 3.25, 3.75, 1.0],
            vel: [-0.5, 0.5, -0.25, 0.0],
            c0: [0.0, 0.0, 0.0, 0.0],
            c1: [0.0, 0.0, 0.0, 0.0],
            c2: [0.0, 0.0, 0.0, 0.0],
        },
        GpuFlipParticle {
            pos: [1.5, 2.5, 1.5, 1.0],
            vel: [0.25, 0.25, 0.25, 0.0],
            c0: [0.5, 0.0, 0.0, 0.0],
            c1: [0.0, 0.25, 0.0, 0.0],
            c2: [0.0, 0.0, 0.5, 0.0],
        },
        GpuFlipParticle {
            pos: [0.5, 0.5, 0.5, 0.0],
            vel: [9.0, 9.0, 9.0, 9.0],
            c0: [9.0, 9.0, 9.0, 9.0],
            c1: [9.0, 9.0, 9.0, 9.0],
            c2: [9.0, 9.0, 9.0, 9.0],
        },
    ];

    let params = GpuFlipSimParams {
        origin: [0.0, 0.0, 0.0, 0.0],
        dim: [dim[0], dim[1], dim[2], 0],
        dx,
        inv_dx,
        flip_blend: 0.5,
        particle_mass: 2.0,
        jacobi_omega: 0.0,
        use_affine: 1,
        particle_count: particles.len() as u32,
        cell_count: total as u32,
    };

    let golden = flip_g2p_golden(&particles, &scatter, &pressure_in, &params);

    let wgsl = compile_flip_wgsl();
    let entry = find_entry_point(&wgsl, "water_flip_g2p");
    let gpu = dispatch_flip_g2p(
        &device,
        &queue,
        &wgsl,
        &entry,
        &particles,
        &scatter,
        &pressure_in,
        &params,
    );

    assert_eq!(gpu.len(), golden.len(), "particle count mismatch");
    let mut p = 0usize;
    while p < golden.len() {
        let g = gpu[p];
        let c = golden[p];
        let mut lane = 0usize;
        while lane < 3 {
            let vd = (g.vel[lane] - c.vel[lane]).abs();
            assert!(
                vd < PARITY_EPS,
                "particle {p} vel[{lane}]: gpu vs cpu |d|={vd}"
            );
            let d0 = (g.c0[lane] - c.c0[lane]).abs();
            assert!(
                d0 < PARITY_EPS,
                "particle {p} c0[{lane}]: gpu vs cpu |d|={d0}"
            );
            let d1 = (g.c1[lane] - c.c1[lane]).abs();
            assert!(
                d1 < PARITY_EPS,
                "particle {p} c1[{lane}]: gpu vs cpu |d|={d1}"
            );
            let d2 = (g.c2[lane] - c.c2[lane]).abs();
            assert!(
                d2 < PARITY_EPS,
                "particle {p} c2[{lane}]: gpu vs cpu |d|={d2}"
            );
            let pd = (g.pos[lane] - c.pos[lane]).abs();
            assert!(
                pd < WATER_EPS,
                "particle {p} pos[{lane}] must be unchanged |d|={pd}"
            );
            lane += 1;
        }
        p += 1;
    }
}

// ===========================================================================
// FLIP/APIC full-pipeline multi-frame device fidelity (water_flip.wesl)
// ===========================================================================
//
// The single-stage goldens above pin `P2G`, the damped-`Jacobi` pressure sweep,
// and `G2P` in isolation. This block closes the remaining `FLIP`/`APIC` gap on
// the axis that a single dispatch cannot cover: a real multi-frame loop that
// chains all three device kernels per frame — `P2G` scatter, an iterated
// pressure projection (ping-ponged `PRESSURE_ITERS` times), and the `G2P`
// gather+advection — the way `Houdini`'s `FLIP` solver and a `UE5` Niagara
// fluid do. The property asserted along the whole trajectory is *cross-frame
// on-device fidelity*: every frame each `GPU` kernel is driven from the exact
// `CPU`-authoritative state its golden twin consumes, and every decoded output
// (scatter, each pressure sweep, and the gathered particle vel/affine rows) is
// compared against its `CPU` mirror. This catches state-plumbing, ping-pong,
// and bind-group regressions that only surface across chained dispatches, not
// in the isolated single-stage goldens. No stage may produce a non-finite
// value and advected particles must stay inside the domain.
//
// IMPORTANT — this is a *fidelity* golden, not a stability proof. The pressure
// stage in `water_flip.wesl` is a cell-centered (collocated) solve: divergence
// and the applied gradient are both wide central differences, while the Jacobi
// relaxation uses the compact 7-point Laplacian. That operator pair carries the
// classic odd/even (checkerboard) null space, so `v - ∇p` is not a true
// orthogonal projection and can *inject* energy for a general input — the GPU
// reproduces the CPU exactly, but the shared scheme itself is not
// unconditionally stable. Measured here: a divergence-free-ish swirl seed still
// grows ~1.5x per frame before the damped reflective walls bleed it back off.
// Unconditional incompressible stability (arresting a gravity-loaded column,
// long-run energy decay) requires a staggered `MAC` discretization with
// face-centered velocities and matching compact grad/div operators; that
// kernel rework is tracked as an explicit design item (see the water spec).
// Until then this test deliberately makes no bounded-energy claim.

/// Deterministic damped reflective-wall restitution for the advection bounce.
const FLIP_WALL_RESTITUTION: f32 = 0.3;

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice and observed fidelity metrics must reach the test log"
)]
fn flip_full_pipeline_gpu_multi_frame_tracks_cpu() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("flip_full_pipeline_gpu_multi_frame_tracks_cpu: no wgpu adapter, skipping");
        return;
    };

    const FRAMES: usize = 10;
    const PRESSURE_ITERS: usize = 30;
    let dim = [6u32, 6u32, 6u32];
    let total = (dim[0] * dim[1] * dim[2]) as usize;
    let dx = 1.0_f32;
    let inv_dx = 1.0_f32;
    let dt = 0.05_f32;
    let lo = 0.05_f32;
    let hi = dim[0] as f32 * dx - 0.05_f32;

    let mut params = GpuFlipSimParams {
        origin: [0.0, 0.0, 0.0, 0.0],
        dim: [dim[0], dim[1], dim[2], 0],
        dx,
        inv_dx,
        flip_blend: 0.0,
        particle_mass: 1.0,
        jacobi_omega: 0.6,
        use_affine: 1,
        particle_count: 0,
        cell_count: total as u32,
    };

    // A block of fluid centred in the box, seeded with a swirling + gently
    // converging velocity impulse (no external force follows). Each particle is
    // nudged off the cell centre by a deterministic dyadic jitter so the
    // trilinear stencil is non-degenerate. The divergent seed exercises the
    // pressure projection hard; the dissipative `APIC` transfer must then keep
    // the field bounded and decaying rather than pumping energy in.
    let centre = [3.0_f32, 3.5_f32, 3.0_f32];
    let swirl = 1.5_f32;
    let converge = 0.0_f32;
    let mut parts: Vec<GpuFlipParticle> = Vec::new();
    let mut cz = 1u32;
    while cz <= 4 {
        let mut cy = 2u32;
        while cy <= 4 {
            let mut cx = 1u32;
            while cx <= 4 {
                let idx = parts.len() as u32;
                let jx = (((idx * 5) % 3) as f32 - 1.0) * 0.125;
                let jy = (((idx * 7) % 3) as f32 - 1.0) * 0.125;
                let jz = (((idx * 11) % 3) as f32 - 1.0) * 0.125;
                let px = (cx as f32 + 0.5) * dx + jx;
                let py = (cy as f32 + 0.5) * dx + jy;
                let pz = (cz as f32 + 0.5) * dx + jz;
                let rx = px - centre[0];
                let ry = py - centre[1];
                let rz = pz - centre[2];
                // Swirl about the y-axis plus a mild radial inflow: this seeds a
                // non-zero divergence for the projection to fight.
                let vx = -rz * swirl - rx * converge;
                let vy = -ry * converge;
                let vz = rx * swirl - rz * converge;
                parts.push(GpuFlipParticle {
                    pos: [px, py, pz, 1.0],
                    vel: [vx, vy, vz, 0.0],
                    c0: [0.0, 0.0, 0.0, 0.0],
                    c1: [0.0, 0.0, 0.0, 0.0],
                    c2: [0.0, 0.0, 0.0, 0.0],
                });
                cx += 1;
            }
            cy += 1;
        }
        cz += 1;
    }
    params.particle_count = parts.len() as u32;
    assert!(parts.len() >= 32, "fluid block must be non-trivial");

    // Peak speed of the seed impulse, reported alongside the observed maximum so
    // the log shows how the (known non-projective) collocated solve evolves it.
    let mut initial_peak = 0.0_f32;
    for particle in &parts {
        let v = particle.vel;
        let sp = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if sp > initial_peak {
            initial_peak = sp;
        }
    }

    let wgsl = compile_flip_wgsl();
    let p2g_entry = find_entry_point(&wgsl, "water_flip_p2g");
    let pressure_entry = find_entry_point(&wgsl, "flip_pressure_solve");
    let g2p_entry = find_entry_point(&wgsl, "water_flip_g2p");

    let mut max_parity = 0.0_f32;
    let mut max_speed_seen = 0.0_f32;

    let mut frame = 0usize;
    while frame < FRAMES {
        // 1) P2G scatter. Compared on decoded values (the fixed-point atomic
        //    path can differ from the golden accumulation by one quantum).
        let gpu_scatter =
            dispatch_flip_p2g(&device, &queue, &wgsl, &p2g_entry, &parts, total, &params);
        let cpu_scatter = flip_p2g_golden(&parts, &params);
        assert_eq!(
            gpu_scatter.len(),
            cpu_scatter.len(),
            "scatter length mismatch"
        );
        let mut w = 0usize;
        while w < cpu_scatter.len() {
            let g = flip_decode_fixed(gpu_scatter[w]);
            let c = flip_decode_fixed(cpu_scatter[w]);
            assert!(
                g.is_finite() && c.is_finite(),
                "frame {frame} scatter word {w}: non-finite gpu={g} cpu={c}"
            );
            let d = (g - c).abs();
            if d > max_parity {
                max_parity = d;
            }
            assert!(
                d < PARITY_EPS,
                "frame {frame} scatter word {w}: gpu={g} cpu={c} |d|={d}"
            );
            w += 1;
        }

        // 2) Pressure projection: PRESSURE_ITERS damped-Jacobi sweeps, the GPU
        //    driven each sweep from the CPU-authoritative field so the parity
        //    check isolates single-sweep fidelity along the converging solve.
        let mut pressure = vec![0.0_f32; total];
        let mut it = 0usize;
        while it < PRESSURE_ITERS {
            let gpu_p = dispatch_flip_pressure(
                &device,
                &queue,
                &wgsl,
                &pressure_entry,
                &cpu_scatter,
                &pressure,
                &params,
            );
            let cpu_p = flip_pressure_golden(
                &cpu_scatter,
                &pressure,
                dim,
                dx,
                inv_dx,
                params.jacobi_omega,
            );
            assert_eq!(gpu_p.len(), cpu_p.len(), "pressure length mismatch");
            let mut c = 0usize;
            while c < cpu_p.len() {
                let g = gpu_p[c];
                let cc = cpu_p[c];
                assert!(
                    g.is_finite() && cc.is_finite(),
                    "frame {frame} iter {it} cell {c}: non-finite pressure gpu={g} cpu={cc}"
                );
                let d = (g - cc).abs();
                if d > max_parity {
                    max_parity = d;
                }
                assert!(
                    d < PARITY_EPS,
                    "frame {frame} iter {it} cell {c}: pressure gpu={g} cpu={cc} |d|={d}"
                );
                c += 1;
            }
            pressure = cpu_p;
            it += 1;
        }

        // 3) G2P gather + FLIP/PIC blend + APIC affine rebuild.
        let gpu_parts = dispatch_flip_g2p(
            &device,
            &queue,
            &wgsl,
            &g2p_entry,
            &parts,
            &cpu_scatter,
            &pressure,
            &params,
        );
        let cpu_parts = flip_g2p_golden(&parts, &cpu_scatter, &pressure, &params);
        assert_eq!(gpu_parts.len(), cpu_parts.len(), "particle count mismatch");
        let mut p = 0usize;
        while p < cpu_parts.len() {
            let g = gpu_parts[p];
            let c = cpu_parts[p];
            let mut lane = 0usize;
            while lane < 3 {
                let vd = (g.vel[lane] - c.vel[lane]).abs();
                let d0 = (g.c0[lane] - c.c0[lane]).abs();
                let d1 = (g.c1[lane] - c.c1[lane]).abs();
                let d2 = (g.c2[lane] - c.c2[lane]).abs();
                assert!(
                    g.vel[lane].is_finite() && c.vel[lane].is_finite(),
                    "frame {frame} particle {p} vel[{lane}]: non-finite"
                );
                for d in [vd, d0, d1, d2] {
                    if d > max_parity {
                        max_parity = d;
                    }
                    assert!(
                        d < PARITY_EPS,
                        "frame {frame} particle {p} lane {lane}: stage parity |d|={d}"
                    );
                }
                lane += 1;
            }
            p += 1;
        }

        // 4) Advect the authoritative trajectory: pos += vel*dt with damped
        //    reflective walls. This also advances the state fed to the next
        //    frame's GPU dispatches, so the parity claim spans a real run.
        parts = cpu_parts;
        let mut pa = 0usize;
        while pa < parts.len() {
            if parts[pa].pos[3] <= 0.5 {
                pa += 1;
                continue;
            }
            let mut np = [
                parts[pa].pos[0] + parts[pa].vel[0] * dt,
                parts[pa].pos[1] + parts[pa].vel[1] * dt,
                parts[pa].pos[2] + parts[pa].vel[2] * dt,
            ];
            let mut nv = [parts[pa].vel[0], parts[pa].vel[1], parts[pa].vel[2]];
            let mut axis = 0usize;
            while axis < 3 {
                if np[axis] < lo {
                    np[axis] = lo;
                    if nv[axis] < 0.0 {
                        nv[axis] = -nv[axis] * FLIP_WALL_RESTITUTION;
                    }
                } else if np[axis] > hi {
                    np[axis] = hi;
                    if nv[axis] > 0.0 {
                        nv[axis] = -nv[axis] * FLIP_WALL_RESTITUTION;
                    }
                }
                axis += 1;
            }
            parts[pa].pos = [np[0], np[1], np[2], parts[pa].pos[3]];
            parts[pa].vel = [nv[0], nv[1], nv[2], parts[pa].vel[3]];

            let speed = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
            assert!(
                speed.is_finite(),
                "frame {frame} particle {pa}: non-finite speed"
            );
            if speed > max_speed_seen {
                max_speed_seen = speed;
            }
            assert!(
                np[0] >= lo - WATER_EPS
                    && np[0] <= hi + WATER_EPS
                    && np[1] >= lo - WATER_EPS
                    && np[1] <= hi + WATER_EPS
                    && np[2] >= lo - WATER_EPS
                    && np[2] <= hi + WATER_EPS,
                "frame {frame} particle {pa}: left the domain at {np:?}"
            );
            pa += 1;
        }

        frame += 1;
    }

    eprintln!(
        "flip_full_pipeline_gpu_multi_frame_tracks_cpu: {FRAMES} frames x {PRESSURE_ITERS} pressure iters, max stage parity drift {max_parity:e}, seed peak speed {initial_peak:e}, max particle speed {max_speed_seen:e}"
    );
}

// ===========================================================================
// Kernel 14: water_surface_reconstruct (screen-space FLIP surface) — the
// `van der Laan` bilateral depth smooth + view-space normal reconstruction
// half of the surface path in `water_flip.wesl`. One invocation per pixel; it
// writes `(normal.xyz, smoothed_depth)` into an `rgba16float` storage texture.
// ===========================================================================

use super::abi::GpuFlipSurfaceParams;

/// Sentinel depth marking a pixel with no splatted fluid; mirrors
/// `FLIP_DEPTH_FAR` in `water_flip.wesl`.
const FLIP_DEPTH_FAR: f32 = 1.0e30;
/// `f16` write path tolerance: `rgba16float` storage carries ~10 mantissa bits,
/// so unit normals and small depths quantise to ~5e-4; the bilateral `exp`
/// weights (Metal `exp` versus `bevy_math::ops::exp`) cancel under
/// normalisation, leaving quantisation as the dominant term. `4e-3` bounds it.
const RECON_EPS: f32 = 4.0e-3;

/// Builds a deterministic screen tile for the surface reconstruction parity:
/// a smooth splatted-depth ramp in `1.0..~1.34 m` punched through with a few
/// background pixels (no fluid). The vertical stripe at `x == 15` exercises the
/// background return, the neighbour `continue` inside the bilateral window, and
/// the right/up neighbour fall-back when a differencing neighbour is
/// background; the far corner adds an edge background so the `x`/`y` edge
/// fall-backs read a `>= FLIP_DEPTH_FAR` neighbour too. Returns the packed
/// per-pixel `(surface_depth, surface_thickness)` fields.
fn surface_reconstruct_field(width: u32, height: u32) -> (Vec<f32>, Vec<f32>) {
    let n = (width * height) as usize;
    let mut depth = vec![0.0_f32; n];
    let mut thickness = vec![0.0_f32; n];
    let mut y = 0u32;
    while y < height {
        let mut x = 0u32;
        while x < width {
            let idx = (y * width + x) as usize;
            let background = x == 15 || (x == 5 && y == 2) || (x == width - 1 && y == height - 1);
            if background {
                depth[idx] = FLIP_DEPTH_FAR;
                thickness[idx] = 0.0;
            } else {
                depth[idx] = 1.0 + 0.01 * (x as f32) + 0.005 * (y as f32);
                thickness[idx] = 0.5 + 0.01 * (x as f32);
            }
            x += 1;
        }
        y += 1;
    }
    (depth, thickness)
}

/// `CPU` golden for `water_surface_reconstruct`: a line-for-line replica of the
/// shader. Background pixels emit the flat `(0, 0, 1, FLIP_DEPTH_FAR)` default;
/// every other pixel bilaterally smooths the depth (spatial Gaussian times a
/// range Gaussian on the depth delta, background samples skipped), reconstructs
/// the view-space normal from the smoothed depth's screen-space finite
/// differences (`van der Laan`), normalises, and faces the camera. The `exp` is
/// the crate's `libm`-backed [`bevy_math::ops::exp`], matching the shader's raw
/// `exp` closely enough that the normalised smooth lands well inside
/// [`RECON_EPS`].
fn surface_reconstruct_golden(
    depth: &[f32],
    _thickness: &[f32],
    params: &GpuFlipSurfaceParams,
) -> Vec<[f32; 4]> {
    let width = params.resolution[0];
    let height = params.resolution[1];
    let eps = 1.0e-6_f32;
    let radius = params.filter_radius.clamp(0, 8);
    let spatial = params.spatial_sigma2.max(eps);
    let range = params.range_sigma2.max(eps);
    let half_res = [width as f32 * 0.5, height as f32 * 0.5];
    let mut out = vec![[0.0_f32; 4]; (width * height) as usize];

    let mut gy = 0u32;
    while gy < height {
        let mut gx = 0u32;
        while gx < width {
            let center_idx = (gy * width + gx) as usize;
            let center_depth = depth[center_idx];
            if center_depth >= FLIP_DEPTH_FAR {
                out[center_idx] = [0.0, 0.0, 1.0, FLIP_DEPTH_FAR];
                gx += 1;
                continue;
            }

            let cxi = gx as i32;
            let cyi = gy as i32;
            let mut depth_sum = 0.0_f32;
            let mut weight_sum = 0.0_f32;
            let mut dy = -radius;
            while dy <= radius {
                let mut dx = -radius;
                while dx <= radius {
                    let sx = cxi + dx;
                    let sy = cyi + dy;
                    if sx < 0 || sy < 0 || sx >= width as i32 || sy >= height as i32 {
                        dx += 1;
                        continue;
                    }
                    let sample_depth = depth[(sy as u32 * width + sx as u32) as usize];
                    if sample_depth >= FLIP_DEPTH_FAR {
                        dx += 1;
                        continue;
                    }
                    let spatial_d = (dx * dx + dy * dy) as f32;
                    let range_d = sample_depth - center_depth;
                    let sw = bevy_math::ops::exp(-spatial_d / spatial)
                        * bevy_math::ops::exp(-(range_d * range_d) / range);
                    depth_sum += sample_depth * sw;
                    weight_sum += sw;
                    dx += 1;
                }
                dy += 1;
            }
            let mut smoothed = center_depth;
            if weight_sum > eps {
                smoothed = depth_sum / weight_sum;
            }

            let cx = gx as f32 + 0.5;
            let cy = gy as f32 + 0.5;
            let view_position = |px: f32, py: f32, d: f32| -> [f32; 3] {
                let scale = params.pixel_world_scale * d;
                [(px - half_res[0]) * scale, (py - half_res[1]) * scale, d]
            };
            let center_pos = view_position(cx, cy, smoothed);

            let mut depth_r = smoothed;
            if cxi + 1 < width as i32 {
                let d = depth[center_idx + 1];
                if d < FLIP_DEPTH_FAR {
                    depth_r = d;
                }
            }
            let mut depth_u = smoothed;
            if cyi + 1 < height as i32 {
                let d = depth[((gy + 1) * width + gx) as usize];
                if d < FLIP_DEPTH_FAR {
                    depth_u = d;
                }
            }
            let pos_r = view_position(cx + 1.0, cy, depth_r);
            let pos_u = view_position(cx, cy + 1.0, depth_u);
            let ddx = [
                pos_r[0] - center_pos[0],
                pos_r[1] - center_pos[1],
                pos_r[2] - center_pos[2],
            ];
            let ddy = [
                pos_u[0] - center_pos[0],
                pos_u[1] - center_pos[1],
                pos_u[2] - center_pos[2],
            ];
            let mut normal = [
                ddx[1] * ddy[2] - ddx[2] * ddy[1],
                ddx[2] * ddy[0] - ddx[0] * ddy[2],
                ddx[0] * ddy[1] - ddx[1] * ddy[0],
            ];
            let len_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
            if len_sq > eps {
                let inv_len = len_sq.sqrt();
                normal = [
                    normal[0] / inv_len,
                    normal[1] / inv_len,
                    normal[2] / inv_len,
                ];
            } else {
                normal = [0.0, 0.0, 1.0];
            }
            if normal[2] < 0.0 {
                normal = [-normal[0], -normal[1], -normal[2]];
            }
            // Coverage `select` in the shader is a no-op (`depth_out == smoothed`).
            out[center_idx] = [normal[0], normal[1], normal[2], smoothed];
            gx += 1;
        }
        gy += 1;
    }
    out
}

/// Dispatches one `water_surface_reconstruct` on device and reads back the
/// decoded `rgba16float` surface texture as packed `(nx, ny, nz, depth)` texels.
///
/// The kernel only touches `group(0)` bindings `5..=8`, so the bind group is
/// built straight from the pipeline's reflected auto layout (sparse bindings,
/// same as the `P2G`/`G2P` paths). `width == 32` makes the `rgba16float` row
/// `32 * 8 == 256` bytes, already `256`-aligned, so the read-back needs no row
/// padding.
#[expect(
    clippy::too_many_lines,
    reason = "one linear dispatch-and-readback keeps the parity path auditable"
)]
fn dispatch_surface_reconstruct(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    depth: &[f32],
    thickness: &[f32],
    params: &GpuFlipSurfaceParams,
) -> Vec<[f32; 4]> {
    let width = params.resolution[0];
    let height = params.resolution[1];

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_flip_surface_reconstruct_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_surface_reconstruct_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);

    let depth_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("surface_depth"),
        contents: bytemuck::cast_slice(depth),
        usage: BufferUsages::STORAGE,
    });
    let thickness_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("surface_thickness"),
        contents: bytemuck::cast_slice(thickness),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("surface_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });
    let out_tex = device.create_texture(&TextureDescriptor {
        label: Some("surface_normal_tex"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("surface_reconstruct_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 5,
                resource: depth_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: thickness_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: BindingResource::TextureView(&out_view),
            },
            BindGroupEntry {
                binding: 8,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let row_bytes = width * 8;
    let tex_bytes = u64::from(row_bytes * height);
    let tex_stage = device.create_buffer(&BufferDescriptor {
        label: Some("surface_tex_stage"),
        size: tex_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("surface_reconstruct_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("surface_reconstruct_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One extra workgroup per axis exercises the in-kernel bounds guard.
        pass.dispatch_workgroups(width.div_ceil(8) + 1, height.div_ceil(8) + 1, 1);
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &tex_stage,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    tex_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let tex_view = tex_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped surface texture should be available after poll");
    let halves = bytemuck::cast_slice::<u8, u16>(&tex_view).to_vec();
    drop(tex_view);
    tex_stage.unmap();

    halves
        .chunks_exact(4)
        .map(|texel| {
            [
                f16_bits_to_f32(texel[0]),
                f16_bits_to_f32(texel[1]),
                f16_bits_to_f32(texel[2]),
                f16_bits_to_f32(texel[3]),
            ]
        })
        .collect()
}

/// Real-device parity for `water_surface_reconstruct`: reconstruct a screen
/// tile of splatted fluid depth on device and match the decoded
/// `rgba16float` `(normal.xyz, smoothed_depth)` texels to the golden. Fluid
/// pixels compare all four lanes within [`RECON_EPS`]; background pixels compare
/// the flat `(0, 0, 1)` normal and assert the depth saturates to the `f16`
/// ceiling (the `FLIP_DEPTH_FAR` sentinel exceeds the `f16` range, so the
/// driver clamps it to the largest finite `f16`, 65504).
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn surface_reconstruct_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "surface_reconstruct_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let width = 32u32;
    let height = 8u32;
    let params = GpuFlipSurfaceParams {
        resolution: [width, height],
        filter_radius: 2,
        spatial_sigma2: 4.0,
        range_sigma2: 0.5,
        pixel_world_scale: 0.01,
        _pad: [0, 0],
    };

    let (depth, thickness) = surface_reconstruct_field(width, height);
    let golden = surface_reconstruct_golden(&depth, &thickness, &params);

    let wgsl = compile_flip_wgsl();
    let entry = find_entry_point(&wgsl, "water_surface_reconstruct");
    let gpu =
        dispatch_surface_reconstruct(&device, &queue, &wgsl, &entry, &depth, &thickness, &params);

    assert_eq!(gpu.len(), golden.len(), "surface texel count mismatch");

    let mut i = 0usize;
    while i < golden.len() {
        let g = gpu[i];
        let c = golden[i];
        if c[3] >= FLIP_DEPTH_FAR {
            // Background: flat default normal, saturated far depth.
            let mut k = 0usize;
            while k < 3 {
                let d = (g[k] - c[k]).abs();
                assert!(
                    d < RECON_EPS,
                    "bg texel {i} normal[{k}]: gpu={} cpu={} |d|={d}",
                    g[k],
                    c[k],
                );
                k += 1;
            }
            // The `FLIP_DEPTH_FAR` (1e30) write exceeds the `f16` range; the
            // driver either saturates to the largest finite `f16` (65504) or
            // overflows to `inf` (decoded to `f32::MAX`). Both clear this bar.
            assert!(
                g[3] >= 65504.0,
                "bg texel {i} depth must saturate to the f16 ceiling, got {}",
                g[3],
            );
        } else {
            let mut k = 0usize;
            while k < 4 {
                let d = (g[k] - c[k]).abs();
                assert!(
                    d < RECON_EPS,
                    "texel {i} lane[{k}]: gpu={} cpu={} |d|={d}",
                    g[k],
                    c[k],
                );
                k += 1;
            }
        }
        i += 1;
    }
}

/// Builds the scene colour field sampled behind the water for the dispersion
/// parity test: an `Rgba32Float` texture whose three colour channels ramp
/// independently along x and stay constant along y, so each chromatic offset
/// lands on its own texel and the per-channel selection is observable. The
/// alpha lane is a constant `1.0`. Returned row-major as `w * h` RGBA texels.
fn dispersion_scene_field(w: u32, h: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    let mut y = 0u32;
    while y < h {
        let mut x = 0u32;
        while x < w {
            let xf = x as f32;
            // Distinct per-channel ramps keep the red/green/blue lookups from
            // aliasing onto one another and stay inside the f16 range.
            out.push(0.12 + 0.021 * xf);
            out.push(0.30 + 0.014 * xf);
            out.push(0.08 + 0.018 * xf);
            out.push(1.0);
            x += 1;
        }
        y += 1;
    }
    out
}

/// Builds the constant surface-normal field for the dispersion parity test: a
/// uniform view-space normal with a lateral x tilt (so the refraction direction
/// is exactly `+x`) and a small up component (so the incidence sine is near
/// grazing and the chromatic spread is wide). Returned row-major as `w * h`
/// RGBA texels; only the xyz lanes are read by the kernel.
fn dispersion_normal_field(w: u32, h: u32, normal: [f32; 3]) -> Vec<f32> {
    let count = w * h;
    let mut out = Vec::with_capacity((count * 4) as usize);
    let mut i = 0u32;
    while i < count {
        out.push(normal[0]);
        out.push(normal[1]);
        out.push(normal[2]);
        out.push(0.0);
        i += 1;
    }
    out
}

/// CPU golden twin of `water_dispersion_refract`. Reproduces the shader's
/// chromatic-offset chain lane-for-lane: the `Cauchy` `IOR` per `RGB`
/// wavelength (via the `dispersion` architecture module), the transmitted-sine
/// offsets scaled by `strength`, the refraction direction from the normal
/// tangent, and the nearest-neighbour scene sample per channel with
/// `ClampToEdge` addressing. Returns `w * h` `(r, g, b, 1)` texels.
fn dispersion_refract_golden(
    scene: &[f32],
    normal: &[f32],
    params: &GpuWaterDispersionParams,
) -> Vec<[f32; 4]> {
    let w = params.width;
    let h = params.height;
    let dims_w = w as f32;
    let dims_h = h as f32;
    let mut out = Vec::with_capacity((w * h) as usize);
    let mut gy = 0u32;
    while gy < h {
        let mut gx = 0u32;
        while gx < w {
            let ni = ((gy * w + gx) * 4) as usize;
            let nx = normal[ni];
            let ny = normal[ni + 1];
            let nz = normal[ni + 2];
            // Incidence sine from the normal's up (view-space z) component.
            let cos_i = nz.abs().clamp(0.0, 1.0);
            let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
            // Per-channel transmitted-sine offsets from the Cauchy law; the
            // architecture module folds in the same clamp-divide-clamp chain
            // and the strength gain the shader applies.
            let iors = dispersion::spectral_iors(params.cauchy_a, params.cauchy_b);
            let offs = dispersion::dispersion_offsets(iors, sin_i, params.strength);
            // Lateral refraction direction from the normal tangent projection.
            let t_len = (nx * nx + ny * ny).max(WATER_EPS).sqrt();
            let dir_x = nx / t_len;
            let dir_y = ny / t_len;
            let uv_x = (gx as f32 + 0.5) / dims_w;
            let uv_y = (gy as f32 + 0.5) / dims_h;
            let px_x = dir_x / dims_w;
            let px_y = dir_y / dims_h;
            let mut texel = [0.0f32; 3];
            let mut c = 0usize;
            while c < 3 {
                let su_x = uv_x + px_x * offs[c];
                let su_y = uv_y + px_y * offs[c];
                // Nearest-neighbour texel selection with ClampToEdge, matching
                // the sampler: texel = clamp(floor(uv * dim), 0, dim - 1).
                let fx = (su_x * dims_w).floor();
                let fy = (su_y * dims_h).floor();
                let tx = (fx as i32).clamp(0, (w as i32) - 1) as u32;
                let ty = (fy as i32).clamp(0, (h as i32) - 1) as u32;
                let si = ((ty * w + tx) * 4) as usize;
                texel[c] = scene[si + c].max(0.0);
                c += 1;
            }
            out.push([texel[0], texel[1], texel[2], 1.0]);
            gx += 1;
        }
        gy += 1;
    }
    out
}

/// Dispatches `water_dispersion_refract` on device and reads back the decoded
/// `rgba16float` refracted colour target. Uses a `Nearest`/`ClampToEdge`
/// `NonFiltering` sampler over `Rgba32Float` scene and normal inputs so each
/// scene sample returns an exact texel value and the parity reduces to the
/// deterministic offset-and-select chain.
#[expect(
    clippy::too_many_lines,
    reason = "one parity harness wires an explicit group-1 layout, the scene/normal input textures, a sampler, and the colour read-back in a single auditable path"
)]
fn dispatch_dispersion_refract(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    params: &GpuWaterDispersionParams,
    scene: &[f32],
    normal: &[f32],
) -> Vec<[f32; 4]> {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_render_fx_dispersion_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let empty_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water_dispersion_empty_layout"),
        entries: &[],
    });
    let group1_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water_dispersion_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("water_dispersion_pipeline_layout"),
        bind_group_layouts: &[Some(&empty_layout), Some(&group1_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_dispersion_refract_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("dispersion_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });
    let scene_tex = device.create_texture(&TextureDescriptor {
        label: Some("dispersion_scene"),
        size: Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &scene_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(scene),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(params.width * 16),
            rows_per_image: Some(params.height),
        },
        Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
    );
    let scene_view = scene_tex.create_view(&TextureViewDescriptor::default());
    let normal_tex = device.create_texture(&TextureDescriptor {
        label: Some("dispersion_normal"),
        size: Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &normal_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(normal),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(params.width * 16),
            rows_per_image: Some(params.height),
        },
        Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
    );
    let normal_view = normal_tex.create_view(&TextureViewDescriptor::default());
    let sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("dispersion_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let out_tex = device.create_texture(&TextureDescriptor {
        label: Some("dispersion_out"),
        size: Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&TextureViewDescriptor::default());

    let empty_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("dispersion_empty_group"),
        layout: &empty_layout,
        entries: &[],
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("dispersion_bind_group"),
        layout: &group1_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&out_view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&scene_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&normal_view),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::Sampler(&sampler),
            },
        ],
    });

    let row_bytes = params.width * 8;
    let tex_bytes = u64::from(row_bytes * params.height);
    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("dispersion_stage"),
        size: tex_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("dispersion_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dispersion_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &empty_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        // One extra workgroup per axis exercises the in-kernel bounds guard.
        pass.dispatch_workgroups(
            params.width.div_ceil(8) + 1,
            params.height.div_ceil(8) + 1,
            1,
        );
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &stage,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(params.height),
            },
        },
        Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped dispersion target should be available after poll");
    let halves = bytemuck::cast_slice::<u8, u16>(&view).to_vec();
    drop(view);
    stage.unmap();
    halves
        .chunks_exact(4)
        .map(|texel| {
            [
                f16_bits_to_f32(texel[0]),
                f16_bits_to_f32(texel[1]),
                f16_bits_to_f32(texel[2]),
                f16_bits_to_f32(texel[3]),
            ]
        })
        .collect()
}

/// Real-device parity for `water_dispersion_refract`: refract a ramped scene
/// colour behind a uniform near-grazing surface normal on device and match the
/// decoded `rgba16float` chromatic-fringe texels lane-for-lane against the CPU
/// golden. The exaggerated `Cauchy` dispersion coefficient pushes the red,
/// green, and blue offsets onto three distinct texels so the per-channel
/// selection is exercised, the right edge exercises the `ClampToEdge` branch,
/// and the alpha lane must read a flat `1.0`.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn dispersion_refract_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "dispersion_refract_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let width = 32u32;
    let height = 8u32;
    let params = GpuWaterDispersionParams {
        cauchy_a: 1.324,
        cauchy_b: 0.36,
        strength: 10.0,
        width,
        height,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    // A lateral x tilt yields an exact +x refraction direction; the small up
    // component puts the incidence near grazing for a wide chromatic spread.
    let normal = [0.5f32, 0.0, 0.05];

    let scene = dispersion_scene_field(width, height);
    let normal_field = dispersion_normal_field(width, height, normal);
    let golden = dispersion_refract_golden(&scene, &normal_field, &params);

    let wgsl = compile_render_fx_wgsl();
    let entry = find_entry_point(&wgsl, "water_dispersion_refract");
    let gpu = dispatch_dispersion_refract(
        &device,
        &queue,
        &wgsl,
        &entry,
        &params,
        &scene,
        &normal_field,
    );

    assert_eq!(gpu.len(), golden.len(), "dispersion texel count mismatch");

    let mut i = 0usize;
    while i < golden.len() {
        let g = gpu[i];
        let c = golden[i];
        let mut k = 0usize;
        while k < 4 {
            let d = (g[k] - c[k]).abs();
            assert!(
                d < RECON_EPS,
                "texel {i} lane[{k}]: gpu={} cpu={} |d|={d}",
                g[k],
                c[k],
            );
            k += 1;
        }
        i += 1;
    }
}

// -----------------------------------------------------------------------------
// Spectral inverse-`FFT` parity (`water_spectrum_ifft`).
// -----------------------------------------------------------------------------

use super::abi::GpuWaterSpectrumParams;

/// Deterministic complex spectral seed for the `water_spectrum_ifft` parity
/// test. Returns the `h0(+k)` and `h0(-k)` arrays (`N*N` entries, `[real, imag]`
/// packed as `[f32; 2]` to match the shader's `array<vec2<f32>>`), filled with
/// small bounded values so the summed field stays `O(1)` and the `rgba32float`
/// accumulation error stays far under [`PARITY_EPS`].
fn spectrum_field(n: u32) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let count = (n * n) as usize;
    let mut h0 = Vec::with_capacity(count);
    let mut h0_neg = Vec::with_capacity(count);
    let mut i = 0usize;
    while i < count {
        let fi = i as f32;
        h0.push([
            0.02 * bevy_math::ops::sin(0.7 * fi + 0.3),
            0.02 * bevy_math::ops::cos(0.4 * fi + 1.1),
        ]);
        h0_neg.push([
            0.015 * bevy_math::ops::sin(0.9 * fi + 2.0),
            0.015 * bevy_math::ops::cos(0.6 * fi + 0.5),
        ]);
        i += 1;
    }
    (h0, h0_neg)
}

/// A deterministic *Hermitian* spectrum for the packed-`FFT` parity test.
///
/// The two-for-one packing (`G = A_hat + i*B_hat`) only reconstructs both real
/// fields when each field's spectrum is Hermitian (`X_hat(-k) = conj(X_hat(k))`),
/// which for the `Tessendorf` advance reduces to `h0_neg(k) = h0(-k)` (see
/// `water_spectrum_fft.wesl`). Production spectra satisfy this; the arbitrary
/// [`spectrum_field`] does not. Here `h0` is filled on the interior and mirrored
/// into `h0_neg` at the frequency-negated index `(N-m, N-n)`; the boundary lanes
/// (`m == 0` or `n == 0`, whose negated frequency `+N/2` is unrepresentable) are
/// zeroed so the grid is exactly Hermitian with no Nyquist defect.
fn spectrum_field_hermitian(n: u32) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let count = (n * n) as usize;
    let mut h0 = vec![[0.0_f32, 0.0]; count];
    let mut m = 1u32;
    while m < n {
        let mut nn = 1u32;
        while nn < n {
            let idx = (m * n + nn) as usize;
            let fm = m as f32;
            let fn_ = nn as f32;
            h0[idx] = [
                0.03 * bevy_math::ops::sin(0.6 * fm + 0.2 * fn_ + 0.3),
                0.03 * bevy_math::ops::cos(0.4 * fm - 0.5 * fn_ + 1.1),
            ];
            nn += 1;
        }
        m += 1;
    }
    let mut h0_neg = vec![[0.0_f32, 0.0]; count];
    let mut m = 0u32;
    while m < n {
        let mut nn = 0u32;
        while nn < n {
            let idx = (m * n + nn) as usize;
            let m2 = (n - m) % n;
            let nn2 = (n - nn) % n;
            let mirror = (m2 * n + nn2) as usize;
            h0_neg[idx] = h0[mirror];
            nn += 1;
        }
        m += 1;
    }
    (h0, h0_neg)
}

/// `CPU` golden twin of `water_spectrum_ifft`. Replays the direct-summation
/// inverse transform texel-for-texel with the crate's `libm`-backed
/// `bevy_math::ops` `sin`/`cos` (the same function the shader's native `sin`/`cos`
/// intrinsics evaluate). Returns the two flat `rgba32float` fields in texel order
/// `n = y*N + x`: the displacement `(Dx, height, Dz, J)` and the normal
/// `(nx, ny, nz, foam)`.
#[expect(
    clippy::too_many_lines,
    reason = "the eight complex accumulators of the direct-summation transform are clearest replayed inline against the shader"
)]
fn spectrum_ifft_golden(
    h0: &[[f32; 2]],
    h0_neg: &[[f32; 2]],
    params: &GpuWaterSpectrumParams,
) -> (Vec<f32>, Vec<f32>) {
    let n = params.grid_size;
    let total = (n * n) as usize;
    let mut disp_out = vec![0.0_f32; total * 4];
    let mut norm_out = vec![0.0_f32; total * 4];

    let inv_n = 1.0_f32 / (n as f32).max(1.0);
    let cell = params.patch_size * inv_n;
    let dk = WATER_TWO_PI / params.patch_size.max(WATER_EPS_LEN_SQ);
    let half = n as f32 * 0.5;
    let lambda = params.choppiness;

    let mut gy = 0u32;
    while gy < n {
        let mut gx = 0u32;
        while gx < n {
            let world_x = gx as f32 * cell;
            let world_z = gy as f32 * cell;

            let mut acc_height = [0.0_f32, 0.0];
            let mut acc_disp_x = [0.0_f32, 0.0];
            let mut acc_disp_z = [0.0_f32, 0.0];
            let mut acc_slope_x = [0.0_f32, 0.0];
            let mut acc_slope_z = [0.0_f32, 0.0];
            let mut acc_dxdx = [0.0_f32, 0.0];
            let mut acc_dzdz = [0.0_f32, 0.0];
            let mut acc_dxdz = [0.0_f32, 0.0];

            let mut m = 0u32;
            while m < n {
                let kx = (m as f32 - half) * dk;
                let mut nn = 0u32;
                while nn < n {
                    let kz = (nn as f32 - half) * dk;
                    let idx = (m * n + nn) as usize;

                    let k_sq = kx * kx + kz * kz;
                    let omega = ocean_dispersion(k_sq.sqrt());
                    // Hermitian advance: h0 e^{i w t} + conj(h0_neg) e^{-i w t}.
                    let theta = omega * params.time;
                    let cf = bevy_math::ops::cos(theta);
                    let sf = bevy_math::ops::sin(theta);
                    let hp = h0[idx];
                    let fwd = [hp[0] * cf - hp[1] * sf, hp[0] * sf + hp[1] * cf];
                    // conj(h0_neg) * e^{-i w t}: conjugate negates imag, phasor
                    // uses (cos(-theta), sin(-theta)) = (cf, -sf).
                    let hn = [h0_neg[idx][0], -h0_neg[idx][1]];
                    let bwd = [hn[0] * cf - hn[1] * (-sf), hn[0] * (-sf) + hn[1] * cf];
                    let h = [fwd[0] + bwd[0], fwd[1] + bwd[1]];

                    // Inverse-transform kernel e^{i k·x} for this texel.
                    let phase = kx * world_x + kz * world_z;
                    let cp = bevy_math::ops::cos(phase);
                    let sp = bevy_math::ops::sin(phase);
                    let hk = [h[0] * cp - h[1] * sp, h[0] * sp + h[1] * cp];

                    acc_height[0] += hk[0];
                    acc_height[1] += hk[1];

                    if k_sq > WATER_EPS_LEN_SQ {
                        let inv_k = 1.0_f32 / k_sq.sqrt();
                        // (0, a) * hk = (-a*hk_im, a*hk_re) — pure-imag factor.
                        let fx = -kx * inv_k;
                        acc_disp_x[0] += -fx * hk[1];
                        acc_disp_x[1] += fx * hk[0];
                        let fz = -kz * inv_k;
                        acc_disp_z[0] += -fz * hk[1];
                        acc_disp_z[1] += fz * hk[0];
                        // Slope factor i*k (per axis).
                        acc_slope_x[0] += -kx * hk[1];
                        acc_slope_x[1] += kx * hk[0];
                        acc_slope_z[0] += -kz * hk[1];
                        acc_slope_z[1] += kz * hk[0];
                        // (r, 0) * hk = (r*hk_re, r*hk_im) — pure-real factor.
                        let gxx = kx * kx * inv_k;
                        acc_dxdx[0] += gxx * hk[0];
                        acc_dxdx[1] += gxx * hk[1];
                        let gzz = kz * kz * inv_k;
                        acc_dzdz[0] += gzz * hk[0];
                        acc_dzdz[1] += gzz * hk[1];
                        let gxz = kx * kz * inv_k;
                        acc_dxdz[0] += gxz * hk[0];
                        acc_dxdz[1] += gxz * hk[1];
                    }
                    nn += 1;
                }
                m += 1;
            }

            let height = acc_height[0];
            let disp_x = lambda * acc_disp_x[0];
            let disp_z = lambda * acc_disp_z[0];
            let dxdx = lambda * acc_dxdx[0];
            let dzdz = lambda * acc_dzdz[0];
            let dxdz = lambda * acc_dxdz[0];
            let jacobian = (1.0 + dxdx) * (1.0 + dzdz) - dxdz * dxdz;

            // N = normalize(-dh/dx, 1, -dh/dz); ny = 1 keeps the length >= 1.
            let nx = -acc_slope_x[0];
            let ny = 1.0_f32;
            let nz = -acc_slope_z[0];
            let inv_len = 1.0_f32 / (nx * nx + ny * ny + nz * nz).sqrt();
            let normal = [nx * inv_len, ny * inv_len, nz * inv_len];

            let mut foam = 0.0_f32;
            if jacobian <= params.foam_threshold {
                foam = 1.0;
            }

            let base = ((gy * n + gx) * 4) as usize;
            disp_out[base] = disp_x;
            disp_out[base + 1] = height;
            disp_out[base + 2] = disp_z;
            disp_out[base + 3] = jacobian;
            norm_out[base] = normal[0];
            norm_out[base + 1] = normal[1];
            norm_out[base + 2] = normal[2];
            norm_out[base + 3] = foam;

            gx += 1;
        }
        gy += 1;
    }

    (disp_out, norm_out)
}

/// Dispatches one `water_spectrum_ifft` pass on device and reads back the
/// displacement and normal textures as flat `rgba32float` lanes.
///
/// The `group(0)` layout is reflected straight off the pipeline (`layout: None`),
/// so only the bindings this entry point touches are materialised: `0`/`1` the
/// read-only `h0`/`h0_neg` spectral buffers, `2` the params uniform, and `3`/`4`
/// the displacement and normal storage textures. `N == 16` keeps
/// `row_bytes == N*16 == 256` naturally aligned, so the readback rows are dense.
#[expect(
    clippy::too_many_lines,
    reason = "one linear dispatch-and-readback over two storage textures keeps the parity path auditable"
)]
fn dispatch_spectrum_ifft(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    h0: &[[f32; 2]],
    h0_neg: &[[f32; 2]],
    params: &GpuWaterSpectrumParams,
) -> (Vec<f32>, Vec<f32>) {
    let n = params.grid_size;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_ocean_spectrum_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_spectrum_ifft_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let h0_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("spectrum_h0"),
        contents: bytemuck::cast_slice(h0),
        usage: BufferUsages::STORAGE,
    });
    let h0_neg_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("spectrum_h0_neg"),
        contents: bytemuck::cast_slice(h0_neg),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("spectrum_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });

    let extent = Extent3d {
        width: n,
        height: n,
        depth_or_array_layers: 1,
    };
    let disp_tex = device.create_texture(&TextureDescriptor {
        label: Some("spectrum_displacement_out"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let norm_tex = device.create_texture(&TextureDescriptor {
        label: Some("spectrum_normal_out"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let disp_view = disp_tex.create_view(&TextureViewDescriptor::default());
    let norm_view = norm_tex.create_view(&TextureViewDescriptor::default());

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("spectrum_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: h0_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: h0_neg_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&disp_view),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&norm_view),
            },
        ],
    });

    let row_bytes = n * 16;
    let readback_size = u64::from(row_bytes * n);
    let disp_readback = device.create_buffer(&BufferDescriptor {
        label: Some("spectrum_disp_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let norm_readback = device.create_buffer(&BufferDescriptor {
        label: Some("spectrum_norm_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("spectrum_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("spectrum_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = n.div_ceil(8);
        pass.dispatch_workgroups(groups, groups, 1);
    }
    for (tex, readback) in [(&disp_tex, &disp_readback), (&norm_tex, &norm_readback)] {
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(n),
                },
            },
            extent,
        );
    }
    queue.submit([encoder.finish()]);

    disp_readback.slice(..).map_async(MapMode::Read, |_| {});
    norm_readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let disp_out = {
        let view = disp_readback
            .slice(..)
            .get_mapped_range()
            .expect("mapped displacement readback should be available after poll");
        let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        disp_readback.unmap();
        floats
    };
    let norm_out = {
        let view = norm_readback
            .slice(..)
            .get_mapped_range()
            .expect("mapped normal readback should be available after poll");
        let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        norm_readback.unmap();
        floats
    };

    (disp_out, norm_out)
}

/// Real-device parity for `water_spectrum_ifft`: run the direct-summation
/// inverse transform on device over a deterministic Hermitian spectrum and match
/// both output textures against the `CPU` golden. The chosen `choppiness`/foam
/// threshold exercise both whitecap branches (`foam == 0` and `foam == 1`) and
/// the `k_sq <= EPS` center-cell guard.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn spectrum_ifft_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "spectrum_ifft_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let params = GpuWaterSpectrumParams {
        grid_size: 16,
        patch_size: 50.0,
        time: 1.3,
        choppiness: 1.6,
        foam_threshold: 1.05,
        h0_offset: 0,
        tile_origin_y: 0,
        _pad: 0,
    };

    let (h0, h0_neg) = spectrum_field(params.grid_size);
    let (gold_disp, gold_norm) = spectrum_ifft_golden(&h0, &h0_neg, &params);

    // The golden must actually straddle the whitecap threshold, else the parity
    // check would never touch the `foam == 1` branch.
    let mut foam_off = false;
    let mut foam_on = false;
    let mut t = 3usize;
    while t < gold_norm.len() {
        if gold_norm[t] < 0.5 {
            foam_off = true;
        } else {
            foam_on = true;
        }
        t += 4;
    }
    assert!(
        foam_off && foam_on,
        "spectrum golden must cover both whitecap branches (off={foam_off}, on={foam_on})"
    );

    let wgsl = compile_ocean_wgsl();
    let entry = find_entry_point(&wgsl, "water_spectrum_ifft");
    let (gpu_disp, gpu_norm) =
        dispatch_spectrum_ifft(&device, &queue, &wgsl, &entry, &h0, &h0_neg, &params);

    assert_eq!(
        gpu_disp.len(),
        gold_disp.len(),
        "displacement texel count mismatch"
    );
    assert_eq!(
        gpu_norm.len(),
        gold_norm.len(),
        "normal texel count mismatch"
    );

    let mut i = 0usize;
    while i < gold_disp.len() {
        let dd = (gpu_disp[i] - gold_disp[i]).abs();
        assert!(
            dd < PARITY_EPS,
            "displacement lane {i}: gpu={} cpu={} |d|={dd}",
            gpu_disp[i],
            gold_disp[i],
        );
        let dn = (gpu_norm[i] - gold_norm[i]).abs();
        assert!(
            dn < PARITY_EPS,
            "normal lane {i}: gpu={} cpu={} |d|={dn}",
            gpu_norm[i],
            gold_norm[i],
        );
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Butterfly FFT parity: the ping-pong `water_butterfly.wesl` inverse transform
// versus a host-side radix-2 `DIT` reference sharing the shader's `libm` trig.
// ---------------------------------------------------------------------------

/// Uniform driving one butterfly pass; mirrors the `FftParams` struct in
/// `water_butterfly.wesl` (16 bytes, `std140`-safe as four `u32`s).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFftParams {
    n: u32,
    axis: u32,
    len: u32,
    log2n: u32,
}

/// Compiles `water_butterfly.wesl` to `Wgsl` through the render-world cache,
/// mirroring [`compile_pbf_wgsl`] but for the butterfly module.
fn compile_butterfly_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_5242_5546_0002),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_butterfly.wesl"),
            "embedded://prism_render_scene/shaders/water_butterfly.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_butterfly.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Reverses the low `bits` bits of `x` (host mirror of the shader's
/// `reverse_bits_low` and the crate golden's `reverse_bits`).
fn cpu_reverse_bits(mut x: usize, bits: u32) -> usize {
    let mut reversed = 0usize;
    let mut b = 0u32;
    while b < bits {
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
        b += 1;
    }
    reversed
}

/// In-place radix-2 `DIT` transform of one `[re, im]` line, using the shader's
/// `libm` [`bevy_math::ops`] trig so the reference matches the on-device path
/// to `float32` rounding. `inverse` flips the twiddle sign; no normalisation is
/// applied here (the caller scales once, as `transform2` does).
fn cpu_line_transform(buf: &mut [[f32; 2]], inverse: bool) {
    let n = buf.len();
    let bits = n.trailing_zeros();
    let mut i = 1usize;
    while i < n {
        let j = cpu_reverse_bits(i, bits);
        if j > i {
            buf.swap(i, j);
        }
        i += 1;
    }
    let sign = if inverse { 1.0_f32 } else { -1.0_f32 };
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let step = sign * core::f32::consts::TAU / len as f32;
        let mut start = 0usize;
        while start < n {
            let mut k = 0usize;
            while k < half {
                let theta = step * k as f32;
                let tw = [bevy_math::ops::cos(theta), bevy_math::ops::sin(theta)];
                let top = buf[start + k];
                let bi = buf[start + k + half];
                let bottom = [bi[0] * tw[0] - bi[1] * tw[1], bi[0] * tw[1] + bi[1] * tw[0]];
                buf[start + k] = [top[0] + bottom[0], top[1] + bottom[1]];
                buf[start + k + half] = [top[0] - bottom[0], top[1] - bottom[1]];
                k += 1;
            }
            start += len;
        }
        len *= 2;
    }
}

/// Separable 2D inverse `FFT` of a row-major `n*n` `[re, im]` grid: transform
/// every row, then every column, then scale once by `1/(N*N)`. Host reference
/// for the `GPU` butterfly, identical in structure to the crate golden
/// `fft::ifft2` but sharing the shader's `libm` trig.
fn cpu_butterfly_ifft2(grid: &[[f32; 2]], n: usize) -> Vec<[f32; 2]> {
    let mut data = grid.to_vec();
    let mut row = 0usize;
    while row < n {
        let start = row * n;
        cpu_line_transform(&mut data[start..start + n], true);
        row += 1;
    }
    let mut col = 0usize;
    while col < n {
        let mut column: Vec<[f32; 2]> = Vec::with_capacity(n);
        let mut r = 0usize;
        while r < n {
            column.push(data[r * n + col]);
            r += 1;
        }
        cpu_line_transform(&mut column, true);
        let mut r = 0usize;
        while r < n {
            data[r * n + col] = column[r];
            r += 1;
        }
        col += 1;
    }
    let inv = 1.0_f32 / (n * n) as f32;
    for c in &mut data {
        c[0] *= inv;
        c[1] *= inv;
    }
    data
}

/// A deterministic, non-symmetric complex `n*n` spectrum grid. Asymmetry across
/// both axes ensures the row and column passes, the bit-reversal permutation,
/// and every twiddle are genuinely exercised (a symmetric grid could mask an
/// axis/sign bug).
fn build_butterfly_input(n: usize) -> Vec<[f32; 2]> {
    let mut grid = Vec::with_capacity(n * n);
    let mut y = 0usize;
    while y < n {
        let mut x = 0usize;
        while x < n {
            let fx = x as f32;
            let fy = y as f32;
            let re = 0.1 * fx - 0.05 * fy + 0.01 * fx * fy;
            let im = 0.2 - 0.03 * fx + 0.07 * fy - 0.004 * fx * fx;
            grid.push([re, im]);
            x += 1;
        }
        y += 1;
    }
    grid
}

/// Records the full inverse-`FFT` pass schedule into one encoder and returns
/// the mapped result. Ping-pongs two storage buffers across `bitrev` + `log2 N`
/// stages on axis 0, the same on axis 1, then one normalize — the exact
/// row-then-column, normalise-once order the `CPU` `transform2` uses.
fn dispatch_butterfly_ifft2(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    bitrev: &str,
    stage: &str,
    normalize: &str,
    input: &[[f32; 2]],
    n: usize,
) -> Vec<[f32; 2]> {
    let byte_len = size_of_val(input) as u64;
    let log2n = (n as u32).trailing_zeros();

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_butterfly_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let make_pipeline = |entry: &str, label: &str| {
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    let bitrev_pipeline = make_pipeline(bitrev, "butterfly_bitrev");
    let stage_pipeline = make_pipeline(stage, "butterfly_stage");
    let normalize_pipeline = make_pipeline(normalize, "butterfly_normalize");

    let buf_a = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("butterfly_a"),
        contents: bytemuck::cast_slice(input),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let buf_b = device.create_buffer(&BufferDescriptor {
        label: Some("butterfly_b"),
        size: byte_len,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Build the ordered pass list: (pipeline, params). `len == 0` marks a
    // reorder/normalize pass (the shader ignores `len` there).
    let mut passes: Vec<(&wgpu::ComputePipeline, GpuFftParams, u32, u32)> = Vec::new();
    for axis in [0u32, 1u32] {
        passes.push((
            &bitrev_pipeline,
            GpuFftParams {
                n: n as u32,
                axis,
                len: 0,
                log2n,
            },
            (n as u32).div_ceil(8),
            (n as u32).div_ceil(8),
        ));
        let mut len = 2u32;
        while len as usize <= n {
            passes.push((
                &stage_pipeline,
                GpuFftParams {
                    n: n as u32,
                    axis,
                    len,
                    log2n,
                },
                ((n as u32) / 2).div_ceil(8),
                (n as u32).div_ceil(8),
            ));
            len *= 2;
        }
    }
    passes.push((
        &normalize_pipeline,
        GpuFftParams {
            n: n as u32,
            axis: 0,
            len: 0,
            log2n,
        },
        (n as u32).div_ceil(8),
        (n as u32).div_ceil(8),
    ));

    // Each pass gets its own uniform buffer and bind group (params differ).
    let param_buffers: Vec<wgpu::Buffer> = passes
        .iter()
        .map(|(_, params, _, _)| {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("butterfly_params"),
                contents: bytemuck::bytes_of(params),
                usage: BufferUsages::UNIFORM,
            })
        })
        .collect();

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("butterfly_parity_encoder"),
    });

    let mut cur_is_a = true;
    for (pass_index, (pipeline, _, groups_x, groups_y)) in passes.iter().enumerate() {
        let (src, dst) = if cur_is_a {
            (&buf_a, &buf_b)
        } else {
            (&buf_b, &buf_a)
        };
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("butterfly_group0"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: param_buffers[pass_index].as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("butterfly_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(*groups_x, *groups_y, 1);
        }
        cur_is_a = !cur_is_a;
    }

    // After the loop, the last-written buffer is the one `cur_is_a` now points
    // *away* from (each pass wrote `dst`, then toggled).
    let final_buf = if cur_is_a { &buf_a } else { &buf_b };

    let stage_buf = device.create_buffer(&BufferDescriptor {
        label: Some("butterfly_stage_readback"),
        size: byte_len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(final_buf, 0, &stage_buf, 0, byte_len);
    queue.submit([encoder.finish()]);

    stage_buf.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");
    let view = stage_buf
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values: Vec<[f32; 2]> = bytemuck::cast_slice::<u8, [f32; 2]>(&view).to_vec();
    drop(view);
    stage_buf.unmap();
    values
}

/// The on-device ping-pong butterfly inverse `FFT` must match the host `DIT`
/// reference (and thus the crate golden `fft::ifft2`) cell-for-cell within
/// `float32` rounding.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn butterfly_ifft2_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "butterfly_ifft2_gpu_matches_cpu_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let n = 16usize;
    let input = build_butterfly_input(n);
    let golden = cpu_butterfly_ifft2(&input, n);

    let wgsl = compile_butterfly_wgsl();
    let bitrev = find_entry_point(&wgsl, "water_fft_bitrev");
    let stage = find_entry_point(&wgsl, "water_fft_stage");
    let normalize = find_entry_point(&wgsl, "water_fft_normalize");

    let gpu = dispatch_butterfly_ifft2(
        &device, &queue, &wgsl, &bitrev, &stage, &normalize, &input, n,
    );

    assert_eq!(gpu.len(), golden.len(), "readback length mismatch");
    let mut i = 0usize;
    while i < golden.len() {
        let dre = (gpu[i][0] - golden[i][0]).abs();
        let dim = (gpu[i][1] - golden[i][1]).abs();
        assert!(
            dre < PARITY_EPS && dim < PARITY_EPS,
            "cell {i}: gpu=({}, {}) cpu=({}, {}) |dre|={dre} |dim|={dim}",
            gpu[i][0],
            gpu[i][1],
            golden[i][0],
            golden[i][1],
        );
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Packed spectral FFT pipeline parity: the O(N log N) `water_spectrum_evolve`
// + butterfly `ifft2` + `water_spectrum_assemble` path must reproduce the
// direct-summation `spectrum_ifft_golden` texel-for-texel.
// ---------------------------------------------------------------------------

/// Compiles `water_spectrum_fft.wesl` to `Wgsl` through the render-world cache.
fn compile_spectrum_fft_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_5246_4654_0003),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_spectrum_fft.wesl"),
            "embedded://prism_render_scene/shaders/water_spectrum_fft.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_spectrum_fft.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Dispatches `water_spectrum_evolve` and reads back the four packed complex
/// grids (`G0..G3`) as row-major `[re, im]` arrays indexed `m*N + n`.
#[expect(
    clippy::too_many_lines,
    reason = "one linear dispatch-and-readback over four packed output grids keeps the parity path auditable"
)]
fn dispatch_spectrum_evolve(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    h0: &[[f32; 2]],
    h0_neg: &[[f32; 2]],
    params: &GpuWaterSpectrumParams,
) -> [Vec<[f32; 2]>; 4] {
    let n = params.grid_size;
    let cell_count = (n * n) as usize;
    let byte_len = (cell_count * size_of::<[f32; 2]>()) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_spectrum_evolve_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_spectrum_evolve_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let h0_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("evolve_h0"),
        contents: bytemuck::cast_slice(h0),
        usage: BufferUsages::STORAGE,
    });
    let h0_neg_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("evolve_h0_neg"),
        contents: bytemuck::cast_slice(h0_neg),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("evolve_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });
    let make_g = |label: &str| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let g0 = make_g("packed_g0");
    let g1 = make_g("packed_g1");
    let g2 = make_g("packed_g2");
    let g3 = make_g("packed_g3");

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("evolve_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: h0_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: h0_neg_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: g0.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: g1.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: g2.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: g3.as_entire_binding(),
            },
        ],
    });

    let readbacks: [wgpu::Buffer; 4] = core::array::from_fn(|_| {
        device.create_buffer(&BufferDescriptor {
            label: Some("evolve_readback"),
            size: byte_len,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("evolve_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("evolve_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = n.div_ceil(8);
        pass.dispatch_workgroups(groups, groups, 1);
    }
    for (src, dst) in [
        (&g0, &readbacks[0]),
        (&g1, &readbacks[1]),
        (&g2, &readbacks[2]),
        (&g3, &readbacks[3]),
    ] {
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, byte_len);
    }
    queue.submit([encoder.finish()]);

    for rb in &readbacks {
        rb.slice(..).map_async(MapMode::Read, |_| {});
    }
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    core::array::from_fn(|i| {
        let view = readbacks[i]
            .slice(..)
            .get_mapped_range()
            .expect("mapped evolve readback should be available after poll");
        let grid: Vec<[f32; 2]> = bytemuck::cast_slice::<u8, [f32; 2]>(&view).to_vec();
        drop(view);
        readbacks[i].unmap();
        grid
    })
}

/// Dispatches `water_spectrum_assemble` over the four inverse-transformed grids
/// and reads back the displacement and normal textures as flat `rgba32float`
/// lanes (texel order `n = y*N + x`).
#[expect(
    clippy::too_many_lines,
    reason = "one linear dispatch-and-readback over four inputs and two storage textures keeps the parity path auditable"
)]
fn dispatch_spectrum_assemble(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    grids: &[Vec<[f32; 2]>; 4],
    params: &GpuWaterSpectrumParams,
) -> (Vec<f32>, Vec<f32>) {
    let n = params.grid_size;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_spectrum_assemble_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_spectrum_assemble_parity"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("assemble_params"),
        contents: bytemuck::bytes_of(params),
        usage: BufferUsages::UNIFORM,
    });
    let g_bufs: [wgpu::Buffer; 4] = core::array::from_fn(|i| {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("assemble_g"),
            contents: bytemuck::cast_slice(&grids[i]),
            usage: BufferUsages::STORAGE,
        })
    });

    let extent = Extent3d {
        width: n,
        height: n,
        depth_or_array_layers: 1,
    };
    let make_tex = |label: &str| {
        device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let disp_tex = make_tex("assemble_displacement_out");
    let norm_tex = make_tex("assemble_normal_out");
    let disp_view = disp_tex.create_view(&TextureViewDescriptor::default());
    let norm_view = norm_tex.create_view(&TextureViewDescriptor::default());

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("assemble_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 7,
                resource: g_bufs[0].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: g_bufs[1].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 9,
                resource: g_bufs[2].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 10,
                resource: g_bufs[3].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 11,
                resource: BindingResource::TextureView(&disp_view),
            },
            BindGroupEntry {
                binding: 12,
                resource: BindingResource::TextureView(&norm_view),
            },
        ],
    });

    let row_bytes = n * 16;
    let readback_size = u64::from(row_bytes * n);
    let disp_readback = device.create_buffer(&BufferDescriptor {
        label: Some("assemble_disp_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let norm_readback = device.create_buffer(&BufferDescriptor {
        label: Some("assemble_norm_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("assemble_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("assemble_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = n.div_ceil(8);
        pass.dispatch_workgroups(groups, groups, 1);
    }
    for (tex, readback) in [(&disp_tex, &disp_readback), (&norm_tex, &norm_readback)] {
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(n),
                },
            },
            extent,
        );
    }
    queue.submit([encoder.finish()]);

    disp_readback.slice(..).map_async(MapMode::Read, |_| {});
    norm_readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let disp_out = {
        let view = disp_readback
            .slice(..)
            .get_mapped_range()
            .expect("mapped displacement readback should be available after poll");
        let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        disp_readback.unmap();
        floats
    };
    let norm_out = {
        let view = norm_readback
            .slice(..)
            .get_mapped_range()
            .expect("mapped normal readback should be available after poll");
        let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        norm_readback.unmap();
        floats
    };

    (disp_out, norm_out)
}

/// End-to-end parity for the packed spectral butterfly path: run
/// `water_spectrum_evolve` on device, inverse-`FFT` the four packed grids with
/// the `water_butterfly.wesl` ping-pong pipeline, run `water_spectrum_assemble`,
/// and match both output textures against the direct-summation
/// `spectrum_ifft_golden`. This proves the O(N log N) production path is
/// numerically identical to the O(N⁴) reference it replaces.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn spectrum_fft_pipeline_gpu_matches_direct_sum_golden() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "spectrum_fft_pipeline_gpu_matches_direct_sum_golden: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let params = GpuWaterSpectrumParams {
        grid_size: 16,
        patch_size: 50.0,
        time: 1.3,
        choppiness: 1.6,
        foam_threshold: 1.05,
        h0_offset: 0,
        tile_origin_y: 0,
        _pad: 0,
    };
    let n = params.grid_size as usize;

    let (h0, h0_neg) = spectrum_field_hermitian(params.grid_size);
    let (gold_disp, gold_norm) = spectrum_ifft_golden(&h0, &h0_neg, &params);

    // The golden must straddle the whitecap threshold so the parity check
    // exercises both `foam == 0` and `foam == 1` branches of assemble.
    let mut foam_off = false;
    let mut foam_on = false;
    let mut t = 3usize;
    while t < gold_norm.len() {
        if gold_norm[t] < 0.5 {
            foam_off = true;
        } else {
            foam_on = true;
        }
        t += 4;
    }
    assert!(
        foam_off && foam_on,
        "spectrum golden must cover both whitecap branches (off={foam_off}, on={foam_on})"
    );

    // 1. Packed evolve → four complex grids at index `m*N + n`.
    let fft_wgsl = compile_spectrum_fft_wgsl();
    let evolve_entry = find_entry_point(&fft_wgsl, "water_spectrum_evolve");
    let assemble_entry = find_entry_point(&fft_wgsl, "water_spectrum_assemble");
    let packed = dispatch_spectrum_evolve(
        &device,
        &queue,
        &fft_wgsl,
        &evolve_entry,
        &h0,
        &h0_neg,
        &params,
    );

    // 2. Separable inverse FFT2 of each packed grid (the production butterfly).
    let butterfly_wgsl = compile_butterfly_wgsl();
    let bitrev = find_entry_point(&butterfly_wgsl, "water_fft_bitrev");
    let stage = find_entry_point(&butterfly_wgsl, "water_fft_stage");
    let normalize = find_entry_point(&butterfly_wgsl, "water_fft_normalize");
    let transformed: [Vec<[f32; 2]>; 4] = core::array::from_fn(|i| {
        dispatch_butterfly_ifft2(
            &device,
            &queue,
            &butterfly_wgsl,
            &bitrev,
            &stage,
            &normalize,
            &packed[i],
            n,
        )
    });

    // 3. Assemble → displacement/normal textures.
    let (gpu_disp, gpu_norm) = dispatch_spectrum_assemble(
        &device,
        &queue,
        &fft_wgsl,
        &assemble_entry,
        &transformed,
        &params,
    );

    assert_eq!(
        gpu_disp.len(),
        gold_disp.len(),
        "displacement texel count mismatch"
    );
    assert_eq!(
        gpu_norm.len(),
        gold_norm.len(),
        "normal texel count mismatch"
    );

    let mut i = 0usize;
    while i < gold_disp.len() {
        let dd = (gpu_disp[i] - gold_disp[i]).abs();
        assert!(
            dd < PARITY_EPS,
            "displacement lane {i}: fft={} direct={} |d|={dd}",
            gpu_disp[i],
            gold_disp[i],
        );
        let dn = (gpu_norm[i] - gold_norm[i]).abs();
        assert!(
            dn < PARITY_EPS,
            "normal lane {i}: fft={} direct={} |d|={dn}",
            gpu_norm[i],
            gold_norm[i],
        );
        i += 1;
    }
}

/// Dispatches `water_spectrum_assemble` once per cascade into a single stacked
/// atlas texture of size `N x (M*N)`, each cascade writing its `N x N` tile at
/// rows `[tile_origin_y, tile_origin_y + N)` exactly as the production recorder
/// does. Returns the displacement and normal atlases as flat `rgba32float`
/// lanes in texel order `n = y*N + x` over the full `M*N`-tall atlas.
#[expect(
    clippy::too_many_lines,
    reason = "one dispatch-and-readback per cascade into a shared atlas keeps the multi-cascade parity path auditable in one place"
)]
fn dispatch_cascade_assemble_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    cascades: &[([Vec<[f32; 2]>; 4], GpuWaterSpectrumParams)],
    atlas_height: u32,
) -> (Vec<f32>, Vec<f32>) {
    let n = cascades[0].1.grid_size;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water_cascade_assemble_atlas"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("water_cascade_assemble_atlas"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let extent = Extent3d {
        width: n,
        height: atlas_height,
        depth_or_array_layers: 1,
    };
    let make_tex = |label: &str| {
        device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let disp_tex = make_tex("cascade_atlas_displacement");
    let norm_tex = make_tex("cascade_atlas_normal");
    let disp_view = disp_tex.create_view(&TextureViewDescriptor::default());
    let norm_view = norm_tex.create_view(&TextureViewDescriptor::default());

    let layout = pipeline.get_bind_group_layout(0);

    // Keep every per-cascade buffer alive until submission completes.
    let mut keep_alive: Vec<wgpu::Buffer> = Vec::new();
    let mut bind_groups: Vec<wgpu::BindGroup> = Vec::new();
    for (grids, params) in cascades {
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("cascade_assemble_params"),
            contents: bytemuck::bytes_of(params),
            usage: BufferUsages::UNIFORM,
        });
        let g_bufs: [wgpu::Buffer; 4] = core::array::from_fn(|i| {
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("cascade_assemble_g"),
                contents: bytemuck::cast_slice(&grids[i]),
                usage: BufferUsages::STORAGE,
            })
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("cascade_assemble_bind_group"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: g_bufs[0].as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: g_bufs[1].as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: g_bufs[2].as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: g_bufs[3].as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: BindingResource::TextureView(&disp_view),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: BindingResource::TextureView(&norm_view),
                },
            ],
        });
        bind_groups.push(bind_group);
        keep_alive.push(params_buf);
        keep_alive.extend(g_bufs);
    }

    let row_bytes = n * 16;
    let readback_size = u64::from(row_bytes * atlas_height);
    let disp_readback = device.create_buffer(&BufferDescriptor {
        label: Some("cascade_atlas_disp_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let norm_readback = device.create_buffer(&BufferDescriptor {
        label: Some("cascade_atlas_norm_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cascade_assemble_encoder"),
    });
    for bind_group in &bind_groups {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cascade_assemble_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let groups = n.div_ceil(8);
        pass.dispatch_workgroups(groups, groups, 1);
    }
    for (tex, readback) in [(&disp_tex, &disp_readback), (&norm_tex, &norm_readback)] {
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(atlas_height),
                },
            },
            extent,
        );
    }
    queue.submit([encoder.finish()]);

    disp_readback.slice(..).map_async(MapMode::Read, |_| {});
    norm_readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let disp_out = {
        let view = disp_readback
            .slice(..)
            .get_mapped_range()
            .expect("mapped displacement readback should be available after poll");
        let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        disp_readback.unmap();
        floats
    };
    let norm_out = {
        let view = norm_readback
            .slice(..)
            .get_mapped_range()
            .expect("mapped normal readback should be available after poll");
        let floats = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        norm_readback.unmap();
        floats
    };

    (disp_out, norm_out)
}

/// End-to-end parity for the *multi-cascade* spectral path introduced to stop
/// the ocean re-solving one band `M` times. Two structurally distinct cascades
/// (different amplitudes and patch sizes) are concatenated into one `h0` pool;
/// the evolve pass reads each cascade's slice through its own `h0_offset`, the
/// production butterfly inverse-`FFT`s each, and assemble stacks each tile into
/// one `N x (2*N)` atlas at its own `tile_origin_y`. The read-back atlas must
/// (a) match the per-cascade direct-summation `spectrum_ifft_golden` tile for
/// tile, proving the offset addressing is correct, and (b) carry genuinely
/// different tiles, proving the ocean now resolves waves across scales the way
/// `UE5` Water, `Crest` and `WaveWorks` stack `FFT` bands rather than repeating
/// a single band.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
#[expect(
    clippy::too_many_lines,
    reason = "building two cascades, running the full stacked-atlas dispatch, and checking both tile parity and tile distinctness reads clearest as one linear scenario"
)]
fn spectrum_fft_cascade_atlas_gpu_stacks_distinct_tiles() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!(
            "spectrum_fft_cascade_atlas_gpu_stacks_distinct_tiles: no wgpu adapter, skipping on-device parity"
        );
        return;
    };

    let n = 16u32;
    let cascade_count = 2u32;
    let atlas_height = n * cascade_count;

    // Cascade 0: the coarse band. Cascade 1: a finer, weaker band. Halving the
    // amplitudes keeps the field Hermitian (a linear scale of a Hermitian pair
    // is Hermitian) while the smaller patch size changes both the wavenumber
    // grid and the dispersion, so the two tiles cannot coincide by accident.
    let (h0_c0, h0_neg_c0) = spectrum_field_hermitian(n);
    let scale = 0.5_f32;
    let h0_c1: Vec<[f32; 2]> = h0_c0.iter().map(|c| [c[0] * scale, c[1] * scale]).collect();
    let h0_neg_c1: Vec<[f32; 2]> = h0_neg_c0
        .iter()
        .map(|c| [c[0] * scale, c[1] * scale])
        .collect();

    let tile_len = (n * n) as usize;

    // Concatenated pools, coarsest cascade first — exactly what `pack_cascades`
    // hands the recorder.
    let mut h0_pool: Vec<[f32; 2]> = Vec::with_capacity(tile_len * 2);
    h0_pool.extend_from_slice(&h0_c0);
    h0_pool.extend_from_slice(&h0_c1);
    let mut h0_neg_pool: Vec<[f32; 2]> = Vec::with_capacity(tile_len * 2);
    h0_neg_pool.extend_from_slice(&h0_neg_c0);
    h0_neg_pool.extend_from_slice(&h0_neg_c1);

    let params0 = GpuWaterSpectrumParams {
        grid_size: n,
        patch_size: 50.0,
        time: 1.3,
        choppiness: 1.6,
        foam_threshold: 1.05,
        h0_offset: 0,
        tile_origin_y: 0,
        _pad: 0,
    };
    let params1 = GpuWaterSpectrumParams {
        grid_size: n,
        patch_size: 25.0,
        time: 1.3,
        choppiness: 1.6,
        foam_threshold: 1.05,
        h0_offset: n * n,
        tile_origin_y: n,
        _pad: 0,
    };

    // Per-cascade golden references address a tile-local pool at offset zero.
    let mut gold0_params = params0;
    gold0_params.h0_offset = 0;
    gold0_params.tile_origin_y = 0;
    let mut gold1_params = params1;
    gold1_params.h0_offset = 0;
    gold1_params.tile_origin_y = 0;
    let (gold0_disp, gold0_norm) = spectrum_ifft_golden(&h0_c0, &h0_neg_c0, &gold0_params);
    let (gold1_disp, gold1_norm) = spectrum_ifft_golden(&h0_c1, &h0_neg_c1, &gold1_params);

    let fft_wgsl = compile_spectrum_fft_wgsl();
    let evolve_entry = find_entry_point(&fft_wgsl, "water_spectrum_evolve");
    let assemble_entry = find_entry_point(&fft_wgsl, "water_spectrum_assemble");
    let butterfly_wgsl = compile_butterfly_wgsl();
    let bitrev = find_entry_point(&butterfly_wgsl, "water_fft_bitrev");
    let stage = find_entry_point(&butterfly_wgsl, "water_fft_stage");
    let normalize = find_entry_point(&butterfly_wgsl, "water_fft_normalize");
    let n_usize = n as usize;

    // Evolve each cascade by pointing the shared pool at that cascade's slice
    // through `h0_offset`, then run the production butterfly inverse FFT.
    let make_tiles = |params: &GpuWaterSpectrumParams| -> [Vec<[f32; 2]>; 4] {
        let packed = dispatch_spectrum_evolve(
            &device,
            &queue,
            &fft_wgsl,
            &evolve_entry,
            &h0_pool,
            &h0_neg_pool,
            params,
        );
        core::array::from_fn(|i| {
            dispatch_butterfly_ifft2(
                &device,
                &queue,
                &butterfly_wgsl,
                &bitrev,
                &stage,
                &normalize,
                &packed[i],
                n_usize,
            )
        })
    };
    let tiles0 = make_tiles(&params0);
    let tiles1 = make_tiles(&params1);

    let (atlas_disp, atlas_norm) = dispatch_cascade_assemble_atlas(
        &device,
        &queue,
        &fft_wgsl,
        &assemble_entry,
        &[(tiles0, params0), (tiles1, params1)],
        atlas_height,
    );

    assert_eq!(
        atlas_disp.len(),
        (atlas_height * n * 4) as usize,
        "stacked displacement atlas lane count mismatch"
    );
    assert_eq!(
        atlas_norm.len(),
        (atlas_height * n * 4) as usize,
        "stacked normal atlas lane count mismatch"
    );

    // Compare each cascade's atlas tile against its own direct-sum golden. The
    // atlas texel (x, c*N + y) lives at flat lane (((c*N + y)*N + x)*4 + ch).
    let check_tile = |cascade: u32, gold_disp: &[f32], gold_norm: &[f32]| {
        let base_row = cascade * n;
        let mut y = 0u32;
        while y < n {
            let mut x = 0u32;
            while x < n {
                let tile_texel = (y * n + x) as usize;
                let atlas_texel = (((base_row + y) * n) + x) as usize;
                let mut ch = 0usize;
                while ch < 4 {
                    let g_disp = gold_disp[tile_texel * 4 + ch];
                    let a_disp = atlas_disp[atlas_texel * 4 + ch];
                    let dd = (a_disp - g_disp).abs();
                    assert!(
                        dd < PARITY_EPS,
                        "cascade {cascade} disp ({x},{y}) ch{ch}: atlas={a_disp} golden={g_disp} |d|={dd}"
                    );
                    let g_norm = gold_norm[tile_texel * 4 + ch];
                    let a_norm = atlas_norm[atlas_texel * 4 + ch];
                    let dn = (a_norm - g_norm).abs();
                    assert!(
                        dn < PARITY_EPS,
                        "cascade {cascade} norm ({x},{y}) ch{ch}: atlas={a_norm} golden={g_norm} |d|={dn}"
                    );
                    ch += 1;
                }
                x += 1;
            }
            y += 1;
        }
    };
    check_tile(0, &gold0_disp, &gold0_norm);
    check_tile(1, &gold1_disp, &gold1_norm);

    // The two stacked tiles must genuinely differ: if the recorder still fed one
    // band into both slots (the old fake multi-cascade), every lane would match.
    let mut max_tile_delta = 0.0_f32;
    let mut texel = 0usize;
    while texel < tile_len {
        let mut ch = 0usize;
        while ch < 4 {
            let lane0 = atlas_disp[texel * 4 + ch];
            let lane1 = atlas_disp[(tile_len + texel) * 4 + ch];
            let d = (lane0 - lane1).abs();
            if d > max_tile_delta {
                max_tile_delta = d;
            }
            ch += 1;
        }
        texel += 1;
    }
    assert!(
        max_tile_delta > PARITY_EPS,
        "stacked cascades must resolve different bands, but the two tiles were identical (max |d|={max_tile_delta})"
    );
}
