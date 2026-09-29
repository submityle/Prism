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
