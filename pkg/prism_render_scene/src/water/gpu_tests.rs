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

use super::abi::GpuFlipSimParams;

/// Fixed-point domain constants mirrored from `water_flip.wesl` so the host can
/// pack the momentum/mass scatter buffer with the exact bit pattern the kernel
/// decodes. Kept private to this parity block.
const FLIP_FIXED_SCALE: f32 = 65536.0;
/// Clamp bound preventing the signed accumulator from overflowing `i32`.
const FLIP_FIXED_LIMIT: f32 = 30000.0;
/// "Effectively zero" threshold shared with the shader's mass/count guards.
const FLIP_EPS: f32 = 1.0e-6;

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
