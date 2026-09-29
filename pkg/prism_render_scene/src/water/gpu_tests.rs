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
