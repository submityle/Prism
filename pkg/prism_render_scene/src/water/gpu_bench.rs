//! Real-device `GPU` timing for the water compute kernels.
//!
//! The sibling [`gpu_tests`](super::gpu_tests) module proves the kernels compute
//! the *right* numbers; this module proves they do so within a *measured* time
//! envelope. The architecture crate's kernel contract
//! ([`prism_render_architecture::water::kernels`]) documents every workgroup
//! tile as a **design target, not a measured value** — a fair label while the
//! `GPU` backend was still offline. The backend is online now (the parity suite
//! dispatches real `Metal`/`Vulkan`/`DX12` compute), so this module closes that
//! gap: it wraps a kernel's compute pass in a two-slot timestamp
//! [`wgpu::QuerySet`], resolves the ticks the device wrote at pass begin/end,
//! scales them by [`wgpu::Queue::get_timestamp_period`], and reports the
//! kernel's **measured** on-device microseconds.
//!
//! The harness is deliberately conservative:
//!
//! * It only needs [`wgpu::Features::TIMESTAMP_QUERY`] (pass-boundary writes),
//!   not the rarer inside-pass/inside-encoder variants, so it runs on any
//!   backend that reports the base timestamp feature and skips with a printed
//!   notice otherwise (headless `CI`, or an adapter without the feature).
//! * It warms the pipeline once (shader upload, first-use compilation, cache
//!   population) before the timed runs, then reports the **median** of several
//!   iterations so a single scheduling hiccup cannot skew the figure.
//! * It asserts only that the measured time is finite, strictly positive, and
//!   under a generous ceiling — a real dispatch always takes a positive,
//!   bounded time, and a zero or absurd figure means the timestamps did not
//!   land. It does not hard-code a target microsecond count, because the honest
//!   budget is the number the device reports, not a guess baked into a test.
//!
//! The measured figures are printed (so they can be captured and used to
//! re-tune the design-target tiles), while the assertions keep the run green
//! and self-checking on any machine with a timestamp-capable device.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePassTimestampWrites,
    ComputePipelineDescriptor, DeviceDescriptor, Extent3d, Features, Instance, InstanceDescriptor,
    InstanceFlags, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType,
    QuerySet, QuerySetDescriptor, QueryType, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor,
};

use super::abi::{
    GpuFlipParticle, GpuFlipSimParams, GpuMacParams, GpuPbfParams, GpuWaterFoamParams,
    GpuWaterSpectrumParams, GpuWaterSweParams,
};

/// Generous upper bound (microseconds) for a single water compute pass over the
/// benchmarked tile sizes. A real dispatch over an `N <= 512` grid completes far
/// under this on any modern `GPU`; the ceiling only catches a pathological
/// measurement (wrong tick scaling, a stalled device), never a healthy run.
const MAX_PASS_MICROS: f64 = 250_000.0;

/// Grid resolutions the spectral kernels are timed at. `64` is a coarse cascade,
/// `256` a production ocean tile; timing both shows how the pass scales with the
/// `N x N` domain the shader launches over.
const BENCH_GRIDS: [u32; 3] = [64, 128, 256];

/// Warm-up dispatches (untimed) before the measured runs, so first-use pipeline
/// compilation and cache population never land inside a timed sample.
const WARMUP_RUNS: u32 = 2;

/// Timed dispatches whose median is reported as the kernel's measured budget.
const TIMED_RUNS: usize = 16;

/// Identity `WESL` -> `Wgsl` shuttle for the render-world [`ShaderCache`]: the
/// water shaders are authored in `WESL` and the cache hands back their `Wgsl`
/// translation, which is what `wgpu` consumes.
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

/// Compiles `water_spectrum_fft.wesl` and returns its `Wgsl` translation.
fn compile_spectrum_fft_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_4245_4e43_4846_4654_0001),
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

/// Finds the compute entry point whose name contains `needle` in `wgsl`.
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

/// Best-effort acquisition of a native compute device that can write timestamp
/// queries. Returns `None` (rather than failing) when there is no adapter or the
/// adapter lacks [`Features::TIMESTAMP_QUERY`], so the suite stays green on hosts
/// that cannot measure while still running the full timed path on any device
/// that can (for example an `Apple` `M`-series `GPU`).
fn try_timing_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
    if !adapter.features().contains(Features::TIMESTAMP_QUERY) {
        return None;
    }
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("water_gpu_bench_device"),
        required_features: Features::TIMESTAMP_QUERY,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Builds a deterministic Hermitian-symmetric initial spectrum for an `N x N`
/// tile: `h0[m*N + n]` carries a bounded complex amplitude and `h0_neg` mirrors
/// it, so the evolve pass has real work to do at every texel. Timing does not
/// depend on the exact amplitudes, only on the full domain being exercised.
fn hermitian_field(n: u32) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
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
                0.02 * bevy_math::ops::sin(0.5 * fm + 0.3 * fn_),
                0.02 * bevy_math::ops::cos(0.4 * fm - 0.2 * fn_),
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
            let mirror = (((n - m) % n) * n + ((n - nn) % n)) as usize;
            h0_neg[idx] = h0[mirror];
            nn += 1;
        }
        m += 1;
    }
    (h0, h0_neg)
}

/// A two-slot timestamp query set plus the resolve/read buffers needed to pull
/// the pass-boundary ticks back to the host.
struct PassTimer {
    query_set: QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
}

impl PassTimer {
    fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&QuerySetDescriptor {
            label: Some("water_bench_timestamps"),
            ty: QueryType::Timestamp,
            count: 2,
        });
        let size = 2 * size_of::<u64>() as u64;
        let resolve = device.create_buffer(&BufferDescriptor {
            label: Some("water_bench_ts_resolve"),
            size,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&BufferDescriptor {
            label: Some("water_bench_ts_read"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve,
            read,
        }
    }

    /// Timestamp-writes descriptor that brackets a compute pass with this set's
    /// two slots.
    fn writes(&self) -> ComputePassTimestampWrites<'_> {
        ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }
    }

    /// Reads back the two resolved ticks and converts their difference to
    /// microseconds using the queue's tick period (nanoseconds per tick).
    fn elapsed_micros(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> f64 {
        self.read.slice(..).map_async(MapMode::Read, |_| {});
        device
            .poll(PollType::wait_indefinitely())
            .expect("device poll should complete the resolved-timestamp copy");
        let ticks = {
            let view = self
                .read
                .slice(..)
                .get_mapped_range()
                .expect("mapped timestamp readback should be available after poll");
            let raw = bytemuck::cast_slice::<u8, u64>(&view).to_vec();
            drop(view);
            self.read.unmap();
            raw
        };
        let delta = ticks[1].saturating_sub(ticks[0]);
        let period_ns = f64::from(queue.get_timestamp_period());
        (delta as f64) * period_ns / 1_000.0
    }
}

/// Median of a small sample (sorts a copy; `TIMED_RUNS` is tiny).
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("timings are finite"));
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    }
}

/// Times the packed `water_spectrum_evolve` pass over an `N x N` tile and
/// returns the median measured microseconds across [`TIMED_RUNS`] runs.
fn measure_evolve(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    n: u32,
) -> f64 {
    let (h0, h0_neg) = hermitian_field(n);
    let cell_count = (n * n) as usize;
    let byte_len = (cell_count * size_of::<[f32; 2]>()) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bench_evolve"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bench_evolve"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let params = GpuWaterSpectrumParams {
        grid_size: n,
        patch_size: 250.0,
        time: 1.3,
        choppiness: 1.4,
        foam_threshold: 1.05,
        h0_offset: 0,
        tile_origin_y: 0,
        _pad: 0,
    };
    let h0_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_evolve_h0"),
        contents: bytemuck::cast_slice(&h0),
        usage: BufferUsages::STORAGE,
    });
    let h0_neg_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_evolve_h0_neg"),
        contents: bytemuck::cast_slice(&h0_neg),
        usage: BufferUsages::STORAGE,
    });
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_evolve_params"),
        contents: bytemuck::bytes_of(&params),
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
    let g = [make_g("g0"), make_g("g1"), make_g("g2"), make_g("g3")];

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("bench_evolve_bind_group"),
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
                resource: g[0].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: g[1].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: g[2].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: g[3].as_entire_binding(),
            },
        ],
    });
    let groups = n.div_ceil(8);
    let timer = PassTimer::new(device);
    time_dispatch(
        device,
        queue,
        &pipeline,
        &bind_group,
        groups,
        groups,
        &timer,
    )
}

/// Times the `water_spectrum_assemble` pass over an `N x N` tile and returns the
/// median measured microseconds across [`TIMED_RUNS`] runs.
fn measure_assemble(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    n: u32,
) -> f64 {
    let cell_count = (n * n) as usize;
    let grid: Vec<[f32; 2]> = (0..cell_count)
        .map(|i| {
            let f = i as f32;
            [
                0.01 * bevy_math::ops::sin(f * 0.017),
                0.01 * bevy_math::ops::cos(f * 0.013),
            ]
        })
        .collect();

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bench_assemble"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bench_assemble"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let params = GpuWaterSpectrumParams {
        grid_size: n,
        patch_size: 250.0,
        time: 1.3,
        choppiness: 1.4,
        foam_threshold: 1.05,
        h0_offset: 0,
        tile_origin_y: 0,
        _pad: 0,
    };
    let params_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_assemble_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let g_bufs: [wgpu::Buffer; 4] = core::array::from_fn(|_| {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("bench_assemble_g"),
            contents: bytemuck::cast_slice(&grid),
            usage: BufferUsages::STORAGE,
        })
    });

    let extent = Extent3d {
        width: n,
        height: n,
        depth_or_array_layers: 1,
    };
    let make_tex = |label: &str| {
        device
            .create_texture(&TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default())
    };
    let disp_view = make_tex("bench_assemble_disp");
    let norm_view = make_tex("bench_assemble_norm");

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("bench_assemble_bind_group"),
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
    let groups = n.div_ceil(8);
    let timer = PassTimer::new(device);
    time_dispatch(
        device,
        queue,
        &pipeline,
        &bind_group,
        groups,
        groups,
        &timer,
    )
}

/// Warms then times a single compute dispatch, returning the median measured
/// microseconds. The pass is bracketed by `timer`'s two timestamp slots, the
/// set is resolved into a host-readable buffer, and the tick delta is scaled by
/// the queue period.
fn time_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    groups_x: u32,
    groups_y: u32,
    timer: &PassTimer,
) -> f64 {
    // The common case: the kernel declares its resources on `group(0)`.
    time_dispatch_at_group(
        device, queue, pipeline, bind_group, 0, groups_x, groups_y, timer,
    )
}

/// Warms then times a single compute dispatch whose single bind group is set
/// at `group_index`. Kernels that declare their resources on a non-zero group
/// (for example `water_foam_advect` on `@group(1)`) bind there while the
/// auto-derived lower groups stay empty and unreferenced. Otherwise identical
/// to [`time_dispatch`]: warm up untimed, then report the median of the timed
/// runs bracketed by `timer`'s two timestamp slots.
fn time_dispatch_at_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    group_index: u32,
    groups_x: u32,
    groups_y: u32,
    timer: &PassTimer,
) -> f64 {
    // Untimed warm-up so first-use compilation never lands in a sample.
    for _ in 0..WARMUP_RUNS {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_warmup_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bench_warmup_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(group_index, bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        queue.submit([encoder.finish()]);
    }
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should drain the warm-up submissions");

    let mut samples = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_timed_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bench_timed_pass"),
                timestamp_writes: Some(timer.writes()),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(group_index, bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        encoder.resolve_query_set(&timer.query_set, 0..2, &timer.resolve, 0);
        encoder.copy_buffer_to_buffer(&timer.resolve, 0, &timer.read, 0, timer.resolve.size());
        queue.submit([encoder.finish()]);
        samples.push(timer.elapsed_micros(device, queue));
    }
    median(samples)
}

/// Measures the ocean spectral kernels (`water_spectrum_evolve` and
/// `water_spectrum_assemble`) on a real device across [`BENCH_GRIDS`] and
/// asserts each pass takes a finite, strictly positive, bounded time. This
/// turns the architecture crate's "design target, not a measured value" note
/// into an actual on-device measurement for the flagship ocean path; the
/// printed medians are the numbers the design-target tiles should be re-tuned
/// against.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the measured budget (and the skip notice) must reach the test log so it can be captured"
)]
fn spectrum_pass_gpu_budget_is_measured() {
    let Some((device, queue)) = try_timing_device() else {
        eprintln!(
            "spectrum_pass_gpu_budget_is_measured: no timestamp-capable wgpu adapter, skipping on-device timing"
        );
        return;
    };

    let wgsl = compile_spectrum_fft_wgsl();
    let evolve_entry = find_entry_point(&wgsl, "water_spectrum_evolve");
    let assemble_entry = find_entry_point(&wgsl, "water_spectrum_assemble");

    for n in BENCH_GRIDS {
        let evolve_us = measure_evolve(&device, &queue, &wgsl, &evolve_entry, n);
        let assemble_us = measure_assemble(&device, &queue, &wgsl, &assemble_entry, n);
        eprintln!(
            "water spectral budget @ {n}x{n}: evolve = {evolve_us:.2} us, assemble = {assemble_us:.2} us (measured, median of {TIMED_RUNS})"
        );
        for (label, us) in [("evolve", evolve_us), ("assemble", assemble_us)] {
            assert!(
                us.is_finite() && us > 0.0 && us < MAX_PASS_MICROS,
                "{label} @ {n}x{n} measured {us} us is not a healthy bounded timing"
            );
        }
    }
}

/// Compiles `water_surface.wesl` and returns its `Wgsl` translation. The
/// shallow-water (`SWE`) step and the foam advection kernels both live in this
/// module, so the surface benches share one compiled translation.
fn compile_surface_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_4245_4e43_4842_5357_0001),
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

/// Builds a deterministic, non-trivial shallow-water state for an `N x N` tile:
/// a raised algebraic bump over an otherwise still `1 m` sheet with sheared
/// velocities, plus a couple of interaction sources. This exercises every
/// branch of the step (interior flux, reflective walls, pressure gradient,
/// upwind advection, damping, source injection) so the timed dispatch does the
/// same work a production frame would. Timing does not depend on the exact
/// amplitudes, only on the full domain being exercised.
fn swe_bench_state(n: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<[f32; 4]>) {
    let nx = n as usize;
    let nz = n as usize;
    let count = nx * nz;
    let mut h = vec![0.0_f32; count];
    let mut u = vec![0.0_f32; count];
    let mut v = vec![0.0_f32; count];
    let cx = (nx as f32 - 1.0) * 0.5;
    let cz = (nz as f32 - 1.0) * 0.5;
    let mut z = 0usize;
    while z < nz {
        let mut x = 0usize;
        while x < nx {
            let i = z * nx + x;
            let fx = x as f32 - cx;
            let fz = z as f32 - cz;
            let r2 = fx * fx + fz * fz;
            let bump = (1.0 - r2 * 0.002).max(0.0);
            h[i] = 1.0 + 0.5 * bump;
            u[i] = 0.005 * fx;
            v[i] = -0.004 * fz;
            x += 1;
        }
        z += 1;
    }
    let mut sources = vec![[0.0_f32; 4]; count];
    let center = (nz / 2) * nx + (nx / 2);
    sources[center] = [0.2, 0.0, 0.0, 0.0];
    let off = (nz / 4) * nx + (nx / 4);
    sources[off] = [0.0, 0.15, -0.1, 0.0];
    (h, u, v, sources)
}

/// Times one `water_swe_step` pass over an `N x N` tile and returns the median
/// measured microseconds across [`TIMED_RUNS`] runs. The eight bindings are
/// assembled from the pipeline's reflected `group(0)` layout in the exact order
/// the sibling parity test (`gpu_tests::dispatch_swe`) uses, so the timed pass
/// is the same dispatch the parity suite already proved numerically faithful.
fn measure_swe_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    n: u32,
) -> f64 {
    let (h, u, v, sources) = swe_bench_state(n);
    let scalar_bytes = (h.len() * size_of::<f32>()) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bench_swe_step"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bench_swe_step"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let storage_read = BufferUsages::STORAGE;
    let h_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_swe_h_in"),
        contents: bytemuck::cast_slice(&h),
        usage: storage_read,
    });
    let u_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_swe_u_in"),
        contents: bytemuck::cast_slice(&u),
        usage: storage_read,
    });
    let v_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_swe_v_in"),
        contents: bytemuck::cast_slice(&v),
        usage: storage_read,
    });
    let make_out = |label: &str| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: scalar_bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let h_out = make_out("bench_swe_h_out");
    let u_out = make_out("bench_swe_u_out");
    let v_out = make_out("bench_swe_v_out");
    let src_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_swe_sources"),
        contents: bytemuck::cast_slice(&sources),
        usage: storage_read,
    });

    let params = GpuWaterSweParams {
        nx: n,
        nz: n,
        dx: 0.5,
        gravity: 9.81,
        damping: 0.2,
        dt: 0.016,
        cfl_number: 0.5,
        // A fixed positive signal speed keeps the explicit step inside its
        // `CFL` bound without needing the host `swe::max_wave_speed` scan; the
        // kernel's branch work is identical for any finite positive value.
        max_wave_speed: 4.0,
    };
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_swe_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("bench_swe_bind_group"),
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

    let groups = n.div_ceil(8);
    let timer = PassTimer::new(device);
    time_dispatch(
        device,
        queue,
        &pipeline,
        &bind_group,
        groups,
        groups,
        &timer,
    )
}

/// Measures the shallow-water step kernel (`water_swe_step`) on a real device
/// across [`BENCH_GRIDS`] and asserts each pass takes a finite, strictly
/// positive, bounded time. The design doc (`docs/prism_water_engine_design_zh.md`
/// §12) budgets `SWE` height-field stepping at `<= 0.3 ms/domain` as a **design
/// target, not a measured value**; this turns that line into an actual
/// on-device measurement. The printed medians are the numbers the design-target
/// budget should be re-tuned against.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the measured budget (and the skip notice) must reach the test log so it can be captured"
)]
fn swe_step_pass_gpu_budget_is_measured() {
    let Some((device, queue)) = try_timing_device() else {
        eprintln!(
            "swe_step_pass_gpu_budget_is_measured: no timestamp-capable wgpu adapter, skipping on-device timing"
        );
        return;
    };

    let wgsl = compile_surface_wgsl();
    let entry = find_entry_point(&wgsl, "swe_step");

    for n in BENCH_GRIDS {
        let step_us = measure_swe_step(&device, &queue, &wgsl, &entry, n);
        eprintln!(
            "water SWE step budget @ {n}x{n}: step = {step_us:.2} us (measured, median of {TIMED_RUNS})"
        );
        assert!(
            step_us.is_finite() && step_us > 0.0 && step_us < MAX_PASS_MICROS,
            "SWE step @ {n}x{n} measured {step_us} us is not a healthy bounded timing"
        );
    }
}

// ===========================================================================
// Foam advection + decay (`water_foam_advect`, water_surface.wesl @group(1))
// ===========================================================================

/// Builds a deterministic, non-trivial foam step for an `N x N` tile: a raised
/// algebraic coverage bump over a low ambient sheet, a sheared surface-flow
/// field (so the semi-Lagrangian backtrace lands on fractional cells and
/// exercises the bilinear resample), and two additive reactive sources. This
/// drives every branch of the step (advection, flow-aware decay, source
/// injection, `0..=1` clamp) so the timed dispatch does the same work a
/// production frame would. Timing depends only on the full domain being
/// exercised, not on the exact amplitudes. Mirrors the shape of the parity
/// suite's `gpu_tests::build_initial_foam`.
fn foam_bench_state(n: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let nx = n as usize;
    let nz = n as usize;
    let count = nx * nz;
    let mut density = vec![0.0_f32; count];
    let mut u = vec![0.0_f32; count];
    let mut v = vec![0.0_f32; count];
    let cx = (nx as f32 - 1.0) * 0.5;
    let cz = (nz as f32 - 1.0) * 0.5;
    let mut z = 0usize;
    while z < nz {
        let mut x = 0usize;
        while x < nx {
            let i = z * nx + x;
            let fx = x as f32 - cx;
            let fz = z as f32 - cz;
            let r2 = fx * fx + fz * fz;
            let bump = (1.0 - r2 * 0.002).max(0.0);
            density[i] = 0.15 + 0.7 * bump;
            u[i] = 0.3 + 0.004 * fx;
            v[i] = -0.2 + 0.003 * fz;
            x += 1;
        }
        z += 1;
    }
    let mut sources = vec![0.0_f32; count];
    sources[(nz / 2) * nx + (nx / 2)] = 0.4;
    sources[(nz / 4) * nx + (nx / 4)] = 0.25;
    (density, u, v, sources)
}

/// Times one `water_foam_advect` pass over an `N x N` tile and returns the
/// median measured microseconds across [`TIMED_RUNS`] runs. The kernel declares
/// its six resources on `@group(1)`, so the bind group is built from the
/// pipeline's reflected `group(1)` layout and set at binding index `1` (the
/// auto-derived `group(0)` is empty and referenced by nothing). The binding
/// order matches the shader declaration exactly, so the timed pass is the same
/// dispatch the sibling parity test (`gpu_tests::dispatch_foam`) already proved
/// numerically faithful.
fn measure_foam_advect(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    n: u32,
) -> f64 {
    let (density, u, v, sources) = foam_bench_state(n);
    let scalar_bytes = (density.len() * size_of::<f32>()) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bench_foam_advect"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bench_foam_advect"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let storage_read = BufferUsages::STORAGE;
    let foam_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_foam_in"),
        contents: bytemuck::cast_slice(&density),
        usage: storage_read,
    });
    let foam_u = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_foam_u"),
        contents: bytemuck::cast_slice(&u),
        usage: storage_read,
    });
    let foam_v = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_foam_v"),
        contents: bytemuck::cast_slice(&v),
        usage: storage_read,
    });
    let foam_sources = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_foam_sources"),
        contents: bytemuck::cast_slice(&sources),
        usage: storage_read,
    });
    let foam_out = device.create_buffer(&BufferDescriptor {
        label: Some("bench_foam_out"),
        size: scalar_bytes,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let params = GpuWaterFoamParams {
        nx: n,
        nz: n,
        dx: 0.5,
        dt: 0.016,
        base_decay: 0.8,
        persistence_floor: 0.1,
        reference_speed: 2.0,
        _pad: 0,
    };
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_foam_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(1);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("bench_foam_bind_group"),
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

    let groups = n.div_ceil(8);
    let timer = PassTimer::new(device);
    time_dispatch_at_group(
        device,
        queue,
        &pipeline,
        &bind_group,
        1,
        groups,
        groups,
        &timer,
    )
}

/// Measures the foam advection + decay kernel (`water_foam_advect`) on a real
/// device across [`BENCH_GRIDS`] and asserts each pass takes a finite, strictly
/// positive, bounded time. The design doc (`docs/prism_water_engine_design_zh.md`
/// §12) budgets dynamic foam advection at `<= 0.3 ms` as a **design target, not
/// a measured value**; this turns that line into an actual on-device
/// measurement. The printed medians are the numbers the design-target budget
/// should be re-tuned against.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the measured budget (and the skip notice) must reach the test log so it can be captured"
)]
fn foam_advect_pass_gpu_budget_is_measured() {
    let Some((device, queue)) = try_timing_device() else {
        eprintln!(
            "foam_advect_pass_gpu_budget_is_measured: no timestamp-capable wgpu adapter, skipping on-device timing"
        );
        return;
    };

    let wgsl = compile_surface_wgsl();
    let entry = find_entry_point(&wgsl, "foam_advect");

    for n in BENCH_GRIDS {
        let advect_us = measure_foam_advect(&device, &queue, &wgsl, &entry, n);
        eprintln!(
            "water foam advect budget @ {n}x{n}: advect = {advect_us:.2} us (measured, median of {TIMED_RUNS})"
        );
        assert!(
            advect_us.is_finite() && advect_us > 0.0 && advect_us < MAX_PASS_MICROS,
            "foam advect @ {n}x{n} measured {advect_us} us is not a healthy bounded timing"
        );
    }
}

/// Compiles `water_pbf.wesl` and returns its `Wgsl` translation. The
/// position-based-fluids density solve (`water_pbf_density_solve`) lives in this
/// module; the bench shares one compiled translation with any sibling pass.
fn compile_pbf_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_4245_4e43_4850_4246_0001),
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

/// Particle-cube side lengths the `PBF` density solve is timed at. A side `s`
/// packs `s^3` particles (`16^3 = 4096`, `25^3 = 15625`, `40^3 = 64000`),
/// bracketing the design doc's "local ~1e5 particle" `PBF`/`FLIP` budget
/// (`docs/prism_water_engine_design_zh.md` §12). Timing all three shows how the
/// per-particle solve scales with the particle population and its neighbourhood
/// density.
const PBF_BENCH_SIDES: [u32; 3] = [16, 25, 40];

/// Packs a deterministic particle set into the shader's single `hash` storage
/// buffer. The layout mirrors `water_pbf.wesl` (and the parity test's
/// `build_pbf_hash`): `hash[2*c]` is cell `c`'s start offset into the index
/// region, `hash[2*c + 1]` its particle count, and the index region
/// (`hash[2*cell_count + start + s]`) lists the binned particle indices in
/// cell-major, ascending-index order. Binning here (rather than reaching into
/// the architecture crate) keeps the bench self-contained; the particle layout
/// is chosen so every cell the solve gathers is populated, so the timed pass
/// does the same 27-cell neighbourhood work a production frame would.
fn pbf_bench_hash(
    positions: &[[f32; 4]],
    origin: [f32; 3],
    cell_size: f32,
    nx: u32,
    ny: u32,
    nz: u32,
) -> Vec<u32> {
    let cell_count = (nx * ny * nz) as usize;
    let mut cells: Vec<Vec<u32>> = vec![Vec::new(); cell_count];
    let mut p = 0usize;
    while p < positions.len() {
        let pos = positions[p];
        let lx = pos[0] - origin[0];
        let ly = pos[1] - origin[1];
        let lz = pos[2] - origin[2];
        if lx >= 0.0 && ly >= 0.0 && lz >= 0.0 {
            let cx = (lx / cell_size) as u32;
            let cy = (ly / cell_size) as u32;
            let cz = (lz / cell_size) as u32;
            if cx < nx && cy < ny && cz < nz {
                let flat = (cz * nx * ny + cy * nx + cx) as usize;
                cells[flat].push(p as u32);
            }
        }
        p += 1;
    }
    let mut hash = vec![0u32; 2 * cell_count];
    let mut index_region: Vec<u32> = Vec::new();
    let mut flat = 0usize;
    while flat < cell_count {
        hash[2 * flat] = index_region.len() as u32;
        hash[2 * flat + 1] = cells[flat].len() as u32;
        for &idx in &cells[flat] {
            index_region.push(idx);
        }
        flat += 1;
    }
    hash.extend_from_slice(&index_region);
    hash
}

/// Builds a deterministic, compressed `PBF` particle cube of `side^3` particles
/// plus its spatial hash and solve parameters. Particles sit on a lattice at
/// half the smoothing radius, so every particle is over-dense (a non-trivial
/// constraint) and its 27-cell neighbourhood is populated — the solve walks the
/// same heavy gather a production incompressibility step would. The grid is
/// sized with a one-cell margin so no particle lands on or past the far
/// boundary. The `w` lane carries a distinct per-particle payload. Timing does
/// not depend on the exact positions, only on the full population being
/// exercised with real neighbours.
fn pbf_bench_state(side: u32) -> (Vec<[f32; 4]>, Vec<u32>, GpuPbfParams) {
    let cell_size = 1.0_f32;
    let spacing = 0.5_f32;
    let origin = [0.0_f32, 0.0, 0.0];
    let s = side as usize;
    let count = s * s * s;
    let base = 0.25_f32;

    let mut positions: Vec<[f32; 4]> = Vec::with_capacity(count);
    let mut z = 0usize;
    while z < s {
        let mut y = 0usize;
        while y < s {
            let mut x = 0usize;
            while x < s {
                let px = base + x as f32 * spacing;
                let py = base + y as f32 * spacing;
                let pz = base + z as f32 * spacing;
                let w = positions.len() as f32 + 0.5;
                positions.push([px, py, pz, w]);
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }

    let max_coord = base + (side as f32 - 1.0) * spacing;
    let cells_per_axis = (max_coord / cell_size) as u32 + 2;
    let grid_nx = cells_per_axis;
    let grid_ny = cells_per_axis;
    let grid_nz = cells_per_axis;

    let hash = pbf_bench_hash(&positions, origin, cell_size, grid_nx, grid_ny, grid_nz);

    let params = GpuPbfParams {
        grid_origin: origin,
        cell_size,
        rest_density: 20.0,
        particle_mass: 1.0,
        smoothing_radius: cell_size,
        relaxation_epsilon: 0.01,
        artificial_pressure_k: 0.1,
        artificial_pressure_delta_q: 0.2,
        artificial_pressure_n: 4,
        particle_count: count as u32,
        grid_nx,
        grid_ny,
        grid_nz,
        _pad: 0,
    };
    (positions, hash, params)
}

/// Times the two-pass `PBF` density solve over a `side^3` particle cube and
/// returns the median measured microseconds across [`TIMED_RUNS`] runs for each
/// pass as `(lambda_us, solve_us)`. Both passes share one explicit five-binding
/// `@group(0)` layout (`positions_in`, `positions_out`, `hash`, `params`,
/// `lambdas`): the `pbf_compute_lambda` pre-pass fills every particle's `XPBD`
/// scaling factor `lambda_i` once, then `water_pbf_density_solve` reads each
/// neighbour's cached `lambda_j` instead of re-deriving it — turning the former
/// O(n * k^2) neighbourhood blow-up into O(n * k). Each pass runs one invocation
/// per particle at `@workgroup_size(64)`, so each dispatch is `ceil(count / 64)`
/// workgroups — the same launch the sibling parity test
/// (`gpu_tests::dispatch_pbf`) already proved numerically faithful.
fn measure_pbf_density(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    side: u32,
) -> (f64, f64) {
    let (positions, hash, params) = pbf_bench_state(side);
    let vec4_bytes = (positions.len() * size_of::<[f32; 4]>()) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bench_pbf_density_solve"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });

    // One explicit five-binding group-0 layout shared by both passes. An
    // auto-derived layout would omit, per entry point, the bindings that entry
    // happens not to touch, so the lambda pre-pass and solve pass would end up
    // with incompatible subsets and could not share a single bind group.
    let storage = |read_only: bool| BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bench_pbf_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                ..storage(true)
            },
            BindGroupLayoutEntry {
                binding: 1,
                ..storage(false)
            },
            BindGroupLayoutEntry {
                binding: 2,
                ..storage(true)
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                ..storage(false)
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("bench_pbf_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        ..Default::default()
    });

    let lambda_entry = find_entry_point(wgsl, "pbf_compute_lambda");
    let lambda_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bench_pbf_compute_lambda"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&lambda_entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bench_pbf_density_solve"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let positions_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_pbf_positions_in"),
        contents: bytemuck::cast_slice(&positions),
        usage: BufferUsages::STORAGE,
    });
    let positions_out = device.create_buffer(&BufferDescriptor {
        label: Some("bench_pbf_positions_out"),
        size: vec4_bytes,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let hash_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_pbf_hash"),
        contents: bytemuck::cast_slice(&hash),
        usage: BufferUsages::STORAGE,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_pbf_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let lambdas = device.create_buffer(&BufferDescriptor {
        label: Some("bench_pbf_lambdas"),
        size: (positions.len() as u64) * size_of::<f32>() as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("bench_pbf_bind_group"),
        layout: &bind_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: positions_in.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: positions_out.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: hash_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: param_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: lambdas.as_entire_binding(),
            },
        ],
    });

    let groups = params.particle_count.div_ceil(64);
    // Time the lambda pre-pass first; its warm-up and timed runs leave the
    // shared `lambdas` buffer populated for the solve measurement that follows.
    let lambda_timer = PassTimer::new(device);
    let lambda_us = time_dispatch(
        device,
        queue,
        &lambda_pipeline,
        &bind_group,
        groups,
        1,
        &lambda_timer,
    );
    let solve_timer = PassTimer::new(device);
    let solve_us = time_dispatch(
        device,
        queue,
        &pipeline,
        &bind_group,
        groups,
        1,
        &solve_timer,
    );
    (lambda_us, solve_us)
}

/// Measures the `PBF` density-constraint solve (`water_pbf_density_solve`) on a
/// real device across [`PBF_BENCH_SIDES`] and asserts each pass takes a finite,
/// strictly positive, bounded time. The design doc
/// (`docs/prism_water_engine_design_zh.md` §12) budgets the `FLIP`/`PBF` sim at
/// `<= 2-4 ms` for a local `~1e5` particle domain as a **design target, not a
/// measured value**; this turns that line into an actual on-device measurement
/// of the solve kernel. The printed medians are the numbers the design-target
/// budget should be re-tuned against.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the measured budget (and the skip notice) must reach the test log so it can be captured"
)]
fn pbf_density_solve_pass_gpu_budget_is_measured() {
    let Some((device, queue)) = try_timing_device() else {
        eprintln!(
            "pbf_density_solve_pass_gpu_budget_is_measured: no timestamp-capable wgpu adapter, skipping on-device timing"
        );
        return;
    };

    let wgsl = compile_pbf_wgsl();
    let entry = find_entry_point(&wgsl, "pbf_density_solve");

    for side in PBF_BENCH_SIDES {
        let count = side * side * side;
        let (lambda_us, solve_us) = measure_pbf_density(&device, &queue, &wgsl, &entry, side);
        let total_us = lambda_us + solve_us;
        eprintln!(
            "water PBF density solve budget @ {count} particles (side {side}): lambda = {lambda_us:.2} us, solve = {solve_us:.2} us, total = {total_us:.2} us (measured, median of {TIMED_RUNS})"
        );
        assert!(
            lambda_us.is_finite() && lambda_us > 0.0 && lambda_us < MAX_PASS_MICROS,
            "pbf compute-lambda @ {count} particles measured {lambda_us} us is not a healthy bounded timing"
        );
        assert!(
            solve_us.is_finite() && solve_us > 0.0 && solve_us < MAX_PASS_MICROS,
            "pbf density solve @ {count} particles measured {solve_us} us is not a healthy bounded timing"
        );
    }
}

/// Cubic grid resolutions (`dim = [D, D, D]`) the staggered-`MAC` pressure
/// projection is timed at. `16^3 = 4096` cells is a coarse interactive splash,
/// `32^3 = 32768` cells (packing `~1e5` staggered faces) a production-scale
/// `FLIP` pressure domain — the `~1e5` figure the design doc
/// (`docs/prism_water_engine_design_zh.md` §12) budgets the `FLIP`/`PBF`
/// solver against. Timing all three resolutions shows how each projection
/// stage scales with the cell/face count the kernel launches over.
const MAC_BENCH_DIMS: [u32; 3] = [16, 24, 32];

/// Compiles `water_flip_mac.wesl` (the staggered-`MAC` projection kernels) and
/// returns its `Wgsl` translation, mirroring [`compile_pbf_wgsl`].
fn compile_flip_mac_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_4245_4e43_464d_4143_0001),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_flip_mac.wesl"),
            "embedded://prism_render_scene/shaders/water_flip_mac.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_flip_mac.wesl failed to compile: {error}"));
    (*module).clone()
}

/// `u`-face element count `(nx+1)*ny*nz`, mirroring `mac_u_count` in the shader.
fn mac_u_count(dim: [u32; 3]) -> u32 {
    (dim[0] + 1) * dim[1] * dim[2]
}

/// `v`-face element count `nx*(ny+1)*nz`.
fn mac_v_count(dim: [u32; 3]) -> u32 {
    dim[0] * (dim[1] + 1) * dim[2]
}

/// `w`-face element count `nx*ny*(nz+1)`.
fn mac_w_count(dim: [u32; 3]) -> u32 {
    dim[0] * dim[1] * (dim[2] + 1)
}

/// Total packed face count over the `[u | v | w]` blocks the shader indexes.
fn mac_face_count(dim: [u32; 3]) -> u32 {
    mac_u_count(dim) + mac_v_count(dim) + mac_w_count(dim)
}

/// Linear `u`-face index for `i in 0..=nx`, mirroring `mac_u_index`.
fn mac_u_index(i: u32, j: u32, k: u32, dim: [u32; 3]) -> u32 {
    (k * dim[1] + j) * (dim[0] + 1) + i
}

/// Linear `v`-face index for `j in 0..=ny`, mirroring `mac_v_index`.
fn mac_v_index(i: u32, j: u32, k: u32, dim: [u32; 3]) -> u32 {
    mac_u_count(dim) + (k * (dim[1] + 1) + j) * dim[0] + i
}

/// Linear `w`-face index for `k in 0..=nz`, mirroring `mac_w_index`.
fn mac_w_index(i: u32, j: u32, k: u32, dim: [u32; 3]) -> u32 {
    mac_u_count(dim) + mac_v_count(dim) + (k * dim[1] + j) * dim[0] + i
}

/// Seeds a staggered face field with a smooth low-frequency divergent flow plus
/// a strong interior checkerboard component, mirroring `seed_mac_faces` in the
/// parity suite. Boundary faces stay zero (the solid tank wall). Timing does not
/// depend on the exact values, only on every interior face carrying real work
/// for the divergence and projection sweeps to touch.
fn mac_bench_faces(dim: [u32; 3]) -> Vec<f32> {
    let mut faces = vec![0.0_f32; mac_face_count(dim) as usize];
    let checker = 0.7_f32;
    // u-faces.
    let mut k = 0u32;
    while k < dim[2] {
        let mut j = 0u32;
        while j < dim[1] {
            let mut i = 0u32;
            while i <= dim[0] {
                if i != 0 && i != dim[0] {
                    let smooth = 0.05 * (2.0 * i as f32 - j as f32 + 0.5 * k as f32);
                    let sign = if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 };
                    faces[mac_u_index(i, j, k, dim) as usize] = smooth + sign * checker;
                }
                i += 1;
            }
            j += 1;
        }
        k += 1;
    }
    // v-faces.
    k = 0;
    while k < dim[2] {
        let mut i = 0u32;
        while i < dim[0] {
            let mut j = 0u32;
            while j <= dim[1] {
                if j != 0 && j != dim[1] {
                    let smooth = 0.04 * (i as f32 - 2.0 * j as f32 + k as f32);
                    let sign = if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 };
                    faces[mac_v_index(i, j, k, dim) as usize] = smooth + sign * checker;
                }
                j += 1;
            }
            i += 1;
        }
        k += 1;
    }
    // w-faces.
    let mut j = 0u32;
    while j < dim[1] {
        let mut i = 0u32;
        while i < dim[0] {
            let mut kk = 0u32;
            while kk <= dim[2] {
                if kk != 0 && kk != dim[2] {
                    let smooth = 0.03 * (i as f32 + j as f32 - 2.0 * kk as f32);
                    let sign = if (i + j + kk) % 2 == 0 { 1.0 } else { -1.0 };
                    faces[mac_w_index(i, j, kk, dim) as usize] = smooth + sign * checker;
                }
                kk += 1;
            }
            i += 1;
        }
        j += 1;
    }
    faces
}

/// Builds the host-side [`GpuMacParams`] uniform for a cubic bench domain. The
/// scalar grid spacing is unit (`dx = inv_dx = 1`) and the damped-`Jacobi`
/// factor is a representative `0.8`; the measured dispatch time is independent
/// of these values, only of the grid dimensions driving the launch size.
fn mac_bench_params(dim: [u32; 3]) -> GpuMacParams {
    let cell_count = dim[0] * dim[1] * dim[2];
    GpuMacParams {
        dim: [dim[0], dim[1], dim[2], cell_count],
        inv_dx: 1.0,
        dx: 1.0,
        jacobi_omega: 0.8,
        _pad: 0.0,
    }
}

/// Warms then times a single compute dispatch over a three-dimensional
/// workgroup grid, bound at `group(0)`. The staggered-`MAC` `mac_divergence`
/// and `mac_pressure` kernels launch one invocation per cell on a
/// `workgroup_size(4, 4, 4)` tile, so they need the `z` dispatch dimension that
/// [`time_dispatch`] (fixed at `z = 1`) does not expose. Otherwise identical:
/// untimed warm-up, then the median of [`TIMED_RUNS`] timed runs bracketed by
/// `timer`'s two timestamp slots.
fn time_dispatch_3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    groups_x: u32,
    groups_y: u32,
    groups_z: u32,
    timer: &PassTimer,
) -> f64 {
    for _ in 0..WARMUP_RUNS {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_warmup_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bench_warmup_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, groups_z);
        }
        queue.submit([encoder.finish()]);
    }
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should drain the warm-up submissions");

    let mut samples = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("bench_timed_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("bench_timed_pass"),
                timestamp_writes: Some(timer.writes()),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, groups_z);
        }
        encoder.resolve_query_set(&timer.query_set, 0..2, &timer.resolve, 0);
        encoder.copy_buffer_to_buffer(&timer.resolve, 0, &timer.read, 0, timer.resolve.size());
        queue.submit([encoder.finish()]);
        samples.push(timer.elapsed_micros(device, queue));
    }
    median(samples)
}

/// Times the three staggered-`MAC` projection stages (`mac_divergence`,
/// `mac_pressure` one `Jacobi` sweep, `mac_project`) over a cubic `dim` domain,
/// returning their measured medians in microseconds. The three kernels share one
/// bind group layout (`faces`, `divergence`, `pressure_in`, `pressure_out`,
/// `mac_params` on `group(0)`); the cell-local stages dispatch over the
/// `4x4x4`-tiled grid while the face-local projection dispatches linearly over
/// the packed `[u | v | w]` faces at `workgroup_size(64)`.
fn measure_mac_projection(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    dim: [u32; 3],
) -> (f64, f64, f64) {
    let faces = mac_bench_faces(dim);
    let cell_count = (dim[0] * dim[1] * dim[2]) as usize;
    let cell_zeros = vec![0.0_f32; cell_count];
    let params = mac_bench_params(dim);

    let faces_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_mac_faces"),
        contents: bytemuck::cast_slice(&faces),
        usage: BufferUsages::STORAGE,
    });
    let divergence_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_mac_divergence"),
        contents: bytemuck::cast_slice(&cell_zeros),
        usage: BufferUsages::STORAGE,
    });
    let pressure_in = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_mac_pressure_in"),
        contents: bytemuck::cast_slice(&cell_zeros),
        usage: BufferUsages::STORAGE,
    });
    let pressure_out = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_mac_pressure_out"),
        contents: bytemuck::cast_slice(&cell_zeros),
        usage: BufferUsages::STORAGE,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_mac_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    // The three MAC entries each touch a different subset of the five group(0)
    // bindings (`mac_divergence` skips the pressure buffers, `mac_project` skips
    // `divergence`), so an auto-derived (`layout: None`) layout would differ per
    // entry and reject a shared five-entry bind group. Declare the full layout
    // explicitly so one bind group feeds all three pipelines; a WGSL entry is
    // allowed to bind a superset of what it reads.
    let storage_rw = BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let storage_ro = BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let uniform = BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let entry = |binding: u32, ty: BindingType| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bench_flip_mac_layout"),
        entries: &[
            entry(0, storage_rw),
            entry(1, storage_rw),
            entry(2, storage_ro),
            entry(3, storage_rw),
            entry(4, uniform),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("bench_flip_mac_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });

    let build = |entry: &str| {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("bench_flip_mac"),
            source: ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("bench_flip_mac"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bench_flip_mac_bind_group"),
            layout: &bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: faces_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: divergence_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pressure_in.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: pressure_out.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });
        (pipeline, bind_group)
    };

    let timer = PassTimer::new(device);
    let gx = dim[0].div_ceil(4);
    let gy = dim[1].div_ceil(4);
    let gz = dim[2].div_ceil(4);

    let div_entry = find_entry_point(wgsl, "mac_divergence");
    let (div_pipeline, div_bind) = build(&div_entry);
    let divergence_us =
        time_dispatch_3d(device, queue, &div_pipeline, &div_bind, gx, gy, gz, &timer);

    let pressure_entry = find_entry_point(wgsl, "mac_pressure");
    let (pressure_pipeline, pressure_bind) = build(&pressure_entry);
    let pressure_us = time_dispatch_3d(
        device,
        queue,
        &pressure_pipeline,
        &pressure_bind,
        gx,
        gy,
        gz,
        &timer,
    );

    let project_entry = find_entry_point(wgsl, "mac_project");
    let (project_pipeline, project_bind) = build(&project_entry);
    let face_groups = mac_face_count(dim).div_ceil(64);
    let project_us = time_dispatch(
        device,
        queue,
        &project_pipeline,
        &project_bind,
        face_groups,
        1,
        &timer,
    );

    (divergence_us, pressure_us, project_us)
}

/// Measures the staggered-`MAC` incompressible pressure projection
/// (`mac_divergence` -> `mac_pressure` -> `mac_project` in `water_flip_mac.wesl`)
/// on a real device across [`MAC_BENCH_DIMS`] and asserts each stage takes a
/// finite, strictly positive, bounded time. The design doc
/// (`docs/prism_water_engine_design_zh.md` §12) budgets the `FLIP`/`PBF` sim at
/// `<= 2-4 ms` for a local `~1e5`-element domain as a **design target, not a
/// measured value**; this turns that line into an actual on-device measurement
/// of the pressure solve — the hottest, least cache-friendly part of a `FLIP`
/// step. The printed medians are the numbers the design-target budget should be
/// re-tuned against.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the measured budget (and the skip notice) must reach the test log so it can be captured"
)]
fn mac_projection_pass_gpu_budget_is_measured() {
    let Some((device, queue)) = try_timing_device() else {
        eprintln!(
            "mac_projection_pass_gpu_budget_is_measured: no timestamp-capable wgpu adapter, skipping on-device timing"
        );
        return;
    };

    let wgsl = compile_flip_mac_wgsl();

    for side in MAC_BENCH_DIMS {
        let dim = [side, side, side];
        let cells = side * side * side;
        let faces = mac_face_count(dim);
        let (divergence_us, pressure_us, project_us) =
            measure_mac_projection(&device, &queue, &wgsl, dim);
        eprintln!(
            "water FLIP staggered-MAC projection budget @ {cells} cells / {faces} faces (dim {side}^3): divergence = {divergence_us:.2} us, pressure(1 Jacobi sweep) = {pressure_us:.2} us, project = {project_us:.2} us (measured, median of {TIMED_RUNS})"
        );
        for (label, micros) in [
            ("mac_divergence", divergence_us),
            ("mac_pressure", pressure_us),
            ("mac_project", project_us),
        ] {
            assert!(
                micros.is_finite() && micros > 0.0 && micros < MAX_PASS_MICROS,
                "flip {label} @ {cells} cells measured {micros} us is not a healthy bounded timing"
            );
        }
    }
}

/// Compiles `water_flip_mac_p2g.wesl` (the face-centered particle-to-grid
/// scatter plus its mass-normalize pass) and returns its `Wgsl` translation,
/// mirroring [`compile_flip_mac_wgsl`].
fn compile_flip_mac_p2g_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_5246_4c50_0004),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_flip_mac_p2g.wesl"),
            "embedded://prism_render_scene/shaders/water_flip_mac_p2g.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_flip_mac_p2g.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Compiles `water_flip_mac_g2p.wesl` (the face-centered grid-to-particle
/// gather) and returns its `Wgsl` translation, mirroring [`compile_flip_mac_wgsl`].
fn compile_flip_mac_g2p_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_5246_4c50_0005),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/water_flip_mac_g2p.wesl"),
            "embedded://prism_render_scene/shaders/water_flip_mac_g2p.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("water_flip_mac_g2p.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Cubic grid resolutions (`dim = [D, D, D]`) the staggered-`MAC` particle-grid
/// transfer kernels are timed at. The interior is filled at half-cell spacing
/// (eight particles per cell), so `D = 12` is a `23^3 = 12167`-particle
/// interactive splash and `D = 24` a `47^3 = 103823`-particle domain — the
/// `~1e5` particle figure the design doc
/// (`docs/prism_water_engine_design_zh.md` §12) budgets the `FLIP`/`PBF`
/// solver against. Timing all three resolutions shows how the scatter/gather
/// passes scale with the particle count (`P2G`/`G2P`) and the face count
/// (`faces_normalize`) the kernels launch over.
const FLIP_TRANSFER_DIMS: [u32; 3] = [12, 18, 24];

/// Builds a deterministic `FLIP`/`APIC` particle cube filling the interior of a
/// `dim` staggered-`MAC` grid at half-cell spacing (eight particles per cell),
/// so every particle scatters to a fully populated eight-corner stencil on all
/// three staggered face lattices. Each particle is active (`pos.w = 1`), carries
/// a smooth divergent velocity, and a non-zero `APIC` affine field so both the
/// `P2G` scatter and the `G2P` gather walk their full affine path. Positions sit
/// a quarter-cell inside the low corner and stop three-quarters of a cell short
/// of the far wall, so every staggered corner stays in bounds (no silent
/// skips). Timing depends only on the full population being exercised, not on
/// the exact values.
fn flip_bench_particles(dim: [u32; 3]) -> Vec<GpuFlipParticle> {
    let spacing = 0.5_f32;
    let base = 0.25_f32;
    let nx = 2 * dim[0] - 1;
    let ny = 2 * dim[1] - 1;
    let nz = 2 * dim[2] - 1;
    let mut particles = Vec::with_capacity((nx * ny * nz) as usize);
    let mut k = 0u32;
    while k < nz {
        let mut j = 0u32;
        while j < ny {
            let mut i = 0u32;
            while i < nx {
                let fi = i as f32;
                let fj = j as f32;
                let fk = k as f32;
                particles.push(GpuFlipParticle {
                    pos: [
                        base + fi * spacing,
                        base + fj * spacing,
                        base + fk * spacing,
                        1.0,
                    ],
                    vel: [
                        0.1 * bevy_math::ops::sin(0.3 * fi + 0.2 * fk),
                        0.1 * bevy_math::ops::cos(0.25 * fj - 0.15 * fi),
                        0.1 * bevy_math::ops::sin(0.2 * fk + 0.1 * fj),
                        0.0,
                    ],
                    c0: [0.05, 0.0, 0.0, 0.0],
                    c1: [0.0, 0.05, 0.0, 0.0],
                    c2: [0.0, 0.0, 0.05, 0.0],
                });
                i += 1;
            }
            j += 1;
        }
        k += 1;
    }
    particles
}

/// Builds the host-side [`GpuFlipSimParams`] uniform for a cubic bench domain.
/// The grid spacing is unit (`dx = inv_dx = 1`), the `APIC` affine field is
/// enabled, and representative `FLIP`/`Jacobi` blend factors are set; the
/// measured dispatch time is independent of these values, only of the particle
/// and cell counts driving the launch sizes.
fn flip_bench_params(dim: [u32; 3], particle_count: u32) -> GpuFlipSimParams {
    let cell_count = dim[0] * dim[1] * dim[2];
    GpuFlipSimParams {
        origin: [0.0, 0.0, 0.0, 0.0],
        dim: [dim[0], dim[1], dim[2], 0],
        dx: 1.0,
        inv_dx: 1.0,
        flip_blend: 0.95,
        particle_mass: 1.0,
        jacobi_omega: 0.8,
        use_affine: 1,
        particle_count,
        cell_count,
    }
}

/// Times the staggered-`MAC` particle-to-grid transfer over a cubic `dim`
/// domain, returning `(p2g_us, normalize_us)` measured medians in microseconds.
///
/// Both kernels share one explicit four-binding `@group(0)` layout
/// (`particles`, `face_scatter` atomics, `faces`, `sim_params`):
/// `water_flip_mac_p2g` scatters each particle's `APIC` velocity onto the three
/// staggered face grids as `[momentum, mass]` signed fixed-point pairs (touching
/// bindings 0, 1, 3), then `water_flip_mac_faces_normalize` divides each face's
/// momentum by its mass into the `f32` velocity buffer (touching bindings 1, 2,
/// 3). An auto-derived (`layout: None`) layout would omit, per entry point, the
/// bindings that entry does not touch, so the two entries would reject a shared
/// bind group; the explicit superset layout lets one bind group feed both (a
/// `WGSL` entry may bind a superset of what it reads). The scatter runs one
/// invocation per particle and the normalize one per face, both at
/// `@workgroup_size(64)`.
fn measure_flip_p2g(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    p2g_entry: &str,
    norm_entry: &str,
    dim: [u32; 3],
) -> (f64, f64) {
    let particles = flip_bench_particles(dim);
    let params = flip_bench_params(dim, particles.len() as u32);
    let face_count = mac_face_count(dim) as usize;
    let scatter_zero = vec![0u32; face_count * 2];
    let faces_zero = vec![0.0_f32; face_count];

    let particle_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_flip_p2g_particles"),
        contents: bytemuck::cast_slice(&particles),
        usage: BufferUsages::STORAGE,
    });
    let scatter_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_flip_p2g_scatter"),
        contents: bytemuck::cast_slice(&scatter_zero),
        usage: BufferUsages::STORAGE,
    });
    let faces_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_flip_p2g_faces"),
        contents: bytemuck::cast_slice(&faces_zero),
        usage: BufferUsages::STORAGE,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_flip_p2g_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    // One explicit four-binding group-0 layout shared by both passes: particles
    // (read-only), the scatter atomics (read-write), the normalized faces
    // (read-write), and the sim params uniform. A per-entry auto-derived layout
    // would drop the bindings each entry skips, so declare the full superset.
    let storage_ro = BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let storage_rw = BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let uniform = BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let entry = |binding: u32, ty: BindingType| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bench_flip_p2g_layout"),
        entries: &[
            entry(0, storage_ro),
            entry(1, storage_rw),
            entry(2, storage_rw),
            entry(3, uniform),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("bench_flip_p2g_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });

    let build = |ep: &str| {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("bench_flip_p2g"),
            source: ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("bench_flip_p2g"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(ep),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bench_flip_p2g_bind_group"),
            layout: &bind_layout,
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
                    resource: faces_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });
        (pipeline, bind_group)
    };

    let (p2g_pipeline, p2g_bind) = build(p2g_entry);
    let p2g_groups = params.particle_count.div_ceil(64);
    let p2g_timer = PassTimer::new(device);
    let p2g_us = time_dispatch(
        device,
        queue,
        &p2g_pipeline,
        &p2g_bind,
        p2g_groups,
        1,
        &p2g_timer,
    );

    let (norm_pipeline, norm_bind) = build(norm_entry);
    let norm_groups = (face_count as u32).div_ceil(64);
    let norm_timer = PassTimer::new(device);
    let normalize_us = time_dispatch(
        device,
        queue,
        &norm_pipeline,
        &norm_bind,
        norm_groups,
        1,
        &norm_timer,
    );

    (p2g_us, normalize_us)
}

/// Times the staggered-`MAC` grid-to-particle gather (`water_flip_mac_g2p`) over
/// a cubic `dim` domain, returning the measured median microseconds. The gather
/// reads the projected and the pre-projection face fields, blends the `PIC` and
/// `FLIP` updates, and writes the particle velocities (plus the `APIC` affine
/// rows) back, so its `@group(0)` touches the particles (read-write, binding 0),
/// the projected faces (binding 1), the pre-projection faces (binding 2), and
/// the params (binding 3). The seed face fields reuse [`mac_bench_faces`]; the
/// gather dispatches one invocation per particle at `@workgroup_size(64)`.
fn measure_flip_g2p(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    dim: [u32; 3],
) -> f64 {
    let particles = flip_bench_particles(dim);
    let params = flip_bench_params(dim, particles.len() as u32);
    let projected = mac_bench_faces(dim);
    let preprojection = mac_bench_faces(dim);

    let particle_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_flip_g2p_particles"),
        contents: bytemuck::cast_slice(&particles),
        usage: BufferUsages::STORAGE,
    });
    let projected_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_flip_g2p_projected"),
        contents: bytemuck::cast_slice(&projected),
        usage: BufferUsages::STORAGE,
    });
    let preproj_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_flip_g2p_preprojection"),
        contents: bytemuck::cast_slice(&preprojection),
        usage: BufferUsages::STORAGE,
    });
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("bench_flip_g2p_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    // Explicit group-0 layout: particles (read-write), projected faces
    // (read-only), pre-projection faces (read-only), params uniform. The single
    // gather entry binds every slot, but declaring the layout explicitly keeps
    // the harness consistent with the P2G and projection benches.
    let storage_ro = BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let storage_rw = BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let uniform = BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let entry_desc = |binding: u32, ty: BindingType| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bench_flip_g2p_layout"),
        entries: &[
            entry_desc(0, storage_rw),
            entry_desc(1, storage_ro),
            entry_desc(2, storage_ro),
            entry_desc(3, uniform),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("bench_flip_g2p_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bench_flip_g2p"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("bench_flip_g2p"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("bench_flip_g2p_bind_group"),
        layout: &bind_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: projected_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: preproj_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let groups = params.particle_count.div_ceil(64);
    let timer = PassTimer::new(device);
    time_dispatch(device, queue, &pipeline, &bind_group, groups, 1, &timer)
}

/// Measures the staggered-`MAC` particle-grid transfer kernels
/// (`water_flip_mac_p2g` -> `water_flip_mac_faces_normalize` in
/// `water_flip_mac_p2g.wesl`, then `water_flip_mac_g2p` in
/// `water_flip_mac_g2p.wesl`) on a real device across [`FLIP_TRANSFER_DIMS`] and
/// asserts each pass takes a finite, strictly positive, bounded time. The design
/// doc (`docs/prism_water_engine_design_zh.md` §12) budgets the `FLIP`/`PBF` sim
/// at `<= 2-4 ms` for a local `~1e5` particle domain as a **design target, not a
/// measured value**; together with the pressure-projection bench this turns the
/// remaining `FLIP` transfer half of that line into actual on-device
/// measurements. The printed medians are the numbers the design-target budget
/// should be re-tuned against.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the measured budget (and the skip notice) must reach the test log so it can be captured"
)]
fn flip_transfer_pass_gpu_budget_is_measured() {
    let Some((device, queue)) = try_timing_device() else {
        eprintln!(
            "flip_transfer_pass_gpu_budget_is_measured: no timestamp-capable wgpu adapter, skipping on-device timing"
        );
        return;
    };

    let p2g_wgsl = compile_flip_mac_p2g_wgsl();
    let p2g_entry = find_entry_point(&p2g_wgsl, "water_flip_mac_p2g");
    let norm_entry = find_entry_point(&p2g_wgsl, "faces_normalize");
    let g2p_wgsl = compile_flip_mac_g2p_wgsl();
    let g2p_entry = find_entry_point(&g2p_wgsl, "water_flip_mac_g2p");

    for side in FLIP_TRANSFER_DIMS {
        let dim = [side, side, side];
        let particle_count = (2 * side - 1).pow(3);
        let faces = mac_face_count(dim);
        let (p2g_us, normalize_us) =
            measure_flip_p2g(&device, &queue, &p2g_wgsl, &p2g_entry, &norm_entry, dim);
        let g2p_us = measure_flip_g2p(&device, &queue, &g2p_wgsl, &g2p_entry, dim);
        eprintln!(
            "water FLIP MAC transfer budget @ {particle_count} particles / {faces} faces (dim {side}^3): p2g scatter = {p2g_us:.2} us, faces_normalize = {normalize_us:.2} us, g2p gather = {g2p_us:.2} us (measured, median of {TIMED_RUNS})"
        );
        for (label, micros) in [
            ("water_flip_mac_p2g", p2g_us),
            ("water_flip_mac_faces_normalize", normalize_us),
            ("water_flip_mac_g2p", g2p_us),
        ] {
            assert!(
                micros.is_finite() && micros > 0.0 && micros < MAX_PASS_MICROS,
                "flip {label} @ {particle_count} particles measured {micros} us is not a healthy bounded timing"
            );
        }
    }
}
