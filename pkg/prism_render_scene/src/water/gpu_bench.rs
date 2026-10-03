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
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindingResource,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePassTimestampWrites, ComputePipelineDescriptor, DeviceDescriptor, Extent3d, Features,
    Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineCompilationOptions, PollType,
    QuerySet, QuerySetDescriptor, QueryType, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor,
};

use super::abi::{GpuWaterSpectrumParams, GpuWaterSweParams};

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
            pass.set_bind_group(0, bind_group, &[]);
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
            pass.set_bind_group(0, bind_group, &[]);
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
