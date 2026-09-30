//! Real-device `GPU` parity coverage for the ray-cone footprint / texture-`LOD`
//! kernel.
//!
//! The sibling [`shader_tests`](super::shader_tests) module proves
//! `shaders/ray_footprint.wesl` parses and type-checks through the render
//! world's [`ShaderCache`], guarding the kernel's *shape*. This test closes the
//! behavioural gap: it packs a batch of ray-cone footprints (each with a
//! per-surface texel size), binds the actual `ray_footprint` compute pipeline on
//! a live `Metal` (or any native `wgpu`) device, dispatches it, reads the
//! results back and asserts them record-for-record against the `CPU` golden
//! [`RayFootprint`] mip math (`projected_width` / `texel_span` / `mip_level` /
//! `mip_floor`).
//!
//! Because the `WESL` kernel and the golden arithmetic share byte-identical
//! operations — the non-finite/negative sanitization, the linear
//! `width = cone_width + hit_distance * cone_spread_angle`, the guarded
//! `span = width / texel`, and the transcendental-free `log2_linear`
//! (power-of-two halving/doubling only) — a green run is direct on-device
//! evidence the ported mip selection matches its reference to `float32`
//! rounding, not merely that it compiles.
//!
//! Adapter acquisition is best-effort: on a headless host with no `wgpu`
//! adapter [`try_solver_device`] returns `None` and the test skips with a
//! printed notice instead of failing.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Instance as WgpuInstance, InstanceDescriptor, InstanceFlags, MapMode,
    PipelineCompilationOptions, PollType, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource,
};

use prism_render_architecture::ray_scene::RayFootprint;

use super::abi::{FOOTPRINT_RESULT_WORDS, FOOTPRINT_WORDS, GpuFootprintParams};

/// Absolute per-scalar tolerance for the `GPU`-versus-`CPU` comparison of the
/// three `f32` result fields. Both paths run the same `float32` arithmetic, so
/// agreement is far tighter in practice; the margin only absorbs a driver's
/// fused-multiply-add contraction of `cone_width + hit_distance * cone_spread`.
const PARITY_EPS: f32 = 1.0e-3;

/// Minimum distance the golden `mip_level` must sit from an integer boundary for
/// the discrete `mip_floor` bucket to be compared for exact equality. Nearer a
/// boundary a legitimate sub-`ULP` difference in `projected_width` can flip the
/// floor, which is not a defect; those records still assert the `f32` fields.
const MIP_FLOOR_GUARD: f32 = 0.02;

/// Maximum mip level the continuous `mip_level` is clamped to (inclusive).
const MAX_MIP: u32 = 8;

/// Tiny deterministic `xorshift` `RNG` so the batch is byte-reproducible across
/// runs and hosts without pulling in a dependency.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        lo + (hi - lo) * unit
    }
}

/// One packed footprint the kernel reads: the three `RayFootprint` slopes plus
/// the surface texel size the mip math divides by. Stored as raw values so the
/// kernel's own sanitization path is exercised (the `CPU` golden reconstructs a
/// [`RayFootprint`], which sanitizes identically).
#[derive(Clone, Copy)]
struct FootprintInput {
    cone_width: f32,
    cone_spread: f32,
    hit_distance: f32,
    texel: f32,
}

/// Assembles a batch that provably exercises every branch: growing distance,
/// varied texel sizes spanning mip 0 through the clamp ceiling, a guarded zero
/// texel, and non-finite / negative inputs that must sanitize to zero.
fn build_footprints() -> Vec<FootprintInput> {
    let mut rng = Rng::new(0x00F0_07C0_DEBA_5E11);
    let mut out = Vec::new();

    // Randomized well-formed cones across a wide dynamic range.
    for _ in 0..48 {
        out.push(FootprintInput {
            cone_width: rng.range(0.001, 2.0),
            cone_spread: rng.range(0.0, 0.4),
            hit_distance: rng.range(0.0, 40.0),
            texel: rng.range(0.01, 1.5),
        });
    }

    // Hand-picked deterministic edge cases.
    let edges = [
        // Sub-texel footprint clamps to mip 0.
        FootprintInput { cone_width: 0.25, cone_spread: 0.0, hit_distance: 0.0, texel: 1.0 },
        // Exactly one texel -> mip 0.
        FootprintInput { cone_width: 1.0, cone_spread: 0.0, hit_distance: 0.0, texel: 1.0 },
        // 16 texels -> mip 4 exactly.
        FootprintInput { cone_width: 16.0, cone_spread: 0.0, hit_distance: 0.0, texel: 1.0 },
        // Far beyond the ceiling clamps to MAX_MIP.
        FootprintInput { cone_width: 4096.0, cone_spread: 0.0, hit_distance: 0.0, texel: 1.0 },
        // Guarded zero texel -> span 0 -> mip 0.
        FootprintInput { cone_width: 2.0, cone_spread: 0.0, hit_distance: 0.0, texel: 0.0 },
        // Negative texel guards the same way.
        FootprintInput { cone_width: 2.0, cone_spread: 0.0, hit_distance: 0.0, texel: -1.0 },
        // Non-finite / negative slopes sanitize to zero (projected_width 0).
        FootprintInput { cone_width: -1.0, cone_spread: f32::NAN, hit_distance: f32::INFINITY, texel: 1.0 },
        // Growing distance drives coarser mips.
        FootprintInput { cone_width: 0.1, cone_spread: 0.05, hit_distance: 30.0, texel: 0.1 },
    ];
    out.extend_from_slice(&edges);
    out
}

/// Packs the batch into the flat `array<u32>` the kernel binds: `FOOTPRINT_WORDS`
/// `f32`-`to_bits` words per record.
fn pack_footprints(inputs: &[FootprintInput]) -> Vec<u32> {
    let mut words = vec![0u32; inputs.len() * FOOTPRINT_WORDS];
    for (i, fp) in inputs.iter().enumerate() {
        let base = i * FOOTPRINT_WORDS;
        words[base] = fp.cone_width.to_bits();
        words[base + 1] = fp.cone_spread.to_bits();
        words[base + 2] = fp.hit_distance.to_bits();
        words[base + 3] = fp.texel.to_bits();
    }
    words
}

/// Streams the `Wgsl` source back out of the shader cache without a device.
fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("the footprint shader is WESL"),
    }
}

/// Compiles `ray_footprint.wesl` and returns its `Wgsl` translation.
fn compile_footprint_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_4655_5450_5249_4e54_0002),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../shaders/ray_footprint.wesl"),
            "embedded://prism_render_scene/shaders/ray_footprint.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("ray_footprint.wesl failed to compile: {error}"));
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

/// Best-effort acquisition of a native compute device and queue.
fn try_solver_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = WgpuInstance::new(InstanceDescriptor {
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

/// One unpacked footprint result the kernel writes back.
#[derive(Clone, Copy)]
struct GpuFootprintResult {
    projected_width: f32,
    texel_span: f32,
    mip_level: f32,
    mip_floor: u32,
}

/// Records the `ray_footprint` dispatch and reads the results back.
///
/// Binds the packed `footprints` input, an `RW` `results` output and the
/// [`GpuFootprintParams`] uniform on `@group(0)` bindings 0..3, dispatches
/// `count.div_ceil(64)` workgroups and maps the results back.
fn dispatch_footprint(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    footprint_words: &[u32],
    count: u32,
) -> Vec<GpuFootprintResult> {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ray_footprint_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("ray_footprint_parity_pipeline"),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let result_words = count as usize * FOOTPRINT_RESULT_WORDS;
    let result_bytes = (result_words * size_of::<u32>()) as u64;

    let footprints_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("fp_inputs"),
        contents: bytemuck::cast_slice(footprint_words),
        usage: BufferUsages::STORAGE,
    });
    let results_buf = device.create_buffer(&BufferDescriptor {
        label: Some("fp_results"),
        size: result_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = GpuFootprintParams {
        footprint_count: count,
        max_mip: MAX_MIP,
        pad0: 0,
        pad1: 0,
    };
    let param_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("fp_params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });

    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("fp_group0"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: footprints_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: results_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: param_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("fp_results_stage"),
        size: result_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("fp_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fp_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&results_buf, 0, &stage, 0, result_bytes);
    queue.submit([encoder.finish()]);

    stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let raw: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    stage.unmap();

    let mut results = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        let base = i * FOOTPRINT_RESULT_WORDS;
        results.push(GpuFootprintResult {
            projected_width: f32::from_bits(raw[base]),
            texel_span: f32::from_bits(raw[base + 1]),
            mip_level: f32::from_bits(raw[base + 2]),
            mip_floor: raw[base + 3],
        });
    }
    results
}

/// Footprint mip parity: `ray_footprint` must reproduce the golden
/// [`RayFootprint`] `projected_width` / `texel_span` / `mip_level` / `mip_floor`
/// for every record in the batch.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn ray_footprint_matches_cpu_golden_on_device() {
    let Some((device, queue)) = try_solver_device() else {
        eprintln!("skipping ray_footprint parity: no wgpu adapter");
        return;
    };

    let inputs = build_footprints();
    let words = pack_footprints(&inputs);

    let wgsl = compile_footprint_wgsl();
    let entry = find_entry_point(&wgsl, "ray_footprint");
    let gpu = dispatch_footprint(&device, &queue, &wgsl, &entry, &words, inputs.len() as u32);

    let mut mip0_count = 0usize;
    let mut clamped_count = 0usize;
    let mut exact_floor_count = 0usize;
    for (i, fp) in inputs.iter().enumerate() {
        // The golden reconstructs a `RayFootprint`, which sanitizes the three
        // slopes exactly as the kernel does on read; `texel` is sanitized inside
        // the span/mip calls.
        let golden = RayFootprint::new(fp.cone_width, fp.cone_spread, fp.hit_distance);
        let cpu_width = golden.projected_width();
        let cpu_span = golden.texel_span(fp.texel);
        let cpu_level = golden.mip_level(fp.texel, MAX_MIP);
        let cpu_floor = golden.mip_floor(fp.texel, MAX_MIP);

        let g = gpu[i];
        assert!(
            (g.projected_width - cpu_width).abs() <= PARITY_EPS,
            "record {i}: GPU projected_width {} != CPU {}",
            g.projected_width,
            cpu_width
        );
        assert!(
            (g.texel_span - cpu_span).abs() <= PARITY_EPS,
            "record {i}: GPU texel_span {} != CPU {}",
            g.texel_span,
            cpu_span
        );
        assert!(
            (g.mip_level - cpu_level).abs() <= PARITY_EPS,
            "record {i}: GPU mip_level {} != CPU {}",
            g.mip_level,
            cpu_level
        );

        if cpu_level <= 0.0 {
            mip0_count += 1;
        }
        if cpu_level >= MAX_MIP as f32 {
            clamped_count += 1;
        }

        // Only assert the discrete bucket when the continuous level is safely
        // away from an integer boundary; nearer a boundary a sub-`ULP` diff can
        // legitimately flip the floor.
        let dist_to_boundary = (cpu_level - cpu_level.round()).abs();
        if dist_to_boundary > MIP_FLOOR_GUARD {
            exact_floor_count += 1;
            assert_eq!(
                g.mip_floor, cpu_floor,
                "record {i}: GPU mip_floor {} != CPU {} (mip_level {cpu_level})",
                g.mip_floor, cpu_floor
            );
        }
    }

    assert!(mip0_count > 0, "batch must exercise the mip-0 / sub-texel path");
    assert!(clamped_count > 0, "batch must exercise the max-mip clamp path");
    assert!(
        exact_floor_count > 0,
        "batch must exercise the discrete mip_floor bucket away from a boundary"
    );
}
