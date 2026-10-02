//! Real-device `GPU` parity coverage for the world-space `ReSTIR` inject kernel.
//!
//! The sibling [`shader_tests`](super::shader_tests) module proves the inject
//! `WESL` source parses and type-checks through the render world's
//! [`ShaderCache`], and mirrors its open-address slot claim on the `CPU`. That
//! guards the *shape* and the *claim arithmetic* of the kernel, but it never
//! runs the kernel on a device: a shader can compile cleanly and still compute
//! the wrong number, or the host's immediate-data / bind-group wiring can be
//! wrong in a way no `CPU` mirror can catch. The test here closes that gap for
//! `world_restir_inject.wesl`'s `inject_main` entry by binding the real compute
//! pipeline on an actual `Metal` (or any native `wgpu`) device, dispatching one
//! invocation per visible point over a deterministic point stream, reading the
//! resulting reservoir table and the parallel `atomic<u32>` slot-state array
//! back, and asserting both byte-for-byte against a serial `CPU` twin built
//! from the authoritative
//! [`prism_render_shading::gi::world_restir::spatial_hash`] hash
//! (`compute_key` / `checksum` / `bucket_index`) and the frozen
//! [`super::abi::GpuWorldRestirReservoir`] `make_reservoir` layout.
//!
//! Determinism under parallel execution: the device runs every invocation
//! concurrently, so the open-address claim order is not fixed, whereas the
//! `CPU` twin probes serially in point order. The two agree only when the final
//! table state is order-independent. This test picks the default
//! `131072`-slot capacity and asserts every fresh claim lands directly in its
//! base bucket with zero linear-probe steps, i.e. the stream's distinct cells
//! occupy pairwise-disjoint buckets. Under that invariant each cell's slot is
//! fixed regardless of scheduling, repeated cells converge on one slot through
//! the shader's first-wins `atomicCompareExchangeWeak`, and the parallel result
//! is identical to the serial golden to the bit.
//!
//! Adapter acquisition is best-effort: on a headless host with no `wgpu`
//! adapter, or an adapter without the `IMMEDIATES` feature (the inject
//! `var<immediate>` block compiles to immediate data), [`try_inject_device`]
//! returns `None` and the test
//! skips with a printed notice instead of failing, so the suite stays green
//! everywhere while still exercising the full dispatch on any machine with a
//! real device that supports immediate data (for example an `Apple` `M`-series
//! `GPU`).

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_math::Vec3;
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use bytemuck::Zeroable;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Features, Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PollType, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};

use prism_render_shading::gi::world_restir::spatial_hash::{self, HashGridParams};
use prism_render_shading::gi::world_restir::world_reservoir::PROBE_LIMIT;

use super::abi::{
    GpuWorldRestirInjectParams, GpuWorldRestirInjectPoint, GpuWorldRestirLight,
    GpuWorldRestirReservoir, GpuWorldRestirSeedParams, WORLD_RESTIR_SEED_WORKGROUP_SIZE,
};

/// Reservoir-table capacity exercised on device.
///
/// Large enough that the stream's handful of well-separated cells hash to
/// pairwise-disjoint buckets (asserted by the golden builder), which is what
/// makes the parallel on-device claim order-independent and therefore
/// comparable byte-for-byte to the serial `CPU` twin.
const CAPACITY: u32 = 131_072;

/// Streams the `Wgsl` source back out of the shader cache without a device.
///
/// Mirrors the closure the parity tests elsewhere use so the `WESL` is composed
/// through the exact render-world pipeline; the compiled `Wgsl` string (rather
/// than a device module) is handed to a raw `wgpu` device the test creates
/// itself.
fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("world_restir shaders are WESL"),
    }
}

/// Compiles `world_restir_inject.wesl` and returns its `Wgsl` translation.
fn compile_inject_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5753_5244_5f49_4e4a_0001),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../shaders/world_restir_inject.wesl"),
            "embedded://prism_render_scene/shaders/world_restir_inject.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("world_restir_inject.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Compiles `world_restir_seed.wesl` and returns its `Wgsl` translation.
fn compile_seed_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5753_5244_5f53_4544_0001),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../shaders/world_restir_seed.wesl"),
            "embedded://prism_render_scene/shaders/world_restir_seed.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("world_restir_seed.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Finds the compute entry point whose name contains `needle` in `wgsl`.
///
/// The `WESL` compiler may prefix module-local names, so the parity test
/// locates the `inject_main` entry by substring rather than assuming a fixed
/// symbol.
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

/// Best-effort acquisition of a native compute device and queue that exposes at
/// least `min_immediate_bytes` of immediate data.
///
/// Returns `None` (rather than panicking) when no adapter is available or the
/// adapter cannot satisfy the `IMMEDIATES` feature / the requested immediate
/// block, so the suite stays green on headless hosts; on a machine with a real
/// `GPU` that supports immediate data this yields a live device the parity
/// tests dispatch against.
fn try_immediate_device(min_immediate_bytes: u32) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
    if !adapter.features().contains(Features::IMMEDIATES) {
        return None;
    }
    let limits = adapter.limits();
    if limits.max_immediate_size < min_immediate_bytes {
        return None;
    }
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        required_features: Features::IMMEDIATES,
        required_limits: limits,
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Best-effort inject device: an immediate-data device whose immediate block
/// fits the `48`-byte [`GpuWorldRestirInjectParams`].
fn try_inject_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    try_immediate_device(size_of::<GpuWorldRestirInjectParams>() as u32)
}

/// The deterministic visible-point stream shared by the `CPU` golden and the
/// device dispatch: six well-separated cells with three exact repeats
/// interleaved so the first-wins `Reuse` branch is exercised on device. Matches
/// the stream the `inject_shader_matches_cpu_golden` `CPU` mirror uses.
fn point_stream() -> [(Vec3, Vec3); 10] {
    [
        (Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        (Vec3::new(3.0, 0.0, 0.0), Vec3::X),
        (Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::Z),
        (Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        (Vec3::new(0.0, 0.0, -3.0), Vec3::new(0.0, -1.0, 0.0)),
        (Vec3::new(3.0, 0.0, 0.0), Vec3::X),
        (
            Vec3::new(3.0, 3.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0).normalize(),
        ),
        (
            Vec3::new(-3.0, 0.0, 3.0),
            Vec3::new(-1.0, 0.0, 1.0).normalize(),
        ),
        (Vec3::new(0.0, 3.0, 0.0), Vec3::Z),
    ]
}

/// Byte-for-byte twin of the shader's `make_reservoir`: the cell geometry plus
/// the `SHARC` checksum and the `valid` flag, every energy / sample lane zeroed.
fn golden_reservoir(
    position: [f32; 3],
    normal: [f32; 3],
    checksum: u32,
) -> GpuWorldRestirReservoir {
    GpuWorldRestirReservoir {
        visible_point: position,
        w: 0.0,
        visible_normal: normal,
        m: 0.0,
        sample_point: [0.0; 3],
        checksum,
        sample_normal: [0.0; 3],
        valid: 1,
        radiance: [0.0; 3],
        _pad0: 0,
    }
}

/// Builds the serial `CPU` golden tables plus the packed inject-point records.
///
/// Reproduces the shader's open-address claim serially in point order
/// (`EMPTY_SLOT = 0`, linear probe bounded by [`PROBE_LIMIT`], first-wins on a
/// matching checksum). Asserts every fresh claim lands at its base bucket with
/// zero probe steps, which both proves the stream's cells are pairwise-disjoint
/// at [`CAPACITY`] and guarantees the parallel device result is order-
/// independent, so the serial golden is the exact device expectation.
fn build_golden(
    camera: Vec3,
    params: &HashGridParams,
) -> (
    Vec<u32>,
    Vec<GpuWorldRestirReservoir>,
    Vec<GpuWorldRestirInjectPoint>,
) {
    let cap = CAPACITY as usize;
    let mut slot_state = vec![0u32; cap];
    let mut reservoirs = vec![GpuWorldRestirReservoir::zeroed(); cap];
    let mut points = Vec::with_capacity(10);

    let steps = PROBE_LIMIT.min(CAPACITY);
    for (position, normal) in point_stream() {
        let pos = position.to_array();
        let nrm = normal.to_array();
        points.push(GpuWorldRestirInjectPoint {
            world_position: pos,
            _pad0: 0.0,
            world_normal: nrm,
            _pad1: 0.0,
        });

        let key = spatial_hash::compute_key(position, normal, camera, params);
        let checksum = spatial_hash::checksum(&key);
        assert_ne!(
            checksum, 0,
            "checksum must be non-zero so EMPTY_SLOT=0 stays unambiguous"
        );
        let base = spatial_hash::bucket_index(&key, CAPACITY);
        let mut i = 0u32;
        while i < steps {
            let idx = ((base + i) % CAPACITY) as usize;
            if slot_state[idx] == 0 {
                assert_eq!(
                    i, 0,
                    "capacity must keep cells in disjoint buckets so the parallel claim is \
                     order-independent"
                );
                slot_state[idx] = checksum;
                reservoirs[idx] = golden_reservoir(pos, nrm, checksum);
                break;
            }
            if slot_state[idx] == checksum {
                break;
            }
            i += 1;
        }
        assert!(i < steps, "the probe window must not fill for this stream");
    }

    (slot_state, reservoirs, points)
}

/// Dispatches `inject_main` once over `points` and reads back the reservoir
/// table and slot-state array as raw bytes.
///
/// The pipeline uses an explicit layout (three storage bindings + a
/// `48`-byte immediate block) because the inject parameters arrive through the
/// `var<immediate>` path, so the bindings and the immediate match the shader's
/// declaration exactly.
#[expect(
    clippy::too_many_lines,
    reason = "one linear dispatch-and-readback keeps the parity path auditable"
)]
fn dispatch_inject(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    points: &[GpuWorldRestirInjectPoint],
    params: &GpuWorldRestirInjectParams,
) -> (Vec<u8>, Vec<u8>) {
    let cap = CAPACITY as u64;
    let reservoir_bytes = cap * size_of::<GpuWorldRestirReservoir>() as u64;
    let slot_bytes = cap * size_of::<u32>() as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("world_restir_inject_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("world_restir_inject_group0"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("world_restir_inject_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: size_of::<GpuWorldRestirInjectParams>() as u32,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("world_restir_inject_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let points_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("inject_points"),
        contents: bytemuck::cast_slice(points),
        usage: BufferUsages::STORAGE,
    });
    let reservoirs_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("inject_reservoirs"),
        contents: &vec![0u8; reservoir_bytes as usize],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let slot_state_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("inject_slot_state"),
        contents: &vec![0u8; slot_bytes as usize],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("inject_group0"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: points_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: reservoirs_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: slot_state_buf.as_entire_binding(),
            },
        ],
    });

    let reservoir_stage = device.create_buffer(&BufferDescriptor {
        label: Some("inject_reservoirs_stage"),
        size: reservoir_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let slot_stage = device.create_buffer(&BufferDescriptor {
        label: Some("inject_slot_state_stage"),
        size: slot_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("inject_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("inject_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(params));
        pass.dispatch_workgroups(params.point_count.max(1).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&reservoirs_buf, 0, &reservoir_stage, 0, reservoir_bytes);
    encoder.copy_buffer_to_buffer(&slot_state_buf, 0, &slot_stage, 0, slot_bytes);
    queue.submit([encoder.finish()]);

    reservoir_stage.slice(..).map_async(MapMode::Read, |_| {});
    slot_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let read_back = |buffer: &wgpu::Buffer| -> Vec<u8> {
        let view = buffer
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let bytes = view.to_vec();
        drop(view);
        buffer.unmap();
        bytes
    };
    (read_back(&reservoir_stage), read_back(&slot_stage))
}

/// A read-only (`read_only = true`) or read-write storage buffer binding for
/// the inject group layout. The atomic slot-state array uses the same plain
/// storage layout entry as a non-atomic read-write buffer in `wgpu`.
fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// One on-device `inject_main` dispatch must reproduce the serial `CPU` golden
/// reservoir table and slot-state array byte-for-byte.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a usable device"
)]
fn inject_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_inject_device() else {
        eprintln!(
            "inject_gpu_matches_cpu_golden: no wgpu adapter with immediate data, skipping \
             on-device parity"
        );
        return;
    };

    let camera = Vec3::new(0.0, 1.5, 4.0);
    let params = HashGridParams::DEFAULT;
    let (golden_slot_state, golden_reservoirs, points) = build_golden(camera, &params);

    let inject_params = GpuWorldRestirInjectParams {
        camera_position: camera.to_array(),
        base_cell_size: params.base_cell_size,
        jitter: params.jitter.to_array(),
        level_scale: params.level_scale,
        capacity: CAPACITY,
        point_count: points.len() as u32,
        normal_resolution: params.normal_resolution,
        frame: 0,
    };

    let wgsl = compile_inject_wgsl();
    let entry = find_entry_point(&wgsl, "inject_main");
    let (gpu_reservoirs, gpu_slot_state) =
        dispatch_inject(&device, &queue, &wgsl, &entry, &points, &inject_params);

    let golden_reservoir_bytes: &[u8] = bytemuck::cast_slice(&golden_reservoirs);
    let golden_slot_bytes: &[u8] = bytemuck::cast_slice(&golden_slot_state);

    assert_eq!(
        gpu_slot_state.len(),
        golden_slot_bytes.len(),
        "slot-state byte length mismatch"
    );
    assert_eq!(
        gpu_reservoirs.len(),
        golden_reservoir_bytes.len(),
        "reservoir byte length mismatch"
    );
    assert!(
        gpu_slot_state == golden_slot_bytes,
        "device slot-state array must match the CPU golden byte-for-byte"
    );
    assert!(
        gpu_reservoirs == golden_reservoir_bytes,
        "device reservoir table must match the CPU golden byte-for-byte"
    );
}

// ---------------------------------------------------------------------------
// Seed pass (`world_restir_seed.wesl` `seed_main`) real-device parity.
// ---------------------------------------------------------------------------

/// Reservoir-table capacity exercised by the seed parity dispatch.
///
/// The seed kernel is fully per-slot independent and order-independent (each
/// invocation reads only its own `src` slot and writes only its own `dst`
/// slot), so a tiny fixed table is enough to prove parity across both the
/// occupied (streaming `RIS`) and the empty (copy-through) branches.
const SEED_CAPACITY: u32 = 4;

/// Absolute + relative tolerance for the seed `RIS` arithmetic.
///
/// The `CPU` mirror and the `GPU` kernel run the identical formulas, so
/// cross-device `f32` rounding (`sqrt` / division / fused-multiply-add
/// contraction) is the only expected divergence; `1e-3` scaled by the lane
/// magnitude absorbs it while still catching a real serialisation or host
/// wiring bug.
const SEED_PARITY_EPS: f32 = 1e-3;

/// The candidate lights the seed `RIS` stream draws from, shared verbatim by
/// the device light buffer and the `CPU` mirror. All emitters sit at positive
/// `z` so a `+Z`-facing cell lights up while a `-Z`-facing cell receives no
/// front-facing candidate (exercising the not-selected branch).
fn seed_lights() -> [(Vec3, f32, Vec3); 3] {
    [
        (Vec3::new(0.0, 0.0, 3.0), 2.0, Vec3::new(1.0, 0.8, 0.6)),
        (Vec3::new(2.0, 1.0, 4.0), 1.5, Vec3::new(0.6, 0.7, 1.0)),
        (Vec3::new(-1.0, 2.0, 2.0), 1.0, Vec3::new(0.9, 0.9, 0.9)),
    ]
}

/// An occupied `src` slot carrying a cell's visible-point geometry and `SHARC`
/// checksum, flagged valid, with every seeded energy / sample lane zeroed (the
/// seed kernel overwrites them). Mirrors how the inject pass pre-seeds a slot.
fn seed_occupied_src(
    visible_point: Vec3,
    visible_normal: Vec3,
    checksum: u32,
) -> GpuWorldRestirReservoir {
    GpuWorldRestirReservoir {
        visible_point: visible_point.to_array(),
        w: 0.0,
        visible_normal: visible_normal.to_array(),
        m: 0.0,
        sample_point: [0.0; 3],
        checksum,
        sample_normal: [0.0; 3],
        valid: 1,
        radiance: [0.0; 3],
        _pad0: 0,
    }
}

/// Asserts a single `f32` lane agrees within [`SEED_PARITY_EPS`].
fn seed_approx_scalar(got: f32, expected: f32, slot: usize, lane: &str) {
    let tol = SEED_PARITY_EPS * (1.0 + expected.abs().max(got.abs()));
    assert!(
        (got - expected).abs() <= tol,
        "slot {slot} {lane}: device {got} vs CPU golden {expected} exceeds tolerance {tol}"
    );
}

/// Asserts a `vec3` lane agrees componentwise within [`SEED_PARITY_EPS`].
fn seed_approx_vec(got: [f32; 3], expected: Vec3, slot: usize, lane: &str) {
    for (g, x) in got.iter().zip(expected.to_array()) {
        seed_approx_scalar(*g, x, slot, lane);
    }
}

/// Dispatches `seed_main` once over `src` + `lights` and reads the whole `dst`
/// reservoir table back as raw bytes.
///
/// The pipeline uses an explicit layout (a read-only `src` storage binding, a
/// read-write `dst` storage binding, a read-only light storage binding, and a
/// `32`-byte immediate block) matching the `WESL` declarations exactly, because
/// the seed tunables arrive through the `var<immediate>` path.
fn dispatch_seed(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    src: &[GpuWorldRestirReservoir],
    lights: &[GpuWorldRestirLight],
    params: &GpuWorldRestirSeedParams,
) -> Vec<GpuWorldRestirReservoir> {
    let reservoir_bytes = (src.len() * size_of::<GpuWorldRestirReservoir>()) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("world_restir_seed_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("world_restir_seed_group0"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, true),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("world_restir_seed_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: size_of::<GpuWorldRestirSeedParams>() as u32,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("world_restir_seed_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let src_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("seed_src_reservoirs"),
        contents: bytemuck::cast_slice(src),
        usage: BufferUsages::STORAGE,
    });
    let dst_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("seed_dst_reservoirs"),
        contents: &vec![0u8; reservoir_bytes as usize],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let lights_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("seed_lights"),
        contents: bytemuck::cast_slice(lights),
        usage: BufferUsages::STORAGE,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("seed_group0"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: dst_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: lights_buf.as_entire_binding(),
            },
        ],
    });

    let dst_stage = device.create_buffer(&BufferDescriptor {
        label: Some("seed_dst_stage"),
        size: reservoir_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("seed_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("seed_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(params));
        pass.dispatch_workgroups(
            params
                .capacity
                .max(1)
                .div_ceil(WORLD_RESTIR_SEED_WORKGROUP_SIZE),
            1,
            1,
        );
    }
    encoder.copy_buffer_to_buffer(&dst_buf, 0, &dst_stage, 0, reservoir_bytes);
    queue.submit([encoder.finish()]);

    dst_stage.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = dst_stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let reservoirs = view
        .chunks_exact(size_of::<GpuWorldRestirReservoir>())
        .map(bytemuck::pod_read_unaligned::<GpuWorldRestirReservoir>)
        .collect();
    drop(view);
    dst_stage.unmap();
    reservoirs
}

/// One on-device `seed_main` dispatch must reproduce the authoritative seed
/// `CPU` golden: each occupied slot's streaming-`RIS` reservoir within
/// [`SEED_PARITY_EPS`], and each empty slot copied through byte-for-byte.
///
/// The expectation comes from [`super::shader_tests::seed_mirror::seed_cell`],
/// the serial `CPU` twin the sibling `shader_tests` module already proves
/// bit-equal to the authoritative `Reservoir` golden under a finite-input
/// sweep. This test closes the remaining gap — that the host immediate-data /
/// bind-group wiring and the on-device arithmetic agree with that twin — by
/// binding the real compute pipeline on an actual device. Acquisition is
/// best-effort (see [`try_immediate_device`]); a headless host prints a skip
/// notice and stays green.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a usable device"
)]
fn seed_gpu_matches_cpu_golden() {
    use super::shader_tests::seed_mirror::{self, Light};

    let Some((device, queue)) = try_immediate_device(size_of::<GpuWorldRestirSeedParams>() as u32)
    else {
        eprintln!(
            "seed_gpu_matches_cpu_golden: no wgpu adapter with immediate data, skipping \
             on-device parity"
        );
        return;
    };

    let light_src = seed_lights();
    let gpu_lights: Vec<GpuWorldRestirLight> = light_src
        .iter()
        .map(|&(position, intensity, color)| GpuWorldRestirLight {
            position: position.to_array(),
            intensity,
            color: color.to_array(),
            _pad0: 0.0,
        })
        .collect();
    let mirror_lights: Vec<Light> = light_src
        .iter()
        .map(|&(position, intensity, color)| Light {
            position,
            intensity,
            color,
        })
        .collect();

    // Occupied cells: slot 0 (`+Z`) and slot 1 (`+Y`) face the emitters and
    // seed a surviving sample; slot 2 (`-Z`) faces away so every candidate's
    // geometric term is zero and the slot keeps its geometry with no sample.
    let occupied = [
        (0usize, Vec3::new(0.0, 0.0, 0.0), Vec3::Z, 0x0000_1111u32),
        (1usize, Vec3::new(2.0, 1.0, -1.0), Vec3::Y, 0x0000_2222u32),
        (
            2usize,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::NEG_Z,
            0x0000_3333u32,
        ),
    ];
    let mut src = vec![GpuWorldRestirReservoir::zeroed(); SEED_CAPACITY as usize];
    for &(slot, visible_point, visible_normal, checksum) in &occupied {
        src[slot] = seed_occupied_src(visible_point, visible_normal, checksum);
    }
    // Slot 3 is empty but carries recognisable bytes so the copy-through branch
    // is proven to preserve the whole slot verbatim.
    src[3] = GpuWorldRestirReservoir {
        visible_point: [1.0, 2.0, 3.0],
        w: 4.0,
        visible_normal: [5.0, 6.0, 7.0],
        m: 8.0,
        sample_point: [9.0, 10.0, 11.0],
        checksum: 0x0000_DEAD,
        sample_normal: [12.0, 13.0, 14.0],
        valid: 0,
        radiance: [15.0, 16.0, 17.0],
        _pad0: 0x0000_BEEF,
    };

    let candidate_count = 8u32;
    let frame = 7u32;
    let intensity = 1.0f32;
    let m_cap = 16.0f32;
    let seed_params = GpuWorldRestirSeedParams {
        capacity: SEED_CAPACITY,
        light_count: gpu_lights.len() as u32,
        candidate_count,
        frame,
        intensity,
        m_cap,
        _pad0: 0.0,
        _pad1: 0.0,
    };

    let wgsl = compile_seed_wgsl();
    let entry = find_entry_point(&wgsl, "seed_main");
    let dst = dispatch_seed(
        &device,
        &queue,
        &wgsl,
        &entry,
        &src,
        &gpu_lights,
        &seed_params,
    );
    assert_eq!(
        dst.len(),
        src.len(),
        "device dst table length must match the src table"
    );

    let mut saw_selected = false;
    let mut saw_not_selected = false;
    for slot in 0..SEED_CAPACITY as usize {
        let got = dst[slot];

        if src[slot].valid == 0 {
            assert_eq!(
                bytemuck::bytes_of(&got),
                bytemuck::bytes_of(&src[slot]),
                "empty slot {slot} must be copied through byte-for-byte"
            );
            continue;
        }

        let visible_point = Vec3::from_array(src[slot].visible_point);
        let visible_normal = Vec3::from_array(src[slot].visible_normal);
        let result = seed_mirror::seed_cell(
            visible_point,
            visible_normal,
            &mirror_lights,
            candidate_count,
            frame,
            slot as u32,
            intensity,
            m_cap,
        );

        assert_eq!(got.valid, 1, "occupied slot {slot} must stay valid");
        assert_eq!(
            got.checksum, src[slot].checksum,
            "occupied slot {slot} must preserve its SHARC checksum"
        );
        assert_eq!(got._pad0, 0, "occupied slot {slot} pad word must be zero");

        match result.sample {
            Some(sample) => {
                saw_selected = true;
                assert!(
                    result.m > 0.0,
                    "slot {slot} selected a candidate but reports zero confidence"
                );
                seed_approx_vec(
                    got.visible_point,
                    sample.visible_point,
                    slot,
                    "visible_point",
                );
                seed_approx_vec(
                    got.visible_normal,
                    sample.visible_normal,
                    slot,
                    "visible_normal",
                );
                seed_approx_vec(got.sample_point, sample.sample_point, slot, "sample_point");
                seed_approx_vec(
                    got.sample_normal,
                    sample.sample_normal,
                    slot,
                    "sample_normal",
                );
                seed_approx_vec(got.radiance, sample.radiance, slot, "radiance");
                seed_approx_scalar(got.w, result.w, slot, "w");
                seed_approx_scalar(got.m, result.m, slot, "m");
            }
            None => {
                saw_not_selected = true;
                // Not-selected occupied slot: cell geometry preserved, every
                // seeded energy / sample lane cleared.
                seed_approx_vec(got.visible_point, visible_point, slot, "visible_point");
                seed_approx_vec(got.visible_normal, visible_normal, slot, "visible_normal");
                seed_approx_vec(got.sample_point, Vec3::ZERO, slot, "sample_point");
                seed_approx_vec(got.sample_normal, Vec3::ZERO, slot, "sample_normal");
                seed_approx_vec(got.radiance, Vec3::ZERO, slot, "radiance");
                seed_approx_scalar(got.w, 0.0, slot, "w");
                seed_approx_scalar(got.m, 0.0, slot, "m");
            }
        }
    }

    assert!(
        saw_selected,
        "the fixture must exercise the selected serialisation branch"
    );
    assert!(
        saw_not_selected,
        "the fixture must exercise the not-selected serialisation branch"
    );
}
