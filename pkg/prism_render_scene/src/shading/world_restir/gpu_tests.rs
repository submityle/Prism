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
    GpuWorldRestirFillParams, GpuWorldRestirInjectParams, GpuWorldRestirInjectPoint,
    GpuWorldRestirLight, GpuWorldRestirReservoir, GpuWorldRestirSeedParams,
    WORLD_RESTIR_SEED_WORKGROUP_SIZE, WORLD_RESTIR_WORKGROUP_SIZE,
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
    let reservoir_bytes = size_of_val(src) as u64;

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

/// Reservoir-table capacity exercised by the fill parity fixture.
///
/// `256` slots keep the dispatch tiny (four `64`-wide workgroups) while giving
/// the handful of occupied cells pairwise-disjoint buckets, so the parallel
/// on-device result is order-independent and byte-comparable to the serial
/// `CPU` twin.
const FILL_CAPACITY: u32 = 256;

/// Serial `CPU` twin of `world_restir_fill.wesl`'s `fill_main`: the
/// authoritative expectation the on-device fill dispatch is checked against.
///
/// This is an arm-for-arm port of the `WESL` kernel's `GRIS` spatial-reuse
/// pass. The hash/`checksum`/bucket math reuses the authoritative golden
/// [`prism_render_shading::gi::world_restir::spatial_hash`] (proven bit-equal
/// to the `WESL` 2x32 port by the sibling `shader_tests`), and the `RIS`
/// estimator (`luminance`, `geometric_term`, `target_function`) is copied from
/// the seed twin verbatim so both kernels resolve the same density.
mod fill_mirror {
    use super::super::shader_tests::seed_mirror::{fmix32, rng01};
    use super::{GpuWorldRestirReservoir, PROBE_LIMIT};
    use bevy_math::{IVec3, Vec3};
    use prism_render_shading::gi::screen_probe::restir::GiSample;
    use prism_render_shading::gi::world_restir::spatial_hash::{self, HashGridKey, HashGridParams};

    /// Smallest positive normal `f32` (golden `f32::MIN_POSITIVE`, `WESL` `MIN_POSITIVE`).
    const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;
    /// Rec. 709 luminance weights (golden/`WESL` `LUMA_R` / `LUMA_G` / `LUMA_B`).
    const LUMA_R: f32 = 0.212_639;
    const LUMA_G: f32 = 0.715_169;
    const LUMA_B: f32 = 0.072_192;

    /// Rec. 709 luminance of a linear RGB triple (`WESL` `luminance`).
    fn luminance(rgb: Vec3) -> f32 {
        LUMA_R * rgb.x.max(0.0) + LUMA_G * rgb.y.max(0.0) + LUMA_B * rgb.z.max(0.0)
    }

    /// Surface-to-surface geometry factor `cos_v * cos_s / dist^2` (`WESL`
    /// `geometric_term`); zero for degenerate / back-facing / `NaN` pairs. Uses
    /// `dist_sq.sqrt().recip()` to reproduce the golden's exact IEEE reciprocal
    /// square root bit-for-bit.
    fn geometric_term(
        visible_point: Vec3,
        visible_normal: Vec3,
        sample_point: Vec3,
        sample_normal: Vec3,
    ) -> f32 {
        let delta = sample_point - visible_point;
        let dist_sq = delta.length_squared();
        if dist_sq > MIN_POSITIVE {
            let inv_dist = dist_sq.sqrt().recip();
            let dir = delta * inv_dist;
            let cos_v = visible_normal.dot(dir).max(0.0);
            let cos_s = sample_normal.dot(-dir).max(0.0);
            let term = cos_v * cos_s / dist_sq;
            // `!term.is_nan()` mirrors the `WESL` `term == term` NaN reject.
            if !term.is_nan() {
                return term.max(0.0);
            }
        }
        0.0
    }

    /// Scalar resampling target `p_hat` (`WESL` `target_function`).
    fn target_function(s: &GiSample) -> f32 {
        let g = geometric_term(
            s.visible_point,
            s.visible_normal,
            s.sample_point,
            s.sample_normal,
        );
        let t = luminance(s.radiance) * g;
        if t.is_nan() {
            0.0
        } else {
            t.max(0.0)
        }
    }

    /// Reconnection-shift target: keep the neighbour's secondary point / normal
    /// / radiance, re-evaluate from the destination's visible point (`WESL`
    /// `reconnection_target`).
    fn reconnection_target(neighbor: &GiSample, visible_point: Vec3, visible_normal: Vec3) -> f32 {
        let mut shifted = *neighbor;
        shifted.visible_point = visible_point;
        shifted.visible_normal = visible_normal;
        target_function(&shifted)
    }

    /// Deserialises a reservoir slot's five vec3 lanes into a `GiSample` (`WESL`
    /// `load_sample`).
    fn load_sample(r: &GpuWorldRestirReservoir) -> GiSample {
        GiSample {
            visible_point: Vec3::from_array(r.visible_point),
            visible_normal: Vec3::from_array(r.visible_normal),
            sample_point: Vec3::from_array(r.sample_point),
            sample_normal: Vec3::from_array(r.sample_normal),
            radiance: Vec3::from_array(r.radiance),
        }
    }

    /// Jittered non-zero integer neighbour offset in `[-radius, radius]^3`
    /// (`WESL` `fill_main` ring step); nudged off-center so a neighbour is
    /// always distinct.
    pub fn neighbor_offset(frame: u32, slot: u32, k: u32, radius: i32) -> IVec3 {
        let span = (2 * radius + 1) as u32;
        let rx = (fmix32(
            frame
                .wrapping_mul(0x27d4_eb4f)
                .wrapping_add(slot.wrapping_mul(0x9e37_79b9))
                .wrapping_add(k.wrapping_mul(0x1656_67b1)),
        ) % span) as i32
            - radius;
        let ry = (fmix32(
            frame
                .wrapping_mul(0x85eb_ca6b)
                .wrapping_add(slot.wrapping_mul(0xc2b2_ae35))
                .wrapping_add(k.wrapping_mul(0x27d4_eb2f)),
        ) % span) as i32
            - radius;
        let rz = (fmix32(
            frame
                .wrapping_mul(0xc2b2_ae35)
                .wrapping_add(slot.wrapping_mul(0x1656_67b1))
                .wrapping_add(k.wrapping_mul(0x85eb_ca6b)),
        ) % span) as i32
            - radius;
        let mut offset = IVec3::new(rx, ry, rz);
        if offset == IVec3::ZERO {
            offset.x = 1;
        }
        offset
    }

    /// Pools one occupied reservoir slot with a jittered ring of spatial
    /// neighbours under `GRIS` reuse and re-finalises `W` (`WESL` `fill_main`);
    /// an invalid slot is returned unchanged (copy-through).
    #[expect(
        clippy::too_many_arguments,
        reason = "arm-for-arm port of the WESL fill_main tunable set keeps each immediate mapping explicit"
    )]
    pub fn fill_slot(
        src: &[GpuWorldRestirReservoir],
        slot: u32,
        camera: Vec3,
        params: &HashGridParams,
        frame: u32,
        spatial_samples: u32,
        spatial_radius: i32,
        m_cap: f32,
    ) -> GpuWorldRestirReservoir {
        let capacity = src.len() as u32;
        let center = src[slot as usize];
        if center.valid == 0 {
            return center;
        }

        let visible_point = Vec3::from_array(center.visible_point);
        let visible_normal = Vec3::from_array(center.visible_normal);
        let center_key = spatial_hash::compute_key(visible_point, visible_normal, camera, params);

        let mut acc_sample = load_sample(&center);
        let mut acc_m = center.m;
        let center_w = center.w;
        let center_p_hat = target_function(&acc_sample);
        let mut acc_w_sum = center_w * acc_m * center_p_hat;

        let radius = spatial_radius.max(0);
        for k in 0..spatial_samples {
            if radius <= 0 {
                break;
            }
            let offset = neighbor_offset(frame, slot, k, radius);
            let nkey = HashGridKey {
                cell_coord: center_key.cell_coord + offset,
                level: center_key.level,
                normal_bin: center_key.normal_bin,
            };
            let ncs = spatial_hash::checksum(&nkey);
            let base = spatial_hash::bucket_index(&nkey, capacity);

            let steps = PROBE_LIMIT.min(capacity);
            let mut found: Option<GpuWorldRestirReservoir> = None;
            for i in 0..steps {
                let idx = ((base + i) % capacity) as usize;
                let cand = src[idx];
                if cand.valid == 0 {
                    break;
                }
                if cand.checksum == ncs {
                    found = Some(cand);
                    break;
                }
            }
            let Some(neighbor) = found else {
                continue;
            };

            let neighbor_m = neighbor.m;
            if neighbor_m <= 0.0 || neighbor_m.is_nan() {
                continue;
            }
            acc_m += neighbor_m;
            let neighbor_sample = load_sample(&neighbor);
            let p_hat = reconnection_target(&neighbor_sample, visible_point, visible_normal);
            let rw = neighbor_m * p_hat.max(0.0) * neighbor.w.max(0.0);
            if rw <= 0.0 || rw.is_nan() {
                continue;
            }
            acc_w_sum += rw;
            let u = rng01(frame, slot, k);
            if u * acc_w_sum <= rw {
                acc_sample = neighbor_sample;
            }
        }

        if m_cap >= 0.0 && acc_m > m_cap {
            acc_m = m_cap;
        }
        let final_p_hat = target_function(&acc_sample);
        let mut final_w = 0.0;
        if acc_m > 0.0 && !acc_w_sum.is_nan() && final_p_hat > 0.0 {
            let w = (acc_w_sum / acc_m) / final_p_hat;
            final_w = if !w.is_nan() && w >= 0.0 { w } else { 0.0 };
        }

        GpuWorldRestirReservoir {
            visible_point: acc_sample.visible_point.to_array(),
            w: final_w,
            visible_normal: acc_sample.visible_normal.to_array(),
            m: acc_m,
            sample_point: acc_sample.sample_point.to_array(),
            checksum: center.checksum,
            sample_normal: acc_sample.sample_normal.to_array(),
            valid: 1,
            radiance: acc_sample.radiance.to_array(),
            _pad0: 0,
        }
    }
}

/// Compiles `world_restir_fill.wesl` and returns its `Wgsl` translation.
fn compile_fill_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5753_5244_5f46_494c_0001),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../shaders/world_restir_fill.wesl"),
            "embedded://prism_render_scene/shaders/world_restir_fill.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("world_restir_fill.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Dispatches `fill_main` once over `src` and reads the whole `dst` reservoir
/// table back as raw bytes.
///
/// The pipeline uses an explicit layout (a read-only `src` storage binding, a
/// read-write `dst` storage binding, and a `64`-byte immediate block) matching
/// the `WESL` declarations exactly; unlike the seed pass the fill kernel pools
/// resident reservoirs only, so it binds no light list.
fn dispatch_fill(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    entry: &str,
    src: &[GpuWorldRestirReservoir],
    params: &GpuWorldRestirFillParams,
) -> Vec<GpuWorldRestirReservoir> {
    let reservoir_bytes = size_of_val(src) as u64;

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("world_restir_fill_parity"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("world_restir_fill_group0"),
        entries: &[storage_entry(0, true), storage_entry(1, false)],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("world_restir_fill_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: size_of::<GpuWorldRestirFillParams>() as u32,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("world_restir_fill_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let src_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("fill_src_reservoirs"),
        contents: bytemuck::cast_slice(src),
        usage: BufferUsages::STORAGE,
    });
    let dst_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("fill_dst_reservoirs"),
        contents: &vec![0u8; reservoir_bytes as usize],
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("fill_group0"),
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
        ],
    });

    let dst_stage = device.create_buffer(&BufferDescriptor {
        label: Some("fill_dst_stage"),
        size: reservoir_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("fill_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fill_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(params));
        pass.dispatch_workgroups(
            params.capacity.max(1).div_ceil(WORLD_RESTIR_WORKGROUP_SIZE),
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

/// One on-device `fill_main` dispatch must reproduce the authoritative fill
/// `CPU` golden: each occupied slot's `GRIS`-pooled reservoir within
/// [`SEED_PARITY_EPS`], and each empty slot copied through byte-for-byte.
///
/// The expectation comes from [`fill_mirror::fill_slot`], the serial `CPU` twin
/// built on the authoritative golden
/// [`prism_render_shading::gi::world_restir::spatial_hash`] and the seed-twin
/// `RIS` estimator. The fixture plants a center cell plus a matching-checksum
/// neighbour at the exact bucket the center's spatial ring probes, so the merge
/// branch (confidence growth + reconnection-shift replacement) and the empty
/// copy-through branch are both exercised. Acquisition is best-effort (see
/// [`try_immediate_device`]); a headless host prints a skip notice and stays
/// green.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a usable device"
)]
fn fill_gpu_matches_cpu_golden() {
    use prism_render_shading::gi::world_restir::spatial_hash::HashGridKey;

    let Some((device, queue)) = try_immediate_device(size_of::<GpuWorldRestirFillParams>() as u32)
    else {
        eprintln!(
            "fill_gpu_matches_cpu_golden: no wgpu adapter with immediate data, skipping \
             on-device parity"
        );
        return;
    };

    let camera = Vec3::ZERO;
    let params = HashGridParams::DEFAULT;
    let frame = 7u32;
    let spatial_samples = 1u32;
    let spatial_radius = 1i32;
    let m_cap = 16.0f32;
    let center_slot = 10u32;
    let witness_slot = 20usize;

    // Center cell geometry: a +Z normal at the origin.
    let center_vp = Vec3::ZERO;
    let center_vn = Vec3::Z;

    // Resolve the neighbour's open-addressed home slot from the authoritative
    // golden hash so the fixture plants a matching-checksum neighbour exactly
    // where the center's spatial ring will probe for it.
    let center_key = spatial_hash::compute_key(center_vp, center_vn, camera, &params);
    let offset = fill_mirror::neighbor_offset(frame, center_slot, 0, spatial_radius);
    let nkey = HashGridKey {
        cell_coord: center_key.cell_coord + offset,
        level: center_key.level,
        normal_bin: center_key.normal_bin,
    };
    let ncs = spatial_hash::checksum(&nkey);
    let base = spatial_hash::bucket_index(&nkey, FILL_CAPACITY);
    assert_ne!(
        base, center_slot,
        "neighbour must not alias the center slot"
    );
    assert_ne!(
        base as usize, witness_slot,
        "neighbour must not alias the witness slot"
    );

    let mut src = vec![GpuWorldRestirReservoir::zeroed(); FILL_CAPACITY as usize];
    src[center_slot as usize] = GpuWorldRestirReservoir {
        visible_point: center_vp.to_array(),
        w: 1.0,
        visible_normal: center_vn.to_array(),
        m: 1.0,
        sample_point: [0.0, 0.0, 0.0],
        checksum: 0x0000_ABCD,
        sample_normal: Vec3::Z.to_array(),
        valid: 1,
        radiance: [0.0, 0.0, 0.0],
        _pad0: 0,
    };
    // Neighbour: a radiant off-axis sample whose reconnection target is
    // positive (so it wins the center's resampling test), with stored visible
    // geometry that makes the pooled reservoir's final target zero.
    src[base as usize] = GpuWorldRestirReservoir {
        visible_point: [7.0, 8.0, 9.0],
        w: 3.0,
        visible_normal: [0.0, 0.0, 0.0],
        m: 2.0,
        sample_point: [0.0, 0.0, 2.0],
        checksum: ncs,
        sample_normal: Vec3::NEG_Z.to_array(),
        valid: 1,
        radiance: [1.0, 1.0, 1.0],
        _pad0: 0,
    };
    // Empty witness slot carrying recognisable bytes so the copy-through branch
    // is proven to preserve the whole slot verbatim.
    src[witness_slot] = GpuWorldRestirReservoir {
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

    let fill_params = GpuWorldRestirFillParams {
        camera_position: camera.to_array(),
        capacity: FILL_CAPACITY,
        jitter: params.jitter.to_array(),
        light_count: 0,
        base_cell_size: params.base_cell_size,
        level_scale: params.level_scale,
        normal_resolution: params.normal_resolution,
        m_cap,
        intensity: 1.0,
        frame,
        spatial_samples,
        spatial_radius: spatial_radius as u32,
    };

    let golden: Vec<GpuWorldRestirReservoir> = (0..FILL_CAPACITY)
        .map(|slot| {
            fill_mirror::fill_slot(
                &src,
                slot,
                camera,
                &params,
                frame,
                spatial_samples,
                spatial_radius,
                m_cap,
            )
        })
        .collect();

    // Sanity: the fixture must actually exercise a GRIS merge at the center
    // (confidence 1 + 2 = 3, and the neighbour's visible geometry won).
    seed_approx_scalar(
        golden[center_slot as usize].m,
        3.0,
        center_slot as usize,
        "m",
    );
    seed_approx_vec(
        golden[center_slot as usize].visible_point,
        Vec3::new(7.0, 8.0, 9.0),
        center_slot as usize,
        "visible_point",
    );

    let wgsl = compile_fill_wgsl();
    let entry = find_entry_point(&wgsl, "fill_main");
    let dst = dispatch_fill(&device, &queue, &wgsl, &entry, &src, &fill_params);
    assert_eq!(
        dst.len(),
        src.len(),
        "device dst table length must match the src table"
    );

    let mut saw_copy_through = false;
    let mut saw_merged = false;
    for slot in 0..FILL_CAPACITY as usize {
        let got = dst[slot];
        let want = golden[slot];

        if src[slot].valid == 0 {
            assert_eq!(
                bytemuck::bytes_of(&got),
                bytemuck::bytes_of(&src[slot]),
                "empty slot {slot} must be copied through byte-for-byte"
            );
            assert_eq!(
                bytemuck::bytes_of(&want),
                bytemuck::bytes_of(&src[slot]),
                "golden empty slot {slot} must also copy through"
            );
            saw_copy_through = true;
            continue;
        }

        assert_eq!(got.valid, 1, "occupied slot {slot} must stay valid");
        assert_eq!(
            got.checksum, want.checksum,
            "occupied slot {slot} must preserve its SHARC checksum"
        );
        assert_eq!(got._pad0, 0, "occupied slot {slot} pad word must be zero");
        seed_approx_vec(
            got.visible_point,
            Vec3::from_array(want.visible_point),
            slot,
            "visible_point",
        );
        seed_approx_vec(
            got.visible_normal,
            Vec3::from_array(want.visible_normal),
            slot,
            "visible_normal",
        );
        seed_approx_vec(
            got.sample_point,
            Vec3::from_array(want.sample_point),
            slot,
            "sample_point",
        );
        seed_approx_vec(
            got.sample_normal,
            Vec3::from_array(want.sample_normal),
            slot,
            "sample_normal",
        );
        seed_approx_vec(
            got.radiance,
            Vec3::from_array(want.radiance),
            slot,
            "radiance",
        );
        seed_approx_scalar(got.w, want.w, slot, "w");
        seed_approx_scalar(got.m, want.m, slot, "m");

        if slot == center_slot as usize {
            saw_merged = true;
        }
    }

    assert!(
        saw_copy_through,
        "the fixture must exercise the copy-through branch"
    );
    assert!(
        saw_merged,
        "the fixture must exercise the GRIS merge branch"
    );
}
