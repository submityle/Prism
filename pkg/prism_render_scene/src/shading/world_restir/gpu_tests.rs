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

use super::abi::{GpuWorldRestirInjectParams, GpuWorldRestirInjectPoint, GpuWorldRestirReservoir};

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

/// Best-effort acquisition of a native compute device and queue with push
/// constants.
///
/// Returns `None` (rather than panicking) when no adapter is available or the
/// adapter cannot satisfy the `IMMEDIATES` feature / `48`-byte immediate block,
/// so the suite stays green on headless hosts; on a machine with a real `GPU`
/// that supports immediate data this yields a live device the parity test
/// dispatches against.
fn try_inject_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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
    if limits.max_immediate_size < size_of::<GpuWorldRestirInjectParams>() as u32 {
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
