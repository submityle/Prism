//! Bounded real-hardware acceptance for the three-pass hair raster WESL.
//!
//! This is a direct wgpu replay of the production shader ABI: ShaderCache
//! compiles the embedded WESL, one group-0 layout binds the segment/depth/id/
//! coverage storage buffers, each pass receives the 20-byte immediate block,
//! and all three vis buffers are copied to mapped staging buffers. It verifies
//! nearest-depth occlusion, deterministic equal-depth ID selection, coverage
//! publication for the winning ID, and untouched clear sentinels.

use std::sync::{mpsc, Arc};

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::raster::{HairRasterDispatchParams, HairSoftRasterAbi};
use prism_render_driver_wgpu::{require_hardware_gpu, WgpuInstance};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, DeviceDescriptor, Features, MapMode, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PollType, PowerPreference, ShaderModuleDescriptor, ShaderSource,
    ShaderStages,
};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 12;
const PIXELS: usize = (WIDTH * HEIGHT) as usize;
const DEPTH_EMPTY: u32 = u32::MAX;
const ID_EMPTY: u32 = u32::MAX;
const COVERAGE_EMPTY: u32 = 0;
const COVERAGE_QUANT: f32 = 65535.0;
const PARAM_BYTES: u32 = 20;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSegment {
    p0: [f32; 2],
    p1: [f32; 2],
    depth0: f32,
    depth1: f32,
    width_px: f32,
    seg_id: u32,
}

struct GpuHost {
    // Keep the instance alive for the adapter/device lifetime.
    _instance: Arc<WgpuInstance>,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("hair raster source is WESL"),
    }
}

fn compile_hair_raster_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_4841_4952_5f52_5354_0003),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../shaders/hair_raster.wesl"),
            "embedded://prism_render_scene/shaders/hair_raster.wesl",
        ),
    );
    (*cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("hair_raster.wesl failed ShaderCache compilation: {error}")))
    .clone()
}

fn entry_name(wgsl: &str, needle: &str) -> String {
    for line in wgsl.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("fn ")
            && let Some(paren) = rest.find('(')
        {
            let name = &rest[..paren];
            if name.contains(needle) {
                return name.to_owned();
            }
        }
    }
    panic!("compiled hair raster WGSL has no entry point containing {needle:?}");
}

fn hardware_host_or_skip() -> Option<GpuHost> {
    let skip = |why: &str| {
        assert!(!require_hardware_gpu(), "PRISM_REQUIRE_GPU=1: {why}");
        eprintln!("SKIP_UNVERIFIED hair_raster physical GPU acceptance: {why}");
        None
    };
    let instance = Arc::new(WgpuInstance::new());
    let Some(adapter) = instance.request_adapter(PowerPreference::HighPerformance, false) else {
        return skip("no adapter");
    };
    let info = adapter.get_info();
    if !matches!(
        info.device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
    ) {
        return skip(&format!(
            "adapter is not classified as physical hardware: {info:?}"
        ));
    }
    if !adapter.features().contains(Features::IMMEDIATES) {
        return skip("adapter does not support IMMEDIATES");
    }
    if adapter.limits().max_immediate_size < PARAM_BYTES {
        return skip("adapter max_immediate_size is below 20 bytes");
    }
    eprintln!("hair_raster REAL HARDWARE GPU: {info:?}");
    let (device, queue) = match block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("hair_raster acceptance device"),
        required_features: Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: PARAM_BYTES,
            ..Default::default()
        },
        ..Default::default()
    })) {
        Ok(device_queue) => device_queue,
        Err(error) => return skip(&format!("device request failed: {error}")),
    };
    Some(GpuHost {
        _instance: instance,
        device,
        queue,
    })
}

fn staging_buffer(host: &GpuHost, label: &str) -> wgpu::Buffer {
    let byte_len = (PIXELS * size_of::<u32>()) as u64;
    host.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn map_words(host: &GpuHost, staging: &wgpu::Buffer) -> Vec<u32> {
    let (tx, rx) = mpsc::channel();
    staging.slice(..).map_async(MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    host.device
        .poll(PollType::wait_indefinitely())
        .expect("hair raster staging readback poll");
    rx.recv()
        .expect("hair raster staging map callback")
        .expect("hair raster staging map succeeds");
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .expect("hair raster staging range is mapped");
    let words = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    words
}

#[expect(
    clippy::too_many_lines,
    reason = "the bounded replay keeps ABI setup, ordered dispatch, and readback auditable"
)]
fn replay(host: &GpuHost, wgsl: &str, segments: &[GpuSegment]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    assert!(segments.len() <= u32::MAX as usize);
    assert!(WIDTH
        .checked_mul(HEIGHT)
        .is_some_and(|pixels| pixels as usize == PIXELS));
    let device = &host.device;
    // Storage bindings must have a non-zero size even for an empty frame; the
    // production segment_count immediate keeps this sentinel record unread.
    let empty_segment = GpuSegment::zeroed();
    let segment_upload = if segments.is_empty() {
        bytemuck::bytes_of(&empty_segment)
    } else {
        bytemuck::cast_slice(segments)
    };
    let segment_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("hair_raster_segments"),
        contents: segment_upload,
        usage: BufferUsages::STORAGE,
    });
    // Per-frame clear contract: depth and ID use all-ones sentinels; zero
    // coverage means no fragment. These are initialized before every replay.
    let depth_init = vec![DEPTH_EMPTY; PIXELS];
    let id_init = vec![ID_EMPTY; PIXELS];
    let coverage_init = vec![COVERAGE_EMPTY; PIXELS];
    let depth = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("hair_raster_depth_clear"),
        contents: bytemuck::cast_slice(&depth_init),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let id = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("hair_raster_id_clear"),
        contents: bytemuck::cast_slice(&id_init),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });
    let coverage = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("hair_raster_coverage_clear"),
        contents: bytemuck::cast_slice(&coverage_init),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    // Exact group-0 ABI from hair_raster.wesl:
    // 0 = read-only segments; 1 = atomic<u32> depth; 2 = atomic<u32> ID;
    // 3 = atomic<u32> coverage. All three vis arrays are read_write storage.
    let storage = |binding, read_only| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("hair_raster_group0"),
        entries: &[
            storage(0, true),
            storage(1, false),
            storage(2, false),
            storage(3, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("hair_raster_pipeline_layout"),
        bind_group_layouts: &[Some(&group_layout)],
        immediate_size: PARAM_BYTES,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("hair_raster_shader_cache_wgsl"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = |entry| {
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        })
    };
    let depth_entry = entry_name(wgsl, "hair_raster_depth");
    let resolve_entry = entry_name(wgsl, "hair_raster_resolve");
    let publish_entry = entry_name(wgsl, "hair_raster_publish");
    let depth_pipeline = pipeline(&depth_entry);
    let resolve_pipeline = pipeline(&resolve_entry);
    let publish_pipeline = pipeline(&publish_entry);
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("hair_raster_group0"),
        layout: &group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: segment_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: depth.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: id.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: coverage.as_entire_binding(),
            },
        ],
    });
    let params = HairRasterDispatchParams::try_new(
        [WIDTH, HEIGHT],
        HairSoftRasterAbi::default(),
        segments.len(),
    )
    .expect("test segments fit the production raster dispatch ABI");
    let workgroups = params
        .dispatch_workgroups(device.limits().max_compute_workgroups_per_dimension)
        .expect("test dispatch fits the device workgroup limit");
    let immediate_bytes = params.encode_le();

    let depth_staging = staging_buffer(host, "hair_raster_depth_readback");
    let id_staging = staging_buffer(host, "hair_raster_id_readback");
    let coverage_staging = staging_buffer(host, "hair_raster_coverage_readback");
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("hair_raster_three_ordered_passes"),
    });
    for (label, pipeline) in [
        ("hair_raster_depth", &depth_pipeline),
        ("hair_raster_resolve", &resolve_pipeline),
        ("hair_raster_publish", &publish_pipeline),
    ] {
        if workgroups > 0 {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_immediates(0, &immediate_bytes);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
    let byte_len = (PIXELS * size_of::<u32>()) as u64;
    for (source, destination) in [
        (&depth, &depth_staging),
        (&id, &id_staging),
        (&coverage, &coverage_staging),
    ] {
        encoder.copy_buffer_to_buffer(source, 0, destination, 0, byte_len);
    }
    host.queue.submit([encoder.finish()]);

    (
        map_words(host, &depth_staging),
        map_words(host, &id_staging),
        map_words(host, &coverage_staging),
    )
}

#[test]
fn hair_raster_shader_immediate_is_the_scalar_twenty_byte_cpu_abi() {
    let source = include_str!("../../shaders/hair_raster.wesl");
    let params = source
        .split_once("struct HairRasterParamsGpu {")
        .and_then(|(_, rest)| rest.split_once("}\n"))
        .map(|(body, _)| body)
        .expect("hair raster shader declares its immediate ABI");
    let fields: Vec<_> = params
        .lines()
        .filter_map(|line| line.trim().strip_suffix(","))
        .filter(|line| !line.starts_with("///"))
        .collect();
    assert_eq!(
        fields,
        [
            "viewport_width: u32",
            "viewport_height: u32",
            "coverage_epsilon: f32",
            "subpixel_samples: u32",
            "segment_count: u32",
        ]
    );
    assert_eq!(
        std::mem::size_of::<HairRasterDispatchParams>(),
        PARAM_BYTES as usize
    );
    assert_eq!(std::mem::offset_of!(GpuSegment, seg_id), 28);
    assert_eq!(std::mem::size_of::<GpuSegment>(), 32);
}

fn pixel(x: u32, y: u32) -> usize {
    (y * WIDTH + x) as usize
}

fn cpu_capsule_coverage(
    center: [f32; 2],
    a: [f32; 2],
    b: [f32; 2],
    radius: f32,
    samples: u32,
) -> (f32, f32) {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let denom = ab[0] * ab[0] + ab[1] * ab[1];
    let mut inside = 0.0;
    let mut best_t = 0.0;
    let mut best_dist2 = f32::MAX;
    for i in 0..samples.max(1) {
        let fi = i as f32;
        let ox = (fi * 0.618_034).fract() - 0.5;
        let oy = (fi * 0.381_966).fract() - 0.5;
        let sample = [center[0] + ox, center[1] + oy];
        let ap = [sample[0] - a[0], sample[1] - a[1]];
        let t = if denom <= 1.0e-12 {
            0.0
        } else {
            ((ap[0] * ab[0] + ap[1] * ab[1]) / denom).clamp(0.0, 1.0)
        };
        let dx = sample[0] - (a[0] + ab[0] * t);
        let dy = sample[1] - (a[1] + ab[1] * t);
        let dist2 = dx * dx + dy * dy;
        let distance = dist2.sqrt();
        inside += (radius + 0.5 - distance).clamp(0.0, 1.0) / samples.max(1) as f32;
        if dist2 < best_dist2 {
            best_dist2 = dist2;
            best_t = t;
        }
    }
    (inside.clamp(0.0, 1.0), best_t)
}

fn cpu_oracle(segments: &[GpuSegment]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut depth = vec![DEPTH_EMPTY; PIXELS];
    let mut ids = vec![ID_EMPTY; PIXELS];
    let mut coverage = vec![COVERAGE_EMPTY; PIXELS];
    for segment in segments {
        let radius = segment.width_px.max(0.0) * 0.5;
        if radius <= 0.0 {
            continue;
        }
        let lo_x = segment.p0[0].min(segment.p1[0]) - (radius + 1.0);
        let lo_y = segment.p0[1].min(segment.p1[1]) - (radius + 1.0);
        let hi_x = segment.p0[0].max(segment.p1[0]) + (radius + 1.0);
        let hi_y = segment.p0[1].max(segment.p1[1]) + (radius + 1.0);
        let x0 = (lo_x.floor() as i32).max(0);
        let y0 = (lo_y.floor() as i32).max(0);
        let x1 = (hi_x.ceil() as i32).min(WIDTH as i32 - 1);
        let y1 = (hi_y.ceil() as i32).min(HEIGHT as i32 - 1);
        if x0 > x1 || y0 > y1 {
            continue;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (cov, t) = cpu_capsule_coverage(
                    [x as f32 + 0.5, y as f32 + 0.5],
                    segment.p0,
                    segment.p1,
                    radius,
                    HairSoftRasterAbi::default().subpixel_samples,
                );
                if cov < HairSoftRasterAbi::default().coverage_epsilon {
                    continue;
                }
                let z = segment.depth0 + (segment.depth1 - segment.depth0) * t;
                let quantized_depth = (z.clamp(0.0, 1.0) * 16_777_215.0) as u32;
                let index = pixel(x as u32, y as u32);
                depth[index] = depth[index].min(quantized_depth);
            }
        }
    }
    for segment in segments {
        let radius = segment.width_px.max(0.0) * 0.5;
        if radius <= 0.0 {
            continue;
        }
        let x0 = ((segment.p0[0].min(segment.p1[0]) - (radius + 1.0)).floor() as i32).max(0);
        let y0 = ((segment.p0[1].min(segment.p1[1]) - (radius + 1.0)).floor() as i32).max(0);
        let x1 = ((segment.p0[0].max(segment.p1[0]) + (radius + 1.0)).ceil() as i32)
            .min(WIDTH as i32 - 1);
        let y1 = ((segment.p0[1].max(segment.p1[1]) + (radius + 1.0)).ceil() as i32)
            .min(HEIGHT as i32 - 1);
        if x0 > x1 || y0 > y1 {
            continue;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (cov, t) = cpu_capsule_coverage(
                    [x as f32 + 0.5, y as f32 + 0.5],
                    segment.p0,
                    segment.p1,
                    radius,
                    HairSoftRasterAbi::default().subpixel_samples,
                );
                if cov < HairSoftRasterAbi::default().coverage_epsilon {
                    continue;
                }
                let z = segment.depth0 + (segment.depth1 - segment.depth0) * t;
                let quantized_depth = (z.clamp(0.0, 1.0) * 16_777_215.0) as u32;
                let index = pixel(x as u32, y as u32);
                if depth[index] == quantized_depth {
                    ids[index] = ids[index].min(segment.seg_id);
                }
            }
        }
    }
    for segment in segments {
        let radius = segment.width_px.max(0.0) * 0.5;
        if radius <= 0.0 {
            continue;
        }
        let x0 = ((segment.p0[0].min(segment.p1[0]) - (radius + 1.0)).floor() as i32).max(0);
        let y0 = ((segment.p0[1].min(segment.p1[1]) - (radius + 1.0)).floor() as i32).max(0);
        let x1 = ((segment.p0[0].max(segment.p1[0]) + (radius + 1.0)).ceil() as i32)
            .min(WIDTH as i32 - 1);
        let y1 = ((segment.p0[1].max(segment.p1[1]) + (radius + 1.0)).ceil() as i32)
            .min(HEIGHT as i32 - 1);
        if x0 > x1 || y0 > y1 {
            continue;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (cov, t) = cpu_capsule_coverage(
                    [x as f32 + 0.5, y as f32 + 0.5],
                    segment.p0,
                    segment.p1,
                    radius,
                    HairSoftRasterAbi::default().subpixel_samples,
                );
                if cov < HairSoftRasterAbi::default().coverage_epsilon {
                    continue;
                }
                let z = segment.depth0 + (segment.depth1 - segment.depth0) * t;
                let quantized_depth = (z.clamp(0.0, 1.0) * 16_777_215.0) as u32;
                let index = pixel(x as u32, y as u32);
                if depth[index] == quantized_depth && ids[index] == segment.seg_id {
                    let quantized_coverage = (cov.clamp(0.0, 1.0) * COVERAGE_QUANT) as u32;
                    coverage[index] = coverage[index].max(quantized_coverage);
                }
            }
        }
    }
    (depth, ids, coverage)
}

fn make_segment(y: f32, depth: f32, id: u32) -> GpuSegment {
    GpuSegment {
        p0: [2.0, y],
        p1: [7.0, y],
        depth0: depth,
        depth1: depth,
        width_px: 1.4,
        seg_id: id,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "physical-adapter skip must be visible in ordinary test logs"
)]
fn hair_raster_three_pass_physical_gpu_acceptance() {
    let Some(host) = hardware_host_or_skip() else {
        return;
    };
    let wgsl = compile_hair_raster_wgsl();
    let scene = [
        make_segment(3.5, 0.8, 2),  // farther, deliberately lower ID
        make_segment(3.5, 0.2, 40), // nearer depth must win regardless of ID
        make_segment(8.5, 0.4, 23),
        make_segment(8.5, 0.4, 7), // equal quantized depth: minimum ID wins
        GpuSegment {
            p0: [-0.25, 1.5],
            p1: [2.0, 1.5],
            depth0: 0.6,
            depth1: 0.6,
            width_px: 1.4,
            seg_id: 11,
        }, // crosses the top-left viewport boundary
        GpuSegment {
            p0: [12.5, 10.5],
            p1: [12.5, 10.5],
            depth0: 0.3,
            depth1: 0.3,
            width_px: 1.4,
            seg_id: 31,
        }, // degenerate segment behaves as an analytic point capsule
    ];
    let (depth_a, id_a, coverage_a) = replay(&host, &wgsl, &scene);
    let expected_a = cpu_oracle(&scene);
    assert_eq!(depth_a, expected_a.0, "full-frame depth matches CPU oracle");
    assert_eq!(id_a, expected_a.1, "full-frame IDs match CPU oracle");
    assert_eq!(
        coverage_a, expected_a.2,
        "full-frame coverage matches CPU oracle"
    );
    let mut reversed = scene;
    reversed.reverse();
    let (depth_b, id_b, coverage_b) = replay(&host, &wgsl, &reversed);
    let expected_b = cpu_oracle(&reversed);
    assert_eq!(
        depth_b, expected_b.0,
        "reversed full-frame depth matches CPU oracle"
    );
    assert_eq!(
        id_b, expected_b.1,
        "reversed full-frame IDs match CPU oracle"
    );
    assert_eq!(
        coverage_b, expected_b.2,
        "reversed full-frame coverage matches CPU oracle"
    );

    let occlusion = pixel(4, 3);
    let tie = pixel(4, 8);
    let expected_depth = |z: f32| (z.clamp(0.0, 1.0) * 16_777_215.0) as u32;
    assert_eq!(
        depth_a[occlusion],
        expected_depth(0.2),
        "nearest depth wins"
    );
    assert_eq!(id_a[occlusion], 40, "depth winner keeps its segment ID");
    assert_eq!(
        depth_a[tie],
        expected_depth(0.4),
        "equal-depth quantized value"
    );
    assert_eq!(
        id_a[tie], 7,
        "equal-depth tie resolves to minimum segment ID"
    );
    assert_eq!(id_b[tie], 7, "tie winner is invariant under input order");
    assert_eq!(depth_b[tie], depth_a[tie]);
    assert_eq!(coverage_b[tie], coverage_a[tie]);
    assert!(coverage_a[occlusion] > 0, "near winner publishes coverage");
    assert!(coverage_a[tie] > 0, "tie winner publishes coverage");
    assert!(coverage_a[occlusion] <= COVERAGE_QUANT as u32);
    assert!(coverage_a[tie] <= COVERAGE_QUANT as u32);

    // The clear contract is observable outside both segment AABBs.
    let untouched = pixel(15, 0);
    assert_eq!(depth_a[untouched], DEPTH_EMPTY);
    assert_eq!(id_a[untouched], ID_EMPTY);
    assert_eq!(coverage_a[untouched], COVERAGE_EMPTY);
    assert_eq!(depth_b[untouched], DEPTH_EMPTY);
    assert_eq!(id_b[untouched], ID_EMPTY);
    assert_eq!(coverage_b[untouched], COVERAGE_EMPTY);

    let empty = replay(&host, &wgsl, &[]);
    assert!(empty.0.iter().all(|&value| value == DEPTH_EMPTY));
    assert!(empty.1.iter().all(|&value| value == ID_EMPTY));
    assert!(empty.2.iter().all(|&value| value == COVERAGE_EMPTY));
}
