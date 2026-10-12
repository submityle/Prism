//! Real-device verification of the `GPU`-driven indirect-dispatch primitive.
//!
//! Two claims are checked on a live adapter (the suite skips when none is
//! available):
//!
//! 1. **Parity** — the device-built [`DispatchArgs`] block equals the
//!    [`cpu_dispatch_args`] golden twin bit-for-bit across the one-dimensional,
//!    boundary, and `y`-tiled regimes. The arithmetic is integer, so parity is
//!    exact rather than toleranced.
//! 2. **End to end** — a count that only exists in device memory drives a real
//!    `dispatch_workgroups_indirect`: the args kernel sizes the launch, and a
//!    consumer kernel bumps exactly `count` elements and no more, with the host
//!    never reading the count back. This is the capability the whole
//!    GPU-driven pipeline rests on, exercised through the actual indirect path.
//!
//! Provenance: standard `wgpu` indirect dispatch. No Unreal Engine source or
//! derived code.

use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::indirect::args::cpu_dispatch_args;
use prism_physics_gpu::indirect::gpu::GpuDispatchArgs;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

/// Lanes per workgroup used by both the args sizing and the consumer kernel.
const WORKGROUP: u32 = 256;

#[test]
fn device_dispatch_args_match_the_cpu_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping indirect-dispatch parity");
        return;
    };
    let pipeline = GpuDispatchArgs::new(&ctx);
    let max_dim = GpuDispatchArgs::device_max_dim(&ctx);

    // A small synthetic max_dim forces the tiling branch without allocating a
    // 65k-wide grid; the device honours whatever cap we pass in `params`.
    let tiling_cap = 8u32;

    let cases: &[(Vec<u32>, u32, u32, u32)] = &[
        // (counts, offset, workgroup_size, max_dim)
        (vec![0], 0, WORKGROUP, max_dim),
        (vec![1], 0, WORKGROUP, max_dim),
        (vec![WORKGROUP], 0, WORKGROUP, max_dim),
        (vec![WORKGROUP + 1], 0, WORKGROUP, max_dim),
        (vec![1000], 0, WORKGROUP, max_dim),
        (vec![100_000], 0, WORKGROUP, max_dim),
        // offset selects a later count word.
        (vec![7, 999, 4096], 2, WORKGROUP, max_dim),
        // tiling: 9 groups' worth of work past an 8-wide cap -> 8x2.
        (vec![9 * 64], 0, 64, tiling_cap),
        // tiling at a non-multiple to exercise the ceil in both dimensions.
        (vec![1000], 0, 64, tiling_cap),
        // `count + wg - 1` wraps at the u32 top: the kernel must use the
        // overflow-safe ceil-div to match `u32::div_ceil` (CR-found).
        (vec![u32::MAX], 0, 2, max_dim),
    ];

    for (counts, offset, wg, cap) in cases {
        let got = pipeline.compute(&ctx, counts, *offset, *wg, *cap);
        let want = cpu_dispatch_args(counts[*offset as usize], *wg, *cap);
        assert_eq!(
            got, want,
            "indirect args mismatch for counts={counts:?} offset={offset} wg={wg} cap={cap}"
        );
    }
}

/// An empty `counts` slice must not create a zero-sized storage binding: the
/// convenience `compute` returns the empty grid instead (CR-found).
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_counts_returns_empty_args() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping indirect-dispatch empty case");
        return;
    };
    let pipeline = GpuDispatchArgs::new(&ctx);
    let max_dim = GpuDispatchArgs::device_max_dim(&ctx);
    assert_eq!(
        pipeline.compute(&ctx, &[], 0, WORKGROUP, max_dim),
        prism_physics_gpu::indirect::args::DispatchArgs::EMPTY
    );
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_count_drives_a_real_indirect_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter; skipping indirect-dispatch end-to-end");
        return;
    };
    let device = ctx.device();
    let args_pipeline = GpuDispatchArgs::new(&ctx);
    let max_dim = GpuDispatchArgs::device_max_dim(&ctx);

    // The work-item count lives only in a device buffer; the host picks the
    // storage capacity but never feeds the count into the launch size.
    let count: u32 = 1000;
    let capacity: u32 = 2048;

    let counts_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("indirect_e2e_counts"),
        contents: bytemuck::bytes_of(&count),
        usage: BufferUsages::STORAGE,
    });
    let data_buf = device.create_buffer(&BufferDescriptor {
        label: Some("indirect_e2e_data"),
        size: u64::from(capacity) * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let consumer_params = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("indirect_e2e_consumer_params"),
        // (count, groups_x_unused, pad, pad): the guard count for the consumer.
        contents: bytemuck::cast_slice(&[count, 0u32, 0u32, 0u32]),
        usage: BufferUsages::UNIFORM,
    });
    let args_buf = GpuDispatchArgs::args_buffer(device, "indirect_e2e_args");

    // Consumer kernel: bump each element below the device-side count exactly
    // once. Launched purely from the GPU-built args block.
    let consumer_src = r#"
struct P { count: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < p.count) {
        data[i] = data[i] + 1u;
    }
}
"#;
    let consumer_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("indirect_e2e_consumer"),
        source: ShaderSource::Wgsl(consumer_src.into()),
    });
    let consumer_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("indirect_e2e_consumer_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
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
        ],
    });
    let consumer_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("indirect_e2e_consumer_pipeline_layout"),
        bind_group_layouts: &[Some(&consumer_layout)],
        immediate_size: 0,
    });
    let consumer_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("indirect_e2e_consumer_pipeline"),
        layout: Some(&consumer_pipeline_layout),
        module: &consumer_module,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let consumer_bind = device.create_bind_group(&BindGroupDescriptor {
        label: Some("indirect_e2e_consumer_bind"),
        layout: &consumer_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: consumer_params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: data_buf.as_entire_binding(),
            },
        ],
    });

    let stage = device.create_buffer(&BufferDescriptor {
        label: Some("indirect_e2e_stage"),
        size: u64::from(capacity) * 4,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // One encoder, one submission: size the launch from the device count, then
    // consume it indirectly. No host round-trip in between.
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("indirect_e2e_encoder"),
    });
    args_pipeline.record(
        device,
        &mut encoder,
        &counts_buf,
        0,
        WORKGROUP,
        max_dim,
        &args_buf,
    );
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("indirect_e2e_consumer_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&consumer_pipeline);
        pass.set_bind_group(0, &consumer_bind, &[]);
        pass.dispatch_workgroups_indirect(&args_buf, 0);
    }
    encoder.copy_buffer_to_buffer(&data_buf, 0, &stage, 0, u64::from(capacity) * 4);
    ctx.queue().submit([encoder.finish()]);

    stage.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    ctx.wait();
    let view = stage.slice(..).get_mapped_range().expect("mapped readback");
    let values = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
    drop(view);
    stage.unmap();

    for (i, &v) in values.iter().enumerate() {
        let expected = u32::from((i as u32) < count);
        assert_eq!(
            v, expected,
            "element {i} was {v}, expected {expected} (count={count})"
        );
    }
}
