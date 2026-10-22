//! Strict physical-GPU parity for the planetary U256 WGSL primitives.
#![cfg(feature = "terrain_scatter")]

#[path = "../src/planet_placement_fixed_math.rs"]
mod fixed_math;

use std::time::Duration;

use fixed_math::U256;
use prism_world_gpu::GpuContext;
use wgpu::util::DeviceExt;

fn context() -> GpuContext {
    let context = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 2,
        max_bindings_per_bind_group: 2,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("physical U256 arithmetic device acquisition failed")
    .expect("physical GPU required; no silent skip");
    assert!(context.evidence().is_hardware());
    eprintln!("[planet_placement_u256_gpu] {:?}", context.evidence());
    context
}

fn read(context: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(30))
        .unwrap()
        .unwrap();
    let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
    buffer.unmap();
    bytes
}

fn append_shader() -> String {
    let mut source = include_str!("../src/shaders/planet_placement_u256.wgsl").to_owned();
    source.push_str(
        r#"
@group(0) @binding(0) var<storage, read> input_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words: array<u32>;

fn load_u256(base: u32) -> U256 {
    var result = u256_zero();
    for (var i = 0u; i < 8u; i += 1u) { result.limb[i] = input_words[base + i]; }
    return result;
}

fn store_u256(base: u32, value: U256) {
    for (var i = 0u; i < 8u; i += 1u) { output_words[base + i] = value.limb[i]; }
}

@compute @workgroup_size(1)
fn main() {
    let a = load_u256(0u);
    let b = load_u256(8u);
    let sum = u256_add(a, b);
    store_u256(0u, sum.value); output_words[8] = select(0u, 1u, sum.overflow);
    let difference = u256_sub(a, b);
    store_u256(9u, difference.value); output_words[17] = select(0u, 1u, difference.underflow);
    let product = u256_mul(a, b);
    store_u256(18u, product.value); output_words[26] = select(0u, 1u, product.overflow);
    let division = u256_div_rem(a, b);
    store_u256(27u, division.quotient); store_u256(35u, division.remainder);
    output_words[43] = select(0u, 1u, division.division_by_zero);
    store_u256(44u, u256_isqrt(a));
    let shifted = u256_shl(a, 84u);
    store_u256(52u, shifted.value); output_words[60] = select(0u, 1u, shifted.overflow);
}
"#,
    );
    source
}

fn read_word(bytes: &[u8], word_offset: usize) -> u32 {
    let offset = word_offset * 4;
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u256(bytes: &[u8], word_offset: usize) -> U256 {
    U256(std::array::from_fn(|index| {
        read_word(bytes, word_offset + index)
    }))
}

#[test]
fn u256_limb_arithmetic_matches_rust_reference_on_metal() {
    let context = context();
    let a = U256([
        0xfedc_ba98,
        0x7654_3210,
        0xffff_ffff,
        0x0000_ffff,
        0x1234_5678,
        0x9abc_def0,
        0x0000_0001,
        0,
    ]);
    // Keep the product below 256 bits while exercising a multi-limb divisor
    // and a quotient whose upper limbs are nonzero.
    let b = U256([0x1357_9bdf, 1, 0, 0, 0, 0, 0, 0]);
    let mut input = vec![0u32; 16];
    input[..8].copy_from_slice(&a.0);
    input[8..].copy_from_slice(&b.0);
    let input_buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("planet U256 arithmetic input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output_buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet U256 arithmetic output"),
        size: 61 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let shader = context
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet U256 arithmetic parity shader"),
            source: wgpu::ShaderSource::Wgsl(append_shader().into()),
        });
    let layout = context
        .device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet U256 arithmetic layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("planet U256 arithmetic pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("planet U256 arithmetic parity pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet U256 arithmetic bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
    let mut encoder = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet U256 arithmetic parity encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("planet U256 arithmetic parity pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let readback = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet U256 arithmetic readback"),
        size: 61 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, 61 * 4);
    context.queue().submit(Some(encoder.finish()));
    context.wait();
    let output = read(&context, &readback);

    let sum = a.checked_add(b).unwrap();
    let difference = a.checked_sub(b).unwrap();
    let product = a.checked_mul(b).unwrap();
    let (quotient, remainder) = a.div_rem(b).unwrap();
    let root = a.isqrt();
    let shifted = a.checked_shl(84);
    assert_eq!(read_u256(&output, 0), sum);
    assert_eq!(read_word(&output, 8), 0);
    assert_eq!(read_u256(&output, 9), difference);
    assert_eq!(read_word(&output, 17), 0);
    assert_eq!(read_u256(&output, 18), product);
    assert_eq!(read_word(&output, 26), 0);
    assert_eq!(read_u256(&output, 27), quotient);
    assert_eq!(read_u256(&output, 35), remainder);
    assert_eq!(read_word(&output, 43), 0);
    assert_eq!(read_u256(&output, 44), root);
    match shifted {
        Ok(expected) => {
            assert_eq!(read_u256(&output, 52), expected);
            assert_eq!(read_word(&output, 60), 0);
        }
        Err(_) => assert_eq!(read_word(&output, 60), 1),
    }
}
