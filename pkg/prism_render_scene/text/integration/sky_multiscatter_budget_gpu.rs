//! Real production sky multiscatter kernel readback against its CPU oracle.
//!
//! Run with `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene
//! --test sky_multiscatter_budget_gpu -- --nocapture` to require a hardware GPU.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_math::Vec3;
use prism_render_shading::{
    gi::{atmosphere::medium::Atmosphere, sky_lut::multiscatter_lut::bake_multiscatter_lut},
    LobeSampleBudget, ShadingQuality,
};

const PARAM_BYTES: u32 = 16;
const ROW_BYTES: u32 = 256;
const LUT_WIDTH: u32 = 4;
const LUT_HEIGHT: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    dir_samples: u32,
    march_samples: u32,
}

fn load(_: &(), source: ShaderCacheSource, _: &ValidateShader) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!(),
    }
}

fn production_wgsl() -> String {
    let mut cache = ShaderCache::new((), load);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_534b_595f_4d53_4c55_5401),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../src/shaders/sky_multiscatter_lut.wesl"),
            "embedded://prism_render_scene/shaders/sky_multiscatter_lut.wesl",
        ),
    );
    (*cache
        .get(0, id, &[])
        .expect("link production sky multiscatter WESL"))
    .clone()
}

fn hardware_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: Default::default(),
    });
    let adapter = match block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(error) => return skip_or_fail(error),
    };
    let info = adapter.get_info();
    eprintln!("sky multiscatter GPU adapter: {info:?}");
    if info.device_type == wgpu::DeviceType::Cpu {
        return skip_or_fail("software adapter is not hardware parity");
    }
    let limits = adapter.limits();
    if !adapter.features().contains(wgpu::Features::IMMEDIATES)
        || limits.max_immediate_size < PARAM_BYTES
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 8
        || limits.max_compute_workgroup_size_y < 8
        || limits.max_texture_dimension_2d < 1
    {
        return skip_or_fail(format!(
            "adapter lacks production kernel limits: {limits:?}"
        ));
    }
    match block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: PARAM_BYTES,
            ..wgpu::Limits::default()
        },
        ..Default::default()
    })) {
        Ok(pair) => Some(pair),
        Err(error) => skip_or_fail(error),
    }
}

fn skip_or_fail(message: impl std::fmt::Display) -> Option<(wgpu::Device, wgpu::Queue)> {
    assert!(
        std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU set: required sky multiscatter GPU unavailable: {message}"
    );
    eprintln!("sky multiscatter GPU SKIP: {message}");
    None
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut mantissa = fraction;
            let mut shift = 0;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            sign | ((127 - 15 - shift) << 23) | ((mantissa & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

fn run_kernel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    params: Params,
) -> Vec<[f32; 4]> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("production sky multiscatter readback output"),
        size: wgpu::Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("production sky multiscatter output"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sky multiscatter readback"),
        size: u64::from(ROW_BYTES * params.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(params.height),
            },
        },
        wgpu::Extent3d {
            width: params.width,
            height: params.height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("map sky LUT readback")
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("complete sky LUT dispatch");
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("sky LUT mapped bytes");
    let mut rgba = Vec::with_capacity((params.width * params.height) as usize);
    for y in 0..params.height {
        let row_start = (y * ROW_BYTES) as usize;
        let row_bytes = (params.width * 8) as usize;
        let row: &[u16] = bytemuck::cast_slice(&mapped[row_start..row_start + row_bytes]);
        for texel in row.chunks_exact(4) {
            rgba.push([
                half_to_f32(texel[0]),
                half_to_f32(texel[1]),
                half_to_f32(texel[2]),
                half_to_f32(texel[3]),
            ]);
        }
    }
    drop(mapped);
    readback.unmap();
    rgba
}

#[test]
fn production_sky_multiscatter_budget_counts_match_cpu_reference() {
    let Some((device, queue)) = hardware_device() else {
        return;
    };
    let source = production_wgsl();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production sky multiscatter kernel"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("production sky multiscatter kernel"),
        layout: None,
        module: &module,
        entry_point: Some("sky_multiscatter_lut"),
        compilation_options: Default::default(),
        cache: None,
    });

    let high = ShadingQuality::High.budget();
    let low = high.with_global_scale(0.25);
    let atmosphere = Atmosphere::earth();
    let irradiance = Vec3::splat(20.0);
    for budget in [high, low] {
        let params = Params {
            width: LUT_WIDTH,
            height: LUT_HEIGHT,
            dir_samples: budget
                .multiscatter_dir_samples
                .max(LobeSampleBudget::MIN_MULTISCATTER_DIR),
            march_samples: budget
                .multiscatter_march_samples
                .max(LobeSampleBudget::MIN_MULTISCATTER_MARCH),
        };
        let actual = run_kernel(&device, &queue, &pipeline, params);
        let expected = bake_multiscatter_lut(
            &atmosphere,
            irradiance,
            params.width as usize,
            params.height as usize,
            params.dir_samples,
            params.march_samples,
        );
        assert_eq!(actual.len(), expected.texels().len());
        for (index, (actual, expected)) in actual.iter().zip(expected.texels()).enumerate() {
            assert_eq!(actual[3], 1.0, "kernel did not write LUT texel {index}");
            for (channel, expected) in actual[..3].iter().zip([expected.x, expected.y, expected.z])
            {
                let tolerance = 0.003_f32.max(expected.abs() * 0.025);
                assert!(
                    (channel - expected).abs() <= tolerance,
                    "budget {budget:?}, texel {index}: GPU {channel} differs from CPU {expected} beyond {tolerance}"
                );
            }
        }
    }
}
