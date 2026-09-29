//! Real-device `GPU` parity coverage for the volumetric-cloud noise-bake kernel.
//!
//! The sibling [`shader_tests`](super::shader_tests) module proves every cloud
//! `WESL` source parses and type-checks through the render world's
//! [`ShaderCache`], but a shader can compile cleanly and still compute the wrong
//! number. The test here closes that gap for `volumetric_noise_bake` by binding
//! the real compute pipeline on an actual `Metal` (or any native `wgpu`) device,
//! baking the Perlin-Worley density volume over a non-trivial grid, reading the
//! result back, and asserting it voxel-for-voxel against the `CPU` golden twin
//! [`prism_render_architecture::volumetric::noise`].
//!
//! The `WESL` kernel and the `CPU` reference share byte-identical integer
//! hashing (`FNV-1a` word mixing plus the `xorshift`-multiply avalanche), the
//! same `GRAD3` gradient table, the same quintic fade, and the same
//! amplitude-normalized `fBm` / `perlin_worley` composition, so a green run is
//! direct on-device evidence that the ported noise matches its reference — not
//! merely that it compiles. The on-device store target is `rgba16float`, so the
//! comparison uses an `fp16`-quantization tolerance rather than demanding
//! bit-exact `float32` agreement.
//!
//! Adapter acquisition is best-effort: on a headless host with no `wgpu`
//! adapter, or one without compute-immediate support, [`try_cloud_device`]
//! returns `None` and the test skips with a printed notice instead of failing,
//! so the suite stays green everywhere while still exercising the full dispatch
//! on any machine with a real device (for example an `Apple` `M`-series `GPU`).

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Extent3d, Features, Instance, InstanceDescriptor, InstanceFlags, MapMode, Origin3d,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};

use prism_render_architecture::volumetric::math::{Vec2, Vec3};
use prism_render_architecture::volumetric::multiscatter::MultiScatterLut;
use prism_render_architecture::volumetric::scatter::OctaveParams;
use prism_render_architecture::volumetric::weather::{advect_semi_lagrangian, WeatherField};
use prism_render_architecture::volumetric::{
    modeling, noise, CloudKind, WeatherMapHandle, WeatherSample,
};

use super::abi::{GpuModelingParams, GpuMsLutParams, GpuNoiseBakeParams, GpuWeatherAdvectParams};

/// Absolute per-voxel tolerance for the `GPU`-versus-`CPU` comparison.
///
/// The two paths run the same deterministic `float32` hashing and arithmetic;
/// the only divergence is the final store to an `rgba16float` texel, whose
/// worst-case round-to-nearest error over `[0, 1]` is well under `5.0e-4`. The
/// margin absorbs that quantization plus a driver's fused-multiply-add
/// contraction freedom in the long `fBm` sums.
const PARITY_EPS: f32 = 2.0e-3;

/// Detail-noise seed mix folded into the kernel, mirrored from the shader's
/// `vc_noise_params.seed ^ 0x5bd1e995u` so the `CPU` golden hashes the same
/// cells for the erosion-detail channel.
const DETAIL_SEED_MIX: u32 = 0x5bd1_e995;

/// Streams the `Wgsl` source back out of the shader cache without a device.
///
/// Mirrors the closure [`shader_tests`](super::shader_tests) uses so the `WESL`
/// is composed through the exact render-world pipeline; here we keep the
/// compiled `Wgsl` string (rather than a device module) so the parity test can
/// hand it to a raw `wgpu` device it created itself.
fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("volumetric cloud shaders are WESL"),
    }
}

/// Compiles `volumetric_clouds.wesl` and returns its `Wgsl` translation.
fn compile_clouds_wgsl() -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let id = AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_564f_4c55_4d45_5450_0003),
    };
    cache.set_shader(
        id,
        Shader::from_wesl(
            include_str!("../../shaders/volumetric_clouds.wesl"),
            "embedded://prism_render_scene/shaders/volumetric_clouds.wesl",
        ),
    );
    let module = cache
        .get(0, id, &[])
        .unwrap_or_else(|error| panic!("volumetric_clouds.wesl failed to compile: {error}"));
    (*module).clone()
}

/// Finds the compute entry point whose name contains `needle` in `wgsl`.
///
/// The `WESL` compiler may prefix module-local names, so the parity test locates
/// the `volumetric_noise_bake` entry by substring rather than assuming a fixed
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

/// Best-effort acquisition of a native compute device and queue with compute
/// immediate-data support.
///
/// Returns `None` (rather than panicking) when no adapter is available or the
/// adapter cannot host the kernel's `var<immediate>` block, so the suite
/// stays green on headless / limited hosts; on a machine with a real `GPU` this
/// yields a live device the parity test dispatches against.
fn try_cloud_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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
    if limits.max_immediate_size < size_of::<GpuNoiseBakeParams>() as u32 {
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

/// Decodes one `IEEE-754` binary16 (`fp16`) bit pattern to `f32`.
///
/// The on-device store target is `rgba16float`; the readback buffer therefore
/// carries `u16` half-floats that must be widened before comparison. Handles
/// subnormals, zero, and the normal range (the kernel never produces
/// `Inf`/`NaN` because every channel is a saturated `[0, 1]` value).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from(bits >> 10) & 0x1f;
    let mant = u32::from(bits & 0x03ff);
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalize into the f32 exponent range.
            let mut e = -1i32;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x0400 != 0 {
                    break;
                }
            }
            let f32_exp = (127 - 15 - e) as u32;
            sign | (f32_exp << 23) | ((m & 0x03ff) << 13)
        }
    } else if exp == 0x1f {
        // Inf / NaN: preserve mantissa payload (unused by this kernel).
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        let f32_exp = exp + (127 - 15);
        sign | (f32_exp << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

/// One on-device noise bake must match the `CPU` golden within `fp16` rounding.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without an immediate-data adapter"
)]
fn noise_bake_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_cloud_device() else {
        eprintln!(
            "noise_bake_gpu_matches_cpu_golden: no immediate-data wgpu adapter, skipping on-device parity"
        );
        return;
    };

    // Width is a multiple of 32 so the `rgba16float` row (8 bytes/texel) is a
    // multiple of the 256-byte copy alignment, keeping the readback dense; every
    // extent is a multiple of the (4, 4, 4) workgroup so no invocation is masked.
    const DIM_X: u32 = 32;
    const DIM_Y: u32 = 8;
    const DIM_Z: u32 = 8;
    const BYTES_PER_TEXEL: u32 = 8;

    let params = GpuNoiseBakeParams {
        dim_x: DIM_X,
        dim_y: DIM_Y,
        dim_z: DIM_Z,
        base_freq: 8.0,
        detail_freq: 16.0,
        seed: 1337,
    };

    let wgsl = compile_clouds_wgsl();
    let entry = find_entry_point(&wgsl, "noise_bake");

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("volumetric_clouds_parity"),
        source: ShaderSource::Wgsl(wgsl.as_str().into()),
    });
    // Explicit `@group(0)` layout (binding 2 = the write-only rgba16float density
    // volume) plus a pipeline layout whose immediate range spans the full
    // `GpuNoiseBakeParams` block. Auto layout (`layout: None`) reflects an
    // immediate range one word short of the 24-byte struct on this driver, so
    // the range is declared explicitly to mirror the production pipeline.
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vc_noise_bake_layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba16Float,
                view_dimension: TextureViewDimension::D3,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("vc_noise_bake_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: size_of::<GpuNoiseBakeParams>() as u32,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("volumetric_noise_bake_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let density = device.create_texture(&TextureDescriptor {
        label: Some("vc_noise_density_out"),
        size: Extent3d {
            width: DIM_X,
            height: DIM_Y,
            depth_or_array_layers: DIM_Z,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let density_view = density.create_view(&TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("vc_noise_bake_bind_group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 2,
            resource: BindingResource::TextureView(&density_view),
        }],
    });

    let row_bytes = DIM_X * BYTES_PER_TEXEL;
    let readback_size = u64::from(row_bytes * DIM_Y * DIM_Z);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("vc_noise_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("vc_noise_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("vc_noise_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(DIM_X / 4, DIM_Y / 4, DIM_Z / 4);
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &density,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(DIM_Y),
            },
        },
        Extent3d {
            width: DIM_X,
            height: DIM_Y,
            depth_or_array_layers: DIM_Z,
        },
    );
    queue.submit([encoder.finish()]);

    readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = readback
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let halves: Vec<u16> = bytemuck::cast_slice::<u8, u16>(&view).to_vec();
    drop(view);
    readback.unmap();

    // Row is dense (row_bytes == DIM_X * 8), so texel n = z*W*H + y*W + x maps to
    // half-word 4*n (r) and 4*n + 1 (g).
    let mut checked = 0u32;
    let mut z = 0u32;
    while z < DIM_Z {
        let mut y = 0u32;
        while y < DIM_Y {
            let mut x = 0u32;
            while x < DIM_X {
                let n = (z * DIM_Y * DIM_X + y * DIM_X + x) as usize;
                let gpu_base = f16_to_f32(halves[4 * n]);
                let gpu_detail = f16_to_f32(halves[4 * n + 1]);

                let uvw = Vec3::new(
                    (x as f32 + 0.5) / DIM_X as f32,
                    (y as f32 + 0.5) / DIM_Y as f32,
                    (z as f32 + 0.5) / DIM_Z as f32,
                );
                let cpu_base = noise::perlin_worley(uvw.scale(params.base_freq), params.seed);
                let cpu_detail = noise::worley_fbm(
                    uvw.scale(params.detail_freq),
                    params.seed ^ DETAIL_SEED_MIX,
                    3,
                );

                let db = (gpu_base - cpu_base).abs();
                let dd = (gpu_detail - cpu_detail).abs();
                assert!(
                    db < PARITY_EPS && dd < PARITY_EPS,
                    "voxel ({x}, {y}, {z}): gpu=(base {gpu_base}, detail {gpu_detail}) \
                     cpu=(base {cpu_base}, detail {cpu_detail}) |d|=({db}, {dd})",
                );
                checked += 1;
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    assert_eq!(
        checked,
        DIM_X * DIM_Y * DIM_Z,
        "every voxel must be compared"
    );
}

/// One on-device density-modeling pass must match the `CPU` golden twin.
///
/// This closes the parity gap for `volumetric_modeling`, the kernel that turns
/// the baked noise volume plus the weather map into the final cloud density.
/// It uploads a deterministic `rgba32float` noise volume (`.x` base shape,
/// `.y` erosion detail) and a per-column `rgba32float` weather coverage map,
/// dispatches the real compute pipeline over a `(4, 4, 4)`-aligned grid, reads
/// the `rgba16float` density back, and compares every voxel against the
/// re-derived [`prism_render_architecture::volumetric::modeling`] pipeline
/// (`combine_coverage` -> `cloud_type_shape` -> `height_gradient` ->
/// `compose_density`). Inputs are exact `float32` (no input quantization), so
/// the only tolerance is the output `fp16` store, absorbed by [`PARITY_EPS`].
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without an immediate-data adapter"
)]
fn modeling_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_cloud_device() else {
        eprintln!(
            "modeling_gpu_matches_cpu_golden: no immediate-data wgpu adapter, skipping on-device parity"
        );
        return;
    };

    // Width is a multiple of 32 so the `rgba16float` output row (8 bytes/texel)
    // meets the 256-byte copy alignment; every extent is a multiple of the
    // (4, 4, 4) workgroup so no invocation is masked.
    const DIM_X: u32 = 32;
    const DIM_Y: u32 = 8;
    const DIM_Z: u32 = 8;
    const OUT_BYTES_PER_TEXEL: u32 = 8;

    // `kind = 1` selects the cumulus height profile in both the `WESL`
    // `vc_height_profile` switch and the `CPU` `HeightProfile::for_kind`; the
    // pairing is by profile value, not enum declaration order.
    let params = GpuModelingParams {
        dim_x: DIM_X,
        dim_y: DIM_Y,
        dim_z: DIM_Z,
        coverage: 0.6,
        cloud_type: 0.5,
        erosion_strength: 0.4,
        kind: 1,
    };

    // Deterministic integer recipes the `CPU` golden re-derives verbatim below.
    // Every span is non-degenerate so no collapsed `smoothstep` / `remap`
    // branch is taken; the standard formulas apply on both sides.
    fn base_at(x: u32, y: u32, z: u32) -> f32 {
        ((x * 13 + y * 7 + z * 5) % 17) as f32 / 16.0
    }
    fn detail_at(x: u32, y: u32, z: u32) -> f32 {
        ((x * 3 + y * 11 + z * 2) % 13) as f32 / 12.0
    }
    fn coverage_at(x: u32, y: u32) -> f32 {
        ((x * 5 + y * 3) % 11) as f32 / 10.0
    }

    // Populate the exact-`float32` noise volume (`.x` base, `.y` detail) and the
    // per-column weather coverage map (`.x` coverage).
    let voxel_count = (DIM_X * DIM_Y * DIM_Z) as usize;
    let mut noise_data = vec![0.0f32; voxel_count * 4];
    let column_count = (DIM_X * DIM_Y) as usize;
    let mut weather_data = vec![0.0f32; column_count * 4];
    let mut z = 0u32;
    while z < DIM_Z {
        let mut y = 0u32;
        while y < DIM_Y {
            let mut x = 0u32;
            while x < DIM_X {
                let n = (z * DIM_Y * DIM_X + y * DIM_X + x) as usize;
                noise_data[4 * n] = base_at(x, y, z);
                noise_data[4 * n + 1] = detail_at(x, y, z);
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    let mut y = 0u32;
    while y < DIM_Y {
        let mut x = 0u32;
        while x < DIM_X {
            let m = (y * DIM_X + x) as usize;
            weather_data[4 * m] = coverage_at(x, y);
            x += 1;
        }
        y += 1;
    }

    let wgsl = compile_clouds_wgsl();
    let entry = find_entry_point(&wgsl, "modeling");
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("volumetric_modeling_parity"),
        source: ShaderSource::Wgsl(wgsl.as_str().into()),
    });

    // Explicit `@group(0)` layout mirroring the modeling kernel's bindings:
    // 3 = noise input (`texture_3d<f32>`), 4 = weather input (`texture_2d<f32>`),
    // 5 = write-only `rgba16float` density volume. The pipeline layout's
    // immediate range spans the full `GpuModelingParams` block, matching the
    // production pipeline (auto layout misreflects the immediate size on this
    // driver).
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vc_modeling_bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D3,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("vc_modeling_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: size_of::<GpuModelingParams>() as u32,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("volumetric_modeling_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let noise_tex = device.create_texture(&TextureDescriptor {
        label: Some("vc_modeling_noise_in"),
        size: Extent3d {
            width: DIM_X,
            height: DIM_Y,
            depth_or_array_layers: DIM_Z,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let weather_tex = device.create_texture(&TextureDescriptor {
        label: Some("vc_modeling_weather_in"),
        size: Extent3d {
            width: DIM_X,
            height: DIM_Y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let density = device.create_texture(&TextureDescriptor {
        label: Some("vc_modeling_density_out"),
        size: Extent3d {
            width: DIM_X,
            height: DIM_Y,
            depth_or_array_layers: DIM_Z,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // `write_texture` has no 256-byte row-alignment requirement, so the dense
    // `rgba32float` uploads go up as-is.
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &noise_tex,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(&noise_data),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(DIM_X * 16),
            rows_per_image: Some(DIM_Y),
        },
        Extent3d {
            width: DIM_X,
            height: DIM_Y,
            depth_or_array_layers: DIM_Z,
        },
    );
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &weather_tex,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(&weather_data),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(DIM_X * 16),
            rows_per_image: Some(DIM_Y),
        },
        Extent3d {
            width: DIM_X,
            height: DIM_Y,
            depth_or_array_layers: 1,
        },
    );

    let noise_view = noise_tex.create_view(&TextureViewDescriptor::default());
    let weather_view = weather_tex.create_view(&TextureViewDescriptor::default());
    let density_view = density.create_view(&TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("vc_modeling_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureView(&noise_view),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&weather_view),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::TextureView(&density_view),
            },
        ],
    });

    let row_bytes = DIM_X * OUT_BYTES_PER_TEXEL;
    let readback_size = u64::from(row_bytes * DIM_Y * DIM_Z);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("vc_modeling_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("vc_modeling_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("vc_modeling_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(DIM_X / 4, DIM_Y / 4, DIM_Z / 4);
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &density,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(DIM_Y),
            },
        },
        Extent3d {
            width: DIM_X,
            height: DIM_Y,
            depth_or_array_layers: DIM_Z,
        },
    );
    queue.submit([encoder.finish()]);

    readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = readback
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let halves: Vec<u16> = bytemuck::cast_slice::<u8, u16>(&view).to_vec();
    drop(view);
    readback.unmap();

    // Row is dense (row_bytes == DIM_X * 8), so texel n = z*W*H + y*W + x maps
    // to half-word 4*n (the density is stored in the `.x` channel).
    let mut checked = 0u32;
    let mut z = 0u32;
    while z < DIM_Z {
        let mut y = 0u32;
        while y < DIM_Y {
            let mut x = 0u32;
            while x < DIM_X {
                let n = (z * DIM_Y * DIM_X + y * DIM_X + x) as usize;
                let gpu_density = f16_to_f32(halves[4 * n]);

                let base = base_at(x, y, z);
                let detail = detail_at(x, y, z);
                let weather_cov = coverage_at(x, y);
                let cov = modeling::combine_coverage(params.coverage, weather_cov);
                // Kernel folds `cloud_type_shape(base, base, ..)`, which equals
                // `base` (lerp of equal endpoints); mirror it exactly anyway.
                let ct = modeling::cloud_type_shape(base, base, params.cloud_type);
                let height_fraction = (z as f32 + 0.5) / DIM_Z as f32;
                let height = modeling::height_gradient(height_fraction, CloudKind::Cumulus);
                let cpu_density =
                    modeling::compose_density(ct, cov, height, detail, params.erosion_strength);

                let d = (gpu_density - cpu_density).abs();
                assert!(
                    d < PARITY_EPS,
                    "voxel ({x}, {y}, {z}): gpu={gpu_density} cpu={cpu_density} |d|={d}",
                );
                checked += 1;
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    assert_eq!(
        checked,
        DIM_X * DIM_Y * DIM_Z,
        "every voxel must be compared"
    );
}

/// Absolute per-cell tolerance for the multi-scatter `LUT` `GPU`-vs-`CPU`
/// comparison.
///
/// Unlike the noise and modeling kernels, the energy-gain kernel is *not* a
/// byte-identical twin: its optical-depth ramp is `1 - exp(-depth)`, and by
/// design the `WESL` uses the native `exp` while the `CPU` reference uses the
/// zero-dependency polynomial [`prism_render_architecture::volumetric::math::exp_approx`].
/// Every other term (the `sqrt`-based diffusion reflectance, the integer-power
/// octave decay, and the pure-`sqrt` `HG` phase ratio) is bit-for-bit shared,
/// so the only divergence is that documented `exp` approximation gap (bounded
/// well under `1.0e-3` over `depth in [0, 8]`) plus the final `fp16` store.
/// This margin covers both with room to spare.
const MS_PARITY_EPS: f32 = 3.0e-3;

/// One on-device multi-scatter energy-gain `LUT` bake must match the `CPU`
/// golden twin within the `exp`-approximation gap.
///
/// The `volumetric_multiscatter_lut_bake` kernel is the isolated, input-free
/// numeric kernel that pre-integrates the Wrenninge octave energy gain into a
/// 3D table (cos-zenith outer, optical-depth middle, albedo inner). This binds
/// the real compute pipeline, bakes the table on-device, reads back the
/// `rgba16float` `.x` gain, and compares every cell against the shipping `CPU`
/// [`MultiScatterLut::build_energy_gain`] — the same reference the render world
/// samples — reconstructing each cell's axis coordinates with the table's own
/// `lerp(min, max, i / (dim - 1))` rule.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without an immediate-data adapter"
)]
fn multiscatter_lut_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_cloud_device() else {
        eprintln!(
            "multiscatter_lut_gpu_matches_cpu_golden: no immediate-data wgpu adapter, skipping on-device parity"
        );
        return;
    };

    // cos axis is a multiple of 32 so the `rgba16float` row (8 bytes/texel) meets
    // the 256-byte copy alignment; every axis is a multiple of the (4, 4, 4)
    // workgroup so no invocation is masked.
    const DIM_COS: u32 = 32;
    const DIM_DEPTH: u32 = 8;
    const DIM_ALBEDO: u32 = 8;
    const BYTES_PER_TEXEL: u32 = 8;
    // Mirrors `VC_MAX_OPTICAL_DEPTH` / `DEFAULT_MAX_OPTICAL_DEPTH`: the middle
    // axis spans `[0, 8]`.
    const MAX_OPTICAL_DEPTH: f32 = 8.0;

    let params = GpuMsLutParams {
        dim_cos: DIM_COS,
        dim_depth: DIM_DEPTH,
        dim_albedo: DIM_ALBEDO,
        attenuation: 0.6,
        contribution: 0.7,
        eccentricity: 0.8,
        octave_count: 4,
    };

    // The `CPU` golden table filled by the exact shipping code path.
    let octave = OctaveParams {
        attenuation: params.attenuation,
        contribution: params.contribution,
        eccentricity_attenuation: params.eccentricity,
        octave_count: params.octave_count,
    };
    let lut = MultiScatterLut::build_energy_gain(
        [DIM_COS as usize, DIM_DEPTH as usize, DIM_ALBEDO as usize],
        octave,
    );

    // The kernel's axis reconstruction: `lerp(min, max, i / (dim - 1))`, or the
    // minimum when the axis has a single cell.
    fn axis_value(min: f32, max: f32, dim: u32, i: u32) -> f32 {
        if dim <= 1 {
            min
        } else {
            min + (max - min) * (i as f32 / (dim - 1) as f32)
        }
    }

    let wgsl = compile_clouds_wgsl();
    let entry = find_entry_point(&wgsl, "multiscatter_lut_bake");
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("volumetric_multiscatter_lut_parity"),
        source: ShaderSource::Wgsl(wgsl.as_str().into()),
    });

    // Explicit `@group(0)` layout: binding 6 = the write-only `rgba16float`
    // energy-gain volume; the pipeline layout's immediate range spans the full
    // `GpuMsLutParams` block (auto layout misreflects the immediate size on this
    // driver).
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vc_mslut_bind_group_layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 6,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba16Float,
                view_dimension: TextureViewDimension::D3,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("vc_mslut_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: size_of::<GpuMsLutParams>() as u32,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("volumetric_multiscatter_lut_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let lut_tex = device.create_texture(&TextureDescriptor {
        label: Some("vc_mslut_out"),
        size: Extent3d {
            width: DIM_COS,
            height: DIM_DEPTH,
            depth_or_array_layers: DIM_ALBEDO,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let lut_view = lut_tex.create_view(&TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("vc_mslut_bind_group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 6,
            resource: BindingResource::TextureView(&lut_view),
        }],
    });

    let row_bytes = DIM_COS * BYTES_PER_TEXEL;
    let readback_size = u64::from(row_bytes * DIM_DEPTH * DIM_ALBEDO);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("vc_mslut_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("vc_mslut_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("vc_mslut_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(DIM_COS / 4, DIM_DEPTH / 4, DIM_ALBEDO / 4);
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &lut_tex,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(DIM_DEPTH),
            },
        },
        Extent3d {
            width: DIM_COS,
            height: DIM_DEPTH,
            depth_or_array_layers: DIM_ALBEDO,
        },
    );
    queue.submit([encoder.finish()]);

    readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = readback
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let halves: Vec<u16> = bytemuck::cast_slice::<u8, u16>(&view).to_vec();
    drop(view);
    readback.unmap();

    // Row is dense (row_bytes == DIM_COS * 8), so cell (ic, id, ia) at
    // (gid.x, gid.y, gid.z) maps to texel n = ia*DEPTH*COS + id*COS + ic and the
    // gain is stored in the `.x` channel (half-word 4*n).
    let mut checked = 0u32;
    let mut ia = 0u32;
    while ia < DIM_ALBEDO {
        let albedo = axis_value(0.0, 1.0, DIM_ALBEDO, ia);
        let mut id = 0u32;
        while id < DIM_DEPTH {
            let depth = axis_value(0.0, MAX_OPTICAL_DEPTH, DIM_DEPTH, id);
            let mut ic = 0u32;
            while ic < DIM_COS {
                let cos = axis_value(-1.0, 1.0, DIM_COS, ic);
                let n = (ia * DIM_DEPTH * DIM_COS + id * DIM_COS + ic) as usize;
                let gpu_gain = f16_to_f32(halves[4 * n]);
                // Sampling at the exact cell coordinate returns that cell's
                // stored value (the trilinear fraction is zero to within f32).
                let cpu_gain = lut.sample(cos, depth, albedo);

                let d = (gpu_gain - cpu_gain).abs();
                assert!(
                    d < MS_PARITY_EPS,
                    "cell (cos {ic}, depth {id}, albedo {ia}): gpu={gpu_gain} cpu={cpu_gain} |d|={d}",
                );
                checked += 1;
                ic += 1;
            }
            id += 1;
        }
        ia += 1;
    }
    assert_eq!(
        checked,
        DIM_COS * DIM_DEPTH * DIM_ALBEDO,
        "every LUT cell must be compared"
    );
}

/// One on-device semi-Lagrangian weather-advect step must match the `CPU`
/// golden twin to within `fp16` storage rounding.
///
/// The `volumetric_weather_advect` kernel back-traces every weather cell along
/// the wind and resamples the previous field with clamp-to-edge bilinear taps,
/// the on-device mirror of [`advect_semi_lagrangian`] +
/// [`WeatherField::sample_bilinear`]. Both paths run the identical clamp /
/// truncate / `min` / `mix` arithmetic in `float32` with no `exp`/`pow`
/// anywhere, and the input map is uploaded as exact `rgba32float`, so the sole
/// divergence is the final `rgba16float` store. The chosen wind makes the
/// left-edge departure cells clamp (exercising clamp-to-edge) while the
/// fractional offsets exercise the four-tap bilinear blend on every channel
/// (coverage / type / precipitation / wind-disturbance).
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without an immediate-data adapter"
)]
fn weather_advect_gpu_matches_cpu_golden() {
    let Some((device, queue)) = try_cloud_device() else {
        eprintln!(
            "weather_advect_gpu_matches_cpu_golden: no immediate-data wgpu adapter, skipping on-device parity"
        );
        return;
    };

    // Width is a multiple of 32 so the `rgba16float` output row (8 bytes/texel)
    // meets the 256-byte copy alignment; both extents are multiples of the
    // (8, 8, 1) workgroup so no invocation is masked.
    const WIDTH: u32 = 32;
    const HEIGHT: u32 = 8;
    const OUT_BYTES_PER_TEXEL: u32 = 8;

    let params = GpuWeatherAdvectParams {
        width: WIDTH,
        height: HEIGHT,
        // Non-integer wind: the negative x back-traces the left columns past the
        // edge (clamp-to-edge path), and the fractional magnitudes land every
        // departure point between cells (four-tap bilinear path).
        wind_x: 1.5,
        wind_y: -0.75,
        dt: 1.0,
    };

    // Deterministic per-cell weather channels the `CPU` golden re-derives
    // verbatim; every value already lies in `[0, 1]` so `from_rgba`'s saturate
    // is the identity and both sides see the same exact `float32` input.
    fn coverage_at(x: u32, y: u32) -> f32 {
        ((x * 5 + y * 3) % 11) as f32 / 10.0
    }
    fn cloud_type_at(x: u32, y: u32) -> f32 {
        ((x * 7 + y * 2) % 9) as f32 / 8.0
    }
    fn precip_at(x: u32, y: u32) -> f32 {
        ((x * 2 + y * 13) % 7) as f32 / 6.0
    }
    fn wind_at(x: u32, y: u32) -> f32 {
        ((x * 3 + y * 5) % 13) as f32 / 12.0
    }

    // Exact-`float32` source map, uploaded to the input texture and mirrored
    // into the `CPU` `WeatherField` so both paths resample identical cells.
    let cell_count = (WIDTH * HEIGHT) as usize;
    let mut src_data = vec![0.0f32; cell_count * 4];
    let mut y = 0u32;
    while y < HEIGHT {
        let mut x = 0u32;
        while x < WIDTH {
            let (r, g, b, a) = (
                coverage_at(x, y),
                cloud_type_at(x, y),
                precip_at(x, y),
                wind_at(x, y),
            );
            let m = (y * WIDTH + x) as usize;
            src_data[4 * m] = r;
            src_data[4 * m + 1] = g;
            src_data[4 * m + 2] = b;
            src_data[4 * m + 3] = a;
            x += 1;
        }
        y += 1;
    }
    // `from_cells` fills row-major (`y * WIDTH + x`), matching the upload order.
    let field = {
        let mut ordered = Vec::with_capacity(cell_count);
        let mut y = 0u32;
        while y < HEIGHT {
            let mut x = 0u32;
            while x < WIDTH {
                ordered.push(WeatherSample::from_rgba(
                    coverage_at(x, y),
                    cloud_type_at(x, y),
                    precip_at(x, y),
                    wind_at(x, y),
                ));
                x += 1;
            }
            y += 1;
        }
        WeatherField::from_cells(WeatherMapHandle(1), WIDTH, HEIGHT, ordered)
            .expect("cell count matches width * height")
    };
    let advected =
        advect_semi_lagrangian(&field, Vec2::new(params.wind_x, params.wind_y), params.dt);

    let wgsl = compile_clouds_wgsl();
    let entry = find_entry_point(&wgsl, "weather_advect");
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("volumetric_weather_advect_parity"),
        source: ShaderSource::Wgsl(wgsl.as_str().into()),
    });

    // Explicit `@group(0)` layout mirroring the advect kernel: binding 0 = the
    // sampled `texture_2d<f32>` source map, binding 1 = the write-only
    // `rgba16float` destination. The pipeline layout's immediate range spans the
    // full `GpuWeatherAdvectParams` block (auto layout misreflects the immediate
    // size on this driver).
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("vc_advect_bind_group_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("vc_advect_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: size_of::<GpuWeatherAdvectParams>() as u32,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("volumetric_weather_advect_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let src_tex = device.create_texture(&TextureDescriptor {
        label: Some("vc_weather_src"),
        size: Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let dst_tex = device.create_texture(&TextureDescriptor {
        label: Some("vc_weather_dst"),
        size: Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // `write_texture` has no 256-byte row-alignment requirement, so the dense
    // `rgba32float` source uploads as-is.
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &src_tex,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(&src_data),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(WIDTH * 16),
            rows_per_image: Some(HEIGHT),
        },
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );

    let src_view = src_tex.create_view(&TextureViewDescriptor::default());
    let dst_view = dst_tex.create_view(&TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("vc_advect_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&src_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&dst_view),
            },
        ],
    });

    let row_bytes = WIDTH * OUT_BYTES_PER_TEXEL;
    let readback_size = u64::from(row_bytes * HEIGHT);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("vc_advect_readback"),
        size: readback_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("vc_advect_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("vc_advect_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, bytemuck::bytes_of(&params));
        pass.dispatch_workgroups(WIDTH / 8, HEIGHT / 8, 1);
    }
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &dst_tex,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(HEIGHT),
            },
        },
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("device poll should complete the submitted work");

    let view = readback
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let halves: Vec<u16> = bytemuck::cast_slice::<u8, u16>(&view).to_vec();
    drop(view);
    readback.unmap();

    // Row is dense (row_bytes == WIDTH * 8), so cell (x, y) maps to texel
    // n = y*WIDTH + x with channel c at half-word 4*n + c.
    let mut checked = 0u32;
    let mut y = 0u32;
    while y < HEIGHT {
        let mut x = 0u32;
        while x < WIDTH {
            let n = (y * WIDTH + x) as usize;
            let gpu = [
                f16_to_f32(halves[4 * n]),
                f16_to_f32(halves[4 * n + 1]),
                f16_to_f32(halves[4 * n + 2]),
                f16_to_f32(halves[4 * n + 3]),
            ];
            let cpu = advected.get(x, y).to_rgba();
            let mut c = 0usize;
            while c < 4 {
                let d = (gpu[c] - cpu[c]).abs();
                assert!(
                    d < PARITY_EPS,
                    "cell ({x}, {y}) channel {c}: gpu={} cpu={} |d|={d}",
                    gpu[c],
                    cpu[c],
                );
                c += 1;
            }
            checked += 1;
            x += 1;
        }
        y += 1;
    }
    assert_eq!(
        checked,
        WIDTH * HEIGHT,
        "every weather cell must be compared"
    );
}
