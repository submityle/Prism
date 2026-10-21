//! Full production resolve dispatch, not a test-owned BRDF entry point.
//! The reference remains a test oracle, never a realtime shader dependency.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{
    Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ShaderDefVal, ValidateShader,
};
use prism_render_material::{
    GpuMaterialHeader, GpuSurfaceParameters, LobeMask, SurfaceParameterBlock,
};
use prism_render_shading::{
    reference_pt_adapter::ReferencePtMaterialAdapter, GpuShadingPrimitive, GpuShadingVertex,
    SurfaceSample,
};
use wgpu::util::DeviceExt;

const EDGE: u32 = 4;
const PIXELS: usize = 16;
const ROW_BYTES: u32 = 256;
const GEOMETRY_OPACITY: f32 = 0.375;
const GENERATION: u32 = 7;
const CAMERA: [f32; 3] = [0.0, 2.0, 2.0];
const NORMAL: [f32; 3] = [0.0, 1.0, 0.0];
const ALBEDO: [f32; 3] = [0.25, 0.5, 0.75];
const POSITIONS: [[f32; 3]; 3] = [[-1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]];

/// Data-only fixture definition shared by the GPU ABI upload and the
/// independent reference construction. This function contains no evaluation
/// logic and does not import or call a production BSDF.
fn fixture_surface(coat: f32, darkening: f32) -> SurfaceSample {
    SurfaceSample {
        base_color: ALBEDO,
        diffuse_roughness: 0.0,
        specular_weight: 0.0,
        clearcoat: coat,
        clearcoat_roughness: 0.2,
        coat_ior: 1.6,
        coat_color: [1.0; 3],
        coat_darkening: darkening,
        geometry_opacity: GEOMETRY_OPACITY,
        ..Default::default()
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    shading_class: u32,
    width: u32,
    height: u32,
    flags: u32,
    view_position: [f32; 4],
    max_full_lobes: u32,
    frame_index: u32,
}

fn load(_: &(), source: ShaderCacheSource, _: &ValidateShader) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!(),
    }
}

fn production_wgsl() -> String {
    let mut cache = ShaderCache::new((), load);
    let mut tag = 0x5052_4953_4d5f_5032_485f_4750_5500_0000u128;
    macro_rules! register {
        ($name:literal) => {{
            tag += 1;
            cache.set_shader(
                AssetId::Uuid {
                    uuid: Uuid::from_u128(tag),
                },
                Shader::from_wesl(
                    include_str!(concat!("../src/shaders/", $name, ".wesl")),
                    concat!("embedded://prism_render_scene/shaders/", $name, ".wesl"),
                ),
            );
        }};
    }
    register!("tangent");
    register!("material_sample");
    register!("material_virtual_texture");
    register!("surface");
    register!("lighting");
    register!("brdf");
    register!("material");
    register!("material_unpack");
    register!("gpu_scene");
    register!("cloth");
    register!("subsurface");
    register!("hair");
    register!("water");
    register!("clearcoat");
    register!("shadow");
    register!("specular_aa");
    register!("thin_film");
    register!("glint");
    register!("dynamic");
    register!("lobe_sampling");
    register!("lobe_budget");
    register!("dynamic_mask");
    register!("measured_brdf");
    register!("measured_lobe");
    cache.set_shader(
        AssetId::Uuid {
            uuid: Uuid::from_u128(tag + 1),
        },
        Shader::from_wesl(
            include_str!("../../../crates/bevy_render/src/maths.wesl"),
            "embedded://bevy_render/maths.wesl",
        ),
    );
    let resolve = AssetId::Uuid {
        uuid: Uuid::from_u128(tag + 2),
    };
    cache.set_shader(
        resolve,
        Shader::from_wesl(
            include_str!("../src/shaders/shading_resolve.wesl"),
            "embedded://prism_render_scene/shaders/shading_resolve.wesl",
        ),
    );
    (*cache
        .get(
            0,
            resolve,
            &[ShaderDefVal::Bool("MATERIAL_BINDLESS".into(), false)],
        )
        .expect("link unchanged production shading_resolve"))
    .clone()
}

fn unavailable(reason: impl std::fmt::Display) -> Option<(wgpu::Device, wgpu::Queue)> {
    assert!(
        std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU: full production resolve GPU unavailable: {reason}"
    );
    eprintln!("P2-h production resolve GPU SKIP: {reason}");
    None
}

fn hardware_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = match block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(error) => return unavailable(error),
    };
    let info = adapter.get_info();
    eprintln!("P2-h production resolve adapter: {info:?}");
    if !matches!(
        info.device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
    ) {
        return unavailable("a physical integrated/discrete GPU is required");
    }
    let mut limits = adapter.limits();
    if !adapter.features().contains(wgpu::Features::IMMEDIATES)
        || limits.max_immediate_size < 40
        || limits.max_bind_groups < 8
        || limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
    {
        return unavailable(format!("full resolve limits not available: {limits:?}"));
    }
    let mut storage_buffers = 0;
    let mut uniform_buffers = 0;
    let mut storage_textures = 0;
    let mut sampled_textures = 0;
    let mut samplers = 0;
    for group in production_layout_entries() {
        if group.len() as u32 > limits.max_bindings_per_bind_group {
            return unavailable("full resolve exceeds max_bindings_per_bind_group");
        }
        for entry in group {
            match entry.ty {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { .. },
                    ..
                } => storage_buffers += 1,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                } => uniform_buffers += 1,
                wgpu::BindingType::StorageTexture { .. } => storage_textures += 1,
                wgpu::BindingType::Texture { .. } => sampled_textures += 1,
                wgpu::BindingType::Sampler(_) => samplers += 1,
                _ => unreachable!("unexpected production fixture binding type"),
            }
        }
    }
    if storage_buffers > limits.max_storage_buffers_per_shader_stage
        || uniform_buffers > limits.max_uniform_buffers_per_shader_stage
        || storage_buffers + uniform_buffers
            > limits.max_buffers_and_acceleration_structures_per_shader_stage
        || storage_textures > limits.max_storage_textures_per_shader_stage
        || sampled_textures > limits.max_sampled_textures_per_shader_stage
        || samplers > limits.max_samplers_per_shader_stage
    {
        return unavailable(format!(
            "full eight-group layout needs {storage_buffers} storage buffers, \
             {uniform_buffers} uniform buffers, {storage_textures} storage textures, \
             {sampled_textures} sampled textures, {samplers} samplers; adapter limits: {limits:?}"
        ));
    }
    for format in [
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rg16Float,
    ] {
        if !adapter
            .get_texture_format_features(format)
            .allowed_usages
            .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            return unavailable(format!("{format:?} storage output unsupported"));
        }
    }
    // Full resolve uses far more storage resources than wgpu's default limits.
    // Request the adapter's actual supported limits, not a reduced test kernel.
    limits.max_immediate_size = 40;
    match block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: limits,
        ..Default::default()
    })) {
        Ok(pair) => Some(pair),
        Err(error) => unavailable(error),
    }
}

enum Resource {
    Buffer(wgpu::Buffer),
    Texture(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

impl Resource {
    fn binding(&self) -> wgpu::BindingResource<'_> {
        match self {
            Self::Buffer(buffer) => buffer.as_entire_binding(),
            Self::Texture(view) => wgpu::BindingResource::TextureView(view),
            Self::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
        }
    }
}

fn buffer(device: &wgpu::Device, bytes: &[u8], uniform: bool) -> Resource {
    Resource::Buffer(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("production resolve fixture ABI row"),
            contents: bytes,
            usage: if uniform {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
            },
        }),
    )
}

fn zero(device: &wgpu::Device, bytes: usize, uniform: bool) -> Resource {
    buffer(device, &vec![0; bytes], uniform)
}

fn texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layers: u32,
    storage: bool,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("production resolve fixture texture"),
        size: wgpu::Extent3d {
            width: EDGE,
            height: EDGE,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | if storage {
                wgpu::TextureUsages::STORAGE_BINDING
            } else {
                wgpu::TextureUsages::TEXTURE_BINDING
            },
        view_formats: &[],
    })
}

fn view(texture: &wgpu::Texture, dimension: wgpu::TextureViewDimension) -> Resource {
    Resource::Texture(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(dimension),
        ..Default::default()
    }))
}

fn sampled(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    dimension: wgpu::TextureViewDimension,
) -> Resource {
    let layers = if dimension == wgpu::TextureViewDimension::Cube {
        6
    } else {
        1
    };
    view(&texture(device, format, layers, false), dimension)
}

fn sampler(device: &wgpu::Device, filtering: bool) -> Resource {
    let filter = if filtering {
        wgpu::FilterMode::Linear
    } else {
        wgpu::FilterMode::Nearest
    };
    Resource::Sampler(device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: filter,
        min_filter: filter,
        ..Default::default()
    }))
}

fn write_texture<T: bytemuck::Pod>(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    texels: &[T],
    bytes_per_pixel: u32,
) {
    queue.write_texture(
        texture.as_image_copy(),
        bytemuck::cast_slice(texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(EDGE * bytes_per_pixel),
            rows_per_image: Some(EDGE),
        },
        wgpu::Extent3d {
            width: EDGE,
            height: EDGE,
            depth_or_array_layers: 1,
        },
    );
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.map(|x| x / length)
}

fn barycentrics(pixel: usize) -> (u32, [f32; 3]) {
    let x = ((0.1 + (pixel % 4) as f32 * 0.1) * 65535.0).round() as u32;
    let y = ((0.1 + (pixel / 4) as f32 * 0.1) * 65535.0).round() as u32;
    let bx = x as f32 / 65535.0;
    let by = y as f32 / 65535.0;
    (x | (y << 16), [bx, by, (1.0 - bx - by).max(0.0)])
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    match exponent {
        0 => sign * f32::from(fraction) * 2.0f32.powi(-24),
        31 if fraction == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + f32::from(fraction) / 1024.0) * 2.0f32.powi(i32::from(exponent) - 15),
    }
}

// Mirror the portable production layouts, including resources that a fixture's
// disabled lighting branches do not sample. In particular, auto-layout would
// incorrectly require FLOAT32_FILTERABLE for the nearest-sampled VSM atlas.
fn production_layout_entries() -> [Vec<wgpu::BindGroupLayoutEntry>; 8] {
    use wgpu::{BindingType as B, TextureSampleType as T, TextureViewDimension as D};
    let storage = |read_only| B::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let uniform = B::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let texture = |sample_type, view_dimension| B::Texture {
        sample_type,
        view_dimension,
        multisampled: false,
    };
    let sampled = |filterable, dimension| texture(T::Float { filterable }, dimension);
    let output = |format| B::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: D::D2,
    };
    let filtering = B::Sampler(wgpu::SamplerBindingType::Filtering);
    let nonfiltering = B::Sampler(wgpu::SamplerBindingType::NonFiltering);
    let rgba = output(wgpu::TextureFormat::Rgba16Float);
    let groups: [Vec<(u32, B)>; 8] = [
        vec![
            (0, texture(T::Uint, D::D2)),
            (1, texture(T::Uint, D::D2)),
            (2, rgba),
            (3, sampled(false, D::D2)),
            (4, sampled(true, D::Cube)),
            (5, filtering),
            (6, sampled(true, D::D2)),
            (7, filtering),
            (8, rgba),
            (9, rgba),
            (10, output(wgpu::TextureFormat::Rg16Float)),
            (11, uniform),
            (12, rgba),
            (13, rgba),
            (14, rgba),
            (15, storage(false)),
            (16, storage(false)),
            (17, storage(false)),
        ],
        vec![
            (0, storage(true)),
            (1, storage(true)),
            (2, storage(true)),
            (5, storage(true)),
            (6, storage(true)),
            (7, sampled(true, D::D2Array)),
            (8, filtering),
        ],
        (0..11).map(|binding| (binding, storage(true))).collect(),
        (0..3).map(|binding| (binding, storage(true))).collect(),
        vec![
            (0, sampled(true, D::D2Array)),
            (1, filtering),
            (2, storage(true)),
            (3, storage(true)),
            (4, storage(true)),
            (5, storage(true)),
        ],
        (0..3).map(|binding| (binding, storage(true))).collect(),
        vec![
            (0, storage(true)),
            (1, sampled(false, D::D2)),
            (2, nonfiltering),
            (3, uniform),
        ],
        vec![
            (0, storage(true)),
            (1, sampled(false, D::D2)),
            (2, sampled(false, D::D2)),
            (3, storage(true)),
            (4, storage(true)),
            (5, storage(true)),
            (6, storage(true)),
            (7, sampled(false, D::D2)),
            (8, uniform),
            (9, uniform),
        ],
    ];
    groups.map(|entries| {
        entries
            .into_iter()
            .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            })
            .collect()
    })
}

fn run_case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    coat: f32,
    darkening: f32,
) -> Vec<[f32; 4]> {
    use wgpu::{TextureFormat as F, TextureViewDimension as D};
    let ids = texture(device, F::Rgba32Uint, 1, false);
    let metadata = texture(device, F::Rgba32Uint, 1, false);
    let output = texture(device, F::Rgba16Float, 1, true);
    write_texture(queue, &ids, &[[0u32, GENERATION, 0, 0]; PIXELS], 16);
    let meta: Vec<_> = (0..PIXELS)
        .map(|i| [0, GENERATION, 0, barycentrics(i).0])
        .collect();
    write_texture(queue, &metadata, &meta, 16);
    // Negative RGB and alpha=0 make early returns / untouched pixels observable.
    write_texture(queue, &output, &[[0xbc00u16, 0xbc00, 0xbc00, 0]; PIXELS], 8);
    let identity4 = [
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let motion: Vec<_> = identity4.into_iter().chain(identity4).collect();
    let mut group0 = vec![
        (0, view(&ids, D::D2)),
        (1, view(&metadata, D::D2)),
        (2, view(&output, D::D2)),
        (3, sampled(device, F::Rgba16Float, D::D2)),
        (4, sampled(device, F::Rgba16Float, D::Cube)),
        (5, sampler(device, true)),
        (6, sampled(device, F::Rgba16Float, D::D2)),
        (7, sampler(device, true)),
    ];
    for binding in [8, 9, 10, 12, 13, 14] {
        let format = if binding == 10 {
            F::Rg16Float
        } else {
            F::Rgba16Float
        };
        group0.push((binding, view(&texture(device, format, 1, true), D::D2)));
    }
    group0.extend([
        (11, buffer(device, bytemuck::cast_slice(&motion), true)),
        (15, zero(device, PIXELS * 64, false)),
        (16, zero(device, PIXELS * 64, false)),
        (17, zero(device, PIXELS * 16, false)),
    ]);

    let fixture = fixture_surface(coat, darkening);
    let params = GpuSurfaceParameters {
        base_color: [
            fixture.base_color[0],
            fixture.base_color[1],
            fixture.base_color[2],
            1.0,
        ],
        base_metalness: 0.0,
        specular_weight: 0.0,
        base_weight: 1.0,
        base_diffuse_roughness: 0.0,
        coat_weight: fixture.clearcoat,
        coat_roughness: fixture.clearcoat_roughness,
        coat_ior: fixture.coat_ior,
        coat_color: [1.0; 4],
        coat_darkening: fixture.coat_darkening,
        geometry_opacity: fixture.geometry_opacity,
        ..Default::default()
    };
    let mask = LobeMask::CLEARCOAT;
    let words = SurfaceParameterBlock::from_full(&params, mask).pack();
    let header = GpuMaterialHeader {
        generation: GENERATION,
        parameter_size: (words.len() * 4) as u32,
        is_active: 1,
        lobe_mask: mask.bits(),
        ..Default::default()
    };
    let group1 = vec![
        (0, buffer(device, bytemuck::bytes_of(&header), false)),
        (1, buffer(device, bytemuck::cast_slice(&words), false)),
        (2, zero(device, 32, false)),
        (5, zero(device, 16, false)),
        (6, zero(device, 32, false)),
        (7, sampled(device, F::Rgba16Float, D::D2Array)),
        (8, sampler(device, true)),
    ];
    let work: Vec<[u32; 8]> = (0..PIXELS as u32)
        .map(|i| [i, 0, 0, 0, 0, u32::MAX, u32::MAX, u32::MAX])
        .collect();
    // gpu_scene.wesl: generation,flags,geometry(index,generation),
    // material(index,generation),render_layers,is_active; identity affine rows.
    let instance = [GENERATION, 0, 0, GENERATION, 0, GENERATION, u32::MAX, 1];
    let affine = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    // RenderShadingGeometryHeader / ShadingGeometryHeader: generation, revision,
    // vertex_offset/count, primitive_offset/count, flags (ACTIVE=1), padding.
    // Like the instance and transform rows above, this is uploaded ABI data,
    // not an alternate reconstruction or shading implementation.
    let geometry = [GENERATION, 0, 0, 3, 0, 1, 1, 0];
    let vertices = POSITIONS.map(|position| GpuShadingVertex {
        position,
        normal: NORMAL,
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv: [position[0], position[2]],
        flags: 0,
        ..Default::default()
    });
    let primitive = GpuShadingPrimitive {
        indices: [0, 1, 2],
        flags: 0,
    };
    let group2 = vec![
        (0, buffer(device, bytemuck::cast_slice(&work), false)),
        (1, buffer(device, bytemuck::bytes_of(&0u32), false)),
        (
            2,
            buffer(device, bytemuck::bytes_of(&(PIXELS as u32)), false),
        ),
        (3, buffer(device, bytemuck::bytes_of(&instance), false)),
        (4, buffer(device, bytemuck::bytes_of(&geometry), false)),
        (5, buffer(device, bytemuck::cast_slice(&vertices), false)),
        (6, buffer(device, bytemuck::bytes_of(&primitive), false)),
        (7, buffer(device, bytemuck::bytes_of(&affine), false)),
        (8, buffer(device, bytemuck::bytes_of(&affine), false)),
        (9, buffer(device, bytemuck::bytes_of(&u32::MAX), false)),
        (10, zero(device, 16, false)),
    ];
    let light = normalize([0.35, 0.85, 0.25]);
    let directional = [light[0], light[1], light[2], 1.0, 1.0, 1.0, 1.0, 0.0];
    // LightEnvironment: ambient=0, one directional, SH=0, no punctual/IBL.
    let mut environment = [0u32; 60];
    environment[3] = 1;
    environment[42] = 1.0f32.to_bits();
    let group3 = vec![
        (0, buffer(device, bytemuck::bytes_of(&directional), false)),
        (1, zero(device, 64, false)),
        (2, buffer(device, bytemuck::bytes_of(&environment), false)),
    ];
    let group4 = vec![
        (0, sampled(device, F::Rgba16Float, D::D2Array)),
        (1, sampler(device, true)),
        (2, zero(device, 368, false)),
        (3, zero(device, 64, false)),
        (4, zero(device, 16, false)),
        (5, zero(device, 128, false)),
    ];
    let group5 = vec![
        (0, zero(device, 128, false)),
        (1, zero(device, 8, false)),
        (2, zero(device, 4, false)),
    ];
    let group6 = vec![
        (0, zero(device, 4, false)),
        (1, sampled(device, F::R32Float, D::D2)),
        (2, sampler(device, false)),
        (3, zero(device, 96, true)),
    ];
    let high = prism_render_shading::ShadingQuality::High.budget();
    let budget = [
        high.sss_separable_taps,
        high.sss_preintegration_steps,
        high.translucency_profile_samples,
        high.multiscatter_dir_samples,
        high.multiscatter_march_samples,
        high.total_samples(),
        1,
        u32::MAX,
    ];
    let group7 = vec![
        (0, zero(device, 80, false)),
        (1, sampled(device, F::Rgba32Float, D::D2)),
        (2, sampled(device, F::Rgba32Float, D::D2)),
        (3, zero(device, 32, false)),
        (4, zero(device, 16, false)),
        (5, zero(device, 56, false)),
        (6, zero(device, 48, false)),
        (7, sampled(device, F::Rgba16Float, D::D2)),
        (8, zero(device, 32, true)),
        (9, buffer(device, bytemuck::bytes_of(&budget), true)),
    ];
    let resources = [
        group0, group1, group2, group3, group4, group5, group6, group7,
    ];
    let groups: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(index, resources)| {
            let entries: Vec<_> = resources
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("full production resolve fixture group"),
                layout: &pipeline.get_bind_group_layout(index as u32),
                entries: &entries,
            })
        })
        .collect();
    assert_eq!(groups.len(), 8);
    let image_bytes = u64::from(ROW_BYTES * EDGE);
    let response_bytes = (PIXELS * 64) as u64;
    let control_bytes = (PIXELS * 16) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("production resolve 4x4 HDR and response/control readback"),
        size: image_bytes + 2 * response_bytes + control_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        for (index, group) in groups.iter().enumerate() {
            pass.set_bind_group(index as u32, group, &[]);
        }
        pass.set_immediates(
            0,
            bytemuck::bytes_of(&Params {
                shading_class: 0,
                width: EDGE,
                height: EDGE,
                flags: 0,
                view_position: [CAMERA[0], CAMERA[1], CAMERA[2], 1.0],
                max_full_lobes: u32::MAX,
                frame_index: 0,
            }),
        );
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(EDGE),
            },
        },
        wgpu::Extent3d {
            width: EDGE,
            height: EDGE,
            depth_or_array_layers: 1,
        },
    );
    for (binding, offset, size) in [
        (15, image_bytes, response_bytes),
        (16, image_bytes + response_bytes, response_bytes),
        (17, image_bytes + 2 * response_bytes, control_bytes),
    ] {
        let (_, Resource::Buffer(source)) = resources[0]
            .iter()
            .find(|(slot, _)| *slot == binding)
            .expect("production response/control output buffer")
        else {
            panic!("production response/control binding must be a buffer");
        };
        encoder.copy_buffer_to_buffer(source, 0, &readback, offset, size);
    }
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("map production resolve readback")
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("finish production resolve dispatch");
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("mapped production pixels");
    let pixels = (0..PIXELS)
        .map(|i| {
            let offset = (i / 4) * ROW_BYTES as usize + (i % 4) * 8;
            std::array::from_fn(|channel| {
                let start = offset + channel * 2;
                half_to_f32(u16::from_le_bytes([mapped[start], mapped[start + 1]]))
            })
        })
        .collect();
    let word = |offset: usize| {
        u32::from_le_bytes(
            mapped[offset..offset + 4]
                .try_into()
                .expect("readback word"),
        )
    };
    for pixel in 0..PIXELS {
        // Coverage is a surface-parameter/response ABI value, not a BSDF lobe
        // or the scene_color alpha (which the production resolve sets to 1).
        for arm in 0..2 {
            let offset = image_bytes as usize + arm * response_bytes as usize + pixel * 64;
            assert_eq!(
                f32::from_bits(word(offset + 15 * 4)),
                GEOMETRY_OPACITY,
                "coat={coat} darkening={darkening} pixel={pixel} arm={arm} opacity"
            );
        }
        let offset = (image_bytes + 2 * response_bytes) as usize + pixel * 16;
        assert_eq!(f32::from_bits(word(offset)), 0.0, "single-arm mask");
        assert_eq!(word(offset + 4), 0, "single-arm channels");
        assert_eq!(f32::from_bits(word(offset + 8)), GEOMETRY_OPACITY);
        assert_eq!(word(offset + 12), 1, "production alpha composition rule");

        // The fixture has no indirect, environment, transmission or emissive
        // contribution. Check every response word, not only opacity, so a
        // stale/untouched response buffer cannot make the image comparison
        // appear valid. The right arm is expected to be an exact single-arm
        // clone because the work item does not request a split response.
        let left_offset = image_bytes as usize + pixel * 64;
        let right_offset = image_bytes as usize + response_bytes as usize + pixel * 64;
        for word_index in 0..16 {
            let left = f32::from_bits(word(left_offset + word_index * 4));
            let right = f32::from_bits(word(right_offset + word_index * 4));
            assert!(
                left.is_finite() && left >= 0.0,
                "coat={coat} darkening={darkening} pixel={pixel} left response[{word_index}]={left}"
            );
            assert_eq!(
                right.to_bits(),
                left.to_bits(),
                "coat={coat} darkening={darkening} pixel={pixel} right response[{word_index}] diverged"
            );
            if (3..15).contains(&word_index) {
                assert_eq!(
                    left,
                    0.0,
                    "coat={coat} darkening={darkening} pixel={pixel} unexpected non-direct response[{word_index}]"
                );
            }
        }
    }
    drop(mapped);
    readback.unmap();
    pixels
}

#[test]
fn production_shading_resolve_links_and_validates_without_a_gpu() {
    // Always exercise the unchanged production import closure, including on a
    // headless runner where the dispatch test explicitly skips.
    let source = production_wgsl();
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    assert!(module.entry_points.iter().any(|entry| {
        entry.name == "shading_resolve"
            && entry.stage == wgpu::naga::ShaderStage::Compute
            && entry.workgroup_size == [64, 1, 1]
    }));
    let groups: std::collections::BTreeSet<_> = module
        .global_variables
        .iter()
        .filter_map(|(_, var)| var.binding.as_ref().map(|binding| binding.group))
        .collect();
    assert_eq!(
        groups,
        (0..8).collect(),
        "retain all production bind groups"
    );
    let layout = production_layout_entries();
    let declared: std::collections::BTreeSet<_> = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| {
            variable
                .binding
                .as_ref()
                .map(|binding| (binding.group, binding.binding))
        })
        .collect();
    let provided: std::collections::BTreeSet<_> = layout
        .iter()
        .enumerate()
        .flat_map(|(group, entries)| {
            entries
                .iter()
                .map(move |entry| (group as u32, entry.binding))
        })
        .collect();
    assert_eq!(declared, provided, "portable production binding ABI drift");
}

#[test]
fn production_shading_resolve_readback_matches_reference_pt_clearcoat() {
    let Some((device, queue)) = hardware_device() else {
        return;
    };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("unchanged production shading_resolve.wesl"),
        source: wgpu::ShaderSource::Wgsl(production_wgsl().into()),
    });
    let layouts = production_layout_entries().map(|entries| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("portable production resolve layout"),
            entries: &entries,
        })
    });
    let layout_refs = layouts.each_ref().map(Some);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("full production resolve pipeline layout"),
        bind_group_layouts: &layout_refs,
        immediate_size: std::mem::size_of::<Params>() as u32,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("full eight-group production resolve"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("shading_resolve"),
        compilation_options: Default::default(),
        cache: None,
    });
    let light = normalize([0.35, 0.85, 0.25]);
    let mut images = Vec::new();
    for (coat, darkening) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)] {
        let reference = ReferencePtMaterialAdapter::from_surface(fixture_surface(coat, darkening))
            .expect("representable Lambert plus white coat");
        let actual = run_case(&device, &queue, &pipeline, coat, darkening);
        assert_eq!(actual.len(), PIXELS);
        let mut max_abs = 0.0f32;
        let mut reference_first = [0.0; 3];
        for (i, rgba) in actual.iter().enumerate() {
            assert!(rgba.iter().all(|x| x.is_finite()), "pixel {i}: {rgba:?}");
            assert_eq!(
                rgba[3], 1.0,
                "pixel {i} was not written by production resolve: {rgba:?}"
            );
            assert!(
                rgba[..3].iter().all(|x| *x >= 0.0),
                "sentinel/invalid RGB at {i}: {rgba:?}"
            );
            let (_, bary) = barycentrics(i);
            let position: [f32; 3] =
                std::array::from_fn(|axis| (0..3).map(|v| POSITIONS[v][axis] * bary[v]).sum());
            let view = normalize(std::array::from_fn(|axis| CAMERA[axis] - position[axis]));
            let expected = reference
                .evaluate(view, light, NORMAL)
                .map(|value| value * light[1]);
            assert!(
                expected
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0),
                "reference produced invalid HDR at pixel {i}: {expected:?}"
            );
            if i == 0 {
                reference_first = expected;
            }
            for channel in 0..3 {
                max_abs = max_abs.max((rgba[channel] - expected[channel]).abs());
                let tolerance = 5.0e-4 + 1.0e-3 * expected[channel].abs();
                assert!((rgba[channel] - expected[channel]).abs() <= tolerance,
                    "coat={coat} darkening={darkening} pixel={i} channel={channel}: GPU={rgba:?} reference={expected:?}");
            }
        }
        eprintln!("P2-h REAL GPU READBACK coat={coat} darkening={darkening}: 16/16 HDR + both response arms + composition controls written, opacity={GEOMETRY_OPACITY}, first={:?}, reference_first={reference_first:?}, max_abs={max_abs:.8}", actual[0]);
        images.push(actual);
    }
    // Passing three isolated comparisons must also prove the controls changed output.
    assert!(images[0]
        .iter()
        .zip(&images[1])
        .any(|(a, b)| (a[0] - b[0]).abs() > 1.0e-3));
    assert!(images[1]
        .iter()
        .zip(&images[2])
        .all(|(a, b)| b[0] < a[0] && b[1] < a[1] && b[2] < a[2]));
}
