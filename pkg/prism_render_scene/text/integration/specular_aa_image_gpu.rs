//! Real production specular-AA image fixture with GPU readback.
//!
//! Each fixture pixel carries a complete production `SurfaceSample`, shading
//! frame, analytic light, and the two same-frame normal-footprint differences.
//! The GPU computes the normal variance through `specular_aa.wesl`, feeds it to
//! the production `brdf.wesl` response, and returns one HDR RGB pixel plus the
//! filtered GGX alpha. The CPU builds the same per-pixel golden through the
//! public shading reference. This is a small deterministic image gate, not a
//! shader-only link test or a CPU dispatch emulation.
//!
//! Run on a real device with:
//! `PRISM_REQUIRE_GPU=1 CARGO_INCREMENTAL=0 cargo test -p prism_render_scene
//! --test specular_aa_image_gpu -- --nocapture`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_math::Vec3;
use prism_render_shading::{
    evaluate_principled_direct_channels, DirectLightSample, ShadingFrame, SurfaceSample,
};
use std::fmt::Write;
use std::mem::size_of_val;
use wgpu::util::DeviceExt;

const WIDTH: usize = 29;
const HEIGHT: usize = 17;
const PIXELS: usize = WIDTH * HEIGHT;
const WORKGROUP: usize = 64;
const OUTPUT_ROWS: usize = 1;
// CPU and Metal both execute the same production equations, but the backend
// may fuse multiply-adds differently from the host reference. 64 ULP keeps
// the image gate strict at HDR magnitudes while admitting that bounded
// implementation-level rounding difference.
const MAX_ULPS: u32 = 64;
const SIGMA2: f32 = 0.159_154_94;
const KAPPA_MAX: f32 = 0.18;
const GUARD: [f32; 4] = [-12_345.0; 4];
// Keep the production group(7)/binding(9) budget ABI bound while this fixture
// evaluates the BRDF.  The default High tier is intentionally used here:
// specular-AA pixels do not enable SSS, but the production BRDF still imports
// and reads the shared budget block before selecting its lobe path.
const GPU_LOBE_BUDGET: [u32; 8] = [13, 160, 48, 64, 16, 301, 0, u32::MAX];

macro_rules! surface_fields {
    ($apply:ident) => {
        $apply! {
            base_color: 3, base_weight: 1, diffuse_roughness: 1, metallic: 1,
            perceptual_roughness: 1, specular_weight: 1, specular_color: 3,
            specular_ior: 1, transmission: 1, index_of_refraction: 1,
            dispersion: 1, thin_walled: 1, transmission_color: 3,
            transmission_scatter: 3, transmission_scatter_anisotropy: 1,
            transmission_dispersion_abbe_number: 1, coat_darkening: 1,
            scatter_anisotropy: 1, ambient_occlusion: 1, emissive: 3,
            clearcoat: 1, clearcoat_roughness: 1, coat_ior: 1, coat_color: 3,
            coat_normal: 3, coat_roughness_anisotropy: 1, coat_tangent: 3,
            sheen: 1, fuzz_roughness: 1, fuzz_color: 3, subsurface: 1,
            subsurface_color: 3, subsurface_radius: 1, subsurface_radius_scale: 3,
            thickness: 1, anisotropy: 1, anisotropy_rotation: 1,
            normal_variance: 1, iridescence: 1, iridescence_thickness_nm: 1,
            iridescence_ior: 1, glint_weight: 1, glint_roughness: 1,
            glint_density: 1, glint_bin_half_width: 1, glint_state: 1,
            glint_cell: 2, glint_footprint_area: 1,
        }
    };
}

trait ScalarWords {
    fn append(self, words: &mut Vec<f32>);
}

impl ScalarWords for f32 {
    fn append(self, words: &mut Vec<f32>) {
        words.push(self);
    }
}

impl<const N: usize> ScalarWords for [f32; N] {
    fn append(self, words: &mut Vec<f32>) {
        words.extend(self);
    }
}

macro_rules! define_transport {
    ($($field:ident: $width:literal),* $(,)?) => {
        const SURFACE_FIELDS: &[(&str, usize)] = &[$((stringify!($field), $width)),*];

        fn surface_words(surface: SurfaceSample) -> Vec<f32> {
            let mut words = Vec::new();
            $(surface.$field.append(&mut words);)*
            words
        }
    };
}

surface_fields!(define_transport);

#[derive(Clone, Copy)]
struct Pixel {
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
    ddx_normal: [f32; 3],
    ddy_normal: [f32; 3],
}

fn pixels() -> Vec<Pixel> {
    let surface = SurfaceSample {
        base_color: [0.92, 0.97, 1.0],
        perceptual_roughness: 0.08,
        specular_ior: 1.5,
        ..Default::default()
    };
    let light = DirectLightSample {
        direction: [0.22, 0.96, 0.17],
        illuminance: [3.0, 2.5, 2.0],
        visibility: 1.0,
    };
    let mut result = Vec::with_capacity(PIXELS);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let u = x as f32 / (WIDTH - 1) as f32;
            let v = y as f32 / (HEIGHT - 1) as f32;
            let normal = [0.22 * (u * 2.0 - 1.0), 1.0, 0.16 * (v * 2.0 - 1.0)];
            let frequency = if (x / 4 + y / 3) % 2 == 0 { 1.0 } else { 0.18 };
            let ddx_normal = [
                0.035 * frequency + (x % 7) as f32 * 0.001,
                0.012 * frequency,
                0.0,
            ];
            let ddy_normal = [
                0.0,
                0.028 * frequency,
                0.025 * frequency + (y % 5) as f32 * 0.001,
            ];
            result.push(Pixel {
                surface,
                frame: ShadingFrame {
                    normal,
                    view: [0.28, 0.95, 0.12],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, 1.0],
                },
                light,
                ddx_normal,
                ddy_normal,
            });
        }
    }
    result
}

fn input_words(pixel: &Pixel) -> Vec<f32> {
    let mut words = surface_words(pixel.surface);
    words.extend(pixel.frame.normal);
    words.extend(pixel.frame.view);
    words.extend(pixel.frame.tangent);
    words.extend(pixel.frame.bitangent);
    words.extend(pixel.light.direction);
    words.extend(pixel.light.illuminance);
    words.push(pixel.light.visibility);
    words.extend(pixel.ddx_normal);
    words.extend(pixel.ddy_normal);
    words
}

fn cpu_pixel(pixel: &Pixel) -> [f32; 4] {
    let mut surface = pixel.surface;
    surface.normal_variance = prism_render_shading::gi::specular_aa::screen_space_variance(
        Vec3::from_array(pixel.ddx_normal),
        Vec3::from_array(pixel.ddy_normal),
        SIGMA2,
    );
    let channels = evaluate_principled_direct_channels(surface, pixel.frame, pixel.light);
    let base_alpha = surface.perceptual_roughness * surface.perceptual_roughness;
    let alpha = prism_render_shading::gi::specular_aa::normal_variance::filter_alpha_sq(
        base_alpha * base_alpha,
        surface.normal_variance,
        KAPPA_MAX,
    )
    .sqrt();
    [
        channels.direct[0],
        channels.direct[1],
        channels.direct[2],
        alpha,
    ]
}

fn shader_id(index: usize) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_5350_4543_494d_4700_0000 + index as u128),
    }
}

fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("production specular AA is WESL"),
    }
}

fn compile_shader() -> String {
    let production = include_str!("../../src/shaders/brdf.wesl");
    let body = production
        .split_once("struct SurfaceSample {")
        .expect("production SurfaceSample")
        .1
        .split_once('}')
        .expect("SurfaceSample closing brace")
        .0;
    let fields: Vec<_> = body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("//"))
        .map(|line| {
            let (field, ty) = line.trim_end_matches(',').split_once(':').unwrap();
            (field.trim(), ty.trim())
        })
        .collect();
    let expected: Vec<_> = SURFACE_FIELDS
        .iter()
        .map(|(field, width)| {
            (
                *field,
                match width {
                    1 => "f32",
                    2 => "vec2<f32>",
                    3 => "vec3<f32>",
                    _ => unreachable!(),
                },
            )
        })
        .collect();
    assert_eq!(
        fields, expected,
        "production SurfaceSample transport drifted"
    );

    let surface_size: usize = SURFACE_FIELDS.iter().map(|(_, width)| width).sum();
    let frame_size = 12;
    let light_size = 7;
    let derivative_offset = surface_size + frame_size + light_size;
    let stride = derivative_offset + 6;
    let mut wrapper = format!(
        r#"
import prism_render_scene::shaders::brdf::{{
    SurfaceSample, ShadingFrame, DirectLightSample, principled_pre,
    principled_direct_channels_with,
}};
import prism_render_scene::shaders::specular_aa::{{
    spec_aa_screen_space_variance, spec_aa_filter_alpha_sq,
}};
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4<f32>>;
fn load3(offset: u32) -> vec3<f32> {{
    return vec3<f32>(inputs[offset], inputs[offset + 1u], inputs[offset + 2u]);
}}
@compute @workgroup_size(64)
fn specular_aa_image(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x >= {pixels}u {{ return; }}
    let base = id.x * {stride}u;
    var surface: SurfaceSample;
"#,
        pixels = PIXELS,
        stride = stride,
    );
    let mut offset = 0;
    for (field, width) in SURFACE_FIELDS {
        let expression = match width {
            1 => format!("inputs[base + {offset}u]"),
            2 => format!(
                "vec2<f32>(inputs[base + {offset}u], inputs[base + {}u])",
                offset + 1
            ),
            3 => format!("load3(base + {offset}u)"),
            _ => unreachable!(),
        };
        writeln!(wrapper, "    surface.{field} = {expression};").unwrap();
        offset += width;
    }
    write!(
        wrapper,
        r#"
    let f = base + {surface_size}u;
    let frame = ShadingFrame(load3(f), load3(f + 3u), load3(f + 6u), load3(f + 9u));
    let light = DirectLightSample(load3(f + 12u), load3(f + 15u), inputs[f + 18u]);
    let d = base + {derivative_offset}u;
    let ddx = load3(d);
    let ddy = load3(d + 3u);
    surface.normal_variance = spec_aa_screen_space_variance(ddx, ddy, {sigma2});
    let pre = principled_pre(surface, frame);
    let channels = principled_direct_channels_with(surface, frame, light, pre);
    let base_alpha = surface.perceptual_roughness * surface.perceptual_roughness;
    let alpha = sqrt(spec_aa_filter_alpha_sq(
        base_alpha * base_alpha,
        surface.normal_variance,
        {kappa},
    ));
    outputs[id.x] = vec4<f32>(channels.direct, alpha);
}}
"#,
        surface_size = surface_size,
        derivative_offset = derivative_offset,
        sigma2 = SIGMA2,
        kappa = KAPPA_MAX,
    )
    .unwrap();

    let wrapper: &'static str = Box::leak(wrapper.into_boxed_str());
    let modules = [
        ("lighting", include_str!("../../src/shaders/lighting.wesl")),
        ("tangent", include_str!("../../src/shaders/tangent.wesl")),
        (
            "specular_aa",
            include_str!("../../src/shaders/specular_aa.wesl"),
        ),
        (
            "lobe_budget",
            include_str!("../../src/shaders/lobe_budget.wesl"),
        ),
        (
            "thin_film",
            include_str!("../../src/shaders/thin_film.wesl"),
        ),
        ("glint", include_str!("../../src/shaders/glint.wesl")),
        ("cloth", include_str!("../../src/shaders/cloth.wesl")),
        (
            "subsurface",
            include_str!("../../src/shaders/subsurface.wesl"),
        ),
        ("brdf", production),
        ("specular_aa_image", wrapper),
    ];
    let mut cache = ShaderCache::new((), keep_wgsl);
    for (index, (name, source)) in modules.iter().enumerate() {
        cache.set_shader(
            shader_id(index),
            Shader::from_wesl(
                *source,
                format!("embedded://prism_render_scene/shaders/{name}.wesl"),
            ),
        );
    }
    (*cache
        .get(0, shader_id(modules.len() - 1), &[])
        .unwrap_or_else(|error| panic!("production specular-AA image WESL failed: {error}")))
    .clone()
}

fn unavailable(message: impl std::fmt::Display) -> Option<(wgpu::Device, wgpu::Queue)> {
    assert!(
        std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU set: required specular-AA GPU unavailable: {message}"
    );
    eprintln!("specular-AA image GPU SKIP: {message}");
    None
}

fn device(bytes: u64) -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: Default::default(),
    });
    let adapter = match block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(error) => return unavailable(error),
    };
    eprintln!("specular-AA image adapter: {:?}", adapter.get_info());
    if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
        return unavailable("software/CPU adapter is not hardware image parity");
    }
    let limits = adapter.limits();
    if limits.max_compute_invocations_per_workgroup < WORKGROUP as u32
        || limits.max_compute_workgroup_size_x < WORKGROUP as u32
        || limits.max_storage_buffers_per_shader_stage < 2
        || limits.max_bind_groups < 8
        || u64::from(limits.max_storage_buffer_binding_size) < bytes
        || limits.max_buffer_size < bytes
    {
        return unavailable(format!(
            "needs x64, eight bind groups, and two {bytes}-byte storage bindings; {limits:?}"
        ));
    }
    match block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        // Request the adapter's advertised limits verbatim.  Asking for an
        // otherwise-default limit set caused wgpu to expose the default four
        // bind groups to validation even though this production shader ABI
        // legitimately uses group 7 for the shared lobe budget.
        required_limits: limits,
        ..Default::default()
    })) {
        Ok(pair) => {
            eprintln!(
                "specular-AA image device max_bind_groups: {}",
                pair.0.limits().max_bind_groups
            );
            assert!(
                pair.0.limits().max_bind_groups >= 8,
                "requested production group-7 ABI but device exposes only {} bind groups",
                pair.0.limits().max_bind_groups
            );
            Some(pair)
        }
        Err(error) => unavailable(error),
    }
}

fn ulps(a: f32, b: f32) -> u32 {
    if !a.is_finite() || !b.is_finite() {
        return u32::MAX;
    }
    fn ordered(value: f32) -> u32 {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            0x8000_0000 - (bits & 0x7fff_ffff)
        } else {
            0x8000_0000 + bits
        }
    }
    ordered(a).abs_diff(ordered(b))
}

#[test]
fn real_gpu_specular_aa_image_matches_cpu_golden() {
    let pixels = pixels();
    assert_eq!(pixels.len(), PIXELS);
    let input: Vec<f32> = pixels.iter().flat_map(input_words).collect();
    let initial = vec![GUARD; PIXELS + WORKGROUP];
    let input_bytes = size_of_val(input.as_slice()) as u64;
    let output_bytes = size_of_val(initial.as_slice()) as u64;
    let Some((device, queue)) = device(input_bytes.max(output_bytes)) else {
        return;
    };
    let source = compile_shader();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production specular-AA image"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("production specular-AA image"),
        layout: None,
        module: &module,
        entry_point: Some("specular_aa_image"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("specular-AA image inputs"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("specular-AA image outputs and guard"),
        contents: bytemuck::cast_slice(&initial),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let lobe_budget_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("production lobe sample budget"),
        contents: bytemuck::cast_slice(&GPU_LOBE_BUDGET),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("specular-AA image"),
        layout: &pipeline.get_bind_group_layout(0),
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
    let budget_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("production lobe sample budget"),
        layout: &pipeline.get_bind_group_layout(7),
        entries: &[wgpu::BindGroupEntry {
            binding: 9,
            resource: lobe_budget_buffer.as_entire_binding(),
        }],
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("specular-AA image readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_bind_group(7, &budget_bind_group, &[]);
        pass.dispatch_workgroups(PIXELS.div_ceil(WORKGROUP) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("map specular-AA image readback")
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("complete specular-AA image dispatch");
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("specular-AA image mapped data");
    let values: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    assert_eq!(
        &values[PIXELS..],
        &initial[PIXELS..],
        "image tail guard was modified"
    );

    let mut compared = 0usize;
    let mut worst = (0u32, 0usize, 0usize);
    let mut max_abs = 0.0f32;
    let mut nonzero_variance = 0usize;
    let mut softened = 0usize;
    for (index, pixel) in pixels.iter().enumerate() {
        let expected = cpu_pixel(pixel);
        let actual = values[index];
        let variance = SIGMA2
            * (Vec3::from_array(pixel.ddx_normal).dot(Vec3::from_array(pixel.ddx_normal))
                + Vec3::from_array(pixel.ddy_normal).dot(Vec3::from_array(pixel.ddy_normal)));
        if variance > 0.0 {
            nonzero_variance += 1;
        }
        let base_alpha = pixel.surface.perceptual_roughness * pixel.surface.perceptual_roughness;
        if actual[3] > base_alpha {
            softened += 1;
        }
        for channel in 0..4 {
            let distance = ulps(actual[channel], expected[channel]);
            let absolute = (actual[channel] - expected[channel]).abs();
            max_abs = max_abs.max(absolute);
            compared += 1;
            if distance > worst.0 {
                worst = (distance, index, channel);
            }
            assert!(
                distance <= MAX_ULPS,
                "pixel {index} channel {channel}: GPU={:.9e} CPU={:.9e}, {distance} ULP",
                actual[channel],
                expected[channel]
            );
        }
    }
    assert_eq!(
        nonzero_variance, PIXELS,
        "fixture did not exercise normal-footprint variance"
    );
    assert_eq!(
        softened, PIXELS,
        "specular-AA image did not widen any high-gloss pixel"
    );
    eprintln!(
        "specular-AA image: {WIDTH}x{HEIGHT}, {compared} channels, worst {} ULP at pixel {} channel {}, max abs {:.9e}",
        worst.0, worst.1, worst.2, max_abs
    );
    drop(mapped);
    readback.unmap();
}

#[test]
fn specular_aa_image_fixture_shape_is_stable() {
    let pixels = pixels();
    assert_eq!(pixels.len(), WIDTH * HEIGHT);
    assert!(pixels.iter().all(|pixel| {
        pixel.ddx_normal.into_iter().all(f32::is_finite)
            && pixel.ddy_normal.into_iter().all(f32::is_finite)
    }));
    assert_eq!(OUTPUT_ROWS, 1);
}

#[test]
fn specular_aa_tagged_variance_contract_covers_fallback_and_realtime_override() {
    use prism_render_shading::gi::specular_aa::normal_variance::{
        bake_normal_variance_texel, normal_variance_from_texel, SEMANTIC_NORMAL_VARIANCE,
    };

    let offline_texel =
        bake_normal_variance_texel([0.5, 0.5, 1.0], &[[0.8, 0.5, 1.0], [0.2, 0.5, 1.0]], 1.0)
            .expect("valid tagged normal footprint");
    let offline = normal_variance_from_texel(SEMANTIC_NORMAL_VARIANCE, offline_texel);
    assert!(offline > 0.0);

    // The CPU side of this image fixture carries the resolved variance in the
    // same SurfaceSample field used by the GPU BRDF. A valid flat realtime
    // footprint is zero and must replace, rather than add to, the offline value.
    let realtime_zero = prism_render_shading::gi::specular_aa::screen_space_variance(
        Vec3::ZERO,
        Vec3::ZERO,
        SIGMA2,
    );
    assert_eq!(realtime_zero, 0.0);
    assert_ne!(
        prism_render_shading::gi::specular_aa::normal_variance::filter_alpha_sq(
            0.08 * 0.08 * 0.08 * 0.08,
            offline,
            KAPPA_MAX,
        ),
        prism_render_shading::gi::specular_aa::normal_variance::filter_alpha_sq(
            0.08 * 0.08 * 0.08 * 0.08,
            realtime_zero,
            KAPPA_MAX,
        )
    );

    // This integration harness executes the production BRDF/specAA module, not
    // the complete visibility resolve. Keep the source gate explicit so the
    // test proves the resolve owns the fallback/override seam without claiming
    // bindless/VT sampling or geometry reconstruction was dispatched here.
    let resolve = include_str!("../../src/shaders/shading_resolve.wesl");
    assert!(resolve.contains("material_is_normal_semantic(record.semantic)"));
    assert!(resolve.contains("shading_resolve_normal_variance("));
    assert!(resolve.contains("let fallback = sampled.offline_normal_variance;"));
    assert!(resolve.contains("return spec_aa_screen_space_variance("));
}
