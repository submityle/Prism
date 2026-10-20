//! Real production principled BRDF -> storage readback -> public CPU reference.
//!
//! The one-ULP distance remains a diagnostic, while the real-device gate uses a
//! documented 32-ULP backend budget. Metal is allowed to contract FMA and use
//! device transcendental implementations, so a production BRDF cannot promise
//! bit-identical CPU results across backends. This exercises the complete
//! `SurfaceSample` input, both aggregate entry
//! points, hoisted GPU/CPU precomputation, and source-defined direct/transmission
//! channels. "Unlit" below means zero incident illuminance, not the separate
//! resolve-stage Unlit program (which does not call this BRDF).
//!
//! Run on a real device with:
//! `PRISM_REQUIRE_GPU=1 CARGO_INCREMENTAL=0 cargo test -p prism_render_scene
//! --test brdf_channels_gpu -- --nocapture`.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use prism_render_shading::{
    evaluate_principled_direct, evaluate_principled_direct_channels,
    evaluate_principled_direct_channels_with, evaluate_principled_direct_with, principled_pre,
    DirectLightSample, PrincipledDirectChannels, ShadingFrame, SurfaceSample,
};
use std::fmt::Write;
use wgpu::util::DeviceExt;

const WORKGROUP: usize = 64;
const OUTPUT_ROWS: usize = 11;
const GUARD: [f32; 4] = [-12345.0; 4];
const MAX_ULPS: u32 = 32;
const OUTPUT_NAMES: [&str; OUTPUT_ROWS] = [
    "radiance",
    "direct",
    "transmission",
    "legacy",
    "legacy_with",
    "dark.radiance",
    "dark.direct",
    "dark.transmission",
    "cpu_pre.radiance",
    "cpu_pre.direct",
    "cpu_pre.transmission",
];

// Single field list generates host scalar serialization AND the WESL adapter.
// Check against the production declaration so added/retyped GPU fields fail
// instead of silently receiving a zero/default value. This is a test transport,
// not a claimed Rust `SurfaceSample` storage ABI (`geometry_opacity` is CPU-only).
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

#[derive(Clone, Copy, Debug)]
struct Case {
    material: &'static str,
    geometry: &'static str,
    lighting: &'static str,
    surface: SurfaceSample,
    frame: ShadingFrame,
    light: DirectLightSample,
}

fn cases() -> Vec<Case> {
    let base = SurfaceSample {
        base_color: [0.25, 0.5, 0.75],
        perceptual_roughness: 0.5,
        ..Default::default()
    };
    let glass = SurfaceSample {
        transmission: 0.5,
        transmission_color: [0.5, 0.75, 1.0],
        thickness: 0.5,
        ..base
    };
    let coat = SurfaceSample {
        clearcoat: 0.5,
        clearcoat_roughness: 0.5,
        coat_ior: 1.5,
        ..base
    };
    let fuzz = SurfaceSample {
        sheen: 0.5,
        fuzz_roughness: 0.5,
        fuzz_color: [0.75, 0.5, 0.25],
        ..base
    };
    let sss = SurfaceSample {
        subsurface: 0.5,
        subsurface_color: [0.75, 0.5, 0.25],
        subsurface_radius: 0.5,
        subsurface_radius_scale: [1.0, 0.5, 0.25],
        scatter_anisotropy: 0.25,
        thickness: 0.25,
        ..base
    };
    let materials = [
        ("opaque", base),
        (
            "rough_diffuse",
            SurfaceSample {
                diffuse_roughness: 0.5,
                ..base
            },
        ),
        (
            "metal",
            SurfaceSample {
                metallic: 1.0,
                ..base
            },
        ),
        ("transmission_half", glass),
        (
            "transmission_full_thin",
            SurfaceSample {
                transmission: 1.0,
                thin_walled: 1.0,
                ..glass
            },
        ),
        (
            "transmission_dispersion",
            SurfaceSample {
                dispersion: 0.5,
                transmission_scatter: [0.125, 0.25, 0.5],
                transmission_scatter_anisotropy: 0.25,
                ..glass
            },
        ),
        ("coat", coat),
        (
            "coat_darkening_zero",
            SurfaceSample {
                coat_darkening: 0.0,
                ..coat
            },
        ),
        (
            "coat_darkening_half",
            SurfaceSample {
                coat_darkening: 0.5,
                ..coat
            },
        ),
        (
            "coat_darkening_full",
            SurfaceSample {
                coat_darkening: 1.0,
                ..coat
            },
        ),
        (
            "coat_weight_zero_darkening_full",
            SurfaceSample {
                clearcoat: 0.0,
                coat_darkening: 1.0,
                ..coat
            },
        ),
        (
            "coat_transmission_darkening_zero",
            SurfaceSample {
                transmission: 1.0,
                transmission_color: [0.5, 0.75, 1.0],
                thickness: 0.5,
                coat_darkening: 0.0,
                ..coat
            },
        ),
        (
            "coat_transmission_darkening_full",
            SurfaceSample {
                transmission: 1.0,
                transmission_color: [0.5, 0.75, 1.0],
                thickness: 0.5,
                coat_darkening: 1.0,
                ..coat
            },
        ),
        ("sheen", fuzz),
        (
            "anisotropy",
            SurfaceSample {
                anisotropy: 0.5,
                anisotropy_rotation: 0.0,
                ..base
            },
        ),
        ("subsurface", sss),
        (
            "mixed",
            SurfaceSample {
                transmission: 0.5,
                clearcoat: 0.5,
                clearcoat_roughness: 0.5,
                sheen: 0.5,
                fuzz_color: fuzz.fuzz_color,
                anisotropy: 0.5,
                ..sss
            },
        ),
        (
            "filtered_iridescent",
            SurfaceSample {
                normal_variance: 0.015625,
                iridescence: 0.5,
                iridescence_thickness_nm: 400.0,
                ..base
            },
        ),
    ];
    let mut result = Vec::new();
    // Dyadic material parameters and symmetric view/light frames make the
    // principal cases operation-order-compatible. Oblique normalized vectors
    // deliberately retain transcendental/backend sensitivity: never relax I7
    // when these expose a wider error.
    for (material, surface) in materials {
        for (geometry, view) in [
            ("normal", [0.0, 1.0, 0.0]),
            ("oblique", [0.6, 0.8, 0.0]),
            ("grazing", [0.8, 0.6, 0.0]),
        ] {
            let frame = ShadingFrame {
                normal: [0.0, 1.0, 0.0],
                view,
                tangent: [1.0, 0.0, 0.0],
                bitangent: [0.0, 0.0, -1.0],
            };
            for (lighting, direction, illuminance, visibility) in [
                ("front", view, [2.0, 1.0, 0.5], 1.0),
                ("partial_visibility", view, [2.0, 1.0, 0.5], 0.5),
                ("shadowed", view, [2.0, 1.0, 0.5], 0.0),
                ("visibility_below_zero", view, [2.0, 1.0, 0.5], -0.5),
                ("visibility_above_one", view, [2.0, 1.0, 0.5], 1.5),
                ("unlit", view, [0.0; 3], 1.0),
                ("backface", [0.0, -1.0, 0.0], [2.0, 1.0, 0.5], 1.0),
            ] {
                for emissive in [[0.0; 3], [0.125, 0.25, 0.5]] {
                    result.push(Case {
                        material,
                        geometry,
                        lighting,
                        surface: SurfaceSample {
                            emissive,
                            ..surface
                        },
                        frame,
                        light: DirectLightSample {
                            direction,
                            illuminance,
                            visibility,
                        },
                    });
                }
            }
        }
    }
    assert_eq!(result.len(), 756);
    assert_ne!(
        result.len() % WORKGROUP,
        0,
        "exercise final workgroup bounds"
    );
    result
}

fn cpu_rows(case: &Case) -> [[f32; 3]; OUTPUT_ROWS] {
    let Case {
        surface,
        frame,
        light,
        ..
    } = *case;
    let pre = principled_pre(&surface, &frame);
    let channels = evaluate_principled_direct_channels(surface, frame, light);
    assert_eq!(
        channels,
        evaluate_principled_direct_channels_with(surface, frame, light, &pre)
    );
    let dark = evaluate_principled_direct_channels(
        SurfaceSample {
            emissive: [0.0; 3],
            ..surface
        },
        frame,
        light,
    );
    assert_eq!(
        channels.direct, dark.direct,
        "CPU source direct excludes emission"
    );
    assert_eq!(
        channels.transmission, dark.transmission,
        "CPU source transmission excludes emission"
    );
    if light.visibility <= 0.0
        || light.illuminance == [0.0; 3]
        || (case.lighting == "backface" && surface.subsurface == 0.0)
    {
        assert_eq!(
            channels,
            PrincipledDirectChannels {
                radiance: surface.emissive,
                ..Default::default()
            }
        );
    }
    [
        channels.radiance,
        channels.direct,
        channels.transmission,
        evaluate_principled_direct(surface, frame, light),
        evaluate_principled_direct_with(surface, frame, light, &pre),
        dark.radiance,
        dark.direct,
        dark.transmission,
        channels.radiance,
        channels.direct,
        channels.transmission,
    ]
}

fn case_words(case: &Case) -> Vec<f32> {
    let mut words = surface_words(case.surface);
    for vector in [
        case.frame.normal,
        case.frame.view,
        case.frame.tangent,
        case.frame.bitangent,
        case.light.direction,
        case.light.illuminance,
    ] {
        words.extend(vector);
    }
    words.push(case.light.visibility);
    let pre = principled_pre(&case.surface, &case.frame);
    words.extend([pre.e_ggx_view, pre.e_ggx_avg]);
    words.extend(pre.e_dielectric);
    words.push(pre.coat_lobe_albedo);
    words
}

fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("production BRDF is WESL"),
    }
}

fn shader_id(index: usize) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(0x5052_4953_4d5f_4252_4446_4348_0000_0000 + index as u128),
    }
}

fn compile_shader(count: usize) -> String {
    let production = include_str!("../../src/shaders/brdf.wesl");
    let body = production
        .split_once("struct SurfaceSample {")
        .expect("production SurfaceSample")
        .1
        .split_once('}')
        .unwrap()
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
        "full production surface transport drifted"
    );

    let surface_size: usize = SURFACE_FIELDS.iter().map(|(_, width)| width).sum();
    let stride = surface_size + 25;
    let mut wrapper = format!(
        r#"
import prism_render_scene::shaders::brdf::{{
    SurfaceSample, ShadingFrame, DirectLightSample, PrincipledPre,
    principled_pre, principled_direct, principled_direct_with,
    principled_direct_channels_with,
}};
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4<f32>>;
fn load3(offset: u32) -> vec3<f32> {{
    return vec3<f32>(inputs[offset], inputs[offset + 1u], inputs[offset + 2u]);
}}
@compute @workgroup_size(64)
fn brdf_channels_reference(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x >= {count}u {{ return; }}
    let base = id.x * {stride}u;
    var surface: SurfaceSample;
"#
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
    let pre = principled_pre(surface, frame);
    let channels = principled_direct_channels_with(surface, frame, light, pre);
    let legacy = principled_direct(surface, frame, light);
    let legacy_with = principled_direct_with(surface, frame, light, pre);
    let cpu_pre = PrincipledPre(inputs[f + 19u], inputs[f + 20u], load3(f + 21u), inputs[f + 24u]);
    let supplied = principled_direct_channels_with(surface, frame, light, cpu_pre);
    surface.emissive = vec3<f32>(0.0);
    let dark = principled_direct_channels_with(surface, frame, light, pre);
    let out = id.x * 11u;
    outputs[out] = vec4<f32>(channels.radiance, 1.0);
    outputs[out + 1u] = vec4<f32>(channels.direct, 1.0);
    outputs[out + 2u] = vec4<f32>(channels.transmission, 1.0);
    outputs[out + 3u] = vec4<f32>(legacy, 1.0);
    outputs[out + 4u] = vec4<f32>(legacy_with, 1.0);
    outputs[out + 5u] = vec4<f32>(dark.radiance, 1.0);
    outputs[out + 6u] = vec4<f32>(dark.direct, 1.0);
    outputs[out + 7u] = vec4<f32>(dark.transmission, 1.0);
    outputs[out + 8u] = vec4<f32>(supplied.radiance, 1.0);
    outputs[out + 9u] = vec4<f32>(supplied.direct, 1.0);
    outputs[out + 10u] = vec4<f32>(supplied.transmission, 1.0);
}}
"#
    )
    .unwrap();
    let wrapper: &'static str = Box::leak(wrapper.into_boxed_str());
    let mut cache = ShaderCache::new((), keep_wgsl);
    let modules = [
        ("lighting", include_str!("../../src/shaders/lighting.wesl")),
        ("tangent", include_str!("../../src/shaders/tangent.wesl")),
        (
            "specular_aa",
            include_str!("../../src/shaders/specular_aa.wesl"),
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
        ("brdf_channels_reference", wrapper),
    ];
    for (i, (name, source)) in modules.iter().enumerate() {
        cache.set_shader(
            shader_id(i),
            Shader::from_wesl(
                *source,
                format!("embedded://prism_render_scene/shaders/{name}.wesl"),
            ),
        );
    }
    (*cache
        .get(0, shader_id(modules.len() - 1), &[])
        .unwrap_or_else(|error| panic!("production BRDF WESL import/link failed: {error}")))
    .clone()
}

fn unavailable(message: impl std::fmt::Display) -> Option<(wgpu::Device, wgpu::Queue)> {
    assert!(
        std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU set: required real BRDF GPU unavailable: {message}"
    );
    eprintln!("BRDF channels GPU SKIP: {message}");
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
    let info = adapter.get_info();
    eprintln!("BRDF channels GPU adapter: {info:?}");
    if info.device_type == wgpu::DeviceType::Cpu {
        return unavailable("software/CPU adapter is not real GPU parity");
    }
    let limits = adapter.limits();
    if limits.max_compute_invocations_per_workgroup < WORKGROUP as u32
        || limits.max_compute_workgroup_size_x < WORKGROUP as u32
        || limits.max_storage_buffers_per_shader_stage < 2
        || u64::from(limits.max_storage_buffer_binding_size) < bytes
        || limits.max_buffer_size < bytes
    {
        return unavailable(format!(
            "needs x64 and two {bytes}-byte storage bindings; {limits:?}"
        ));
    }
    match block_on(adapter.request_device(&Default::default())) {
        Ok(device_queue) => Some(device_queue),
        Err(error) => unavailable(error),
    }
}

/// Ordered IEEE-754 distance with the two signed zeros treated as equal.
/// Nonfinite results are failures even if both backends produce the same NaN.
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
fn ulp_metric_and_full_surface_fixture_contract() {
    assert_eq!(ulps(-0.0, 0.0), 0);
    assert_eq!(ulps(0.0, f32::from_bits(1)), 1);
    assert_eq!(ulps(-f32::from_bits(1), f32::from_bits(1)), 2);
    assert_eq!(ulps(1.0, f32::from_bits(1.0_f32.to_bits() + 1)), 1);
    assert_eq!(ulps(f32::NAN, f32::NAN), u32::MAX);
    let cases = cases();
    let stride = SURFACE_FIELDS.iter().map(|(_, width)| width).sum::<usize>() + 25;
    for case in &cases {
        assert_eq!(case_words(case).len(), stride);
        assert!(
            cpu_rows(case)
                .iter()
                .flatten()
                .all(|value| value.is_finite() && *value >= 0.0),
            "{case:?}"
        );
    }
    assert!(
        cases.iter().any(|case| case.material == "subsurface"
            && case.lighting == "backface"
            && cpu_rows(case)[1].iter().any(|value| *value > 0.0)),
        "exercise real subsurface backlight, not only early-out"
    );
    assert!(
        cases
            .iter()
            .any(|case| cpu_rows(case)[2].iter().all(|value| *value > 0.0)),
        "exercise nonzero RGB transmission"
    );
}

#[test]
fn real_gpu_principled_channels_match_cpu_within_backend_budget() {
    let cases = cases();
    let words: Vec<_> = cases.iter().flat_map(case_words).collect();
    let initial = vec![GUARD; (cases.len() + WORKGROUP) * OUTPUT_ROWS];
    let output_bytes = std::mem::size_of_val(initial.as_slice()) as u64;
    let input_bytes = std::mem::size_of_val(words.as_slice()) as u64;
    let Some((device, queue)) = device(output_bytes.max(input_bytes)) else {
        return;
    };
    let source = compile_shader(cases.len());
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production BRDF channel reference"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("BRDF channels reference"),
        layout: None,
        module: &module,
        entry_point: Some("brdf_channels_reference"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("full surface, frame, light and CPU pre"),
        contents: bytemuck::cast_slice(&words),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BRDF RGB planes and dispatch guard"),
        contents: bytemuck::cast_slice(&initial),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BRDF channels reference"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BRDF channel readback"),
        size: output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(cases.len().div_ceil(WORKGROUP) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("map real BRDF GPU readback")
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("complete BRDF GPU dispatch");
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .expect("BRDF readback data");
    let values: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    assert_eq!(
        &values[cases.len() * OUTPUT_ROWS..],
        &initial[cases.len() * OUTPUT_ROWS..],
        "bounded last workgroup leaves guards intact"
    );
    let mut mismatches = Vec::new();
    let mut worst = (0, String::new());
    let mut compared = 0;
    for (index, case) in cases.iter().enumerate() {
        let expected = cpu_rows(case);
        let rows = &values[index * OUTPUT_ROWS..(index + 1) * OUTPUT_ROWS];
        assert_eq!(rows[1], rows[6], "GPU direct excludes emission: {case:?}");
        assert_eq!(
            rows[2], rows[7],
            "GPU transmission excludes emission: {case:?}"
        );
        for (row, (actual, expected)) in rows.iter().zip(expected).enumerate() {
            assert_eq!(
                actual[3], 1.0,
                "case {index} output row {row} was not written"
            );
            for channel in 0..3 {
                let distance = ulps(actual[channel], expected[channel]);
                compared += 1;
                if distance > worst.0 {
                    worst = (
                        distance,
                        format!(
                            "case {index} {}/{}/{} emissive={:?} {}[{channel}]: GPU={:.9e} (0x{:08x}), CPU={:.9e} (0x{:08x}), {distance} ULP",
                            case.material,
                            case.geometry,
                            case.lighting,
                            case.surface.emissive,
                            OUTPUT_NAMES[row],
                            actual[channel],
                            actual[channel].to_bits(),
                            expected[channel],
                            expected[channel].to_bits(),
                        ),
                    );
                }
                if distance > MAX_ULPS {
                    let message = format!("case {index} {}/{}/{} emissive={:?} {}[{channel}]: GPU={:.9e} (0x{:08x}), CPU={:.9e} (0x{:08x}), {distance} ULP", case.material, case.geometry, case.lighting, case.surface.emissive, OUTPUT_NAMES[row], actual[channel], actual[channel].to_bits(), expected[channel], expected[channel].to_bits());
                    mismatches.push(message);
                }
            }
        }
    }
    eprintln!("BRDF channels: {} full-surface cases, {compared} RGB components, {} exceed {MAX_ULPS} ULP; worst {} ULP", cases.len(), mismatches.len(), worst.0);
    for message in mismatches.iter().take(16) {
        eprintln!("{message}");
    }
    assert!(
        mismatches.is_empty(),
        "BRDF CPU/GPU parity exceeded the documented {MAX_ULPS}-ULP backend budget for {}/{} components; worst: {}",
        mismatches.len(),
        compared,
        worst.1
    );
    drop(mapped);
    readback.unmap();
}
