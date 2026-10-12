//! **Real-device parity for the runtime dynamic channels (§17.1).**
//!
//! The four header words (`dynamic_{wetness,snow,damage,dissolve}`) remap the
//! frozen über surface analytically, on both lines. The parity risk is not the
//! idea but the arithmetic: `mix` spelled one way and not the other, a different
//! channel order, or a dissolve mask whose hash/quantisation drifts by one cell.
//! This test assembles a compute kernel from the **shipping WESL** (`dynamic.wesl`
//! plus `brdf.wesl`'s real `SurfaceSample` and `lobe_sampling.wesl`'s hash chain)
//! and diffs every modulated field against the CPU golden over a grid.
//!
//! On a host without a usable adapter the test skips — pass `PRISM_REQUIRE_GPU=1`
//! to make a missing adapter fail instead.

use futures_lite::future::block_on;
use prism_render_material::DynamicChannels;
use prism_render_shading::{apply_dynamic_channels, dissolve_coverage, SurfaceSample};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineLayoutDescriptor, PollType,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

const DYNAMIC_WESL: &str = include_str!("../../prism_render_scene/src/shaders/dynamic.wesl");
const BRDF_WESL: &str = include_str!("../../prism_render_scene/src/shaders/brdf.wesl");
const DYNAMIC_MASK_WESL: &str = include_str!("../../prism_render_scene/src/shaders/dynamic_mask.wesl");
const LOBE_SAMPLING_WESL: &str =
    include_str!("../../prism_render_scene/src/shaders/lobe_sampling.wesl");

fn extract_fn(source: &str, label: &str, name: &str) -> String {
    let needle = format!("fn {name}(");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("{name} missing from {label}"));
    let open = source[start..].find('{').expect("fn body") + start;
    let mut depth = 0usize;
    for (i, ch) in source[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..open + i + 1].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated {name}");
}

fn extract_struct(source: &str, label: &str, name: &str) -> String {
    let needle = format!("struct {name} {{");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("{name} missing from {label}"));
    let open = start + needle.len() - 1;
    let mut depth = 0usize;
    for (i, ch) in source[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..open + i + 1].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated {name}");
}

/// One grid sample: the surface's authored fields, the four channels, and the
/// shade-time cell the procedural dissolve mask hashes.
struct Sample {
    surface: SurfaceSample,
    channels: DynamicChannels,
    cell: [f32; 2],
}

/// A representative surface: metal, a coat, sheen, emission (so the dissolve rim
/// has something to tint) and a non-trivial roughness.
fn base_surface() -> SurfaceSample {
    SurfaceSample {
        base_color: [0.55, 0.42, 0.31],
        diffuse_roughness: 0.65,
        metallic: 0.8,
        perceptual_roughness: 0.35,
        specular_weight: 0.7,
        specular_ior: 1.45,
        sheen: 0.4,
        clearcoat: 0.8,
        emissive: [0.25, 0.12, 0.05],
        geometry_opacity: 1.0,
        ..SurfaceSample::default()
    }
}

fn samples() -> Vec<Sample> {
    let mut out = Vec::new();
    for channels in [
        DynamicChannels::NONE,
        DynamicChannels {
            wetness: 1.0,
            ..DynamicChannels::NONE
        },
        DynamicChannels {
            snow: 1.0,
            ..DynamicChannels::NONE
        },
        DynamicChannels {
            damage: 1.0,
            ..DynamicChannels::NONE
        },
        DynamicChannels {
            wetness: 0.35,
            snow: 0.5,
            damage: 0.2,
            dissolve: 0.0,
        },
        DynamicChannels {
            wetness: 0.2,
            snow: 0.15,
            damage: 0.7,
            dissolve: 0.42,
        },
        DynamicChannels {
            dissolve: 0.9,
            ..DynamicChannels::NONE
        },
    ] {
        for cell in [[0.0_f32, 0.0], [0.3, 0.5], [12.25, -4.75], [0.999, 0.001]] {
            out.push(Sample {
                surface: base_surface(),
                channels,
                cell,
            });
        }
    }
    // A diffuse, coat-free surface as a second shape of input.
    for channels in [
        DynamicChannels {
            damage: 0.6,
            ..DynamicChannels::NONE
        },
        DynamicChannels {
            snow: 0.4,
            damage: 0.4,
            dissolve: 0.25,
            ..DynamicChannels::NONE
        },
    ] {
        let mut surface = base_surface();
        surface.metallic = 0.0;
        surface.clearcoat = 0.0;
        surface.sheen = 0.0;
        surface.emissive = [0.0; 3];
        for cell in [[0.5_f32, 0.25], [3.75, 8.125]] {
            out.push(Sample {
                surface,
                channels,
                cell,
            });
        }
    }
    out
}

/// The modulated fields the kernel writes, in this order.
const FIELDS: usize = 13;

#[test]
fn dynamic_channels_match_on_device() {
    let samples = samples();
    let count = samples.len();

    let mut shader = String::from(
        "@group(0) @binding(0) var<storage, read_write> out_fields: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> in_surface: array<vec4<f32>>;\n\
         @group(0) @binding(2) var<storage, read> in_channels: array<vec4<f32>>;\n\
         @group(0) @binding(3) var<storage, read> in_cell: array<vec2<f32>>;\n\
         @group(0) @binding(4) var<storage, read_write> out_coverage: array<f32>;\n",
    );
    shader.push_str(&extract_struct(BRDF_WESL, "brdf.wesl", "SurfaceSample"));
    shader.push('\n');
    for name in [
        "ls_hash_u32",
        "ls_hash_combine",
        "ls_uniform_from_hash",
    ] {
        shader.push_str(&extract_fn(LOBE_SAMPLING_WESL, "lobe_sampling.wesl", name));
        shader.push('\n');
    }
    for name in ["dyn_quantize_cell", "dyn_dissolve_mask", "dyn_dissolve_coverage"] {
        shader.push_str(&extract_fn(DYNAMIC_MASK_WESL, "dynamic_mask.wesl", name));
        shader.push('\n');
    }
    for name in ["DISSOLVE_MASK_CELLS", "DISSOLVE_EDGE_WIDTH"] {
        let start = DYNAMIC_MASK_WESL
            .find(&format!("const {name}:"))
            .unwrap_or_else(|| panic!("{name} missing from dynamic_mask.wesl"));
        let end = DYNAMIC_MASK_WESL[start..].find(';').expect("const") + start + 1;
        shader.push_str(&DYNAMIC_MASK_WESL[start..end]);
        shader.push('\n');
    }
    shader.push_str(&extract_fn(DYNAMIC_WESL, "dynamic.wesl", "dyn_mix"));
    shader.push('\n');
    // The constants the modulation reads, taken verbatim.
    for name in [
        "WET_BASE_DARKEN",
        "WET_ROUGHNESS_SCALE",
        "WATER_IOR",
        "SNOW_ALBEDO",
        "SNOW_ROUGHNESS",
        "SNOW_DIFFUSE_ROUGHNESS",
        "SNOW_SPECULAR_WEIGHT",
        "SNOW_SHEEN",
        "DAMAGE_BASE_DARKEN",
        "DAMAGE_METAL_LOSS",
        "DAMAGE_ROUGHEN",
        "DISSOLVE_EDGE_GAIN",
    ] {
        let start = DYNAMIC_WESL
            .find(&format!("const {name}:"))
            .unwrap_or_else(|| panic!("{name} missing from dynamic.wesl"));
        let end = DYNAMIC_WESL[start..].find(';').expect("const") + start + 1;
        shader.push_str(&DYNAMIC_WESL[start..end]);
        shader.push('\n');
    }
    shader.push_str(&extract_fn(
        DYNAMIC_WESL,
        "dynamic.wesl",
        "apply_dynamic_channels",
    ));
    shader.push_str(
        "\n@compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
             let idx = gid.x;\n\
             if (idx >= arrayLength(&in_cell)) { return; }\n\
             let g0 = in_surface[idx * 4u];\n\
             let g1 = in_surface[idx * 4u + 1u];\n\
             let g2 = in_surface[idx * 4u + 2u];\n\
             let g3 = in_surface[idx * 4u + 3u];\n\
             var surface: SurfaceSample;\n\
             surface.base_color = g0.xyz;\n\
             surface.diffuse_roughness = g0.w;\n\
             surface.metallic = g1.x;\n\
             surface.perceptual_roughness = g1.y;\n\
             surface.specular_weight = g1.z;\n\
             surface.specular_ior = g1.w;\n\
             surface.sheen = g2.x;\n\
             surface.clearcoat = g2.y;\n\
             surface.emissive = g3.xyz;\n\
             let modulated = apply_dynamic_channels(surface, in_channels[idx], in_cell[idx]);\n\
             let base = idx * FIELDS;\n\
             out_fields[base + 0u] = modulated.base_color.x;\n\
             out_fields[base + 1u] = modulated.base_color.y;\n\
             out_fields[base + 2u] = modulated.base_color.z;\n\
             out_fields[base + 3u] = modulated.diffuse_roughness;\n\
             out_fields[base + 4u] = modulated.metallic;\n\
             out_fields[base + 5u] = modulated.perceptual_roughness;\n\
             out_fields[base + 6u] = modulated.specular_weight;\n\
             out_fields[base + 7u] = modulated.specular_ior;\n\
             out_fields[base + 8u] = modulated.sheen;\n\
             out_fields[base + 9u] = modulated.clearcoat;\n\
             out_fields[base + 10u] = modulated.emissive.x;\n\
             out_fields[base + 11u] = modulated.emissive.y;\n\
             out_fields[base + 12u] = modulated.emissive.z;\n\
             out_coverage[idx] = dyn_dissolve_coverage(in_channels[idx].w, in_cell[idx]);\n\
         }\n",
    );
    shader = shader.replace("FIELDS", &FIELDS.to_string());

    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let Some(adapter) = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()
    else {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        eprintln!("skipping dynamic-channel GPU parity: no usable adapter");
        return;
    };
    let (device, queue) =
        block_on(adapter.request_device(&DeviceDescriptor::default())).expect("device");
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("prism_dynamic_channels_parity"),
        source: ShaderSource::Wgsl(shader.into()),
    });

    let mut surface_bytes = Vec::new();
    let mut channel_bytes = Vec::new();
    let mut cell_bytes = Vec::new();
    for sample in &samples {
        let s = sample.surface;
        for value in [
            s.base_color[0], s.base_color[1], s.base_color[2], s.diffuse_roughness,
        ] {
            surface_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [
            s.metallic,
            s.perceptual_roughness,
            s.specular_weight,
            s.specular_ior,
        ] {
            surface_bytes.extend_from_slice(&value.to_le_bytes());
        }
        // The remaining fields the modulation touches, one `vec4` group each so
        // the kernel's surface is exactly the CPU's.
        for value in [s.sheen, s.clearcoat, 0.0, 0.0] {
            surface_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [s.emissive[0], s.emissive[1], s.emissive[2], 0.0] {
            surface_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [
            sample.channels.wetness,
            sample.channels.snow,
            sample.channels.damage,
            sample.channels.dissolve,
        ] {
            channel_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in sample.cell {
            cell_bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    let make_buffer = |label: &str, bytes: &[u8]| {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytes);
        buffer
    };
    let in_surface = make_buffer("in_surface", &surface_bytes);
    let in_channels = make_buffer("in_channels", &channel_bytes);
    let in_cell = make_buffer("in_cell", &cell_bytes);

    let out_bytes = (count * FIELDS * 4) as u64;
    let out_fields = device.create_buffer(&BufferDescriptor {
        label: Some("out_fields"),
        size: out_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let coverage_bytes = (count * 4) as u64;
    let out_coverage = device.create_buffer(&BufferDescriptor {
        label: Some("out_coverage"),
        size: coverage_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: out_bytes + coverage_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let storage = |binding: u32, read_only: bool| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage(0, false),
            storage(1, true),
            storage(2, true),
            storage(3, true),
            storage(4, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("dynamic_channels_parity"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: out_fields.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: in_surface.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: in_channels.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: in_cell.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: out_coverage.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_fields, 0, &readback, 0, out_bytes);
    encoder.copy_buffer_to_buffer(&out_coverage, 0, &readback, out_bytes, coverage_bytes);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(PollType::wait_indefinitely());
    rx.recv().expect("map callback").expect("map");
    let data = slice
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let floats: Vec<f32> = data[..count * FIELDS * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let coverage: Vec<f32> = data[count * FIELDS * 4..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for (index, sample) in samples.iter().enumerate() {
        let mut want = sample.surface;
        apply_dynamic_channels(&mut want, sample.channels, sample.cell);
        let got = &floats[index * FIELDS..(index + 1) * FIELDS];
        let expected = [
            want.base_color[0],
            want.base_color[1],
            want.base_color[2],
            want.diffuse_roughness,
            want.metallic,
            want.perceptual_roughness,
            want.specular_weight,
            want.specular_ior,
            want.sheen,
            want.clearcoat,
            want.emissive[0],
            want.emissive[1],
            want.emissive[2],
        ];
        // The dissolve *coverage* half of §17.1 lives where alpha is consumed,
        // so it is compared through its own entry point — the same value the
        // raster passes multiply into the material's alpha.
        let want_coverage = dissolve_coverage(sample.channels.dissolve, sample.cell);
        assert!(
            (coverage[index] - want_coverage).abs() <= 1.0e-6,
            "case {index} coverage: gpu {} vs cpu {want_coverage}",
            coverage[index]
        );
        for (field, (got, want)) in got.iter().zip(expected).enumerate() {
            // Closed-form `a*(1-t) + b*t` arithmetic: contraction can move this
            // by one ULP, so the bound is here to catch a real formula or order
            // drift, not to paper over rounding.
            let tolerance = 1.0e-6 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "case {index} field {field}: gpu {got} vs cpu {want}"
            );
        }
    }
}
