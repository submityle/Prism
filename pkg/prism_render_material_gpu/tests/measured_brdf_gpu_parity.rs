//! **Real-device parity for the measured (tabulated) BRDF lobe (§17.6).**
//!
//! The lobe is a table lookup: Rusinkiewicz coordinates, a grid mapping, eight
//! bracketing texels fetched from a material texture, and a trilinear mix. The
//! risk is not the idea but the *addressing* — the atlas layout (`width =
//! theta_h`, `height = theta_d * phi_d`, texel `(i, j*phi_d + k)` = cell
//! `(i, j, k)`), the cell-centre remap, and the blend order. This test builds a
//! real atlas of a synthetic measured table, runs the **shipping WESL**
//! (`measured_lobe.wesl` + `measured_brdf.wesl`, plus `brdf.wesl`'s real
//! `SurfaceSample`/`ShadingFrame`/`DirectLightSample`) in a compute kernel, and
//! diffs every sample against the CPU golden that shares that kernel.
//!
//! On a host without a usable adapter the test skips — pass `PRISM_REQUIRE_GPU=1`
//! to make a missing adapter fail instead.

use futures_lite::future::block_on;
use prism_render_shading::{
    evaluate_measured_direct, DirectLightSample, MeasuredBrdf, MeasuredBrdfDims, ShadingFrame,
    SyntheticBrdfParams,
};
use std::num::NonZeroU32;
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    DeviceDescriptor, Extent3d, Instance, InstanceDescriptor, InstanceFlags, MapMode,
    PipelineLayoutDescriptor, PollType, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

const MEASURED_LOBE_WESL: &str =
    include_str!("../../prism_render_scene/src/shaders/measured_lobe.wesl");
const MEASURED_BRDF_WESL: &str =
    include_str!("../../prism_render_scene/src/shaders/measured_brdf.wesl");
const BRDF_WESL: &str = include_str!("../../prism_render_scene/src/shaders/brdf.wesl");

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

fn extract_const(source: &str, label: &str, name: &str) -> String {
    let start = source
        .find(&format!("const {name}:"))
        .unwrap_or_else(|| panic!("{name} missing from {label}"));
    let end = source[start..].find(';').expect("const terminator") + start + 1;
    source[start..end].to_string()
}

/// The table under test. Small on purpose: the arithmetic is resolution
/// independent, and a small atlas keeps the texture trivially resolvable.
const DIMS: MeasuredBrdfDims = MeasuredBrdfDims {
    theta_h: 8,
    theta_d: 8,
    phi_d: 8,
};

fn table() -> MeasuredBrdf {
    MeasuredBrdf::from_analytic(
        DIMS,
        SyntheticBrdfParams {
            base_color: [0.75, 0.45, 0.2],
            roughness: 0.35,
            metallic: 0.4,
            reflectance: 0.5,
        },
    )
}

/// Pack the table into the documented atlas: `(i, j * phi_d + k)`.
fn atlas_rgba(table: &MeasuredBrdf) -> Vec<f32> {
    let cells = table.cells();
    let mut out = vec![0.0_f32; DIMS.theta_h * DIMS.theta_d * DIMS.phi_d * 4];
    for i in 0..DIMS.theta_h {
        for j in 0..DIMS.theta_d {
            for k in 0..DIMS.phi_d {
                let cell = cells[(i * DIMS.theta_d + j) * DIMS.phi_d + k];
                let texel = (i + (j * DIMS.phi_d + k) * DIMS.theta_h) * 4;
                out[texel] = cell[0];
                out[texel + 1] = cell[1];
                out[texel + 2] = cell[2];
                out[texel + 3] = 1.0;
            }
        }
    }
    out
}

/// A direction on the upper hemisphere from two angles.
fn direction(theta: f32, phi: f32) -> [f32; 3] {
    let (s, c) = theta.sin_cos();
    [s * phi.cos(), s * phi.sin(), c]
}

fn frame() -> ShadingFrame {
    ShadingFrame {
        normal: [0.0, 1.0, 0.0],
        view: [0.0, 1.0, 0.0],
        tangent: [1.0, 0.0, 0.0],
        bitangent: [0.0, 0.0, -1.0],
    }
}

#[test]
fn measured_lobe_matches_on_device() {
    let table = table();
    let frame = frame();
    // A grid of light directions; the view is the frame's normal, so the
    // half-vector, the difference elevation and the folded azimuth all move.
    let mut lights = Vec::new();
    for theta_i in 0..=6 {
        for phi_i in 0..8 {
            let theta = theta_i as f32 / 6.0 * 1.4;
            let phi = phi_i as f32 / 8.0 * core::f32::consts::TAU;
            lights.push(DirectLightSample {
                direction: direction(theta, phi),
                illuminance: [3.0, 2.5, 2.0],
                visibility: 1.0,
            });
        }
    }
    let count = lights.len();

    let mut shader = String::from(
        // The shipping shaders declare this (`material_sample.wesl`), and the
        // lobe reads the material texture heap exactly as they do.
        "enable wgpu_binding_array;\n\
         @group(0) @binding(0) var material_texture_heap: binding_array<texture_2d<f32>>;\n\
         @group(0) @binding(1) var<storage, read_write> out_color: array<vec4<f32>>;\n\
         @group(0) @binding(2) var<storage, read> in_light: array<vec4<f32>>;\n",
    );
    shader.push_str(&extract_struct(BRDF_WESL, "brdf.wesl", "ShadingFrame"));
    shader.push('\n');
    shader.push_str(&extract_struct(BRDF_WESL, "brdf.wesl", "DirectLightSample"));
    shader.push('\n');
    for name in [
        "MB_PI",
        "MB_FRAC_PI_2",
        "MB_TAU",
        "MB_MIN_COS",
        "MB_MIN_LEN_SQ",
        "MB_MIN_ROUGHNESS",
        "MB_MIN_POSITIVE",
    ] {
        shader.push_str(&extract_const(MEASURED_BRDF_WESL, "measured_brdf.wesl", name));
        shader.push('\n');
    }
    for name in [
        "mb_is_finite",
        "mb_clamp01",
        "mb_pow5",
        "mb_lerp3",
        "mb_normalize_or",
        "mb_normalize_or_zero",
        "mb_half_vector",
        "mb_rotate_about_z",
        "mb_rotate_about_y",
        "mb_wrap_tau",
        "mb_fold_phi_d",
        "mb_theta_h_to_unit",
        "mb_unit_to_theta_h",
        "mb_directions_to_rusinkiewicz",
        "mb_unit_to_grid",
        "mb_grid_lower",
        "mb_grid_frac",
        "mb_grid_coord_theta_h",
        "mb_grid_coord_theta_d",
        "mb_grid_coord_phi_d",
        "mb_trilinear",
    ] {
        shader.push_str(&extract_fn(MEASURED_BRDF_WESL, "measured_brdf.wesl", name));
        shader.push('\n');
    }
    shader.push_str(&extract_const(
        MEASURED_LOBE_WESL,
        "measured_lobe.wesl",
        "SEMANTIC_MEASURED_BRDF",
    ));
    shader.push('\n');
    for name in ["measured_normalize_or", "measured_texel", "measured_direct"] {
        shader.push_str(&extract_fn(MEASURED_LOBE_WESL, "measured_lobe.wesl", name));
        shader.push('\n');
    }
    shader.push_str(
        "\n@compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
             let idx = gid.x;\n\
             if (idx >= arrayLength(&in_light)) { return; }\n\
             var frame: ShadingFrame;\n\
             frame.normal = vec3<f32>(0.0, 1.0, 0.0);\n\
             frame.view = vec3<f32>(0.0, 1.0, 0.0);\n\
             frame.tangent = vec3<f32>(1.0, 0.0, 0.0);\n\
             frame.bitangent = vec3<f32>(0.0, 0.0, -1.0);\n\
             var light: DirectLightSample;\n\
             light.direction = in_light[idx].xyz;\n\
             light.illuminance = vec3<f32>(3.0, 2.5, 2.0);\n\
             light.visibility = 1.0;\n\
             let color = measured_direct(0u, 8.0, 1.75, 0.8, frame, light);\n\
             out_color[idx] = vec4<f32>(color, 1.0);\n\
         }\n",
    );

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
        eprintln!("skipping measured-BRDF GPU parity: no usable adapter");
        return;
    };
    // The lobe reads the material texture heap, which is a `binding_array`, so
    // the device needs the corresponding feature — exactly what the shipping
    // renderer requests for the material bind group.
    let wanted = wgpu::Features::TEXTURE_BINDING_ARRAY
        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    let features = adapter.features() & wanted;
    assert!(
        features == wanted || std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU is set but the adapter cannot index a texture binding array"
    );
    if features != wanted {
        eprintln!("skipping measured-BRDF GPU parity: adapter has no bindless texture arrays");
        return;
    }
    // One element each: the lobe only ever addresses the material's own table.
    let mut limits = wgpu::Limits::default();
    limits.max_binding_array_elements_per_shader_stage = 1;
    limits.max_binding_array_sampler_elements_per_shader_stage = 1;
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        required_features: features,
        required_limits: limits,
        ..Default::default()
    }))
    .expect("device");
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("prism_measured_brdf_parity"),
        source: ShaderSource::Wgsl(shader.into()),
    });

    // The atlas, as a real texture: width = theta_h, height = theta_d * phi_d.
    let pixels = atlas_rgba(&table);
    let width = DIMS.theta_h as u32;
    let height = (DIMS.theta_d * DIMS.phi_d) as u32;
    let atlas = device.create_texture(&TextureDescriptor {
        label: Some("measured_atlas"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &atlas,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(height),
        },
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let atlas_view = atlas.create_view(&TextureViewDescriptor::default());

    let light_bytes: Vec<u8> = lights
        .iter()
        .flat_map(|light| {
            [
                light.direction[0], light.direction[1], light.direction[2], 0.0,
            ]
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    let in_light = device.create_buffer(&BufferDescriptor {
        label: Some("in_light"),
        size: light_bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&in_light, 0, &light_bytes);
    let out_bytes = (count * 16) as u64;
    let out_color = device.create_buffer(&BufferDescriptor {
        label: Some("out_color"),
        size: out_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: out_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZeroU32::new(1),
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
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("measured_brdf_parity"),
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
                resource: BindingResource::TextureViewArray(&[&atlas_view]),
            },
            BindGroupEntry {
                binding: 1,
                resource: out_color.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: in_light.as_entire_binding(),
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
    encoder.copy_buffer_to_buffer(&out_color, 0, &readback, 0, out_bytes);
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
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut non_zero = 0usize;
    for (index, light) in lights.iter().enumerate() {
        let want = evaluate_measured_direct(&table, 1.75, 0.8, &frame, light);
        let got = &floats[index * 4..index * 4 + 3];
        if want.iter().any(|value| *value > 1.0e-4) {
            non_zero += 1;
        }
        for channel in 0..3 {
            let tolerance = 1.0e-4 * want[channel].abs().max(1.0);
            assert!(
                (got[channel] - want[channel]).abs() <= tolerance,
                "sample {index} channel {channel}: gpu {} vs cpu {} (dir {:?})",
                got[channel],
                want[channel],
                light.direction
            );
        }
    }
    // A gate over an all-zero response would pass vacuously, so assert the grid
    // actually exercises the lobe.
    assert!(non_zero > count / 4, "only {non_zero}/{count} samples lit");
}
