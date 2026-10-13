//! **Real-device parity for the procedural glint lobe (§17.6).**
//!
//! The discrete (counting) microfacet estimator is what makes countable flake
//! (metallic flake, snow, sand, glitter) twinkle instead of forming a smooth
//! statistical highlight, and it is *unbiased* in expectation — its ensemble
//! mean is the smooth GGX NDF, so it re-shapes the specular lobe without
//! changing its energy. That contract only holds if the CPU golden in
//! `prism_render_shading` and the WESL twin agree on the whole chain: the seed
//! derivation (space cell x half-vector slope quantisation), the Poisson draw,
//! the acceptance-bin solid angle and the blend.
//!
//! This test assembles a compute kernel from the **shipping WESL text**
//! (`glint.wesl` plus `brdf.wesl`'s seed derivation) and diffs it against the
//! CPU golden over a grid. The seed must match exactly (it is a hash of integer
//! bins); the estimator to float tolerance. On a host without a usable adapter
//! the test skips — pass `PRISM_REQUIRE_GPU=1` to make a missing adapter fail.

use futures_lite::future::block_on;
use prism_render_shading::{
    glint_bin_solid_angle, glint_ggx_ndf, glint_microfacet_count, glint_ndf_blended,
    glint_ndf_blended_from_density, glint_shade_seed, SurfaceSample,
};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineLayoutDescriptor, PollType,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

const GLINT_WESL: &str = include_str!("../../prism_render_scene/src/shaders/glint.wesl");
const BRDF_WESL: &str = include_str!("../../prism_render_scene/src/shaders/brdf.wesl");

/// Extract one `fn NAME(...) -> ... { ... }` body verbatim.
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

/// Extract a `struct NAME { ... }` declaration verbatim.
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

/// Extract a `const NAME: T = ...;` declaration verbatim.
fn extract_const(source: &str, label: &str, name: &str) -> String {
    let start = source
        .find(&format!("const {name}:"))
        .unwrap_or_else(|| panic!("{name} missing from {label}"));
    let end = source[start..].find(';').expect("const terminator") + start + 1;
    source[start..end].to_string()
}

/// One sample: the glint parameters, the half-vector, and a smooth density to
/// exercise the arbitrary-density entry point (the anisotropic path).
struct Sample {
    cell: [f32; 2],
    half: [f32; 3],
    density: f32,
    roughness: f32,
    footprint: f32,
    bin_half_width: f32,
    weight: f32,
    n_dot_h: f32,
    alpha: f32,
    d_smooth: f32,
}

fn samples() -> Vec<Sample> {
    let mut out = Vec::new();
    for cell in [[0.0_f32, 0.0], [0.31, 0.77], [12.5, -3.25], [0.999, 0.001]] {
        for half in [
            [0.0_f32, 1.0, 0.0],
            [0.2, 0.9, 0.1],
            [0.7, 0.3, -0.4],
            [-0.5, 0.5, 0.7],
        ] {
            for (density, roughness) in [(4.0_f32, 0.1_f32), (400.0, 0.15), (1.0e5, 0.6)] {
                for footprint in [1.0e-6_f32, 1.0e-4, 0.01] {
                    for bin_half_width in [0.01_f32, 0.05, 0.3] {
                        for (weight, n_dot_h, alpha, d_smooth) in [
                            (0.0_f32, 0.9_f32, 0.02_f32, 120.0_f32),
                            (1.0, 0.9, 0.02, 120.0),
                            (1.0, 0.35, 0.35, 3.0),
                            (0.4, 0.7, 0.1, 25.0),
                        ] {
                            out.push(Sample {
                                cell,
                                half,
                                density,
                                roughness,
                                footprint,
                                bin_half_width,
                                weight,
                                n_dot_h,
                                alpha,
                                d_smooth,
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

#[test]
fn glint_estimator_and_seed_match_on_device() {
    let samples = samples();
    let count = samples.len();

    // The kernel is assembled from the shipping sources: the whole glint kernel
    // chain plus `brdf.wesl`'s seed derivation.
    let mut shader = String::from(
        "@group(0) @binding(0) var<storage, read_write> out_seed: array<u32>;\n\
         @group(0) @binding(1) var<storage, read_write> out_iso: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> out_density: array<f32>;\n\
         @group(0) @binding(3) var<storage, read> in_cell: array<vec2<f32>>;\n\
         @group(0) @binding(4) var<storage, read> in_half: array<vec4<f32>>;\n\
         @group(0) @binding(5) var<storage, read> in_params: array<vec4<f32>>;\n\
         @group(0) @binding(6) var<storage, read> in_misc: array<vec4<f32>>;\n",
    );
    for name in [
        "GLINT_PI",
        "GLINT_MIN_ALPHA",
        "GLINT_MIN_COS",
        "GLINT_MIN_POSITIVE",
        "GLINT_KNUTH_MAX_LAMBDA",
        "GLINT_POISSON_MAX_ITERS",
        "GLINT_SEED_SALT",
    ] {
        shader.push_str(&extract_const(GLINT_WESL, "glint.wesl", name));
        shader.push('\n');
    }
    shader.push_str(&extract_const(BRDF_WESL, "brdf.wesl", "MIN_N_DOT"));
    shader.push('\n');
    for name in [
        "glint_is_finite",
        "glint_hash_u32",
        "glint_hash_combine",
        "glint_uniform_from_hash",
        "glint_uniform_at",
        "glint_quantize_axis",
        "glint_half_slope",
        "glint_seed",
        "glint_bin_solid_angle",
        "glint_microfacet_count",
        "glint_ggx_ndf",
        "glint_poisson_knuth",
        "glint_poisson_gaussian",
        "glint_poisson_sample",
        "glint_ndf_from_density",
        "glint_ndf_blended_from_density",
        "glint_ndf_blended",
    ] {
        shader.push_str(&extract_fn(GLINT_WESL, "glint.wesl", name));
        shader.push('\n');
    }
    // The struct the seed derivation reads, taken verbatim from the shipping
    // shader so a field rename/removal there breaks this gate instead of
    // silently diverging.
    shader.push_str(&extract_struct(BRDF_WESL, "brdf.wesl", "SurfaceSample"));
    shader.push('\n');
    shader.push_str(&extract_fn(BRDF_WESL, "brdf.wesl", "brdf_glint_shade_seed"));
    shader.push_str(
        "\n@compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
             let idx = gid.x;\n\
             if (idx >= arrayLength(&in_cell)) { return; }\n\
             let params = in_params[idx];\n\
             let misc = in_misc[idx];\n\
             var surface: SurfaceSample;\n\
             surface.glint_cell = in_cell[idx];\n\
             surface.glint_density = params.x;\n\
             surface.glint_roughness = params.y;\n\
             surface.glint_footprint_area = params.z;\n\
             surface.glint_bin_half_width = params.w;\n\
             surface.glint_weight = misc.x;\n\
             let n = vec3<f32>(0.0, 1.0, 0.0);\n\
             let tangent = vec3<f32>(1.0, 0.0, 0.0);\n\
             let bitangent = vec3<f32>(0.0, 0.0, -1.0);\n\
             let seed = brdf_glint_shade_seed(surface, in_half[idx].xyz, tangent, bitangent, n);\n\
             let facets = glint_microfacet_count(surface.glint_density, surface.glint_footprint_area);\n\
             let dw = glint_bin_solid_angle(surface.glint_bin_half_width);\n\
             out_seed[idx] = seed;\n\
             out_iso[idx] = glint_ndf_blended(misc.y, misc.z, facets, dw, surface.glint_weight, seed);\n\
             out_density[idx] = glint_ndf_blended_from_density(misc.w, facets, dw, surface.glint_weight, seed);\n\
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
        eprintln!("skipping glint GPU parity: no usable adapter");
        return;
    };
    let (device, queue) =
        block_on(adapter.request_device(&DeviceDescriptor::default())).expect("device");
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("prism_glint_parity"),
        source: ShaderSource::Wgsl(shader.into()),
    });

    let mut cell_bytes = Vec::new();
    let mut half_bytes = Vec::new();
    let mut params_bytes = Vec::new();
    let mut misc_bytes = Vec::new();
    for sample in &samples {
        for value in sample.cell {
            cell_bytes.extend_from_slice(&value.to_le_bytes());
        }
        // `array<vec4<f32>>` (16-byte stride): a plain `vec3` array would be
        // padded to 16 bytes by WGSL and silently shift every element after the
        // first — the mismatch this gate is here to catch.
        for value in [sample.half[0], sample.half[1], sample.half[2], 0.0] {
            half_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [
            sample.density,
            sample.roughness,
            sample.footprint,
            sample.bin_half_width,
        ] {
            params_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [sample.weight, sample.n_dot_h, sample.alpha, sample.d_smooth] {
            misc_bytes.extend_from_slice(&value.to_le_bytes());
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
    let in_cell = make_buffer("in_cell", &cell_bytes);
    let in_half = make_buffer("in_half", &half_bytes);
    let in_params = make_buffer("in_params", &params_bytes);
    let in_misc = make_buffer("in_misc", &misc_bytes);

    let out_seed_bytes = (count * 4) as u64;
    let out_float_bytes = (count * 4) as u64;
    let out_seed = device.create_buffer(&BufferDescriptor {
        label: Some("out_seed"),
        size: out_seed_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_iso = device.create_buffer(&BufferDescriptor {
        label: Some("out_iso"),
        size: out_float_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_density = device.create_buffer(&BufferDescriptor {
        label: Some("out_density"),
        size: out_float_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: out_seed_bytes + 2 * out_float_bytes,
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
            storage(1, false),
            storage(2, false),
            storage(3, true),
            storage(4, true),
            storage(5, true),
            storage(6, true),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("glint_parity"),
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
                resource: out_seed.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: out_iso.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: out_density.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: in_cell.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: in_half.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: in_params.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 6,
                resource: in_misc.as_entire_binding(),
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
    encoder.copy_buffer_to_buffer(&out_seed, 0, &readback, 0, out_seed_bytes);
    encoder.copy_buffer_to_buffer(&out_iso, 0, &readback, out_seed_bytes, out_float_bytes);
    encoder.copy_buffer_to_buffer(
        &out_density,
        0,
        &readback,
        out_seed_bytes + out_float_bytes,
        out_float_bytes,
    );
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
    let gpu_seeds: Vec<u32> = data[..count * 4]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let floats: Vec<f32> = data[count * 4..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for (index, sample) in samples.iter().enumerate() {
        let surface = SurfaceSample {
            glint_weight: sample.weight,
            glint_roughness: sample.roughness,
            glint_density: sample.density,
            glint_bin_half_width: sample.bin_half_width,
            glint_cell: sample.cell,
            glint_footprint_area: sample.footprint,
            ..SurfaceSample::default()
        };
        let n = [0.0, 1.0, 0.0];
        let seed = glint_shade_seed(&surface, sample.half, [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], n);
        // The seed is an integer hash of quantised bins: it must agree exactly,
        // which is what makes the sparkle pattern identical on both lines.
        assert_eq!(gpu_seeds[index], seed, "case {index}: seed mismatch");

        let facets = glint_microfacet_count(sample.density, sample.footprint);
        let dw = glint_bin_solid_angle(sample.bin_half_width);
        let want_iso = glint_ndf_blended(
            sample.n_dot_h,
            sample.alpha,
            facets,
            dw,
            sample.weight,
            seed,
        );
        let want_density =
            glint_ndf_blended_from_density(sample.d_smooth, facets, dw, sample.weight, seed);
        // `weight == 0` pins the bypass on both lines: the blend must return the
        // smooth density itself rather than a rounded estimator draw.
        if sample.weight == 0.0 {
            // Weight 0 returns the *smooth* GGX density, not an estimator draw.
            // The device may contract the NDF's arithmetic into an FMA (measured:
            // one ULP), so this comparison is bounded rather than exact.
            let want_plain = glint_ggx_ndf(sample.n_dot_h, sample.alpha);
            let tolerance = 1.0e-6 * want_plain.abs().max(1.0);
            assert!(
                (floats[index] - want_plain).abs() <= tolerance,
                "case {index}: weight-0 bypass gpu {} vs cpu {want_plain}",
                floats[index]
            );
            assert_eq!(
                floats[count + index],
                sample.d_smooth,
                "case {index}: weight-0 density bypass"
            );
        }

        for (label, want, got) in [
            ("iso", want_iso, floats[index]),
            ("density", want_density, floats[count + index]),
        ] {
            let tolerance = 1.0e-5 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "case {index} {label}: gpu {got} vs cpu {want}"
            );
        }
    }
}
