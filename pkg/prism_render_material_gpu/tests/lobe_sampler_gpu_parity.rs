//! **Real-device parity for the P3-e single-lobe sampler (§17.4).**
//!
//! The sampler is tiny — a blue-noise draw, a mixture-state verdict for the
//! blend-shaped lobe and an `1/p` factor for the additive group — but every part
//! of it is a *decision*, not a magnitude: a drift in the sequence, the temporal
//! stride or the `select` argument order would silently change which lobes a
//! pixel evaluates, and the estimator's mean would still look plausible. This
//! test runs the shipping WESL (`lobe_sampling.wesl`) over a grid of pixels,
//! frames and weight pairs and asserts the CPU twin draws **exactly** the same
//! state and factor.
//!
//! On a host without a usable adapter the test skips — pass `PRISM_REQUIRE_GPU=1`
//! to make a missing adapter fail instead.

use futures_lite::future::block_on;
use prism_render_shading::{
    select_expensive_lobe_factors, LobeSamplingPolicy, ShadingQuality,
};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineLayoutDescriptor, PollType,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

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

/// One case: the pixel, the frame and the authored expensive-lobe weights.
#[derive(Clone, Copy)]
struct Case {
    pixel: [u32; 2],
    frame: u32,
    weights: [f32; 2],
    /// `max_full_lobes` as the pass ships it (`u32::MAX` = full evaluation).
    max_full_lobes: u32,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for max_full_lobes in [u32::MAX, 1, 2] {
        for (glint, measured) in [
            (0.0_f32, 0.0_f32),
            (0.35, 0.0),
            (0.0, 0.8),
            (0.35, 0.6),
            (1.0, 1.0),
            (0.05, 0.95),
        ] {
            for pixel in [[0u32, 0u32], [1, 0], [37, 19], [1920, 1080]] {
                for frame in [0u32, 1, 7, 255] {
                    out.push(Case {
                        pixel,
                        frame,
                        weights: [glint, measured],
                        max_full_lobes,
                    });
                }
            }
        }
    }
    out
}

#[test]
fn lobe_sampler_factors_match_on_device() {
    let cases = cases();
    let count = cases.len();

    let mut shader = String::from(
        "@group(0) @binding(0) var<storage, read_write> out_factors: array<vec4<f32>>;\n\
         @group(0) @binding(1) var<storage, read> in_cases: array<vec4<f32>>;\n\
         @group(0) @binding(2) var<storage, read> in_extra: array<vec2<u32>>;\n",
    );
    for name in [
        "LS_SEED_SALT",
        "LS_GOLDEN_STRIDE",
        "LS_TEMPORAL_SALT",
        "LS_SEQUENCE_PERIOD",
        "LS_SEQUENCE_MASK",
        "LS_R2_A1",
        "LS_R2_A2",
        "LS_MIN_WEIGHT",
        "LS_ONE_MINUS_EPSILON",
        "LS_NO_LOBE",
    ] {
        let start = LOBE_SAMPLING_WESL
            .find(&format!("const {name}:"))
            .unwrap_or_else(|| panic!("{name} missing"));
        let end = LOBE_SAMPLING_WESL[start..].find(';').expect("const") + start + 1;
        shader.push_str(&LOBE_SAMPLING_WESL[start..end]);
        shader.push('\n');
    }
    for name in [
        "ls_is_finite",
        "ls_hash_u32",
        "ls_hash_combine",
        "ls_uniform_from_hash",
        "ls_uniform_at",
        "ls_sanitize_weight",
        "ls_clamp_unit",
        "ls_rotate_left_16",
        "ls_r2_sample_2d",
        "ls_pixel_seed",
        "ls_temporal_stride",
        "ls_blue_noise_2d",
        "ls_blue_noise_1d",
        "ls_should_use_stochastic",
        "ls_select_expensive_lobe_factors",
    ] {
        shader.push_str(&extract_fn(LOBE_SAMPLING_WESL, "lobe_sampling.wesl", name));
        shader.push('\n');
    }
    shader.push_str(
        "\n@compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
             let idx = gid.x;\n\
             if (idx >= arrayLength(&in_cases)) { return; }\n\
             let entry = in_cases[idx];\n\
             let extra = in_extra[idx];\n\
             let pixel = vec2<u32>(bitcast<u32>(entry.x), bitcast<u32>(entry.y));\n\
             let factors = ls_select_expensive_lobe_factors(\n\
                 vec2<f32>(entry.z, entry.w),\n\
                 extra.y,\n\
                 pixel,\n\
                 extra.x,\n\
                 0u,\n\
                 LS_SEED_SALT,\n\
             );\n\
             out_factors[idx] = vec4<f32>(factors, 0.0, 0.0);\n\
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
        eprintln!("skipping lobe-sampler GPU parity: no usable adapter");
        return;
    };
    let (device, queue) =
        block_on(adapter.request_device(&DeviceDescriptor::default())).expect("device");
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("prism_lobe_sampler_parity"),
        source: ShaderSource::Wgsl(shader.into()),
    });

    // Pack each case as (pixel.x bits, pixel.y bits, glint weight, measured
    // weight) plus a second buffer for the frame and the tier word.  The extras
    // travel as a second `vec2` so the shader reads exactly what the CPU sees.
    let mut case_bytes = Vec::new();
    let mut extra_bytes = Vec::new();
    for case in &cases {
        for value in [
            f32::from_bits(case.pixel[0]),
            f32::from_bits(case.pixel[1]),
            case.weights[0],
            case.weights[1],
        ] {
            case_bytes.extend_from_slice(&value.to_le_bytes());
        }
        // Frame and tier travel as raw integers: a NaN round-trip through a
        // float register could canonicalise on some hardware, and the tier word
        // legitimately uses `u32::MAX`.
        for value in [case.frame, case.max_full_lobes] {
            extra_bytes.extend_from_slice(&value.to_le_bytes());
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
    let in_cases = make_buffer("in_cases", &case_bytes);
    let in_extra = make_buffer("in_extra", &extra_bytes);
    let out_bytes = (count * 16) as u64;
    let out_factors = device.create_buffer(&BufferDescriptor {
        label: Some("out_factors"),
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
        entries: &[storage(0, false), storage(1, true), storage(2, true)],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("lobe_sampler_parity"),
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
                resource: out_factors.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: in_cases.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: in_extra.as_entire_binding(),
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
    encoder.copy_buffer_to_buffer(&out_factors, 0, &readback, 0, out_bytes);
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
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut stochastic = 0usize;
    for (index, case) in cases.iter().enumerate() {
        // The canonical tiers *are* the three budgets under test: Low = 1,
        // Medium = 2, High/Ultra = evaluate everything.
        let policy = LobeSamplingPolicy::for_quality(match case.max_full_lobes {
            u32::MAX => ShadingQuality::High,
            2 => ShadingQuality::Medium,
            _ => ShadingQuality::Low,
        });
        assert_eq!(policy.max_full_lobes(), case.max_full_lobes);
        let want =
            select_expensive_lobe_factors(policy, case.weights, case.pixel, case.frame, 0);
        let got = [floats[index * 4], floats[index * 4 + 1]];
        if want != prism_render_shading::ExpensiveLobeFactors::FULL {
            stochastic += 1;
        }
        assert_eq!(
            got[0], want.glint_state,
            "case {index} (pixel {:?} frame {} weights {:?}): glint state",
            case.pixel, case.frame, case.weights
        );
        assert_eq!(
            got[1], want.measured,
            "case {index} (pixel {:?} frame {} weights {:?}): measured factor",
            case.pixel, case.frame, case.weights
        );
    }
    // The grid is deliberately dominated by the full-quality cases (two of the
    // three tiers evaluate everything, and only the two-heavy-lobe weight pairs
    // are sampled at all), so the guard is "clearly non-vacuous", not a share.
    assert!(
        stochastic >= 32,
        "the grid must exercise the sampler ({stochastic}/{} stochastic)",
        cases.len()
    );
}
