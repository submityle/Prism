//! **Real-device parity for the analytic per-pixel texture LOD.**
//!
//! The material resolve is a compute dispatch, so there are no `dpdx`/`dpdy`
//! derivatives and the mip level is derived analytically from the triangle's
//! projected vertices (`prism_uv_gradients` / `prism_texture_lod` in
//! `material_sample.wesl`, roadmap P1-a / §7.1). Those functions are mirrored by
//! the CPU golden in `prism_render_shading::texture_lod`, and the two are only
//! construction-verified (the shader compiles) until this test runs the
//! **shipping WESL text** on the GPU and diffs it against that golden over a
//! grid of triangles.
//!
//! Reading the functions out of `material_sample.wesl` rather than restating
//! them here makes this a gate on the shipping source, not on a copy. On a host
//! without a usable adapter the test skips — pass `PRISM_REQUIRE_GPU=1` to turn
//! a missing adapter into a failure instead.

use futures_lite::future::block_on;
use prism_render_shading::{texture_lod, texture_uv_gradients};
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineLayoutDescriptor, PollType,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

const MATERIAL_SAMPLE_WESL: &str =
    include_str!("../../prism_render_scene/src/shaders/material_sample.wesl");

/// Extract one `fn NAME(...) -> ... { ... }` body verbatim from the WESL source.
fn extract_fn(name: &str) -> String {
    let src = MATERIAL_SAMPLE_WESL;
    let needle = format!("fn {name}(");
    let start = src
        .find(&needle)
        .unwrap_or_else(|| panic!("{name} missing from material_sample.wesl"));
    let open = src[start..].find('{').expect("fn body") + start;
    let mut depth = 0usize;
    for (i, ch) in src[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return src[start..open + i + 1].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated {name}");
}

/// Extract a scalar `const NAME: f32 = ...;` declaration verbatim.
fn extract_const(name: &str) -> String {
    let src = MATERIAL_SAMPLE_WESL;
    let start = src
        .find(&format!("const {name}:"))
        .unwrap_or_else(|| panic!("{name} missing from material_sample.wesl"));
    let end = src[start..].find(';').expect("const terminator") + start + 1;
    src[start..end].to_string()
}

/// One triangle case: three UVs, three screen-space positions (pixels) and a
/// texture size.
struct Case {
    uv: [[f32; 2]; 3],
    screen: [[f32; 2]; 3],
    dimensions: [f32; 2],
}

/// Rotate `p` about the origin by the `(sin, cos)` pair. Exact literals instead
/// of `sin_cos` so the grid is identical everywhere and the workspace's
/// libm-determinism lint stays quiet.
fn rotate(p: [f32; 2], (s, c): (f32, f32)) -> [f32; 2] {
    [p[0] * c - p[1] * s, p[0] * s + p[1] * c]
}

/// Axes plus 15°, ~36.87° (a 3-4-5 rotation), 90°, 180° and 270°.
const ROTATIONS: [(f32, f32); 6] = [
    (0.0, 1.0),
    (0.258_819_04, 0.965_925_8),
    (0.6, 0.8),
    (1.0, 0.0),
    (0.0, -1.0),
    (-1.0, 0.0),
];

/// The grid: right triangles scaled and rotated, sheared triangles (to exercise
/// the off-diagonal gradients) and degenerate/sliver cases that must all land on
/// the base level.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for screen_extent in [0.125_f32, 0.5, 1.0, 2.0, 7.0, 64.0, 512.0, 4096.0] {
        for uv_extent in [0.125_f32, 1.0, 8.0] {
            for rotation in ROTATIONS {
                let base_screen = [[0.0, 0.0], [screen_extent, 0.0], [0.0, screen_extent]];
                let base_uv = [[0.0, 0.0], [uv_extent, 0.0], [0.0, uv_extent]];
                out.push(Case {
                    uv: base_uv,
                    screen: [
                        rotate(base_screen[0], rotation),
                        rotate(base_screen[1], rotation),
                        rotate(base_screen[2], rotation),
                    ],
                    dimensions: [1024.0, 1024.0],
                });
            }
        }
    }
    // Non-square textures and anisotropic-in-UV triangles.
    out.push(Case {
        uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        screen: [[0.0, 0.0], [16.0, 0.0], [0.0, 4.0]],
        dimensions: [2048.0, 256.0],
    });
    out.push(Case {
        uv: [[0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        screen: [[0.0, 0.0], [2.0, 0.0], [0.0, 8.0]],
        dimensions: [512.0, 512.0],
    });
    // Sheared UVs (a skewed atlas cell).
    out.push(Case {
        uv: [[0.0, 0.0], [1.0, 0.25], [0.25, 1.0]],
        screen: [[0.0, 0.0], [32.0, 0.0], [0.0, 32.0]],
        dimensions: [1024.0, 1024.0],
    });
    // Degenerate: collinear and coincident vertices.
    out.push(Case {
        uv: [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        screen: [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dimensions: [4096.0, 4096.0],
    });
    out.push(Case {
        uv: [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        screen: [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]],
        dimensions: [4096.0, 4096.0],
    });
    out.push(Case {
        uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        screen: [[0.0, 0.0], [1.0e-7, 0.0], [0.0, 1.0e-7]],
        dimensions: [8192.0, 8192.0],
    });
    // Magnified beyond 1:1 (must clamp to the base level).
    out.push(Case {
        uv: [[0.0, 0.0], [0.02, 0.0], [0.0, 0.02]],
        screen: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dimensions: [64.0, 64.0],
    });
    // Non-finite inputs. These are the cases where Rust's `f32::max` (which
    // *ignores* NaN) and WGSL's `max` (which propagates it) would disagree, so
    // the guard has to be and stay explicit on both sides.
    out.push(Case {
        uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        screen: [[0.0, 0.0], [f32::NAN, 0.0], [0.0, 16.0]],
        dimensions: [2048.0, 2048.0],
    });
    out.push(Case {
        uv: [[0.0, 0.0], [f32::INFINITY, 0.0], [0.0, 1.0]],
        screen: [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]],
        dimensions: [2048.0, 2048.0],
    });
    out
}

#[test]
fn analytic_texture_lod_matches_on_device() {
    let cases = cases();
    let count = cases.len();

    let mut shader = String::from(
        "@group(0) @binding(0) var<storage, read_write> out_grad: array<vec4<f32>>;\n\
         @group(0) @binding(1) var<storage, read_write> out_lod: array<f32>;\n\
         @group(0) @binding(2) var<storage, read> in_uv: array<vec2<f32>>;\n\
         @group(0) @binding(3) var<storage, read> in_screen: array<vec2<f32>>;\n\
         @group(0) @binding(4) var<storage, read> in_dimensions: array<vec2<f32>>;\n",
    );
    shader.push_str(&extract_const("PRISM_MIN_SCREEN_AREA"));
    shader.push('\n');
    shader.push_str(&extract_const("PRISM_FINITE_RANGE"));
    shader.push('\n');
    shader.push_str(&extract_fn("prism_lod_in_range"));
    shader.push('\n');
    shader.push_str(&extract_fn("prism_uv_gradients"));
    shader.push('\n');
    shader.push_str(&extract_fn("prism_texture_lod"));
    shader.push_str(
        "\n@compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
             let idx = gid.x;\n\
             if (idx >= arrayLength(&in_dimensions)) { return; }\n\
             let base = idx * 3u;\n\
             let gradients = prism_uv_gradients(\n\
                 array<vec2<f32>, 3>(in_uv[base], in_uv[base + 1u], in_uv[base + 2u]),\n\
                 array<vec2<f32>, 3>(\n\
                     in_screen[base], in_screen[base + 1u], in_screen[base + 2u]),\n\
             );\n\
             out_grad[idx] = vec4<f32>(gradients[0], gradients[1]);\n\
             out_lod[idx] = prism_texture_lod(gradients, in_dimensions[idx]);\n\
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
        eprintln!("skipping texture-LOD GPU parity: no usable adapter");
        return;
    };
    let (device, queue) =
        block_on(adapter.request_device(&DeviceDescriptor::default())).expect("device");

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("prism_texture_lod_parity"),
        source: ShaderSource::Wgsl(shader.into()),
    });

    let uv_bytes: Vec<u8> = cases
        .iter()
        .flat_map(|case| case.uv.iter())
        .flat_map(|v| [v[0].to_le_bytes(), v[1].to_le_bytes()].concat())
        .collect();
    let screen_bytes: Vec<u8> = cases
        .iter()
        .flat_map(|case| case.screen.iter())
        .flat_map(|v| [v[0].to_le_bytes(), v[1].to_le_bytes()].concat())
        .collect();
    let dimension_bytes: Vec<u8> = cases
        .iter()
        .flat_map(|case| {
            [
                case.dimensions[0].to_le_bytes(),
                case.dimensions[1].to_le_bytes(),
            ]
            .concat()
        })
        .collect();

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
    let in_uv = make_buffer("in_uv", &uv_bytes);
    let in_screen = make_buffer("in_screen", &screen_bytes);
    let in_dimensions = make_buffer("in_dimensions", &dimension_bytes);

    let out_grad_bytes = (count * 16) as u64;
    let out_lod_bytes = (count * 4) as u64;
    let out_grad = device.create_buffer(&BufferDescriptor {
        label: Some("out_grad"),
        size: out_grad_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let out_lod = device.create_buffer(&BufferDescriptor {
        label: Some("out_lod"),
        size: out_lod_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: out_grad_bytes + out_lod_bytes,
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
            storage(2, true),
            storage(3, true),
            storage(4, true),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("texture_lod_parity"),
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
                resource: out_grad.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: out_lod.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: in_uv.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: in_screen.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: in_dimensions.as_entire_binding(),
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
    encoder.copy_buffer_to_buffer(&out_grad, 0, &readback, 0, out_grad_bytes);
    encoder.copy_buffer_to_buffer(&out_lod, 0, &readback, out_grad_bytes, out_lod_bytes);
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

    for (index, case) in cases.iter().enumerate() {
        let want_grad = texture_uv_gradients(case.uv, case.screen);
        let got_grad = [
            floats[index * 4],
            floats[index * 4 + 1],
            floats[index * 4 + 2],
            floats[index * 4 + 3],
        ];
        for (component, want) in [
            want_grad[0][0],
            want_grad[0][1],
            want_grad[1][0],
            want_grad[1][1],
        ]
        .into_iter()
        .enumerate()
        {
            let got = got_grad[component];
            // `*`, `-` and `/` only: contraction cannot move these, so the bound
            // is here to catch a real formula/order drift, not to paper over
            // rounding.
            let tol = 1.0e-6 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "case {index} gradient[{component}]: gpu {got} vs cpu {want}"
            );
        }

        let want_lod = texture_lod(want_grad, case.dimensions);
        let got_lod = floats[count * 4 + index];
        // `sqrt` + `log2`: the device may fuse/approximate these, so the bound
        // is looser (still ~4 orders of magnitude below one mip step).
        assert!(
            (got_lod - want_lod).abs() <= 1.0e-4,
            "case {index} lod: gpu {got_lod} vs cpu {want_lod} (grad {want_grad:?})"
        );
    }
}
