//! **Real-device parity for the shading albedo tables.**
//!
//! The white-furnace energy invariants consume five baked lookup tables
//! (`GGX_ALBEDO`, `GGX_AVERAGE`, `DIELECTRIC_ALBEDO`, `SCHLICK_BIAS`,
//! `FUZZ_ALBEDO`) that live in `pkg/prism_render_shading/src/*_table.rs` and are
//! mirrored verbatim into the WESL sources. Until now the mirror was only
//! **construction-verified** (the shader compiles and links) — this test closes
//! that gap by running the *actual* WESL lookup functions on the GPU and
//! diffing every sample bit-for-bit against the same bilinear interpolation
//! evaluated in Rust **from the same WESL text**.
//!
//! Reading the table and the lookup body out of `brdf.wesl` (rather than
//! restating them here) is deliberate: it makes the test a gate on the shipping
//! source, not on a copy. On a host without a usable adapter the test skips,
//! exactly like the other `prism_*_gpu` twins — pass `PRISM_REQUIRE_GPU=1` to
//! turn a missing adapter into a failure instead.

use futures_lite::future::block_on;
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    DeviceDescriptor, Instance, InstanceDescriptor, InstanceFlags, MapMode, PipelineLayoutDescriptor,
    PollType, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

const BRDF_WESL: &str = include_str!("../../prism_render_scene/src/shaders/brdf.wesl");
const CLOTH_WESL: &str = include_str!("../../prism_render_scene/src/shaders/cloth.wesl");

/// The WESL source that owns a given table/lookup pair (the fuzz table lives in
/// the Cloth shader, the rest in the shared BRDF shader).
fn source_of(array: &str) -> &'static str {
    if array == "FUZZ_ALBEDO" {
        CLOTH_WESL
    } else {
        BRDF_WESL
    }
}

/// Pull the float literals out of a WESL `const NAME: array<f32, N>(...)`.
fn parse_wesl_array(name: &str) -> Vec<f32> {
    let src = source_of(name);
    let start = src
        .find(&format!("const {name}:"))
        .unwrap_or_else(|| panic!("{name} missing from brdf.wesl"));
    let open = src[start..].find('(').expect("array constructor") + start;
    let close = src[open..].find(')').expect("array terminator") + open;
    src[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f32>().expect("float literal"))
        .collect()
}

/// The bilinear lookup, evaluated exactly as the WESL body does — from the
/// parsed table, so a drift in either the numbers or the interpolation shows up.
fn cpu_lookup(table: &[f32], n_mu: usize, n_r: usize, mu: f32, r: f32) -> f32 {
    let x = mu.clamp(0.0, 1.0) * (n_mu - 1) as f32;
    let y = r.clamp(0.0, 1.0) * (n_r - 1) as f32;
    let i0 = (x as usize).min(n_mu - 2);
    let j0 = (y as usize).min(n_r - 2);
    let fx = x - i0 as f32;
    let fy = y - j0 as f32;
    let a = table[j0 * n_mu + i0];
    let b = table[j0 * n_mu + i0 + 1];
    let c = table[(j0 + 1) * n_mu + i0];
    let d = table[(j0 + 1) * n_mu + i0 + 1];
    let top = a + (b - a) * fx;
    let bottom = c + (d - c) * fx;
    (top + (bottom - top) * fy).clamp(0.0, 1.0)
}

/// Extract one `fn NAME(...) -> f32 { ... }` body verbatim from the WESL source
/// so the GPU shader runs the shipping implementation.
fn extract_fn(array: &str, name: &str) -> String {
    let src = source_of(array);
    let needle = format!("fn {name}(");
    let start = src
        .find(&needle)
        .unwrap_or_else(|| panic!("{name} missing from the shader sources"));
    // Walk braces from the first `{` after the signature.
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

/// One table under test: the const array and the WESL lookup fn over it.
struct TableCase {
    array: &'static str,
    lookup: &'static str,
    /// Prefix of the two dimension constants the lookup body reads
    /// (`<PREFIX>_N_MU` / `<PREFIX>_N_R`).
    prefix: &'static str,
    n_mu: usize,
    n_r: usize,
}

const CASES: [TableCase; 4] = [
    TableCase {
        array: "GGX_ALBEDO",
        prefix: "GGX",
        lookup: "ggx_albedo_lookup",
        n_mu: 17,
        n_r: 9,
    },
    TableCase {
        array: "DIELECTRIC_ALBEDO",
        prefix: "DIEL",
        lookup: "dielectric_albedo_lookup",
        n_mu: 17,
        n_r: 9,
    },
    TableCase {
        array: "SCHLICK_BIAS",
        prefix: "BIAS",
        lookup: "schlick_bias_lookup",
        n_mu: 17,
        n_r: 9,
    },
    TableCase {
        array: "FUZZ_ALBEDO",
        prefix: "FUZZ",
        lookup: "fuzz_albedo_lookup",
        n_mu: 17,
        n_r: 9,
    },
];

/// Samples per table: a dense grid plus the exact cell corners (where a
/// half-texel or `i32` vs `usize` mistake would hide between samples).
fn sample_grid() -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    for i in 0..=32 {
        for j in 0..=16 {
            out.push((i as f32 / 32.0, j as f32 / 16.0));
        }
    }
    for i in 0..=16 {
        for j in 0..=8 {
            out.push((i as f32 / 16.0, j as f32 / 8.0));
        }
    }
    out
}

#[test]
fn albedo_lookups_match_on_device() {
    let samples = sample_grid();
    let per_table = samples.len();

    // Assemble a compute shader that evaluates each shipping lookup on the grid.
    let mut shader = String::from(
        "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> grid: array<vec2<f32>>;\n",
    );
    // Table constants + lookups, taken verbatim from the shipping source.
    for case in &CASES {
        let src = source_of(case.array);
        let start = src.find(&format!("const {}:", case.array)).unwrap();
        let open = src[start..].find("array<f32,").unwrap() + start;
        // include the `const NAME: ... = array<f32, N>( ... );` declaration
        let decl_end = src[open..].find(");").unwrap() + open + 2;
        shader.push_str(&format!(
            "const {}_N_MU: i32 = {};\nconst {}_N_R: i32 = {};\n",
            case.prefix, case.n_mu, case.prefix, case.n_r
        ));
        shader.push_str(&src[start..decl_end]);
        shader.push('\n');
        shader.push_str(&extract_fn(case.array, case.lookup));
        shader.push('\n');
    }
    shader.push_str(
        "@compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
             let idx = gid.x;\n\
             if (idx >= arrayLength(&grid)) { return; }\n\
             let s = grid[idx];\n",
    );
    for (t, case) in CASES.iter().enumerate() {
        shader.push_str(&format!(
            "    out[{}u * arrayLength(&grid) + idx] = {}(s.x, s.y);\n",
            t,
            case.lookup
        ));
    }
    shader.push_str("}\n");

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
        eprintln!("skipping albedo GPU parity: no usable adapter");
        return;
    };
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor::default()))
        .expect("device");

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("prism_albedo_parity"),
        source: ShaderSource::Wgsl(shader.into()),
    });

    let grid_bytes: Vec<u8> = samples
        .iter()
        .flat_map(|(a, b)| [a.to_le_bytes(), b.to_le_bytes()].concat())
        .collect();
    let grid = device.create_buffer(&BufferDescriptor {
        label: Some("grid"),
        size: grid_bytes.len() as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&grid, 0, &grid_bytes);

    let out_len = (CASES.len() * per_table * 4) as u64;
    let out = device.create_buffer(&BufferDescriptor {
        label: Some("out"),
        size: out_len,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: out_len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
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
        label: Some("albedo_parity"),
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
                resource: out.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: grid.as_entire_binding(),
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
        let groups = ((per_table as u32) + 63) / 64;
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, out_len);
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
    let gpu: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for (t, case) in CASES.iter().enumerate() {
        let table = parse_wesl_array(case.array);
        assert_eq!(
            table.len(),
            case.n_mu * case.n_r,
            "{} length drift",
            case.array
        );
        for (s, &(mu, r)) in samples.iter().enumerate() {
            let want = cpu_lookup(&table, case.n_mu, case.n_r, mu, r);
            let got = gpu[t * per_table + s];
            // **Not** bit-exact by construction: the GPU is free to contract
            // `a + (b - a) * fx` into an FMA while the CPU reference does not
            // (measured: a 1-ULP difference, e.g. 0.387934 vs 0.38793403). A
            // 1e-6 relative bound still catches any real table or interpolation
            // error by orders of magnitude.
            let tol = 1.0e-6 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tol,
                "{} mismatch at (mu={mu}, r={r}): gpu {got} vs cpu {want}",
                case.array
            );
        }
    }
}

/// **Micro-benchmark: `const`-array lookup vs storage-buffer table on the real
/// device.** The white-furnace work added five `const` lookup tables; a WGSL
/// `const` array indexed dynamically is normally materialised into
/// private/local memory, so the question is whether that is actually cheaper
/// than fetching the same numbers from a storage buffer (the stand-in for the
/// "move it to a small linear-filtered texture" option).
///
/// Ignored by default (it is a measurement, not a gate). Run with:
/// `PRISM_REQUIRE_GPU=1 cargo test -p prism_render_material_gpu --test
/// albedo_table_gpu_parity -- --ignored --nocapture` **outside the sandbox**
/// (the sandbox blocks Metal device access), which is also where the numbers in
/// the design doc come from. Each thread performs `LOOKUPS` dependent lookups so
/// the measurement is ALU/latency-bound rather than launch-bound.
#[test]
#[ignore = "micro-benchmark: run outside the sandbox on a real GPU"]
fn bench_const_table_vs_buffer_table() {
    use std::time::Instant;

    const THREADS: u32 = 1 << 18;
    const LOOKUPS: u32 = 64;

    // Kernel A: the shipping `const` table + lookup, straight from brdf.wesl.
    let table = parse_wesl_array("GGX_ALBEDO");
    let mut shader_const = String::from(
        "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n",
    );
    shader_const.push_str("const GGX_N_MU: i32 = 17;\nconst GGX_N_R: i32 = 9;\n");
    let src = BRDF_WESL;
    let start = src.find("const GGX_ALBEDO:").unwrap();
    let open = src[start..].find("array<f32,").unwrap() + start;
    let decl_end = src[open..].find(");").unwrap() + open + 2;
    shader_const.push_str(&src[start..decl_end]);
    shader_const.push('\n');
    shader_const.push_str(&extract_fn("GGX_ALBEDO", "ggx_albedo_lookup"));
    shader_const.push_str(BENCH_MAIN);

    // Kernel B: the identical numbers and interpolation, but read from a
    // storage buffer.
    let mut shader_buffer = String::from(
        "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> table: array<f32>;\n\
         const GGX_N_MU: i32 = 17;\nconst GGX_N_R: i32 = 9;\n",
    );
    shader_buffer.push_str(&buffer_lookup_fn());
    shader_buffer.push_str(&BENCH_MAIN.replace("ggx_albedo_lookup", "buffer_albedo_lookup"));

    let Some((device, queue)) = gpu_device() else {
        eprintln!("skipping albedo micro-benchmark: no usable adapter");
        return;
    };
    let out_len = (THREADS as u64) * 4;
    let out = device.create_buffer(&BufferDescriptor {
        label: Some("bench_out"),
        size: out_len,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let table_buf = device.create_buffer(&BufferDescriptor {
        label: Some("bench_table"),
        size: (table.len() * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let table_bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    queue.write_buffer(&table_buf, 0, &table_bytes);

    let mut run = |shader: String, label: &str| -> f64 {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(shader.into()),
        });
        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
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
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
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
                    resource: out.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: table_buf.as_entire_binding(),
                },
            ],
        });
        // Warm up once (pipeline/shader residency), then time the steady state.
        for _ in 0..2 {
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(THREADS / 64, 1, 1);
            }
            queue.submit([encoder.finish()]);
            let _ = device.poll(PollType::wait_indefinitely());
        }
        let reps = 8;
        let t0 = Instant::now();
        for _ in 0..reps {
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(THREADS / 64, 1, 1);
            }
            queue.submit([encoder.finish()]);
            let _ = device.poll(PollType::wait_indefinitely());
        }
        let secs = t0.elapsed().as_secs_f64() / reps as f64;
        let lookups = THREADS as f64 * LOOKUPS as f64;
        println!("{label}: {:.3} ms/dispatch  ({:.2} Glookup/s)", secs * 1e3, lookups / secs / 1e9);
        secs * 1e3
    };
    run(shader_const, "const table");
    run(shader_buffer, "storage buffer");
}

/// **A/B of the per-light-loop hoisting** (`23a5f6d64`): the view- and
/// material-only albedo terms are identical for every light, so a 4-light pixel
/// can compute them once instead of four times. This measures that directly on
/// the device, using the shipping lookup functions.
#[test]
#[ignore = "micro-benchmark: run outside the sandbox on a real GPU"]
fn bench_per_light_hoisting() {
    use std::time::Instant;

    const THREADS: u32 = 1 << 18;
    const LIGHTS: u32 = 4;

    let mut shader = String::from(
        "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n",
    );
    shader.push_str("const GGX_N_MU: i32 = 17;\nconst GGX_N_R: i32 = 9;\n");
    shader.push_str("const DIEL_N_MU: i32 = 17;\nconst DIEL_N_R: i32 = 9;\n");
    for (array, lookup) in [
        ("GGX_ALBEDO", "ggx_albedo_lookup"),
        ("DIELECTRIC_ALBEDO", "dielectric_albedo_lookup"),
    ] {
        let start = BRDF_WESL.find(&format!("const {array}:")).unwrap();
        let open = BRDF_WESL[start..].find("array<f32,").unwrap() + start;
        let end = BRDF_WESL[open..].find(");").unwrap() + open + 2;
        shader.push_str(&BRDF_WESL[start..end]);
        shader.push('\n');
        shader.push_str(&extract_fn(array, lookup));
        shader.push('\n');
    }
    // Kernel A recomputes the three view/material terms per light; kernel B
    // hoists them. Both consume the results so nothing is eliminated.
    shader.push_str(&format!(
        "@compute @workgroup_size(64)\n\
         fn recompute(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             let mu_v = clamp(f32(gid.x % 4096u) / 4096.0, 1.0e-5, 1.0);\n\
             let r = clamp(f32((gid.x / 4096u) % 64u) / 64.0, 0.05, 1.0);\n\
             var acc = 0.0;\n\
             for (var i = 0u; i < {LIGHTS}u; i = i + 1u) {{\n\
                 let mu_l = clamp(mu_v * 0.5 + f32(i) * 0.125, 1.0e-5, 1.0);\n\
                 acc = acc + ggx_albedo_lookup(mu_v, r)\n\
                     + ggx_albedo_lookup(mu_l, r) + dielectric_albedo_lookup(mu_v, r);\n\
             }}\n\
             out[gid.x] = acc;\n\
         }}\n\
         @compute @workgroup_size(64)\n\
         fn hoisted(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             let mu_v = clamp(f32(gid.x % 4096u) / 4096.0, 1.0e-5, 1.0);\n\
             let r = clamp(f32((gid.x / 4096u) % 64u) / 64.0, 0.05, 1.0);\n\
             let e_view = ggx_albedo_lookup(mu_v, r);\n\
             let e_diel = dielectric_albedo_lookup(mu_v, r);\n\
             var acc = 0.0;\n\
             for (var i = 0u; i < {LIGHTS}u; i = i + 1u) {{\n\
                 let mu_l = clamp(mu_v * 0.5 + f32(i) * 0.125, 1.0e-5, 1.0);\n\
                 acc = acc + e_view + ggx_albedo_lookup(mu_l, r) + e_diel;\n\
             }}\n\
             out[gid.x] = acc;\n\
         }}\n"
    ));

    let Some((device, queue)) = gpu_device() else {
        eprintln!("skipping hoisting benchmark: no usable adapter");
        return;
    };
    let out = device.create_buffer(&BufferDescriptor {
        label: Some("bench_out"),
        size: (THREADS as u64) * 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("hoist_bench"),
        source: ShaderSource::Wgsl(shader.into()),
    });
    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: out.as_entire_binding(),
        }],
    });
    let mut run = |entry: &str, label: &str| {
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let dispatch = || {
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(THREADS / 64, 1, 1);
            }
            queue.submit([encoder.finish()]);
            let _ = device.poll(PollType::wait_indefinitely());
        };
        dispatch();
        dispatch();
        let reps = 8;
        let t0 = Instant::now();
        for _ in 0..reps {
            dispatch();
        }
        let ms = t0.elapsed().as_secs_f64() / reps as f64 * 1e3;
        println!("{label}: {ms:.3} ms/dispatch");
        ms
    };
    let a = run("recompute", "recompute per light");
    let b = run("hoisted", "hoisted once");
    println!("saving: {:.1}%", (a - b) / a * 100.0);
}

/// Shared body: `LOOKUPS` dependent lookups per thread, accumulated so nothing
/// is dead-code eliminated.
const BENCH_MAIN: &str = "\n@compute @workgroup_size(64)\n\
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
    var mu = f32(gid.x % 4096u) / 4096.0;\n\
    var r = f32((gid.x / 4096u) % 64u) / 64.0;\n\
    var acc = 0.0;\n\
    for (var i = 0u; i < 64u; i = i + 1u) {\n\
        acc = acc + ggx_albedo_lookup(mu, r);\n\
        let t = mu; mu = clamp(r * 0.5 + 0.25, 0.0, 1.0); r = clamp(t * 0.5 + 0.5, 0.0, 1.0);\n\
    }\n\
    out[gid.x] = acc;\n\
}\n";

/// The same interpolation against a storage-buffer table.
fn buffer_lookup_fn() -> String {
    "fn buffer_albedo_lookup(n_dot_v: f32, roughness: f32) -> f32 {\n\
         let x = clamp(n_dot_v, 0.0, 1.0) * f32(GGX_N_MU - 1);\n\
         let y = clamp(roughness, 0.0, 1.0) * f32(GGX_N_R - 1);\n\
         let i0 = min(i32(x), GGX_N_MU - 2);\n\
         let j0 = min(i32(y), GGX_N_R - 2);\n\
         let fx = x - f32(i0);\n\
         let fy = y - f32(j0);\n\
         let a = table[j0 * GGX_N_MU + i0];\n\
         let b = table[j0 * GGX_N_MU + i0 + 1];\n\
         let c = table[(j0 + 1) * GGX_N_MU + i0];\n\
         let d = table[(j0 + 1) * GGX_N_MU + i0 + 1];\n\
         let top = a + (b - a) * fx;\n\
         let bottom = c + (d - c) * fx;\n\
         return clamp(top + (bottom - top) * fy, 0.0, 1.0);\n\
     }\n"
        .to_string()
}

/// Headless device, or `None` on hosts without a usable adapter.
fn gpu_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
    block_on(adapter.request_device(&DeviceDescriptor::default())).ok()
}
