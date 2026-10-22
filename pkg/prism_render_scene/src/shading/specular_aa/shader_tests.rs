//! WESL link coverage for the shared specular-AA helper module.
//!
//! This deliberately links a tiny compute entry point against the module rather
//! than merely parsing its source. Every helper is referenced by the live entry
//! point, so import visibility, signatures, and type-checking are covered.
//! Numeric fidelity is guaranteed by exact mirroring of the CPU golden's
//! operation order, clamps, floors, and finite-sanitization branches.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

fn load_source(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("specular-AA shaders are WESL"),
    }
}

fn shader_id(tag: u128) -> AssetId<Shader> {
    AssetId::Uuid {
        uuid: Uuid::from_u128(tag),
    }
}

const LINK_TEST_SOURCE: &str = r#"
import prism_render_scene::shaders::specular_aa::{
    SPEC_AA_MIN_ALPHA, SPEC_AA_SIGMA2, SPEC_AA_KAPPA_MAX,
    TOKSVIG_MIN_LEN, TOKSVIG_MAX_SHININESS,
    spec_aa_is_finite,
    toksvig_sanitize_len,
    toksvig_sanitize_shininess,
    spec_aa_roughness_to_alpha,
    spec_aa_safe_sq_len,
    spec_aa_screen_space_variance,
    spec_aa_kernel_roughness_sq,
    spec_aa_filter_alpha_sq,
    spec_aa_delta_alpha_sq_from_derivatives,
    spec_aa_geometric_specular_aa_roughness,
    toksvig_factor,
    toksvig_effective_shininess,
    toksvig_shininess_from_alpha,
    toksvig_alpha_from_shininess,
    toksvig_delta_alpha_sq,
    toksvig_effective_alpha_sq_from_len,
    toksvig_roughness,
    toksvig_combine_roughness,
};

@group(0) @binding(0)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn specular_aa_link_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) {
        return;
    }
    let ddx = vec3<f32>(0.2, 0.05, 0.01);
    let ddy = vec3<f32>(0.03, 0.22, 0.0);
    let alpha = spec_aa_roughness_to_alpha(0.2);
    let sq_len = spec_aa_safe_sq_len(ddx);
    let variance = spec_aa_screen_space_variance(ddx, ddy, SPEC_AA_SIGMA2);
    let kernel = spec_aa_kernel_roughness_sq(variance, SPEC_AA_KAPPA_MAX);
    let filtered_alpha_sq = spec_aa_filter_alpha_sq(alpha * alpha, variance, SPEC_AA_KAPPA_MAX);
    let geometric_delta = spec_aa_delta_alpha_sq_from_derivatives(ddx, ddy, SPEC_AA_SIGMA2, SPEC_AA_KAPPA_MAX);
    let geometric_roughness = spec_aa_geometric_specular_aa_roughness(0.2, ddx, ddy, SPEC_AA_SIGMA2, SPEC_AA_KAPPA_MAX);
    let factor = toksvig_factor(0.7, 100.0);
    let effective_shininess = toksvig_effective_shininess(100.0, 0.7);
    let shininess = toksvig_shininess_from_alpha(alpha);
    let effective_alpha = toksvig_alpha_from_shininess(shininess);
    let toksvig_delta = toksvig_delta_alpha_sq(0.7);
    let toksvig_alpha_sq = toksvig_effective_alpha_sq_from_len(alpha, 0.7);
    let toksvig_rough = toksvig_roughness(0.2, 0.7);
    let combined = toksvig_combine_roughness(geometric_roughness, toksvig_rough);
    output[0] = sq_len;
    output[1] = variance;
    output[2] = kernel;
    output[3] = filtered_alpha_sq;
    output[4] = geometric_delta;
    output[5] = geometric_roughness;
    output[6] = factor;
    output[7] = effective_shininess;
    output[8] = shininess;
    output[9] = effective_alpha;
    output[10] = toksvig_delta;
    output[11] = toksvig_alpha_sq;
    output[12] = toksvig_rough;
    output[13] = combined;
    output[14] = select(0.0, 1.0, spec_aa_is_finite(alpha));
    output[15] = toksvig_sanitize_len(TOKSVIG_MIN_LEN);
    output[16] = toksvig_sanitize_shininess(TOKSVIG_MAX_SHININESS);
    output[17] = SPEC_AA_MIN_ALPHA;
}
"#;

/// Links the shared module through a real compute entry point, exercising every
/// exported helper so none can disappear behind dead-code elimination.
#[test]
fn specular_aa_wesl_compiles_and_links_every_helper() {
    let mut cache = ShaderCache::new((), load_source);

    let module = shader_id(0x5052_4953_4d5f_5350_4543_4141_0001);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../../shaders/specular_aa.wesl"),
            "embedded://prism_render_scene/shaders/specular_aa.wesl",
        ),
    );

    let link_test = shader_id(0x5052_4953_4d5f_5350_4543_4141_0002);
    cache.set_shader(
        link_test,
        Shader::from_wesl(
            LINK_TEST_SOURCE,
            "embedded://prism_render_scene/shading/specular_aa/link_test.wesl",
        ),
    );

    cache.get(0, link_test, &[]).unwrap_or_else(|error| {
        panic!("specular_aa.wesl failed to compile/link every helper: {error}")
    });
}

const GPU_PARITY_SOURCE: &str = r#"
import prism_render_scene::shaders::specular_aa::{
    SPEC_AA_SIGMA2, SPEC_AA_KAPPA_MAX,
    spec_aa_screen_space_variance,
    spec_aa_roughness_to_alpha,
    spec_aa_filter_alpha_sq,
    spec_aa_geometric_specular_aa_roughness,
    toksvig_factor,
    toksvig_roughness,
};

@group(0) @binding(0)
var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> input: array<vec4<f32>>;

@compute @workgroup_size(1)
fn specular_aa_gpu_parity(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let base = input[index * 3u];
    let normal = input[index * 3u + 1u];
    let extra = input[index * 3u + 2u];
    let ddx = vec3<f32>(base.y, base.z, base.w);
    let ddy = normal.xyz;
    let variance = spec_aa_screen_space_variance(ddx, ddy, SPEC_AA_SIGMA2);
    let alpha = spec_aa_roughness_to_alpha(base.x);
    let filtered_alpha_sq = spec_aa_filter_alpha_sq(
        alpha * alpha,
        variance,
        SPEC_AA_KAPPA_MAX,
    );
    output[index] = vec4<f32>(
        spec_aa_geometric_specular_aa_roughness(
            base.x,
            ddx,
            ddy,
            SPEC_AA_SIGMA2,
            SPEC_AA_KAPPA_MAX,
        ),
        toksvig_roughness(base.x, normal.w),
        filtered_alpha_sq,
        toksvig_factor(normal.w, extra.x),
    );
}
"#;

#[derive(Clone, Copy)]
struct SpecularAaGpuCase {
    roughness: f32,
    ddx: [f32; 3],
    ddy: [f32; 3],
    average_normal_len: f32,
    shininess: f32,
}

fn compile_gpu_parity_wgsl() -> String {
    let mut cache = ShaderCache::new((), load_source);
    let module = shader_id(0x5052_4953_4d5f_5350_4543_4141_0011);
    cache.set_shader(
        module,
        Shader::from_wesl(
            include_str!("../../shaders/specular_aa.wesl"),
            "embedded://prism_render_scene/shaders/specular_aa.wesl",
        ),
    );
    let parity = shader_id(0x5052_4953_4d5f_5350_4543_4141_0012);
    cache.set_shader(
        parity,
        Shader::from_wesl(
            GPU_PARITY_SOURCE,
            "embedded://prism_render_scene/shading/specular_aa/gpu_parity.wesl",
        ),
    );
    (*cache
        .get(0, parity, &[])
        .unwrap_or_else(|error| panic!("specular-AA GPU parity shader failed to compile: {error}")))
    .clone()
}

fn parity_entry_point(wgsl: &str) -> String {
    wgsl.lines()
        .filter_map(|line| line.trim_start().strip_prefix("fn "))
        .filter_map(|rest| rest.split_once('(').map(|(name, _)| name))
        .find(|name| name.contains("specular_aa_gpu_parity"))
        .map(str::to_owned)
        .unwrap_or_else(|| panic!("compiled shader has no specular-AA parity entry point"))
}

/// Dispatches the production WESL helper module on a real device and compares
/// its readback against the backend-neutral specular-AA goldens. The input
/// layout deliberately carries both derivative variance and Toksvig inputs, so
/// the test covers the two production sources that are combined by resolve.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "headless CI needs an explicit skip diagnostic when no GPU exists"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the complete compile-dispatch-readback-parity path stays auditable in one test"
)]
fn specular_aa_gpu_readback_matches_cpu_goldens() {
    use bevy_platform::future::block_on;
    use prism_math::Vec3;
    use prism_render_shading::gi::specular_aa::normal_variance::filter_alpha_sq;
    use prism_render_shading::gi::specular_aa::{
        geometric_specular_aa_roughness, screen_space_variance, toksvig_factor, toksvig_roughness,
        DEFAULT_KAPPA_MAX, DEFAULT_SIGMA2,
    };
    use wgpu::util::{BufferInitDescriptor, DeviceExt};
    use wgpu::{
        BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
        CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
        DeviceDescriptor, Instance, InstanceDescriptor, InstanceFlags, MapMode,
        PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, RequestAdapterOptions,
        ShaderModuleDescriptor, ShaderSource, ShaderStages,
    };

    let cases = [
        SpecularAaGpuCase {
            roughness: 0.02,
            ddx: [0.20, -0.10, 0.03],
            ddy: [-0.04, 0.11, 0.07],
            average_normal_len: 0.92,
            shininess: 50.0,
        },
        SpecularAaGpuCase {
            roughness: 0.30,
            ddx: [0.0, 0.0, 0.0],
            ddy: [0.0, 0.0, 0.0],
            average_normal_len: 1.0,
            shininess: 100.0,
        },
        SpecularAaGpuCase {
            roughness: 0.80,
            ddx: [0.70, 0.40, 0.30],
            ddy: [-0.20, 0.50, 0.60],
            average_normal_len: 0.25,
            shininess: 10_000.0,
        },
        SpecularAaGpuCase {
            roughness: 1.0,
            ddx: [0.15, 0.02, -0.08],
            ddy: [0.04, -0.18, 0.10],
            average_normal_len: 0.5,
            shininess: 0.0,
        },
        SpecularAaGpuCase {
            roughness: 0.0,
            ddx: [0.45, 0.35, -0.20],
            ddy: [-0.31, 0.26, 0.41],
            average_normal_len: 0.0001,
            shininess: 1_000_000.0,
        },
        SpecularAaGpuCase {
            roughness: 0.55,
            ddx: [0.01, 0.02, 0.03],
            ddy: [0.04, 0.05, 0.06],
            average_normal_len: 0.73,
            shininess: 700.0,
        },
    ];

    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::METAL | Backends::VULKAN | Backends::DX12,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: BackendOptions::default(),
    });
    let Some(adapter) = block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()
    else {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        eprintln!("specular_aa_gpu_readback_matches_cpu_goldens: no adapter, skipping");
        return;
    };
    let Some((device, queue)) = block_on(adapter.request_device(&DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok() else {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU is set but request_device failed"
        );
        eprintln!("specular_aa_gpu_readback_matches_cpu_goldens: device unavailable, skipping");
        return;
    };

    let input_words: Vec<[f32; 4]> = cases
        .iter()
        .flat_map(|case| {
            [
                [case.roughness, case.ddx[0], case.ddx[1], case.ddx[2]],
                [
                    case.ddy[0],
                    case.ddy[1],
                    case.ddy[2],
                    case.average_normal_len,
                ],
                [case.shininess, 0.0, 0.0, 0.0],
            ]
        })
        .collect();
    let input = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("specular_aa_gpu_parity_input"),
        contents: bytemuck::cast_slice(&input_words),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    let output_size = (cases.len() * size_of::<[f32; 4]>()) as u64;
    let output = device.create_buffer(&BufferDescriptor {
        label: Some("specular_aa_gpu_parity_output"),
        size: output_size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("specular_aa_gpu_parity_readback"),
        size: output_size,
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
    let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("specular_aa_gpu_parity_bind_group_layout"),
        entries: &[storage(0, false), storage(1, true)],
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("specular_aa_gpu_parity_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let wgsl = compile_gpu_parity_wgsl();
    let entry = parity_entry_point(&wgsl);
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("specular_aa_gpu_parity_shader"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("specular_aa_gpu_parity_pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(&entry),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("specular_aa_gpu_parity_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("specular_aa_gpu_parity_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("specular_aa_gpu_parity_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(cases.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(PollType::wait_indefinitely());
    receiver
        .recv()
        .expect("GPU readback callback")
        .expect("GPU readback map");
    let data = slice.get_mapped_range().expect("mapped GPU readback range");
    let values: Vec<f32> = data
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 bytes")))
        .collect();

    const EPSILON: f32 = 3.0e-5;
    for (index, case) in cases.iter().enumerate() {
        let ddx = Vec3::from_array(case.ddx);
        let ddy = Vec3::from_array(case.ddy);
        let variance = screen_space_variance(ddx, ddy, DEFAULT_SIGMA2);
        let alpha = prism_render_shading::gi::spec_gi::ggx_lobe::roughness_to_alpha(case.roughness);
        let expected = [
            geometric_specular_aa_roughness(
                case.roughness,
                ddx,
                ddy,
                DEFAULT_SIGMA2,
                DEFAULT_KAPPA_MAX,
            ),
            toksvig_roughness(case.roughness, case.average_normal_len),
            filter_alpha_sq(alpha * alpha, variance, DEFAULT_KAPPA_MAX),
            toksvig_factor(case.average_normal_len, case.shininess),
        ];
        let actual = &values[index * 4..index * 4 + 4];
        for lane in 0..4 {
            assert!(
                (actual[lane] - expected[lane]).abs() <= EPSILON,
                "case {index} lane {lane}: GPU {} CPU {} (roughness {}, len {})",
                actual[lane],
                expected[lane],
                case.roughness,
                case.average_normal_len,
            );
        }
    }
    drop(data);
    readback.unmap();
}

#[test]
fn shading_resolve_routes_normal_variance_through_production_specular_aa() {
    let resolve = include_str!("../../shaders/shading_resolve.wesl");
    let import = resolve
        .find("import prism_render_scene::shaders::specular_aa::{")
        .expect("shading resolve must import the shared specular-AA module");
    let call = resolve
        .find("spec_aa_screen_space_variance(")
        .expect("shading resolve must call the shared variance helper");
    let publication = resolve
        .find("surface.normal_variance = shading_resolve_normal_variance(")
        .expect("shading resolve must publish the filtered normal variance");
    assert!(import < call && call < publication);
    assert!(resolve.contains("geometry.uv + uv_gradients[0]"));
    assert!(resolve.contains("geometry.uv + uv_gradients[1]"));
}
