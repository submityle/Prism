//! Dispatches the production pipelines and compares every output channel with
//! the authoritative CPU compositor, including separately typed auxiliary data.

use bevy_platform::future::block_on;
use bevy_render::renderer::RenderDevice;
use core::mem::size_of;
use prism_render_material::{BlendDomain, ChannelMask, MaterialRenderClass};
use prism_render_scene::response_gpu::{
    GpuAuxiliaryMerge, GpuCompositionJob, GpuResponseOutput, ResponseCompositionPipelines,
};
use prism_render_shading::response_output::*;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        display: None,
        backend_options: wgpu::BackendOptions::default(),
    });
    let adapter = match block_on(instance.request_adapter(&Default::default())) {
        Ok(adapter) => adapter,
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "required GPU adapter unavailable: {error}"
            );
            eprintln!("response GPU parity skipped: {error}");
            return None;
        }
    };
    eprintln!("response GPU adapter: {:?}", adapter.get_info());
    let limits = adapter.limits();
    let required_auxiliary_bytes = 14_520_u64 * size_of::<GpuAuxiliaryMerge>() as u64;
    let capability_error = if limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_storage_buffer_binding_size < required_auxiliary_bytes
        || limits.max_buffer_size < required_auxiliary_bytes
    {
        Some(format!(
            "response composition needs workgroup x64 and {required_auxiliary_bytes} byte storage buffers; adapter limits are {limits:?}"
        ))
    } else {
        None
    };
    if let Some(error) = capability_error {
        assert!(
            std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
            "PRISM_REQUIRE_GPU set: response GPU capability gate failed: {error}"
        );
        eprintln!("response GPU skipped: {error}");
        return None;
    }
    match block_on(adapter.request_device(&Default::default())) {
        Ok(device_queue) => Some(device_queue),
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "PRISM_REQUIRE_GPU set: response GPU device request failed: {error}"
            );
            eprintln!("response GPU skipped: device request failed: {error}");
            None
        }
    }
}

fn upload<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn read<T: bytemuck::Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<T> {
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.expect("readback map"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU completion");
    let view = buffer.slice(..).get_mapped_range().expect("mapped range");
    let result = bytemuck::cast_slice::<u8, T>(&view).to_vec();
    drop(view);
    buffer.unmap();
    result
}

fn fixture() -> (ResponseOutput, ResponseOutput, AuxiliaryMergeRules) {
    let mut left = ResponseOutput {
        direct: [0.0, 2.0, 19.0],
        indirect_diffuse: [7.0, 0.25, 0.75],
        reflection: [0.125, 0.5, 13.0],
        transmission: [1.0, 17.0, 0.0],
        emission: [23.0, 3.0, 0.625],
        alpha: 0.2,
        auxiliary: Auxiliary::default(),
    };
    let mut right = ResponseOutput {
        direct: [11.0, 5.0, 0.0],
        indirect_diffuse: [0.5, 3.0, 29.0],
        reflection: [31.0, 2.0, 0.375],
        transmission: [5.0, 0.25, 37.0],
        emission: [0.0, 41.0, 0.25],
        alpha: 0.8,
        auxiliary: Auxiliary::default(),
    };
    let mut rules = AuxiliaryMergeRules::default();
    for (name, a, b, rule) in [
        (
            "scalar_lerp",
            AuxiliaryValue::Scalar(0.125),
            AuxiliaryValue::Scalar(9.0),
            AuxiliaryMergeRule::Linear,
        ),
        (
            "vector_lerp",
            AuxiliaryValue::Vector3([2.0, 0.0, -1.0]),
            AuxiliaryValue::Vector3([0.5, 7.0, 3.0]),
            AuxiliaryMergeRule::Linear,
        ),
        (
            "vector_min",
            AuxiliaryValue::Vector3([2.0, -1.0, 0.0]),
            AuxiliaryValue::Vector3([3.0, 0.0, -2.0]),
            AuxiliaryMergeRule::Min,
        ),
        (
            "scalar_max",
            AuxiliaryValue::Scalar(-2.0),
            AuxiliaryValue::Scalar(8.0),
            AuxiliaryMergeRule::Max,
        ),
        (
            "unsigned_min",
            AuxiliaryValue::Unsigned(u32::MAX),
            AuxiliaryValue::Unsigned(17),
            AuxiliaryMergeRule::Min,
        ),
        (
            "unsigned_max",
            AuxiliaryValue::Unsigned(33),
            AuxiliaryValue::Unsigned(u32::MAX),
            AuxiliaryMergeRule::Max,
        ),
        (
            "flags",
            AuxiliaryValue::Flags(0x80000001),
            AuxiliaryValue::Flags(0x40000002),
            AuxiliaryMergeRule::BitOr,
        ),
        (
            "equal",
            AuxiliaryValue::Unsigned(42),
            AuxiliaryValue::Unsigned(42),
            AuxiliaryMergeRule::RequireEqual,
        ),
        (
            "left",
            AuxiliaryValue::Flags(23),
            AuxiliaryValue::Flags(27),
            AuxiliaryMergeRule::Left,
        ),
        (
            "right",
            AuxiliaryValue::Unsigned(2),
            AuxiliaryValue::Unsigned(3),
            AuxiliaryMergeRule::Right,
        ),
    ] {
        left.auxiliary.fields.insert(name.into(), a);
        right.auxiliary.fields.insert(name.into(), b);
        rules.fields.insert(name.into(), rule);
    }
    rules
        .fields
        .insert("shared".into(), AuxiliaryMergeRule::Shared);
    rules
        .shared
        .fields
        .insert("shared".into(), AuxiliaryValue::Vector3([0.0, 1.0, 0.0]));
    (left, right, rules)
}

fn near(a: f32, b: f32) {
    assert!(
        (a - b).abs() <= 2e-6 * b.abs().max(1.0),
        "GPU {a} != CPU {b}"
    );
}

#[test]
fn production_composition_gpu_matches_cpu_all_domains_alpha_and_auxiliary() {
    let Some((device, queue)) = device() else {
        return;
    };
    let pipelines = ResponseCompositionPipelines::new(&RenderDevice::from(device.clone()));
    let (left, right, rules) = fixture();
    let lighting = LightingContext {
        direct: &[],
        indirect_diffuse_irradiance: None,
        reflection_radiance: None,
        transmission_radiance: None,
        shared_terms: &(),
    };
    let a = lighting.bind(left.clone()).unwrap();
    let b = lighting.bind(right.clone()).unwrap();
    let mut jobs = Vec::new();
    let mut expected = Vec::new();
    let domains = [BlendDomain::FullRadiance, BlendDomain::DirectOnly]
        .into_iter()
        .chain((1..=31).map(|mask| BlendDomain::CustomChannelMask(ChannelMask(mask))));
    for domain in domains {
        for alpha_rule in [
            AlphaMergeRule::Shared,
            AlphaMergeRule::Left,
            AlphaMergeRule::Right,
            AlphaMergeRule::Min,
            AlphaMergeRule::Max,
        ] {
            for mask in [
                0.0,
                f32::EPSILON,
                0.125,
                0.25,
                0.5,
                0.875,
                1.0 - f32::EPSILON,
                1.0,
            ] {
                let options = CompositionOptions {
                    mask,
                    domain,
                    alpha_rule,
                    shared_alpha: SharedSurfaceAlpha::new(0.625, MaterialRenderClass::Opaque)
                        .unwrap(),
                    auxiliary_rules: &rules,
                };
                jobs.push(GpuCompositionJob::new(&left, &right, options).unwrap());
                expected.push(blend_responses(&a, &b, options).unwrap());
            }
        }
    }
    let left_buffer = upload(
        &device,
        "left responses",
        &jobs.iter().map(|j| j.left).collect::<Vec<_>>(),
    );
    let right_buffer = upload(
        &device,
        "right responses",
        &jobs.iter().map(|j| j.right).collect::<Vec<_>>(),
    );
    let controls = upload(
        &device,
        "composition controls",
        &jobs.iter().map(|j| j.control).collect::<Vec<_>>(),
    );
    let auxiliary = upload(
        &device,
        "typed auxiliary",
        &jobs
            .iter()
            .flat_map(|j| j.auxiliary.iter().copied())
            .collect::<Vec<_>>(),
    );
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("production radiance bindings"),
        layout: &pipelines.radiance.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: left_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: right_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: controls.as_entire_binding(),
            },
        ],
    });
    let aux_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("production auxiliary bindings"),
        layout: &pipelines.auxiliary.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 3,
            resource: auxiliary.as_entire_binding(),
        }],
    });
    let staging = |size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("response parity readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    };
    let radiance_stage = staging(left_buffer.size());
    let auxiliary_stage = staging(auxiliary.size());
    let start = std::time::Instant::now();
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("response composition parity"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.radiance);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((jobs.len() as u32).div_ceil(64), 1, 1);
        pass.set_pipeline(&pipelines.auxiliary);
        pass.set_bind_group(0, &aux_bind, &[]);
        pass.dispatch_workgroups((auxiliary.size() / 64).div_ceil(64) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&left_buffer, 0, &radiance_stage, 0, left_buffer.size());
    encoder.copy_buffer_to_buffer(&auxiliary, 0, &auxiliary_stage, 0, auxiliary.size());
    queue.submit([encoder.finish()]);
    let radiance = read::<GpuResponseOutput>(&device, &radiance_stage);
    let auxiliary = read::<GpuAuxiliaryMerge>(&device, &auxiliary_stage);
    eprintln!(
        "{} cases, {} auxiliary jobs, host submit+readback {:?}, radiance {} bytes",
        jobs.len(),
        auxiliary.len(),
        start.elapsed(),
        left_buffer.size() + right_buffer.size()
    );
    let mut offset = 0;
    for ((job, actual), expected) in jobs.iter().zip(radiance).zip(expected) {
        let aux = job
            .decode_auxiliary(&auxiliary[offset..offset + job.auxiliary.len()])
            .unwrap();
        offset += job.auxiliary.len();
        let actual = actual.unpack(aux);
        for (a, b) in [
            actual.direct,
            actual.indirect_diffuse,
            actual.reflection,
            actual.transmission,
            actual.emission,
        ]
        .into_iter()
        .flatten()
        .zip(
            [
                expected.direct,
                expected.indirect_diffuse,
                expected.reflection,
                expected.transmission,
                expected.emission,
            ]
            .into_iter()
            .flatten(),
        ) {
            near(a, b);
        }
        assert_eq!(actual.alpha, expected.alpha);
        for (name, value) in &actual.auxiliary.fields {
            match (value, &expected.auxiliary.fields[name]) {
                (AuxiliaryValue::Scalar(a), AuxiliaryValue::Scalar(b)) => near(*a, *b),
                (AuxiliaryValue::Vector3(a), AuxiliaryValue::Vector3(b)) => {
                    for (a, b) in a.iter().zip(b) {
                        near(*a, *b);
                    }
                }
                (a, b) => assert_eq!(a, b),
            }
        }
    }
}

#[test]
fn packing_rejects_invalid_inputs_and_auxiliary_loss() {
    let (left, right, mut rules) = fixture();
    fn options(
        mask: f32,
        domain: BlendDomain,
        rules: &AuxiliaryMergeRules,
    ) -> CompositionOptions<'_> {
        CompositionOptions {
            mask,
            domain,
            alpha_rule: AlphaMergeRule::Shared,
            shared_alpha: SharedSurfaceAlpha::new(0.5, MaterialRenderClass::Opaque).unwrap(),
            auxiliary_rules: rules,
        }
    }
    for mask in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
        assert!(GpuCompositionJob::new(
            &left,
            &right,
            options(mask, BlendDomain::FullRadiance, &rules)
        )
        .is_err());
    }
    for bits in [0, 32, 64, u32::MAX] {
        assert!(GpuCompositionJob::new(
            &left,
            &right,
            options(
                0.5,
                BlendDomain::CustomChannelMask(ChannelMask(bits)),
                &rules
            )
        )
        .is_err());
    }
    rules.fields.remove("flags");
    assert!(matches!(
        GpuCompositionJob::new(
            &left,
            &right,
            options(0.5, BlendDomain::FullRadiance, &rules)
        ),
        Err(CompositionError::MissingAuxiliaryRule(_))
    ));
}
