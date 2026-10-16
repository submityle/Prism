use super::*;
use prism_mesh::deformation::{DeformedVertex, JointTransform};
use prism_mesh::{MorphDelta, MorphTarget, SkinInfluence, SkinningData};
use prism_render_driver_null::NullInstance;
use prism_render_driver_wgpu::WgpuInstance;
use rhi::{DeviceDescriptor, Extent3d, Instance, RequestAdapterOptions};

#[test]
fn cache_is_owned_send_sync_and_preflights_all_reported_limits() {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<DeformationCache>();
    let mesh = fixture(65);
    assert!(DeformationCache::buffer_sizes(&mesh, rhi::Limits::baseline()).is_ok());
    for limits in [
        rhi::Limits {
            max_compute_invocations_per_workgroup: 63,
            ..rhi::Limits::baseline()
        },
        rhi::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..rhi::Limits::baseline()
        },
        rhi::Limits {
            max_bind_groups: 0,
            ..rhi::Limits::baseline()
        },
        rhi::Limits {
            max_bindings_per_bind_group: 7,
            ..rhi::Limits::baseline()
        },
        rhi::Limits {
            max_uniform_buffer_binding_size: 15,
            ..rhi::Limits::baseline()
        },
    ] {
        assert_eq!(
            DeformationCache::buffer_sizes(&mesh, limits),
            Err(DeformationCacheError::ComputeLimit)
        );
    }
    assert!(matches!(
        DeformationCache::buffer_sizes(
            &mesh,
            rhi::Limits {
                max_storage_buffer_binding_size: 16,
                ..rhi::Limits::baseline()
            }
        ),
        Err(DeformationCacheError::BufferLimit { .. })
    ));
}

fn fixture(count: usize) -> DeformationMesh {
    fixture_with_profile(count, 8)
}

fn fixture_with_profile(count: usize, profile: usize) -> DeformationMesh {
    let positions: Vec<_> = (0..count).map(|i| [i as f32 * 0.001, 2.0, 3.0]).collect();
    let normals = vec![[1.0, 1.0, 0.0]; count];
    let dense =
        MorphTarget::new_dense("dense", vec![MorphDelta::position([0.5, -1.0, 0.0]); count])
            .unwrap();
    let indices: Vec<_> = (0..count as u32).step_by(7).collect();
    let sparse = MorphTarget::new_sparse(
        "sparse",
        count as u32,
        indices.clone(),
        vec![MorphDelta::position([0.0, 0.5, 2.0]); indices.len()],
    )
    .unwrap();
    let lists = (0..count)
        .map(|i| {
            if i % 11 == 0 {
                return vec![];
            }
            (0..profile as u32)
                .map(|joint_index| SkinInfluence {
                    joint_index,
                    weight: if i % 13 == 0 {
                        0.0
                    } else {
                        (joint_index + 1) as f32
                    },
                })
                .collect()
        })
        .collect();
    let skin = SkinningData::new(count as u32, 8, lists).unwrap();
    DeformationMesh::new(
        &positions,
        Some(&normals),
        &[dense, sparse],
        Some(&skin),
        profile,
    )
    .unwrap()
}

fn joints() -> Vec<JointTransform> {
    (0..8)
        .map(|i| {
            JointTransform::new([
                [0.0, -2.0, 0.0, i as f32 * 0.1],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.5, -1.0],
            ])
            .unwrap()
        })
        .collect()
}

fn assert_close(actual: &[u8], expected: &[DeformedVertex]) {
    assert_eq!(actual.len(), size_of_val(expected));
    for (i, (&a, &b)) in bytemuck::cast_slice::<u8, f32>(actual)
        .iter()
        .zip(bytemuck::cast_slice::<DeformedVertex, f32>(expected))
        .enumerate()
    {
        assert!(
            (a - b).abs() <= 3e-5 * b.abs().max(1.0),
            "component {i}: GPU {a}, CPU {b}"
        );
    }
}

#[test]
fn null_cache_releases_every_owned_resource_and_rejects_bad_frames() {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .unwrap();
    let (device, queue) = instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .unwrap();
    let device = Arc::new(device);
    let before = device.live_resource_count();
    {
        let mut cache = DeformationCache::new(device.clone(), &queue, fixture(65)).unwrap();
        assert!(cache.allocated_bytes() > 65 * 64);
        assert_eq!(cache.output(), None);
        let joints = joints();
        let pose = DeformationPose {
            morph_weights: &[1.0, 0.5],
            joints: &joints,
            corrections: None,
        };
        let first = cache.submit_pose(&queue, 1, pose).unwrap();
        assert!(!first.history_valid);
        assert_eq!(first.previous, first.current);
        assert_eq!(
            cache.submit_pose(&queue, 1, pose),
            Err(DeformationCacheError::FrameNotIncreasing)
        );
        assert_eq!(cache.output(), Some(first));
        let second = cache.submit_pose(&queue, 2, pose).unwrap();
        assert!(second.history_valid);
        assert_eq!(second.previous, first.current);
        assert_ne!(second.current, second.previous);
        cache.invalidate_history();
        assert!(!cache.output().unwrap().history_valid);
        assert!(!cache.submit_pose(&queue, 3, pose).unwrap().history_valid);
        assert!(cache.submit_pose(&queue, 4, pose).unwrap().history_valid);
        assert!(!cache.submit_pose(&queue, 6, pose).unwrap().history_valid);
    }
    assert_eq!(device.live_resource_count(), before);
    let layout = DeformationOutput::vertex_layout(0, 1);
    assert_eq!(layout.array_stride, 32);
    assert!(layout.is_within_stride());
}

#[test]
fn deformation_source_shares_immutable_gpu_resources_between_instances() {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .unwrap();
    let (device, queue) = instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .unwrap();
    let device = Arc::new(device);
    let baseline = device.live_resource_count();
    let source = Arc::new(DeformationSource::new(device.clone(), &queue, fixture(65)).unwrap());
    let source_resources = device.live_resource_count() - baseline;
    assert_eq!(
        source_resources, 7,
        "three streams plus four pipeline objects"
    );

    let mut first = DeformationCache::from_source(source.clone(), &queue).unwrap();
    let mut second = DeformationCache::from_source(source.clone(), &queue).unwrap();
    assert!(Arc::ptr_eq(first.source(), second.source()));
    assert_eq!(first.source().allocated_bytes(), source.allocated_bytes());
    assert_eq!(
        device.live_resource_count(),
        baseline + source_resources + 16,
        "each instance owns six buffers and two bind groups; immutable streams are shared"
    );

    let joints = joints();
    let pose = DeformationPose {
        morph_weights: &[0.25, -0.5],
        joints: &joints,
        corrections: None,
    };
    let first_output = first.submit_pose(&queue, 1, pose).unwrap();
    let second_output = second.submit_pose(&queue, 1, pose).unwrap();
    assert_eq!(
        first_output.vertex_count,
        second.source().mesh().vertex_count()
    );
    assert_ne!(first_output.current, second_output.current);
    assert_ne!(first_output.previous, second_output.previous);

    drop(first);
    assert_eq!(
        device.live_resource_count(),
        baseline + source_resources + 8,
        "dropping one instance must not release the shared source"
    );
    drop(second);
    assert_eq!(
        device.live_resource_count(),
        baseline + source_resources,
        "the explicit source owner keeps immutable resources alive"
    );
    drop(source);
    assert_eq!(device.live_resource_count(), baseline);
}

#[test]
fn batch_deformation_matches_cpu_on_a_real_gpu_and_retains_history() {
    let Ok(runtime) =
        prism_render_runtime::RenderRuntime::<WgpuInstance>::wgpu(Extent3d::new_2d(8, 8))
    else {
        assert!(
            !crate::tests::require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found"
        );
        return;
    };
    let queue = runtime.queue();
    for (count, profile) in [(1, 4), (65, 4), (65, 8), (50_000, 8)] {
        let mesh = fixture_with_profile(count, profile);
        let joints = joints();
        let corrections = vec![
            DeformationCorrection {
                position: [0.25, 0.0, -0.5, 0.0],
                normal: [0.0, 0.25, 0.0, 0.0]
            };
            count
        ];
        let pose = DeformationPose {
            morph_weights: &[0.75, -0.25],
            joints: &joints,
            corrections: Some(&corrections),
        };
        let expected = mesh.evaluate(pose).unwrap();
        let mut cache =
            DeformationCache::new(runtime.device_handle(), queue, mesh.clone()).unwrap();
        let bytes = count as u64 * 32;
        let first = cache.submit_pose(queue, 10, pose).unwrap();
        assert!(runtime.wait_idle());
        assert_close(&queue.read_buffer(first.current, bytes), &expected);
        assert!(!first.history_valid);

        // A rejected newer pose must leave current/history buffers intact.
        let invalid = DeformationPose {
            morph_weights: &[f32::NAN, 0.0],
            ..pose
        };
        assert!(matches!(
            cache.submit_pose(queue, 11, invalid),
            Err(DeformationCacheError::Input(_))
        ));
        assert_eq!(cache.output(), Some(first));
        assert_close(&queue.read_buffer(first.current, bytes), &expected);
        let overflow = DeformationPose {
            morph_weights: &[f32::MAX, 0.0],
            ..pose
        };
        assert_eq!(
            cache.submit_pose(queue, 11, overflow),
            Err(DeformationCacheError::Input(
                DeformationError::ArithmeticRange
            ))
        );
        assert_eq!(cache.output(), Some(first));

        // Turning correction OFF must clear its persistent buffer.
        let next_pose = DeformationPose {
            morph_weights: &[-0.25, 0.5],
            corrections: None,
            ..pose
        };
        let second_expected = mesh.evaluate(next_pose).unwrap();
        let second = cache.submit_pose(queue, 11, next_pose).unwrap();
        assert!(runtime.wait_idle());
        assert!(second.history_valid);
        assert_eq!(second.previous, first.current);
        assert_close(&queue.read_buffer(second.previous, bytes), &expected);
        assert_close(&queue.read_buffer(second.current, bytes), &second_expected);
        let third = cache.submit_pose(queue, 12, pose).unwrap();
        assert!(runtime.wait_idle());
        assert_eq!(
            third.current, first.current,
            "two outputs reused, no reallocation"
        );
        assert_close(&queue.read_buffer(third.previous, bytes), &second_expected);
        assert_close(&queue.read_buffer(third.current, bytes), &expected);
        cache.invalidate_history();
        let teleported = cache.submit_pose(queue, 13, pose).unwrap();
        assert!(!teleported.history_valid);
        assert_eq!(teleported.current, teleported.previous);
        assert!(!cache.submit_pose(queue, 15, pose).unwrap().history_valid);
        assert!(runtime.wait_idle());
    }
    let mesh = DeformationMesh::new(&[], None, &[], None, 4).unwrap();
    let mut cache = DeformationCache::new(runtime.device_handle(), queue, mesh).unwrap();
    let view = cache
        .submit_pose(
            queue,
            1,
            DeformationPose {
                morph_weights: &[],
                joints: &[],
                corrections: None,
            },
        )
        .unwrap();
    assert_eq!(view.vertex_count, 0);

    // Morph-only meshes have neither skin input nor authored normals.
    let mesh = DeformationMesh::new(
        &[[1.0, 2.0, 3.0]; 3],
        None,
        &[MorphTarget::new_dense("only", vec![MorphDelta::position([2.0, 0.0, 0.0]); 3]).unwrap()],
        None,
        4,
    )
    .unwrap();
    let pose = DeformationPose {
        morph_weights: &[-0.5],
        joints: &[],
        corrections: None,
    };
    let expected = mesh.evaluate(pose).unwrap();
    let mut cache = DeformationCache::new(runtime.device_handle(), queue, mesh).unwrap();
    let output = cache.submit_pose(queue, 1, pose).unwrap();
    assert!(runtime.wait_idle());
    assert_close(&queue.read_buffer(output.current, 3 * 32), &expected);
}

#[test]
fn deformed_current_and_previous_are_vertex_inputs_for_a_real_gpu_draw() {
    use prism_render_graph::{
        BufferDesc, ImportedBuffer, ImportedTexture, PassFlags, RenderGraph, TextureDesc,
    };
    let extent = Extent3d::new_2d(16, 16);
    let Ok(runtime) = prism_render_runtime::RenderRuntime::<WgpuInstance>::wgpu(extent) else {
        assert!(
            !crate::tests::require_gpu(),
            "GPU required for deformation draw"
        );
        return;
    };
    let device = runtime.device();
    let queue = runtime.queue();
    let positions = [
        [-0.75, -0.5, 0.5],
        [-0.25, -0.5, 0.5],
        [-0.25, 0.5, 0.5],
        [-0.75, -0.5, 0.5],
        [-0.25, 0.5, 0.5],
        [-0.75, 0.5, 0.5],
    ];
    let skin = SkinningData::new(
        6,
        4,
        vec![
            (0..4)
                .map(|joint_index| SkinInfluence {
                    joint_index,
                    weight: (joint_index + 1) as f32
                })
                .collect();
            6
        ],
    )
    .unwrap();
    let mesh = DeformationMesh::new(
        &positions,
        Some(&[[0.0, 0.0, 2.0]; 6]),
        &[
            MorphTarget::new_dense("shift", vec![MorphDelta::position([1.0, 0.0, 0.0]); 6])
                .unwrap(),
        ],
        Some(&skin),
        4,
    )
    .unwrap();
    let mut cache = DeformationCache::new(runtime.device_handle(), queue, mesh).unwrap();
    let identity = [JointTransform::identity(); 4];
    cache
        .submit_pose(
            queue,
            1,
            DeformationPose {
                morph_weights: &[0.0],
                joints: &identity,
                corrections: None,
            },
        )
        .unwrap();
    let shifted = [JointTransform::new([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    .unwrap(); 4];
    let output = cache
        .submit_pose(
            queue,
            2,
            DeformationPose {
                morph_weights: &[0.5],
                joints: &shifted,
                corrections: None,
            },
        )
        .unwrap();
    assert!(output.history_valid);
    // No CPU readback/copy/repacking between compute and this render pass.
    let shader = device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(
        "struct VsOut { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32> }
         @vertex fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>, @location(2) previous: vec3<f32>) -> VsOut {
             var o: VsOut; o.position = vec4<f32>(p, 1.0); o.color = vec3<f32>(p.x - previous.x, n.z, 0.0); return o;
         }
         @fragment fn fs(o: VsOut) -> @location(0) vec4<f32> { return vec4<f32>(o.color, 1.0); }"
    ));
    let pipeline = device.create_render_pipeline(&rhi::RenderPipelineDescriptor {
        label: Some("mesh-deformed-velocity".into()),
        layout: None,
        vertex: rhi::VertexState {
            module: shader,
            entry_point: "vs".into(),
            buffers: vec![
                DeformationOutput::vertex_layout(0, 1),
                rhi::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: rhi::VertexStepMode::Vertex,
                    attributes: vec![rhi::VertexAttribute {
                        format: rhi::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 2,
                    }],
                },
            ],
        },
        primitive: rhi::PrimitiveState::default(),
        depth_stencil: None,
        multisample: rhi::MultisampleState::default(),
        fragment: Some(rhi::FragmentState {
            module: shader,
            entry_point: "fs".into(),
            targets: vec![Some(rhi::ColorTargetState::opaque(
                rhi::TextureFormat::Rgba8Unorm,
            ))],
        }),
    });
    let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
        extent,
        rhi::TextureFormat::Rgba8Unorm,
        rhi::TextureUsages::RENDER_ATTACHMENT | rhi::TextureUsages::COPY_SRC,
    ));
    let view = device.create_texture_view(texture, &Default::default());
    let mut graph = RenderGraph::new(extent);
    let target = graph.import_texture(
        "deformed-target",
        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm).with_usage(rhi::TextureUsages::COPY_SRC),
        ImportedTexture {
            texture,
            default_view: view,
            entry_state: rhi::TextureState::initial(),
            exit_state: rhi::TextureState::copy_src(),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    let current = graph.import_buffer(
        "deformed-current",
        BufferDesc::new(6 * 32, rhi::BufferUsages::VERTEX),
        ImportedBuffer {
            buffer: output.current,
            entry_state: rhi::BufferState::initial(),
        },
    );
    let previous = graph.import_buffer(
        "deformed-previous",
        BufferDesc::new(6 * 32, rhi::BufferUsages::VERTEX),
        ImportedBuffer {
            buffer: output.previous,
            entry_state: rhi::BufferState::initial(),
        },
    );
    graph.add_raster_pass("deformed-motion", PassFlags::SIDE_EFFECT, move |builder| {
        let _ = builder.color_attachment(target);
        let _ = builder.read_vertex(current);
        let _ = builder.read_vertex(previous);
        move |ctx| {
            ctx.set_render_pipeline(pipeline);
            ctx.set_vertex_buffer(0, ctx.buffer(current), 0);
            ctx.set_vertex_buffer(1, ctx.buffer(previous), 0);
            ctx.draw(0..6, 0..1);
        }
    });
    let recorded = graph.execute(device, queue).unwrap();
    assert!(runtime.wait_idle());
    recorded.release(device);
    let pixels = queue.read_texture_rgba8(texture, 16, 16);
    let pixel = |x: usize, y: usize| &pixels[(y * 16 + x) * 4..(y * 16 + x + 1) * 4];
    assert_eq!(
        pixel(12, 8),
        [255, 255, 0, 255],
        "new position, previous position, and normalized normal all reach the shader"
    );
    assert_eq!(
        pixel(4, 8),
        [0, 0, 0, 255],
        "bind-pose position is not rendered"
    );
    device.destroy_render_pipeline(pipeline);
    device.destroy_shader_module(shader);
    device.destroy_texture_view(view);
    device.destroy_texture(texture);
}
