//! Strict production Decal GPU test.
//!
//! This exercises the embedded production shader, production pipeline layout,
//! authenticated domain bindings, SceneFabric lowering/recording, dispatch,
//! and a direct staging readback. Software adapters are never accepted as a
//! pass; `PRISM_REQUIRE_GPU=1` turns the same condition into a failure.

use bevy_asset::Handle;
use bevy_platform::future::block_on;
use bevy_render::{
    render_resource::{Buffer, PipelineCache},
    renderer::{RenderAdapter, RenderDevice, RenderQueue},
};
use bevy_shader::Shader;
use prism_math::{Mat4, Vec3};
use prism_render_architecture::{
    motion::MotionSource,
    render_fabric::{DecalInstance, DecalMaterial, DecalReceiver, DomainObjectHandle},
};
use prism_render_driver::{Adapter, Extent3d};
use prism_render_driver_wgpu::{require_hardware_gpu, AdapterEvidence, WgpuAdapter, WgpuInstance};
use prism_render_shading::gi::decal::DecalProjector;
use std::sync::{Arc, Mutex};

use crate::{
    decal::{
        bind_groups::create_decal_bind_group,
        pipeline::create_decal_resolve_pipeline,
        recorder::{
            build_decal_resource_publication, build_decal_resource_publication_with_extent,
            build_decal_resource_publications_from_owner, prepare_decal_resolve,
            prepare_decal_resolves, DecalPreparedResourcePublication, DecalResolvePublication,
        },
    },
    fabric_domains::{
        build_decal_domain_items, gpu::DomainRecordReadyResource, owners::DecalRenderWorldOwner,
        DecalDomainInput, DomainFrameIdentity,
    },
    gpu::rhi::SceneRhi,
    shading::composition::scene_fabric,
};

struct Host {
    _instance: WgpuInstance,
    adapter: RenderAdapter,
    device: RenderDevice,
    queue: RenderQueue,
}

fn publish_test_resources(
    rhi: &SceneRhi,
    frame: u64,
    identity: DomainFrameIdentity,
    items: &[prism_render_architecture::render_fabric::FrameRenderItem],
    projector: &DecalProjector,
    material: &DecalMaterial,
    receiver: &DecalReceiver,
) -> DecalPreparedResourcePublication {
    build_decal_resource_publication(rhi, frame, identity, items, projector, material, receiver)
        .expect("production Decal resource publication")
}

fn host_or_skip() -> Option<Host> {
    let instance = WgpuInstance::new();
    let Some(adapter) = instance.request_adapter(wgpu::PowerPreference::HighPerformance, false)
    else {
        assert!(!require_hardware_gpu(), "PRISM_REQUIRE_GPU=1: no adapter");
        eprintln!("SKIP_UNVERIFIED Decal production GPU test: no adapter");
        return None;
    };
    let evidence = AdapterEvidence::from_info(WgpuAdapter::new(adapter.clone()).info());
    if !evidence.is_hardware() {
        assert!(
            !require_hardware_gpu(),
            "PRISM_REQUIRE_GPU=1 requires physical hardware: {evidence}"
        );
        eprintln!("SKIP_UNVERIFIED Decal production GPU test: {evidence}");
        return None;
    }
    eprintln!("Decal production REAL HARDWARE GPU: {evidence}");
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("physical Decal device");
    Some(Host {
        _instance: instance,
        adapter: RenderAdapter::new(adapter),
        device: RenderDevice::new(device),
        queue: RenderQueue::new(queue),
    })
}

fn readback(host: &Host, buffer: &Buffer) -> [f32; 12] {
    let device = host.device.wgpu_device();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("decal parity staging"),
        size: 48,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, 48);
    host.queue.submit([encoder.finish()]);
    let status = Arc::new(Mutex::new(None));
    let callback_status = Arc::clone(&status);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            *callback_status.lock().unwrap() = Some(result);
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert_eq!(status.lock().unwrap().take(), Some(Ok(())));
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let values = core::array::from_fn(|index| {
        f32::from_le_bytes(mapped[index * 4..index * 4 + 4].try_into().unwrap())
    });
    drop(mapped);
    staging.unmap();
    values
}

#[test]
fn decal_scene_fabric_gpu_matches_cpu_and_preserves_miss_sentinel_contract() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let rhi = SceneRhi::from_bevy(&host.adapter, &host.device, &host.queue);
    let scene = scene_fabric(
        prism_render_architecture::backend::capability::CapabilitySet::WEBGPU_BASELINE,
    );

    let frame = 41;
    let identity = DomainFrameIdentity {
        object: DomainObjectHandle::new(9, 1),
        history_key: 3,
        motion_source: MotionSource::Rigid,
    };
    let material = DecalMaterial {
        albedo: [0.8, 0.2, 0.1, 1.0],
        normal: [0.0, 0.0, 1.0, 0.0],
        scalar_blend: [0.25, 0.75, 0.8, 0.0],
    };
    let mut cache = PipelineCache::new(host.device.clone(), true);
    let shader = Handle::<Shader>::default();
    cache.set_shader(
        shader.id(),
        Shader::from_wesl(
            include_str!("../shaders/decal_resolve.wesl"),
            "embedded://prism_render_scene/shaders/decal_resolve.wesl",
        ),
    );
    let pipeline = create_decal_resolve_pipeline(&host.device, &cache, shader);
    cache.process_queue();
    let native_pipeline = cache
        .get_compute_pipeline(pipeline.resolve())
        .expect("Decal production pipeline must compile");

    let run_case = |frame: u64, projector: DecalProjector, receiver: DecalReceiver| {
        let inputs = [DecalDomainInput {
            identity,
            projector: &projector,
            material: &material,
            receiver: &receiver,
        }];
        let items = build_decal_domain_items(&inputs).unwrap();
        let instance = DecalInstance::new(
            projector.world_to_decal.to_cols_array(),
            projector.projection_dir.to_array(),
            projector.angle_fade_start,
            projector.angle_fade_end,
            projector.edge_fade,
            0,
        );
        let expected = prism_render_architecture::render_fabric::resolve_decal(
            &receiver, &instance, &material,
        );
        let publication = publish_test_resources(
            &rhi, frame, identity, &items, &projector, &material, &receiver,
        );
        let output_buffer = publication
            .domain_view
            .resources()
            .last()
            .unwrap()
            .record_ready()
            .unwrap();
        let output_buffer = match output_buffer {
            DomainRecordReadyResource::Buffer { owner, .. } => owner.buffer().clone(),
            DomainRecordReadyResource::Texture { .. } => panic!("Decal output must be a buffer"),
        };
        let resource_ids: Vec<_> = publication
            .domain_view
            .frame_bindings()
            .filter_map(|binding| match binding.physical {
                prism_render_architecture::render_fabric::PhysicalResource::Buffer {
                    id, ..
                } => Some(id),
                prism_render_architecture::render_fabric::PhysicalResource::Texture(_) => None,
            })
            .collect();
        let view = publication.domain_view;
        let runtime = prepare_decal_resolve(
            &scene,
            &rhi,
            &host.device,
            &pipeline,
            native_pipeline,
            frame,
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            identity,
            &items,
            &view,
            publication.workgroups,
        )
        .unwrap();
        let submission = runtime.submit_tracked(&rhi);
        for id in &resource_ids {
            assert!(rhi.device().wgpu_buffer(*id).size() > 0);
        }
        submission
            .wait_for_completion(&rhi)
            .expect("Decal SceneFabric submission completed");
        let actual = readback(&host, &output_buffer);
        for (actual, expected) in actual[..4].iter().zip(expected.albedo) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        for (actual, expected) in actual[4..8].iter().zip(expected.normal) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        for (actual, expected) in actual[8..12].iter().zip(expected.scalar_coverage) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        drop(view);
        for id in &resource_ids {
            let retired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = rhi.device().wgpu_buffer(*id);
            }));
            assert!(retired.is_err(), "Decal resource lease {id:?} must retire");
        }
        actual
    };

    let front = run_case(
        frame,
        DecalProjector::new(
            Mat4::from_translation(Vec3::new(-0.25, 0.0, 0.0)),
            0.2,
            0.0,
            0.5,
        ),
        DecalReceiver {
            world_position: [0.25, 0.0, 0.0, 1.0],
            world_normal: [0.0, 0.0, -1.0, 0.0],
            albedo: [0.1, 0.2, 0.3, 1.0],
            scalar: [0.6, 0.1, 7.0, 0.0],
        },
    );
    assert_ne!(front, [0.0; 12]);

    let miss = run_case(
        frame + 1,
        DecalProjector::new(Mat4::IDENTITY, 0.2, 0.0, 0.5),
        DecalReceiver {
            world_position: [2.0, 0.0, 0.0, 1.0],
            world_normal: [0.0, 0.0, -1.0, 0.0],
            albedo: [0.1, 0.2, 0.3, 1.0],
            scalar: [0.6, 0.1, 7.0, 0.0],
        },
    );
    assert_eq!(
        miss,
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, -1.0, 0.0, 0.6, 0.1, 0.0, 0.0]
    );

    let back_face = run_case(
        frame + 2,
        DecalProjector::new(Mat4::IDENTITY, 0.2, 0.0, 0.5),
        DecalReceiver {
            world_position: [0.0, 0.0, 0.0, 1.0],
            world_normal: [0.0, 0.0, 1.0, 0.0],
            albedo: [0.1, 0.2, 0.3, 1.0],
            scalar: [0.6, 0.1, 7.0, 0.0],
        },
    );
    assert_eq!(
        back_face,
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 1.0, 0.0, 0.6, 0.1, 0.0, 0.0]
    );
}

#[test]
fn decal_multi_object_fanout_gpu_matches_cpu_and_keeps_identity_scoped_buffers() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let rhi = SceneRhi::from_bevy(&host.adapter, &host.device, &host.queue);
    let scene = scene_fabric(
        prism_render_architecture::backend::capability::CapabilitySet::WEBGPU_BASELINE,
    );
    let frame = 77;
    let mut cache = PipelineCache::new(host.device.clone(), true);
    let shader = Handle::<Shader>::default();
    cache.set_shader(
        shader.id(),
        Shader::from_wesl(
            include_str!("../shaders/decal_resolve.wesl"),
            "embedded://prism_render_scene/shaders/decal_resolve.wesl",
        ),
    );
    let pipeline = create_decal_resolve_pipeline(&host.device, &cache, shader);
    cache.process_queue();
    let native_pipeline = cache
        .get_compute_pipeline(pipeline.resolve())
        .expect("Decal fan-out pipeline must compile");

    let make_publication = |object: u32,
                            projector: DecalProjector,
                            material: DecalMaterial,
                            receiver: DecalReceiver| {
        let identity = DomainFrameIdentity {
            object: DomainObjectHandle::new(object, object + 10),
            history_key: object + 100,
            motion_source: MotionSource::Rigid,
        };
        let inputs = [DecalDomainInput {
            identity,
            projector: &projector,
            material: &material,
            receiver: &receiver,
        }];
        let items = build_decal_domain_items(&inputs).unwrap();
        let instance = DecalInstance::new(
            projector.world_to_decal.to_cols_array(),
            projector.projection_dir.to_array(),
            projector.angle_fade_start,
            projector.angle_fade_end,
            projector.edge_fade,
            0,
        );
        let expected = prism_render_architecture::render_fabric::resolve_decal(
            &receiver, &instance, &material,
        );
        let prepared = build_decal_resource_publication(
            &rhi, frame, identity, &items, &projector, &material, &receiver,
        )
        .unwrap();
        let ids: Vec<_> = prepared
            .domain_view
            .frame_bindings()
            .filter_map(|binding| match binding.physical {
                prism_render_architecture::render_fabric::PhysicalResource::Buffer {
                    id, ..
                } => Some(id),
                prism_render_architecture::render_fabric::PhysicalResource::Texture(_) => None,
            })
            .collect();
        assert_eq!(ids.len(), 6);
        let output = prepared
            .domain_view
            .resources()
            .last()
            .unwrap()
            .record_ready()
            .unwrap();
        let output = match output {
            DomainRecordReadyResource::Buffer { owner, .. } => owner.buffer().clone(),
            DomainRecordReadyResource::Texture { .. } => panic!("Decal output must be a buffer"),
        };
        let publication = DecalResolvePublication::new(
            prepared.frame,
            prepared.extent,
            prepared.identity,
            items,
            prepared.domain_view,
            prepared.workgroups,
        );
        (publication, ids, output, expected)
    };

    let first = make_publication(
        2,
        DecalProjector::new(
            Mat4::from_translation(Vec3::new(-0.2, 0.0, 0.0)),
            0.2,
            0.0,
            0.5,
        ),
        DecalMaterial {
            albedo: [0.8, 0.1, 0.0, 1.0],
            scalar_blend: [0.3, 0.2, 0.7, 0.0],
            ..DecalMaterial::default()
        },
        DecalReceiver {
            world_position: [0.2, 0.0, 0.0, 1.0],
            world_normal: [0.0, 0.0, -1.0, 0.0],
            albedo: [0.1, 0.2, 0.3, 1.0],
            scalar: [0.6, 0.1, 0.0, 0.0],
        },
    );
    let second = make_publication(
        9,
        DecalProjector::new(
            Mat4::from_translation(Vec3::new(0.1, 0.0, 0.0)),
            0.2,
            0.0,
            0.5,
        ),
        DecalMaterial {
            albedo: [0.0, 0.7, 0.2, 1.0],
            scalar_blend: [0.8, 0.4, 0.6, 1.0],
            ..DecalMaterial::default()
        },
        DecalReceiver {
            world_position: [-0.1, 0.0, 0.0, 1.0],
            world_normal: [0.0, 0.0, -1.0, 0.0],
            albedo: [0.2, 0.1, 0.05, 1.0],
            scalar: [0.3, 0.2, 0.0, 0.0],
        },
    );
    let publications = vec![first.0, second.0];
    let views = publications
        .iter()
        .map(|publication| {
            create_decal_bind_group(&rhi, &host.device, &pipeline, &publication.domain_view)
                .expect("Decal object bind group")
        })
        .collect::<Vec<_>>();
    let runtime = prepare_decal_resolves(&scene, &rhi, native_pipeline, &publications, &views)
        .expect("multi-object Decal SceneFabric preparation");
    assert_eq!(runtime.plan().pass_count(), 1);
    let submission = runtime.submit_tracked(&rhi);
    submission
        .wait_for_completion(&rhi)
        .expect("multi-object Decal SceneFabric submission completed");

    for (output, expected) in [(first.2, first.3), (second.2, second.3)] {
        let actual = readback(&host, &output);
        for (actual, expected) in actual[..4].iter().zip(expected.albedo) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        for (actual, expected) in actual[4..8].iter().zip(expected.normal) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        for (actual, expected) in actual[8..12].iter().zip(expected.scalar_coverage) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
    }
    let mut all_ids = first.1;
    all_ids.extend(second.1);
    for (index, id) in all_ids.iter().enumerate() {
        assert!(
            !all_ids[..index].contains(id),
            "each object owns six non-aliased buffers"
        );
    }
    assert_eq!(
        all_ids.len(),
        12,
        "each object owns six non-aliased buffers"
    );

    let mut owner = DecalRenderWorldOwner::default();
    owner
        .register_with_data(
            DomainFrameIdentity {
                object: DomainObjectHandle::new(40, 1),
                history_key: 140,
                motion_source: MotionSource::Rigid,
            },
            DecalProjector::new(Mat4::IDENTITY, 0.2, 0.0, 0.5),
            DecalMaterial::default(),
            DecalReceiver::default(),
        )
        .unwrap();
    owner
        .register_with_data(
            DomainFrameIdentity {
                object: DomainObjectHandle::new(41, 1),
                history_key: 141,
                motion_source: MotionSource::Rigid,
            },
            DecalProjector::new(Mat4::IDENTITY, 0.2, 0.0, 0.5),
            DecalMaterial::default(),
            DecalReceiver::default(),
        )
        .unwrap();
    let owner_publications =
        build_decal_resource_publications_from_owner(&rhi, frame + 1, &owner).unwrap();
    assert_eq!(owner_publications.len(), 2);
    assert!(owner_publications
        .iter()
        .all(|publication| publication.domain_view.resources().len() == 6));
}

#[test]
fn decal_resize_and_wrong_frame_publications_are_rejected_without_reusing_old_identity() {
    let Some(host) = host_or_skip() else {
        return;
    };
    let rhi = SceneRhi::from_bevy(&host.adapter, &host.device, &host.queue);
    let identity = DomainFrameIdentity {
        object: DomainObjectHandle::new(31, 4),
        history_key: 8,
        motion_source: MotionSource::Rigid,
    };
    let projector = DecalProjector::new(Mat4::IDENTITY, 0.2, 0.0, 0.5);
    let material = DecalMaterial::default();
    let receiver = DecalReceiver::default();
    let inputs = [DecalDomainInput {
        identity,
        projector: &projector,
        material: &material,
        receiver: &receiver,
    }];
    let items = build_decal_domain_items(&inputs).unwrap();
    let old = build_decal_resource_publication_with_extent(
        &rhi,
        90,
        identity,
        &items,
        &projector,
        &material,
        &receiver,
        Extent3d::new_2d(1, 1),
        [1, 1, 1],
    )
    .unwrap();
    let resized = build_decal_resource_publication_with_extent(
        &rhi,
        91,
        identity,
        &items,
        &projector,
        &material,
        &receiver,
        Extent3d::new_2d(4, 2),
        [1, 1, 1],
    )
    .unwrap();
    assert_ne!(old.extent, resized.extent);
    assert_ne!(
        old.domain_view.frame_bindings().next().unwrap().physical,
        resized
            .domain_view
            .frame_bindings()
            .next()
            .unwrap()
            .physical,
        "resize creates a new physical generation instead of recycling an in-flight buffer"
    );
    let mut prepared = super::recorder::DecalPreparedResources::default();
    assert!(prepared
        .publish(
            old.frame,
            old.identity,
            old.extent,
            old.domain_view,
            old.workgroups,
        )
        .is_ok());
    let wrong = match prepared.take_for(91, identity) {
        Err(error) => error,
        Ok(_) => panic!("wrong-frame Decal publication must be rejected"),
    };
    assert_eq!(
        wrong,
        "Decal prepared resource frame does not match Render Fabric frame"
    );
    assert!(prepared.take_for(90, identity).unwrap().is_some());
    assert!(prepared.take_for(90, identity).unwrap().is_none());
    drop(resized);
}
