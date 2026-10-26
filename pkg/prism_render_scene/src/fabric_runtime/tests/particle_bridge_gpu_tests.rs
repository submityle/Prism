//! Particle publication -> SceneFabric -> real GPU completion slice.

use super::multipass_gpu::{hardware_context, read_word_at};
use super::*;
use bevy_render::{
    render_resource::{BindGroup, Buffer, ComputePipeline},
    renderer::RenderQueue,
};
use prism_particle_render::{
    particle_resource_payload, ParticleFabricGpuBuffer, ParticleFabricGpuDraw,
    ParticleFabricScenePublication, PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
    PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
};
use prism_render_architecture::{
    motion::MotionSource,
    particle::indirect_draw::RendererGeometry,
    render_fabric::{
        builtin_catalog, builtin_roots, DomainObjectHandle, DomainPayload, FrameRenderItem,
        RenderDomain, RenderObjectId, ResourcePayload, PARTICLE_ALIVE_RESOURCE,
        PARTICLE_STREAM_RESOURCE,
    },
};
use prism_render_driver::{BufferDescriptor, BufferState, BufferUsages, PipelineStages};
use std::panic::{catch_unwind, AssertUnwindSafe};

const FRAME: u64 = 71;
const STREAM_WORD: u32 = 0x517e_a11e;

fn scene() -> SceneFabric {
    let catalog = builtin_catalog().expect("built-in catalog");
    let roots = builtin_roots(&catalog).expect("built-in roots");
    SceneFabric(
        DeviceFabric::new(
            catalog,
            roots,
            CapabilitySet::WEBGPU_BASELINE,
            RenderQualityTier::High,
        )
        .expect("frozen built-in selection"),
    )
}

fn item(object: RenderObjectId, payload: DomainPayload) -> FrameRenderItem {
    FrameRenderItem {
        object_id: object,
        domain: RenderDomain::Particle,
        history_key: 0,
        motion_source: MotionSource::Particle,
        payload,
    }
}

fn resource_item(object: RenderObjectId, payload: ResourcePayload) -> FrameRenderItem {
    item(object, DomainPayload::Resource(payload))
}

fn gpu_buffer_metadata(id: prism_render_driver::BufferId, size: u64) -> ParticleFabricGpuBuffer {
    let semantic = particle_resource_payload(
        PARTICLE_STREAM_RESOURCE,
        PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
    )
    .expect("stream template");
    ParticleFabricGpuBuffer {
        semantic,
        buffer: id,
        version: id.generation(),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        entry_state: BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
    }
}

fn publication(
    stream: &prism_render_scene::SceneRhiBuffer,
    alive: &prism_render_scene::SceneRhiBuffer,
) -> ParticleFabricScenePublication {
    let object = RenderObjectId::Particle(DomainObjectHandle::new(7, 1));
    let stream_semantic = particle_resource_payload(
        PARTICLE_STREAM_RESOURCE,
        PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
    )
    .expect("stream template");
    let alive_semantic = particle_resource_payload(
        PARTICLE_ALIVE_RESOURCE,
        PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
    )
    .expect("alive template");
    let items = vec![
        resource_item(object, stream_semantic),
        resource_item(object, alive_semantic),
    ];
    let stream_metadata = gpu_buffer_metadata(stream.id(), stream.buffer().size());
    let alive_metadata = ParticleFabricGpuBuffer {
        semantic: alive_semantic,
        buffer: alive.id(),
        version: alive.id().generation(),
        size: alive.buffer().size(),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        entry_state: BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
    };
    ParticleFabricScenePublication::new(
        FRAME,
        items,
        vec![ParticleFabricGpuDraw {
            object_id: object,
            renderer_index: 0,
            history_key: 0,
            bounds_min: [-1.0; 3],
            bounds_max: [1.0; 3],
            geometry: RendererGeometry::Sprite {
                alive_count: 1,
                first_instance: 0,
            },
            stream: stream_metadata,
            alive: alive_metadata,
        }],
    )
}

#[test]
fn particle_publication_error_clears_previous_frame_slot() {
    let mut slot = crate::particle::ParticleSceneFabricSlot::default();
    slot.publish(Ok(ParticleFabricScenePublication::new(
        FRAME,
        Vec::new(),
        Vec::new(),
    )))
    .unwrap();
    assert!(!slot.is_empty());
    let error = crate::particle::ParticleSceneFabricError::FrameMismatch {
        expected: FRAME,
        actual: FRAME + 1,
    };
    assert!(slot.publish(Err(error)).is_err());
    assert!(slot.is_empty());
}

#[test]
fn particle_publication_wrong_frame_read_clears_previous_frame_slot() {
    let mut slot = crate::particle::ParticleSceneFabricSlot::default();
    slot.publish(Ok(ParticleFabricScenePublication::new(
        FRAME,
        Vec::new(),
        Vec::new(),
    )))
    .unwrap();
    assert!(slot.take_for_frame(FRAME + 1).is_none());
    assert!(slot.is_empty());
}

#[test]
fn particle_scene_fabric_real_gpu_bridge_render_and_retirement() {
    let Some((device, queue, adapter, render_device)) = hardware_context() else {
        return;
    };
    let rhi = SceneRhi::from_bevy(&adapter, &render_device, &RenderQueue::new(queue.clone()));
    let scene = scene();
    let stream = rhi.create_buffer(&BufferDescriptor::new(
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    ));
    let alive = rhi.create_buffer(&BufferDescriptor::new(
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    ));
    queue.write_buffer(stream.buffer(), 0, &[0; 256]);
    queue.write_buffer(alive.buffer(), 0, &[0; 256]);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particle-scene-fabric-bridge-render"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
                @group(0) @binding(0) var<storage, read_write> stream: array<u32>;
                @group(0) @binding(1) var<storage, read_write> alive: array<u32>;
                @compute @workgroup_size(1) fn bridge() {
                    stream[0] = 0x517e_a11eu;
                }
                @compute @workgroup_size(1) fn render() {
                    alive[0] = stream[0];
                }
            "#
            .into(),
        ),
    });
    let native_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("particle-scene-fabric-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("bridge"),
        compilation_options: Default::default(),
        cache: None,
    });
    let render_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("particle-scene-fabric-render-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("render"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = native_pipeline.get_bind_group_layout(0);
    let bridge_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("particle-scene-fabric-bridge-group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: stream.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: alive.buffer().as_entire_binding(),
            },
        ],
    });
    let render_layout = render_pipeline.get_bind_group_layout(0);
    let render_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("particle-scene-fabric-render-group"),
        layout: &render_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: stream.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: alive.buffer().as_entire_binding(),
            },
        ],
    });
    let pipeline = ComputePipeline::from(native_pipeline);
    let render_pipeline = ComputePipeline::from(render_pipeline);
    let bridge_group = BindGroup::from(bridge_group);
    let render_group = BindGroup::from(render_group);
    let publication = publication(&stream, &alive);

    let wrong_frame = crate::particle::prepare_particle_fabric(
        &scene,
        &rhi,
        crate::particle::ParticleSceneFabricInput {
            frame: FRAME + 1,
            extent: Extent3d::new_2d(1, 1),
            publication: &publication,
            stream: Some(crate::particle::ParticleSceneFabricBuffer {
                id: stream.id(),
                buffer: stream.buffer(),
            }),
            alive: Some(crate::particle::ParticleSceneFabricBuffer {
                id: alive.id(),
                buffer: alive.buffer(),
            }),
            bridge: crate::particle::ParticleSceneFabricDispatch {
                pipeline: &pipeline,
                bind_group: &bridge_group,
                bind_group_index: 0,
                workgroups: [1, 1, 1],
            },
            renderer: crate::particle::ParticleSceneFabricDispatch {
                pipeline: &render_pipeline,
                bind_group: &render_group,
                bind_group_index: 0,
                workgroups: [1, 1, 1],
            },
        },
    );
    assert!(matches!(
        wrong_frame,
        Err(crate::particle::ParticleSceneFabricError::FrameMismatch { .. })
    ));

    let runtime = crate::particle::prepare_particle_fabric(
        &scene,
        &rhi,
        crate::particle::ParticleSceneFabricInput {
            frame: FRAME,
            extent: Extent3d::new_2d(1, 1),
            publication: &publication,
            stream: Some(crate::particle::ParticleSceneFabricBuffer {
                id: stream.id(),
                buffer: stream.buffer(),
            }),
            alive: Some(crate::particle::ParticleSceneFabricBuffer {
                id: alive.id(),
                buffer: alive.buffer(),
            }),
            bridge: crate::particle::ParticleSceneFabricDispatch {
                pipeline: &pipeline,
                bind_group: &bridge_group,
                bind_group_index: 0,
                workgroups: [1, 1, 1],
            },
            renderer: crate::particle::ParticleSceneFabricDispatch {
                pipeline: &render_pipeline,
                bind_group: &render_group,
                bind_group_index: 0,
                workgroups: [1, 1, 1],
            },
        },
    )
    .expect("Particle SceneFabric prepare")
    .expect("non-empty Particle publication");
    assert_eq!(runtime.frame(), FRAME);
    assert_eq!(runtime.plan().pass_count(), 2);
    let imported_stream = runtime._buffer_leases[0].id();
    let imported_alive = runtime._buffer_leases[1].id();
    let submission = runtime.submit_tracked(&rhi);
    assert!(rhi.device().wgpu_buffer(imported_stream).size() >= 256);
    assert!(rhi.device().wgpu_buffer(imported_alive).size() >= 256);
    submission
        .wait_for_completion(&rhi)
        .expect("Particle SceneFabric completion");
    assert_eq!(
        read_word_at(&device, &queue, alive.buffer(), 0),
        STREAM_WORD
    );
    assert!(catch_unwind(AssertUnwindSafe(|| {
        rhi.device().wgpu_buffer(imported_stream)
    }))
    .is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| {
        rhi.device().wgpu_buffer(imported_alive)
    }))
    .is_err());
}
