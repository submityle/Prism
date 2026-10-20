use prism_render_architecture::particle::indirect_draw::RendererGeometry;
use prism_render_architecture::particle::renderers::RendererKind;
use prism_render_architecture::particle::SortStrategy;
use prism_render_driver::{DeviceDescriptor, Instance, Pass, RequestAdapterOptions};
use prism_render_driver_null::NullInstance;
use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};
use prism_render_gpudriven::particle::renderer_matrix::{
    ParticleAuxiliaryGpu, ParticleCullView, ParticleMeshVertex, ParticleRenderInput,
    ParticleRendererGpu, ParticleRendererRoute, RENDERER_MATRIX_TARGET_SIZE,
};

fn route(kind: RendererKind) -> ParticleRendererRoute {
    let geometry = match kind {
        RendererKind::Sprite => RendererGeometry::Sprite {
            alive_count: 1,
            first_instance: 0,
        },
        RendererKind::Mesh => RendererGeometry::Mesh {
            alive_count: 1,
            index_count: 6,
            first_instance: 0,
        },
        RendererKind::Ribbon => RendererGeometry::Ribbon {
            chain_count: 1,
            segments_per_chain: 2,
            first_instance: 0,
        },
        RendererKind::Beam => RendererGeometry::Beam {
            chain_count: 1,
            segments_per_chain: 2,
            first_instance: 0,
        },
        RendererKind::Light | RendererKind::Decal | RendererKind::Volume => unreachable!(),
    };
    ParticleRendererRoute::new(kind, SortStrategy::ViewDepthBitonic, geometry)
        .expect("valid renderer route")
}

fn inputs() -> [ParticleRenderInput; 3] {
    [
        ParticleRenderInput::new([-0.15, 0.0, 0.0], 0.35, 0.2, 0),
        ParticleRenderInput::new([0.0, 0.05, 0.0], 0.35, 0.4, 1),
        ParticleRenderInput::new([0.15, 0.1, 0.0], 0.35, 0.6, 2),
    ]
}

#[test]
fn null_rhi_records_each_surface_raster_and_auxiliary_phase() {
    let instance = NullInstance::new();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("null adapter");
    let (device, queue) = instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("null device");

    for kind in [
        RendererKind::Sprite,
        RendererKind::Mesh,
        RendererKind::Ribbon,
        RendererKind::Beam,
    ] {
        let stage = ParticleRendererGpu::new(&device, 8, route(kind)).expect("surface resources");
        stage
            .set_inputs(&queue, &inputs(), ParticleCullView::unrestricted())
            .expect("surface inputs");
        let recorded = stage
            .execute(&device, &queue)
            .expect("surface graph executes");
        assert!(recorded.commands().passes.iter().any(|pass| matches!(pass,
            Pass::Render { descriptor, commands } if descriptor.label.as_deref().is_some_and(|label| label.contains("particle-renderer-raster"))
                && commands.iter().any(|command| matches!(command, prism_render_driver::RenderCommand::DrawIndirect { .. } | prism_render_driver::RenderCommand::DrawIndexedIndirect { .. }))
        )), "{kind:?} records a raster indirect consumer");
        recorded.release(&device);
        stage.destroy(&device);
    }

    for kind in [
        RendererKind::Light,
        RendererKind::Decal,
        RendererKind::Volume,
    ] {
        let stage = ParticleAuxiliaryGpu::new(&device, kind, 8).expect("auxiliary resources");
        stage
            .set_inputs(&queue, &inputs(), 1.25)
            .expect("auxiliary inputs");
        let recorded = stage
            .execute(&device, &queue)
            .expect("auxiliary graph executes");
        let dispatches = recorded
            .commands()
            .passes
            .iter()
            .filter(|pass| matches!(pass, Pass::Compute { .. }))
            .count();
        assert_eq!(
            dispatches, 1,
            "{kind:?} has its own executable phase dispatch"
        );
        recorded.release(&device);
        stage.destroy(&device);
    }
}

#[test]
fn physical_gpu_rasterizes_mesh_ribbon_and_beam_shader_paths() {
    let pair = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(HardwareDeviceError::NoAdapter) if !require_hardware_gpu() => return,
        Err(HardwareDeviceError::NonHardware(_)) if !require_hardware_gpu() => return,
        Err(error) => panic!("physical GPU required for renderer matrix test: {error}"),
    };
    let (device, queue) = pair;
    for kind in [RendererKind::Mesh, RendererKind::Ribbon, RendererKind::Beam] {
        let mut stage =
            ParticleRendererGpu::new(&device, 8, route(kind)).expect("GPU raster stage");
        if kind == RendererKind::Mesh {
            stage
                .set_mesh_geometry(
                    &queue,
                    &[
                        ParticleMeshVertex::new([-0.48, -0.40], [0.0, 0.0]),
                        ParticleMeshVertex::new([0.48, -0.40], [1.0, 0.0]),
                        ParticleMeshVertex::new([0.48, 0.40], [1.0, 1.0]),
                        ParticleMeshVertex::new([-0.48, 0.40], [0.0, 1.0]),
                    ],
                    &[0, 1, 2, 0, 2, 3],
                )
                .expect("custom mesh geometry uploads");
        }
        stage
            .set_inputs(&queue, &inputs(), ParticleCullView::unrestricted())
            .expect("GPU inputs");
        let recorded = stage.execute(&device, &queue).expect("GPU graph execution");
        assert!(device.poll_wait(), "{kind:?} render completes on hardware");
        let pixels = queue.read_texture_rgba8(
            stage.raster_target(),
            RENDERER_MATRIX_TARGET_SIZE,
            RENDERER_MATRIX_TARGET_SIZE,
        );
        assert!(
            pixels
                .chunks_exact(4)
                .any(|pixel| pixel[..3].iter().any(|channel| *channel > 8)),
            "{kind:?} paints pixels"
        );
        recorded.release(&device);
        stage.destroy(&device);
    }
}

#[test]
fn physical_gpu_executes_auxiliary_renderer_compute_phases() {
    let pair = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(HardwareDeviceError::NoAdapter) if !require_hardware_gpu() => return,
        Err(HardwareDeviceError::NonHardware(_)) if !require_hardware_gpu() => return,
        Err(error) => panic!("physical GPU required for auxiliary renderer test: {error}"),
    };
    let (device, queue) = pair;
    for kind in [
        RendererKind::Light,
        RendererKind::Decal,
        RendererKind::Volume,
    ] {
        let stage = ParticleAuxiliaryGpu::new(&device, kind, 8).expect("GPU auxiliary stage");
        stage
            .set_inputs(&queue, &inputs(), 1.25)
            .expect("GPU inputs");
        let recorded = stage.execute(&device, &queue).expect("GPU phase dispatch");
        assert!(device.poll_wait(), "{kind:?} phase completes on hardware");
        assert_eq!(queue.read_buffer(stage.output(), 16).len(), 16);
        recorded.release(&device);
        stage.destroy(&device);
    }
}
