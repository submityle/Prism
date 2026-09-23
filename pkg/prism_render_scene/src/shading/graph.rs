use prism_render_architecture::frame_graph::{
    AccessKind, GpuFrameGraphBuilder, PassDescriptor, QueueClass, ResourceAccess,
    ResourceDescriptor, ResourceLifetime,
};

pub(crate) fn shading_frame_graph() -> GpuFrameGraphBuilder {
    let mut graph = GpuFrameGraphBuilder::default();
    let scene = resource(&mut graph, "gpu_scene", ResourceLifetime::Imported);
    let geometry = resource(&mut graph, "geometry", ResourceLifetime::Imported);
    let materials = resource(&mut graph, "materials", ResourceLifetime::Imported);
    let early_commands = resource(&mut graph, "early_commands", ResourceLifetime::Imported);
    let late_commands = resource(&mut graph, "late_commands", ResourceLifetime::Imported);
    let depth = resource(&mut graph, "main_depth", ResourceLifetime::Imported);
    let visibility = resource(&mut graph, "visibility_buffer", ResourceLifetime::Transient);
    let classification = resource(&mut graph, "material_classification", ResourceLifetime::Transient);
    let indirect_dispatch = resource(&mut graph, "shading_indirect_dispatch", ResourceLifetime::Transient);
    let scene_color = resource(&mut graph, "hdr_scene_color", ResourceLifetime::Imported);

    let raster = graph.add_pass(PassDescriptor {
        name: "visibility_raster".into(),
        queue: QueueClass::Graphics,
        accesses: vec![
            access(scene, AccessKind::StorageRead),
            access(geometry, AccessKind::StorageRead),
            access(materials, AccessKind::StorageRead),
            access(early_commands, AccessKind::IndirectRead),
            access(late_commands, AccessKind::IndirectRead),
            access(depth, AccessKind::DepthAttachment),
            access(visibility, AccessKind::ColorAttachment),
        ],
        depends_on: vec![],
    });
    let classify = graph.add_pass(PassDescriptor {
        name: "material_classification".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(visibility, AccessKind::StorageRead),
            access(materials, AccessKind::StorageRead),
            access(classification, AccessKind::StorageWrite),
            access(indirect_dispatch, AccessKind::StorageWrite),
        ],
        depends_on: vec![raster],
    });
    graph.add_pass(PassDescriptor {
        name: "shading_resolve".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(visibility, AccessKind::StorageRead),
            access(depth, AccessKind::SampledRead),
            access(scene, AccessKind::StorageRead),
            access(geometry, AccessKind::StorageRead),
            access(materials, AccessKind::StorageRead),
            access(classification, AccessKind::StorageRead),
            access(indirect_dispatch, AccessKind::IndirectRead),
            access(scene_color, AccessKind::StorageWrite),
        ],
        depends_on: vec![classify],
    });
    graph
}

fn resource(
    graph: &mut GpuFrameGraphBuilder,
    name: &'static str,
    lifetime: ResourceLifetime,
) -> prism_render_architecture::frame_graph::ResourceId {
    graph.add_resource(ResourceDescriptor {
        name: name.into(),
        lifetime,
        size: 0,
        alignment: 16,
    })
}

fn access(
    resource: prism_render_architecture::frame_graph::ResourceId,
    kind: AccessKind,
) -> ResourceAccess {
    ResourceAccess { resource, kind }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shading_graph_orders_raster_classification_and_resolve_hazards() {
        let graph = shading_frame_graph();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.execution_order.len(), 3);
        assert_eq!(graph.passes()[0].name, "visibility_raster");
        assert_eq!(graph.passes()[1].depends_on, [prism_render_architecture::frame_graph::PassId(0)]);
        assert_eq!(graph.passes()[2].depends_on, [prism_render_architecture::frame_graph::PassId(1)]);
        assert!(compiled.barriers.iter().any(|barrier| barrier.queue_transfer));
    }
}
