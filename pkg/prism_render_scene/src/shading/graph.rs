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
    let visibility_ids = resource(&mut graph, "visibility_ids", ResourceLifetime::Transient);
    let visibility_metadata = resource(
        &mut graph,
        "visibility_metadata",
        ResourceLifetime::Transient,
    );
    let pixel_classification = resource(
        &mut graph,
        "material_pixel_classification",
        ResourceLifetime::Transient,
    );
    let shading_work = resource(
        &mut graph,
        "material_shading_work",
        ResourceLifetime::Transient,
    );
    let class_counts = resource(
        &mut graph,
        "material_class_counts",
        ResourceLifetime::Transient,
    );
    let class_offsets = resource(
        &mut graph,
        "material_class_offsets",
        ResourceLifetime::Transient,
    );
    let class_cursors = resource(
        &mut graph,
        "material_class_cursors",
        ResourceLifetime::Transient,
    );
    let classification_diagnostics = resource(
        &mut graph,
        "material_classification_diagnostics",
        ResourceLifetime::Transient,
    );
    let indirect_dispatch = resource(
        &mut graph,
        "shading_indirect_dispatch",
        ResourceLifetime::Transient,
    );
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
            access(visibility_ids, AccessKind::ColorAttachment),
            access(visibility_metadata, AccessKind::ColorAttachment),
        ],
        depends_on: vec![],
    });
    let classify = graph.add_pass(PassDescriptor {
        name: "material_classification_count".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(visibility_ids, AccessKind::SampledRead),
            access(visibility_metadata, AccessKind::SampledRead),
            access(materials, AccessKind::StorageRead),
            access(pixel_classification, AccessKind::StorageWrite),
            access(class_counts, AccessKind::StorageWrite),
            access(classification_diagnostics, AccessKind::StorageWrite),
        ],
        depends_on: vec![raster],
    });
    let prefix = graph.add_pass(PassDescriptor {
        name: "material_classification_prefix".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(class_counts, AccessKind::StorageRead),
            access(class_offsets, AccessKind::StorageWrite),
            access(class_cursors, AccessKind::StorageWrite),
            access(indirect_dispatch, AccessKind::StorageWrite),
        ],
        depends_on: vec![classify],
    });
    let scatter = graph.add_pass(PassDescriptor {
        name: "material_classification_scatter".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            // Scatter also writes each accepted pixel's compact work index.
            access(pixel_classification, AccessKind::StorageWrite),
            access(class_offsets, AccessKind::StorageRead),
            access(class_cursors, AccessKind::StorageWrite),
            access(shading_work, AccessKind::StorageWrite),
            access(classification_diagnostics, AccessKind::StorageWrite),
        ],
        depends_on: vec![prefix],
    });
    graph.add_pass(PassDescriptor {
        name: "shading_resolve".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(visibility_ids, AccessKind::StorageRead),
            access(visibility_metadata, AccessKind::StorageRead),
            access(depth, AccessKind::SampledRead),
            access(scene, AccessKind::StorageRead),
            access(geometry, AccessKind::StorageRead),
            access(materials, AccessKind::StorageRead),
            access(shading_work, AccessKind::StorageRead),
            access(indirect_dispatch, AccessKind::IndirectRead),
            access(scene_color, AccessKind::StorageWrite),
        ],
        depends_on: vec![scatter],
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
        assert_eq!(compiled.execution_order.len(), 5);
        assert_eq!(graph.passes()[0].name, "visibility_raster");
        assert_eq!(
            graph
                .resources()
                .iter()
                .filter(|resource| resource.name.starts_with("visibility_"))
                .count(),
            2
        );
        assert_eq!(
            graph.passes()[1].depends_on,
            [prism_render_architecture::frame_graph::PassId(0)]
        );
        assert_eq!(
            graph.passes()[2].depends_on,
            [prism_render_architecture::frame_graph::PassId(1)]
        );
        assert_eq!(
            graph.passes()[3].depends_on,
            [prism_render_architecture::frame_graph::PassId(2)]
        );
        assert_eq!(
            graph.passes()[4].depends_on,
            [prism_render_architecture::frame_graph::PassId(3)]
        );
        assert!(compiled
            .barriers
            .iter()
            .any(|barrier| barrier.queue_transfer));
        let resource = |name: &str| {
            prism_render_architecture::frame_graph::ResourceId(
                graph
                    .resources()
                    .iter()
                    .position(|resource| resource.name == name)
                    .unwrap() as u32,
            )
        };
        let has_barrier = |name: &str, source: u32, destination: u32| {
            compiled.barriers.iter().any(|barrier| {
                barrier.resource == resource(name)
                    && barrier.source == prism_render_architecture::frame_graph::PassId(source)
                    && barrier.destination
                        == prism_render_architecture::frame_graph::PassId(destination)
            })
        };
        assert!(has_barrier("visibility_ids", 0, 1));
        assert!(has_barrier("material_class_counts", 1, 2));
        assert!(has_barrier("material_pixel_classification", 1, 3));
        assert!(has_barrier("material_class_offsets", 2, 3));
        assert!(has_barrier("material_class_cursors", 2, 3));
        assert!(has_barrier("material_shading_work", 3, 4));
        assert!(has_barrier("shading_indirect_dispatch", 2, 4));
    }
}
