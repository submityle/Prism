use prism_render_architecture::frame_graph::{
    AccessKind, GpuFrameGraphBuilder, PassDescriptor, QueueClass, ResourceAccess,
    ResourceDescriptor, ResourceLifetime,
};

pub(crate) fn visibility_frame_graph() -> GpuFrameGraphBuilder {
    let mut graph = GpuFrameGraphBuilder::default();
    let scene = graph.add_resource(ResourceDescriptor {
        name: "gpu_scene".into(),
        lifetime: ResourceLifetime::Imported,
        size: 0,
        alignment: 16,
    });
    let views = graph.add_resource(ResourceDescriptor {
        name: "visibility_views".into(),
        lifetime: ResourceLifetime::Imported,
        size: 0,
        alignment: 16,
    });
    let work = graph.add_resource(ResourceDescriptor {
        name: "visible_work".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let ranges = graph.add_resource(ResourceDescriptor {
        name: "visibility_ranges".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    graph.add_pass(PassDescriptor {
        name: "unified_visibility".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            ResourceAccess {
                resource: scene,
                kind: AccessKind::StorageRead,
            },
            ResourceAccess {
                resource: views,
                kind: AccessKind::StorageRead,
            },
            ResourceAccess {
                resource: work,
                kind: AccessKind::StorageWrite,
            },
            ResourceAccess {
                resource: ranges,
                kind: AccessKind::StorageWrite,
            },
        ],
        depends_on: vec![],
    });
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_graph_declares_visibility_hazards() {
        let graph = visibility_frame_graph();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.execution_order.len(), 1);
        assert_eq!(graph.passes()[0].accesses.len(), 4);
    }
}
