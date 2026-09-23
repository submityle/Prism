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
    let materials = graph.add_resource(ResourceDescriptor {
        name: "material_headers".into(),
        lifetime: ResourceLifetime::Imported,
        size: 0,
        alignment: 16,
    });
    let geometry = graph.add_resource(ResourceDescriptor {
        name: "geometry_lods".into(),
        lifetime: ResourceLifetime::Imported,
        size: 0,
        alignment: 16,
    });
    let work = graph.add_resource(ResourceDescriptor {
        name: "visibility_parity_work".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let ranges = graph.add_resource(ResourceDescriptor {
        name: "visibility_parity_ranges".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let indexed_indirect = graph.add_resource(ResourceDescriptor {
        name: "visibility_indexed_indirect".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let non_indexed_indirect = graph.add_resource(ResourceDescriptor {
        name: "visibility_non_indexed_indirect".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let late_bin_headers = graph.add_resource(ResourceDescriptor {
        name: "visibility_late_bin_headers".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let late_indexed_indirect = graph.add_resource(ResourceDescriptor {
        name: "visibility_late_indexed_indirect".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let late_non_indexed_indirect = graph.add_resource(ResourceDescriptor {
        name: "visibility_late_non_indexed_indirect".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let previous_hzb = graph.add_resource(ResourceDescriptor {
        name: "previous_hzb".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let current_hzb = graph.add_resource(ResourceDescriptor {
        name: "current_hzb".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let late_candidates = graph.add_resource(ResourceDescriptor {
        name: "visibility_late_candidates".into(),
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    });
    let early = graph.add_pass(PassDescriptor {
        name: "unified_visibility_previous_hzb".into(),
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
                resource: materials,
                kind: AccessKind::StorageRead,
            },
            ResourceAccess {
                resource: geometry,
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
            ResourceAccess {
                resource: indexed_indirect,
                kind: AccessKind::StorageWrite,
            },
            ResourceAccess {
                resource: non_indexed_indirect,
                kind: AccessKind::StorageWrite,
            },
            ResourceAccess {
                resource: previous_hzb,
                kind: AccessKind::SampledRead,
            },
            ResourceAccess {
                resource: late_candidates,
                kind: AccessKind::StorageWrite,
            },
        ],
        depends_on: vec![],
    });
    let current = graph.add_pass(PassDescriptor {
        name: "unified_visibility_current_hzb".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            ResourceAccess {
                resource: current_hzb,
                kind: AccessKind::SampledRead,
            },
            ResourceAccess {
                resource: late_candidates,
                kind: AccessKind::StorageRead,
            },
        ],
        depends_on: vec![early],
    });
    graph.add_pass(PassDescriptor {
        name: "unified_visibility_late_compact".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            ResourceAccess {
                resource: late_candidates,
                kind: AccessKind::StorageRead,
            },
            ResourceAccess {
                resource: geometry,
                kind: AccessKind::StorageRead,
            },
            ResourceAccess {
                resource: late_bin_headers,
                kind: AccessKind::StorageWrite,
            },
            ResourceAccess {
                resource: late_indexed_indirect,
                kind: AccessKind::StorageWrite,
            },
            ResourceAccess {
                resource: late_non_indexed_indirect,
                kind: AccessKind::StorageWrite,
            },
        ],
        depends_on: vec![current],
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
        assert_eq!(compiled.execution_order.len(), 3);
        assert_eq!(graph.passes()[0].accesses.len(), 10);
        assert_eq!(graph.passes()[1].depends_on[0].0, 0);
        assert_eq!(graph.passes()[2].depends_on[0].0, 1);
    }
}
