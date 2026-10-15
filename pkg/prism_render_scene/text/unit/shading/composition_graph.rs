//! The same declaration is compiled once regardless of whether split inputs exist.

use super::*;

/// Composition owns the HDR WAW dependency and imported typed planes, not transient history.
#[test]
fn composition_is_resident_and_follows_water_response_producer() {
    let graph = shading_frame_graph();
    let compiled = graph.compile().unwrap();
    let find_pass = |name| {
        graph
            .passes()
            .iter()
            .position(|pass| pass.name == name)
            .unwrap()
    };
    let resolve = find_pass("shading_resolve");
    let water = find_pass("water_surface_raster");
    let composition = find_pass(COMPOSITION_PASS_NAME);
    assert_eq!(
        graph.passes()[composition].depends_on,
        [prism_render_graph::frame_graph::PassId(water as u32)]
    );
    assert_eq!(compiled.declared_pass_count, graph.passes().len());
    let hdr = graph
        .resources()
        .iter()
        .position(|resource| resource.name == "hdr_scene_color")
        .unwrap();
    let (hdr_is_texture, hdr_local_index) = logical_resource_index(&graph, hdr);
    assert!(hdr_is_texture);
    assert!(compiled.logical_barriers.iter().any(|barrier| {
        matches!(
            barrier.resource,
            prism_render_graph::plan::LogicalResource::Texture(resource)
                if resource.get() as usize == hdr_local_index
        ) && barrier.source == water
            && barrier.destination == composition
    }));
    for plane in [
        CompositionPlane::Left,
        CompositionPlane::Right,
        CompositionPlane::Controls,
        CompositionPlane::Auxiliary,
        CompositionPlane::WaterCoverage,
    ] {
        let index = graph
            .resources()
            .iter()
            .position(|resource| resource.name == plane.graph_name())
            .unwrap();
        assert_eq!(
            graph.resources()[index].lifetime,
            ResourceLifetime::Imported
        );
        assert!(graph.passes()[composition]
            .accesses
            .iter()
            .any(|access| access.resource.0 as usize == index));
    }
    let coverage_index = graph
        .resources()
        .iter()
        .position(|resource| resource.name == CompositionPlane::WaterCoverage.graph_name())
        .unwrap();
    assert_eq!(
        graph.passes()[composition]
            .accesses
            .iter()
            .find(|access| access.resource.0 as usize == coverage_index)
            .unwrap()
            .kind,
        AccessKind::StorageRead
    );
    let position = |pass| compiled.order.iter().position(|id| *id == pass).unwrap();
    assert!(position(resolve) < position(composition));
    assert!(position(water) < position(composition));

    for plane in [
        CompositionPlane::Left,
        CompositionPlane::Right,
        CompositionPlane::Controls,
        CompositionPlane::WaterCoverage,
    ] {
        let index = graph
            .resources()
            .iter()
            .position(|resource| resource.name == plane.graph_name())
            .unwrap();
        assert!(graph.passes()[water].accesses.iter().any(|access| {
            access.resource.0 as usize == index && access.kind == AccessKind::StorageWrite
        }));
    }
}
