//! The same declaration is compiled once regardless of whether split inputs exist.

use super::*;

/// Composition owns the HDR WAW dependency and imported typed planes, not transient history.
#[test]
fn composition_is_resident_and_follows_resolve_hdr_write() {
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
    let composition = find_pass(COMPOSITION_PASS_NAME);
    assert_eq!(
        graph.passes()[composition].depends_on,
        [prism_render_graph::frame_graph::PassId(resolve as u32)]
    );
    assert_eq!(compiled.declared_pass_count, graph.passes().len());
    let hdr = graph
        .resources()
        .iter()
        .position(|resource| resource.name == "hdr_scene_color")
        .unwrap();
    assert!(compiled
        .barriers
        .iter()
        .any(|barrier| barrier.resource.0 as usize == hdr
            && barrier.source.0 as usize == resolve
            && barrier.destination.0 as usize == composition));
    for plane in [
        CompositionPlane::Left,
        CompositionPlane::Right,
        CompositionPlane::Controls,
        CompositionPlane::Auxiliary,
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
    let position = |pass| {
        compiled
            .execution_order
            .iter()
            .position(|id| id.0 as usize == pass)
            .unwrap()
    };
    assert!(position(resolve) < position(composition));
}
