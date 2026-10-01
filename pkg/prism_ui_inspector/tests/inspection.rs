//! End-to-end integration tests for `prism_ui_inspector`.

extern crate alloc;

use alloc::string::ToString;

use prism_ui::layout::{AvailableSpace, Size};
use prism_ui::{Element, RecordingBackend, Ui};
use prism_ui_devtools::{render_tree, snapshot, OpTrace};
use prism_ui_inspector::{
    find_by_class, find_by_kind, paths_of, resolve, NodePath, PerfReport, Query, TreeMetrics,
};

fn dashboard() -> Element {
    Element::box_()
        .class("app")
        .child(
            Element::box_()
                .class("sidebar")
                .child(Element::text("Home"))
                .child(Element::text("Settings")),
        )
        .child(
            Element::box_().class("content").child(
                Element::box_()
                    .class("card")
                    .class("primary")
                    .child(Element::text("Welcome back")),
            ),
        )
}

#[test]
fn query_addresses_nodes_by_path() {
    let snap = snapshot(&dashboard());

    // Every node has a path, enumerated deterministically.
    let paths = paths_of(&snap);
    assert_eq!(paths.len(), snap.node_count());

    // The `.card.primary` box lives at /1/0.
    let card = Query::new()
        .kind("Box")
        .with_class("card")
        .with_class("primary")
        .find_first(&snap)
        .expect("card present");
    assert_eq!(card.0, NodePath::from_indices([1, 0]));

    // Resolving the reported path returns the same node.
    let resolved = resolve(&snap, &card.0).expect("resolves");
    assert_eq!(resolved, card.1);

    // The path round-trips through its textual form.
    assert_eq!("/1/0".parse::<NodePath>().unwrap(), card.0);
    assert_eq!(card.0.to_string(), "/1/0");
}

#[test]
fn free_functions_and_counts() {
    let snap = snapshot(&dashboard());

    let boxes = find_by_kind(&snap, "Box");
    assert_eq!(boxes.len(), 4);

    let cards = find_by_class(&snap, "card");
    assert_eq!(cards.len(), 1);

    let texts = Query::new().kind("Text").count(&snap);
    assert_eq!(texts, 3);

    let welcome = Query::new().text_contains("Welcome").find_all(&snap);
    assert_eq!(welcome.len(), 1);
    assert_eq!(welcome[0].0, NodePath::from_indices([1, 0, 0]));
}

#[test]
fn perf_report_from_recorded_ops() {
    let view = dashboard();
    let snap = snapshot(&view);

    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(&view);
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(1024.0),
        AvailableSpace::Definite(768.0),
    ));

    let trace = OpTrace::from_ops(ui.backend().ops());
    let report = PerfReport::from_trace(&trace);

    // One create per node, one set_text per text node, one layout per node.
    assert_eq!(report.creates, snap.node_count());
    assert_eq!(report.set_texts, 3);
    assert_eq!(report.set_layouts, snap.node_count());
    assert_eq!(report.removes, 0);
    assert_eq!(report.reorders, 0);
    assert_eq!(report.total, trace.total());

    // Churn is purely the creates here (no removes/reorders).
    assert_eq!(report.churn(), report.creates);
    assert!(report.churn_percent() <= 100);
    assert!(report.report().contains("ops total:"));
}

#[test]
fn tree_metrics_and_render_tree() {
    let snap = snapshot(&dashboard());
    let metrics = TreeMetrics::from_snapshot(&snap);

    assert_eq!(metrics.node_count, snap.node_count());
    assert_eq!(metrics.depth, snap.depth());
    assert_eq!(metrics.max_fan_out, 2);
    assert_eq!(metrics.kind_histogram.get("Box").copied(), Some(4));
    assert_eq!(metrics.kind_histogram.get("Text").copied(), Some(3));

    // render_tree output is deterministic and lists every node line.
    let rendered = render_tree(&snap);
    let lines = rendered.lines().count();
    assert_eq!(lines, snap.node_count());
    assert!(rendered.contains("Text \"Welcome back\""));

    let report = metrics.report();
    assert!(report.contains("nodes: 7"));
    assert!(report.contains("Box=4"));
}
