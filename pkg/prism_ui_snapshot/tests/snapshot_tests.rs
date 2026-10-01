//! End-to-end integration tests for `prism_ui_snapshot`.
//!
//! These exercise the public API the way a downstream crate would: build a
//! real [`prism_ui::Element`] view or a real [`prism_ui_layout::LayoutTree`],
//! capture it, serialize to golden text, and assert matches, mismatches, and
//! diffs.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

use prism_ui::Element;
use prism_ui_devtools::snapshot;
use prism_ui_layout::{
    AvailableSpace, Dimension, Display, FlexDirection, LayoutStyle, LayoutTree, Size,
};
use prism_ui_snapshot::{
    capture_layout, parse_layout, parse_tree, serialize_layout, serialize_tree, LayoutQuery,
    Snapshot,
};

fn sample_view() -> Element {
    Element::box_()
        .class("app")
        .child(Element::text("title"))
        .child(
            Element::box_()
                .class("row")
                .child(Element::text("left"))
                .child(Element::text("right")),
        )
}

#[test]
fn tree_serialization_round_trips() {
    let captured = snapshot(&sample_view());
    let text = serialize_tree(&captured);
    // Deterministic: serializing twice gives the same text.
    assert_eq!(serialize_tree(&captured), text);
    // Reversible: parsing yields an equal snapshot.
    let restored = parse_tree(&text).expect("valid snapshot text");
    assert_eq!(restored, captured);
    assert_eq!(serialize_tree(&restored), text);
}

#[test]
fn golden_match_and_mismatch() {
    let captured = snapshot(&sample_view());
    let golden = serialize_tree(&captured);

    let matched = Snapshot::from_tree(&captured).assert_matches(&golden);
    assert!(matched.is_match());
    assert_eq!(matched.diff(), None);

    // A view with changed text no longer matches the golden file.
    let changed = snapshot(
        &Element::box_()
            .class("app")
            .child(Element::text("title"))
            .child(
                Element::box_()
                    .class("row")
                    .child(Element::text("LEFT"))
                    .child(Element::text("right")),
            ),
    );
    let mismatch = Snapshot::from_tree(&changed).assert_matches(&golden);
    assert!(mismatch.is_mismatch());
    assert!(mismatch.into_result().is_err());
}

#[test]
fn diff_shows_added_and_removed_lines() {
    let base = snapshot(&Element::box_().child(Element::text("a")));
    let extended = snapshot(
        &Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b")),
    );
    let golden = serialize_tree(&base);
    let comparison = Snapshot::from_tree(&extended).assert_matches(&golden);
    let diff = comparison.diff().expect("mismatch diff");

    // Shared lines are kept, and the extra text node appears as an insertion
    // annotated with its node path.
    assert!(diff.contains(" 1/1 Box | kind=Box text=- classes=\n"));
    assert!(diff.contains("+ -/3 Box>Text | kind=Text text=+b classes=\n"));
}

#[test]
fn layout_snapshot_end_to_end() {
    let mut tree = LayoutTree::new();
    let header = tree.new_leaf(LayoutStyle {
        size: Size::new(Dimension::Points(200.0), Dimension::Points(30.0)),
        ..LayoutStyle::default()
    });
    let body = tree.new_leaf(LayoutStyle {
        size: Size::new(Dimension::Points(200.0), Dimension::Points(70.0)),
        ..LayoutStyle::default()
    });
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            ..LayoutStyle::default()
        },
        &[header, body],
    );
    tree.compute_layout(
        root,
        Size::new(
            AvailableSpace::Definite(200.0),
            AvailableSpace::Definite(100.0),
        ),
    );

    let query = LayoutQuery::new("root", root)
        .child(LayoutQuery::new("header", header))
        .child(LayoutQuery::new("body", body));
    let captured = capture_layout(&tree, &query);

    let golden = serialize_layout(&captured);
    // Column layout stacks the body below the header.
    assert!(golden.contains("label=header order=0 x=0.00 y=0.00 w=200.00 h=30.00\n"));
    assert!(golden.contains("label=body order=1 x=0.00 y=30.00 w=200.00 h=70.00\n"));

    // Golden comparison matches, and the text round-trips.
    let comparison = Snapshot::from_layout(&captured).assert_matches(&golden);
    assert!(comparison.is_match());
    let restored = parse_layout(&golden).expect("valid layout text");
    assert_eq!(serialize_layout(&restored), golden);

    // A different available height changes the body rectangle and the golden
    // comparison then fails with a diff.
    let mut taller = LayoutTree::new();
    let h2 = taller.new_leaf(LayoutStyle {
        size: Size::new(Dimension::Points(200.0), Dimension::Points(30.0)),
        ..LayoutStyle::default()
    });
    let b2 = taller.new_leaf(LayoutStyle {
        size: Size::new(Dimension::Points(200.0), Dimension::Points(120.0)),
        ..LayoutStyle::default()
    });
    let r2 = taller.new_node(
        LayoutStyle {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            ..LayoutStyle::default()
        },
        &[h2, b2],
    );
    taller.compute_layout(
        r2,
        Size::new(
            AvailableSpace::Definite(200.0),
            AvailableSpace::Definite(200.0),
        ),
    );
    let query2 = LayoutQuery::new("root", r2)
        .child(LayoutQuery::new("header", h2))
        .child(LayoutQuery::new("body", b2));
    let changed = capture_layout(&taller, &query2);
    let mismatch = Snapshot::from_layout(&changed).assert_matches(&golden);
    assert!(mismatch.is_mismatch());
    assert!(mismatch.diff().unwrap().contains("label=body"));
}
