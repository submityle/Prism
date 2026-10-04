//! Flexbox layout-solve benchmark (roadmap "基准即规格").
//!
//! The design spec (`docs/prism_ui_loom_design_zh.md`) makes a fast,
//! deterministic layout pass the hot path of the whole UI runtime: every
//! frame that changes structure or available space re-solves the box tree
//! before paint. This benchmark builds a representative nested-flex UI (a
//! long list of panels, each with a fixed icon, a flexible multi-line body,
//! and a trailing badge) and times [`LayoutTree::compute_layout`], which the
//! solver documents as *always a full solve*.
//!
//! Two scenarios are reported:
//!
//! * **`full_solve`** — best-of-N full solves at a fixed viewport. Reports the
//!   lowest-noise pass and the per-node cost.
//! * **`resize_sweep`** — a responsive-resize sweep across a range of viewport
//!   widths, which forces flex grow/shrink to redistribute space every pass.
//!
//! A quantized geometry checksum guards against a "fast" run that skipped
//! work: the full-solve checksum must be non-zero (anti-vacuous) and bit-
//! identical across repeated identical-input passes (determinism), and the
//! resize sweep must observe the geometry actually change across widths.
//!
//! Dependency-free `harness = false` plain `main` using [`std::time::Instant`]:
//!
//! ```text
//! cargo bench -p prism_ui_layout --bench layout_solve
//! ```
//!
//! # Provenance
//!
//! This is an original benchmark authored for Prism. It implements the CSS
//! flexbox model from scratch and contains **no Unreal Engine source or
//! derived code** and wraps no third-party layout engine.
#![expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports its timing results to stdout"
)]

use std::hint::black_box;
use std::time::Instant;

use prism_ui_layout::{
    AvailableSpace, Dimension, Display, Edges, FlexDirection, LayoutStyle, LayoutTree, NodeId, Size,
};

/// Panels in the scrollable list.
const PANELS: usize = 256;
/// Text lines inside each panel body.
const LINES: usize = 3;
/// Timed passes; the reported figure is the best (lowest-noise) pass.
const PASSES: usize = 50;
/// Viewport widths visited by the resize sweep.
const SWEEP_WIDTHS: usize = 64;

/// A fixed-size leaf (icon / badge / line) in points.
fn fixed(w: f32, h: f32) -> LayoutStyle {
    LayoutStyle {
        size: Size::new(Dimension::Points(w), Dimension::Points(h)),
        ..LayoutStyle::default()
    }
}

/// Builds a representative nested-flex UI and returns the tree, its root, and
/// every node id (used by the checksum). The structure is a vertical list of
/// panels; each panel is a horizontal row of `[icon, body, badge]` where the
/// body is a flexible vertical stack of text lines. Gaps, padding, and margins
/// are present so the solve touches the full box model, not just raw sizes.
fn build_ui() -> (LayoutTree, NodeId, Vec<NodeId>) {
    let mut tree = LayoutTree::new();
    let mut all = Vec::new();

    let edges = |v: f32| Edges {
        left: Dimension::Points(v),
        right: Dimension::Points(v),
        top: Dimension::Points(v),
        bottom: Dimension::Points(v),
    };

    let mut panels = Vec::with_capacity(PANELS);
    for _ in 0..PANELS {
        let icon = tree.new_leaf(fixed(32.0, 32.0));
        all.push(icon);

        let mut lines = Vec::with_capacity(LINES);
        for i in 0..LINES {
            // Lines grow to fill the body width; heights are fixed.
            let line = tree.new_leaf(LayoutStyle {
                size: Size::new(Dimension::Auto, Dimension::Points(14.0)),
                flex_grow: 1.0,
                margin: edges((i % 2) as f32),
                ..LayoutStyle::default()
            });
            all.push(line);
            lines.push(line);
        }
        let body = tree.new_node(
            LayoutStyle {
                display: Display::Flex,
                flex_direction: FlexDirection::Column,
                flex_grow: 1.0,
                gap: Size::new(0.0, 4.0),
                padding: edges(6.0),
                ..LayoutStyle::default()
            },
            &lines,
        );
        all.push(body);

        let badge = tree.new_leaf(fixed(48.0, 20.0));
        all.push(badge);

        let panel = tree.new_node(
            LayoutStyle {
                display: Display::Flex,
                flex_direction: FlexDirection::Row,
                gap: Size::new(8.0, 0.0),
                padding: edges(8.0),
                margin: edges(2.0),
                ..LayoutStyle::default()
            },
            &[icon, body, badge],
        );
        all.push(panel);
        panels.push(panel);
    }

    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            gap: Size::new(0.0, 4.0),
            padding: edges(12.0),
            ..LayoutStyle::default()
        },
        &panels,
    );
    all.push(root);
    (tree, root, all)
}

/// A quantized, order-sensitive checksum over every box's resolved geometry.
/// Quantizing to 1/64 px keeps it bit-stable across runs while still changing
/// whenever any box moves or resizes, so it is a meaningful anti-vacuous and
/// determinism oracle.
fn checksum(tree: &LayoutTree, nodes: &[NodeId]) -> u64 {
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, &n) in nodes.iter().enumerate() {
        let l = tree.layout(n);
        let q = |v: f32| (v * 64.0) as i64 as u64;
        for field in [l.location.x, l.location.y, l.size.width, l.size.height] {
            acc ^= q(field).wrapping_add(i as u64);
            acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    acc
}

fn viewport(w: f32, h: f32) -> Size<AvailableSpace> {
    Size::new(AvailableSpace::Definite(w), AvailableSpace::Definite(h))
}

fn main() {
    let (mut tree, root, nodes) = build_ui();
    let node_count = nodes.len();
    let space = viewport(1280.0, 800.0);

    // Warm up and capture the reference checksum for the determinism guard.
    tree.compute_layout(root, space);
    let reference = checksum(&tree, &nodes);
    assert!(reference != 0, "layout checksum is vacuously zero");

    // --- Scenario 1: full_solve (best-of-N, fixed viewport). --------------
    let mut best = f64::INFINITY;
    for _ in 0..PASSES {
        let start = Instant::now();
        tree.compute_layout(black_box(root), black_box(space));
        let elapsed = start.elapsed().as_secs_f64();
        best = best.min(elapsed);
        // Determinism guard: identical inputs must reproduce the geometry.
        assert_eq!(
            checksum(&tree, &nodes),
            reference,
            "full solve is non-deterministic"
        );
    }
    let per_node_ns = best * 1.0e9 / node_count as f64;
    println!("prism_ui_layout::layout_solve");
    println!("  tree           : {PANELS} panels, {node_count} boxes");
    println!(
        "  full_solve     : {:.3} ms  ({per_node_ns:.1} ns/box)  [best of {PASSES}]",
        best * 1.0e3
    );

    // --- Scenario 2: resize_sweep (responsive relayout). ------------------
    // Sweeping the width forces flex grow/shrink to redistribute space; we
    // confirm the geometry actually changes so the solve is not a no-op.
    let mut distinct = 0u32;
    let mut prev = reference;
    let mut sink = 0u64;
    let start = Instant::now();
    for i in 0..SWEEP_WIDTHS {
        let w = 480.0 + (i as f32) * 24.0;
        tree.compute_layout(root, viewport(w, 800.0));
        let c = checksum(&tree, &nodes);
        if c != prev {
            distinct += 1;
        }
        prev = c;
        sink ^= c;
    }
    let sweep = start.elapsed().as_secs_f64();
    black_box(sink);
    assert!(
        distinct >= (SWEEP_WIDTHS as u32) / 2,
        "resize sweep did not re-solve geometry across widths (distinct = {distinct})"
    );
    println!(
        "  resize_sweep   : {:.3} ms  ({:.3} ms/solve over {SWEEP_WIDTHS} widths, {distinct} distinct)",
        sweep * 1.0e3,
        sweep * 1.0e3 / SWEEP_WIDTHS as f64
    );

    // Restore the fixed-viewport layout and re-verify the reference checksum.
    tree.compute_layout(root, space);
    assert_eq!(
        checksum(&tree, &nodes),
        reference,
        "re-solving the reference viewport did not reproduce the reference geometry"
    );
}
