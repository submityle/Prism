//! Numeric integration tests for the flexbox solver.
#![allow(
    clippy::std_instead_of_alloc,
    reason = "integration tests run under std and do not need alloc"
)]

use prism_ui_layout::{
    AlignItems, AvailableSpace, Dimension, Display, Edges, FlexDirection, JustifyContent,
    LayoutStyle, LayoutTree, Measure, Position, Size,
};

/// Shorthand for a definite available space.
fn space(width: f32, height: f32) -> Size<AvailableSpace> {
    Size::new(
        AvailableSpace::Definite(width),
        AvailableSpace::Definite(height),
    )
}

/// A fixed-size style helper.
fn fixed(width: f32, height: f32) -> LayoutStyle {
    LayoutStyle {
        size: Size::new(Dimension::Points(width), Dimension::Points(height)),
        ..LayoutStyle::default()
    }
}

fn approx(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-3, "expected {b}, got {a}");
}

#[test]
fn single_row_with_gap_and_padding() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(fixed(20.0, 20.0));
    let b = tree.new_leaf(fixed(30.0, 20.0));
    let c = tree.new_leaf(fixed(40.0, 20.0));
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            flex_direction: FlexDirection::Row,
            gap: Size::new(10.0, 0.0),
            padding: Edges::splat(Dimension::Points(5.0)),
            align_items: AlignItems::Start,
            ..LayoutStyle::default()
        },
        &[a, b, c],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    // Padding offsets the first child; gaps separate the rest.
    approx(tree.layout(a).location.x, 5.0);
    approx(tree.layout(b).location.x, 5.0 + 20.0 + 10.0);
    approx(tree.layout(c).location.x, 5.0 + 20.0 + 10.0 + 30.0 + 10.0);
    approx(tree.layout(a).location.y, 5.0);
    approx(tree.layout(a).size.width, 20.0);
}

#[test]
fn flex_grow_distributes_proportionally() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(LayoutStyle {
        flex_grow: 1.0,
        flex_basis: Dimension::Points(0.0),
        size: Size::new(Dimension::Auto, Dimension::Points(10.0)),
        ..LayoutStyle::default()
    });
    let b = tree.new_leaf(LayoutStyle {
        flex_grow: 2.0,
        flex_basis: Dimension::Points(0.0),
        size: Size::new(Dimension::Auto, Dimension::Points(10.0)),
        ..LayoutStyle::default()
    });
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            ..LayoutStyle::default()
        },
        &[a, b],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    // 300 units split 1:2 → 100 / 200.
    approx(tree.layout(a).size.width, 100.0);
    approx(tree.layout(b).size.width, 200.0);
    approx(tree.layout(a).location.x, 0.0);
    approx(tree.layout(b).location.x, 100.0);
}

#[test]
fn flex_shrink_resolves_overflow() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(LayoutStyle {
        flex_shrink: 1.0,
        size: Size::new(Dimension::Points(200.0), Dimension::Points(10.0)),
        ..LayoutStyle::default()
    });
    let b = tree.new_leaf(LayoutStyle {
        flex_shrink: 1.0,
        size: Size::new(Dimension::Points(200.0), Dimension::Points(10.0)),
        ..LayoutStyle::default()
    });
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            ..LayoutStyle::default()
        },
        &[a, b],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    // 400 wanted, 300 available: equal shrink factors × equal base → 150 each.
    approx(tree.layout(a).size.width, 150.0);
    approx(tree.layout(b).size.width, 150.0);
    approx(tree.layout(b).location.x, 150.0);
}

#[test]
fn justify_content_center() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(fixed(50.0, 10.0));
    let b = tree.new_leaf(fixed(50.0, 10.0));
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            justify_content: JustifyContent::Center,
            ..LayoutStyle::default()
        },
        &[a, b],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    // Content width 100, free 200, leading 100.
    approx(tree.layout(a).location.x, 100.0);
    approx(tree.layout(b).location.x, 150.0);
}

#[test]
fn justify_content_space_between() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(fixed(50.0, 10.0));
    let b = tree.new_leaf(fixed(50.0, 10.0));
    let c = tree.new_leaf(fixed(50.0, 10.0));
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            justify_content: JustifyContent::SpaceBetween,
            ..LayoutStyle::default()
        },
        &[a, b, c],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    // 150 used, 150 free split across 2 gaps → 75 each.
    approx(tree.layout(a).location.x, 0.0);
    approx(tree.layout(b).location.x, 50.0 + 75.0);
    approx(tree.layout(c).location.x, 300.0 - 50.0);
}

#[test]
fn align_items_center_on_cross_axis() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(fixed(20.0, 20.0));
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            align_items: AlignItems::Center,
            ..LayoutStyle::default()
        },
        &[a],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    // Cross axis is 100 tall; a 20-tall item centers at y = 40.
    approx(tree.layout(a).location.y, 40.0);
}

#[test]
fn align_items_stretch_fills_cross_axis() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(LayoutStyle {
        size: Size::new(Dimension::Points(20.0), Dimension::Auto),
        ..LayoutStyle::default()
    });
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            align_items: AlignItems::Stretch,
            ..LayoutStyle::default()
        },
        &[a],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    approx(tree.layout(a).size.height, 100.0);
    approx(tree.layout(a).location.y, 0.0);
}

#[test]
fn column_direction_stacks_children() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(fixed(20.0, 30.0));
    let b = tree.new_leaf(fixed(20.0, 40.0));
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            gap: Size::new(0.0, 5.0),
            align_items: AlignItems::Start,
            ..LayoutStyle::default()
        },
        &[a, b],
    );
    tree.compute_layout(root, space(100.0, 300.0));

    approx(tree.layout(a).location.y, 0.0);
    approx(tree.layout(b).location.y, 30.0 + 5.0);
    approx(tree.layout(a).location.x, 0.0);
}

#[test]
fn percentage_size_resolves_against_parent() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(LayoutStyle {
        size: Size::new(Dimension::Percent(0.5), Dimension::Percent(0.25)),
        ..LayoutStyle::default()
    });
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            align_items: AlignItems::Start,
            ..LayoutStyle::default()
        },
        &[a],
    );
    tree.compute_layout(root, space(200.0, 80.0));

    approx(tree.layout(a).size.width, 100.0);
    approx(tree.layout(a).size.height, 20.0);
}

#[test]
fn measure_function_sizes_leaf() {
    let mut tree = LayoutTree::new();
    let measured: fn(Size<Option<f32>>, Size<AvailableSpace>) -> Size<f32> =
        |_known, _available| Size::new(42.0, 17.0);
    let a = tree.new_leaf_with_measure(LayoutStyle::default(), measured);
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            align_items: AlignItems::Start,
            ..LayoutStyle::default()
        },
        &[a],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    approx(tree.layout(a).size.width, 42.0);
    approx(tree.layout(a).size.height, 17.0);
}

#[test]
fn custom_measure_impl_is_used() {
    struct Square(f32);
    impl Measure for Square {
        fn measure(
            &self,
            _known: Size<Option<f32>>,
            _available: Size<AvailableSpace>,
        ) -> Size<f32> {
            Size::new(self.0, self.0)
        }
    }

    let mut tree = LayoutTree::new();
    let a = tree.new_leaf_with_measure(LayoutStyle::default(), Square(25.0));
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            align_items: AlignItems::Start,
            ..LayoutStyle::default()
        },
        &[a],
    );
    tree.compute_layout(root, space(100.0, 100.0));

    approx(tree.layout(a).size.width, 25.0);
    approx(tree.layout(a).size.height, 25.0);
}

#[test]
fn absolute_child_positioned_by_inset() {
    let mut tree = LayoutTree::new();
    let abs = tree.new_leaf(LayoutStyle {
        position: Position::Absolute,
        inset: Edges::new(
            Dimension::Points(10.0),
            Dimension::Auto,
            Dimension::Points(15.0),
            Dimension::Auto,
        ),
        size: Size::new(Dimension::Points(30.0), Dimension::Points(30.0)),
        ..LayoutStyle::default()
    });
    let flow = tree.new_leaf(fixed(20.0, 20.0));
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            padding: Edges::splat(Dimension::Points(5.0)),
            align_items: AlignItems::Start,
            ..LayoutStyle::default()
        },
        &[abs, flow],
    );
    tree.compute_layout(root, space(200.0, 100.0));

    // Absolute child offset from the content box origin (padding = 5).
    approx(tree.layout(abs).location.x, 15.0);
    approx(tree.layout(abs).location.y, 20.0);
    // The in-flow child ignores the absolute sibling and starts at the origin.
    approx(tree.layout(flow).location.x, 5.0);
}

#[test]
fn reverse_row_places_from_end() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(fixed(40.0, 10.0));
    let b = tree.new_leaf(fixed(60.0, 10.0));
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            flex_direction: FlexDirection::RowReverse,
            ..LayoutStyle::default()
        },
        &[a, b],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    // First declared child sits at the far end.
    approx(tree.layout(a).location.x, 300.0 - 40.0);
    approx(tree.layout(b).location.x, 300.0 - 40.0 - 60.0);
}

#[test]
fn min_max_size_clamps_grow() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(LayoutStyle {
        flex_grow: 1.0,
        flex_basis: Dimension::Points(0.0),
        max_size: Size::new(Dimension::Points(80.0), Dimension::Auto),
        size: Size::new(Dimension::Auto, Dimension::Points(10.0)),
        ..LayoutStyle::default()
    });
    let b = tree.new_leaf(LayoutStyle {
        flex_grow: 1.0,
        flex_basis: Dimension::Points(0.0),
        size: Size::new(Dimension::Auto, Dimension::Points(10.0)),
        ..LayoutStyle::default()
    });
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            ..LayoutStyle::default()
        },
        &[a, b],
    );
    tree.compute_layout(root, space(300.0, 100.0));

    // `a` is capped at 80; `b` absorbs the remaining 220.
    approx(tree.layout(a).size.width, 80.0);
    approx(tree.layout(b).size.width, 220.0);
}

#[test]
fn wrapping_creates_multiple_lines() {
    let mut tree = LayoutTree::new();
    let a = tree.new_leaf(fixed(60.0, 20.0));
    let b = tree.new_leaf(fixed(60.0, 20.0));
    let c = tree.new_leaf(fixed(60.0, 20.0));
    let root = tree.new_node(
        LayoutStyle {
            display: Display::Flex,
            flex_wrap: prism_ui_layout::FlexWrap::Wrap,
            align_content: prism_ui_layout::AlignContent::Start,
            align_items: AlignItems::Start,
            ..LayoutStyle::default()
        },
        &[a, b, c],
    );
    // Only two 60-wide items fit per 150-wide line.
    tree.compute_layout(root, space(150.0, 200.0));

    approx(tree.layout(a).location.x, 0.0);
    approx(tree.layout(b).location.x, 60.0);
    approx(tree.layout(a).location.y, 0.0);
    // Third item wraps to the next line.
    approx(tree.layout(c).location.x, 0.0);
    approx(tree.layout(c).location.y, 20.0);
}
