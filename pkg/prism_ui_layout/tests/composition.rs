//! Integration tests exercising the §9.3 incremental path and the §9.4
//! composable layout protocol together, through the public crate API only.

use prism_ui_layout::{
    AbsoluteProtocol, Alignment, AvailableSpace, Constraints, Dimension, Edges, FlexDirection,
    FlexProtocol, GridProtocol, LayoutProtocol, LayoutStyle, LayoutTree, Modifier, ModifierChain,
    Point, Rect, Size, StackProtocol, WrapProtocol,
};

fn viewport(w: f32, h: f32) -> Size<AvailableSpace> {
    Size::new(AvailableSpace::Definite(w), AvailableSpace::Definite(h))
}

fn fixed(w: f32, h: f32) -> LayoutStyle {
    LayoutStyle {
        size: Size::new(Dimension::Points(w), Dimension::Points(h)),
        ..LayoutStyle::default()
    }
}

#[test]
fn incremental_local_change_only_resolves_dirty_subtree() {
    let mut tree = LayoutTree::new();
    // Content-measured leaf (not a boundary) nested under a fixed wrapper.
    let leaf = tree.new_leaf_with_measure(
        LayoutStyle::default(),
        |_known: Size<Option<f32>>, _avail: Size<AvailableSpace>| Size::new(40.0, 40.0),
    );
    let boundary = tree.new_node(fixed(80.0, 80.0), &[leaf]);
    let sibling = tree.new_leaf(fixed(10.0, 10.0));
    let root = tree.new_node(LayoutStyle::default(), &[boundary, sibling]);

    tree.compute_layout_incremental(root, viewport(200.0, 200.0));
    let sibling_before = *tree.layout(sibling);

    tree.mark_needs_layout(leaf);
    tree.compute_layout_incremental(root, viewport(200.0, 200.0));

    assert!(tree.was_relayouted(leaf));
    assert!(tree.was_relayouted(boundary));
    assert!(!tree.was_relayouted(root));
    assert!(!tree.was_relayouted(sibling));
    // The untouched sibling keeps its exact geometry.
    assert_eq!(*tree.layout(sibling), sibling_before);
}

#[test]
fn paint_only_change_does_not_relayout() {
    let mut tree = LayoutTree::new();
    let leaf = tree.new_leaf(fixed(20.0, 20.0));
    let root = tree.new_node(LayoutStyle::default(), &[leaf]);

    tree.compute_layout_incremental(root, viewport(100.0, 100.0));
    tree.mark_needs_paint(leaf);
    tree.compute_layout_incremental(root, viewport(100.0, 100.0));

    assert!(!tree.was_relayouted(leaf));
    assert!(!tree.was_relayouted(root));
}

#[test]
fn flex_grid_stack_absolute_wrap_place_deterministically() {
    let children = [Size::new(20.0, 10.0), Size::new(30.0, 10.0)];
    let bounds = Rect::new(Point::ZERO, Size::new(300.0, 100.0));

    let flex = FlexProtocol::new(FlexDirection::Row);
    let flex_rects = flex.place(&children, bounds);
    assert_eq!(flex_rects[0].location, Point::ZERO);
    assert_eq!(flex_rects[1].location.x, 20.0);

    let grid = GridProtocol::new(1);
    let grid_rects = grid.place(&children, bounds);
    // Single column: second item stacks below the first.
    assert_eq!(grid_rects[0].location, Point::ZERO);
    assert_eq!(grid_rects[1].location, Point::new(0.0, 10.0));

    let stack = StackProtocol::new(Alignment::TOP_LEFT);
    let stack_rects = stack.place(&children, bounds);
    // Z-overlap: every child shares the same origin.
    assert_eq!(stack_rects[0].location, Point::ZERO);
    assert_eq!(stack_rects[1].location, Point::ZERO);

    let abs = AbsoluteProtocol::new(alloc_insets());
    let abs_rects = abs.place(&children, bounds);
    assert_eq!(abs_rects[0].location, Point::new(5.0, 5.0));

    let wrap = WrapProtocol::new(0.0, 0.0);
    let wrap_rects = wrap.place(&children, bounds);
    assert_eq!(wrap_rects.len(), 2);
}

fn alloc_insets() -> Vec<Edges<Dimension>> {
    vec![Edges::new(
        Dimension::Points(5.0),
        Dimension::Auto,
        Dimension::Points(5.0),
        Dimension::Auto,
    )]
}

#[test]
fn modifier_order_is_semantic() {
    let content = Size::new(50.0, 30.0);
    let available = Size::new(500.0, 500.0);

    let padding_then_size = ModifierChain::new()
        .then(Modifier::Padding(Edges::splat(10.0)))
        .then(Modifier::Size {
            width: Some(100.0),
            height: Some(100.0),
        })
        .resolve(available, content);

    let size_then_padding = ModifierChain::new()
        .then(Modifier::Size {
            width: Some(100.0),
            height: Some(100.0),
        })
        .then(Modifier::Padding(Edges::splat(10.0)))
        .resolve(available, content);

    // Both force a 100x100 content box, but the outer footprint matches
    // because padding contributes equally; the real order sensitivity shows
    // up with aspect ratio below. Here we assert the content box is forced.
    assert_eq!(padding_then_size.content.size, Size::new(100.0, 100.0));
    assert_eq!(size_then_padding.content.size, Size::new(100.0, 100.0));

    let size_then_aspect = ModifierChain::new()
        .then(Modifier::Size {
            width: Some(100.0),
            height: Some(40.0),
        })
        .then(Modifier::AspectRatio(1.0))
        .resolve(available, content);
    let aspect_then_size = ModifierChain::new()
        .then(Modifier::AspectRatio(1.0))
        .then(Modifier::Size {
            width: Some(100.0),
            height: Some(40.0),
        })
        .resolve(available, content);
    assert_ne!(size_then_aspect.content.size, aspect_then_size.content.size);
    assert_eq!(size_then_aspect.content.size, Size::new(100.0, 100.0));
    assert_eq!(aspect_then_size.content.size, Size::new(100.0, 40.0));
}

#[test]
fn protocol_measure_constrains_result() {
    let proto = GridProtocol::new(2);
    let children = [
        Size::new(40.0, 20.0),
        Size::new(30.0, 25.0),
        Size::new(10.0, 10.0),
    ];
    let size = proto.measure(&children, Constraints::loose(Size::new(1000.0, 1000.0)));
    // Two columns (40 + 30 = 70 wide); rows: max(20,25)=25 and 10 -> 35 tall.
    assert_eq!(size, Size::new(70.0, 35.0));
}
