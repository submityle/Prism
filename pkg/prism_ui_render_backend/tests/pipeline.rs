//! End-to-end integration: drive a [`RetainedScene`] with a realistic
//! [`BackendOp`] stream, then assert the lowered draw list rasterises and
//! batches as expected. This exercises the public pipeline the way the runtime
//! and a renderer actually consume it.

use prism_ui::{Backend, BackendId, BackendOp, ElementKind, PaintStyle};
use prism_ui_layout::{Point, Size};
use prism_ui_render_backend::{batch, instance_count, rasterize, scene::RetainedScene};
use prism_ui_style::Color;

/// Builds a parent box containing two coloured child boxes laid out
/// side-by-side, matching how layout/paint would feed the backend.
fn build_scene() -> RetainedScene {
    let mut scene = RetainedScene::default();

    scene.apply(BackendOp::Create {
        id: BackendId(1),
        kind: ElementKind::Box,
        parent: None,
        index: 0,
    });
    scene.apply(BackendOp::SetLayout {
        id: BackendId(1),
        location: Point::new(0.0, 0.0),
        size: Size::new(40.0, 20.0),
    });

    for (i, (id, x, color)) in [
        (2u64, 0.0, Color::rgba(1.0, 0.0, 0.0, 1.0)),
        (3u64, 20.0, Color::rgba(0.0, 1.0, 0.0, 1.0)),
    ]
    .into_iter()
    .enumerate()
    {
        scene.apply(BackendOp::Create {
            id: BackendId(id),
            kind: ElementKind::Box,
            parent: Some(BackendId(1)),
            index: i,
        });
        scene.apply(BackendOp::SetLayout {
            id: BackendId(id),
            location: Point::new(x, 0.0),
            size: Size::new(20.0, 20.0),
        });
        let paint = PaintStyle {
            background_color: Some(color),
            ..PaintStyle::default()
        };
        scene.apply(BackendOp::SetPaint {
            id: BackendId(id),
            paint,
        });
    }

    scene
}

#[test]
fn pipeline_lowers_rasterises_and_batches() {
    let scene = build_scene();
    assert_eq!(scene.len(), 3);

    let list = scene.to_draw_list();
    // Parent has no paint, so only the two painted children emit rects.
    assert_eq!(list.len(), 2);

    // Two consecutive same-kind rects coalesce into a single instanced batch.
    let batches = batch(&list);
    assert_eq!(instance_count(&batches), 2);
    assert_eq!(batches.len(), 1);

    // Absolute positions: child 2 is on the left, child 3 is on the right.
    let fb = rasterize(&list, 40, 20);
    let left = fb.pixel(10, 10);
    let right = fb.pixel(30, 10);
    assert!(left[0] > 0.9 && left[1] < 0.1, "left pixel should be red");
    assert!(
        right[1] > 0.9 && right[0] < 0.1,
        "right pixel should be green"
    );
}

#[test]
fn reorder_swaps_paint_order_without_recreating() {
    let mut scene = build_scene();
    // Make the two children overlap so paint order is observable.
    scene.apply(BackendOp::SetLayout {
        id: BackendId(3),
        location: Point::new(0.0, 0.0),
        size: Size::new(20.0, 20.0),
    });

    let before = rasterize(&scene.to_draw_list(), 20, 20);
    // Child 3 (green) painted last wins the overlap.
    assert!(before.pixel(10, 10)[1] > 0.9);

    scene.apply(BackendOp::Reorder {
        parent: BackendId(1),
        order: vec![BackendId(3), BackendId(2)],
    });
    let after = rasterize(&scene.to_draw_list(), 20, 20);
    // Now child 2 (red) paints last and wins.
    assert!(after.pixel(10, 10)[0] > 0.9);
}
