//! Integration-like contract tests for the renderer-independent M1 plan.
//!
//! Detailed solver tests live beside the implementation so they can exercise
//! private arithmetic helpers; this module pins the public API and its error
//! behavior as consumed by a renderer host.

use crate::{solve, DockLayout, RenderPlanError, Viewport};

#[test]
fn zero_viewport_is_rejected_with_original_dimensions() {
    let error = solve(&DockLayout::default(), Viewport::new(0, 400)).unwrap_err();
    assert_eq!(
        error,
        RenderPlanError::InvalidViewport(Viewport::new(0, 400))
    );
}

#[test]
fn empty_root_is_a_valid_empty_render_surface() {
    let plan = solve(&DockLayout::default(), Viewport::new(640, 480)).unwrap();
    assert_eq!(
        plan.main_rect,
        crate::Rect {
            x: 0,
            y: 0,
            width: 640,
            height: 480,
        }
    );
    assert!(plan.panels.is_empty());
    assert!(plan.documents.is_empty());
    assert!(plan.tabs.is_empty());
}

#[test]
fn invalid_viewport_extent_that_cannot_fit_signed_coordinates_is_rejected() {
    let error = solve(&DockLayout::default(), Viewport::new(u32::MAX, 480)).unwrap_err();
    assert_eq!(error, RenderPlanError::GeometryOverflow);
}
