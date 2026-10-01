//! Integration tests for the overlay manager and focus trap.

use prism_ui::{Element, ElementKind, Key};
use prism_ui_overlay::{FocusTrap, OverlayEntry, OverlayKind, OverlayManager};

fn class_of(el: &Element) -> &str {
    el.class_names().first().map(String::as_str).unwrap_or("")
}

#[test]
fn z_order_stacks_across_kinds_and_insertion() {
    let mut m = OverlayManager::new();
    let tip = m.push(OverlayKind::Tooltip, Element::text("tip"));
    let modal_a = m.push(OverlayKind::Modal, Element::text("a"));
    let toast = m.push(OverlayKind::Toast, Element::text("toast"));
    let modal_b = m.push(OverlayKind::Modal, Element::text("b"));
    let pop = m.push(OverlayKind::Popover, Element::text("pop"));

    let order: Vec<_> = m.iter().map(OverlayEntry::id).collect();
    assert_eq!(order, vec![modal_a, modal_b, pop, tip, toast]);
    assert_eq!(m.top().unwrap().id(), toast);
    assert_eq!(m.len(), 5);
}

#[test]
fn push_dismiss_by_id_and_dismiss_top() {
    let mut m = OverlayManager::new();
    let a = m.push(OverlayKind::Popover, Element::text("a"));
    let b = m.push(OverlayKind::Modal, Element::text("b"));
    let c = m.push(OverlayKind::Toast, Element::text("c"));

    assert!(m.dismiss(b));
    assert!(!m.dismiss(b));
    assert_eq!(m.len(), 2);

    // Toast is top-most.
    assert_eq!(m.dismiss_top(), Some(c));
    assert_eq!(m.dismiss_top(), Some(a));
    assert!(m.is_empty());
    assert_eq!(m.dismiss_top(), None);
}

#[test]
fn render_composition_ordering_and_backdrop() {
    let mut m = OverlayManager::new();
    m.push(OverlayKind::Modal, Element::text("dialog"));
    m.push(OverlayKind::Toast, Element::text("toast"));

    let view = m.render(Element::box_().class("app"));

    // Root box: base + portal.
    assert_eq!(view.kind(), &ElementKind::Box);
    assert_eq!(view.child_elements().len(), 2);
    assert_eq!(class_of(&view.child_elements()[0]), "app");

    let portal = &view.child_elements()[1];
    assert_eq!(class_of(portal), "prism-overlay-portal");

    // Backdrop for the modal, modal layer, then toast layer (no backdrop).
    let layers = portal.child_elements();
    assert_eq!(layers.len(), 3);
    assert_eq!(class_of(&layers[0]), "prism-overlay-backdrop");
    assert_eq!(class_of(&layers[1]), "prism-overlay-layer");
    assert_eq!(class_of(&layers[2]), "prism-overlay-layer");

    // The modal layer wraps the pushed content.
    assert_eq!(layers[1].child_elements()[0].text_content(), Some("dialog"));
}

#[test]
fn render_keys_are_stable_and_distinct() {
    let mut m = OverlayManager::new();
    let modal = m.push(OverlayKind::Modal, Element::text("d"));

    let rendered = m.render(Element::box_());
    let portal = &rendered.child_elements()[1];
    let backdrop_key = portal.child_elements()[0].explicit_key().cloned();
    let layer_key = portal.child_elements()[1].explicit_key().cloned();

    let raw = modal.to_raw();
    assert_eq!(
        backdrop_key,
        Some(Key::Str(format!("prism-overlay-backdrop-{raw}")))
    );
    assert_eq!(
        layer_key,
        Some(Key::Str(format!("prism-overlay-layer-{raw}")))
    );

    // Re-rendering yields identical keys so the reconciler reuses nodes.
    let rendered2 = m.render(Element::box_());
    let portal2 = &rendered2.child_elements()[1];
    assert_eq!(
        portal2.child_elements()[0].explicit_key().cloned(),
        backdrop_key
    );
}

#[test]
fn render_empty_manager_has_empty_portal() {
    let m = OverlayManager::new();
    let view = m.render(Element::text("only base"));
    assert_eq!(view.child_elements().len(), 2);
    assert_eq!(view.child_elements()[0].text_content(), Some("only base"));
    assert!(view.child_elements()[1].child_elements().is_empty());
}

#[test]
fn escape_dismisses_top_dismissible() {
    let mut m = OverlayManager::new();
    let modal = m.push(OverlayKind::Modal, Element::text("m"));
    let pinned = m.push_with(OverlayKind::Toast, Element::text("sticky"), false);

    // Toast is top-most but pinned -> escape falls through to the modal.
    assert_eq!(m.on_escape(), Some(modal));
    assert_eq!(m.on_escape(), None);
    assert!(m.get(pinned).is_some());
    assert_eq!(m.len(), 1);
}

#[test]
fn scrim_click_dismisses_top_modal_only() {
    let mut m = OverlayManager::new();
    let under = m.push(OverlayKind::Modal, Element::text("under"));
    let over = m.push(OverlayKind::Modal, Element::text("over"));
    m.push(OverlayKind::Toast, Element::text("toast"));

    // Scrim click targets the top-most modal (`over`), not the toast above it.
    assert_eq!(m.on_scrim_click(), Some(over));
    assert_eq!(m.on_scrim_click(), Some(under));
    assert_eq!(m.on_scrim_click(), None);
}

#[test]
fn scrim_click_blocked_by_pinned_top_modal() {
    let mut m = OverlayManager::new();
    let _under = m.push(OverlayKind::Modal, Element::text("under"));
    m.push_with(OverlayKind::Modal, Element::text("pinned"), false);

    // Top modal is pinned; the click must not fall through to the one beneath.
    assert_eq!(m.on_scrim_click(), None);
    assert_eq!(m.len(), 2);
}

#[test]
fn focus_trap_cycles_with_wraparound() {
    let mut trap = FocusTrap::new(vec![
        Key::Str("ok".into()),
        Key::Str("cancel".into()),
        Key::Int(7),
    ]);
    assert_eq!(trap.len(), 3);
    assert_eq!(trap.current(), Some(&Key::Str("ok".into())));
    assert_eq!(trap.next(), Some(&Key::Str("cancel".into())));
    assert_eq!(trap.next(), Some(&Key::Int(7)));
    assert_eq!(trap.next(), Some(&Key::Str("ok".into())));
    assert_eq!(trap.prev(), Some(&Key::Int(7)));
}

#[test]
fn focus_trap_empty_is_graceful() {
    let mut trap: FocusTrap<Key> = FocusTrap::new(Vec::new());
    assert!(trap.is_empty());
    assert_eq!(trap.current(), None);
    assert_eq!(trap.next(), None);
    assert_eq!(trap.prev(), None);
}
