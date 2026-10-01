//! End-to-end integration tests for `prism_ui_hotreload`.
//!
//! These drive the public API as an external consumer would: building element
//! trees, seeding per-node state, reloading, and diffing stylesheets.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::string::String;

use prism_ui::{Element, Key};
use prism_ui_hotreload::{
    diff_classes, identity::paths_of, ClassChange, HotReloader, NodePath, StateStore,
};
use prism_ui_style::{Class, MatchContext, StyleProp, StyleSheet, StyleValue, TokenStore};

/// Finds the path of the node whose explicit string key equals `key`.
fn path_for_key(root: &Element, key: &str) -> NodePath {
    paths_of(root)
        .into_iter()
        .find(|(_, element)| element.explicit_key() == Some(&Key::Str(String::from(key))))
        .map(|(path, _)| path)
        .expect("node with key must exist")
}

#[test]
fn keyed_list_reload_preserves_and_drops_state() {
    let old = Element::box_()
        .child(Element::text("keep").key_str("a"))
        .child(Element::box_().key_str("b"))
        .child(Element::text("gone").key_str("c"));

    let a_path = path_for_key(&old, "a");
    let b_path = path_for_key(&old, "b");
    let c_path = path_for_key(&old, "c");

    let mut reloader: HotReloader<i64> = HotReloader::new(old.clone());
    reloader.state_mut().insert(a_path.clone(), 1);
    reloader.state_mut().insert(b_path.clone(), 2);
    reloader.state_mut().insert(c_path.clone(), 3);

    let new = Element::box_()
        .child(Element::text("keep").key_str("a"))
        .child(Element::text("changed").key_str("b"))
        .child(Element::text("new").key_str("d"));

    let report = reloader.reload(new.clone());

    // Root + "a" preserved; "d" added; "b" recreated and "c" removed drop state.
    assert_eq!(report.preserved, 2);
    assert_eq!(report.added, 1);
    assert_eq!(report.dropped, 2);

    assert_eq!(reloader.current(), &new);
    assert_eq!(reloader.state().get(&a_path), Some(&1));
    assert!(!reloader.state().contains(&b_path));
    assert!(!reloader.state().contains(&c_path));
    assert_eq!(reloader.state().len(), 1);
}

#[test]
fn positional_kind_change_recreates_via_remove_and_add() {
    let old = Element::box_().child(Element::box_());
    let box_child = paths_of(&old)[1].0.clone();

    let mut reloader: HotReloader<&str> = HotReloader::new(old.clone());
    reloader.state_mut().insert(box_child.clone(), "state");

    let new = Element::box_().child(Element::text("now text"));
    let plan = reloader.plan_for(&new);
    assert_eq!(plan.counts().removed, 1);
    assert_eq!(plan.counts().added, 1);
    assert_eq!(plan.counts().recreated, 0);

    let report = reloader.reload(new);
    assert_eq!(report.dropped, 1);
    assert!(!reloader.state().contains(&box_child));
}

#[test]
fn deeply_nested_subtree_is_preserved() {
    let build = || {
        Element::box_().child(
            Element::box_()
                .key_str("panel")
                .child(Element::text("deep").key_str("label")),
        )
    };
    let old = build();
    let label_path = path_for_key(&old, "label");

    let mut reloader: HotReloader<u32> = HotReloader::new(old);
    reloader.state_mut().insert(label_path.clone(), 7);

    // Re-render an identical tree: nothing should change.
    let report = reloader.reload(build());
    assert_eq!(report.dropped, 0);
    assert_eq!(report.added, 0);
    assert_eq!(reloader.state().get(&label_path), Some(&7));
    assert!(reloader.plan_for(&reloader.current().clone()).is_noop());
}

#[test]
fn stylesheet_hot_swap_classifies_each_class() {
    let old = StyleSheet::new()
        .with_class(Class::new("btn").with(StyleProp::Width, StyleValue::px(100.0)))
        .with_class(Class::new("old").with(StyleProp::Height, StyleValue::px(10.0)));
    let new = StyleSheet::new()
        .with_class(Class::new("btn").with(StyleProp::Width, StyleValue::px(140.0)))
        .with_class(Class::new("fresh").with(StyleProp::Height, StyleValue::px(20.0)));

    let tokens = TokenStore::new();
    let ctx = MatchContext::new(800.0);
    let diff = diff_classes(&old, &new, &tokens, &["btn", "old", "fresh"], &ctx).unwrap();

    assert_eq!(diff.all().len(), 3);
    assert!(matches!(diff.all()[0], ClassChange::Changed { .. }));
    assert!(matches!(diff.all()[1], ClassChange::Removed(_)));
    assert!(matches!(diff.all()[2], ClassChange::Added(_)));
    assert_eq!(diff.changed().len(), 3);
    assert!(diff.report().contains("3 changed"));
}

#[test]
fn state_store_apply_plan_matches_reloader() {
    // Driving the plan + store directly should match HotReloader's bookkeeping.
    let old = Element::box_().child(Element::text("a").key_str("a"));
    let new = Element::box_()
        .child(Element::text("a").key_str("a"))
        .child(Element::text("b").key_str("b"));
    let a_path = path_for_key(&old, "a");

    let mut store: StateStore<&str> = StateStore::new();
    store.insert(a_path.clone(), "kept");
    let plan = prism_ui_hotreload::plan(&old, &new);
    let report = store.apply_plan(&plan);

    assert_eq!(report.added, 1);
    assert_eq!(report.dropped, 0);
    assert_eq!(report.preserved, 2);
    assert!(store.contains(&a_path));
}
