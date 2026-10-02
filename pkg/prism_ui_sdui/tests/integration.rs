//! End-to-end integration tests for the server-driven UI sandbox.
//!
//! These exercise the public API exactly as a host would: build an untrusted
//! [`RemoteDocument`], negotiate a schema version, run it through the sandbox
//! and decode the sanitized tree into a `prism_ui::Element`.

use prism_ui::ElementKind;
use prism_ui_sdui::{
    decode, negotiate, render, CapabilitySet, Diagnostic, DiagnosticKind, RemoteDocument,
    RemoteNode, Resolution, Sandbox, Version, VersionSet, FALLBACK_CLASS, KIND_FALLBACK,
};

fn client() -> VersionSet {
    [Version::new(2, 0), Version::new(2, 2)]
        .into_iter()
        .collect()
}

fn sandbox() -> Sandbox {
    Sandbox::new(
        CapabilitySet::new()
            .allow_kinds(["box", "text"])
            .allow_styles(["card", "hero"])
            .allow_event("tap"),
    )
}

#[test]
fn full_pipeline_renders_whitelisted_tree() {
    let doc = RemoteDocument::new(
        Version::new(2, 2),
        RemoteNode::new("box")
            .style("card")
            .event("tap")
            .child(RemoteNode::text("Welcome"))
            .child(RemoteNode::new("box").style("hero")),
    );

    let outcome = render(&client(), &sandbox(), &doc);

    assert_eq!(outcome.resolution, Resolution::Exact(Version::new(2, 2)));
    assert!(outcome.diagnostics.is_empty());
    assert_eq!(outcome.element.kind(), &ElementKind::Box);
    assert_eq!(outcome.element.class_names(), &["card".to_string()]);
    assert_eq!(
        outcome.element.child_elements()[0].text_content(),
        Some("Welcome")
    );
    assert_eq!(
        outcome.element.child_elements()[1].class_names(),
        &["hero".to_string()]
    );
}

#[test]
fn full_pipeline_rejects_disallowed_kind_and_records_diagnostic() {
    let doc = RemoteDocument::new(
        Version::new(2, 0),
        RemoteNode::new("box")
            .child(RemoteNode::text("ok"))
            .child(RemoteNode::new("script").child(RemoteNode::text("payload"))),
    );

    let outcome = render(&client(), &sandbox(), &doc);

    assert_eq!(outcome.diagnostics.len(), 1);
    let diag: &Diagnostic = &outcome.diagnostics[0];
    assert_eq!(diag.kind, DiagnosticKind::RejectedKind);
    assert_eq!(diag.capability, "script");
    assert_eq!(diag.path, vec![1]);

    // The rejected node renders as a fallback element, and its untrusted child
    // ("payload") never made it into the output tree.
    let rejected = &outcome.element.child_elements()[1];
    assert_eq!(rejected.class_names(), &[FALLBACK_CLASS.to_string()]);
    assert_eq!(rejected.child_elements().len(), 1);
    assert_ne!(rejected.child_elements()[0].text_content(), Some("payload"));
}

#[test]
fn full_pipeline_strips_disallowed_styles_and_events() {
    let doc = RemoteDocument::new(
        Version::new(2, 0),
        RemoteNode::new("box")
            .style("card")
            .style("danger")
            .event("tap")
            .event("exfiltrate"),
    );

    let outcome = render(&client(), &sandbox(), &doc);

    // Only the whitelisted style survives as a class; events never become
    // classes at all.
    assert_eq!(outcome.element.class_names(), &["card".to_string()]);
    let kinds: Vec<DiagnosticKind> = outcome.diagnostics.iter().map(|d| d.kind).collect();
    assert_eq!(
        kinds,
        vec![DiagnosticKind::StrippedStyle, DiagnosticKind::StrippedEvent]
    );
}

#[test]
fn full_pipeline_downgrades_when_client_is_behind() {
    let doc = RemoteDocument::new(
        Version::new(2, 7),
        RemoteNode::new("box").child(RemoteNode::text("hi")),
    );

    let outcome = render(&client(), &sandbox(), &doc);

    assert_eq!(
        outcome.resolution,
        Resolution::Downgraded {
            requested: Version::new(2, 7),
            resolved: Version::new(2, 2),
        }
    );
    // The content itself is whitelisted, so it still renders cleanly.
    assert!(outcome.diagnostics.is_empty());
    assert_eq!(
        outcome.element.child_elements()[0].text_content(),
        Some("hi")
    );
}

#[test]
fn full_pipeline_short_circuits_unsupported_family() {
    let doc = RemoteDocument::new(
        Version::new(5, 0),
        RemoteNode::new("box").child(RemoteNode::new("script")),
    );

    let outcome = render(&client(), &sandbox(), &doc);

    assert_eq!(
        outcome.resolution,
        Resolution::Unsupported {
            requested: Version::new(5, 0)
        }
    );
    // The untrusted tree is never even sanitized; we fall straight back.
    assert!(outcome.diagnostics.is_empty());
    assert_eq!(outcome.element.kind(), &ElementKind::Box);
    assert_eq!(outcome.element.class_names(), &[FALLBACK_CLASS.to_string()]);
}

#[test]
fn negotiate_then_decode_matches_render() {
    // Driving the stages by hand must agree with the all-in-one `render`.
    let doc = RemoteDocument::new(
        Version::new(2, 2),
        RemoteNode::new("box").child(RemoteNode::text("x")),
    );
    let sandbox = sandbox();

    let resolution = negotiate(&client(), doc.version());
    assert!(resolution.is_renderable());
    let sanitized = sandbox.sanitize(doc.root());
    let by_hand = decode(&sanitized.root);

    let outcome = render(&client(), &sandbox, &doc);
    assert_eq!(by_hand.kind(), outcome.element.kind());
    assert_eq!(
        by_hand.child_elements()[0].text_content(),
        outcome.element.child_elements()[0].text_content()
    );
}

#[test]
fn hostile_depth_is_bounded_and_never_panics() {
    // A pathologically deep tree must not blow the stack; the sandbox caps the
    // descent and emits a depth diagnostic instead.
    let mut node = RemoteNode::new("box");
    for _ in 0..5_000 {
        node = RemoteNode::new("box").child(node);
    }
    let doc = RemoteDocument::new(Version::new(2, 0), node);
    let bounded = Sandbox::new(CapabilitySet::new().allow_kind("box")).with_max_depth(32);

    let outcome = render(&client(), &bounded, &doc);

    assert!(outcome
        .diagnostics
        .iter()
        .any(|d| d.kind == DiagnosticKind::DepthExceeded));
    // A fallback placeholder sits at the capped frontier.
    let mut cursor = &outcome.element;
    let mut saw_fallback = false;
    while let Some(child) = cursor.child_elements().first() {
        if cursor.class_names() == [FALLBACK_CLASS.to_string()] {
            saw_fallback = true;
            break;
        }
        cursor = child;
    }
    assert!(saw_fallback || cursor.class_names() == [FALLBACK_CLASS.to_string()]);
}

#[test]
fn impersonating_the_reserved_fallback_kind_is_rejected() {
    // A remote node may not smuggle itself in as the trusted placeholder kind.
    let doc = RemoteDocument::new(
        Version::new(2, 0),
        RemoteNode::new(KIND_FALLBACK).child(RemoteNode::text("stay out")),
    );

    let outcome = render(&client(), &sandbox(), &doc);

    assert_eq!(outcome.diagnostics.len(), 1);
    assert_eq!(outcome.diagnostics[0].kind, DiagnosticKind::RejectedKind);
    assert_eq!(outcome.element.class_names(), &[FALLBACK_CLASS.to_string()]);
}
