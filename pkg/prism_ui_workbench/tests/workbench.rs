//! End-to-end integration tests for the Loom component workbench.
//!
//! These exercise the public surface the way a user of the crate would:
//! building stories, registering them hierarchically, driving renders through
//! the isolated harness, and asserting the deterministic tree output.

#![forbid(unsafe_code)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use prism_ui::{Element, ElementKind};
use prism_ui_component::ContextMap;
use prism_ui_workbench::{
    render_story, render_story_default, render_story_with_context, ArgSet, ControlError,
    ControlValue, Story, Workbench,
};

/// Builds a small button story whose class and label are control-driven.
fn button_story(name: &str) -> Story {
    Story::builder(name)
        .arg("filled", ControlValue::Bool(true))
        .arg("label", ControlValue::Text("Click".to_string()))
        .arg(
            "size",
            ControlValue::select("md", ["sm", "md", "lg"]).expect("md is a valid option"),
        )
        .build(|ctx| {
            let variant = if ctx.bool_arg("filled").unwrap_or(false) {
                "btn-filled"
            } else {
                "btn-outline"
            };
            let size = ctx.selected_arg("size").unwrap_or("md");
            let label = ctx.text_arg("label").unwrap_or("Button");
            Element::box_()
                .class(variant)
                .class(format!("size-{size}"))
                .child(Element::text(label.to_string()))
        })
}

#[test]
fn registers_and_lists_stories_hierarchically() {
    let mut workbench = Workbench::new();
    workbench.add("Forms/Button", button_story("Secondary"));
    workbench.add("Forms/Button", button_story("Primary"));
    workbench.add("Forms/Input", button_story("Text"));
    workbench.add("Layout/Stack", button_story("Vertical"));

    // Groups come back in sorted, stable order.
    let groups: Vec<&str> = workbench.groups().collect();
    assert_eq!(groups, ["Forms/Button", "Forms/Input", "Layout/Stack"]);

    // Stories within a group are sorted by name.
    let button_names: Vec<&str> = workbench.group("Forms/Button").map(Story::name).collect();
    assert_eq!(button_names, ["Primary", "Secondary"]);

    // The flat listing walks groups then names deterministically.
    let flat: Vec<(&str, &str)> = workbench
        .stories()
        .map(|(group, name, _)| (group, name))
        .collect();
    assert_eq!(
        flat,
        [
            ("Forms/Button", "Primary"),
            ("Forms/Button", "Secondary"),
            ("Forms/Input", "Text"),
            ("Layout/Stack", "Vertical"),
        ]
    );

    assert_eq!(workbench.group_count(), 3);
    assert_eq!(workbench.len(), 4);
    assert!(workbench.contains("Forms/Button", "Primary"));
    assert!(!workbench.contains("Forms/Button", "Ghost"));
}

#[test]
fn args_drive_render_changes() {
    let story = button_story("Primary");

    // Default args: filled + md.
    let default = render_story_default(&story);
    assert_eq!(
        default.element().class_names(),
        &["btn-filled".to_string(), "size-md".to_string()]
    );
    assert_eq!(
        default.element().child_elements()[0].text_content(),
        Some("Click")
    );

    // Mutate three controls of three different kinds, then re-render.
    let mut args = story.default_args().clone();
    args.set_bool("filled", false).expect("bool kind matches");
    args.set_text("label", "Submit").expect("text kind matches");
    args.select("size", "lg").expect("lg is an option");

    let edited = render_story(&story, &args);
    assert_eq!(
        edited.element().class_names(),
        &["btn-outline".to_string(), "size-lg".to_string()]
    );
    assert_eq!(
        edited.element().child_elements()[0].text_content(),
        Some("Submit")
    );
    assert_eq!(edited.element().kind(), &ElementKind::Box);
}

#[test]
fn control_validation_is_enforced() {
    let story = button_story("Primary");
    let mut args = story.default_args().clone();

    // Wrong kind is rejected and leaves the value untouched.
    let err = args
        .set("filled", ControlValue::Number(1.0))
        .expect_err("bool control rejects a number");
    assert!(matches!(err, ControlError::TypeMismatch { .. }));
    assert_eq!(args.get_bool("filled"), Some(true));

    // Out-of-set option is rejected.
    let err = args.select("size", "xl").expect_err("xl is not an option");
    assert!(matches!(err, ControlError::UnknownOption { .. }));
    assert_eq!(args.get_selected("size"), Some("md"));

    // Unknown key is rejected.
    let err = args
        .set("ghost", ControlValue::Bool(true))
        .expect_err("ghost does not exist");
    assert!(matches!(err, ControlError::Missing { .. }));
}

#[test]
fn two_stories_render_in_isolation() {
    // Story A stands up two reactive nodes (signal + memo).
    let story_a = Story::builder("A")
        .arg("start", ControlValue::Number(10.0))
        .build(|ctx| {
            let start = ctx.number_arg("start").unwrap_or(0.0);
            let signal = ctx.runtime().signal(start);
            let memo = ctx.runtime().memo({
                let signal = signal.clone();
                move || signal.get() * 2.0
            });
            Element::box_().child(Element::text(format!("{}", memo.get())))
        });

    // Story B stands up three reactive nodes (signal + two memos).
    let story_b = Story::builder("B")
        .arg("start", ControlValue::Number(4.0))
        .build(|ctx| {
            let start = ctx.number_arg("start").unwrap_or(0.0);
            let signal = ctx.runtime().signal(start);
            let plus = ctx.runtime().memo({
                let signal = signal.clone();
                move || signal.get() + 1.0
            });
            let times = ctx.runtime().memo({
                let plus = plus.clone();
                move || plus.get() * 3.0
            });
            Element::box_().child(Element::text(format!("{}", times.get())))
        });

    let a1 = render_story_default(&story_a);
    let b1 = render_story_default(&story_b);
    let a2 = render_story_default(&story_a);

    // Each story's isolated runtime holds only its own nodes.
    assert_eq!(a1.live_nodes(), 2);
    assert_eq!(b1.live_nodes(), 3);
    // Re-rendering A does not accumulate B's nodes (or A's prior nodes).
    assert_eq!(a2.live_nodes(), a1.live_nodes());

    // Each produced the expected value from its own args.
    assert_eq!(a1.element().child_elements()[0].text_content(), Some("20"));
    assert_eq!(b1.element().child_elements()[0].text_content(), Some("15"));
}

#[test]
fn render_tree_output_is_deterministic_and_nested() {
    let story = Story::builder("Card")
        .arg("title", ControlValue::Text("Hello".to_string()))
        .build(|ctx| {
            let title = ctx.text_arg("title").unwrap_or("Untitled");
            Element::box_()
                .class("card")
                .child(Element::text(title.to_string()))
                .child(
                    Element::box_()
                        .class("body")
                        .child(Element::text("line one"))
                        .child(Element::text("line two")),
                )
        });

    let result = render_story_default(&story);

    assert_eq!(result.node_count(), 5);
    assert_eq!(result.depth(), 3);
    assert_eq!(
        result.tree(),
        "Box\n  Text \"Hello\"\n  Box\n    Text \"line one\"\n    Text \"line two\"\n",
    );
}

#[test]
fn context_values_are_injectable_through_the_harness() {
    struct Locale {
        greeting: &'static str,
    }

    let story = Story::builder("Localised").build(|ctx| {
        let greeting = ctx
            .inject::<Locale>()
            .map_or("hi", |locale| locale.greeting);
        Element::box_().child(Element::text(greeting.to_string()))
    });

    let mut context = ContextMap::new();
    context.provide(Locale {
        greeting: "bonjour",
    });

    let localised = render_story_with_context(&story, story.default_args(), &context);
    assert_eq!(
        localised.element().child_elements()[0].text_content(),
        Some("bonjour")
    );

    // Without the context the story falls back to its default.
    let bare = render_story(&story, &ArgSet::new());
    assert_eq!(
        bare.element().child_elements()[0].text_content(),
        Some("hi")
    );
}

#[test]
fn empty_argset_defaults_are_used() {
    let args = ArgSet::new();
    assert!(args.is_empty());
    let story = Story::builder("Fallback").build(|ctx| {
        let label = ctx.text_arg("missing").unwrap_or("default");
        Element::text(label.to_string())
    });
    let result = render_story(&story, &args);
    assert_eq!(result.element().text_content(), Some("default"));
}

#[test]
fn string_helper_is_used() {
    // Keeps the `String` import exercised and documents the expected label type.
    let name: String = Story::builder("Named")
        .build(|_ctx| Element::box_())
        .name()
        .to_string();
    assert_eq!(name, "Named");
}
