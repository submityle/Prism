//! Workbench stories for [`Button`] (K6: every control ships stories).
//!
//! These double as a living gallery and a snapshot test: each story drives the
//! control through its args and asserts the rendered element carries the right
//! kit classes. Running under the Workbench harness proves the control renders
//! with a real reactive runtime, not just in isolation.

use prism_ui::Element;
use prism_ui_component::Component;
use prism_ui_component_kit::basics::{Button, ButtonProps};
use prism_ui_component_kit::kit::{ButtonVariant, ControlSize};
use prism_ui_workbench::{render_story, ControlValue, Story, StoryContext};

fn variant_from(name: &str) -> ButtonVariant {
    match name {
        "filled" => ButtonVariant::Filled,
        "tinted" => ButtonVariant::Tinted,
        "gray" => ButtonVariant::Gray,
        "glass" => ButtonVariant::Glass,
        _ => ButtonVariant::Plain,
    }
}

fn size_from(name: &str) -> ControlSize {
    match name {
        "sm" => ControlSize::Small,
        "lg" => ControlSize::Large,
        _ => ControlSize::Medium,
    }
}

fn render(ctx: &StoryContext<'_>) -> Element {
    let variant = variant_from(ctx.selected_arg("variant").unwrap_or("filled"));
    let size = size_from(ctx.selected_arg("size").unwrap_or("md"));
    let label = ctx.text_arg("label").unwrap_or("Button").to_string();
    let disabled = ctx.bool_arg("disabled").unwrap_or(false);
    Button.render(
        &ButtonProps::new(label)
            .variant(variant)
            .size(size)
            .disabled(disabled),
    )
}

fn button_story() -> Story {
    Story::builder("basics/Button")
        .arg(
            "variant",
            ControlValue::select(
                "filled",
                ["filled", "tinted", "gray", "glass", "plain"].map(String::from),
            )
            .unwrap(),
        )
        .arg(
            "size",
            ControlValue::select("md", ["sm", "md", "lg"].map(String::from)).unwrap(),
        )
        .arg("label", ControlValue::Text("Save changes".into()))
        .arg("disabled", ControlValue::Bool(false))
        .build(render)
}

#[test]
fn default_story_renders_filled_medium() {
    let story = button_story();
    let result = render_story(&story, story.default_args());
    assert!(result.node_count() > 0, "story rendered an empty tree");
    let classes = result.element().class_names();
    assert!(classes.iter().any(|c| c == "pk-button"));
    assert!(classes.iter().any(|c| c == "pk-button--filled"));
    assert!(classes.iter().any(|c| c == "pk-button--md"));
}

#[test]
fn glass_large_disabled_story() {
    let story = button_story();
    let args = story
        .default_args()
        .clone()
        .with(
            "variant",
            ControlValue::select(
                "glass",
                ["filled", "tinted", "gray", "glass", "plain"].map(String::from),
            )
            .unwrap(),
        )
        .with("size", ControlValue::select("lg", ["sm", "md", "lg"].map(String::from)).unwrap())
        .with("disabled", ControlValue::Bool(true));
    let result = render_story(&story, &args);
    let classes = result.element().class_names();
    assert!(classes.iter().any(|c| c == "pk-button--glass"));
    assert!(classes.iter().any(|c| c == "pk-button--lg"));
    assert!(classes.iter().any(|c| c == "is-disabled"));
}
