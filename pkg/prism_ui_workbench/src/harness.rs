//! The isolated render harness.
//!
//! [`render_story`] drives a [`Story`] to a [`prism_ui::Element`] tree inside a
//! freshly created, private reactive [`Runtime`] and a component
//! [`ComponentCtx`]. Because every call builds its own runtime and context, the
//! signals, memos and effects one story creates can never leak into another —
//! each render is a clean room. The result also carries a deterministic
//! [`prism_ui_devtools::render_tree`] rendering for snapshot and debug use.

#![forbid(unsafe_code)]

use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::{render_with_context, ComponentCtx, ContextMap};
use prism_ui_devtools::{render_tree, snapshot};
use prism_ui_reactive::Runtime;

use crate::controls::ArgSet;
use crate::story::{Story, StoryContext};

/// The outcome of rendering a story in isolation.
///
/// Holds the produced [`Element`] tree, a pretty-printed tree rendering, and
/// cheap structural metrics captured from the snapshot, plus the number of
/// reactive nodes that were live in the isolated runtime at the end of the
/// render (useful for asserting isolation).
#[derive(Clone, Debug, PartialEq)]
pub struct RenderResult {
    element: Element,
    tree: String,
    node_count: usize,
    depth: usize,
    live_nodes: usize,
}

impl RenderResult {
    /// Borrows the rendered element tree.
    #[must_use]
    pub fn element(&self) -> &Element {
        &self.element
    }

    /// Consumes the result, yielding the owned element tree.
    #[must_use]
    pub fn into_element(self) -> Element {
        self.element
    }

    /// The deterministic, indented tree rendering for snapshots and debugging.
    #[must_use]
    pub fn tree(&self) -> &str {
        &self.tree
    }

    /// The total number of nodes in the rendered tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// The depth of the rendered tree (a lone root has depth `1`).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// The number of reactive nodes live in the isolated runtime after render.
    #[must_use]
    pub fn live_nodes(&self) -> usize {
        self.live_nodes
    }
}

/// Renders `story` with `args` in a private runtime and an empty context.
///
/// This is the common entry point; use [`render_story_with_context`] to seed
/// dependency injection values the story may read.
#[must_use]
pub fn render_story(story: &Story, args: &ArgSet) -> RenderResult {
    let context = ContextMap::new();
    render_story_with_context(story, args, &context)
}

/// Renders `story` with `args` against a caller-provided `context`.
///
/// A fresh reactive [`Runtime`] is created for this render alone and dropped
/// when it returns, guaranteeing isolation from any other story's render. The
/// `context` is scoped with [`ContextMap::child`] so the story cannot mutate the
/// caller's map.
#[must_use]
pub fn render_story_with_context(
    story: &Story,
    args: &ArgSet,
    context: &ContextMap,
) -> RenderResult {
    let runtime = Runtime::new();
    let scoped = context.child();
    let component = ComponentCtx::new(&scoped);

    let element = render_with_context(&component, |component| {
        let ctx = StoryContext::new(args, &runtime, component);
        story.render(&ctx)
    });

    let snap = snapshot(&element);
    let node_count = snap.node_count();
    let depth = snap.depth();
    let tree = render_tree(&snap);
    let live_nodes = runtime.live_nodes();

    RenderResult {
        element,
        tree,
        node_count,
        depth,
        live_nodes,
    }
}

/// Renders `story` with its own [`Story::default_args`] in a private runtime.
#[must_use]
pub fn render_story_default(story: &Story) -> RenderResult {
    render_story(story, story.default_args())
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::controls::ControlValue;
    use crate::story::Story;
    use prism_ui::{Element, ElementKind};

    #[test]
    fn render_produces_element_and_tree() {
        let story = Story::builder("Greeting")
            .arg("name", ControlValue::Text("Loom".into()))
            .build(|ctx| {
                let name = ctx.text_arg("name").unwrap_or("world");
                Element::box_()
                    .class("greeting")
                    .child(Element::text(alloc::format!("Hello, {name}")))
            });

        let result = render_story_default(&story);
        assert_eq!(result.element().kind(), &ElementKind::Box);
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.depth(), 2);
        assert_eq!(result.tree(), "Box\n  Text \"Hello, Loom\"\n");
    }

    #[test]
    fn renders_are_isolated_across_calls() {
        // A story that allocates reactive nodes in the ambient runtime.
        let story = Story::builder("Reactive")
            .arg("start", ControlValue::Number(1.0))
            .build(|ctx| {
                let start = ctx.number_arg("start").unwrap_or(0.0);
                let signal = ctx.runtime().signal(start);
                let memo = ctx.runtime().memo({
                    let signal = signal.clone();
                    move || signal.get() + 1.0
                });
                let value = memo.get();
                Element::text(alloc::format!("{value}"))
            });

        let first = render_story_default(&story);
        let second = render_story_default(&story);

        // Each render stands up the same number of live nodes: no accumulation
        // across calls proves the runtimes are independent.
        assert_eq!(first.live_nodes(), 2);
        assert_eq!(second.live_nodes(), first.live_nodes());
        assert_eq!(first.element().text_content(), Some("2"));
    }

    #[test]
    fn provided_context_is_injectable_but_scoped() {
        struct Theme {
            accent: &'static str,
        }

        let story = Story::builder("Themed").build(|ctx| {
            let accent = ctx.inject::<Theme>().map_or("none", |theme| theme.accent);
            Element::box_().class(accent)
        });

        let mut context = ContextMap::new();
        context.provide(Theme { accent: "violet" });

        let result = render_story_with_context(&story, story.default_args(), &context);
        assert_eq!(result.element().class_names(), &["violet".to_string()]);
    }
}
