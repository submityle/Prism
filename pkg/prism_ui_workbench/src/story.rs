//! The [`Story`] model: a named, parameterised component use case.
//!
//! A story pairs a human-readable name with a render function that turns the
//! ambient [`StoryContext`] (its current [`ArgSet`], an isolated reactive
//! [`Runtime`], and a component [`ComponentCtx`]) into a [`prism_ui::Element`]
//! tree. Changing an argument and re-rendering is how a single component's many
//! states are explored — exactly the Storybook model, expressed over Loom's
//! data-only element trees.
//!
//! Build stories with [`StoryBuilder`], which collects default arguments and
//! takes the render closure in its terminal [`StoryBuilder::build`] call so a
//! story is never left half-constructed.

#![forbid(unsafe_code)]

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::string::String;

use prism_ui::Element;
use prism_ui_component::ComponentCtx;
use prism_ui_reactive::Runtime;

use crate::controls::{ArgSet, ControlValue};

/// The boxed render function a [`Story`] holds.
///
/// It is invoked once per render with a freshly built [`StoryContext`], and is
/// expected to be pure with respect to that context so repeated renders of the
/// same arguments are deterministic.
type RenderFn = Box<dyn Fn(&StoryContext<'_>) -> Element>;

/// The ambient environment handed to a story's render function.
///
/// It bundles the three things a story may read while rendering:
///
/// * the current [`ArgSet`] (its controls),
/// * an isolated reactive [`Runtime`] it may use for signals and memos, and
/// * a component [`ComponentCtx`] for dependency injection and slots.
///
/// The harness builds a fresh context — including a brand-new [`Runtime`] — for
/// every render, which is what keeps stories isolated from one another.
pub struct StoryContext<'a> {
    args: &'a ArgSet,
    runtime: &'a Runtime,
    component: &'a ComponentCtx<'a>,
}

impl<'a> StoryContext<'a> {
    /// Bundles the pieces a story renders against into a context.
    #[must_use]
    pub fn new(args: &'a ArgSet, runtime: &'a Runtime, component: &'a ComponentCtx<'a>) -> Self {
        Self {
            args,
            runtime,
            component,
        }
    }

    /// The current argument set driving this render.
    #[must_use]
    pub fn args(&self) -> &ArgSet {
        self.args
    }

    /// The isolated reactive runtime for this render.
    #[must_use]
    pub fn runtime(&self) -> &Runtime {
        self.runtime
    }

    /// The component context for dependency injection and slots.
    #[must_use]
    pub fn component(&self) -> &ComponentCtx<'a> {
        self.component
    }

    /// Injects a shared value of type `T` from the component context.
    #[must_use]
    pub fn inject<T: 'static>(&self) -> Option<Rc<T>> {
        self.component.inject::<T>()
    }

    /// Reads a boolean argument by key.
    #[must_use]
    pub fn bool_arg(&self, key: &str) -> Option<bool> {
        self.args.get_bool(key)
    }

    /// Reads a numeric argument by key.
    #[must_use]
    pub fn number_arg(&self, key: &str) -> Option<f64> {
        self.args.get_number(key)
    }

    /// Reads a text argument by key.
    #[must_use]
    pub fn text_arg(&self, key: &str) -> Option<&str> {
        self.args.get_text(key)
    }

    /// Reads the selected option of a select argument by key.
    #[must_use]
    pub fn selected_arg(&self, key: &str) -> Option<&str> {
        self.args.get_selected(key)
    }
}

/// A named, parameterised use case that renders a component in one state.
///
/// Construct via [`Story::builder`] / [`StoryBuilder`]. A story owns its default
/// [`ArgSet`] and a render function; the [`harness`](crate::harness) drives the
/// render inside an isolated runtime.
pub struct Story {
    name: String,
    default_args: ArgSet,
    render: RenderFn,
}

impl Story {
    /// Starts building a story with the given name.
    #[must_use]
    pub fn builder(name: impl Into<String>) -> StoryBuilder {
        StoryBuilder::new(name)
    }

    /// The story's name (its leaf label within a workbench group).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The story's default arguments, used when no overrides are supplied.
    #[must_use]
    pub fn default_args(&self) -> &ArgSet {
        &self.default_args
    }

    /// Renders the story against `ctx`, producing its element tree.
    #[must_use]
    pub fn render(&self, ctx: &StoryContext<'_>) -> Element {
        (self.render)(ctx)
    }
}

impl core::fmt::Debug for Story {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Story")
            .field("name", &self.name)
            .field("default_args", &self.default_args)
            .finish_non_exhaustive()
    }
}

/// A builder that accumulates a [`Story`]'s name and default arguments.
///
/// The render closure is supplied last, to [`StoryBuilder::build`], so the
/// builder can never yield a story without a render function.
pub struct StoryBuilder {
    name: String,
    default_args: ArgSet,
}

impl StoryBuilder {
    /// Creates a builder for a story named `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            default_args: ArgSet::new(),
        }
    }

    /// Adds a default argument control under `key`.
    #[must_use]
    pub fn arg(mut self, key: impl Into<String>, value: ControlValue) -> Self {
        self.default_args = self.default_args.with(key, value);
        self
    }

    /// Replaces the builder's default argument set wholesale.
    #[must_use]
    pub fn args(mut self, args: ArgSet) -> Self {
        self.default_args = args;
        self
    }

    /// Finalises the story with its render function.
    #[must_use]
    pub fn build<F>(self, render: F) -> Story
    where
        F: Fn(&StoryContext<'_>) -> Element + 'static,
    {
        Story {
            name: self.name,
            default_args: self.default_args,
            render: Box::new(render),
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use prism_ui::ElementKind;
    use prism_ui_component::ContextMap;

    fn render_once(story: &Story, args: &ArgSet) -> Element {
        let runtime = Runtime::new();
        let context = ContextMap::new();
        let component = ComponentCtx::new(&context);
        let ctx = StoryContext::new(args, &runtime, &component);
        story.render(&ctx)
    }

    #[test]
    fn builder_sets_name_and_defaults() {
        let story = Story::builder("Primary")
            .arg("filled", ControlValue::Bool(true))
            .build(|_ctx| Element::box_());

        assert_eq!(story.name(), "Primary");
        assert_eq!(story.default_args().get_bool("filled"), Some(true));
    }

    #[test]
    fn render_depends_on_args() {
        let story = Story::builder("Button")
            .arg("filled", ControlValue::Bool(true))
            .build(|ctx| {
                let class = if ctx.bool_arg("filled").unwrap_or(false) {
                    "filled"
                } else {
                    "outline"
                };
                Element::box_().class(class)
            });

        let filled = render_once(&story, story.default_args());
        assert_eq!(filled.class_names(), &["filled".to_string()]);

        let mut args = story.default_args().clone();
        args.set_bool("filled", false).expect("same kind");
        let outline = render_once(&story, &args);
        assert_eq!(outline.class_names(), &["outline".to_string()]);
        assert_eq!(outline.kind(), &ElementKind::Box);
    }

    #[test]
    fn render_can_use_isolated_runtime() {
        let story = Story::builder("Counter")
            .arg("start", ControlValue::Number(3.0))
            .build(|ctx| {
                let start = ctx.number_arg("start").unwrap_or(0.0);
                let count = ctx.runtime().signal(start);
                let doubled = ctx.runtime().memo({
                    let count = count.clone();
                    move || count.get() * 2.0
                });
                let value = doubled.get();
                Element::box_().child(Element::text(alloc::format!("{value}")))
            });

        let element = render_once(&story, story.default_args());
        assert_eq!(element.child_elements()[0].text_content(), Some("6"));
    }
}
