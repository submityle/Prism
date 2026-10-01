//! The declarative view spec.
//!
//! An [`Element`] is a lightweight, cheap-to-build description of *what* a piece
//! of UI should look like this frame. It carries no identity and performs no
//! allocation beyond its own fields; identity, layout and painting are resolved
//! later by the [`Ui`](crate::Ui) runtime against the retained tree.
//!
//! This is deliberately data-only (the "Widget" half of Flutter's
//! Widget/Element split): building an `Element` tree every frame is cheap, and
//! the runtime reconciles it against the retained state so unchanged nodes are
//! reused.

use alloc::string::String;
use alloc::vec::Vec;

use prism_ui_style::{StyleProp, StyleValue};

/// A reconciliation key identifying a child across re-renders.
///
/// Explicitly keyed children (e.g. list items) should use [`Key::Int`] or
/// [`Key::Str`]. Children without an explicit key are assigned a positional
/// [`Key::Index`] by the runtime, which is stable as long as the sibling
/// structure is stable.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Key {
    /// A positional key assigned by sibling order.
    Index(usize),
    /// An explicit integer key.
    Int(i64),
    /// An explicit string key.
    Str(String),
}

/// The kind of box an [`Element`] represents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElementKind {
    /// A layout box with no intrinsic content.
    Box,
    /// A text run; the string lives in [`Element::text`].
    Text,
    /// An application-defined element kind, identified by name. The runtime
    /// treats it like a box but forwards the name to the backend so a custom
    /// visual can be attached.
    Custom(String),
}

/// A declarative description of a UI node and its subtree.
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    pub(crate) kind: ElementKind,
    pub(crate) key: Option<Key>,
    pub(crate) classes: Vec<String>,
    pub(crate) inline: Vec<(StyleProp, StyleValue)>,
    pub(crate) text: Option<String>,
    pub(crate) children: Vec<Element>,
}

impl Element {
    fn with_kind(kind: ElementKind) -> Self {
        Self {
            kind,
            key: None,
            classes: Vec::new(),
            inline: Vec::new(),
            text: None,
            children: Vec::new(),
        }
    }

    /// Creates a layout box.
    #[must_use]
    pub fn box_() -> Self {
        Self::with_kind(ElementKind::Box)
    }

    /// Creates a text run with the given content.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        let mut el = Self::with_kind(ElementKind::Text);
        el.text = Some(content.into());
        el
    }

    /// Creates a custom, backend-defined element kind.
    #[must_use]
    pub fn custom(name: impl Into<String>) -> Self {
        Self::with_kind(ElementKind::Custom(name.into()))
    }

    /// Sets an explicit integer reconciliation key.
    #[must_use]
    pub fn key_int(mut self, key: i64) -> Self {
        self.key = Some(Key::Int(key));
        self
    }

    /// Sets an explicit string reconciliation key.
    #[must_use]
    pub fn key_str(mut self, key: impl Into<String>) -> Self {
        self.key = Some(Key::Str(key.into()));
        self
    }

    /// Adds a style class, applied by the cascade in order.
    #[must_use]
    pub fn class(mut self, name: impl Into<String>) -> Self {
        self.classes.push(name.into());
        self
    }

    /// Adds an inline style override, applied on top of classes.
    #[must_use]
    pub fn style(mut self, prop: StyleProp, value: StyleValue) -> Self {
        self.inline.push((prop, value));
        self
    }

    /// Appends a child element.
    #[must_use]
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Appends many children.
    #[must_use]
    pub fn children<I: IntoIterator<Item = Element>>(mut self, children: I) -> Self {
        self.children.extend(children);
        self
    }

    /// The element's kind.
    #[must_use]
    pub fn kind(&self) -> &ElementKind {
        &self.kind
    }

    /// The element's explicit key, if any.
    #[must_use]
    pub fn explicit_key(&self) -> Option<&Key> {
        self.key.as_ref()
    }

    /// The element's text content, if it is a text run.
    #[must_use]
    pub fn text_content(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// The element's children.
    #[must_use]
    pub fn child_elements(&self) -> &[Element] {
        &self.children
    }

    /// The element's style class names, in application order.
    #[must_use]
    pub fn class_names(&self) -> &[String] {
        &self.classes
    }

    /// The element's inline style overrides, in application order.
    #[must_use]
    pub fn inline_pairs(&self) -> &[(StyleProp, StyleValue)] {
        &self.inline
    }

    /// Resolves the key used for reconciliation at sibling position `index`.
    pub(crate) fn resolved_key(&self, index: usize) -> Key {
        self.key.clone().unwrap_or(Key::Index(index))
    }
}
