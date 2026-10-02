//! The untrusted remote node schema.
//!
//! A [`RemoteDocument`] is the on-the-wire shape a server sends: a declared
//! schema [`Version`](crate::Version) plus a tree of [`RemoteNode`]s. Every
//! field is considered fully untrusted until it has passed through the
//! [`Sandbox`](crate::Sandbox); nothing here grants any capability on its own.

use alloc::string::String;
use alloc::vec::Vec;

use crate::version::Version;

/// The built-in layout-box node kind.
pub const KIND_BOX: &str = "box";

/// The built-in text-run node kind.
pub const KIND_TEXT: &str = "text";

/// The reserved node kind used for safe fallback placeholders.
///
/// The sandbox emits nodes of this kind when it prunes untrusted content, and
/// [`decode`](crate::decode) maps them to an error-boundary-style element. A
/// remote document that names this kind directly is treated like any other
/// non-whitelisted kind unless it is explicitly allowed.
pub const KIND_FALLBACK: &str = "sdui.fallback";

/// A single untrusted node in a remote description.
///
/// A node carries its component `kind`, optional text content, a list of style
/// tokens, a list of event names and its children. The builder methods are
/// convenient for constructing trees in tests and in native host code; data
/// arriving over the wire would populate the same shape through a (future)
/// decoder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RemoteNode {
    kind: String,
    text: Option<String>,
    styles: Vec<String>,
    events: Vec<String>,
    children: Vec<RemoteNode>,
}

impl RemoteNode {
    /// Creates a node of the given `kind` with no attributes or children.
    #[must_use]
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            ..Self::default()
        }
    }

    /// Creates a [`KIND_TEXT`] node carrying `content`.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            kind: String::from(KIND_TEXT),
            text: Some(content.into()),
            ..Self::default()
        }
    }

    /// Creates a [`KIND_FALLBACK`] placeholder node carrying `reason` as its
    /// text content.
    #[must_use]
    pub fn fallback(reason: impl Into<String>) -> Self {
        Self {
            kind: String::from(KIND_FALLBACK),
            text: Some(reason.into()),
            ..Self::default()
        }
    }

    /// Sets the node's text content.
    #[must_use]
    pub fn with_text(mut self, content: impl Into<String>) -> Self {
        self.text = Some(content.into());
        self
    }

    /// Adds a style token the node wishes to apply.
    #[must_use]
    pub fn style(mut self, token: impl Into<String>) -> Self {
        self.styles.push(token.into());
        self
    }

    /// Adds an event name the node wishes to bind.
    #[must_use]
    pub fn event(mut self, name: impl Into<String>) -> Self {
        self.events.push(name.into());
        self
    }

    /// Appends a child node.
    #[must_use]
    pub fn child(mut self, child: RemoteNode) -> Self {
        self.children.push(child);
        self
    }

    /// Appends many child nodes.
    #[must_use]
    pub fn children<I: IntoIterator<Item = RemoteNode>>(mut self, children: I) -> Self {
        self.children.extend(children);
        self
    }

    /// The node's component kind.
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The node's text content, if any.
    #[must_use]
    pub fn text_content(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// The node's requested style tokens, in order.
    #[must_use]
    pub fn style_tokens(&self) -> &[String] {
        &self.styles
    }

    /// The node's requested event names, in order.
    #[must_use]
    pub fn event_names(&self) -> &[String] {
        &self.events
    }

    /// The node's children.
    #[must_use]
    pub fn child_nodes(&self) -> &[RemoteNode] {
        &self.children
    }
}

/// A complete remote description: a declared schema version and a root node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteDocument {
    version: Version,
    root: RemoteNode,
}

impl RemoteDocument {
    /// Creates a document authored against `version` with the given `root`.
    #[must_use]
    pub fn new(version: Version, root: RemoteNode) -> Self {
        Self { version, root }
    }

    /// The schema version the document was authored against.
    #[must_use]
    pub fn version(&self) -> Version {
        self.version
    }

    /// The document's root node.
    #[must_use]
    pub fn root(&self) -> &RemoteNode {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::{RemoteDocument, RemoteNode, KIND_FALLBACK, KIND_TEXT};
    use crate::version::Version;

    #[test]
    fn builder_populates_all_fields() {
        let node = RemoteNode::new("button")
            .with_text("Buy")
            .style("primary")
            .event("tap")
            .child(RemoteNode::text("child"));
        assert_eq!(node.kind(), "button");
        assert_eq!(node.text_content(), Some("Buy"));
        assert_eq!(node.style_tokens(), &["primary".to_string()]);
        assert_eq!(node.event_names(), &["tap".to_string()]);
        assert_eq!(node.child_nodes().len(), 1);
    }

    #[test]
    fn text_and_fallback_constructors_set_kind() {
        assert_eq!(RemoteNode::text("hi").kind(), KIND_TEXT);
        let fb = RemoteNode::fallback("why");
        assert_eq!(fb.kind(), KIND_FALLBACK);
        assert_eq!(fb.text_content(), Some("why"));
    }

    #[test]
    fn document_exposes_version_and_root() {
        let doc = RemoteDocument::new(Version::new(2, 0), RemoteNode::new("box"));
        assert_eq!(doc.version(), Version::new(2, 0));
        assert_eq!(doc.root().kind(), "box");
    }
}
