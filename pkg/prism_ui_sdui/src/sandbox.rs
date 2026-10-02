//! The structural sandbox.
//!
//! [`Sandbox::sanitize`] walks an untrusted [`RemoteNode`] tree and rewrites it
//! into a tree that references only whitelisted capabilities. Nothing is
//! silently dropped: a rejected node is replaced by a safe
//! [`KIND_FALLBACK`](crate::KIND_FALLBACK) placeholder, stripped styles and
//! events are recorded, and every rewrite produces a [`Diagnostic`] describing
//! what happened and where.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::capability::CapabilitySet;
use crate::schema::RemoteNode;

/// The default maximum nesting depth the sandbox will descend into.
///
/// Untrusted input can be arbitrarily deep; capping the recursion keeps
/// sanitation from exhausting the stack on hostile input. Nodes below the cap
/// are replaced by placeholders rather than visited.
pub const DEFAULT_MAX_DEPTH: usize = 128;

/// Why a node or attribute was rewritten during sanitation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticKind {
    /// The node's component kind was not whitelisted; the whole node (and its
    /// subtree) was replaced by a placeholder.
    RejectedKind,
    /// A style token was not whitelisted and was removed from the node.
    StrippedStyle,
    /// An event name was not whitelisted and was removed from the node.
    StrippedEvent,
    /// The node sat below the configured depth cap and was replaced by a
    /// placeholder without descending further.
    DepthExceeded,
}

/// A record of one rewrite the sandbox performed.
///
/// The `path` is the chain of child indices from the sanitized root to the
/// affected node (an empty path denotes the root). `capability` is the
/// offending kind, style token or event name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// Child-index path from the root to the affected node.
    pub path: Vec<usize>,
    /// What kind of rewrite occurred.
    pub kind: DiagnosticKind,
    /// The offending capability name.
    pub capability: String,
}

impl Diagnostic {
    pub(crate) fn new(
        path: Vec<usize>,
        kind: DiagnosticKind,
        capability: impl Into<String>,
    ) -> Self {
        Self {
            path,
            kind,
            capability: capability.into(),
        }
    }

    /// A human-readable description of this diagnostic.
    #[must_use]
    pub fn message(&self) -> String {
        match self.kind {
            DiagnosticKind::RejectedKind => {
                format!("rejected non-whitelisted node kind `{}`", self.capability)
            }
            DiagnosticKind::StrippedStyle => {
                format!("stripped non-whitelisted style token `{}`", self.capability)
            }
            DiagnosticKind::StrippedEvent => {
                format!("stripped non-whitelisted event `{}`", self.capability)
            }
            DiagnosticKind::DepthExceeded => {
                format!("replaced node `{}` exceeding depth cap", self.capability)
            }
        }
    }
}

/// The result of [`Sandbox::sanitize`]: the rewritten tree plus diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SanitizeResult {
    /// The sanitized tree, safe to hand to [`decode`](crate::decode).
    pub root: RemoteNode,
    /// Every rewrite the sandbox performed, in pre-order of discovery.
    pub diagnostics: Vec<Diagnostic>,
}

impl SanitizeResult {
    /// Whether the input passed through untouched (no rewrites were needed).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

/// A capability-whitelist sandbox over a [`CapabilitySet`].
#[derive(Clone, Debug)]
pub struct Sandbox {
    caps: CapabilitySet,
    max_depth: usize,
}

impl Sandbox {
    /// Creates a sandbox enforcing `caps` with the default depth cap.
    #[must_use]
    pub fn new(caps: CapabilitySet) -> Self {
        Self {
            caps,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Overrides the maximum nesting depth the sandbox will descend into.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// The capabilities this sandbox enforces.
    #[must_use]
    pub fn capabilities(&self) -> &CapabilitySet {
        &self.caps
    }

    /// The configured maximum nesting depth.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Rewrites `root` into a tree referencing only whitelisted capabilities.
    ///
    /// This never panics and never recurses deeper than
    /// [`max_depth`](Sandbox::max_depth), so it is safe to run on fully
    /// untrusted input.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_ui_sdui::{CapabilitySet, RemoteNode, Sandbox, KIND_FALLBACK};
    ///
    /// let sandbox = Sandbox::new(CapabilitySet::new().allow_kind("box"));
    /// let remote = RemoteNode::new("box").child(RemoteNode::new("script"));
    /// let result = sandbox.sanitize(&remote);
    ///
    /// // The disallowed `script` child becomes a safe placeholder.
    /// assert_eq!(result.diagnostics.len(), 1);
    /// assert_eq!(result.root.child_nodes()[0].kind(), KIND_FALLBACK);
    /// ```
    #[must_use]
    pub fn sanitize(&self, root: &RemoteNode) -> SanitizeResult {
        let mut diagnostics = Vec::new();
        let mut path = Vec::new();
        let root = self.sanitize_node(root, 0, &mut path, &mut diagnostics);
        SanitizeResult { root, diagnostics }
    }

    fn sanitize_node(
        &self,
        node: &RemoteNode,
        depth: usize,
        path: &mut Vec<usize>,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> RemoteNode {
        if depth > self.max_depth {
            diagnostics.push(Diagnostic::new(
                path.clone(),
                DiagnosticKind::DepthExceeded,
                node.kind(),
            ));
            return RemoteNode::fallback("depth limit exceeded");
        }

        if !self.caps.allows_kind(node.kind()) {
            diagnostics.push(Diagnostic::new(
                path.clone(),
                DiagnosticKind::RejectedKind,
                node.kind(),
            ));
            // Prune the untrusted subtree: a rejected node's children are not
            // visited, so a blocked kind cannot smuggle content through.
            return RemoteNode::fallback(format!("blocked kind `{}`", node.kind()));
        }

        let mut safe = RemoteNode::new(node.kind());
        if let Some(text) = node.text_content() {
            safe = safe.with_text(text);
        }

        for token in node.style_tokens() {
            if self.caps.allows_style(token) {
                safe = safe.style(token.clone());
            } else {
                diagnostics.push(Diagnostic::new(
                    path.clone(),
                    DiagnosticKind::StrippedStyle,
                    token.clone(),
                ));
            }
        }

        for event in node.event_names() {
            if self.caps.allows_event(event) {
                safe = safe.event(event.clone());
            } else {
                diagnostics.push(Diagnostic::new(
                    path.clone(),
                    DiagnosticKind::StrippedEvent,
                    event.clone(),
                ));
            }
        }

        for (index, child) in node.child_nodes().iter().enumerate() {
            path.push(index);
            let sanitized_child = self.sanitize_node(child, depth + 1, path, diagnostics);
            safe = safe.child(sanitized_child);
            path.pop();
        }

        safe
    }
}

#[cfg(test)]
mod tests {
    use super::{DiagnosticKind, Sandbox};
    use crate::capability::CapabilitySet;
    use crate::schema::{RemoteNode, KIND_FALLBACK};

    fn sandbox() -> Sandbox {
        Sandbox::new(
            CapabilitySet::new()
                .allow_kinds(["box", "text"])
                .allow_style("card")
                .allow_event("tap"),
        )
    }

    #[test]
    fn whitelisted_tree_passes_through_unchanged() {
        let remote = RemoteNode::new("box")
            .style("card")
            .event("tap")
            .child(RemoteNode::text("hello"));
        let result = sandbox().sanitize(&remote);
        assert!(result.is_clean());
        assert_eq!(result.root, remote);
    }

    #[test]
    fn rejected_kind_becomes_placeholder_with_diagnostic() {
        let remote =
            RemoteNode::new("box").child(RemoteNode::new("script").child(RemoteNode::text("x")));
        let result = sandbox().sanitize(&remote);
        assert_eq!(result.root.child_nodes()[0].kind(), KIND_FALLBACK);
        assert_eq!(result.diagnostics.len(), 1);
        let diag = &result.diagnostics[0];
        assert_eq!(diag.kind, DiagnosticKind::RejectedKind);
        assert_eq!(diag.capability, "script");
        assert_eq!(diag.path, vec![0]);
    }

    #[test]
    fn rejected_node_subtree_is_pruned() {
        // The blocked node's own child must not survive sanitation.
        let remote = RemoteNode::new("evil").child(RemoteNode::new("box"));
        let result = sandbox().sanitize(&remote);
        assert_eq!(result.root.kind(), KIND_FALLBACK);
        assert!(result.root.child_nodes().is_empty());
    }

    #[test]
    fn non_whitelisted_style_and_event_are_stripped() {
        let remote = RemoteNode::new("box")
            .style("card")
            .style("danger")
            .event("tap")
            .event("exfiltrate");
        let result = sandbox().sanitize(&remote);
        assert_eq!(result.root.style_tokens(), &["card".to_string()]);
        assert_eq!(result.root.event_names(), &["tap".to_string()]);
        assert_eq!(result.diagnostics.len(), 2);
        assert_eq!(result.diagnostics[0].kind, DiagnosticKind::StrippedStyle);
        assert_eq!(result.diagnostics[0].capability, "danger");
        assert_eq!(result.diagnostics[1].kind, DiagnosticKind::StrippedEvent);
        assert_eq!(result.diagnostics[1].capability, "exfiltrate");
    }

    #[test]
    fn diagnostic_paths_track_nested_position() {
        let remote = RemoteNode::new("box")
            .child(RemoteNode::new("box"))
            .child(RemoteNode::new("box").child(RemoteNode::new("script")));
        let result = sandbox().sanitize(&remote);
        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(result.diagnostics[0].path, vec![1, 0]);
    }

    #[test]
    fn deeply_nested_input_does_not_panic_and_caps_depth() {
        // Build a tree far deeper than the cap; sanitation must stay bounded.
        let mut node = RemoteNode::new("box");
        for _ in 0..1_000 {
            node = RemoteNode::new("box").child(node);
        }
        let sandbox = Sandbox::new(CapabilitySet::new().allow_kind("box")).with_max_depth(16);
        let result = sandbox.sanitize(&node);
        // At least one depth-exceeded placeholder was emitted.
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.kind == DiagnosticKind::DepthExceeded));
    }

    #[test]
    fn diagnostic_messages_are_descriptive() {
        let remote = RemoteNode::new("script");
        let result = sandbox().sanitize(&remote);
        assert_eq!(
            result.diagnostics[0].message(),
            "rejected non-whitelisted node kind `script`"
        );
    }

    #[test]
    fn fallback_kind_is_not_trusted_by_default() {
        // A remote node that impersonates the reserved placeholder kind is
        // rejected like any other non-whitelisted kind.
        let remote = RemoteNode::new(KIND_FALLBACK);
        let result = sandbox().sanitize(&remote);
        assert_eq!(result.root.kind(), KIND_FALLBACK);
        assert_eq!(result.diagnostics[0].kind, DiagnosticKind::RejectedKind);
    }
}
