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

use crate::budget::NodeBudget;
use crate::capability::CapabilitySet;
use crate::schema::RemoteNode;

/// The default maximum nesting depth the sandbox will descend into.
///
/// Untrusted input can be arbitrarily deep; capping the recursion keeps
/// sanitation from exhausting the stack on hostile input. Nodes below the cap
/// are replaced by placeholders rather than visited.
pub const DEFAULT_MAX_DEPTH: usize = 128;

/// The default upper bound on the total number of nodes a single sanitation
/// pass will emit.
///
/// Where [`DEFAULT_MAX_DEPTH`] bounds how *deep* an untrusted tree may be, this
/// bounds how *large* it may be overall. A document that is shallow but
/// extremely wide is cheap to encode yet expensive to materialize; capping the
/// node count keeps sanitation (and the downstream element tree it feeds) within
/// a predictable memory envelope. The default is generous enough for any
/// plausible real UI while still foreclosing breadth-based amplification.
pub const DEFAULT_MAX_NODES: usize = 65_536;

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
    /// The node budget was exhausted before this node could be emitted; it and
    /// its following siblings were truncated from the output.
    BudgetExceeded,
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
            DiagnosticKind::BudgetExceeded => {
                format!("truncated node `{}` exceeding node budget", self.capability)
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
    max_nodes: usize,
}

impl Sandbox {
    /// Creates a sandbox enforcing `caps` with the default depth cap.
    #[must_use]
    pub fn new(caps: CapabilitySet) -> Self {
        Self {
            caps,
            max_depth: DEFAULT_MAX_DEPTH,
            max_nodes: DEFAULT_MAX_NODES,
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

    /// Overrides the maximum number of nodes the sandbox will emit in a single
    /// pass.
    ///
    /// The value is clamped to at least `1` so the sandbox can always emit a
    /// root. See [`DEFAULT_MAX_NODES`] for the rationale behind the cap.
    #[must_use]
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes.max(1);
        self
    }

    /// The configured maximum number of emitted nodes.
    #[must_use]
    pub fn max_nodes(&self) -> usize {
        self.max_nodes
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
        let mut budget = NodeBudget::new(self.max_nodes);
        let root = self
            .sanitize_node(root, 0, &mut budget, &mut path, &mut diagnostics)
            // `max_nodes` is clamped to at least `1`, so a fresh budget always
            // has room for the root; this fallback only guards the degenerate
            // path and keeps the output total within the node cap.
            .unwrap_or_else(|| RemoteNode::fallback("node budget exhausted"));
        SanitizeResult { root, diagnostics }
    }

    fn sanitize_node(
        &self,
        node: &RemoteNode,
        depth: usize,
        budget: &mut NodeBudget,
        path: &mut Vec<usize>,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Option<RemoteNode> {
        // Reserve one unit of the node budget for whatever single node this call
        // commits (a real node or a placeholder). When the budget is already
        // exhausted no node can be emitted, so the caller truncates this slot.
        if !budget.try_consume() {
            return None;
        }

        if depth > self.max_depth {
            diagnostics.push(Diagnostic::new(
                path.clone(),
                DiagnosticKind::DepthExceeded,
                node.kind(),
            ));
            return Some(RemoteNode::fallback("depth limit exceeded"));
        }

        if !self.caps.allows_kind(node.kind()) {
            diagnostics.push(Diagnostic::new(
                path.clone(),
                DiagnosticKind::RejectedKind,
                node.kind(),
            ));
            // Prune the untrusted subtree: a rejected node's children are not
            // visited, so a blocked kind cannot smuggle content through.
            return Some(RemoteNode::fallback(format!(
                "blocked kind `{}`",
                node.kind()
            )));
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
            let sanitized_child =
                self.sanitize_node(child, depth + 1, budget, path, diagnostics);
            match sanitized_child {
                Some(sanitized_child) => safe = safe.child(sanitized_child),
                None => {
                    // The budget ran out mid-tree. Record the truncation of the
                    // first dropped child (its siblings follow it) and stop; the
                    // drop is diagnosed, never silent.
                    diagnostics.push(Diagnostic::new(
                        path.clone(),
                        DiagnosticKind::BudgetExceeded,
                        child.kind(),
                    ));
                    path.pop();
                    break;
                }
            }
            path.pop();
        }

        Some(safe)
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

    fn count_nodes(node: &RemoteNode) -> usize {
        1 + node.child_nodes().iter().map(count_nodes).sum::<usize>()
    }

    /// Builds a deterministic pseudo-random tree of untrusted nodes so budget
    /// behaviour can be checked against many shapes without external input.
    fn fuzz_tree(seed: u64, nodes: usize) -> RemoteNode {
        let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            // SplitMix64 — a small, deterministic generator (no `prism_math`
            // dependency needed for test-only shaping).
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        let kinds = ["box", "text", "script", "card", "sdui.fallback"];
        let mut roots: Vec<RemoteNode> = Vec::new();
        for _ in 0..nodes {
            let kind = kinds[(next() as usize) % kinds.len()];
            let mut node = RemoteNode::new(kind);
            // Attach a few existing nodes as children to grow depth and breadth.
            let take = (next() as usize) % 4;
            for _ in 0..take {
                if let Some(child) = roots.pop() {
                    node = node.child(child);
                }
            }
            roots.push(node);
        }
        // Fold any leftovers under a single whitelisted root.
        let mut root = RemoteNode::new("box");
        for child in roots {
            root = root.child(child);
        }
        root
    }

    #[test]
    fn output_never_exceeds_node_budget() {
        // Core budget invariant: for any input and any limit, the sanitized
        // tree holds at most `limit` nodes, and the limit is never breached.
        let caps = CapabilitySet::new()
            .allow_kinds(["box", "text", "card", "sdui.fallback"]);
        for seed in 0..40u64 {
            let tree = fuzz_tree(seed, 60);
            let total = count_nodes(&tree);
            for &limit in &[1usize, 2, 5, 13, 64, 256] {
                let sandbox = Sandbox::new(caps.clone()).with_max_nodes(limit);
                let result = sandbox.sanitize(&tree);
                let emitted = count_nodes(&result.root);
                assert!(
                    emitted <= limit,
                    "seed {seed} limit {limit}: emitted {emitted} > limit (input {total})"
                );
                // Whenever a budget truncation is reported the budget must have
                // been spent in full, so the output sits exactly at the cap.
                let budget_hit = result
                    .diagnostics
                    .iter()
                    .any(|d| d.kind == DiagnosticKind::BudgetExceeded);
                if budget_hit {
                    assert_eq!(
                        emitted, limit,
                        "seed {seed} limit {limit}: truncated but emitted {emitted} != limit"
                    );
                }
            }
        }
    }

    #[test]
    fn within_budget_whitelisted_tree_is_untouched() {
        // A fully whitelisted tree that fits inside the budget must pass through
        // byte-for-byte with no budget diagnostics.
        let remote = RemoteNode::new("box")
            .child(RemoteNode::text("a"))
            .child(RemoteNode::new("box").child(RemoteNode::text("b")));
        let total = count_nodes(&remote);
        let sandbox = Sandbox::new(
            CapabilitySet::new().allow_kinds(["box", "text"]),
        )
        .with_max_nodes(total);
        let result = sandbox.sanitize(&remote);
        assert_eq!(result.root, remote);
        assert!(result
            .diagnostics
            .iter()
            .all(|d| d.kind != DiagnosticKind::BudgetExceeded));
    }

    #[test]
    fn exceeding_budget_truncates_siblings_with_diagnostic() {
        // Four whitelisted children but a budget for root + two of them: the
        // remaining siblings are truncated and the drop is recorded.
        let remote = RemoteNode::new("box")
            .child(RemoteNode::new("box"))
            .child(RemoteNode::new("box"))
            .child(RemoteNode::new("box"))
            .child(RemoteNode::new("box"));
        let sandbox =
            Sandbox::new(CapabilitySet::new().allow_kind("box")).with_max_nodes(3);
        let result = sandbox.sanitize(&remote);
        assert_eq!(count_nodes(&result.root), 3);
        assert_eq!(result.root.child_nodes().len(), 2);
        let budget_diags: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| d.kind == DiagnosticKind::BudgetExceeded)
            .collect();
        assert_eq!(budget_diags.len(), 1);
        // The diagnostic points at the first dropped child (index 2).
        assert_eq!(budget_diags[0].path, vec![2]);
    }

    #[test]
    fn max_nodes_is_clamped_to_at_least_one() {
        // A zero budget would otherwise leave no room for the root; the clamp
        // guarantees a single safe fallback instead of an empty result.
        let sandbox =
            Sandbox::new(CapabilitySet::new().allow_kind("box")).with_max_nodes(0);
        assert_eq!(sandbox.max_nodes(), 1);
        let result = sandbox.sanitize(&RemoteNode::new("box").child(RemoteNode::new("box")));
        assert_eq!(count_nodes(&result.root), 1);
    }

    #[test]
    fn budget_diagnostic_message_is_descriptive() {
        let remote = RemoteNode::new("box")
            .child(RemoteNode::new("box"))
            .child(RemoteNode::new("box"));
        let sandbox =
            Sandbox::new(CapabilitySet::new().allow_kind("box")).with_max_nodes(2);
        let result = sandbox.sanitize(&remote);
        let diag = result
            .diagnostics
            .iter()
            .find(|d| d.kind == DiagnosticKind::BudgetExceeded)
            .expect("a budget diagnostic");
        assert_eq!(diag.message(), "truncated node `box` exceeding node budget");
    }

    #[test]
    fn sanitation_is_deterministic() {
        let tree = fuzz_tree(7, 50);
        let caps = CapabilitySet::new().allow_kinds(["box", "text", "card"]);
        let sandbox = Sandbox::new(caps).with_max_nodes(32);
        let a = sandbox.sanitize(&tree);
        let b = sandbox.sanitize(&tree);
        assert_eq!(a, b);
    }
}
