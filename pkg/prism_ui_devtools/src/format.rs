//! Deterministic pretty-printing of a [`TreeSnapshot`].
//!
//! [`render_tree`] turns a [`TreeSnapshot`] into an indented, multi-line
//! `String` suitable for debugging and golden/snapshot tests. Output is
//! deterministic: siblings keep their order and each level is indented by two
//! spaces.

use alloc::string::String;
use core::fmt::Write as _;

use crate::snapshot::{SnapshotNode, TreeSnapshot};

/// Appends the rendered form of `node` (and its subtree) to `out`.
fn render_node(node: &SnapshotNode, depth: usize, out: &mut String) {
    for _ in 0..depth {
        out.push_str("  ");
    }
    match &node.text {
        Some(text) => {
            // `{text:?}` yields a quoted, escaped form, e.g. `Text "hello"`.
            let _ = write!(out, "{} {text:?}", node.kind);
        }
        None => {
            let _ = write!(out, "{}", node.kind);
        }
    }
    out.push('\n');
    for child in &node.children {
        render_node(child, depth + 1, out);
    }
}

/// Renders a [`TreeSnapshot`] to an indented, multi-line `String`.
///
/// Each node occupies one line, indented by two spaces per level of depth. Text
/// nodes render their content in quotes after the kind label, e.g.
/// `Text "hello"`. Every line, including the last, ends with a newline.
#[must_use]
pub fn render_tree(tree: &TreeSnapshot) -> String {
    let mut out = String::new();
    render_node(&tree.root, 0, &mut out);
    out
}
