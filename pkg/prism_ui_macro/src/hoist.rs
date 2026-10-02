//! Static-subtree classification for hoisting (§9.2).
//!
//! `SolidJS` clones a *static template* for subtrees that never change, skipping
//! per-node construction and diffing. Loom applies the same idea at lowering
//! time: a subtree with **no `$` reactive reads, no `for_each(..)` splice, and
//! only compile-time-constant literal values** is [`StaticClass::Static`] and
//! can be hoisted into a cloneable constant template. Everything else is a
//! dynamic "island" that participates in reconciliation.
//!
//! The core routine [`classify_subtree`] is a pure function over the parsed
//! [`Node`] AST, so it is exhaustively unit-testable without expanding the
//! macro.

use syn::Expr;

use crate::ast::{Attr, Child, ContentExpr, Node, NodeKind, StyleVal};

/// Whether a subtree can be hoisted into a cloneable constant template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StaticClass {
    /// The subtree has no dynamic inputs and can be hoisted and cloned.
    Static,
    /// The subtree reads a signal, splices a dynamic list, or embeds a runtime
    /// expression, so it must be constructed and reconciled at run time.
    Dynamic,
}

impl StaticClass {
    /// Returns `true` for [`StaticClass::Static`].
    #[cfg(test)]
    pub(crate) fn is_static(self) -> bool {
        matches!(self, StaticClass::Static)
    }
}

/// Classifies a node and its entire subtree as [`StaticClass::Static`] or
/// [`StaticClass::Dynamic`].
///
/// The result is `Static` only when every descendant is static; a single
/// dynamic input anywhere in the subtree makes the whole subtree dynamic
/// (hoisting a template requires the *entire* cloned fragment to be constant).
pub(crate) fn classify_subtree(node: &Node) -> StaticClass {
    if !node_self_is_static(node) {
        return StaticClass::Dynamic;
    }
    for child in &node.children {
        match child {
            // A dynamic list cannot be baked into a constant template.
            Child::ForEach(_) => return StaticClass::Dynamic,
            Child::Node(child_node) => {
                if classify_subtree(child_node) == StaticClass::Dynamic {
                    return StaticClass::Dynamic;
                }
            }
        }
    }
    StaticClass::Static
}

/// Classifies a node's *own* inputs (kind payload and attributes), ignoring its
/// children.
fn node_self_is_static(node: &Node) -> bool {
    let kind_static = match &node.kind {
        NodeKind::Box => true,
        NodeKind::Text(content) | NodeKind::Custom(content) => content_is_static(content),
    };
    kind_static && node.attrs.iter().all(attr_is_static)
}

/// A content expression is static when it is not a reactive read and its
/// expression is a plain literal (so it can be embedded in a constant).
fn content_is_static(content: &ContentExpr) -> bool {
    !content.reactive && expr_is_literal(&content.expr)
}

/// An attribute is static when all of its embedded expressions are literals.
fn attr_is_static(attr: &Attr) -> bool {
    match attr {
        Attr::Class(names) => names.iter().all(expr_is_literal),
        // Literal keys are always constant.
        Attr::KeyInt(_) | Attr::KeyStr(_) => true,
        Attr::Style(entries) => entries
            .iter()
            .all(|entry| style_val_is_static(&entry.value)),
    }
}

/// A style value is static when it is a literal number, a keyword, or a
/// constructor call whose arguments are all literals.
fn style_val_is_static(value: &StyleVal) -> bool {
    match value {
        StyleVal::Number(_) | StyleVal::Keyword(_) => true,
        StyleVal::Call { args, .. } => args.iter().all(expr_is_literal),
    }
}

/// Returns `true` when `expr` is a plain literal (`"s"`, `42`, `1.5`, `true`),
/// the only expressions that can be embedded in a hoisted constant template.
fn expr_is_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Lit(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_str;

    use crate::ast::LoomInput;

    /// Parses a `loom!` body into its root [`Node`] for classification tests.
    fn root(src: &str) -> Node {
        parse_str::<LoomInput>(src).expect("valid loom input").node
    }

    #[test]
    fn empty_box_is_static() {
        assert!(classify_subtree(&root("box {}")).is_static());
    }

    #[test]
    fn literal_text_is_static() {
        assert!(classify_subtree(&root(r#"text("hello")"#)).is_static());
    }

    #[test]
    fn reactive_text_is_dynamic() {
        assert_eq!(
            classify_subtree(&root("text($label)")),
            StaticClass::Dynamic
        );
    }

    #[test]
    fn runtime_expression_text_is_dynamic() {
        // No `$`, but a non-literal expression cannot be a constant.
        assert_eq!(
            classify_subtree(&root("text(user.clone())")),
            StaticClass::Dynamic
        );
    }

    #[test]
    fn literal_classes_and_styles_are_static() {
        let node = root(
            r#"box {
                class: "card", "elevated";
                key: 7;
                style: { width: px(10.0); flex_direction: column; };
            }"#,
        );
        assert!(classify_subtree(&node).is_static());
    }

    #[test]
    fn non_literal_class_is_dynamic() {
        let node = root(r#"box { class: computed_name(); }"#);
        assert_eq!(classify_subtree(&node), StaticClass::Dynamic);
    }

    #[test]
    fn non_literal_style_argument_is_dynamic() {
        let node = root(r#"box { style: { width: px(dynamic_width); }; }"#);
        assert_eq!(classify_subtree(&node), StaticClass::Dynamic);
    }

    #[test]
    fn for_each_child_makes_subtree_dynamic() {
        let node = root(r#"box { text("header"); for_each(items); }"#);
        assert_eq!(classify_subtree(&node), StaticClass::Dynamic);
    }

    #[test]
    fn dynamic_descendant_poisons_static_ancestor() {
        let node = root(
            r#"box {
                box { text("static"); }
                box { text($dynamic); }
            }"#,
        );
        assert_eq!(classify_subtree(&node), StaticClass::Dynamic);
    }

    #[test]
    fn fully_literal_tree_is_static() {
        let node = root(
            r#"box {
                class: "outer";
                text("a");
                box { class: "inner"; text("b"); }
            }"#,
        );
        assert!(classify_subtree(&node).is_static());
    }

    #[test]
    fn classification_is_deterministic() {
        let src = r#"box { class: "x"; text("y"); }"#;
        assert_eq!(classify_subtree(&root(src)), classify_subtree(&root(src)));
    }
}
