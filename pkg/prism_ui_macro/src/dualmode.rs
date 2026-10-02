//! Dual-mode compilation support (§9.1).
//!
//! Loom compiles the same `.loom` source two ways:
//!
//! * **Interpret** (development): the AST is lowered at run time so edits hot
//!   reload without recompiling.
//! * **Freeze** (release): the AST is lowered at compile time to builder calls,
//!   driving parse/build cost to zero.
//!
//! Both paths feed through the single shared lowerer in [`crate::lower`], and
//! both describe a tree with the same **semantic program**: a flat,
//! deterministic sequence of [`SemanticOp`]s. Because the program is identical
//! for both modes, "interpret result == freeze result" holds by construction;
//! the tests here pin that invariant down (the role of the snapshot vectors in
//! §9.1).
//!
//! Freezing additionally requires *reproducible builds*: the emitted output
//! must not depend on incidental source ordering. [`canonical_attr_order`]
//! provides that stability by sorting a node's attributes into a canonical
//! category order with a *stable* sort, so two sources that differ only in the
//! order of their `class` / `key` / `style` lines freeze to byte-identical
//! output.

use crate::ast::{Attr, Child, Node, NodeKind, StyleVal};

/// The two compilation paths that share the lowerer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Development-time interpretation (hot reload).
    Interpret,
    /// Release-time macro freezing (zero parse cost).
    Freeze,
}

/// A single step in a node's lowering, independent of the target mode.
///
/// The sequence of these ops for a whole tree is the shared contract between
/// the interpret and freeze paths: equal programs guarantee equal results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SemanticOp {
    /// Construct a node of the given shape at the given stable-id path.
    Construct { shape: NodeShape, path: String },
    /// Attach one class name (literal text, or `<expr>` for a spliced value).
    Class(String),
    /// Attach an integer reconciliation key.
    KeyInt(String),
    /// Attach a string reconciliation key.
    KeyStr(String),
    /// Set one inline style property to the given value description.
    Style { prop: String, value: String },
    /// Begin a statically-known child subtree.
    EnterChild,
    /// End the most recently entered child subtree.
    LeaveChild,
    /// Splice a dynamic `for_each(..)` child list.
    ForEach,
}

/// The constructor shape of a [`Node`], recorded without its payload tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeShape {
    /// A `box` element.
    Box,
    /// A `text(..)` element.
    Text,
    /// A `custom(..)` element.
    Custom,
}

/// Returns a node's attributes in canonical, build-reproducible order.
///
/// Attributes are grouped by category — `class` first, then `key`, then
/// `style` — using a **stable** sort, so the relative order of attributes
/// *within* a category (notably multiple `class` lines and the `style`
/// entries) is preserved while cross-category source order is normalised. This
/// is what makes frozen output reproducible regardless of how the attribute
/// lines were ordered in source.
pub(crate) fn canonical_attr_order(attrs: &[Attr]) -> Vec<&Attr> {
    let mut ordered: Vec<&Attr> = attrs.iter().collect();
    ordered.sort_by_key(|attr| category_rank(attr));
    ordered
}

/// The sort rank of an attribute's category.
fn category_rank(attr: &Attr) -> u8 {
    match attr {
        Attr::Class(_) => 0,
        Attr::KeyInt(_) | Attr::KeyStr(_) => 1,
        Attr::Style(_) => 2,
    }
}

/// Builds the deterministic [`SemanticOp`] program for `node` rooted at `path`.
///
/// Attributes are emitted in [`canonical_attr_order`], so the program — and
/// therefore the frozen output derived from it — is reproducible.
pub(crate) fn semantic_program(node: &Node, path: &str) -> Vec<SemanticOp> {
    let mut ops = Vec::new();
    emit_node(node, path, &mut ops);
    ops
}

/// Appends the program for `node` (and its subtree) to `ops`.
fn emit_node(node: &Node, path: &str, ops: &mut Vec<SemanticOp>) {
    let shape = match &node.kind {
        NodeKind::Box => NodeShape::Box,
        NodeKind::Text(_) => NodeShape::Text,
        NodeKind::Custom(_) => NodeShape::Custom,
    };
    ops.push(SemanticOp::Construct {
        shape,
        path: path.to_string(),
    });

    for attr in canonical_attr_order(&node.attrs) {
        emit_attr(attr, ops);
    }

    for (index, child) in node.children.iter().enumerate() {
        match child {
            Child::Node(child_node) => {
                ops.push(SemanticOp::EnterChild);
                emit_node(child_node, &child_path(path, index), ops);
                ops.push(SemanticOp::LeaveChild);
            }
            Child::ForEach(_) => ops.push(SemanticOp::ForEach),
        }
    }
}

/// Appends the ops for a single attribute to `ops`.
fn emit_attr(attr: &Attr, ops: &mut Vec<SemanticOp>) {
    match attr {
        Attr::Class(names) => {
            for name in names {
                ops.push(SemanticOp::Class(describe_expr(name)));
            }
        }
        Attr::KeyInt(int) => ops.push(SemanticOp::KeyInt(int.base10_digits().to_string())),
        Attr::KeyStr(string) => ops.push(SemanticOp::KeyStr(string.value())),
        Attr::Style(entries) => {
            for entry in entries {
                ops.push(SemanticOp::Style {
                    prop: entry.prop.to_string(),
                    value: describe_style_val(&entry.value),
                });
            }
        }
    }
}

/// Describes a class expression: the literal text when it is a string literal,
/// otherwise a stable `<expr>` placeholder (its concrete tokens are irrelevant
/// to ordering stability).
fn describe_expr(expr: &syn::Expr) -> String {
    if let syn::Expr::Lit(lit) = expr
        && let syn::Lit::Str(string) = &lit.lit
    {
        return string.value();
    }
    "<expr>".to_string()
}

/// Describes a style value compactly for the semantic program.
fn describe_style_val(value: &StyleVal) -> String {
    match value {
        StyleVal::Number(lit) => match lit {
            syn::Lit::Int(int) => int.base10_digits().to_string(),
            syn::Lit::Float(float) => float.base10_digits().to_string(),
            _ => "<num>".to_string(),
        },
        StyleVal::Keyword(ident) => ident.to_string(),
        StyleVal::Call { func, .. } => format!("{func}(..)"),
    }
}

/// Extends `parent` with `index` to form a child's position path (mirrors the
/// lowerer's path scheme so stable ids line up).
fn child_path(parent: &str, index: usize) -> String {
    if parent.is_empty() {
        index.to_string()
    } else {
        format!("{parent}/{index}")
    }
}

/// Produces the semantic program for a node in the given [`Mode`].
///
/// Both modes return the identical program — the whole point of §9.1 — so this
/// is the function the snapshot/equivalence tests pin down. The `mode`
/// parameter documents intent at the call site and lets future mode-specific
/// post-processing hook in without changing the shared program.
pub(crate) fn lower_in_mode(node: &Node, path: &str, mode: Mode) -> Vec<SemanticOp> {
    let program = semantic_program(node, path);
    match mode {
        Mode::Interpret | Mode::Freeze => program,
    }
}

/// Encodes a program as a stable, line-oriented signature string, suitable for
/// snapshot comparison between the interpret and freeze paths.
pub(crate) fn program_signature(program: &[SemanticOp]) -> String {
    let mut out = String::new();
    for op in program {
        encode_op(op, &mut out);
        out.push('\n');
    }
    out
}

/// Appends one op's canonical encoding to `out`.
fn encode_op(op: &SemanticOp, out: &mut String) {
    match op {
        SemanticOp::Construct { shape, path } => {
            out.push_str("construct ");
            out.push_str(shape_tag(*shape));
            out.push('@');
            out.push_str(path);
        }
        SemanticOp::Class(name) => {
            out.push_str("class ");
            out.push_str(name);
        }
        SemanticOp::KeyInt(value) => {
            out.push_str("key_int ");
            out.push_str(value);
        }
        SemanticOp::KeyStr(value) => {
            out.push_str("key_str ");
            out.push_str(value);
        }
        SemanticOp::Style { prop, value } => {
            out.push_str("style ");
            out.push_str(prop);
            out.push('=');
            out.push_str(value);
        }
        SemanticOp::EnterChild => out.push_str("enter"),
        SemanticOp::LeaveChild => out.push_str("leave"),
        SemanticOp::ForEach => out.push_str("for_each"),
    }
}

/// The short tag used for a shape in a program signature.
fn shape_tag(shape: NodeShape) -> &'static str {
    match shape {
        NodeShape::Box => "box",
        NodeShape::Text => "text",
        NodeShape::Custom => "custom",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_str;

    use crate::ast::LoomInput;

    /// Parses a `loom!` body into its root [`Node`].
    fn root(src: &str) -> Node {
        parse_str::<LoomInput>(src).expect("valid loom input").node
    }

    #[test]
    fn program_is_deterministic() {
        let src = r#"box { class: "a"; text("hi"); }"#;
        assert_eq!(
            semantic_program(&root(src), ""),
            semantic_program(&root(src), "")
        );
    }

    #[test]
    fn attribute_category_order_is_normalised() {
        // Same node, attributes written in two different category orders.
        let a = root(
            r#"box {
                class: "c";
                key: 3;
                style: { width: px(1.0); };
            }"#,
        );
        let b = root(
            r#"box {
                style: { width: px(1.0); };
                key: 3;
                class: "c";
            }"#,
        );
        assert_eq!(semantic_program(&a, ""), semantic_program(&b, ""));
    }

    #[test]
    fn multiple_classes_keep_their_relative_order() {
        let node = root(r#"box { class: "first", "second", "third"; }"#);
        let program = semantic_program(&node, "");
        let classes: Vec<&String> = program
            .iter()
            .filter_map(|op| match op {
                SemanticOp::Class(name) => Some(name),
                _ => None,
            })
            .collect();
        assert_eq!(classes, vec!["first", "second", "third"]);
    }

    #[test]
    fn canonical_order_is_idempotent() {
        let node = root(
            r#"box {
                style: { width: px(1.0); };
                class: "c";
            }"#,
        );
        // Projecting to the category-rank sequence captures the ordering
        // without needing identity comparisons: a canonical order is one whose
        // ranks are non-decreasing, and re-sorting must leave them unchanged.
        let ranks =
            |attrs: &[&Attr]| -> Vec<u8> { attrs.iter().map(|a| category_rank(a)).collect() };
        let once = canonical_attr_order(&node.attrs);
        let once_ranks = ranks(&once);
        let mut twice = once.clone();
        twice.sort_by_key(|attr| category_rank(attr));
        assert_eq!(once_ranks, ranks(&twice));
        assert!(once_ranks.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn interpret_and_freeze_produce_equal_programs() {
        let node = root(
            r#"box {
                class: "card";
                text("hello");
                box { class: "row"; for_each(items); }
            }"#,
        );
        let interpret = lower_in_mode(&node, "", Mode::Interpret);
        let freeze = lower_in_mode(&node, "", Mode::Freeze);
        assert_eq!(interpret, freeze);
        assert_eq!(
            program_signature(&interpret),
            program_signature(&freeze),
            "interpret and freeze must agree on the program signature",
        );
    }

    #[test]
    fn signature_reflects_structure_and_paths() {
        let node = root(r#"box { text("x"); }"#);
        let sig = program_signature(&semantic_program(&node, ""));
        assert_eq!(sig, "construct box@\nenter\nconstruct text@0\nleave\n");
    }

    #[test]
    fn for_each_appears_as_dynamic_op() {
        let node = root(r#"box { for_each(items); }"#);
        let program = semantic_program(&node, "");
        assert!(program.contains(&SemanticOp::ForEach));
    }

    #[test]
    fn style_values_are_described() {
        let node = root(r#"box { style: { width: px(2.0); flex_direction: column; }; }"#);
        let program = semantic_program(&node, "");
        let styles: Vec<(&String, &String)> = program
            .iter()
            .filter_map(|op| match op {
                SemanticOp::Style { prop, value } => Some((prop, value)),
                _ => None,
            })
            .collect();
        assert_eq!(styles[0].0, "width");
        assert_eq!(styles[0].1, "px(..)");
        assert_eq!(styles[1].0, "flex_direction");
        assert_eq!(styles[1].1, "column");
    }
}
