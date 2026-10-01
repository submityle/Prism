//! Deterministic snapshots of computed layout geometry.
//!
//! The [`prism_ui_layout`] solver stores a [`Layout`] per box inside a
//! [`LayoutTree`], but it does not expose a public walk over the arena. A
//! caller therefore describes the subtree it cares about with a [`LayoutQuery`]
//! (mirroring the hierarchy it built), and [`capture_layout`] reads each box's
//! resolved rectangle into an owned [`LayoutSnapshotNode`].
//!
//! [`serialize_layout`] renders that owned tree into a deterministic text
//! format, one box per line:
//!
//! ```text
//! label=<escaped> order=<n> x=<f> y=<f> w=<f> h=<f>
//! ```
//!
//! Floating-point coordinates are formatted with a fixed two decimal places by
//! [`format_fixed`], using only arithmetic and comparison so the output is
//! fully deterministic and free of transcendental functions. [`parse_layout`]
//! reverses the serialization, giving a stable round trip.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use prism_ui_layout::{Layout, LayoutTree, NodeId, Rect};

use crate::escape::{escape, unescape};
use crate::serialize::INDENT_UNIT;

/// Number of fractional digits used when formatting coordinates.
const DECIMALS: u32 = 2;

/// Describes which boxes of a [`LayoutTree`] to capture, and under what labels.
///
/// A query mirrors the hierarchy the caller built in the tree: each query names
/// one [`NodeId`] and lists the child queries to capture beneath it.
#[derive(Clone, Debug)]
pub struct LayoutQuery {
    /// Human-readable label recorded for this box in the snapshot.
    pub label: String,
    /// Handle of the box in the source [`LayoutTree`].
    pub node: NodeId,
    /// Child boxes to capture, in order.
    pub children: Vec<LayoutQuery>,
}

impl LayoutQuery {
    /// Creates a childless query naming `node` with `label`.
    #[must_use]
    pub fn new(label: impl Into<String>, node: NodeId) -> Self {
        Self {
            label: label.into(),
            node,
            children: Vec::new(),
        }
    }

    /// Appends a child query and returns the builder.
    #[must_use]
    pub fn child(mut self, child: LayoutQuery) -> Self {
        self.children.push(child);
        self
    }
}

/// An owned snapshot of one box's computed geometry and its children.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutSnapshotNode {
    /// Human-readable label carried from the originating [`LayoutQuery`].
    pub label: String,
    /// Paint order of the box among its siblings.
    pub order: u32,
    /// Resolved rectangle (location and size) of the box.
    pub rect: Rect,
    /// Captured child boxes, in order.
    pub children: Vec<LayoutSnapshotNode>,
}

/// Reads the resolved geometry for `query` out of `tree` into an owned tree.
///
/// [`LayoutTree::compute_layout`] must have been called first; otherwise the
/// captured rectangles are the solver's zeroed defaults.
#[must_use]
pub fn capture_layout(tree: &LayoutTree, query: &LayoutQuery) -> LayoutSnapshotNode {
    let layout: &Layout = tree.layout(query.node);
    let mut children = Vec::with_capacity(query.children.len());
    for child in &query.children {
        children.push(capture_layout(tree, child));
    }
    LayoutSnapshotNode {
        label: query.label.clone(),
        order: layout.order,
        rect: Rect::new(layout.location, layout.size),
        children,
    }
}

/// Serializes a captured layout tree into the deterministic text format.
#[must_use]
pub fn serialize_layout(root: &LayoutSnapshotNode) -> String {
    let mut out = String::new();
    write_layout(&mut out, root, 0);
    out
}

/// Appends the serialized form of `node` (and its subtree) to `out`.
fn write_layout(out: &mut String, node: &LayoutSnapshotNode, depth: usize) {
    for _ in 0..depth * INDENT_UNIT {
        out.push(' ');
    }
    out.push_str("label=");
    out.push_str(&escape(&node.label));
    out.push_str(" order=");
    out.push_str(&node.order.to_string());
    out.push_str(" x=");
    out.push_str(&format_fixed(node.rect.location.x));
    out.push_str(" y=");
    out.push_str(&format_fixed(node.rect.location.y));
    out.push_str(" w=");
    out.push_str(&format_fixed(node.rect.size.width));
    out.push_str(" h=");
    out.push_str(&format_fixed(node.rect.size.height));
    out.push('\n');
    for child in &node.children {
        write_layout(out, child, depth + 1);
    }
}

/// Formats an `f32` with exactly [`DECIMALS`] fractional digits.
///
/// Rounds half away from zero using integer arithmetic only, so no
/// transcendental or disallowed floating-point methods are involved and the
/// result is deterministic across platforms. Non-finite inputs are rendered as
/// a fixed `nan` or `inf`/`-inf` token so serialization never panics.
#[must_use]
pub fn format_fixed(value: f32) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value == f32::INFINITY {
        return "inf".to_string();
    }
    if value == f32::NEG_INFINITY {
        return "-inf".to_string();
    }

    let mut scale: u64 = 1;
    let mut remaining = DECIMALS;
    while remaining > 0 {
        scale *= 10;
        remaining -= 1;
    }
    let scale_f = scale as f32;

    let negative = value < 0.0;
    let magnitude = if negative { -value } else { value };
    let scaled = magnitude * scale_f + 0.5;
    let units = scaled as u64;
    let whole = units / scale;
    let frac = units % scale;

    let sign = if negative && units != 0 { "-" } else { "" };
    let mut out = String::new();
    out.push_str(sign);
    out.push_str(&whole.to_string());
    out.push('.');
    let frac_text = frac.to_string();
    for _ in 0..(DECIMALS as usize - frac_text.len()) {
        out.push('0');
    }
    out.push_str(&frac_text);
    out
}

/// An error produced while parsing the layout text format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutParseError {
    /// A line's leading indentation was not a multiple of two spaces.
    BadIndent {
        /// One-based line number of the offending line.
        line: usize,
    },
    /// A line did not contain the six expected `key=value` fields.
    MalformedLine {
        /// One-based line number of the offending line.
        line: usize,
    },
    /// A field held an invalid escape sequence or a malformed number.
    BadValue {
        /// One-based line number of the offending line.
        line: usize,
    },
    /// A node was indented more deeply than its parent allows.
    OrphanNode {
        /// One-based line number of the offending line.
        line: usize,
    },
    /// More than one node appeared at indentation level zero.
    MultipleRoots {
        /// One-based line number of the second root.
        line: usize,
    },
    /// The input contained no nodes.
    Empty,
}

impl fmt::Display for LayoutParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutParseError::BadIndent { line } => {
                write!(
                    f,
                    "line {line}: indentation is not a multiple of two spaces"
                )
            }
            LayoutParseError::MalformedLine { line } => write!(
                f,
                "line {line}: expected `label=` `order=` `x=` `y=` `w=` `h=` fields"
            ),
            LayoutParseError::BadValue { line } => {
                write!(f, "line {line}: a field held an invalid value")
            }
            LayoutParseError::OrphanNode { line } => {
                write!(f, "line {line}: node is indented too deeply for its parent")
            }
            LayoutParseError::MultipleRoots { line } => {
                write!(f, "line {line}: a second root node is not allowed")
            }
            LayoutParseError::Empty => write!(f, "layout snapshot text contained no nodes"),
        }
    }
}

/// Parses text produced by [`serialize_layout`] back into a layout tree.
///
/// # Errors
///
/// Returns a [`LayoutParseError`] when the text does not conform to the
/// grammar.
pub fn parse_layout(text: &str) -> Result<LayoutSnapshotNode, LayoutParseError> {
    let mut parsed: Vec<(usize, LayoutSnapshotNode)> = Vec::new();
    for (index, raw_line) in text.lines().enumerate() {
        if raw_line.is_empty() {
            continue;
        }
        let line_no = index + 1;
        let spaces = raw_line.chars().take_while(|&c| c == ' ').count();
        if spaces % INDENT_UNIT != 0 {
            return Err(LayoutParseError::BadIndent { line: line_no });
        }
        let depth = spaces / INDENT_UNIT;
        let node = parse_layout_line(&raw_line[spaces..], line_no)?;
        parsed.push((depth, node));
    }
    if parsed.is_empty() {
        return Err(LayoutParseError::Empty);
    }
    build_layout(parsed)
}

/// Parses a single de-indented content line into a childless node.
fn parse_layout_line(
    content: &str,
    line_no: usize,
) -> Result<LayoutSnapshotNode, LayoutParseError> {
    let mut fields = content.split(' ');
    let label = take_field(&mut fields, "label=", line_no)?;
    let order_text = take_field(&mut fields, "order=", line_no)?;
    let x_text = take_field(&mut fields, "x=", line_no)?;
    let y_text = take_field(&mut fields, "y=", line_no)?;
    let w_text = take_field(&mut fields, "w=", line_no)?;
    let h_text = take_field(&mut fields, "h=", line_no)?;
    if fields.next().is_some() {
        return Err(LayoutParseError::MalformedLine { line: line_no });
    }

    let label = unescape(label).ok_or(LayoutParseError::BadValue { line: line_no })?;
    let order: u32 = order_text
        .parse()
        .map_err(|_| LayoutParseError::BadValue { line: line_no })?;
    let x = parse_fixed(x_text).ok_or(LayoutParseError::BadValue { line: line_no })?;
    let y = parse_fixed(y_text).ok_or(LayoutParseError::BadValue { line: line_no })?;
    let w = parse_fixed(w_text).ok_or(LayoutParseError::BadValue { line: line_no })?;
    let h = parse_fixed(h_text).ok_or(LayoutParseError::BadValue { line: line_no })?;

    Ok(LayoutSnapshotNode {
        label,
        order,
        rect: Rect::new(
            prism_ui_layout::Point::new(x, y),
            prism_ui_layout::Size::new(w, h),
        ),
        children: Vec::new(),
    })
}

/// Reads the next space-separated field and strips its `key=` prefix.
fn take_field<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
    prefix: &str,
    line_no: usize,
) -> Result<&'a str, LayoutParseError> {
    let token = fields
        .next()
        .ok_or(LayoutParseError::MalformedLine { line: line_no })?;
    token
        .strip_prefix(prefix)
        .ok_or(LayoutParseError::MalformedLine { line: line_no })
}

/// Parses a fixed-point coordinate written by [`format_fixed`].
fn parse_fixed(text: &str) -> Option<f32> {
    let (negative, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let mut parts = body.split('.');
    let whole_text = parts.next()?;
    let frac_text = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    if frac_text.len() != DECIMALS as usize {
        return None;
    }
    let whole: u64 = whole_text.parse().ok()?;
    let frac: u64 = frac_text.parse().ok()?;

    let mut scale: u64 = 1;
    let mut remaining = DECIMALS;
    while remaining > 0 {
        scale *= 10;
        remaining -= 1;
    }
    let value = whole as f32 + (frac as f32) / (scale as f32);
    Some(if negative { -value } else { value })
}

/// Assembles a preorder list of `(depth, node)` pairs into a layout tree.
fn build_layout(
    parsed: Vec<(usize, LayoutSnapshotNode)>,
) -> Result<LayoutSnapshotNode, LayoutParseError> {
    let mut stack: Vec<LayoutSnapshotNode> = Vec::new();
    let mut completed_root: Option<LayoutSnapshotNode> = None;

    for (position, (depth, node)) in parsed.into_iter().enumerate() {
        let line_no = position + 1;
        while stack.len() > depth {
            let finished = stack.pop().unwrap_or_else(empty_node);
            match stack.last_mut() {
                Some(parent) => parent.children.push(finished),
                None => {
                    if completed_root.is_some() {
                        return Err(LayoutParseError::MultipleRoots { line: line_no });
                    }
                    completed_root = Some(finished);
                }
            }
        }
        if stack.len() != depth {
            return Err(LayoutParseError::OrphanNode { line: line_no });
        }
        if depth == 0 && (completed_root.is_some() || !stack.is_empty()) {
            return Err(LayoutParseError::MultipleRoots { line: line_no });
        }
        stack.push(node);
    }

    while stack.len() > 1 {
        let finished = stack.pop().unwrap_or_else(empty_node);
        if let Some(parent) = stack.last_mut() {
            parent.children.push(finished);
        }
    }

    match (stack.pop(), completed_root) {
        (Some(root), None) | (None, Some(root)) => Ok(root),
        _ => Err(LayoutParseError::Empty),
    }
}

/// Fallback node for pops the surrounding length checks already guarantee.
fn empty_node() -> LayoutSnapshotNode {
    LayoutSnapshotNode {
        label: String::new(),
        order: 0,
        rect: Rect::new(
            prism_ui_layout::Point::new(0.0, 0.0),
            prism_ui_layout::Size::new(0.0, 0.0),
        ),
        children: Vec::new(),
    }
}

impl LayoutParseError {
    /// Renders this error as an owned, human-readable string.
    #[must_use]
    pub fn message(&self) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        capture_layout, format_fixed, parse_fixed, parse_layout, serialize_layout, LayoutQuery,
        LayoutSnapshotNode,
    };
    use alloc::vec::Vec;
    use prism_ui_layout::{
        AvailableSpace, Dimension, Display, LayoutStyle, LayoutTree, Point, Rect, Size,
    };

    #[test]
    fn formats_fixed_two_decimals() {
        assert_eq!(format_fixed(0.0), "0.00");
        assert_eq!(format_fixed(1.5), "1.50");
        assert_eq!(format_fixed(12.344), "12.34");
        assert_eq!(format_fixed(12.345), "12.35");
        assert_eq!(format_fixed(-3.2), "-3.20");
        assert_eq!(format_fixed(100.0), "100.00");
    }

    #[test]
    fn fixed_round_trips_values() {
        for v in [0.0f32, 1.5, 12.34, 100.0, 3.25, 250.75] {
            let text = format_fixed(v);
            let parsed = parse_fixed(&text).expect("parse");
            assert_eq!(format_fixed(parsed), text);
        }
    }

    #[test]
    fn captures_and_serializes_real_layout() {
        let mut tree = LayoutTree::new();
        let child_a = tree.new_leaf(LayoutStyle {
            size: Size::new(Dimension::Points(40.0), Dimension::Points(20.0)),
            ..LayoutStyle::default()
        });
        let child_b = tree.new_leaf(LayoutStyle {
            size: Size::new(Dimension::Points(60.0), Dimension::Points(20.0)),
            ..LayoutStyle::default()
        });
        let root = tree.new_node(
            LayoutStyle {
                display: Display::Flex,
                ..LayoutStyle::default()
            },
            &[child_a, child_b],
        );
        tree.compute_layout(
            root,
            Size::new(
                AvailableSpace::Definite(200.0),
                AvailableSpace::Definite(100.0),
            ),
        );

        let query = LayoutQuery::new("root", root)
            .child(LayoutQuery::new("a", child_a))
            .child(LayoutQuery::new("b", child_b));
        let snapshot = capture_layout(&tree, &query);

        // First child sits at the origin; second is offset by the first's width.
        assert_eq!(snapshot.children[0].rect.location, Point::new(0.0, 0.0));
        assert_eq!(snapshot.children[0].rect.size, Size::new(40.0, 20.0));
        assert_eq!(snapshot.children[1].rect.location, Point::new(40.0, 0.0));

        let text = serialize_layout(&snapshot);
        let parsed = parse_layout(&text).expect("round trip");
        assert_eq!(serialize_layout(&parsed), text);
    }

    #[test]
    fn parse_layout_round_trips_handcrafted_tree() {
        let root = LayoutSnapshotNode {
            label: "root".into(),
            order: 0,
            rect: Rect::new(Point::new(0.0, 0.0), Size::new(200.0, 100.0)),
            children: alloc::vec![LayoutSnapshotNode {
                label: "child one".into(),
                order: 0,
                rect: Rect::new(Point::new(10.0, 20.0), Size::new(30.5, 40.25)),
                children: Vec::new(),
            }],
        };
        let text = serialize_layout(&root);
        let parsed = parse_layout(&text).expect("round trip");
        assert_eq!(parsed, root);
    }

    #[test]
    fn rejects_empty_layout() {
        assert!(parse_layout("").is_err());
    }
}
