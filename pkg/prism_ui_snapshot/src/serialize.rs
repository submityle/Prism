//! Deterministic text serialization of a [`TreeSnapshot`].
//!
//! The format is a line-per-node indented tree. Each node occupies one line,
//! indented with two spaces per depth level, and carries three
//! space-separated, escaped fields:
//!
//! ```text
//! kind=<escaped> text=<marker+escaped|-> classes=<,escaped,escaped...>
//! ```
//!
//! * `kind=` holds the node kind label.
//! * `text=` is `-` when the node has no text, or `+` followed by the escaped
//!   text when it does. This distinguishes a missing text run from an empty
//!   one.
//! * `classes=` is empty for no classes, otherwise each class name is written
//!   as a leading comma followed by its escaped value. The leading comma lets
//!   an empty list be told apart from a single empty-string class.
//!
//! Because every field is escaped by [`crate::escape`], no field can contain a
//! space, comma, or newline, so the grammar is unambiguous for any input. The
//! traversal is depth-first in sibling order, so the same tree always produces
//! exactly the same string. [`parse_tree`] reverses [`serialize_tree`] into an
//! equal [`TreeSnapshot`].

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use prism_ui_devtools::{SnapshotNode, TreeSnapshot};

use crate::escape::{escape, unescape};

/// Number of spaces used for each level of indentation.
pub(crate) const INDENT_UNIT: usize = 2;

/// Serializes a [`TreeSnapshot`] into the deterministic text format.
///
/// The returned string ends every node line, including the last, with a
/// newline.
#[must_use]
pub fn serialize_tree(tree: &TreeSnapshot) -> String {
    let mut out = String::new();
    write_node(&mut out, &tree.root, 0);
    out
}

/// Appends the serialized form of `node` (and its subtree) to `out`.
fn write_node(out: &mut String, node: &SnapshotNode, depth: usize) {
    for _ in 0..depth * INDENT_UNIT {
        out.push(' ');
    }
    out.push_str("kind=");
    out.push_str(&escape(&node.kind));
    out.push_str(" text=");
    match &node.text {
        Some(text) => {
            out.push('+');
            out.push_str(&escape(text));
        }
        None => out.push('-'),
    }
    out.push_str(" classes=");
    for class in &node.classes {
        out.push(',');
        out.push_str(&escape(class));
    }
    out.push('\n');
    for child in &node.children {
        write_node(out, child, depth + 1);
    }
}

/// An error produced while parsing the text snapshot format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// A line's leading indentation was not a multiple of two spaces.
    BadIndent {
        /// One-based line number of the offending line.
        line: usize,
    },
    /// A line did not contain the three expected `key=value` fields.
    MalformedLine {
        /// One-based line number of the offending line.
        line: usize,
    },
    /// A field contained an invalid escape sequence.
    BadEscape {
        /// One-based line number of the offending line.
        line: usize,
    },
    /// A node was indented more deeply than its parent allows, leaving it
    /// without a valid parent.
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

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::BadIndent { line } => {
                write!(
                    f,
                    "line {line}: indentation is not a multiple of two spaces"
                )
            }
            ParseError::MalformedLine { line } => {
                write!(f, "line {line}: expected `kind=` `text=` `classes=` fields")
            }
            ParseError::BadEscape { line } => {
                write!(f, "line {line}: invalid escape sequence in a field")
            }
            ParseError::OrphanNode { line } => {
                write!(f, "line {line}: node is indented too deeply for its parent")
            }
            ParseError::MultipleRoots { line } => {
                write!(f, "line {line}: a second root node is not allowed")
            }
            ParseError::Empty => write!(f, "snapshot text contained no nodes"),
        }
    }
}

/// Parses text produced by [`serialize_tree`] back into a [`TreeSnapshot`].
///
/// Blank lines are ignored. The parse is exact: `parse_tree(&serialize_tree(t))`
/// returns a [`TreeSnapshot`] equal to `t`.
///
/// # Errors
///
/// Returns a [`ParseError`] when the text does not conform to the grammar, for
/// example on bad indentation, missing fields, invalid escapes, orphaned
/// nodes, multiple roots, or empty input.
pub fn parse_tree(text: &str) -> Result<TreeSnapshot, ParseError> {
    let mut parsed: Vec<(usize, SnapshotNode)> = Vec::new();
    for (index, raw_line) in text.lines().enumerate() {
        if raw_line.is_empty() {
            continue;
        }
        let line_no = index + 1;
        let spaces = raw_line.chars().take_while(|&c| c == ' ').count();
        if spaces % INDENT_UNIT != 0 {
            return Err(ParseError::BadIndent { line: line_no });
        }
        let depth = spaces / INDENT_UNIT;
        let node = parse_line(&raw_line[spaces..], line_no)?;
        parsed.push((depth, node));
    }
    if parsed.is_empty() {
        return Err(ParseError::Empty);
    }
    build_tree(parsed)
}

/// Parses a single de-indented content line into a childless [`SnapshotNode`].
fn parse_line(content: &str, line_no: usize) -> Result<SnapshotNode, ParseError> {
    let mut parts = content.split(' ');
    let kind_token = parts.next().unwrap_or("");
    let text_token = parts
        .next()
        .ok_or(ParseError::MalformedLine { line: line_no })?;
    let classes_token = parts
        .next()
        .ok_or(ParseError::MalformedLine { line: line_no })?;
    if parts.next().is_some() {
        return Err(ParseError::MalformedLine { line: line_no });
    }

    let kind_value = kind_token
        .strip_prefix("kind=")
        .ok_or(ParseError::MalformedLine { line: line_no })?;
    let text_value = text_token
        .strip_prefix("text=")
        .ok_or(ParseError::MalformedLine { line: line_no })?;
    let classes_value = classes_token
        .strip_prefix("classes=")
        .ok_or(ParseError::MalformedLine { line: line_no })?;

    let kind = unescape(kind_value).ok_or(ParseError::BadEscape { line: line_no })?;

    let text = match text_value.chars().next() {
        Some('-') => {
            if text_value.len() != 1 {
                return Err(ParseError::MalformedLine { line: line_no });
            }
            None
        }
        Some('+') => {
            Some(unescape(&text_value[1..]).ok_or(ParseError::BadEscape { line: line_no })?)
        }
        _ => return Err(ParseError::MalformedLine { line: line_no }),
    };

    let classes = if classes_value.is_empty() {
        Vec::new()
    } else {
        let body = classes_value
            .strip_prefix(',')
            .ok_or(ParseError::MalformedLine { line: line_no })?;
        let mut classes = Vec::new();
        for token in body.split(',') {
            classes.push(unescape(token).ok_or(ParseError::BadEscape { line: line_no })?);
        }
        classes
    };

    Ok(SnapshotNode {
        kind,
        text,
        classes,
        children: Vec::new(),
    })
}

/// Assembles a preorder list of `(depth, node)` pairs into a tree.
fn build_tree(parsed: Vec<(usize, SnapshotNode)>) -> Result<TreeSnapshot, ParseError> {
    let mut stack: Vec<SnapshotNode> = Vec::new();
    let mut completed_root: Option<SnapshotNode> = None;

    for (position, (depth, node)) in parsed.into_iter().enumerate() {
        let line_no = position + 1;
        while stack.len() > depth {
            let finished = stack.pop().unwrap_or_else(unreachable_node);
            match stack.last_mut() {
                Some(parent) => parent.children.push(finished),
                None => {
                    if completed_root.is_some() {
                        return Err(ParseError::MultipleRoots { line: line_no });
                    }
                    completed_root = Some(finished);
                }
            }
        }
        if stack.len() != depth {
            return Err(ParseError::OrphanNode { line: line_no });
        }
        if depth == 0 && (completed_root.is_some() || !stack.is_empty()) {
            return Err(ParseError::MultipleRoots { line: line_no });
        }
        stack.push(node);
    }

    while stack.len() > 1 {
        let finished = stack.pop().unwrap_or_else(unreachable_node);
        if let Some(parent) = stack.last_mut() {
            parent.children.push(finished);
        }
    }

    let ((Some(root), None) | (None, Some(root))) = (stack.pop(), completed_root) else {
        return Err(ParseError::Empty);
    };
    Ok(TreeSnapshot { root })
}

/// Fallback for a pop that the surrounding length checks already guarantee
/// succeeds; it yields an empty node rather than panicking.
fn unreachable_node() -> SnapshotNode {
    SnapshotNode {
        kind: String::new(),
        text: None,
        classes: Vec::new(),
        children: Vec::new(),
    }
}

impl ParseError {
    /// Renders this error as an owned, human-readable string.
    #[must_use]
    pub fn message(&self) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_tree, serialize_tree, ParseError};
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use prism_ui_devtools::{SnapshotNode, TreeSnapshot};

    fn node(
        kind: &str,
        text: Option<&str>,
        classes: &[&str],
        children: Vec<SnapshotNode>,
    ) -> SnapshotNode {
        SnapshotNode {
            kind: kind.into(),
            text: text.map(Into::into),
            classes: classes.iter().map(|c| String::from(*c)).collect(),
            children,
        }
    }

    #[test]
    fn serializes_simple_tree() {
        let tree = TreeSnapshot {
            root: node(
                "Box",
                None,
                &["card"],
                vec![node("Text", Some("hi"), &[], Vec::new())],
            ),
        };
        assert_eq!(
            serialize_tree(&tree),
            "kind=Box text=- classes=,card\n  kind=Text text=+hi classes=\n",
        );
    }

    #[test]
    fn round_trips_tree_with_special_characters() {
        let tree = TreeSnapshot {
            root: node(
                "Cus tom",
                Some("line1\nline2, with comma\tand tab"),
                &["a b", "c,d", ""],
                vec![
                    node("Text", Some(""), &[], Vec::new()),
                    node(
                        "Box",
                        None,
                        &["x"],
                        vec![node("Leaf", Some("deep"), &[], Vec::new())],
                    ),
                ],
            ),
        };
        let text = serialize_tree(&tree);
        let parsed = parse_tree(&text).expect("round trip");
        assert_eq!(parsed, tree);
        // Re-serialization is stable.
        assert_eq!(serialize_tree(&parsed), text);
    }

    #[test]
    fn empty_text_differs_from_missing_text() {
        let with_empty = TreeSnapshot {
            root: node("Text", Some(""), &[], Vec::new()),
        };
        let without = TreeSnapshot {
            root: node("Text", None, &[], Vec::new()),
        };
        assert_ne!(serialize_tree(&with_empty), serialize_tree(&without));
        assert_eq!(
            parse_tree(&serialize_tree(&with_empty)).unwrap(),
            with_empty
        );
        assert_eq!(parse_tree(&serialize_tree(&without)).unwrap(), without);
    }

    #[test]
    fn single_empty_class_differs_from_no_classes() {
        let one_empty = TreeSnapshot {
            root: node("Box", None, &[""], Vec::new()),
        };
        let none = TreeSnapshot {
            root: node("Box", None, &[], Vec::new()),
        };
        assert_ne!(serialize_tree(&one_empty), serialize_tree(&none));
        assert_eq!(parse_tree(&serialize_tree(&one_empty)).unwrap(), one_empty);
    }

    #[test]
    fn rejects_bad_indentation() {
        let err = parse_tree("kind=Box text=- classes=\n   kind=Text text=+a classes=\n");
        assert_eq!(err, Err(ParseError::BadIndent { line: 2 }));
    }

    #[test]
    fn rejects_orphan_node() {
        let err = parse_tree("kind=Box text=- classes=\n    kind=Text text=+a classes=\n");
        assert_eq!(err, Err(ParseError::OrphanNode { line: 2 }));
    }

    #[test]
    fn rejects_multiple_roots() {
        let err = parse_tree("kind=Box text=- classes=\nkind=Box text=- classes=\n");
        assert_eq!(err, Err(ParseError::MultipleRoots { line: 2 }));
    }

    #[test]
    fn rejects_empty_input() {
        assert_eq!(parse_tree(""), Err(ParseError::Empty));
        assert_eq!(parse_tree("\n\n"), Err(ParseError::Empty));
    }

    #[test]
    fn rejects_malformed_line() {
        assert_eq!(
            parse_tree("kind=Box text=-\n"),
            Err(ParseError::MalformedLine { line: 1 }),
        );
    }

    #[test]
    fn error_messages_are_available() {
        let err = ParseError::BadIndent { line: 7 };
        assert!(err.message().contains("line 7"));
    }
}
