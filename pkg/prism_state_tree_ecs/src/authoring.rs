//! Strict, deterministic text authoring for [`prism_state_tree_core::TreeDef`].
//!
//! The format intentionally has no third-party dependency. It is a small,
//! line-oriented interchange format that is stable in diffs and cheap to
//! validate before an asset reaches the runtime compiler:
//!
//! ```text
//! prism_state_tree version=1 root=0 nodes=3
//! node id=0 parent=- kind=compound children=1,2 regions=- initial=1 history=none transitions=0
//! node id=1 parent=0 kind=atomic children=- regions=- initial=- history=none transitions=1
//! transition source=1 target=state:2 trigger=event:7 guard=none priority=0 kind=external
//! node id=2 parent=0 kind=atomic children=- regions=- initial=- history=none transitions=0
//! ```
//!
//! Records use a fixed key order and reject unknown, missing, duplicated, or
//! reordered fields. Values never contain whitespace, so parsing is linear in
//! the input size and does not need an escaping layer. The parser also runs
//! the core structural compiler before returning, making the returned
//! definition suitable for immediate compilation.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::{fmt, fmt::Write as _};

use prism_state_tree_core::{
    HistoryMode, NodeKind, PlanError, StateId, StateNode, Transition, TransitionKind,
    TransitionTarget, TreeDef, Trigger,
};

/// Current version of the standalone state-tree text format.
pub const TEXT_FORMAT_VERSION: u32 = 1;

/// A strict text-authoring parse or validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TreeDefTextError {
    /// The input did not contain a header record.
    Empty,
    /// The first record did not match the versioned format header.
    InvalidHeader {
        /// One-based source line.
        line: usize,
    },
    /// A required record was missing at the end of the input.
    UnexpectedEnd {
        /// Record kind that was required.
        expected: &'static str,
    },
    /// A record had the wrong number or order of fields.
    MalformedRecord {
        /// One-based source line.
        line: usize,
    },
    /// A field contained a value that could not be parsed as its scalar type.
    InvalidNumber {
        /// One-based source line.
        line: usize,
        /// Canonical field name.
        field: &'static str,
    },
    /// A field contained a syntactically valid but unsupported value.
    InvalidValue {
        /// One-based source line.
        line: usize,
        /// Canonical field name.
        field: &'static str,
    },
    /// A record appeared after all declared nodes and transitions.
    ExtraInput {
        /// One-based source line.
        line: usize,
    },
    /// The parsed definition failed the core semantic compiler checks.
    InvalidDefinition {
        /// The precise core validation failure.
        error: PlanError,
    },
}

impl fmt::Display for TreeDefTextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("state-tree text is empty"),
            Self::InvalidHeader { line } => write!(f, "line {line}: invalid state-tree header"),
            Self::UnexpectedEnd { expected } => {
                write!(f, "unexpected end of input; expected {expected} record")
            }
            Self::MalformedRecord { line } => write!(f, "line {line}: malformed record"),
            Self::InvalidNumber { line, field } => {
                write!(f, "line {line}: invalid number in `{field}`")
            }
            Self::InvalidValue { line, field } => {
                write!(f, "line {line}: invalid value in `{field}`")
            }
            Self::ExtraInput { line } => write!(f, "line {line}: unexpected extra input"),
            Self::InvalidDefinition { error } => write!(f, "invalid tree definition: {error:?}"),
        }
    }
}

/// Serializes a structurally valid [`TreeDef`] into canonical text.
///
/// Validation happens before any output is produced. The canonical output is
/// deterministic: nodes retain their vector order and transitions retain
/// declaration order. Compiler-assigned transition declaration ordinals are
/// intentionally omitted because they are derived data, not authoring data.
pub fn serialize_tree_def(def: &TreeDef) -> Result<String, TreeDefTextError> {
    def.clone()
        .compile()
        .map_err(|error| TreeDefTextError::InvalidDefinition { error })?;

    let mut text = String::new();
    write_record(
        &mut text,
        "prism_state_tree",
        &["version", "root", "nodes"],
        &[
            TEXT_FORMAT_VERSION.to_string(),
            def.root.0.to_string(),
            def.nodes.len().to_string(),
        ],
    );
    for node in &def.nodes {
        let (kind, children, regions, initial, history) = match &node.kind {
            NodeKind::Atomic => ("atomic", Vec::new(), Vec::new(), None, None),
            NodeKind::Compound {
                children,
                initial,
                history,
            } => (
                "compound",
                children.clone(),
                Vec::new(),
                Some(*initial),
                *history,
            ),
            NodeKind::Parallel {
                children,
                regions,
                history,
            } => (
                "parallel",
                children.clone(),
                regions.clone(),
                None,
                *history,
            ),
        };
        write_record(
            &mut text,
            "node",
            &[
                "id",
                "parent",
                "kind",
                "children",
                "regions",
                "initial",
                "history",
                "transitions",
            ],
            &[
                node.id.0.to_string(),
                option_id(node.parent),
                kind.to_string(),
                id_list(&children),
                id_list(&regions),
                option_id(initial),
                history_name(history).to_string(),
                node.transitions.len().to_string(),
            ],
        );
        for transition in &node.transitions {
            write_record(
                &mut text,
                "transition",
                &["source", "target", "trigger", "guard", "priority", "kind"],
                &[
                    transition.source.0.to_string(),
                    target_text(transition.target),
                    trigger_text(transition.trigger),
                    guard_text(transition.guard),
                    transition.priority.to_string(),
                    transition_kind_name(transition.kind).to_string(),
                ],
            );
        }
    }
    Ok(text)
}

/// Parses canonical state-tree text and performs full core validation.
///
/// The parser accepts the final newline emitted by [`serialize_tree_def`],
/// but rejects blank lines, comments, unknown fields, alternate field order,
/// trailing records, and semantically invalid definitions.
pub fn parse_tree_def(text: &str) -> Result<TreeDef, TreeDefTextError> {
    let mut lines = text.lines().enumerate();
    let (header_index, header) = lines.next().ok_or(TreeDefTextError::Empty)?;
    let header_line = header_index + 1;
    let header_values = parse_record(header, "prism_state_tree", &["version", "root", "nodes"])
        .ok_or(TreeDefTextError::InvalidHeader { line: header_line })?;
    let version = parse_u32(header_values[0], header_line, "version")?;
    if version != TEXT_FORMAT_VERSION {
        return Err(TreeDefTextError::InvalidHeader { line: header_line });
    }
    let root = StateId(parse_u32(header_values[1], header_line, "root")?);
    let node_count = parse_usize(header_values[2], header_line, "nodes")?;

    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let (index, line) = lines
            .next()
            .ok_or(TreeDefTextError::UnexpectedEnd { expected: "node" })?;
        let line_no = index + 1;
        let values = parse_record(
            line,
            "node",
            &[
                "id",
                "parent",
                "kind",
                "children",
                "regions",
                "initial",
                "history",
                "transitions",
            ],
        )
        .ok_or(TreeDefTextError::MalformedRecord { line: line_no })?;
        let id = StateId(parse_u32(values[0], line_no, "id")?);
        let parent = parse_optional_id(values[1], line_no, "parent")?;
        let kind_name = values[2];
        let children = parse_id_list(values[3], line_no, "children")?;
        let regions = parse_id_list(values[4], line_no, "regions")?;
        let initial = parse_optional_id(values[5], line_no, "initial")?;
        let history = parse_history(values[6], line_no, "history")?;
        let transition_count = parse_usize(values[7], line_no, "transitions")?;
        let kind = match kind_name {
            "atomic" => {
                if !children.is_empty()
                    || !regions.is_empty()
                    || initial.is_some()
                    || history.is_some()
                {
                    return Err(TreeDefTextError::InvalidValue {
                        line: line_no,
                        field: "kind",
                    });
                }
                NodeKind::Atomic
            }
            "compound" => {
                if !regions.is_empty() || initial.is_none() {
                    return Err(TreeDefTextError::InvalidValue {
                        line: line_no,
                        field: "kind",
                    });
                }
                NodeKind::Compound {
                    children,
                    initial: initial.unwrap_or(StateId(0)),
                    history,
                }
            }
            "parallel" => {
                if initial.is_some() {
                    return Err(TreeDefTextError::InvalidValue {
                        line: line_no,
                        field: "kind",
                    });
                }
                NodeKind::Parallel {
                    children,
                    regions,
                    history,
                }
            }
            _ => {
                return Err(TreeDefTextError::InvalidValue {
                    line: line_no,
                    field: "kind",
                })
            }
        };
        let mut node = StateNode {
            id,
            parent,
            kind,
            transitions: Vec::with_capacity(transition_count),
        };
        for _ in 0..transition_count {
            let (transition_index, transition_line) =
                lines.next().ok_or(TreeDefTextError::UnexpectedEnd {
                    expected: "transition",
                })?;
            let transition_line_no = transition_index + 1;
            let transition_values = parse_record(
                transition_line,
                "transition",
                &["source", "target", "trigger", "guard", "priority", "kind"],
            )
            .ok_or(TreeDefTextError::MalformedRecord {
                line: transition_line_no,
            })?;
            let source = StateId(parse_u32(
                transition_values[0],
                transition_line_no,
                "source",
            )?);
            let target = parse_target(transition_values[1], transition_line_no)?;
            let trigger = parse_trigger(transition_values[2], transition_line_no)?;
            let guard = parse_guard(transition_values[3], transition_line_no)?;
            let priority = parse_u16(transition_values[4], transition_line_no, "priority")?;
            let kind = parse_transition_kind(transition_values[5], transition_line_no)?;
            node.transitions.push(Transition::new(
                source, target, trigger, guard, priority, kind,
            ));
        }
        nodes.push(node);
    }
    if let Some((index, _)) = lines.next() {
        return Err(TreeDefTextError::ExtraInput { line: index + 1 });
    }

    let definition = TreeDef { root, nodes };
    definition
        .clone()
        .compile()
        .map_err(|error| TreeDefTextError::InvalidDefinition { error })?;
    Ok(definition)
}

fn write_record(text: &mut String, name: &str, keys: &[&str], values: &[String]) {
    text.push_str(name);
    for (key, value) in keys.iter().zip(values) {
        let _ = write!(text, " {key}={value}");
    }
    text.push('\n');
}

fn option_id(value: Option<StateId>) -> String {
    value.map_or_else(|| "-".to_string(), |id| id.0.to_string())
}

fn id_list(values: &[StateId]) -> String {
    if values.is_empty() {
        return "-".to_string();
    }
    let mut text = String::new();
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            text.push(',');
        }
        let _ = write!(text, "{}", value.0);
    }
    text
}

fn history_name(value: Option<HistoryMode>) -> &'static str {
    match value {
        None => "none",
        Some(HistoryMode::Shallow) => "shallow",
        Some(HistoryMode::Deep) => "deep",
    }
}

fn target_text(value: TransitionTarget) -> String {
    match value {
        TransitionTarget::State(id) => format_value("state:", id.0),
        TransitionTarget::History { owner, depth } => {
            let mut text = format_value("history:", owner.0);
            text.push(':');
            text.push_str(history_name(Some(depth)));
            text
        }
        TransitionTarget::SelfState => "self".to_string(),
    }
}

fn trigger_text(value: Trigger) -> String {
    match value {
        Trigger::Event(id) => format_value("event:", id.0),
        Trigger::Eventless => "eventless".to_string(),
    }
}

fn guard_text(value: Option<prism_state_tree_core::GuardId>) -> String {
    value.map_or_else(|| "none".to_string(), |id| format_value("id:", id.0))
}

fn transition_kind_name(value: TransitionKind) -> &'static str {
    match value {
        TransitionKind::External => "external",
        TransitionKind::Internal => "internal",
    }
}

fn format_value(prefix: &str, value: u32) -> String {
    let mut text = String::from(prefix);
    let _ = write!(text, "{value}");
    text
}

fn parse_record<'a>(line: &'a str, name: &str, keys: &[&str]) -> Option<Vec<&'a str>> {
    let mut tokens = line.split(' ');
    if tokens.next()? != name {
        return None;
    }
    let mut values = Vec::with_capacity(keys.len());
    for key in keys {
        let token = tokens.next()?;
        let (actual_key, value) = token.split_once('=')?;
        if actual_key != *key || value.is_empty() {
            return None;
        }
        values.push(value);
    }
    if tokens.next().is_some() {
        return None;
    }
    Some(values)
}

fn parse_u32(value: &str, line: usize, field: &'static str) -> Result<u32, TreeDefTextError> {
    value
        .parse()
        .map_err(|_| TreeDefTextError::InvalidNumber { line, field })
}

fn parse_u16(value: &str, line: usize, field: &'static str) -> Result<u16, TreeDefTextError> {
    value
        .parse()
        .map_err(|_| TreeDefTextError::InvalidNumber { line, field })
}

fn parse_usize(value: &str, line: usize, field: &'static str) -> Result<usize, TreeDefTextError> {
    value
        .parse()
        .map_err(|_| TreeDefTextError::InvalidNumber { line, field })
}

fn parse_optional_id(
    value: &str,
    line: usize,
    field: &'static str,
) -> Result<Option<StateId>, TreeDefTextError> {
    if value == "-" {
        Ok(None)
    } else {
        Ok(Some(StateId(parse_u32(value, line, field)?)))
    }
}

fn parse_id_list(
    value: &str,
    line: usize,
    field: &'static str,
) -> Result<Vec<StateId>, TreeDefTextError> {
    if value == "-" {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    for item in value.split(',') {
        if item.is_empty() {
            return Err(TreeDefTextError::InvalidValue { line, field });
        }
        result.push(StateId(parse_u32(item, line, field)?));
    }
    Ok(result)
}

fn parse_history(
    value: &str,
    line: usize,
    field: &'static str,
) -> Result<Option<HistoryMode>, TreeDefTextError> {
    match value {
        "none" => Ok(None),
        "shallow" => Ok(Some(HistoryMode::Shallow)),
        "deep" => Ok(Some(HistoryMode::Deep)),
        _ => Err(TreeDefTextError::InvalidValue { line, field }),
    }
}

fn parse_target(value: &str, line: usize) -> Result<TransitionTarget, TreeDefTextError> {
    if value == "self" {
        return Ok(TransitionTarget::SelfState);
    }
    let (kind, id) = value
        .split_once(':')
        .ok_or(TreeDefTextError::InvalidValue {
            line,
            field: "target",
        })?;
    match kind {
        "state" => Ok(TransitionTarget::State(StateId(parse_u32(
            id, line, "target",
        )?))),
        "history" => {
            let mut parts = id.split(':');
            let owner = StateId(parse_u32(parts.next().unwrap_or(""), line, "target")?);
            let depth = match parts.next() {
                Some("shallow") => HistoryMode::Shallow,
                Some("deep") => HistoryMode::Deep,
                _ => {
                    return Err(TreeDefTextError::InvalidValue {
                        line,
                        field: "target",
                    })
                }
            };
            if parts.next().is_some() {
                return Err(TreeDefTextError::InvalidValue {
                    line,
                    field: "target",
                });
            }
            Ok(TransitionTarget::History { owner, depth })
        }
        _ => Err(TreeDefTextError::InvalidValue {
            line,
            field: "target",
        }),
    }
}

fn parse_trigger(value: &str, line: usize) -> Result<Trigger, TreeDefTextError> {
    if value == "eventless" {
        return Ok(Trigger::Eventless);
    }
    let (kind, id) = value
        .split_once(':')
        .ok_or(TreeDefTextError::InvalidValue {
            line,
            field: "trigger",
        })?;
    if kind != "event" {
        return Err(TreeDefTextError::InvalidValue {
            line,
            field: "trigger",
        });
    }
    Ok(Trigger::Event(prism_state_tree_core::EventId(parse_u32(
        id, line, "trigger",
    )?)))
}

fn parse_guard(
    value: &str,
    line: usize,
) -> Result<Option<prism_state_tree_core::GuardId>, TreeDefTextError> {
    if value == "none" {
        return Ok(None);
    }
    let (kind, id) = value
        .split_once(':')
        .ok_or(TreeDefTextError::InvalidValue {
            line,
            field: "guard",
        })?;
    if kind != "id" {
        return Err(TreeDefTextError::InvalidValue {
            line,
            field: "guard",
        });
    }
    Ok(Some(prism_state_tree_core::GuardId(parse_u32(
        id, line, "guard",
    )?)))
}

fn parse_transition_kind(value: &str, line: usize) -> Result<TransitionKind, TreeDefTextError> {
    match value {
        "external" => Ok(TransitionKind::External),
        "internal" => Ok(TransitionKind::Internal),
        _ => Err(TreeDefTextError::InvalidValue {
            line,
            field: "kind",
        }),
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn definition() -> TreeDef {
        TreeDef {
            root: StateId(0),
            nodes: vec![
                StateNode::parallel(
                    StateId(0),
                    None,
                    vec![StateId(1), StateId(2)],
                    vec![StateId(1), StateId(2)],
                    Some(HistoryMode::Deep),
                ),
                StateNode::compound(
                    StateId(1),
                    Some(StateId(0)),
                    vec![StateId(3), StateId(4)],
                    StateId(3),
                    Some(HistoryMode::Shallow),
                )
                .with_transition(Transition::new(
                    StateId(1),
                    TransitionTarget::History {
                        owner: StateId(0),
                        depth: HistoryMode::Deep,
                    },
                    Trigger::Eventless,
                    Some(prism_state_tree_core::GuardId(9)),
                    12,
                    TransitionKind::Internal,
                )),
                StateNode::atomic(StateId(2), Some(StateId(0))).with_transition(Transition::new(
                    StateId(2),
                    TransitionTarget::SelfState,
                    Trigger::Event(prism_state_tree_core::EventId(7)),
                    None,
                    0,
                    TransitionKind::External,
                )),
                StateNode::atomic(StateId(3), Some(StateId(1))).with_transition(Transition::new(
                    StateId(3),
                    TransitionTarget::State(StateId(4)),
                    Trigger::Event(prism_state_tree_core::EventId(8)),
                    None,
                    u16::MAX,
                    TransitionKind::External,
                )),
                StateNode::atomic(StateId(4), Some(StateId(1))),
            ],
        }
    }

    #[test]
    fn canonical_text_roundtrips_all_tree_def_fields() {
        let original = definition();
        let text = serialize_tree_def(&original).unwrap();
        let decoded = parse_tree_def(&text).unwrap();
        assert_eq!(serialize_tree_def(&decoded).unwrap(), text);
        assert_eq!(decoded.root, original.root);
        assert_eq!(decoded.nodes, original.nodes);
    }

    #[test]
    fn parser_rejects_unknown_or_reordered_fields() {
        let text = serialize_tree_def(&definition()).unwrap();
        let unknown = text.replacen("root=0", "root=0 extra=1", 1);
        assert!(matches!(
            parse_tree_def(&unknown),
            Err(TreeDefTextError::InvalidHeader { .. })
        ));
        let reordered = text.replacen("version=1 root=0 nodes=5", "root=0 version=1 nodes=5", 1);
        assert!(matches!(
            parse_tree_def(&reordered),
            Err(TreeDefTextError::InvalidHeader { .. })
        ));
    }

    #[test]
    fn parser_rejects_semantically_invalid_tree() {
        let text = serialize_tree_def(&definition()).unwrap().replacen(
            "id=4 parent=1",
            "id=4 parent=99",
            1,
        );
        assert!(matches!(
            parse_tree_def(&text),
            Err(TreeDefTextError::InvalidDefinition {
                error: PlanError::UnknownState(StateId(99))
            })
        ));
    }

    #[test]
    fn parser_rejects_blank_lines_and_trailing_records() {
        let text = serialize_tree_def(&definition()).unwrap();
        assert!(matches!(
            parse_tree_def(&text.replacen('\n', "\n\n", 1)),
            Err(TreeDefTextError::MalformedRecord { .. })
        ));
        assert!(matches!(
            parse_tree_def(&(text + "garbage\n")),
            Err(TreeDefTextError::ExtraInput { .. })
        ));
    }
}
