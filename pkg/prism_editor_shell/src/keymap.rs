//! Keyboard sequences, context expressions, and conflict detection.

use std::collections::{BTreeMap, BTreeSet};

/// A normalized keyboard modifier.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Modifier {
    /// Control modifier.
    Ctrl,
    /// Alt/Option modifier.
    Alt,
    /// Shift modifier.
    Shift,
    /// Meta/Command/Windows modifier.
    Meta,
}

impl Modifier {
    fn parse(token: &str) -> Option<Self> {
        match token.to_lowercase().as_str() {
            "ctrl" | "control" => Some(Self::Ctrl),
            "alt" | "option" => Some(Self::Alt),
            "shift" => Some(Self::Shift),
            "meta" | "cmd" | "command" | "super" | "win" | "windows" => Some(Self::Meta),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Ctrl => "Ctrl",
            Self::Alt => "Alt",
            Self::Shift => "Shift",
            Self::Meta => "Meta",
        }
    }
}

/// One normalized keyboard stroke such as `Ctrl+Shift+P`.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct KeyStroke {
    modifiers: Vec<Modifier>,
    key: String,
}

impl KeyStroke {
    /// Parses a stroke from canonical or common modifier aliases.
    pub fn parse(input: &str) -> Result<Self, KeyCodeError> {
        let mut modifiers = Vec::new();
        let mut key = None;
        for token in input
            .split('+')
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            if let Some(modifier) = Modifier::parse(token) {
                if modifiers.contains(&modifier) {
                    return Err(KeyCodeError::DuplicateModifier(token.to_owned()));
                }
                modifiers.push(modifier);
            } else if key.replace(token.to_owned()).is_some() {
                return Err(KeyCodeError::MultipleKeys(input.to_owned()));
            }
        }
        let Some(key) = key else {
            return Err(KeyCodeError::MissingKey(input.to_owned()));
        };
        modifiers.sort();
        Ok(Self {
            modifiers,
            key: key.to_uppercase(),
        })
    }

    /// Returns the normalized modifier list.
    #[must_use]
    pub fn modifiers(&self) -> &[Modifier] {
        &self.modifiers
    }

    /// Returns the normalized key name.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }
}

impl std::fmt::Display for KeyStroke {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for modifier in &self.modifiers {
            write!(formatter, "{}+", modifier.as_str())?;
        }
        formatter.write_str(&self.key)
    }
}

/// A one- or multi-stroke keyboard sequence.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct KeySequence(Vec<KeyStroke>);

impl KeySequence {
    /// Parses `Ctrl+K Ctrl+C` or `Ctrl+K, Ctrl+C`.
    pub fn parse(input: &str) -> Result<Self, KeyCodeError> {
        let strokes = input
            .split([',', ' '])
            .map(str::trim)
            .filter(|stroke| !stroke.is_empty())
            .map(KeyStroke::parse)
            .collect::<Result<Vec<_>, _>>()?;
        if strokes.is_empty() {
            return Err(KeyCodeError::EmptySequence);
        }
        Ok(Self(strokes))
    }

    /// Returns the sequence strokes.
    #[must_use]
    pub fn strokes(&self) -> &[KeyStroke] {
        &self.0
    }
}

impl std::fmt::Display for KeySequence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, stroke) in self.0.iter().enumerate() {
            if index != 0 {
                formatter.write_str(" ")?;
            }
            stroke.fmt(formatter)?;
        }
        Ok(())
    }
}

/// An error raised when parsing a keyboard sequence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KeyCodeError {
    /// The sequence has no strokes.
    EmptySequence,
    /// A stroke has no non-modifier key.
    MissingKey(String),
    /// A stroke contains more than one key.
    MultipleKeys(String),
    /// A modifier appears more than once.
    DuplicateModifier(String),
}

impl std::fmt::Display for KeyCodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySequence => formatter.write_str("key sequence cannot be empty"),
            Self::MissingKey(input) => write!(formatter, "key stroke has no key: {input}"),
            Self::MultipleKeys(input) => write!(formatter, "key stroke has multiple keys: {input}"),
            Self::DuplicateModifier(modifier) => {
                write!(formatter, "duplicate modifier: {modifier}")
            }
        }
    }
}

impl std::error::Error for KeyCodeError {}

/// A set of active context keys.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ContextKeySet(BTreeSet<String>);

impl ContextKeySet {
    /// Creates an empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a key after trimming it.
    pub fn insert(&mut self, key: impl Into<String>) {
        let key = key.into();
        let key = key.trim();
        if !key.is_empty() {
            self.0.insert(key.to_owned());
        }
    }

    /// Tests whether a key is active.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.0.contains(key)
    }
}

/// A boolean expression over editor context keys.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ContextExpr {
    /// Always matches.
    #[default]
    Any,
    /// Matches one named key.
    Key(String),
    /// Negates a child expression.
    Not(Box<Self>),
    /// All child expressions must match.
    All(Vec<Self>),
    /// At least one child expression must match.
    AnyOf(Vec<Self>),
}

impl ContextExpr {
    /// Parses identifiers with `!`, `&`, `|`, and parentheses.
    pub fn parse(input: &str) -> Result<Self, ContextParseError> {
        Parser::new(input).parse()
    }

    /// Evaluates this expression against active context keys.
    #[must_use]
    pub fn matches(&self, keys: &ContextKeySet) -> bool {
        match self {
            Self::Any => true,
            Self::Key(key) => keys.contains(key),
            Self::Not(child) => !child.matches(keys),
            Self::All(children) => children.iter().all(|child| child.matches(keys)),
            Self::AnyOf(children) => children.iter().any(|child| child.matches(keys)),
        }
    }

    fn dnf(&self, negate: bool) -> Vec<Vec<Literal>> {
        match (self, negate) {
            (Self::Any, false) => vec![Vec::new()],
            (Self::Any, true) => Vec::new(),
            (Self::Key(key), negate) => vec![vec![Literal {
                key: key.clone(),
                positive: !negate,
            }]],
            (Self::Not(child), negate) => child.dnf(!negate),
            (Self::All(children), false) => {
                combine_dnf(children.iter().map(|child| child.dnf(false)))
            }
            (Self::AnyOf(children), false) => {
                children.iter().flat_map(|child| child.dnf(false)).collect()
            }
            (Self::All(children), true) => {
                children.iter().flat_map(|child| child.dnf(true)).collect()
            }
            (Self::AnyOf(children), true) => {
                combine_dnf(children.iter().map(|child| child.dnf(true)))
            }
        }
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.dnf(false).iter().any(|left| {
            other
                .dnf(false)
                .iter()
                .any(|right| clauses_compatible(left, right))
        })
    }
}

/// Parser error for a context expression.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextParseError {
    /// Byte offset at which parsing stopped.
    pub position: usize,
    /// Human-readable reason.
    pub message: String,
}

impl std::fmt::Display for ContextParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "context parse error at {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for ContextParseError {}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Literal {
    key: String,
    positive: bool,
}

fn clauses_compatible(left: &[Literal], right: &[Literal]) -> bool {
    let mut values = BTreeMap::<&str, bool>::new();
    left.iter().chain(right).all(|literal| {
        values
            .insert(literal.key.as_str(), literal.positive)
            .is_none_or(|existing| existing == literal.positive)
    })
}

fn combine_dnf(parts: impl Iterator<Item = Vec<Vec<Literal>>>) -> Vec<Vec<Literal>> {
    let mut result = vec![Vec::new()];
    for part in parts {
        let mut next = Vec::new();
        for prefix in &result {
            for clause in &part {
                let mut combined = prefix.clone();
                combined.extend(clause.iter().cloned());
                next.push(combined);
            }
        }
        result = next;
        if result.is_empty() {
            break;
        }
    }
    result
}

struct Parser<'a> {
    input: &'a str,
    position: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, position: 0 }
    }

    fn parse(mut self) -> Result<ContextExpr, ContextParseError> {
        self.skip_spaces();
        if self.position == self.input.len() {
            return Ok(ContextExpr::Any);
        }
        let expression = self.parse_or()?;
        self.skip_spaces();
        if self.position != self.input.len() {
            return self.error("unexpected character");
        }
        Ok(expression)
    }

    fn parse_or(&mut self) -> Result<ContextExpr, ContextParseError> {
        let mut children = vec![self.parse_and()?];
        loop {
            self.skip_spaces();
            if !self.consume('|') {
                break;
            }
            children.push(self.parse_and()?);
        }
        Ok(if children.len() == 1 {
            children.remove(0)
        } else {
            ContextExpr::AnyOf(children)
        })
    }

    fn parse_and(&mut self) -> Result<ContextExpr, ContextParseError> {
        let mut children = vec![self.parse_unary()?];
        loop {
            self.skip_spaces();
            if !self.consume('&') {
                break;
            }
            children.push(self.parse_unary()?);
        }
        Ok(if children.len() == 1 {
            children.remove(0)
        } else {
            ContextExpr::All(children)
        })
    }

    fn parse_unary(&mut self) -> Result<ContextExpr, ContextParseError> {
        self.skip_spaces();
        if self.consume('!') {
            return Ok(ContextExpr::Not(Box::new(self.parse_unary()?)));
        }
        if self.consume('(') {
            let expression = self.parse_or()?;
            self.skip_spaces();
            if !self.consume(')') {
                return self.error("expected ')'");
            }
            return Ok(expression);
        }
        self.parse_key()
    }

    fn parse_key(&mut self) -> Result<ContextExpr, ContextParseError> {
        self.skip_spaces();
        let start = self.position;
        while let Some(character) = self.input[self.position..].chars().next() {
            if character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':' | '/') {
                self.position += character.len_utf8();
            } else {
                break;
            }
        }
        if start == self.position {
            return self.error("expected context key");
        }
        Ok(ContextExpr::Key(
            self.input[start..self.position].to_owned(),
        ))
    }

    fn skip_spaces(&mut self) {
        while self.input[self.position..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
        {
            self.position += self.input[self.position..]
                .chars()
                .next()
                .expect("checked")
                .len_utf8();
        }
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.input[self.position..].starts_with(expected) {
            self.position += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn error<T>(&self, message: &str) -> Result<T, ContextParseError> {
        Err(ContextParseError {
            position: self.position,
            message: message.to_owned(),
        })
    }
}

/// One command key binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyBinding {
    /// Command to invoke.
    pub command_id: String,
    /// Keyboard sequence.
    pub sequence: KeySequence,
    /// Context in which this binding is active.
    pub when: ContextExpr,
    /// Higher priority bindings are returned first for matching contexts.
    pub priority: i32,
}

impl KeyBinding {
    /// Creates a binding with an always-active context and zero priority.
    #[must_use]
    pub fn new(command_id: impl Into<String>, sequence: KeySequence) -> Self {
        Self {
            command_id: command_id.into(),
            sequence,
            when: ContextExpr::Any,
            priority: 0,
        }
    }

    /// Sets the activation context.
    #[must_use]
    pub fn when(mut self, when: ContextExpr) -> Self {
        self.when = when;
        self
    }

    /// Sets dispatch priority.
    #[must_use]
    pub const fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// A token that can unregister one key binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KeyBindingRegistration {
    serial: u64,
}

/// A conflict between two active bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingConflict {
    /// Command already in the keymap.
    pub existing_command: String,
    /// Command that would be added.
    pub incoming_command: String,
    /// Shared keyboard sequence.
    pub sequence: KeySequence,
}

/// Error raised while registering a key binding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KeymapError {
    /// A binding has no command ID.
    EmptyCommandId,
    /// The binding overlaps an existing binding.
    Conflict(BindingConflict),
}

impl std::fmt::Display for KeymapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCommandId => formatter.write_str("key binding command id cannot be empty"),
            Self::Conflict(conflict) => write!(
                formatter,
                "key conflict between {} and {} on {}",
                conflict.existing_command, conflict.incoming_command, conflict.sequence
            ),
        }
    }
}

impl std::error::Error for KeymapError {}

struct RegisteredBinding {
    serial: u64,
    binding: KeyBinding,
}

/// Deterministic keymap with conservative registration-time conflict rejection.
#[derive(Default)]
pub struct Keymap {
    bindings: Vec<RegisteredBinding>,
    next_serial: u64,
}

impl Keymap {
    /// Registers a binding unless its sequence and context overlap an existing one.
    pub fn register(&mut self, binding: KeyBinding) -> Result<KeyBindingRegistration, KeymapError> {
        if binding.command_id.trim().is_empty() {
            return Err(KeymapError::EmptyCommandId);
        }
        if let Some(existing) = self.bindings.iter().find(|existing| {
            existing.binding.sequence == binding.sequence
                && existing.binding.when.overlaps(&binding.when)
        }) {
            return Err(KeymapError::Conflict(BindingConflict {
                existing_command: existing.binding.command_id.clone(),
                incoming_command: binding.command_id,
                sequence: binding.sequence,
            }));
        }
        let serial = self.allocate_serial();
        self.bindings.push(RegisteredBinding { serial, binding });
        Ok(KeyBindingRegistration { serial })
    }

    /// Removes a registered binding.
    pub fn unregister(&mut self, registration: KeyBindingRegistration) -> bool {
        let Some(index) = self
            .bindings
            .iter()
            .position(|binding| binding.serial == registration.serial)
        else {
            return false;
        };
        self.bindings.remove(index);
        true
    }

    /// Returns all commands matching a sequence and active context.
    #[must_use]
    pub fn resolve(&self, sequence: &KeySequence, context: &ContextKeySet) -> Vec<String> {
        let mut result = self
            .bindings
            .iter()
            .filter(|binding| {
                binding.binding.sequence == *sequence && binding.binding.when.matches(context)
            })
            .map(|binding| (binding.binding.priority, binding.binding.command_id.clone()))
            .collect::<Vec<_>>();
        result.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
        result.into_iter().map(|(_, command)| command).collect()
    }

    /// Returns the number of registered bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns whether no bindings are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    fn allocate_serial(&mut self) -> u64 {
        self.next_serial = self.next_serial.saturating_add(1);
        self.next_serial
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_normalizes_strokes_and_contexts() {
        let sequence = KeySequence::parse("Command+Shift+P, ctrl+k").expect("sequence");
        assert_eq!(sequence.to_string(), "Shift+Meta+P Ctrl+K");
        let expression = ContextExpr::parse("editor.focus & !modal | palette").expect("context");
        let mut keys = ContextKeySet::new();
        keys.insert("editor.focus");
        assert!(expression.matches(&keys));
        keys.insert("modal");
        assert!(!expression.matches(&keys));
        keys.insert("palette");
        assert!(expression.matches(&keys));
    }

    #[test]
    fn overlapping_contexts_are_rejected_but_disjoint_contexts_are_allowed() {
        let sequence = KeySequence::parse("Ctrl+P").expect("sequence");
        let mut keymap = Keymap::default();
        keymap
            .register(
                KeyBinding::new("palette.open", sequence.clone())
                    .when(ContextExpr::parse("editor.focus").expect("context")),
            )
            .expect("first binding");
        let conflict = keymap.register(
            KeyBinding::new("other", sequence.clone())
                .when(ContextExpr::parse("editor.focus | modal").expect("context")),
        );
        assert!(matches!(conflict, Err(KeymapError::Conflict(_))));
        keymap
            .register(
                KeyBinding::new("modal.command", sequence)
                    .when(ContextExpr::parse("modal & !editor.focus").expect("context")),
            )
            .expect("disjoint binding");
    }

    #[test]
    fn resolution_is_priority_then_id_sorted() {
        let sequence = KeySequence::parse("Ctrl+K").expect("sequence");
        let mut keymap = Keymap::default();
        keymap
            .register(KeyBinding::new("z.command", sequence.clone()).priority(1))
            .expect("register");
        keymap
            .register(
                KeyBinding::new("a.command", KeySequence::parse("Ctrl+L").expect("sequence"))
                    .priority(10),
            )
            .expect("register");
        let keys = ContextKeySet::new();
        assert_eq!(keymap.resolve(&sequence, &keys), ["z.command"]);
    }
}
