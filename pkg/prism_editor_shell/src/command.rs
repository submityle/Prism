//! Command palette entries and deterministic text search.

use std::collections::BTreeMap;

/// A command exposed by the editor shell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandEntry {
    /// Stable command identifier.
    pub id: String,
    /// Human-readable command title.
    pub title: String,
    /// Optional grouping label shown by a command palette.
    pub category: Option<String>,
    /// Additional terms that participate in search.
    pub keywords: Vec<String>,
    /// Whether the command is currently eligible for invocation.
    pub enabled: bool,
}

impl CommandEntry {
    /// Creates an enabled command without a category or extra keywords.
    #[must_use]
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            category: None,
            keywords: Vec::new(),
            enabled: true,
        }
    }

    /// Sets the optional category.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Adds a search keyword.
    #[must_use]
    pub fn with_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.keywords.push(keyword.into());
        self
    }

    /// Sets whether the command can currently be invoked.
    #[must_use]
    pub const fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// A token that can unregister one command entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommandRegistration {
    serial: u64,
}

/// An error raised while modifying a command index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandIndexError {
    /// The identifier is already registered.
    DuplicateId(String),
    /// The identifier is empty or contains only whitespace.
    EmptyId,
}

impl std::fmt::Display for CommandIndexError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(formatter, "duplicate command id {id}"),
            Self::EmptyId => formatter.write_str("command id cannot be empty"),
        }
    }
}

impl std::error::Error for CommandIndexError {}

/// One result returned by command palette search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandSearchResult {
    /// Matching command identifier.
    pub id: String,
    /// Matching command title.
    pub title: String,
    /// Optional category.
    pub category: Option<String>,
    /// Lower scores are better matches.
    pub score: u32,
    /// Whether the command can currently be invoked.
    pub enabled: bool,
}

struct IndexedCommand {
    serial: u64,
    entry: CommandEntry,
}

/// Deterministic command index used by a renderer-independent command panel.
#[derive(Default)]
pub struct CommandIndex {
    entries: BTreeMap<String, IndexedCommand>,
    next_serial: u64,
}

impl CommandIndex {
    /// Registers an entry and returns a disposable registration token.
    pub fn register(
        &mut self,
        entry: CommandEntry,
    ) -> Result<CommandRegistration, CommandIndexError> {
        if entry.id.trim().is_empty() {
            return Err(CommandIndexError::EmptyId);
        }
        if self.entries.contains_key(&entry.id) {
            return Err(CommandIndexError::DuplicateId(entry.id));
        }
        let serial = self.allocate_serial();
        self.entries
            .insert(entry.id.clone(), IndexedCommand { serial, entry });
        Ok(CommandRegistration { serial })
    }

    /// Removes the entry owned by `registration`.
    pub fn unregister(&mut self, registration: CommandRegistration) -> bool {
        let Some(id) = self
            .entries
            .iter()
            .find_map(|(id, entry)| (entry.serial == registration.serial).then(|| id.clone()))
        else {
            return false;
        };
        self.entries.remove(&id).is_some()
    }

    /// Returns an entry by stable identifier.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&CommandEntry> {
        self.entries.get(id).map(|indexed| &indexed.entry)
    }

    /// Returns the number of registered commands.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether no commands are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Searches enabled and disabled commands using deterministic token scoring.
    ///
    /// An empty query returns all commands. Search terms must all match one of
    /// the ID, title, category, or keyword fields. Results are ordered by score,
    /// then title, then ID; `limit` truncates after sorting.
    #[must_use]
    pub fn search(&self, query: &str, limit: usize) -> Vec<CommandSearchResult> {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(str::to_lowercase)
            .filter(|term| !term.is_empty())
            .collect();
        let mut results = self
            .entries
            .values()
            .filter_map(|indexed| {
                let score = search_score(&indexed.entry, &terms)?;
                Some(CommandSearchResult {
                    id: indexed.entry.id.clone(),
                    title: indexed.entry.title.clone(),
                    category: indexed.entry.category.clone(),
                    score,
                    enabled: indexed.entry.enabled,
                })
            })
            .collect::<Vec<_>>();
        results.sort_by(|left, right| {
            left.score
                .cmp(&right.score)
                .then_with(|| left.title.cmp(&right.title))
                .then_with(|| left.id.cmp(&right.id))
        });
        results.truncate(limit);
        results
    }

    fn allocate_serial(&mut self) -> u64 {
        self.next_serial = self.next_serial.saturating_add(1);
        self.next_serial
    }
}

fn search_score(entry: &CommandEntry, terms: &[String]) -> Option<u32> {
    if terms.is_empty() {
        return Some(0);
    }
    let id = entry.id.to_lowercase();
    let title = entry.title.to_lowercase();
    let category = entry.category.as_deref().unwrap_or_default().to_lowercase();
    let keywords = entry
        .keywords
        .iter()
        .map(|keyword| keyword.to_lowercase())
        .collect::<Vec<_>>();
    let mut score = 0;
    for term in terms {
        let term_score = if id == *term || title == *term {
            0
        } else if id.starts_with(term) || title.starts_with(term) {
            1
        } else if id.split(['.', '/', ':', '_', '-']).any(|word| word == term)
            || title.split_whitespace().any(|word| word == term)
        {
            2
        } else if id.contains(term) || title.contains(term) {
            3
        } else if category.contains(term) {
            4
        } else if keywords.iter().any(|keyword| keyword.contains(term)) {
            5
        } else {
            return None;
        };
        score += term_score;
    }
    Some(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_is_scored_and_deterministically_sorted() {
        let mut index = CommandIndex::default();
        index
            .register(CommandEntry::new("scene.save", "Save Scene"))
            .expect("register");
        index
            .register(CommandEntry::new("asset.save", "Save Asset"))
            .expect("register");
        index
            .register(
                CommandEntry::new("scene.save_as", "Save As")
                    .with_keyword("scene")
                    .enabled(false),
            )
            .expect("register");

        let results = index.search("save", 10);
        assert_eq!(
            results
                .iter()
                .map(|result| result.id.as_str())
                .collect::<Vec<_>>(),
            ["scene.save_as", "asset.save", "scene.save"]
        );
        assert!(!results[0].enabled);
    }

    #[test]
    fn registration_can_be_removed_without_removing_a_replacement() {
        let mut index = CommandIndex::default();
        let first = index
            .register(CommandEntry::new("edit", "Edit"))
            .expect("register");
        assert!(index.unregister(first));
        let second = index
            .register(CommandEntry::new("edit", "Edit Again"))
            .expect("replacement");
        assert!(!index.unregister(first));
        assert_eq!(index.get("edit").expect("entry").title, "Edit Again");
        assert!(index.unregister(second));
    }
}
