//! A single locale's message catalog.
//!
//! A [`Catalog`] maps string keys to [`Message`]s. A message is either a simple
//! template or a set of plural variants keyed by [`PluralCategory`].

use alloc::collections::BTreeMap;
use alloc::string::String;

use crate::plural::PluralCategory;

/// A single catalog entry.
#[derive(Clone, Debug)]
pub enum Message {
    /// A plain template with optional `{name}` placeholders.
    Simple(String),
    /// Per-category plural variants, selected by a count at call time.
    Plural(BTreeMap<PluralCategory, String>),
}

/// A collection of messages for one locale.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    messages: BTreeMap<String, Message>,
}

impl Catalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a simple message, returning `self` for chaining.
    #[must_use]
    pub fn with(mut self, key: impl Into<String>, template: impl Into<String>) -> Self {
        self.insert(key, template);
        self
    }

    /// Insert a plural message from `(category, template)` pairs, returning
    /// `self` for chaining.
    #[must_use]
    pub fn with_plural<S>(
        mut self,
        key: impl Into<String>,
        variants: impl IntoIterator<Item = (PluralCategory, S)>,
    ) -> Self
    where
        S: Into<String>,
    {
        self.insert_plural(key, variants);
        self
    }

    /// Insert or replace a simple message in place.
    pub fn insert(&mut self, key: impl Into<String>, template: impl Into<String>) -> &mut Self {
        self.messages
            .insert(key.into(), Message::Simple(template.into()));
        self
    }

    /// Insert or replace a plural message in place.
    pub fn insert_plural<S>(
        &mut self,
        key: impl Into<String>,
        variants: impl IntoIterator<Item = (PluralCategory, S)>,
    ) -> &mut Self
    where
        S: Into<String>,
    {
        let map = variants
            .into_iter()
            .map(|(category, template)| (category, template.into()))
            .collect();
        self.messages.insert(key.into(), Message::Plural(map));
        self
    }

    /// Look up the message bound to `key`, if any.
    pub fn message(&self, key: &str) -> Option<&Message> {
        self.messages.get(key)
    }
}
