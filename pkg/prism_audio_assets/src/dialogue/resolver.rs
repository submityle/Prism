//! Runtime dialogue resolution: key values to a concrete media reference.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the "programmatic dialogue, media selected at runtime" model of
//! design section 35. The game emits a semantic request keyed by role, emotion,
//! language, and variant id; the resolver maps it to a [`MediaRef`] drawn from
//! the active voice bank. Resolution is non-real-time (table lookup); a miss
//! yields silence plus a warning count rather than a panic.

#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};

use alloc::collections::BTreeMap;

use crate::bank::entry::EntryId;
use crate::bank::manifest::BankId;
use crate::dialogue::localization::LanguageId;

/// A bank-qualified reference to a resolved media entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MediaRef {
    /// The bank the media lives in (usually the active voice bank).
    pub bank: BankId,
    /// The bank-local media entry.
    pub entry: EntryId,
}

impl MediaRef {
    /// Creates a new media reference.
    #[must_use]
    pub fn new(bank: BankId, entry: EntryId) -> Self {
        Self { bank, entry }
    }
}

/// The lookup key for a dialogue line.
///
/// A line is addressed by the speaking role, the emotional colour, the spoken
/// language, and an explicit variant id used to disambiguate alternate takes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DialogueKey {
    /// Speaking character / role, for example `hero`.
    pub role: String,
    /// Emotional colour, for example `angry` or `neutral`.
    pub emotion: String,
    /// Spoken language.
    pub language: LanguageId,
    /// Variant id disambiguating alternate takes of the same line.
    pub variant: u32,
}

impl DialogueKey {
    /// Creates a dialogue key.
    #[must_use]
    pub fn new(role: &str, emotion: &str, language: LanguageId, variant: u32) -> Self {
        Self {
            role: role.to_string(),
            emotion: emotion.to_string(),
            language,
            variant,
        }
    }
}

/// The result of resolving a [`DialogueKey`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DialogueResolution {
    /// A concrete media reference was found.
    Resolved(MediaRef),
    /// No media matched; the caller must play silence and warn.
    Silence {
        /// The key that failed to resolve, for telemetry.
        key: DialogueKey,
    },
}

impl DialogueResolution {
    /// Returns the resolved media reference, if any.
    #[must_use]
    pub fn media(&self) -> Option<MediaRef> {
        match self {
            DialogueResolution::Resolved(media) => Some(*media),
            DialogueResolution::Silence { .. } => None,
        }
    }

    /// Returns `true` when resolution produced silence.
    #[must_use]
    pub fn is_silence(&self) -> bool {
        matches!(self, DialogueResolution::Silence { .. })
    }
}

/// A table-driven dialogue resolver.
///
/// Entries map an exact [`DialogueKey`] to a [`MediaRef`]. When an exact match
/// is absent the resolver tries the same role/emotion/language with variant `0`
/// as a generic take before giving up and reporting silence. Every miss bumps
/// [`DialogueResolver::miss_count`] so the telemetry ring can warn.
#[derive(Debug, Clone, Default)]
pub struct DialogueResolver {
    table: BTreeMap<DialogueKey, MediaRef>,
    miss_count: u64,
}

impl DialogueResolver {
    /// Creates an empty resolver.
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: BTreeMap::new(),
            miss_count: 0,
        }
    }

    /// Binds `key` to `media`, replacing any prior binding.
    pub fn insert(&mut self, key: DialogueKey, media: MediaRef) {
        self.table.insert(key, media);
    }

    /// Returns the number of bound keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns `true` when no keys are bound.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Returns how many resolutions have fallen through to silence.
    #[must_use]
    pub fn miss_count(&self) -> u64 {
        self.miss_count
    }

    /// Resolves `key` to a media reference, counting and reporting misses.
    ///
    /// Resolution order: exact key, then the role/emotion/language generic take
    /// (variant `0`), then silence.
    pub fn resolve(&mut self, key: &DialogueKey) -> DialogueResolution {
        if let Some(media) = self.table.get(key) {
            return DialogueResolution::Resolved(*media);
        }
        if key.variant != 0 {
            let generic = DialogueKey {
                role: key.role.clone(),
                emotion: key.emotion.clone(),
                language: key.language.clone(),
                variant: 0,
            };
            if let Some(media) = self.table.get(&generic) {
                return DialogueResolution::Resolved(*media);
            }
        }
        self.miss_count += 1;
        DialogueResolution::Silence { key: key.clone() }
    }

    /// Resolves without mutating the miss counter (for speculative lookups).
    #[must_use]
    pub fn peek(&self, key: &DialogueKey) -> Option<MediaRef> {
        self.table.get(key).copied()
    }
}
