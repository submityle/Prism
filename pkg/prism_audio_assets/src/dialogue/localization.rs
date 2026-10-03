//! Language-partitioned voice banks with hot language switching.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the "localization media partitioned by language, switching the
//! language only swaps the voice bank" requirement of design section 35.
//! Logic and timelines are language independent; only the voice [`BankId`]
//! bound to the active language changes. Requesting an unregistered language
//! falls back to the default and raises a telemetry-friendly warning flag
//! rather than going silent.

#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};

use alloc::collections::BTreeMap;

use crate::bank::manifest::BankId;

/// A stable language identifier such as `en`, `zh`, or `ja`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LanguageId(pub String);

impl LanguageId {
    /// Creates a language id from any string-like value.
    #[must_use]
    pub fn new(code: &str) -> Self {
        Self(code.to_string())
    }

    /// Returns the language code as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The outcome of a language switch request.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SwitchOutcome {
    /// The requested language was registered and is now active.
    Switched(LanguageId),
    /// The requested language was not registered; the default is now active.
    FellBackToDefault {
        /// The language that was requested but not registered.
        requested: LanguageId,
        /// The default language that was activated instead.
        fallback: LanguageId,
    },
}

impl SwitchOutcome {
    /// Returns `true` when the switch fell back to the default language.
    #[must_use]
    pub fn is_fallback(&self) -> bool {
        matches!(self, SwitchOutcome::FellBackToDefault { .. })
    }
}

/// A set of per-language voice banks with a designated default.
///
/// Each language maps to the [`BankId`] of its voice bank. The active language
/// selects which bank the dialogue resolver draws media from; switching only
/// changes that binding, leaving all event logic and timelines untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct LanguageBankSet {
    banks: BTreeMap<LanguageId, BankId>,
    default: LanguageId,
    active: LanguageId,
    fallback_count: u64,
}

impl LanguageBankSet {
    /// Creates a set whose default language maps to `default_bank`.
    ///
    /// The default language starts active.
    #[must_use]
    pub fn new(default: LanguageId, default_bank: BankId) -> Self {
        let mut banks = BTreeMap::new();
        banks.insert(default.clone(), default_bank);
        Self {
            banks,
            default: default.clone(),
            active: default,
            fallback_count: 0,
        }
    }

    /// Registers (or replaces) the voice bank bound to `language`.
    pub fn register(&mut self, language: LanguageId, bank: BankId) {
        self.banks.insert(language, bank);
    }

    /// Returns `true` when `language` has a registered voice bank.
    #[must_use]
    pub fn has_language(&self, language: &LanguageId) -> bool {
        self.banks.contains_key(language)
    }

    /// Returns the currently active language.
    #[must_use]
    pub fn active_language(&self) -> &LanguageId {
        &self.active
    }

    /// Returns the default language.
    #[must_use]
    pub fn default_language(&self) -> &LanguageId {
        &self.default
    }

    /// Returns the voice bank bound to the active language.
    #[must_use]
    pub fn active_bank(&self) -> BankId {
        // The active language is always a registered key by construction.
        self.banks
            .get(&self.active)
            .copied()
            .unwrap_or_else(|| self.banks[&self.default])
    }

    /// Returns the voice bank bound to `language`, if registered.
    #[must_use]
    pub fn bank_for(&self, language: &LanguageId) -> Option<BankId> {
        self.banks.get(language).copied()
    }

    /// Switches the active language.
    ///
    /// When `language` is not registered, the default language is activated and
    /// the fallback counter is incremented so telemetry can warn without any
    /// panic or silence.
    pub fn set_active(&mut self, language: LanguageId) -> SwitchOutcome {
        if self.banks.contains_key(&language) {
            self.active = language.clone();
            SwitchOutcome::Switched(language)
        } else {
            self.active = self.default.clone();
            self.fallback_count += 1;
            SwitchOutcome::FellBackToDefault {
                requested: language,
                fallback: self.default.clone(),
            }
        }
    }

    /// Returns how many times a switch has fallen back to the default.
    #[must_use]
    pub fn fallback_count(&self) -> u64 {
        self.fallback_count
    }

    /// Returns the number of registered languages.
    #[must_use]
    pub fn language_count(&self) -> usize {
        self.banks.len()
    }
}
