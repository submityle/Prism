//! The declarative auto-mix ruleset: categories plus ducking rules.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! CRIWARE, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Aggregates the design section 46.8 category axis ([`super::category`]) and
//! the declarative ducking rules ([`super::rule`]) into a single authoring
//! artifact. Validation runs at authoring time; [`AutoMixRuleset::compile`]
//! lowers a validated ruleset onto the design section 12 modulation matrix (see
//! [`super::compiler`]).

use alloc::vec::Vec;

use super::category::{CategoryId, CategorySet};
use super::rule::DuckingRule;

/// Errors produced while validating or compiling an [`AutoMixRuleset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum AutoMixError {
    /// A rule referenced a trigger category that is not registered.
    UnknownTrigger {
        /// Index of the offending rule within the ruleset.
        rule_index: usize,
        /// The out-of-range category id.
        category: CategoryId,
    },
    /// A rule referenced a target category that is not registered.
    UnknownTarget {
        /// Index of the offending rule within the ruleset.
        rule_index: usize,
        /// The out-of-range category id.
        category: CategoryId,
    },
    /// A rule tried to duck a category against itself.
    SelfDuck {
        /// Index of the offending rule within the ruleset.
        rule_index: usize,
        /// The category used as both trigger and target.
        category: CategoryId,
    },
}

/// A validated-on-compile collection of categories and ducking rules.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AutoMixRuleset {
    /// The category registry addressed by every rule.
    categories: CategorySet,
    /// The declarative ducking rules in authoring order.
    rules: Vec<DuckingRule>,
}

impl AutoMixRuleset {
    /// Creates an empty ruleset.
    #[must_use]
    pub fn new() -> Self {
        Self {
            categories: CategorySet::new(),
            rules: Vec::new(),
        }
    }

    /// Registers a category by name and priority, returning its id.
    ///
    /// Registration is idempotent by name; see [`CategorySet::add`].
    pub fn add_category(&mut self, name: &str, priority: i32) -> CategoryId {
        self.categories.add(name, priority)
    }

    /// Appends a ducking rule.
    pub fn add_rule(&mut self, rule: DuckingRule) {
        self.rules.push(rule);
    }

    /// Returns the category registry.
    #[must_use]
    pub fn categories(&self) -> &CategorySet {
        &self.categories
    }

    /// Returns the declarative rules in authoring order.
    #[must_use]
    pub fn rules(&self) -> &[DuckingRule] {
        &self.rules
    }

    /// Validates every rule against the category registry.
    ///
    /// Checks that each rule's trigger and target are registered and that no
    /// rule ducks a category against itself. Returns the first error found in
    /// authoring order, or `Ok(())` when the ruleset is well formed.
    ///
    /// # Errors
    ///
    /// Returns [`AutoMixError`] describing the first malformed rule.
    pub fn validate(&self) -> Result<(), AutoMixError> {
        for (rule_index, rule) in self.rules.iter().enumerate() {
            if !self.categories.contains(rule.trigger) {
                return Err(AutoMixError::UnknownTrigger {
                    rule_index,
                    category: rule.trigger,
                });
            }
            if !self.categories.contains(rule.target) {
                return Err(AutoMixError::UnknownTarget {
                    rule_index,
                    category: rule.target,
                });
            }
            if rule.trigger == rule.target {
                return Err(AutoMixError::SelfDuck {
                    rule_index,
                    category: rule.trigger,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_formed_ruleset_validates() {
        let mut set = AutoMixRuleset::new();
        let dialogue = set.add_category("dialogue", 100);
        let music = set.add_category("music", 10);
        set.add_rule(DuckingRule::new(dialogue, music, 9.0));
        assert_eq!(set.validate(), Ok(()));
        assert_eq!(set.rules().len(), 1);
        assert_eq!(set.categories().len(), 2);
    }

    #[test]
    fn self_duck_is_rejected() {
        let mut set = AutoMixRuleset::new();
        let a = set.add_category("a", 1);
        set.add_rule(DuckingRule::new(a, a, 6.0));
        assert_eq!(
            set.validate(),
            Err(AutoMixError::SelfDuck {
                rule_index: 0,
                category: a,
            })
        );
    }

    #[test]
    fn unknown_target_is_rejected() {
        let mut set = AutoMixRuleset::new();
        let a = set.add_category("a", 1);
        set.add_rule(DuckingRule::new(a, CategoryId(9), 6.0));
        assert_eq!(
            set.validate(),
            Err(AutoMixError::UnknownTarget {
                rule_index: 0,
                category: CategoryId(9),
            })
        );
    }

    #[test]
    fn unknown_trigger_is_rejected() {
        let mut set = AutoMixRuleset::new();
        let a = set.add_category("a", 1);
        set.add_rule(DuckingRule::new(CategoryId(7), a, 6.0));
        assert_eq!(
            set.validate(),
            Err(AutoMixError::UnknownTrigger {
                rule_index: 0,
                category: CategoryId(7),
            })
        );
    }
}
