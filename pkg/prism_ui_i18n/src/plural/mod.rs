//! `CLDR`-conformant plural category selection.
//!
//! This module implements Unicode `CLDR` plural selection on top of the integer
//! [`PluralOperands`] model, so every decision stays in pure integer arithmetic
//! — no floating-point or transcendental math — preserving determinism.
//!
//! Two rule *types* are supported:
//! * **cardinal** — "1 file", "2 files" ([`CardinalFamily`]).
//! * **ordinal** — "1st", "2nd", "3rd" ([`OrdinalFamily`]).
//!
//! The high-level entry point is [`PluralRules`], which pairs a resolved rule
//! family with a selection method. Resolve rules for a locale with
//! [`PluralRules::cardinal`] / [`PluralRules::ordinal`], then select a category
//! from an integer ([`PluralRules::select`]), a signed integer
//! ([`PluralRules::select_i64`]), explicit operands
//! ([`PluralRules::select_operands`]), or an exact decimal string
//! ([`PluralRules::select_decimal`]).
//!
//! The legacy unit variants [`PluralRules::English`], [`PluralRules::Slavic`],
//! [`PluralRules::Asian`], and [`PluralRules::OtherOnly`] remain for source
//! compatibility and map onto the corresponding `CLDR` families.

mod cardinal;
mod operands;
mod ordinal;
pub mod registry;

#[cfg(all(test, feature = "std"))]
mod tests;

pub use cardinal::CardinalFamily;
pub use operands::PluralOperands;
pub use ordinal::OrdinalFamily;

/// The six `CLDR` plural categories.
///
/// A category selects which message variant to render; the full set is defined
/// by `CLDR`, and any individual language uses only a subset (English uses just
/// `one`/`other`, Arabic uses all six).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PluralCategory {
    /// The `zero` category (e.g. Arabic, Welsh, Latvian).
    Zero,
    /// The `one` (singular) category.
    One,
    /// The `two` (dual) category.
    Two,
    /// The `few` (paucal) category.
    Few,
    /// The `many` category.
    Many,
    /// The `other` (general plural / fallback) category.
    Other,
}

/// Whether a rule selects cardinal ("2 files") or ordinal ("2nd") categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PluralType {
    /// Cardinal counting rules.
    Cardinal,
    /// Ordinal position rules.
    Ordinal,
}

/// A resolved plural rule: a `CLDR` rule family plus the selection it performs.
///
/// Construct from a locale with [`PluralRules::cardinal`] /
/// [`PluralRules::ordinal`], or use one of the legacy unit variants. Select a
/// category with [`select`](PluralRules::select) and friends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluralRules {
    /// Legacy alias for the Germanic cardinal family (`one: i = 1 and v = 0`).
    English,
    /// Legacy alias for the East-Slavic cardinal family (one/few/many).
    Slavic,
    /// Legacy alias for the other-only cardinal family (East-Asian style).
    Asian,
    /// Legacy alias for the degenerate other-only cardinal family.
    OtherOnly,
    /// An explicit cardinal rule family.
    Cardinal(CardinalFamily),
    /// An explicit ordinal rule family.
    Ordinal(OrdinalFamily),
}

impl PluralRules {
    /// Resolve the cardinal rules for `locale` (e.g. `"en"`, `"pt-BR"`, `"ru"`).
    ///
    /// Unmapped languages resolve to the `CLDR` root (other-only) rule.
    pub fn cardinal(locale: &str) -> Self {
        PluralRules::Cardinal(registry::cardinal_family(locale))
    }

    /// Resolve the ordinal rules for `locale`.
    ///
    /// Unmapped languages resolve to the `CLDR` root (other-only) rule.
    pub fn ordinal(locale: &str) -> Self {
        PluralRules::Ordinal(registry::ordinal_family(locale))
    }

    /// Whether these rules are cardinal or ordinal.
    pub fn plural_type(&self) -> PluralType {
        match self {
            PluralRules::Ordinal(_) => PluralType::Ordinal,
            _ => PluralType::Cardinal,
        }
    }

    /// Select the plural category for a non-negative integer count.
    pub fn select(&self, n: u64) -> PluralCategory {
        self.select_operands(&PluralOperands::from_u64(n))
    }

    /// Select the plural category for a signed integer count (uses `|n|`).
    pub fn select_i64(&self, n: i64) -> PluralCategory {
        self.select_operands(&PluralOperands::from_i64(n))
    }

    /// Select the plural category from an exact decimal string such as
    /// `"1.50"`. Returns [`PluralCategory::Other`] if the string is malformed.
    pub fn select_decimal(&self, s: &str) -> PluralCategory {
        match PluralOperands::parse(s) {
            Some(ops) => self.select_operands(&ops),
            None => PluralCategory::Other,
        }
    }

    /// Select the plural category from fully specified [`PluralOperands`].
    pub fn select_operands(&self, ops: &PluralOperands) -> PluralCategory {
        match self {
            PluralRules::English => CardinalFamily::Germanic.select(ops),
            PluralRules::Slavic => CardinalFamily::EastSlavic.select(ops),
            PluralRules::Asian | PluralRules::OtherOnly => {
                CardinalFamily::OtherOnly.select(ops)
            }
            PluralRules::Cardinal(family) => family.select(ops),
            PluralRules::Ordinal(family) => family.select(ops),
        }
    }
}
