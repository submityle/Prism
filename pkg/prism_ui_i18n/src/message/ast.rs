//! Abstract syntax tree for the `ICU`-style message format.
//!
//! A compiled message is a flat list of [`Node`]s. Argument nodes (`select` /
//! `plural` / `selectordinal`) carry nested sub-messages (their arms), each of
//! which is itself a `Vec<Node>`, giving the recursive structure `ICU`
//! `MessageFormat` requires. The AST is produced by
//! [`parser`](super::parser) and consumed by [`eval`](super::eval); it holds no
//! runtime values and performs no formatting itself.

use alloc::string::String;
use alloc::vec::Vec;

use crate::plural::PluralCategory;

/// One element of a (sub-)message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Node {
    /// Literal text (brace escapes already resolved).
    Text(String),
    /// A `#` token. Rendered as the active plural value minus its offset when
    /// evaluated inside a `plural`/`selectordinal` arm, otherwise literal `#`.
    Pound,
    /// A bare `{name}` substitution.
    Arg(String),
    /// A `{name, select, key{..} other{..}}` string switch.
    Select {
        /// The argument whose string value selects an arm.
        name: String,
        /// The arms; the `other` arm is stored with key `"other"`.
        arms: Vec<SelectArm>,
    },
    /// A `{name, plural, ...}` or `{name, selectordinal, ...}` numeric switch.
    Plural {
        /// The argument whose integer value selects an arm.
        name: String,
        /// `true` for `selectordinal` (uses ordinal rules), `false` for
        /// cardinal `plural`.
        ordinal: bool,
        /// The `offset:` value subtracted before keyword selection and `#`.
        offset: i64,
        /// The arms; an `other` arm (as [`PluralSelector::Category`] with
        /// [`PluralCategory::Other`]) is always present after a successful
        /// parse.
        arms: Vec<PluralArm>,
    },
}

/// One arm of a `select` switch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SelectArm {
    /// The literal key this arm matches, or `"other"` for the fallback.
    pub key: String,
    /// The arm's sub-message.
    pub body: Vec<Node>,
}

/// One arm of a `plural`/`selectordinal` switch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PluralArm {
    /// What this arm matches.
    pub selector: PluralSelector,
    /// The arm's sub-message.
    pub body: Vec<Node>,
}

/// A `plural`/`selectordinal` arm selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PluralSelector {
    /// An exact `=N` literal match against the original argument value.
    Exact(i64),
    /// A `CLDR` keyword match (`zero`/`one`/`two`/`few`/`many`/`other`).
    Category(PluralCategory),
}
