//! A deterministic dialogue decision tree over State/Switch values.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the dialogue decision tree of design section 35: a set of
//! State/Switch values (who is speaking, where, what mood) walks a tree to a
//! concrete line or a random variant, with a generic fallback branch so a
//! missing variant never produces silence. Variant selection uses a
//! deterministic `xorshift64` generator seeded through `SplitMix64` so the same
//! state and seed always pick the same take. Evaluation is non-real-time.

#[cfg(not(feature = "std"))]
use alloc::string::{String, ToString};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use alloc::collections::BTreeMap;

use crate::dialogue::resolver::MediaRef;

/// A snapshot of State/Switch values used to walk the tree.
///
/// Keys name a dimension (for example `who`, `area`, `mood`) and values name
/// the current setting (for example `hero`, `cave`, `angry`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DialogueState {
    values: BTreeMap<String, String>,
}

impl DialogueState {
    /// Creates an empty state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    /// Sets `key` to `value` and returns the state for chaining.
    #[must_use]
    pub fn with(mut self, key: &str, value: &str) -> Self {
        self.values.insert(key.to_string(), value.to_string());
        self
    }

    /// Sets `key` to `value`.
    pub fn set(&mut self, key: &str, value: &str) {
        self.values.insert(key.to_string(), value.to_string());
    }

    /// Returns the value bound to `key`, if any.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
}

/// An index into a [`DialogueDecisionTree`]'s node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeIndex(pub u32);

/// A single node in the decision tree.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DecisionNode {
    /// Branch on the value of `key`.
    Branch {
        /// The State/Switch dimension this node inspects.
        key: String,
        /// Matching value arms, in declaration order.
        arms: Vec<(String, NodeIndex)>,
        /// The generic fallback taken when no arm matches (and the key is
        /// absent). Avoids silence on missing variants.
        fallback: Option<NodeIndex>,
    },
    /// A leaf holding one or more interchangeable takes.
    Leaf {
        /// Candidate media; one is chosen deterministically at evaluation.
        variants: Vec<MediaRef>,
    },
}

/// Errors raised while building or validating a tree.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum TreeError {
    /// A node referenced an index outside the arena.
    DanglingIndex(NodeIndex),
    /// The tree has no nodes, so there is no root.
    Empty,
}

/// A deterministic `xorshift64` generator seeded through `SplitMix64`.
///
/// Reused only as a small, well-known uniform integer source for picking a
/// variant; it contains no third-party code.
struct VariantRng {
    state: u64,
}

impl VariantRng {
    #[must_use]
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let state = if z == 0 { 0x9E37_79B9_7F4A_7C15 } else { z };
        Self { state }
    }

    #[must_use]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a uniform index in `[0, bound)` for a non-zero `bound`.
    #[must_use]
    fn index(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// A dialogue decision tree stored as a flat node arena.
///
/// The arena layout keeps the tree `no_std`-friendly and serialisable without
/// recursive pointer types. Node `0` is the root.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DialogueDecisionTree {
    nodes: Vec<DecisionNode>,
}

impl DialogueDecisionTree {
    /// Creates an empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Appends a node and returns its index.
    pub fn push(&mut self, node: DecisionNode) -> NodeIndex {
        let index = NodeIndex(self.nodes.len() as u32);
        self.nodes.push(node);
        index
    }

    /// Returns the number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when the tree has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Validates that every referenced index is in range and a root exists.
    pub fn validate(&self) -> Result<(), TreeError> {
        if self.nodes.is_empty() {
            return Err(TreeError::Empty);
        }
        let count = self.nodes.len() as u32;
        for node in &self.nodes {
            if let DecisionNode::Branch { arms, fallback, .. } = node {
                for (_, index) in arms {
                    if index.0 >= count {
                        return Err(TreeError::DanglingIndex(*index));
                    }
                }
                if let Some(index) = fallback
                    && index.0 >= count
                {
                    return Err(TreeError::DanglingIndex(*index));
                }
            }
        }
        Ok(())
    }

    /// Evaluates the tree against `state`, choosing a variant with `seed`.
    ///
    /// Returns the selected [`MediaRef`], or `None` when the walk dead-ends at a
    /// branch with no matching arm and no fallback, or at an empty leaf. The
    /// `seed` makes variant selection reproducible.
    #[must_use]
    pub fn evaluate(&self, state: &DialogueState, seed: u64) -> Option<MediaRef> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut current = NodeIndex(0);
        // Bound the walk by the node count to guarantee termination even if a
        // hand-built arena contains a cycle.
        for _ in 0..=self.nodes.len() {
            match self.nodes.get(current.0 as usize)? {
                DecisionNode::Branch {
                    key,
                    arms,
                    fallback,
                } => {
                    let next = match state.get(key) {
                        Some(value) => arms
                            .iter()
                            .find(|(arm_value, _)| arm_value == value)
                            .map(|(_, index)| *index)
                            .or(*fallback),
                        None => *fallback,
                    };
                    current = next?;
                }
                DecisionNode::Leaf { variants } => {
                    if variants.is_empty() {
                        return None;
                    }
                    if variants.len() == 1 {
                        return Some(variants[0]);
                    }
                    let mut rng = VariantRng::new(seed);
                    return Some(variants[rng.index(variants.len())]);
                }
            }
        }
        None
    }
}
