//! Static binding slots and deterministic dependency scheduling for ST2.
//!
//! This module deliberately does not know how a host stores values. A compiler or
//! ECS adapter resolves a [`BindingPath`] once, assigns a stable [`BindingSlot`],
//! and reports changed slots to [`DependencyPlan`] before evaluator execution.

use alloc::{string::String, vec, vec::Vec};
use core::fmt;

/// A validated, canonical field path used by a binding slot.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BindingPath(String);

/// A path validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BindingPathError {
    /// The path has no segments.
    Empty,
    /// A segment is empty or contains an unsupported character.
    InvalidSegment {
        /// The zero-based segment index that failed validation.
        index: usize,
    },
}

impl fmt::Display for BindingPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("binding path is empty"),
            Self::InvalidSegment { index } => write!(f, "invalid binding path segment {index}"),
        }
    }
}

impl BindingPath {
    /// Parses a dotted path and rejects ambiguous whitespace or punctuation.
    pub fn parse(input: &str) -> Result<Self, BindingPathError> {
        if input.is_empty() {
            return Err(BindingPathError::Empty);
        }
        let mut canonical = String::new();
        for (index, segment) in input.split('.').enumerate() {
            if segment.is_empty()
                || !segment.bytes().enumerate().all(|(position, byte)| {
                    (position == 0 && (byte.is_ascii_alphabetic() || byte == b'_'))
                        || (position > 0 && (byte.is_ascii_alphanumeric() || byte == b'_'))
                })
            {
                return Err(BindingPathError::InvalidSegment { index });
            }
            if index != 0 {
                canonical.push('.');
            }
            canonical.push_str(segment);
        }
        Ok(Self(canonical))
    }

    /// Returns the canonical dotted representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Stable index into an instance's compiled binding frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct BindingSlot(pub u32);

/// One evaluator's declared input dependencies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluatorDependency {
    /// Stable evaluator identifier, used for deterministic ordering.
    pub evaluator: u32,
    /// Slots whose changes make this evaluator eligible.
    pub inputs: Vec<BindingSlot>,
}

/// Deterministic reverse dependency index for dirty evaluator execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyPlan {
    evaluators: Vec<EvaluatorDependency>,
    reverse: Vec<(BindingSlot, Vec<u32>)>,
}

impl DependencyPlan {
    /// Builds a dependency index. Duplicate evaluator ids or input slots are rejected.
    pub fn compile(mut evaluators: Vec<EvaluatorDependency>) -> Result<Self, DependencyError> {
        evaluators.sort_by_key(|entry| entry.evaluator);
        for pair in evaluators.windows(2) {
            if pair[0].evaluator == pair[1].evaluator {
                return Err(DependencyError::DuplicateEvaluator(pair[0].evaluator));
            }
        }
        let mut edges = Vec::new();
        for entry in &mut evaluators {
            entry.inputs.sort_unstable();
            if entry.inputs.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(DependencyError::DuplicateInput(entry.evaluator));
            }
            for &slot in &entry.inputs {
                edges.push((slot, entry.evaluator));
            }
        }
        edges.sort_unstable();
        let mut reverse = Vec::<(BindingSlot, Vec<u32>)>::new();
        for (slot, evaluator) in edges {
            if let Some((last_slot, ids)) = reverse.last_mut()
                && *last_slot == slot
            {
                ids.push(evaluator);
                continue;
            }
            reverse.push((slot, vec![evaluator]));
        }
        Ok(Self {
            evaluators,
            reverse,
        })
    }

    /// Returns evaluator ids affected by changed slots, sorted and deduplicated.
    #[must_use]
    pub fn dirty_evaluators(&self, changed: &[BindingSlot]) -> Vec<u32> {
        let mut dirty = Vec::new();
        // The convenience API allocates; use `dirty_evaluators_into` in a hot loop.
        for &slot in changed {
            if let Ok(index) = self.reverse.binary_search_by_key(&slot, |(key, _)| *key) {
                let ids = &self.reverse[index].1;
                dirty.extend_from_slice(ids);
            }
        }
        dirty.sort_unstable();
        dirty.dedup();
        dirty
    }

    /// Writes sorted, unique dirty evaluator ids into a preallocated buffer.
    ///
    /// No allocation occurs, including on capacity error. On error the buffer
    /// is empty; on success its previous content is replaced. Its capacity must
    /// accommodate all matching edges (including duplicate changed slots).
    pub fn dirty_evaluators_into(
        &self,
        changed: &[BindingSlot],
        output: &mut Vec<u32>,
    ) -> Result<(), DependencyError> {
        output.clear();
        for &slot in changed {
            if let Ok(index) = self.reverse.binary_search_by_key(&slot, |(key, _)| *key) {
                let ids = &self.reverse[index].1;
                if ids.len() > output.capacity() - output.len() {
                    output.clear();
                    return Err(DependencyError::OutputCapacity);
                }
                output.extend_from_slice(ids);
            }
        }
        output.sort_unstable();
        output.dedup();
        Ok(())
    }

    /// Returns the compiled evaluator declarations in stable id order.
    #[must_use]
    pub fn evaluators(&self) -> &[EvaluatorDependency] {
        &self.evaluators
    }
}

/// Dependency graph compilation errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DependencyError {
    /// Two declarations use the same evaluator id.
    DuplicateEvaluator(u32),
    /// One evaluator lists the same input slot twice.
    DuplicateInput(u32),
    /// The supplied result buffer cannot hold all affected edges.
    OutputCapacity,
}

impl fmt::Display for DependencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateEvaluator(id) => write!(f, "duplicate evaluator {id}"),
            Self::DuplicateInput(id) => write!(f, "evaluator {id} repeats an input slot"),
            Self::OutputCapacity => f.write_str("dirty evaluator result buffer is full"),
        }
    }
}
