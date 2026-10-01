//! Summaries of a [`BackendOp`] stream.
//!
//! [`OpTrace`] tallies a slice of [`BackendOp`]s by variant, giving a cheap,
//! deterministic view of what a flush did without re-walking the retained tree.
//! It is handy for asserting the central Loom contract — *cost ∝ change* — in
//! tests and for human-readable debug output via [`OpTrace::summary`].

use alloc::string::String;
use core::fmt::Write as _;

use prism_ui::backend::BackendOp;

/// Per-variant counts for a slice of [`BackendOp`]s.
///
/// Build one with [`OpTrace::from_ops`]. Each accessor returns the number of
/// ops of the corresponding [`BackendOp`] variant; [`OpTrace::total`] returns
/// the grand total.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpTrace {
    creates: usize,
    removes: usize,
    set_texts: usize,
    set_layouts: usize,
    set_paints: usize,
    reorders: usize,
}

impl OpTrace {
    /// Tallies `ops` by [`BackendOp`] variant.
    #[must_use]
    pub fn from_ops(ops: &[BackendOp]) -> Self {
        let mut trace = Self::default();
        for op in ops {
            match op {
                BackendOp::Create { .. } => trace.creates += 1,
                BackendOp::Remove { .. } => trace.removes += 1,
                BackendOp::SetText { .. } => trace.set_texts += 1,
                BackendOp::SetLayout { .. } => trace.set_layouts += 1,
                BackendOp::SetPaint { .. } => trace.set_paints += 1,
                BackendOp::Reorder { .. } => trace.reorders += 1,
            }
        }
        trace
    }

    /// Number of [`BackendOp::Create`] ops.
    #[must_use]
    pub fn creates(&self) -> usize {
        self.creates
    }

    /// Number of [`BackendOp::Remove`] ops.
    #[must_use]
    pub fn removes(&self) -> usize {
        self.removes
    }

    /// Number of [`BackendOp::SetText`] ops.
    #[must_use]
    pub fn set_texts(&self) -> usize {
        self.set_texts
    }

    /// Number of [`BackendOp::SetLayout`] ops.
    #[must_use]
    pub fn set_layouts(&self) -> usize {
        self.set_layouts
    }

    /// Number of [`BackendOp::SetPaint`] ops.
    #[must_use]
    pub fn set_paints(&self) -> usize {
        self.set_paints
    }

    /// Number of [`BackendOp::Reorder`] ops.
    #[must_use]
    pub fn reorders(&self) -> usize {
        self.reorders
    }

    /// Grand total of all recorded ops.
    #[must_use]
    pub fn total(&self) -> usize {
        self.creates
            + self.removes
            + self.set_texts
            + self.set_layouts
            + self.set_paints
            + self.reorders
    }

    /// Renders a deterministic, human-readable one-line summary.
    ///
    /// The format is stable and suitable for golden tests, e.g.
    /// `3 ops (create=1, remove=0, set_text=1, set_layout=1, set_paint=0, reorder=0)`.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut out = String::new();
        let _ = write!(
            out,
            "{} ops (create={}, remove={}, set_text={}, set_layout={}, set_paint={}, reorder={})",
            self.total(),
            self.creates,
            self.removes,
            self.set_texts,
            self.set_layouts,
            self.set_paints,
            self.reorders,
        );
        out
    }
}
