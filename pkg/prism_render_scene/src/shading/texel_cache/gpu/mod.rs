//! GPU-facing contracts for the object/texture-space shade cache.
//!
//! This module intentionally owns a page-table ABI separate from the
//! material virtual-texture ABI.  The shade writer validates generation and
//! epoch proofs before touching its atlas, so a stale material or an atlas
//! slot from another cache cannot become sampleable by accident.

pub mod page_table;
pub mod shade;

#[cfg(test)]
mod shade_tests;

#[cfg(test)]
mod material_vt_tests;
