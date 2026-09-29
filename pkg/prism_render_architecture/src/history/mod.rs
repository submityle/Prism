//! Shared temporal-history identity, epochs, and invalidation.
//!
//! Modern renderers amortize expensive effects across frames by reusing the
//! previous frame's result: temporal anti-aliasing (`TAA`), temporal upscalers,
//! screen-space denoisers, and reprojected shadows or reflections all keep a
//! *history* buffer. That reuse is only valid while the reprojection
//! assumptions hold, so the architecture needs a single, `CPU`-verifiable
//! contract for *which* views own history and *when* that history stops being
//! trustworthy. This module is that contract; the `GPU`-side history textures
//! and the reprojection kernels themselves are out of scope and pending the
//! `GPU` backend.
//!
//! The contract has three cooperating pieces:
//!
//! 1. **Invalidation** ([`InvalidationMask`] / [`InvalidationReason`]) — a
//!    compact bitset of the reasons history reuse can break, from a hard camera
//!    cut down to a category-scoped lighting change; see [`invalidation`].
//! 2. **Epochs** ([`HistoryEpochs`] / [`EpochCategory`]) — monotonic version
//!    counters per subsystem whose diff against a consumer's cached snapshot
//!    yields exactly the categories that moved; see [`epochs`].
//! 3. **Registry** ([`ViewHistoryRegistry`]) — a generational table that owns
//!    each view's [`ViewHistoryId`], its cached epochs, and its pending
//!    invalidation, and resolves the effective per-frame invalidation; see
//!    [`registry`].
//!
//! Every operation is deterministic integer work: bit manipulation on the
//! masks, wrapping counter comparisons on the epochs, and free-list slot
//! recycling with generation bumps in the registry. No floating-point or
//! transcendental math is involved.

pub mod epochs;
pub mod invalidation;
pub mod registry;

pub use epochs::{EpochCategory, HistoryEpochs};
pub use invalidation::{InvalidationMask, InvalidationReason};
pub use registry::{ResolvedHistory, ViewHistoryId, ViewHistoryRegistry, ViewHistoryState};
