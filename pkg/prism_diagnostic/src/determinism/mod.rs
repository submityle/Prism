//! §24.4 deterministic replay & trace comparison (anti-desync).
//!
//! Rollback networking, recordings, and deterministic simulation share one
//! brutal failure mode: two runs that *should* be identical diverge. This
//! module is the diagnostic-layer tooling to catch and localize that desync. It
//! delivers the three §24.4 pieces as pure `core`/`alloc` integer arithmetic
//! (deterministic, `no_std` + `alloc`, no `unsafe`), always compiled regardless
//! of crate features:
//!
//! 1. **Deterministic trace** ([`hash`] + [`trace`]): a stable 64-bit
//!    [`StateHasher`] (`FNV`-1a, fixed-order fold, no per-process seed) hashes
//!    each frame's key state, and [`DeterminismTrace`] records one digest per
//!    frame.
//! 2. **Input + seed recording** ([`record`]): [`InputRecorder`] records the
//!    per-frame input digest + random seed, and [`InputReplay`] feeds the
//!    identical sequence back for precise reproduction of an intermittent bug.
//! 3. **Double-run comparison** ([`compare`]): walk two traces frame-by-frame
//!    and report the first divergence as [`TraceDiff::Identical`],
//!    [`TraceDiff::Diverged`] (the desync root cause), or
//!    [`TraceDiff::LengthMismatch`].
//!
//! This intentionally mirrors the auditing concept in `prism_time`'s
//! `multiworld` layer (same `FNV`-1a constants and the same
//! identical/diverged/length-mismatch comparison shape) **without** a
//! dependency edge: `prism_time`'s audit hashes simulation-time state, whereas
//! this is the general-purpose trace primitive any subsystem can feed. The
//! existing [`crate::replay`] markers are the complementary *labeled-marker*
//! stream; this module is the *per-frame state-hash* stream.

pub mod hash;
pub mod record;
pub mod trace;

pub use hash::{fnv1a_64, StateHasher};
pub use record::{FrameInput, InputRecorder, InputReplay};
pub use trace::{compare, DeterminismTrace, FrameHash, TraceDiff};
