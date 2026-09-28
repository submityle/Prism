//! Offline physics caching: bake once, replay for free (milestone M4.5).
//!
//! This module implements Prism's **offline mode** for deformable simulation.
//! A soft body is expensive to solve every frame, but many uses (cutscenes,
//! background props, deterministic replays) do not need live simulation, only
//! its recorded result. The cache pipeline pays the solver cost once at author
//! time and stores a compact, deterministic trajectory that can be played back
//! with zero solver work.
//!
//! # Pipeline
//!
//! - [`PositionQuantizer`] maps continuous positions onto a fixed integer grid
//!   for compact, exactly reproducible storage.
//! - [`CacheTrack`] stores one body's history as periodic keyframes plus sparse
//!   per-frame deltas.
//! - [`PhysicsCache`] bundles the frame step, quantizer, and per-body tracks
//!   into a shippable offline asset.
//! - [`Baker`] runs the M4 soft solver forward and records a [`PhysicsCache`].
//! - [`Player`] samples a cache at a running playback time with linear
//!   interpolation and no solving.
//! - [`trajectory_hash`] produces a deterministic [`GoldenDigest`] for
//!   golden-replay regression tests.
//!
//! # Provenance
//!
//! This module and its submodules contain **no Unreal Engine source or derived code**.
//! Quantization, keyframe/delta compression, fixed-step baking, keyframe
//! interpolation, and FNV-1a hashing are all standard, publicly documented
//! techniques implemented from first principles.

pub mod asset;
pub mod bake;
pub mod golden;
pub mod playback;
pub mod quantize;
pub mod track;

pub use asset::PhysicsCache;
pub use bake::{BakeConfig, Baker};
pub use golden::{trajectory_hash, verify, GoldenDigest};
pub use playback::{PlaybackConfig, Player};
pub use quantize::{PositionQuantizer, DEFAULT_STEP};
pub use track::{CacheTrack, FrameRecord};
