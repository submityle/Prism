//! World-space ReSTIR: spatial-hash reservoirs for persistent, view-independent
//! global-illumination reuse (SHARC-style hashed radiance cache + GRIS merge).
//!
//! # Submodules
//! * [`spatial_hash`] — SHARC-style spatial hash keys: level-scaled, jittered
//!   world-position + normal quantisation into `cell_coord`/`hash_key`/`checksum`.
//! * [`world_reservoir`] — per-cell streaming RIS fill, an open-addressed
//!   reservoir table, and cross-cell GRIS spatial reuse built on
//!   [`crate::gi::screen_probe::restir`].
//!
//! # Conventions
//! * Reservoirs reuse [`crate::gi::screen_probe::restir::Reservoir`]; hash keys
//!   quantise world position + normal into a sparse grid (jittered, level-scaled).
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe);
//!   randomness, if any, is supplied by the caller.

pub mod spatial_hash;
pub mod world_reservoir;
