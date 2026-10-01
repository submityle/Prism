//! Path-space ReSTIR (ReSTIR PT): multi-bounce path reuse via reconnection
//! shift mapping with the associated Jacobian determinant and GRIS resampling.
//!
//! # Conventions
//! * Reuses [`crate::gi::screen_probe::restir::Reservoir`]; a reservoir sample
//!   is a reconnectable path suffix anchored at a reconnection vertex.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe);
//!   randomness, if any, is supplied by the caller as `u in [0, 1)`.
//!
//! * [`vertex`] — path vertices ([`vertex::PathVertex`]) and reconnectable path
//!   suffixes ([`vertex::PathSuffix`]) that pack losslessly into the shared
//!   [`crate::gi::screen_probe::restir::GiSample`] payload.
//! * [`shift_map`] — the reconnection shift map and its geometric Jacobian
//!   `|J| = (cos_dst/cos_src)·(dist_src²/dist_dst²)` with grazing-angle and
//!   determinant clamps (`J(a→b)·J(b→a)=1`, same-pixel `J=1`).
//! * [`path_reservoir`] — reconnection-shift GRIS resampling: streaming RIS
//!   fill, finalize, and Jacobian-weighted neighbour reuse `m·p̂·W·|J|`.

pub mod vertex;
pub mod shift_map;
pub mod path_reservoir;
