//! Fixed-point determinism kernel (milestone M4).
//!
//! This module is Prism's **bit-exact numerical substrate**: a complete signed
//! fixed-point number set plus integer-only transcendental functions, designed
//! so that the same inputs produce byte-identical results on every CPU and OS.
//! It is the numerical foundation of "four-way determinism" (ECS / tasks /
//! time / transform) for lockstep netcode, rollback replay, and
//! server-authoritative simulation (design doc §6/§13/§22-M4).
//!
//! ## Contents
//! - [`Fixed`] — the primary Q32.32 scalar (`i64`-backed) with a full operator
//!   set, saturating/wrapping/checked arithmetic, and exact raw-bit I/O.
//! - [`I16F16`] — a lightweight Q16.16 scalar (`i32`-backed) for compact angles
//!   and parameters, widening losslessly to [`Fixed`].
//! - Deterministic transcendentals on [`Fixed`]: [`Fixed::sqrt`],
//!   [`Fixed::sin`]/[`Fixed::cos`]/[`Fixed::sin_cos`]/[`Fixed::tan`],
//!   [`Fixed::atan2`], [`Fixed::exp`], and [`Fixed::ln`] — all pure integer
//!   math (table-free range reduction + polynomial / digit-by-digit roots).
//! - [`FxVec2`]/[`FxVec3`]/[`FxVec4`] — fixed-point vectors.
//! - [`StateHasher`] — a deterministic FNV-1a digest of fixed-point state for
//!   cross-peer desync detection.
//! - [`KahanSum`]/[`NeumaierSum`] and [`kahan_sum`]/[`neumaier_sum`] —
//!   compensated `f32`/`f64` reduction (design doc §24.4), the low-drift
//!   single-platform companion to the bit-exact fixed path.
//!
//! ## Determinism contract
//! Nothing on the [`Fixed`] arithmetic or transcendental path uses hardware
//! float. Float↔fixed conversions ([`Fixed::from_f64`] etc.) exist only for
//! authoring/debug and are explicitly excluded from the deterministic
//! guarantee. Construct deterministic values with [`Fixed::from_bits`] /
//! [`Fixed::from_int`].

pub mod compensated;
pub mod double_double;
pub mod hash;
pub mod i16f16;
pub mod scalar;
pub mod transcendental;
pub mod trig;
pub mod vec;

pub use compensated::{CompensableFloat, KahanSum, NeumaierSum, kahan_sum, neumaier_sum};
pub use double_double::DoubleDouble;
pub use hash::StateHasher;
pub use i16f16::I16F16;
pub use scalar::Fixed;
pub use vec::{FxVec2, FxVec3, FxVec4, fxvec2, fxvec3, fxvec4};
