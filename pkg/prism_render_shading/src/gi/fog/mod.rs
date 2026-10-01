//! Analytic atmospheric fog (CPU golden reference).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; never `f32::exp()` style.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Defensive clamping everywhere: guard divide-by-zero, enforce
//!   non-negativity and finiteness, and fall back gracefully on degenerate
//!   input so no `NaN`/`inf` ever escapes.
//!
//! Implements closed-form exponential height fog, distance-based fog, and
//! sun inscattering with a Henyey-Greenstein phase function.


pub mod exponential;
pub mod height_fog;
pub mod inscatter;

pub use exponential::{
    apply_fog_color, exponential_fog_factor, exponential_squared_fog_factor, linear_fog_factor,
    DistanceFog, DistanceFogMode,
};
pub use height_fog::{FogIntegral, HeightFog};
pub use inscatter::{AtmosphericFog, Inscatter};
