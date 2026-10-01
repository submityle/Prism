//! Height-field parallax mapping (CPU golden reference).
//!
//! # Conventions
//! * Pure, deterministic functions: no RNG, IO, GPU, or `unsafe`.
//! * Transcendental functions via [`bevy_math::ops`]; never `f32::exp()` style.
//! * Storage layouts mirror the GPU twin (f32 fields, explicit alignment).
//! * Defensive clamping everywhere: guard divide-by-zero, enforce
//!   non-negativity and finiteness, and fall back gracefully on degenerate
//!   input so no `NaN`/`inf` ever escapes.
//! * A height field is a closure `height_at(uv) -> f32` returning the surface
//!   height in `[0, 1]`: `1` on the polygon plane / ridge, `0` at the deepest
//!   valley.  The march works in depth `= 1 - height`, increasing downward from
//!   the plane.  Tangent-space directions use `+z` along the geometric normal.
//!
//! Implements parallax occlusion mapping, relief mapping via binary search, and self-shadowing of height fields.

pub mod pom;
pub mod relief;
pub mod self_shadow;

pub use pom::{parallax_occlusion, PomConfig, PomSample};
pub use relief::{relief_map, ReliefConfig, ReliefSample};
pub use self_shadow::{self_shadow, SelfShadowConfig};
