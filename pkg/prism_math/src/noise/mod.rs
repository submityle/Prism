//! Procedural noise (M5): classic Perlin and Simplex gradient noise with
//! fractal (fBm / turbulence / ridged) helpers.
//!
//! Every generator is **deterministic given its seed** — the permutation table
//! is built from the seed via [`crate::rng::Xoshiro256StarStar`], so the same
//! seed yields the same field on every platform. These are purely classical
//! procedural algorithms (no ML), intended for terrain, particles, procedural
//! textures, and camera shake.
//!
//! Both [`Perlin`] and [`Simplex`] implement the [`Noise2`]/[`Noise3`] traits,
//! so the [`Fractal`] octave helpers work with either source.

mod fbm;
mod perlin;
mod perm;
mod simplex;

pub use fbm::{Fractal, Noise2, Noise3};
pub use perlin::Perlin;
pub use simplex::Simplex;
