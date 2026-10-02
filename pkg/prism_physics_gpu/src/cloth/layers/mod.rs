//! `GPU` multi-layer garment coupling for cloth particles, the faithful twin of
//! [`prism_physics_core`]'s
//! [`resolve_layer_coupling_jacobi`](prism_physics_core::resolve_layer_coupling_jacobi).
//!
//! A dressed character stacks garments — shirt under jacket, lining under skirt
//! — each simulating its own cloth with its own *layer number*. The inter-layer
//! coupling tier keeps a higher-numbered (outer) layer on the outward side of
//! the lower-numbered (inner) one, at least `thickness` apart, so garments stay
//! stacked instead of sinking through each other. Intra-layer contacts are the
//! job of the self-collision tiers ([`super::self_collision_point`] and
//! [`super`]); this tier resolves only cross-layer pairs.
//!
//! The Gauss-Seidel core cannot map to a compute kernel because every
//! invocation must read the *same* frozen snapshot, so this module runs the
//! Jacobi reformulation the hardware actually executes.
//!
//! # Layout
//!
//! - [`prep`] — the host-side deterministic broad phase (uniform spatial hash,
//!   27-cell neighbor enumeration, same-layer filtering, and the per-particle
//!   `CSR` adjacency) laid out in the golden's exact reduction order.
//! - [`cpu`] — the [`cpu_cloth_layer_coupling`] golden twin, delegating to
//!   [`prism_physics_core`].
//! - [`gpu`] — the real-device [`GpuClothLayerCoupling`] single-pass pipeline.
//!
//! # Correctness model
//!
//! The cross-layer adjacency is built on the host (integer-exact), so the only
//! floating-point work on the `GPU` is the separating-push arithmetic. The
//! single own-slot pass reads only a frozen snapshot, so the result is
//! independent of invocation order (a Jacobi iteration, never Gauss-Seidel) and
//! matches the golden's own parallel-safe pass up to a few `ULP` of
//! `sqrt`/division rounding, verified by the parity suite within a tight
//! tolerance.
//!
//! # Provenance
//!
//! The layer-number stacking constraint and inverse-mass-weighted separation
//! are standard position-based dynamics; the Jacobi own-slot accumulate is
//! standard parallel position-based dynamics; uniform spatial hashing is the
//! classical Teschner et al. 2003 scheme. No Unreal Engine source or derived
//! code.

pub mod cpu;
pub mod gpu;
pub mod prep;

pub use cpu::cpu_cloth_layer_coupling;
pub use gpu::GpuClothLayerCoupling;
pub use prep::{build as build_cloth_layer_prep, ClothLayerPrep};
