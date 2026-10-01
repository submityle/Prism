//! `GPU` virtual-particle cloth self-collision.
//!
//! Cloth self-intersection is resolved with the `NvCloth` virtual-particle
//! technique: every triangle is seeded with barycentric sample points (face
//! centroid plus edge midpoints) so a vertex that tries to tunnel through a
//! face interior still finds a nearby sample to push against. This module runs
//! one *parallel-safe Jacobi* pass of that tier on the `GPU`, the faithful twin
//! of [`prism_physics_core`]'s `resolve_self_collision_virtual_jacobi`: the
//! Gauss-Seidel core cannot map to a compute kernel because every invocation
//! must read the *same* frozen snapshot, so the Jacobi reformulation is what
//! the hardware actually runs.
//!
//! # Layout
//!
//! - [`prep`] — the host-side deterministic build of the sample table, uniform
//!   hash, and phase-2 incidence list, laid out in the golden's exact order.
//! - [`cpu`] — the [`cpu_cloth_self_collision_jacobi`] golden twin (delegating
//!   to [`prism_physics_core`]) and its independent brute-force anchor.
//! - [`gpu`] — the real-device [`GpuClothSelfCollision`] pipeline pair.
//! - [`layout`] — the shared bind-group layout helpers.
//!
//! # Correctness model
//!
//! The kernel is paired with the [`cpu_cloth_self_collision_jacobi`] twin
//! running the identical pass. Cell assignment and bucketing are integer-exact
//! (built on the host in [`prep`]), so the only floating-point divergence is in
//! the separating-push arithmetic (`GPU` fused multiply-add and division/sqrt
//! rounding), and parity is verified within a tight tolerance rather than
//! bit-for-bit — the same model the `XPBD` solver uses.
//!
//! # Provenance
//!
//! The virtual-particle technique is the published `NvCloth` method; the Jacobi
//! own-slot accumulate/apply split is standard parallel position-based
//! dynamics; uniform spatial hashing is the classical Teschner et al. 2003
//! scheme. No Unreal Engine source or derived code.

pub mod cpu;
pub mod gpu;
pub mod layout;
pub mod prep;

pub use cpu::cpu_cloth_self_collision_jacobi;
pub use gpu::GpuClothSelfCollision;
pub use prep::{build as build_cloth_prep, ClothPrep};

/// Which sample pairs one cloth self-collision pass resolves.
///
/// Mirrors `prism_physics_core`'s internal `PairScope`: the engine keeps that
/// type crate-private, so this is the `GPU` crate's public re-statement of the
/// same two modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClothSelfCollisionScope {
    /// Every pair, including real-vertex versus real-vertex, so the pass is a
    /// self-contained self-collision tier (reduces to the point-to-point tier
    /// when there are no virtual particles).
    All,
    /// Only pairs where at least one sample is a virtual particle, so the pass
    /// augments an existing friction point-to-point tier without stripping its
    /// tangential friction from the real-vertex pairs.
    VirtualOnly,
}
