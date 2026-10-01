//! Position-level contact resolution for the unified soft-body kernel.
//!
//! The substep XPBD solver ([`crate::soft::solver`]) advances particles under
//! gravity and projects their internal constraints, but two cloth layers (or
//! two folds of the same garment) that drift into one another are not coupled
//! by any distance or bending constraint. This module adds the missing
//! *contact* tier: a deterministic uniform spatial hash that pushes
//! interpenetrating particle pairs apart, optionally rubbing their tangential
//! slide with position-level Coulomb friction so stacked layers grip instead of
//! shearing freely.
//!
//! The resolver is deliberately expressed as array-in / array-out math over the
//! particle store's raw columns ([`crate::soft::particle::ParticleStorage`]):
//! the same positions and inverse masses always produce bit-identical results,
//! so the stage is CPU-golden-testable and maps cleanly onto a future
//! data-parallel backend. Pinned particles (`inverse_mass <= 0`) are never
//! moved, degenerate inputs (non-positive `cell_size`/`thickness`, fewer than
//! two particles, coincident pairs, both-pinned pairs) fall back
//! deterministically, and no path can produce a [`f32::NAN`].
//!
//! Only [`f32::sqrt`] is used; no transcendental functions are called. This is
//! the discrete, position-level approximation of self-collision: it resolves
//! interpenetration measured at the current positions, with the `thickness`
//! buffer providing the tunnelling margin. A continuous-collision-detection
//! (CCD) sweep against per-particle motion segments is a heavier future slot
//! layered on the same spatial hash.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! uniform spatial hash and the inverse-mass-weighted separation are standard
//! position-based-dynamics techniques; the tangential-friction projection is
//! the one published by Macklin et al. (2014), "Unified Particle Physics for
//! Real-Time Applications".

mod body;
mod coupling;
mod friction;
mod ccd;
mod self_ccd;
mod self_collision;
mod virtual_particles;

pub use body::{
    apply_backstop, closest_point_on_segment, project_out_of_half_space, project_out_of_sphere,
    resolve_backstops, resolve_body_collisions, resolve_body_collisions_with_friction, Backstop,
    BodyCollider,
};
pub use ccd::{capsule_toi, half_space_toi, resolve_ccd, sphere_toi, CcdParams};
pub use coupling::{resolve_two_way_coupling, CouplingBody};
pub use self_ccd::{resolve_self_ccd, swept_pair_toi, SelfCcdParams};
pub use self_collision::{resolve_self_collision, resolve_self_collision_with_friction};
pub use virtual_particles::{
    generate_virtual_particles, resolve_self_collision_virtual,
    resolve_self_collision_virtual_augment, VirtualParticle, VirtualParticlePattern,
};

use glam::Vec3;

use crate::math::scalar::Real;

/// Squared-length floor below which a separation vector is treated as zero, so
/// a direction is never recovered from a (near) zero-length vector. Shared by
/// the self-collision passes.
pub(crate) const EPS_LEN_SQ: Real = 1e-12;

/// Returns the integer grid cell containing `pos` for a uniform hash of the
/// given (strictly positive) `cell_size`.
///
/// Each coordinate is scaled by `1 / cell_size` and floored, so a particle at
/// the origin lands in cell `(0, 0, 0)` and the mapping is translation-stable
/// within a cell. Callers guarantee `cell_size > 0`.
pub(crate) fn cell_of(pos: Vec3, cell_size: Real) -> (i32, i32, i32) {
    let inv = 1.0 / cell_size;
    let cx = (pos.x * inv).floor() as i32;
    let cy = (pos.y * inv).floor() as i32;
    let cz = (pos.z * inv).floor() as i32;
    (cx, cy, cz)
}
