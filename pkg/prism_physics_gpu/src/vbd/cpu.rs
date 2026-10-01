//! `CPU` golden twin for the Vertex Block Descent (VBD) GPU kernel.
//!
//! This twin delegates to the [`prism_physics_core`] reference
//! ([`VbdSolver::step_colored`]) so the parity suite compares the kernel against
//! the exact authority it must reproduce, not a second re-derivation. The GPU
//! kernel reimplements the same colour-major Gauss-Seidel block descent in
//! `WGSL`; a passing parity test is therefore direct evidence the kernel
//! computes the golden's result.
//!
//! # Provenance
//!
//! The VBD formulation follows Chen et al., "Vertex Block Descent" (SIGGRAPH
//! 2024). No Unreal Engine source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::vbd::{SpringSet, VbdColoring, VbdConfig, VbdSolver};
use prism_physics_core::ParticleStorage;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Steps one VBD substep schedule on the `CPU` via the golden
/// [`VbdSolver::step_colored`] and returns the advanced `(positions,
/// velocities)`.
///
/// The input `particles` is cloned so the caller's storage is untouched; the
/// GPU twin takes the same `(particles, springs, config, coloring, dt)` inputs
/// and must return matching arrays.
#[must_use]
pub fn cpu_vbd(
    particles: &ParticleStorage,
    springs: &SpringSet,
    config: &VbdConfig,
    coloring: &VbdColoring,
    dt: Real,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut storage = particles.clone();
    let solver = VbdSolver::new();
    solver.step_colored(&mut storage, springs, config, coloring, dt);
    (storage.positions().to_vec(), storage.velocities().to_vec())
}
