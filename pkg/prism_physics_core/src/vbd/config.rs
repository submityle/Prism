//! Configuration for the Vertex Block Descent solver.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Substep
//! counts, block-descent iteration counts, and linear velocity damping are
//! standard, publicly documented time-integration parameters.

use glam::Vec3;

use crate::math::scalar::Real;

/// Tunables controlling how a Vertex Block Descent body is advanced each frame.
///
/// The frame time step is divided into [`substeps`](Self::substeps) equal
/// substeps. Each substep forms the implicit-Euler inertial target, then runs
/// [`iterations`](Self::iterations) Gauss-Seidel vertex-block sweeps. Unlike
/// XPBD, VBD stays unconditionally stable even at very high stiffness and low
/// iteration counts, so a single substep with a handful of iterations is
/// usually enough.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VbdConfig {
    /// Uniform acceleration (metres per second squared) applied to every
    /// non-pinned vertex, typically gravity.
    pub gravity: Vec3,
    /// Number of equal substeps the frame time step is divided into. Clamped to
    /// at least `1` when stepping.
    pub substeps: u32,
    /// Number of Gauss-Seidel vertex-block sweeps per substep. Clamped to at
    /// least `1` when stepping.
    pub iterations: u32,
    /// Linear velocity damping coefficient (per second). Each substep scales
    /// the recovered velocity by `(1 - damping * h).max(0)`.
    pub damping: Real,
}

impl VbdConfig {
    /// Default gravity (Earth-like, downward along `-Y`) in metres per second
    /// squared.
    pub const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
    /// Default substep count.
    pub const DEFAULT_SUBSTEPS: u32 = 1;
    /// Default block-descent iterations per substep.
    pub const DEFAULT_ITERATIONS: u32 = 8;
    /// Default linear velocity damping (per second).
    pub const DEFAULT_DAMPING: Real = 0.5;
}

impl Default for VbdConfig {
    fn default() -> Self {
        VbdConfig {
            gravity: Self::DEFAULT_GRAVITY,
            substeps: Self::DEFAULT_SUBSTEPS,
            iterations: Self::DEFAULT_ITERATIONS,
            damping: Self::DEFAULT_DAMPING,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_spec() {
        let c = VbdConfig::default();
        assert_eq!(c.gravity, Vec3::new(0.0, -9.81, 0.0));
        assert_eq!(c.substeps, 1);
        assert_eq!(c.iterations, 8);
        assert_eq!(c.damping, 0.5);
    }
}
