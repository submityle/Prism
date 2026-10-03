//! Tunable parameters for the XPBD solver.
//!
//! [`XpbdConfig`] groups the knobs that shape how the extended
//! position-based-dynamics solver resolves contacts: how many position-solve
//! iterations run per sub-step, how compliant (soft) contacts are, and the
//! relative normal speed below which restitution is suppressed so resting
//! stacks do not jitter.
//!
//! # Provenance
//!
//! The compliance and restitution-threshold concepts follow Müller et al.,
//! *Detailed Rigid Body Simulation with Extended Position Based Dynamics*
//! (2020). This file contains no Unreal Engine source or derived code.

/// Configuration for [`XpbdSolver`](super::XpbdSolver).
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct XpbdConfig {
    /// Number of position-solve iterations per sub-step.
    ///
    /// With sub-stepping, a single iteration per sub-step already converges
    /// well (the "small steps" regime), so the default is `1`. Raising it
    /// stiffens contacts at the cost of more work per sub-step.
    pub position_iterations: u32,
    /// Contact compliance (inverse stiffness) in metres per newton.
    ///
    /// `0.0` makes contacts perfectly rigid; small positive values soften them.
    /// It enters the position solve as `alpha_tilde = compliance / h^2`.
    pub contact_compliance: f32,
    /// Relative normal speed below which restitution is treated as zero.
    ///
    /// Contacts whose pre-solve closing speed is under this threshold are
    /// resolved as fully inelastic, which removes the residual bouncing that
    /// would otherwise keep a resting stack awake.
    pub restitution_threshold: f32,
    /// Solve independent simulation islands across worker threads.
    ///
    /// This only takes effect when the crate is built with the `parallel`
    /// feature; without it the solver always runs the single-threaded island
    /// pass regardless of this flag. The parallel path is numerically
    /// bit-identical to the serial path (islands touch disjoint dynamic
    /// bodies), so toggling it never changes results, only throughput.
    pub parallel_islands: bool,
    /// Colour a single island's constraint graph and solve one colour at a time
    /// so a *large* island can relax across worker threads instead of pinning
    /// one thread with a serial Gauss-Seidel sweep.
    ///
    /// Islands with disjoint dynamic bodies already run on separate threads via
    /// [`parallel_islands`](Self::parallel_islands); this flag targets the
    /// opposite case — one dense island (a tall stack, a big pile) that other
    /// agents' workloads cannot be split across. Constraints are partitioned so
    /// no two in a colour write the same dynamic body, then colours are applied
    /// in order (Gauss-Seidel across colours, order-independent within a
    /// colour).
    ///
    /// Unlike [`parallel_islands`](Self::parallel_islands), enabling this
    /// **changes the solve order** within an island, so the floating-point
    /// result differs from the natural-index sweep (it converges to the same
    /// rest state, not the same bits). It is therefore gated and defaults to
    /// `false`, which keeps the existing bit-identical island behaviour and all
    /// position goldens intact. Only takes effect with the `parallel` feature.
    pub parallel_within_island: bool,
}

impl XpbdConfig {
    /// The default position-solve iteration count.
    pub const DEFAULT_POSITION_ITERATIONS: u32 = 1;

    /// The default restitution suppression threshold, in metres per second.
    pub const DEFAULT_RESTITUTION_THRESHOLD: f32 = 0.5;
}

impl Default for XpbdConfig {
    fn default() -> Self {
        XpbdConfig {
            position_iterations: Self::DEFAULT_POSITION_ITERATIONS,
            contact_compliance: 0.0,
            restitution_threshold: Self::DEFAULT_RESTITUTION_THRESHOLD,
            parallel_islands: true,
            parallel_within_island: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_rigid_single_iteration() {
        let c = XpbdConfig::default();
        assert_eq!(c.position_iterations, 1);
        assert_eq!(c.contact_compliance, 0.0);
        assert!((c.restitution_threshold - 0.5).abs() < 1e-6);
        assert!(c.parallel_islands);
        assert!(
            !c.parallel_within_island,
            "intra-island colouring is opt-in so default stays bit-identical"
        );
    }
}
