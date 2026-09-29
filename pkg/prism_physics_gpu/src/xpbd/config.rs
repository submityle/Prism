//! Solver configuration and error type for the `GPU` `XPBD` distance solver.
//!
//! [`XpbdConfig`] carries the same substep `XPBD` tunables as the `CPU`
//! reference in [`prism_physics_core`](prism_physics_core::soft::solver): a
//! uniform acceleration, a substep count, a projection-iteration count, and a
//! linear velocity damping coefficient. The frame time step is split into equal
//! substeps; each substep predicts positions, resets the per-constraint
//! Lagrange multipliers, runs the projection sweeps, then recovers velocities
//! from the net motion.
//!
//! Provenance: standard substep `XPBD` parameters (Müller et al.). No Unreal
//! Engine source or derived code.

use glam::Vec3;

/// Tunables controlling how the distance-constraint system is advanced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XpbdConfig {
    /// Uniform acceleration (metres per second squared) applied to every
    /// non-pinned particle each substep, typically gravity.
    pub gravity: Vec3,
    /// Number of equal substeps the frame time step is divided into. Clamped to
    /// at least `1` when stepping.
    pub substeps: u32,
    /// Number of projection sweeps over the coloured constraint batches per
    /// substep. Clamped to at least `1` when stepping.
    pub iterations: u32,
    /// Linear velocity damping coefficient (per second). Each substep scales
    /// velocity by `(1 - damping * h).max(0)`.
    pub damping: f32,
}

impl XpbdConfig {
    /// Default gravity (Earth-like, downward along `-Y`).
    pub const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);
    /// Default substep count.
    pub const DEFAULT_SUBSTEPS: u32 = 8;
    /// Default projection iterations per substep.
    pub const DEFAULT_ITERATIONS: u32 = 1;
    /// Default linear velocity damping (per second).
    pub const DEFAULT_DAMPING: f32 = 0.5;

    /// Creates a configuration.
    #[must_use]
    pub fn new(gravity: Vec3, substeps: u32, iterations: u32, damping: f32) -> XpbdConfig {
        XpbdConfig {
            gravity,
            substeps,
            iterations,
            damping,
        }
    }

    /// Returns the effective substep count (at least `1`).
    #[must_use]
    pub fn effective_substeps(&self) -> u32 {
        self.substeps.max(1)
    }

    /// Returns the effective iteration count (at least `1`).
    #[must_use]
    pub fn effective_iterations(&self) -> u32 {
        self.iterations.max(1)
    }

    /// Validates the configuration, returning the first violated invariant.
    ///
    /// # Errors
    ///
    /// Returns [`XpbdError::InvalidConfig`] when `gravity` or `damping` is not
    /// finite. Zero substep or iteration counts are permitted and clamped to
    /// `1` at step time.
    pub fn validate(&self) -> Result<(), XpbdError> {
        if !self.gravity.is_finite() {
            return Err(XpbdError::InvalidConfig("gravity must be finite"));
        }
        if !self.damping.is_finite() {
            return Err(XpbdError::InvalidConfig("damping must be finite"));
        }
        Ok(())
    }
}

impl Default for XpbdConfig {
    fn default() -> Self {
        XpbdConfig {
            gravity: Self::DEFAULT_GRAVITY,
            substeps: Self::DEFAULT_SUBSTEPS,
            iterations: Self::DEFAULT_ITERATIONS,
            damping: Self::DEFAULT_DAMPING,
        }
    }
}

/// Errors the `GPU` `XPBD` solver can report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XpbdError {
    /// A configuration invariant was violated; carries a static reason.
    InvalidConfig(&'static str),
    /// A constraint referenced a particle index outside the state arrays.
    ConstraintOutOfRange {
        /// Index of the offending constraint.
        constraint: u32,
        /// The particle index it referenced.
        particle: u32,
        /// The number of particles in the state.
        particle_count: u32,
    },
    /// The constraint graph required more colours than the solver supports.
    ///
    /// The greedy colouring packs one colour bit per particle into a `u64`, so
    /// a physically meaningful mesh (whose vertex degree is far below `64`)
    /// never trips this; it guards against pathological fully-connected inputs.
    TooManyColours {
        /// The colour index that exceeded the supported maximum.
        colour: u32,
        /// The maximum number of colours supported.
        maximum: u32,
    },
}

impl core::fmt::Display for XpbdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            XpbdError::InvalidConfig(reason) => write!(f, "invalid XPBD config: {reason}"),
            XpbdError::ConstraintOutOfRange {
                constraint,
                particle,
                particle_count,
            } => write!(
                f,
                "constraint {constraint} references particle {particle} but only \
                 {particle_count} particles exist"
            ),
            XpbdError::TooManyColours { colour, maximum } => write!(
                f,
                "constraint graph needs colour {colour} but only {maximum} colours are supported"
            ),
        }
    }
}

impl core::error::Error for XpbdError {}
