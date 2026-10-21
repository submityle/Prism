//! Rigid and multibody aerodynamics for M12.
//!
//! This module is backend-neutral: it evaluates force and torque from a body
//! snapshot and a sampled flow field, then exposes a narrow sink callback for a
//! fixed-step world adapter. It does not know about `PhysicsWorld`, the solver,
//! storage layout, or an integrator. The same pure evaluator can therefore be
//! used by CPU, GPU preparation, offline baking, and deterministic tests.

mod apply;
mod field;
mod fluid;
mod registry;
mod sanitize;
mod surface;

pub use apply::{
    apply_aero_forces, compute_aero_wrench, AeroBodyState, AeroForceSink, AeroModel, AeroWrench,
};
pub use field::{
    Downwash, FlowField, FlowJet, GroundEffect, GroundEffectScale, Gust, Real, Slipstream,
    Turbulence, WindField, WindSample,
};
pub use fluid::{AddedMass, BuoyancyVolume, FluidPlane, MagnusParams};
pub use registry::AeroBodyConfig;
pub use surface::{AeroSurface, ControlSurface, FluidKind, LiftCurve, Thruster};
