//! Free-body dynamics integration.
//!
//! This module contains the semi-implicit Euler
//! [`integrator::Integrator`] used to advance dynamic bodies under gravity and
//! damping.

pub mod assembly;
pub mod field;
pub mod force;
pub mod integrator;

pub use assembly::{
    AssemblyError, BodySnapshot, BodyState, BoundaryLoadError, BreakEvent, BreakLimit,
    BreakableJointSet, CompoundBody, CompoundError, Connection, ConnectionId, GrabDrive, Pose,
    SolverJointLoad,
};
pub use integrator::Integrator;
