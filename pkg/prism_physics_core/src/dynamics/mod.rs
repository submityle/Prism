//! Free-body dynamics integration.
//!
//! This module contains the real (non-stub) semi-implicit Euler
//! [`integrator::Integrator`] used to advance dynamic bodies under gravity and
//! damping.

pub mod integrator;

pub use integrator::Integrator;
