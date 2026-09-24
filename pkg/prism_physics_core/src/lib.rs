//! Backend-neutral core kernel for Prism's next-generation physics engine.
//!
//! This crate provides the M0 foundation: a Structure-of-Arrays rigid-body
//! state store, math helpers, a real semi-implicit Euler integrator, and the
//! reserved extension points (solver / backend / driver / constraint / island)
//! that later milestones build upon.
//!
//! # Design overview
//!
//! - [`math`] holds the scalar configuration ([`Real`]) and the [`Isometry`]
//!   rigid transform.
//! - [`state`] holds the generational [`BodyHandle`], body description types,
//!   and the Structure-of-Arrays [`BodyStorage`].
//! - [`dynamics`] holds the real semi-implicit Euler [`Integrator`].
//! - [`collider`] holds analytic [`ColliderShape`]s and a shared
//!   [`ShapeRegistry`].
//! - [`world`] bundles the storage, colliders, and [`WorldConfig`] into a
//!   single [`PhysicsWorld`].
//! - [`solver`], [`backend`], [`driver`], [`constraint`], and [`island`] are
//!   the reserved extension points. Each is a genuine trait or enum interface
//!   with a working reference implementation where one is specified; none of
//!   them use `todo!()`, `unimplemented!()`, or hollow stub logic.
//!
//! # Provenance
//!
//! This crate is engine-agnostic and contains **no Unreal Engine source or
//! derived code**. All rigid-body dynamics math (semi-implicit Euler
//! integration, closed-form inertia tensors, quaternion integration, and
//! union-find island building) is implemented from standard, publicly
//! documented physics and computer-science knowledge.
#![forbid(unsafe_code)]

pub mod backend;
pub mod collider;
pub mod config;
pub mod constraint;
pub mod driver;
pub mod dynamics;
pub mod island;
pub mod math;
pub mod solver;
pub mod state;
pub mod world;

// Curated, prelude-style re-exports of the most commonly used public types.
pub use backend::{CpuBackend, PhysicsBackend};
pub use collider::{ColliderHandle, ColliderShape, ShapeRegistry};
pub use config::WorldConfig;
pub use constraint::{Constraint, ConstraintKind};
pub use driver::{CacheHandle, DriveMode, DriveOutcome, SimulationDriver};
pub use dynamics::Integrator;
pub use island::{islands_from_pairs, IslandBuilder, IslandId, IslandSet};
pub use math::scalar::{approx_eq, Real, EPSILON, PI, TAU};
pub use math::transform::Isometry;
pub use solver::{IntegrateOnlySolver, Solver, SolverRegistry};
pub use state::body::{BodyDesc, BodyKind, MassProperties};
pub use state::handle::BodyHandle;
pub use state::storage::BodyStorage;
pub use world::PhysicsWorld;
