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

// `prism_physics_core` is a std crate, but the workspace `std_instead_of_alloc`
// lint prefers `alloc` collections where they exist (e.g. `BTreeMap`). Pulling
// in `alloc` makes those canonical paths resolvable from this std crate.
extern crate alloc;

pub mod backend;
pub mod cache;
pub mod ccd;
pub mod collide;
pub mod collider;
pub mod command;
pub mod config;
pub mod constraint;
pub mod driver;
pub mod dsl;
pub mod dynamics;
pub mod events;
pub mod fluid;
pub mod fracture;
pub mod island;
pub mod joint;
pub mod lod;
pub mod math;
pub mod mpm;
pub mod pipeline;
pub mod query;
pub mod reconstruct;
pub mod reduced;
pub mod sleep;
pub mod snapshot;
pub mod soft;
pub mod solver;
pub mod state;
pub mod vbd;
pub mod world;

// Curated, prelude-style re-exports of the most commonly used public types.
pub use backend::{CpuBackend, PhysicsBackend};
pub use cache::{
    trajectory_hash, BakeConfig, Baker, CacheTrack, FrameRecord, GoldenDigest, PhysicsCache,
    PlaybackConfig, Player, PositionQuantizer,
};
pub use ccd::CcdConfig;
pub use collide::{generate_contact, ContactManifold, ContactPoint, MAX_MANIFOLD_POINTS};
pub use collider::material::PhysicsMaterial;
pub use collider::{ColliderHandle, ColliderShape, ShapeRegistry};
pub use command::{CommandQueue, PhysicsCommand};
pub use config::WorldConfig;
pub use constraint::{Constraint, ConstraintKind};
pub use driver::fixed_step::{AdvanceReport, FixedStepPipeline};
pub use driver::{CacheHandle, DriveMode, DriveOutcome, SimulationDriver};
pub use dsl::{
    compile_constraint, compile_expr, eval as dsl_eval, parse_constraint, parse_expression,
    tokenize, BinOp, BuiltinFn, CompiledConstraint, ConstraintDecl, DslConstraint, DslError,
    Environment, Expr, HotReloadRegistry, OpCode, ParamSpec, ParamValue, ParameterStore, Program,
    SpannedToken, Token, VarBinding, VarKind,
};
pub use dynamics::Integrator;
pub use events::{ContactEventTracker, ContactPair, Observer, ObserverRegistry, PhysicsEvent};
pub use fluid::{
    advect, clamp_to_fluid_domain, grid_to_particle as fluid_grid_to_particle,
    max_fluid_divergence, particle_to_grid as fluid_particle_to_grid, project, CellType,
    FluidConfig, FluidSolver, MacGrid, MarkerParticles, TransferMode,
};
pub use fracture::{
    fracture_aabb, fracture_convex, scatter_impact, scatter_uniform, shatter_box,
    shatter_box_impact, ConvexPolyhedron, DeterministicRng, FractureConfig, Fragment,
    MassProperties as FractureMassProperties, Plane as FracturePlane,
};
pub use island::{islands_from_pairs, IslandBuilder, IslandId, IslandSet};
pub use joint::{
    AngleLimit, DistanceJoint, FixedJoint, Joint, JointAnchor, JointDesc, JointHandle, JointKind,
    JointStorage, LinearLimit, Motor, MotorTarget, PrismaticJoint, RevoluteJoint, SphericalJoint,
};
pub use lod::{LodConfig, SpatialController, TemporalController};
pub use math::scalar::{approx_eq, Real, EPSILON, PI, TAU};
pub use math::transform::Isometry;
pub use mpm::{
    apply_grid_boundary, clamp_particles, cofactor, corotated_pf, corotated_piola,
    grid_to_particle as mpm_grid_to_particle, hardening_factor,
    particle_to_grid as mpm_particle_to_grid, polar_rotation, snow_return_mapping, svd3,
    symmetric_eigen, BoundaryCondition, Grid, MaterialPoints, MpmConfig, MpmMaterial, MpmSolver,
    PlasticUpdate, QuadraticWeights, SnowPlasticity, Svd3,
};
pub use pipeline::detect_contacts;
pub use query::{PointProjection, QueryFilter, RayHit, SweepHit};
pub use reconstruct::{triangulate, ScalarField, SurfaceMesh};
pub use reduced::{
    ReducedConfig, ReducedMode, ReducedModel, ReducedState, SymmetricEigen, SymmetricMatrix,
};
pub use sleep::SleepConfig;
pub use snapshot::buffer::TripleBuffer;
pub use snapshot::hash::{hash_state, locate_divergence, StateHash};
pub use snapshot::pose::{lerp_pose, BodyPose};
pub use snapshot::StateSnapshot;
pub use soft::body::SoftBody;
pub use soft::build::{Cloth, ClothGrid, Rope, RopeGrid, SoftBox, SoftBoxGrid};
pub use soft::constraint::{
    AttachmentConstraint, BendingConstraint, ConstraintSet, DistanceConstraint,
    LongRangeConstraint, ParticleConstraint, SoftConstraintKind, StrainLimitConstraint,
    TetraVolumeConstraint,
};
pub use soft::collision::{
    apply_backstop, closest_point_on_segment, project_out_of_half_space, project_out_of_sphere,
    resolve_backstops, resolve_body_collisions, resolve_body_collisions_with_friction,
    resolve_self_collision, resolve_self_collision_with_friction, Backstop, BodyCollider,
};
pub use soft::particle::{ParticleHandle, ParticleStorage};
pub use soft::solver::{SelfCollisionParams, SoftContacts, SoftSolver, SoftSolverConfig};
pub use solver::{IntegrateOnlySolver, Solver, SolverRegistry, XpbdConfig, XpbdSolver};
pub use state::body::{BodyDesc, BodyKind, MassProperties};
pub use state::handle::BodyHandle;
pub use state::storage::BodyStorage;
pub use state::view::BodySolverView;
pub use vbd::{
    outer, SpringContribution, SpringElement, SpringSet, VbdBody, VbdConfig, VbdSolver,
    VertexSystem,
};
pub use world::PhysicsWorld;
