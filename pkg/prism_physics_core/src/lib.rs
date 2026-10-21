//! Backend-neutral core kernel for Prism's next-generation physics engine.
//!
//! This crate provides the rigid-body state store, math helpers, semi-implicit
//! Euler integrator, solver, backend, driver, and island primitives used by the
//! current physics pipeline.
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
//! - [`solver`], [`backend`], [`driver`], and [`island`] expose the execution
//!   contracts used by the current pipeline, with CPU reference implementations
//!   where each contract requires one.
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

pub mod aero;
pub mod backend;
pub mod cache;
pub mod ccd;
pub mod collide;
pub mod collider;
pub mod collision_filter;
pub mod command;
pub mod config;
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
pub use aero::{
    compute_aero_wrench, AddedMass, AeroBodyConfig, AeroBodyState, AeroForceSink, AeroModel,
    AeroSurface, AeroWrench, BuoyancyVolume, ControlSurface, Downwash, FlowField, FlowJet,
    FluidKind, FluidPlane, GroundEffect, GroundEffectScale, Gust, LiftCurve, MagnusParams,
    Slipstream, Thruster, Turbulence, WindField, WindSample,
};
pub use backend::{CpuBackend, PhysicsBackend};
pub use cache::{
    trajectory_hash, BakeConfig, Baker, CacheTrack, FrameRecord, GoldenDigest, PhysicsCache,
    PlaybackConfig, Player, PositionQuantizer,
};
pub use ccd::CcdConfig;
pub use collide::{
    generate_contact, generate_contact_in, ContactManifold, ContactPoint, MAX_MANIFOLD_POINTS,
};
pub use collider::material::PhysicsMaterial;
pub use collider::{ColliderHandle, ColliderShape, ShapeRegistry};
pub use collision_filter::{
    CollisionExclusionError, CollisionExclusionGroupId, CollisionExclusionMembership,
    CollisionExclusions,
};
pub use command::{CommandQueue, PhysicsCommand};
pub use config::WorldConfig;
pub use driver::fixed_step::{AdvanceReport, FixedStepPipeline};
pub use driver::playback::{PlaybackError, PlaybackRepeat, PlaybackSample, TimedPlayback};
pub use driver::{
    CacheFrame, CacheHandle, CacheRegistrationError, DriveMode, DriveOutcome, DriverCache,
    SimulationDriver, DRIVER_CACHE_VERSION, MAX_DRIVER_CACHE_FRAMES, MAX_OFFLINE_SUBSTEPS,
    MAX_REGISTERED_CACHES,
};
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
    AngleLimit, DistanceJoint, FixedJoint, GearJoint, GearJointError, GearJointState, Joint,
    JointAnchor, JointDesc, JointHandle, JointKind, JointStorage, LinearLimit, Motor, MotorTarget,
    PrismaticJoint, RevoluteJoint, SphericalJoint,
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
pub use soft::aero::{
    apply_aero_forces, apply_aero_to_columns, triangle_aero_force, turbulence_offset, AeroParams,
};
pub use soft::body::SoftBody;
pub use soft::build::{Cloth, ClothGrid, Rope, RopeGrid, SoftBox, SoftBoxGrid};
pub use soft::collision::{
    accumulate_vertex_normals, apply_backstop, capsule_toi, closest_point_on_segment,
    couple_particle_against_body, generate_virtual_particles, half_space_toi,
    project_out_of_half_space, project_out_of_sphere, resolve_backstops, resolve_body_collisions,
    resolve_body_collisions_with_friction, resolve_ccd, resolve_layer_coupling,
    resolve_layer_coupling_jacobi, resolve_self_ccd, resolve_self_ccd_jacobi,
    resolve_self_collision, resolve_self_collision_jacobi, resolve_self_collision_virtual,
    resolve_self_collision_virtual_augment, resolve_self_collision_virtual_augment_jacobi,
    resolve_self_collision_virtual_jacobi, resolve_self_collision_with_friction,
    resolve_self_collision_with_friction_jacobi, resolve_two_way_coupling, sphere_toi,
    swept_pair_toi, Backstop, BodyCollider, CcdParams, ConvexProxy, CouplingBody,
    CouplingContribution, LayerParams, Plane, SelfCcdParams, VirtualParticle,
    VirtualParticlePattern, MAX_CONVEX_PLANES,
};
pub use soft::constraint::{
    mesh_volume, project_bending, project_long_range, project_pressure, project_strain_limit,
    AttachmentConstraint, BendingConstraint, ConstraintSet, DistanceConstraint,
    LongRangeConstraint, ParticleConstraint, PressureConstraint, SoftConstraintKind,
    StrainLimitConstraint, TetraVolumeConstraint,
};
pub use soft::damage::{
    apply_plasticity, apply_tearing, plastic_rest_length, tear_flag, tear_flags, tear_report,
    PlasticParams, TearReport, TearingParams,
};
pub use soft::particle::{ParticleHandle, ParticleStorage};
pub use soft::rigid_coupling::{
    convex_proxy_from_cuboid, couple_cloth_to_rigid, couple_cloth_to_rigid_angular,
    couple_cloth_to_rigid_friction, gather_rigid_proxies, AngularCouplingReport,
    ClothRigidCouplingConfig, CouplingReport, FrictionCouplingReport, RigidProxy,
};
pub use soft::solver::{
    SelfCollisionParams, SoftContacts, SoftSolver, SoftSolverConfig, VirtualSelfCollisionParams,
};
pub use soft::storage::{SoftBodyHandle, SoftBodyStorage};
pub use soft::topology::{
    TearableMesh, TopologyEdge, TopologyMeshError, TopologyTearEvent, TopologyTearEventKind,
    TopologyTearingConfig,
};
pub use solver::{IntegrateOnlySolver, Solver, SolverRegistry, XpbdConfig, XpbdSolver};
pub use state::body::{BodyDesc, BodyKind, MassProperties, SolverMassProperties};
pub use state::handle::BodyHandle;
pub use state::storage::BodyStorage;
pub use state::view::BodySolverView;
pub use vbd::{
    outer, SpringContribution, SpringElement, SpringSet, VbdBody, VbdConfig, VbdSolver,
    VertexSystem,
};
pub use world::{
    AeroApplyReport, AeroBodyRegistration, AeroBodySample, AeroRegistrationError, AeroStepReport,
    AssemblyMemberRemap, AssemblySplitReport, JointBoundaryFailure, PhysicsWorld,
};
