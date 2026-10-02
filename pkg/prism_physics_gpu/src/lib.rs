//! Optional `wgpu` compute backend for Prism's next-generation physics engine.
//!
//! This crate is milestone `M5`: it moves the mass-parallel stages of the
//! simulation (broad-phase neighbour finding, the position-based constraint
//! solver, and the `FLIP`/`APIC` fluid particle-grid transfer) onto the `GPU`
//! through `wgpu` compute pipelines, targeting the
//! hundred-thousand-to-million particle regime that the pure `CPU` reference in
//! [`prism_physics_core`] cannot reach in real time.
//!
//! # Correctness model
//!
//! Every kernel here is paired with a `CPU` golden twin. The twin lives next to
//! the kernel (for example [`broadphase::cpu_broadphase`] and
//! [`xpbd::cpu_solve`]) and runs the identical arithmetic, so a passing
//! real-device parity test is direct evidence that the ported kernel computes
//! the same result as the reference, not merely that its `WGSL` compiles. Each
//! `CPU` twin is in turn anchored against an independent brute-force reference
//! in its own unit tests, closing the loop from first principles.
//!
//! The strength of the parity claim depends on the kernel. The broad phase is
//! integer-exact: its candidate-pair set matches the twin bit-for-bit. The
//! `XPBD` solver is floating-point: `GPU` reassociation (fused multiply-add,
//! differing division and square-root rounding) perturbs the low bits, so its
//! parity is verified within a tight tolerance rather than byte-for-byte. The
//! `FLIP`/`APIC` fluid transfer sits between the two: its fixed-point momentum
//! and weight accumulators are integer-exact and order-independent, and only
//! the final per-face division and trilinear gather are floating point, so its
//! transfer round-trip is likewise checked within a tight tolerance.
//!
//! # Provenance
//!
//! All algorithms are standard, openly published techniques (Teschner et al.
//! 2003 spatial hashing for the broad phase; extended position-based dynamics,
//! Müller et al., for the constraint solver, with textbook greedy first-fit
//! graph colouring). This crate contains no Unreal Engine source or derived
//! code.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
#![forbid(unsafe_code)]

extern crate alloc;

pub mod broadphase;
pub mod buffer;
pub mod bvh;
pub mod cfl;
pub mod cloth;
pub mod collider;
pub mod contacts;
pub mod context;
pub mod fluid;
pub mod fracture;
pub mod grid;
pub mod island;
pub mod mpm;
pub mod narrowphase;
pub mod radix;
pub mod rigid;
pub mod scan;
pub mod vbd;
pub mod xpbd;

pub use broadphase::{cpu_broadphase, BroadphaseConfig, BroadphaseError, CandidatePair, Particle};
pub use bvh::{
    cpu_build_lbvh, cpu_bvh_aabb_overlap, cpu_refit_lbvh, cpu_bvh_pairs, cpu_bvh_raycast_any,
    cpu_bvh_raycast_closest, lbvh_sah_cost, lbvh_sah_cost_weighted, surface_area, Aabb, BvhQueryError,
    GpuBvhOverlap, GpuBvhQuery, GpuBvhRaycast, GpuBvhRefit, GpuBvhSahCost, GpuLbvh, GpuResidentLbvh,
    Lbvh,
    OverlapQueryError, Ray, RayHit, FrameUpdate, RebuildDecision, RefitQualityTracker, ResidentBvhDriver, SceneBounds,
    UpdateAction, NO_PARENT,
};
pub use cfl::{cpu_cfl_dt, cpu_max_speed, CflConfig, GpuCflReduce};
pub use cloth::{
    build_cloth_aero_prep, build_vertex_triangle_adjacency, colour_bending, colour_long_range,
    colour_strain_limit, cpu_cloth_aero, cpu_cloth_backstops, cpu_cloth_bending,
    cpu_cloth_body_collision, cpu_cloth_ccd, cpu_cloth_coupling, cpu_cloth_layer_coupling, cpu_cloth_long_range,
    cpu_cloth_plasticity, cpu_cloth_pressure, cpu_cloth_self_ccd, cpu_cloth_self_collision_jacobi,
    cpu_cloth_self_collision_point, cpu_cloth_strain_limit, cpu_cloth_tearing, pack_backstops,
    pack_body_colliders, BendingColoring, ClothAeroParams, ClothAeroPrep, ClothAeroTriangle,
    ClothBendingConstraint, ClothLongRangeConstraint, ClothPlasticEdge, ClothPrep,
    ClothSelfCollisionScope, ClothStrainLimitConstraint, ClothTearEdge, GpuBackstop,
    GpuBodyCollider, GpuClothAero, GpuClothBending, GpuClothBodyCollision, GpuClothCcd,
    GpuClothCoupling, GpuClothLayerCoupling, GpuClothLongRange, GpuClothPlasticity, GpuClothPressure, GpuClothSelfCcd,
    GpuClothSelfCollision, GpuClothSelfCollisionPoint, GpuClothStrainLimit, GpuClothTearing,
    LongRangeColoring, StrainLimitColoring, VertexTriangleAdjacency,
};
pub use collider::{
    cpu_capsule_trimesh_collide, cpu_obb_trimesh_collide, cpu_obb_trimesh_manifold_collide,
    cpu_sphere_trimesh_collide, GpuCapsuleTrimeshCollider, GpuObbTrimeshCollider,
    GpuSphereTrimeshCollider, Trimesh,
};
pub use contacts::{
    contact_constraints, contact_constraints_with_friction, cpu_resolve_contacts,
    cpu_resolve_contacts_warm, ContactCache, ContactConstraint, ContactKey, GpuContactSolver,
    GpuContactWarmSolver,
};
pub use context::GpuContext;
pub use fluid::{
    grid_to_particle, particle_to_grid, CellType, FluidConfig, FluidError, FluidParticles,
    GoldenGrid, GpuAdvect, GpuExtrapolate, GpuFluidApicStep, GpuFluidSolver, GpuFluidStep,
    GpuGridOps, GpuPressureSolver, GridDims, PressureConfig, TransferMode,
};
pub use fracture::{
    cpu_aggregate_fragments, cpu_assign_cells, cpu_bounds_fragments, AggregateConfig, BoundsConfig,
    CellAssignment, FragmentAggregate, FragmentBounds, GpuFragmentAggregate, GpuFragmentBounds,
    GpuVoronoiAssign, VoronoiAssignConfig, NO_CELL,
};
pub use grid::{cpu_grid_sort, GpuUniformGrid, GridBuild, GridConfig, GridError};
pub use island::{build_islands, Island, IslandSet, SleepConfig, SleepState};
pub use mpm::{
    BoundaryMode, ConstitutiveOutput, G2pParticles, GpuMpmConstitutive, GpuMpmG2p,
    GpuMpmGridUpdate, GpuMpmP2g, GpuMpmResident, GpuMpmStep, P2gGrid, StepConfig, StepInputs,
    StepParticles,
};
pub use narrowphase::{
    cpu_capsule_capsule_manifold, cpu_capsule_capsule_narrowphase, cpu_capsule_halfspace_manifold, cpu_capsule_heightfield_manifold, cpu_obb_heightfield_manifold,
    cpu_capsule_narrowphase, cpu_capsule_obb_manifold, cpu_capsule_obb_narrowphase,
    cpu_capsule_triangle_manifold, cpu_capsule_triangle_narrowphase,
    cpu_halfspace_narrowphase, cpu_narrowphase, cpu_obb_halfspace_manifold,
    cpu_obb_halfspace_narrowphase, cpu_obb_narrowphase, cpu_obb_obb_manifold,
    cpu_obb_obb_narrowphase, cpu_obb_triangle_manifold, cpu_obb_triangle_narrowphase, cpu_sphere_heightfield_narrowphase, cpu_sphere_triangle_narrowphase, Capsule, CapsuleCapsulePair, CellRange,
    CapsuleObbPair, CapsulePlanePair, CapsuleTrianglePair, Contact, ContactManifold,
    GpuCapsuleCapsuleManifoldNarrowphase, GpuCapsuleCapsuleNarrowphase,
    GpuCapsuleHalfspaceNarrowphase, GpuCapsuleNarrowphase, GpuCapsuleObbManifoldNarrowphase,
    GpuCapsuleHeightfieldManifoldNarrowphase, GpuCapsuleObbNarrowphase, GpuCapsuleTriangleManifoldNarrowphase, GpuCapsuleTriangleNarrowphase, GpuHalfspaceNarrowphase, GpuNarrowphase, GpuSphereHeightfieldNarrowphase,
    GpuObbHalfspaceManifoldNarrowphase, GpuObbHalfspaceNarrowphase, GpuObbNarrowphase,
    GpuObbObbManifoldNarrowphase, GpuObbObbNarrowphase, GpuObbTriangleManifoldNarrowphase, GpuObbTriangleNarrowphase, GpuSphereTriangleNarrowphase,
    Heightfield, HeightfieldCapsulePair, HeightfieldObbPair, HeightfieldSpherePair, ManifoldPoint, Obb, ObbObbPair, ObbTrianglePair,
    ObbPlanePair, Plane, SphereCapsulePair, SphereObbPair, SpherePlanePair, SphereTrianglePair,
    Triangle, XzAabb, MAX_MANIFOLD_POINTS,
};
pub use radix::{cpu_radix_sort_keys, cpu_radix_sort_pairs, GpuRadixSort};
pub use rigid::{
    cpu_integrate, cpu_integrate_gyro, cpu_solve_contacts, cpu_solve_contacts_tgs,
    cpu_solve_joints_angular_slerp_drive, cpu_solve_joints_cylindrical,
    cpu_solve_joints_cylindrical_drive, cpu_solve_joints_cylindrical_limit, cpu_solve_joints_d6,
    cpu_solve_joints_d6_driven, cpu_solve_joints_distance, cpu_solve_joints_elliptical_cone_twist,
    cpu_solve_joints_fixed, cpu_solve_joints_gear, cpu_solve_joints_hinge_limit,
    cpu_solve_joints_prismatic, cpu_solve_joints_prismatic_drive, cpu_solve_joints_prismatic_limit,
    cpu_solve_joints_rack_pinion, cpu_solve_joints_revolute, cpu_solve_joints_revolute_drive,
    cpu_solve_joints_revolute_motor, cpu_solve_joints_revolute_servo, cpu_solve_joints_spherical,
    cpu_solve_joints_swing_twist, cpu_solve_joints_universal, AngularSlerpDriveJoint,
    ContactSolverConfig, CylindricalDriveJoint, CylindricalJoint, CylindricalLimitJoint, D6Drive,
    D6DriveSet, D6Joint, D6Motion, DistanceJoint, EllipticalConeTwistJoint, FixedJoint, GearJoint,
    GpuAngularSlerpDriveJointSolver, GpuCylindricalDriveJointSolver, GpuCylindricalJointSolver,
    GpuCylindricalLimitJointSolver, GpuD6DriveJointSolver, GpuD6JointSolver,
    GpuDistanceJointSolver, GpuEllipticalConeTwistJointSolver, GpuFixedJointSolver,
    GpuGearJointSolver, GpuHingeLimitJointSolver, GpuPrismaticDriveJointSolver,
    GpuPrismaticJointSolver, GpuPrismaticLimitJointSolver, GpuRackPinionJointSolver,
    GpuRevoluteDriveJointSolver, GpuRevoluteJointSolver, GpuRevoluteMotorJointSolver,
    GpuRevoluteServoJointSolver, GpuRigidContactSolver, GpuRigidIntegrator,
    GpuRigidTgsContactSolver, GpuSphericalJointSolver, GpuSwingTwistJointSolver,
    GpuUniversalJointSolver, GyroscopicConfig, GyroscopicMode, HingeLimitJoint, IntegratorConfig,
    JointColouring, JointSolverConfig, PrismaticDriveJoint, PrismaticJoint, PrismaticLimitJoint,
    RackPinionJoint, RevoluteDriveJoint, RevoluteJoint, RevoluteMotorJoint, RevoluteServoJoint,
    RigidBodyState, RigidContact, RigidContactColouring, RigidError, SphericalJoint,
    SwingTwistJoint, TgsContactConfig, UniversalJoint, MAX_JOINT_BATCHES,
};
pub use scan::{cpu_compact, cpu_exclusive_scan, GpuScan};
pub use vbd::{build_vbd_prep, cpu_vbd, GpuSpring, GpuVbd, VbdPrep};
pub use xpbd::{
    cpu_solve, cpu_solve_warm, tgs_solve, Colouring, DistanceCache, DistanceConstraint,
    DistanceKey, GpuIslandedTgsSolver, GpuIslandedXpbdSolver, GpuTgsSolver, GpuXpbdSolver,
    GpuXpbdWarmSolver, IslandStep, IslandedSolver, IslandedTgsSolver, ParticleState, SoftParams,
    TgsConfig, XpbdConfig, XpbdError,
};
