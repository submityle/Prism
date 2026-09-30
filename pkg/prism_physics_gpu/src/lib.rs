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

pub mod broadphase;
pub mod buffer;
pub mod bvh;
pub mod cfl;
pub mod contacts;
pub mod context;
pub mod fluid;
pub mod fracture;
pub mod grid;
pub mod mpm;
pub mod narrowphase;
pub mod radix;
pub mod scan;
pub mod xpbd;

pub use broadphase::{cpu_broadphase, BroadphaseConfig, BroadphaseError, CandidatePair, Particle};
pub use bvh::{
    cpu_build_lbvh, cpu_bvh_pairs, cpu_bvh_raycast_any, cpu_bvh_raycast_closest, Aabb,
    BvhQueryError, GpuBvhQuery, GpuBvhRaycast, GpuLbvh, GpuResidentLbvh, Lbvh, Ray, RayHit,
    SceneBounds, NO_PARENT,
};
pub use cfl::{cpu_cfl_dt, cpu_max_speed, CflConfig, GpuCflReduce};
pub use contacts::{
    contact_constraints, contact_constraints_with_friction, cpu_resolve_contacts,
    ContactConstraint, GpuContactSolver,
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
pub use mpm::{
    BoundaryMode, ConstitutiveOutput, G2pParticles, GpuMpmConstitutive, GpuMpmG2p,
    GpuMpmGridUpdate, GpuMpmP2g, GpuMpmResident, GpuMpmStep, P2gGrid, StepConfig, StepInputs,
    StepParticles,
};
pub use narrowphase::{
    cpu_capsule_capsule_narrowphase, cpu_capsule_narrowphase, cpu_halfspace_narrowphase,
    cpu_narrowphase, cpu_obb_halfspace_narrowphase, cpu_obb_narrowphase, cpu_obb_obb_narrowphase,
    Capsule, CapsuleCapsulePair, Contact, GpuCapsuleCapsuleNarrowphase, GpuCapsuleNarrowphase,
    GpuHalfspaceNarrowphase, GpuNarrowphase, GpuObbHalfspaceNarrowphase, GpuObbNarrowphase,
    GpuObbObbNarrowphase, Obb, ObbObbPair, ObbPlanePair, Plane, SphereCapsulePair, SphereObbPair,
    SpherePlanePair,
};
pub use radix::{cpu_radix_sort_keys, cpu_radix_sort_pairs, GpuRadixSort};
pub use scan::{cpu_compact, cpu_exclusive_scan, GpuScan};
pub use xpbd::{
    cpu_solve, Colouring, DistanceConstraint, GpuXpbdSolver, ParticleState, XpbdConfig, XpbdError,
};
