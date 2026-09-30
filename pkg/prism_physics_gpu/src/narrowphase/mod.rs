//! Narrow phase: candidate pairs and shape colliders to contact manifolds.
//!
//! The broad phase ([`crate::broadphase`]) prunes the `O(n^2)` collision test to
//! a set of candidate pairs whose bounding volumes *might* touch. The narrow
//! phase closes the remaining gap: it takes each candidate and produces the
//! precise contact a solver needs — a unit normal, a penetration depth, and a
//! world contact point — or rejects it when the shapes do not actually
//! penetrate. This is the missing link between the broad phase and the `XPBD`
//! constraint solver ([`crate::xpbd`]).
//!
//! # Shape library
//!
//! The narrow phase is organised one collider pair per module, each a
//! self-contained vertical slice: a shared geometry function, a `CPU` golden
//! twin, a `WGSL` kernel, and a device pipeline that runs the identical
//! arithmetic.
//!
//! - [`sphere`] / [`GpuNarrowphase`]: sphere versus sphere, the dynamic-dynamic
//!   primitive driving particle-particle collision.
//! - [`halfspace`] / [`GpuHalfspaceNarrowphase`]: sphere versus an infinite
//!   plane, the canonical static collider for grounds, walls, and frustum faces.
//! - [`capsule`] / [`GpuCapsuleNarrowphase`]: sphere versus a capsule (a segment
//!   swept by a radius), collapsing to sphere-sphere against the closest point
//!   on the segment.
//! - [`capsule_capsule`] / [`GpuCapsuleCapsuleNarrowphase`]: capsule versus
//!   capsule (dynamic-dynamic), finding the closest point pair between the two
//!   segments and collapsing to sphere-sphere there.
//! - [`obb`] / [`GpuObbNarrowphase`]: sphere versus an oriented bounding box,
//!   clamping the sphere centre in the box frame with an interior push-out
//!   fallback through the least-penetrated face.
//! - [`obb_halfspace`] / [`GpuObbHalfspaceNarrowphase`]: oriented bounding box
//!   versus a halfspace, using the box support function along the plane normal
//!   to report the deepest penetrating vertex.
//! - [`obb_obb`] / [`GpuObbObbNarrowphase`]: oriented bounding box versus
//!   oriented bounding box (dynamic-dynamic), a fifteen-axis separating-axis
//!   test reporting the minimum-translation contact.
//! - [`obb_obb_manifold`] / [`cpu_obb_obb_manifold`]: promotes the OBB-OBB
//!   contact to a multi-point [`ContactManifold`] by clipping the incident face
//!   against the reference face, the manifold a solver needs for stable
//!   stacking.
//!
//! Every pair emits one contact slot per input candidate: a passing real-device
//! parity test is direct evidence the ported kernel builds the same manifolds as
//! its twin.
//!
//! # Dense output
//!
//! Emitting a slot per candidate (rather than compacting on the fly) keeps the
//! contact index aligned with the input index, which the parity tests rely on
//! and which lets a downstream [`crate::scan`] compaction stream the survivors
//! without a second pass. It is a deliberate design choice, not a stub: the
//! `None` / invalid slots carry a real "no penetration" decision.
//!
//! Provenance: textbook collision-manifold construction; no Unreal Engine source
//! or derived code.

mod capsule;
mod capsule_capsule;
mod capsule_capsule_gpu;
mod capsule_gpu;
mod contact;
mod cpu;
mod gpu;
mod halfspace;
mod halfspace_gpu;
mod layout;
mod manifold;
mod obb;
mod obb_gpu;
mod obb_halfspace;
mod obb_halfspace_gpu;
mod obb_obb;
mod obb_obb_gpu;
mod obb_obb_manifold;
mod obb_obb_manifold_gpu;
mod sphere;

pub use capsule::{cpu_capsule_narrowphase, Capsule, SphereCapsulePair};
pub use capsule_capsule::{cpu_capsule_capsule_narrowphase, CapsuleCapsulePair};
pub use capsule_capsule_gpu::GpuCapsuleCapsuleNarrowphase;
pub use capsule_gpu::GpuCapsuleNarrowphase;
pub use contact::Contact;
pub use cpu::cpu_narrowphase;
pub use gpu::GpuNarrowphase;
pub use halfspace::{cpu_halfspace_narrowphase, Plane, SpherePlanePair};
pub use halfspace_gpu::GpuHalfspaceNarrowphase;
pub use manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};
pub use obb::{cpu_obb_narrowphase, Obb, SphereObbPair};
pub use obb_gpu::GpuObbNarrowphase;
pub use obb_halfspace::{cpu_obb_halfspace_narrowphase, ObbPlanePair};
pub use obb_halfspace_gpu::GpuObbHalfspaceNarrowphase;
pub use obb_obb::{cpu_obb_obb_narrowphase, ObbObbPair};
pub use obb_obb_gpu::GpuObbObbNarrowphase;
pub use obb_obb_manifold::cpu_obb_obb_manifold;
pub use obb_obb_manifold_gpu::GpuObbObbManifoldNarrowphase;
