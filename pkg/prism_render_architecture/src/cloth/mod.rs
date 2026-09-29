//! Cloth / garment subsystem contracts (AAA XPBD cloth engine).
//!
//! Cloth is a full triangle-mesh garment engine and a first-class subsystem,
//! separate from hair: the primitive is a *triangle-mesh particle*, not a
//! strand. The pipeline mirrors production cloth engines (UE5 `Chaos` Cloth,
//! NVIDIA `NvCloth` / `PhysX` Clothing, `Marvelous` Designer / CLO, Havok Cloth,
//! Houdini `Vellum`, and the SIGGRAPH 2024 VBD solver) at the algorithm level,
//! without reusing any of their code:
//!
//! 1. **Asset & seaming** — 2D panels stitched into a 3D garment plus woven
//!    material parameters (warp/weft stiffness, bend, density, drag) and paint
//!    constraints (max-distance / backstop / blend-weight / anim-drive); see
//!    [`asset`].
//! 2. **Constraint graph** — stretch (warp/weft anisotropic), bend, shear,
//!    long-range attachment (LRA) and tether constraints, graph-colored into
//!    batches that are parallel within a color and serial across colors; see
//!    [`constraints`].
//! 3. **Dynamics** — an XPBD substepping solver (predict, project per color
//!    batch, update velocity, strain-limit) parameterized by compliance; see
//!    [`dynamics`]. The per-frame vertex work is arbitrated by
//!    [`crate::deformation::schedule`]; this subsystem only emits requests, it
//!    does not own the budget.
//! 4. **Collision** — body proxies (sphere / capsule / plane), self-collision
//!    via a spatial hash with continuous detection, and backstop planes; see
//!    [`collision`].
//! 5. **LOD** — a ladder from full simulation to reduced simulation to a
//!    skinned proxy, selected by screen coverage and clamped by an authored
//!    native form; see [`lod`].
//! 6. **Embedding** — render-mesh vertices follow the coarse sim mesh through
//!    barycentric embedding; see [`embed`].
//!
//! This module owns the geometry, LOD, and simulation-binding contracts and the
//! data types shared across the sibling modules. It references, never
//! reimplements, the shared deformation budget, the material `cloth` shading
//! closures, and the transparency (OIT) routing.

pub mod asset;
pub mod bending;
pub mod collision;
pub mod constraints;
pub mod dynamics;
pub mod embed;
pub mod lod;
pub mod pipeline;
pub mod pressure;
pub mod sleep;
pub mod vbd;
pub mod wind;

use alloc::vec::Vec;

use crate::deformation::DeformationHandle;

/// Squared-length threshold below which a vector is treated as zero, so
/// normalization and constraint projection never divide by (near) zero and
/// never propagate `NaN`.
pub const EPS_LEN_SQ: f32 = 1e-12;

/// A hand-rolled three-component vector.
///
/// `prism_render_architecture` is a dependency-free contracts crate, so the
/// cloth math is spelled out here rather than pulled from a linear-algebra
/// dependency. Only `sqrt` is used (allowed by the workspace determinism lint);
/// no transcendental functions are called.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The cloth math API is specified with named add/sub methods for call-site uniformity; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product `self × rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] when only
    /// comparisons are needed.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Squared distance between two points.
    #[must_use]
    pub fn distance_squared(self, rhs: Self) -> f32 {
        self.sub(rhs).length_squared()
    }

    /// Euclidean distance between two points.
    #[must_use]
    pub fn distance(self, rhs: Self) -> f32 {
        self.sub(rhs).length()
    }

    /// Returns the unit vector along `self`, or [`Vec3::ZERO`] when `self` is
    /// (numerically) the zero vector, so normalization never yields `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > EPS_LEN_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }
}

/// Identifies one cloth piece: a garment bound to a skinned body.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ClothPieceHandle(pub u32);

/// Discrete cloth level-of-detail tiers, coarsening with distance / coverage.
///
/// The first two tiers stay simulation-based; [`ClothLodTier::SkinnedProxy`] is
/// a static skinned shell that skips cloth dynamics entirely.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ClothLodTier {
    /// Full-resolution simulation with self-collision.
    FullSim,
    /// Reduced-resolution simulation; weak or no self-collision.
    ReducedSim,
    /// Pure skinned shell that does not simulate.
    SkinnedProxy,
}

impl ClothLodTier {
    /// Returns `true` when this tier runs the cloth solver and therefore drives
    /// deformation through the shared budget.
    #[must_use]
    pub fn is_simulated(self) -> bool {
        matches!(self, ClothLodTier::FullSim | ClothLodTier::ReducedSim)
    }

    /// Coarseness rank: `0` is the finest (full sim), `2` the coarsest (skinned
    /// proxy). Used to clamp a coverage-selected tier so a garment never
    /// simulates finer than the geometry it was authored with.
    #[must_use]
    pub fn coarseness(self) -> u8 {
        match self {
            ClothLodTier::FullSim => 0,
            ClothLodTier::ReducedSim => 1,
            ClothLodTier::SkinnedProxy => 2,
        }
    }

    /// Returns whichever of the two tiers is coarser (higher coarseness rank).
    #[must_use]
    pub fn coarser_of(self, other: ClothLodTier) -> ClothLodTier {
        if other.coarseness() > self.coarseness() {
            other
        } else {
            self
        }
    }
}

/// Authoring description of one cloth piece at its finest LOD.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClothPiece {
    /// Stable identity of this garment.
    pub handle: ClothPieceHandle,
    /// Simulated sim-mesh vertices; the solve cost and the deformation charge.
    pub sim_vertex_count: u32,
    /// Render-mesh vertices embedded into the sim mesh; drives raster cost.
    pub render_vertex_count: u32,
    /// Constraints in the sim-mesh constraint graph at the finest tier.
    pub constraint_count: u32,
    /// Deformation-cache entry that cloth dynamics writes into.
    pub deformation: DeformationHandle,
    /// The finest representation this garment actually has geometry for.
    ///
    /// A fully authored garment sets [`ClothLodTier::FullSim`] and uses the
    /// whole ladder. A background NPC outfit authored to only ever skin sets
    /// [`ClothLodTier::SkinnedProxy`], so LOD never promotes it to a simulation
    /// it does not own. This is an authoring (geometry) choice, orthogonal to
    /// the PBR/NPR shading response, so both styles honor it identically.
    pub native_form: ClothLodTier,
}

/// One simulated cloth particle (a sim-mesh vertex).
///
/// XPBD stores an explicit velocity: the integrator predicts a position from
/// the current position and velocity, projects constraints against the
/// prediction, then recovers velocity from the position delta. A particle with
/// `inverse_mass == 0` (equivalently non-positive) is *pinned*: infinitely
/// heavy, never moved by integration or any constraint, which is how attachment
/// points, waistbands, and anim-driven vertices are modeled.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClothParticle {
    /// World-space position.
    pub position: Vec3,
    /// World-space velocity.
    pub velocity: Vec3,
    /// Inverse mass (`1/m`); `0` pins the particle.
    pub inverse_mass: f32,
}

impl ClothParticle {
    /// Builds a movable particle with the given inverse mass.
    #[must_use]
    pub fn new(position: Vec3, inverse_mass: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            inverse_mass: inverse_mass.max(0.0),
        }
    }

    /// Builds a pinned particle (`inverse_mass == 0`) at a fixed position.
    #[must_use]
    pub fn pinned(position: Vec3) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            inverse_mass: 0.0,
        }
    }

    /// Returns `true` when the particle is pinned and must never move.
    #[must_use]
    pub fn is_pinned(self) -> bool {
        self.inverse_mass <= 0.0
    }
}

/// XPBD compliance `α`, the inverse of stiffness (`0` is perfectly rigid).
///
/// Compliance makes the solver step-size independent: the effective constraint
/// stiffness is `1 / (compliance + tiny)` and higher compliance yields a softer
/// response. A woven fabric sets small compliance along warp/weft (stiff) and
/// larger compliance for bending (floppy).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Compliance(pub f32);

impl Compliance {
    /// A perfectly rigid constraint.
    pub const RIGID: Self = Self(0.0);

    /// Returns the non-negative compliance value (negative inputs clamp to 0).
    #[must_use]
    pub fn value(self) -> f32 {
        self.0.max(0.0)
    }
}

/// The kind of a two-particle distance constraint in the cloth graph.
///
/// Every cloth constraint in this engine is expressed as a positional distance
/// constraint between two sim-mesh particles, which keeps one solver kernel and
/// one graph-coloring routine covering the whole ladder:
///
/// * [`ConstraintKind::Stretch`] — a structural warp/weft grid edge; the woven
///   anisotropy lives in per-edge compliance.
/// * [`ConstraintKind::Bend`] — a cross-diagonal distance constraint spanning
///   the two triangles that share an interior edge (a distance-based bending
///   model; the full dihedral formulation is a future high-fidelity slot).
/// * [`ConstraintKind::Shear`] — a quad-diagonal constraint resisting in-plane
///   shear.
/// * [`ConstraintKind::Lra`] — a long-range attachment capping a particle's
///   distance to a pinned anchor, preventing stretch under fast motion.
/// * [`ConstraintKind::Tether`] — a one-sided leash to an anchor that only
///   pulls when over-extended.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ConstraintKind {
    /// Structural warp/weft edge.
    Stretch,
    /// Cross-diagonal bending resistance.
    Bend,
    /// Quad-diagonal shear resistance.
    Shear,
    /// Long-range attachment to an anchor.
    Lra,
    /// One-sided tether leash to an anchor.
    Tether,
}

impl ConstraintKind {
    /// Returns `true` for constraints that only pull when over-extended and
    /// never push when compressed (LRA and tether leashes). The solver skips
    /// projection for these when the current distance is within the rest
    /// length, so an anchor never yanks slack cloth inward.
    #[must_use]
    pub fn is_one_sided(self) -> bool {
        matches!(self, ConstraintKind::Lra | ConstraintKind::Tether)
    }
}

/// A two-particle positional distance constraint.
///
/// `a` and `b` index into the particle array. The constraint drives the
/// distance between the two particles toward `rest_length`; `compliance`
/// softens it (XPBD) and `kind` tags it for one-sided handling and for
/// per-kind diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constraint {
    /// First particle index.
    pub a: u32,
    /// Second particle index.
    pub b: u32,
    /// Target rest distance between the two particles.
    pub rest_length: f32,
    /// XPBD compliance for this constraint.
    pub compliance: Compliance,
    /// Constraint category.
    pub kind: ConstraintKind,
}

impl Constraint {
    /// Builds a constraint between two particle indices.
    #[must_use]
    pub fn new(
        a: u32,
        b: u32,
        rest_length: f32,
        compliance: Compliance,
        kind: ConstraintKind,
    ) -> Self {
        Self {
            a,
            b,
            rest_length,
            compliance,
            kind,
        }
    }
}

/// A contiguous run of constraints that share no particle and can therefore be
/// projected in parallel (Jacobi within a color); colors are applied in order
/// (Gauss-Seidel across colors).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ColorBatch {
    /// Index of the first constraint of this batch in [`ConstraintGraph::constraints`].
    pub start: u32,
    /// Number of constraints in this batch.
    pub len: u32,
}

impl ColorBatch {
    /// The half-open range `start..start+len` as `usize` for slicing.
    #[must_use]
    pub fn range(self) -> core::ops::Range<usize> {
        let start = self.start as usize;
        start..start + self.len as usize
    }
}

/// A graph-colored constraint set ready for the solver.
///
/// `constraints` is reordered so that every [`ColorBatch`] in `batches` names a
/// contiguous run whose constraints touch pairwise-disjoint particles. The
/// solver walks `batches` in order; within a batch, projections are
/// independent and map cleanly onto a GPU dispatch. The ordering is fully
/// deterministic given a fixed input constraint list.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintGraph {
    /// Constraints reordered so each color batch is contiguous.
    pub constraints: Vec<Constraint>,
    /// Color batches in application order.
    pub batches: Vec<ColorBatch>,
}

impl ConstraintGraph {
    /// Total constraints across every color.
    #[must_use]
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Number of colors (serial solver passes).
    #[must_use]
    pub fn color_count(&self) -> usize {
        self.batches.len()
    }

    /// Returns `true` when there are no constraints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// The constraints belonging to a given color batch, or an empty slice when
    /// the batch range falls outside the constraint array (never panics).
    #[must_use]
    pub fn batch(&self, batch: ColorBatch) -> &[Constraint] {
        let range = batch.range();
        self.constraints.get(range).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_basic_algebra_is_exact() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.add(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b.sub(a), Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a.dot(b), 32.0);
    }

    #[test]
    fn vec3_cross_is_right_handed() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn vec3_length_and_distance() {
        let a = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(a.length_squared(), 25.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(Vec3::ZERO.distance(a), 5.0);
    }

    #[test]
    fn normalize_zero_vector_is_zero_not_nan() {
        let n = Vec3::ZERO.normalize_or_zero();
        assert_eq!(n, Vec3::ZERO);
        let unit = Vec3::new(0.0, 5.0, 0.0).normalize_or_zero();
        assert_eq!(unit, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn tier_simulation_flag() {
        assert!(ClothLodTier::FullSim.is_simulated());
        assert!(ClothLodTier::ReducedSim.is_simulated());
        assert!(!ClothLodTier::SkinnedProxy.is_simulated());
    }

    #[test]
    fn tier_coarseness_ranks_finest_to_coarsest() {
        assert_eq!(ClothLodTier::FullSim.coarseness(), 0);
        assert_eq!(ClothLodTier::ReducedSim.coarseness(), 1);
        assert_eq!(ClothLodTier::SkinnedProxy.coarseness(), 2);
    }

    #[test]
    fn coarser_of_returns_higher_rank() {
        assert_eq!(
            ClothLodTier::FullSim.coarser_of(ClothLodTier::SkinnedProxy),
            ClothLodTier::SkinnedProxy
        );
        assert_eq!(
            ClothLodTier::SkinnedProxy.coarser_of(ClothLodTier::FullSim),
            ClothLodTier::SkinnedProxy
        );
        assert_eq!(
            ClothLodTier::ReducedSim.coarser_of(ClothLodTier::ReducedSim),
            ClothLodTier::ReducedSim
        );
    }

    #[test]
    fn particle_pinning_is_detected() {
        let free = ClothParticle::new(Vec3::ZERO, 2.0);
        assert!(!free.is_pinned());
        assert!(ClothParticle::pinned(Vec3::ZERO).is_pinned());
        // Negative inverse mass clamps to pinned rather than misbehaving.
        assert!(ClothParticle::new(Vec3::ZERO, -1.0).is_pinned());
    }

    #[test]
    fn compliance_clamps_negative_to_zero() {
        assert_eq!(Compliance(-3.0).value(), 0.0);
        assert_eq!(Compliance::RIGID.value(), 0.0);
        assert_eq!(Compliance(0.25).value(), 0.25);
    }

    #[test]
    fn one_sided_constraint_kinds() {
        assert!(ConstraintKind::Lra.is_one_sided());
        assert!(ConstraintKind::Tether.is_one_sided());
        assert!(!ConstraintKind::Stretch.is_one_sided());
        assert!(!ConstraintKind::Bend.is_one_sided());
        assert!(!ConstraintKind::Shear.is_one_sided());
    }

    #[test]
    fn graph_batch_slicing_is_bounds_safe() {
        let graph = ConstraintGraph {
            constraints: vec![
                Constraint::new(0, 1, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
                Constraint::new(2, 3, 1.0, Compliance::RIGID, ConstraintKind::Stretch),
            ],
            batches: vec![ColorBatch { start: 0, len: 2 }],
        };
        assert_eq!(graph.constraint_count(), 2);
        assert_eq!(graph.color_count(), 1);
        assert_eq!(graph.batch(ColorBatch { start: 0, len: 2 }).len(), 2);
        // Out-of-range batch yields an empty slice instead of panicking.
        assert!(graph.batch(ColorBatch { start: 5, len: 3 }).is_empty());
    }
}
