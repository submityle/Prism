//! `XPBD`/`VBD` constraint-solve contracts and large-scale fracture data
//! (design §10).
//!
//! Position-based dynamics is the shared kinematic backbone the particle
//! subsystem leans on for cloth-like sheets, ropes, soft bodies, and rigid
//! debris. Production stacks — `Houdini` Vellum, `Niagara`'s physics modules,
//! and `EmberGen`'s destruction layer — all express these effects as a *set of
//! constraints* projected over a few substeps, and this module is the `CPU`-
//! verifiable contract for the *data* those solves consume and produce. It does
//! **not** reimplement the shared position solver kernel: it models the
//! constraint primitives (their compliance, rest value, and participating
//! particles), the `XPBD` substep projection math, the batching handshake with
//! the graph-colored [`ConstraintColoring`] already produced by
//! [`super::stages`], the `VBD` (Vertex Block Descent) high-stability tier and
//! its selection matrix, the tearing threshold that drives a topology rebuild,
//! and the fracture-chunk / rigid-approximation contract for mass destruction.
//!
//! Everything here is pure and deterministic: only ordinary arithmetic and
//! `sqrt` (through the hand-rolled [`Vec3`] math) are used, so a future `GPU`
//! kernel that projects the same constraints in the same color order produces
//! bit-identical corrections. Distance constraints reuse
//! [`super::stages::ParticleConstraint`] and its coloring verbatim so the
//! particle and cloth subsystems batch identically.

use alloc::vec::Vec;

use super::stages::{
    color_particle_constraints, ColorBatch, ConstraintColoring, ParticleConstraint,
};
use super::{Vec3, EPS_LEN_SQ};

/// Small positive guard against division by a (near) zero denominator.
///
/// Distinct from [`EPS_LEN_SQ`], which guards *squared lengths*; this guards
/// scalar quantities such as an inverse-mass sum, a substep `dt`, or a rest
/// length before it appears in a denominator.
const EPS: f32 = 1e-9;

/// Finite mass assigned to a particle whose inverse mass is (numerically) zero.
///
/// An inverse mass of zero models an immovable / kinematic particle. When such
/// a particle contributes to a fracture chunk's rigid approximation it must
/// still carry weight in the center-of-mass average, so it is treated as very
/// heavy rather than massless.
const STATIC_MASS: f32 = 1.0e9;

/// The four constraint primitives the particle solver projects (design §10).
///
/// Each maps to a scalar constraint function `C` whose zero the solver drives
/// the positions toward; the variants differ only in how many particles they
/// touch and how `C` is formed, not in the projection machinery.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ConstraintKind {
    /// Two-particle separation constraint (`C = |a - b| - rest`).
    Distance,
    /// Bending / dihedral constraint over an edge-adjacent particle stencil.
    Bend,
    /// Volume-preservation constraint over a tetrahedron of four particles.
    Volume,
    /// Direction-dependent stiffness: stiff along an axis, compliant across it.
    Anisotropic,
}

/// Inverse-stiffness (compliance) of a constraint, `α = 1 / stiffness`.
///
/// `XPBD` replaces a stiffness coefficient with its inverse so a perfectly
/// rigid constraint is the finite value `α = 0` rather than an infinite
/// stiffness. During a substep the raw compliance is scaled by `1 / dt²`
/// (see [`Compliance::scaled_for_substep`]) to make the effective stiffness
/// independent of the substep count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Compliance {
    /// Inverse stiffness in metres-per-newton units; `0` is perfectly rigid.
    pub alpha: f32,
}

impl Compliance {
    /// A perfectly rigid constraint (`α = 0`).
    pub const RIGID: Self = Self { alpha: 0.0 };

    /// Wraps a raw compliance value, clamping negatives to `0` (rigid).
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: if alpha > 0.0 { alpha } else { 0.0 },
        }
    }

    /// Builds a compliance from a stiffness, `α = 1 / stiffness`.
    ///
    /// A non-positive stiffness is treated as infinitely stiff and yields the
    /// rigid compliance `α = 0`.
    #[must_use]
    pub fn from_stiffness(stiffness: f32) -> Self {
        if stiffness > EPS {
            Self {
                alpha: 1.0 / stiffness,
            }
        } else {
            Self::RIGID
        }
    }

    /// Returns `true` when this constraint is (numerically) rigid.
    #[must_use]
    pub fn is_rigid(self) -> bool {
        self.alpha <= EPS
    }

    /// The substep-scaled compliance `α̃ = α / dt²` used in the projection
    /// denominator and Lagrange-multiplier regularizer.
    ///
    /// Scaling by `1 / dt²` per substep is what makes an `XPBD` material's
    /// stiffness independent of how many substeps are taken. A non-positive
    /// `substep_dt` is degenerate and yields `0` (treated as rigid) rather than
    /// dividing by zero.
    #[must_use]
    pub fn scaled_for_substep(self, substep_dt: f32) -> f32 {
        let dt2 = substep_dt * substep_dt;
        if dt2 > EPS {
            self.alpha / dt2
        } else {
            0.0
        }
    }
}

/// Free-function form of [`Compliance::scaled_for_substep`] for call sites that
/// already hold a raw `α`.
///
/// Returns `α̃ = α / dt²`, guarding a non-positive `substep_dt` by returning
/// `0` (rigid).
#[must_use]
pub fn effective_compliance(alpha: f32, substep_dt: f32) -> f32 {
    Compliance::new(alpha).scaled_for_substep(substep_dt)
}

/// Maximum number of particles a single constraint primitive references.
///
/// Distance / anisotropic constraints use two, while bend and volume
/// constraints use a four-particle stencil; the fixed-size array keeps the
/// contract `alloc`-free per primitive.
pub const MAX_CONSTRAINT_PARTICLES: usize = 4;

/// A fully described constraint primitive: kind, participants, rest value, and
/// compliance (design §10).
///
/// `rest_value` is interpreted per [`ConstraintKind`]: a rest length for
/// [`ConstraintKind::Distance`] and [`ConstraintKind::Anisotropic`], a rest
/// dihedral measure for [`ConstraintKind::Bend`], and a rest (signed) volume for
/// [`ConstraintKind::Volume`]. `axis` is meaningful only for the anisotropic
/// kind and is [`Vec3::ZERO`] otherwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintPrimitive {
    /// Which primitive this is.
    pub kind: ConstraintKind,
    /// Participating particle indices; only the first `particle_count` are live.
    pub particles: [u32; MAX_CONSTRAINT_PARTICLES],
    /// Number of live entries in `particles`.
    pub particle_count: u8,
    /// Kind-dependent rest target (length, dihedral, or volume).
    pub rest_value: f32,
    /// Inverse stiffness of this constraint.
    pub compliance: Compliance,
    /// Stiff direction for [`ConstraintKind::Anisotropic`]; else [`Vec3::ZERO`].
    pub axis: Vec3,
}

impl ConstraintPrimitive {
    /// Builds a two-particle distance constraint.
    #[must_use]
    pub fn distance(a: u32, b: u32, rest_length: f32, compliance: Compliance) -> Self {
        Self {
            kind: ConstraintKind::Distance,
            particles: [a, b, 0, 0],
            particle_count: 2,
            rest_value: rest_length,
            compliance,
            axis: Vec3::ZERO,
        }
    }

    /// Builds a four-particle bending constraint over an edge-adjacent stencil.
    #[must_use]
    pub fn bend(a: u32, b: u32, c: u32, d: u32, rest: f32, compliance: Compliance) -> Self {
        Self {
            kind: ConstraintKind::Bend,
            particles: [a, b, c, d],
            particle_count: 4,
            rest_value: rest,
            compliance,
            axis: Vec3::ZERO,
        }
    }

    /// Builds a four-particle tetrahedral volume-preservation constraint.
    #[must_use]
    pub fn volume(
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        rest_volume: f32,
        compliance: Compliance,
    ) -> Self {
        Self {
            kind: ConstraintKind::Volume,
            particles: [a, b, c, d],
            particle_count: 4,
            rest_value: rest_volume,
            compliance,
            axis: Vec3::ZERO,
        }
    }

    /// Builds a two-particle anisotropic constraint stiff along `axis`.
    ///
    /// The axis is normalized (a zero axis collapses to isotropic, i.e.
    /// [`Vec3::ZERO`]).
    #[must_use]
    pub fn anisotropic(
        a: u32,
        b: u32,
        axis: Vec3,
        rest_length: f32,
        compliance: Compliance,
    ) -> Self {
        Self {
            kind: ConstraintKind::Anisotropic,
            particles: [a, b, 0, 0],
            particle_count: 2,
            rest_value: rest_length,
            compliance,
            axis: axis.normalize_or_zero(),
        }
    }

    /// The live participating-particle indices.
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        let count = self.particle_count as usize;
        &self.particles[..count]
    }

    /// Lowers a distance constraint to the shared [`ParticleConstraint`] so it
    /// can be graph-colored by [`super::stages`]; other kinds return [`None`].
    #[must_use]
    pub fn as_particle_constraint(&self) -> Option<ParticleConstraint> {
        match self.kind {
            ConstraintKind::Distance => Some(ParticleConstraint::new(
                self.particles[0],
                self.particles[1],
                self.rest_value,
                self.compliance.alpha,
            )),
            ConstraintKind::Bend | ConstraintKind::Volume | ConstraintKind::Anisotropic => None,
        }
    }
}

/// Substep / iteration schedule for one `XPBD` frame (design §10).
///
/// A frame's timestep is split into `substeps` equal substeps, and each substep
/// runs `iterations` solver sweeps over the colored constraint batches. More
/// substeps (rather than more iterations) is the `XPBD`-recommended way to
/// stiffen a material without changing its rest behavior, because the per-
/// substep compliance scaling absorbs the smaller `dt`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubstepSchedule {
    /// Full frame timestep in seconds.
    pub dt: f32,
    /// Number of equal substeps per frame.
    pub substeps: u32,
    /// Solver sweeps over the colored batches per substep.
    pub iterations: u32,
}

impl SubstepSchedule {
    /// Builds a substep schedule.
    #[must_use]
    pub fn new(dt: f32, substeps: u32, iterations: u32) -> Self {
        Self {
            dt,
            substeps,
            iterations,
        }
    }

    /// Per-substep timestep `dt / substeps`.
    ///
    /// Zero substeps is degenerate and yields `0` rather than dividing by zero.
    #[must_use]
    pub fn substep_dt(self) -> f32 {
        if self.substeps == 0 {
            0.0
        } else {
            self.dt / self.substeps as f32
        }
    }

    /// Substep-scaled compliance `α̃` for a constraint under this schedule.
    #[must_use]
    pub fn effective_compliance(self, compliance: Compliance) -> f32 {
        compliance.scaled_for_substep(self.substep_dt())
    }
}

/// The position correction and Lagrange-multiplier increment produced by one
/// `XPBD` distance projection.
///
/// `delta_a` / `delta_b` are added to the two particle positions and
/// `delta_lambda` is accumulated into the constraint's running multiplier `λ`,
/// which the next iteration feeds back through the compliance regularizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceCorrection {
    /// Position delta for particle `a`.
    pub delta_a: Vec3,
    /// Position delta for particle `b`.
    pub delta_b: Vec3,
    /// Increment to the constraint's accumulated Lagrange multiplier.
    pub delta_lambda: f32,
}

impl DistanceCorrection {
    /// A no-op correction (used for degenerate / already-satisfied inputs).
    pub const ZERO: Self = Self {
        delta_a: Vec3::ZERO,
        delta_b: Vec3::ZERO,
        delta_lambda: 0.0,
    };
}

/// Projects a single `XPBD` distance constraint, returning the position deltas
/// and multiplier increment (design §10).
///
/// Implements the standard compliant projection: with constraint value
/// `C = |a - b| - rest`, gradient `n = (a - b) / |a - b|`, inverse-mass sum
/// `w = w_a + w_b`, and substep-scaled compliance `α̃`, the multiplier
/// increment is
///
/// `Δλ = (−C − α̃·λ) / (w + α̃)`,
///
/// and the position deltas are `Δa = w_a·Δλ·n`, `Δb = −w_b·Δλ·n`. A degenerate
/// separation (endpoints coincident) or a zero effective inverse-mass sum
/// yields [`DistanceCorrection::ZERO`], so the projection never divides by zero
/// and never emits `NaN`. Only `sqrt` (via [`Vec3`]) is used.
#[must_use]
pub fn project_distance(
    pos_a: Vec3,
    pos_b: Vec3,
    inv_mass_a: f32,
    inv_mass_b: f32,
    rest: f32,
    alpha_tilde: f32,
    lambda: f32,
) -> DistanceCorrection {
    let delta = pos_a.sub(pos_b);
    let len_sq = delta.length_squared();
    if len_sq <= EPS_LEN_SQ {
        return DistanceCorrection::ZERO;
    }
    let len = len_sq.sqrt();
    let normal = delta.scale(1.0 / len);
    let c = len - rest;
    let denom = inv_mass_a + inv_mass_b + alpha_tilde;
    if denom <= EPS {
        return DistanceCorrection::ZERO;
    }
    let delta_lambda = (-c - alpha_tilde * lambda) / denom;
    DistanceCorrection {
        delta_a: normal.scale(inv_mass_a * delta_lambda),
        delta_b: normal.scale(-inv_mass_b * delta_lambda),
        delta_lambda,
    }
}

/// Reorders the distance constraints of `primitives` into graph-colored batches
/// via [`super::stages::color_particle_constraints`].
///
/// Non-distance primitives are skipped (they are projected on their own stencil
/// schedules); the surviving distance constraints are lowered to
/// [`ParticleConstraint`] and colored so that, within a color, no two
/// constraints touch a shared particle and their projections can run in
/// parallel without a write conflict. The batching is identical to the cloth
/// kernel's, so replay and networked simulation agree.
#[must_use]
pub fn color_distance_constraints(primitives: &[ConstraintPrimitive]) -> ConstraintColoring {
    let mut lowered = Vec::new();
    for primitive in primitives {
        if let Some(constraint) = primitive.as_particle_constraint() {
            lowered.push(constraint);
        }
    }
    color_particle_constraints(&lowered)
}

/// Borrows the constraints of one color [`ColorBatch`] out of a coloring.
///
/// A convenience over [`ConstraintColoring`]'s flat `constraints` list: a
/// scheduler dispatches one color at a time, and this returns the contiguous
/// run for `batch` so a per-color solve iterates exactly its members.
#[must_use]
pub fn batch_constraints(
    coloring: &ConstraintColoring,
    batch: ColorBatch,
) -> &[ParticleConstraint] {
    let start = batch.start as usize;
    let end = start + batch.len as usize;
    &coloring.constraints[start..end]
}

/// The solver tier chosen for a constraint island (design §10).
///
/// The particle solver exposes two position-solve backends behind one
/// contract: the default `XPBD` Gauss-Seidel-over-colors path and a higher-
/// stability `VBD` (Vertex Block Descent) path. This module models the *choice*
/// and the convergence contract; it does not implement either kernel.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SolverTier {
    /// Extended Position-Based Dynamics: cheap, parallel over graph colors,
    /// but only conditionally stable at extreme stiffness / contact density.
    Xpbd,
    /// Vertex Block Descent: an energy-descent solve that is unconditionally
    /// stable and robust under stiff, densely-coupled contacts at higher cost.
    Vbd,
}

/// The selection matrix that picks a [`SolverTier`] for an island (design §10).
///
/// `XPBD` is the default; the solver escalates to `VBD` when the material is
/// very stiff (its stiffness ratio exceeds `stiff_ratio_threshold`), when
/// contacts are densely coupled (average constraints-per-particle exceed
/// `contact_density_threshold`) so Gauss-Seidel converges slowly, or when the
/// caller demands high stability outright.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverSelection {
    /// Stiffness ratio above which `VBD` is preferred for stability.
    pub stiff_ratio_threshold: f32,
    /// Constraints-per-particle above which `VBD` is preferred.
    pub contact_density_threshold: f32,
}

impl SolverSelection {
    /// Builds a selection matrix from explicit thresholds.
    #[must_use]
    pub fn new(stiff_ratio_threshold: f32, contact_density_threshold: f32) -> Self {
        Self {
            stiff_ratio_threshold,
            contact_density_threshold,
        }
    }

    /// A balanced default: escalate past a `1000x` stiffness ratio or an
    /// average of eight constraints per particle.
    #[must_use]
    pub fn standard() -> Self {
        Self::new(1000.0, 8.0)
    }

    /// Chooses the tier for an island.
    ///
    /// Returns [`SolverTier::Vbd`] when `force_high_stability` is set, when
    /// `stiffness_ratio` exceeds the stiffness threshold, or when
    /// `contact_density` exceeds the density threshold; otherwise
    /// [`SolverTier::Xpbd`].
    #[must_use]
    pub fn select(
        self,
        stiffness_ratio: f32,
        contact_density: f32,
        force_high_stability: bool,
    ) -> SolverTier {
        let stiff = stiffness_ratio > self.stiff_ratio_threshold;
        let dense = contact_density > self.contact_density_threshold;
        if force_high_stability || stiff || dense {
            SolverTier::Vbd
        } else {
            SolverTier::Xpbd
        }
    }
}

/// The convergence characteristics a [`SolverTier`] contracts for (design §10).
///
/// A scheduler uses this to budget iterations and decide whether a color-
/// parallel dispatch is safe: `XPBD` is only conditionally stable but parallel
/// across colors and cheap, whereas `VBD` is unconditionally stable and block-
/// parallel at a higher per-iteration cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvergenceContract {
    /// The tier this contract describes.
    pub tier: SolverTier,
    /// Whether the tier is stable for any timestep / stiffness.
    pub unconditionally_stable: bool,
    /// Whether iterations parallelize across graph colors / vertex blocks.
    pub parallelizable: bool,
    /// Relative per-iteration cost against the `XPBD` baseline of `1`.
    pub cost_multiplier: f32,
}

impl ConvergenceContract {
    /// The convergence contract for a tier.
    #[must_use]
    pub fn for_tier(tier: SolverTier) -> Self {
        match tier {
            SolverTier::Xpbd => Self {
                tier,
                unconditionally_stable: false,
                parallelizable: true,
                cost_multiplier: 1.0,
            },
            SolverTier::Vbd => Self {
                tier,
                unconditionally_stable: true,
                parallelizable: true,
                cost_multiplier: 3.0,
            },
        }
    }
}

/// The strain threshold above which a constraint tears (design §10).
///
/// Strain is the relative stretch `(|a - b| − rest) / rest`; once it exceeds
/// `max_strain` the constraint is marked broken and a [`TopologyUpdateRequest`]
/// is emitted so the island's connectivity can be rebuilt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TearThreshold {
    /// Maximum tolerated relative stretch before the constraint breaks.
    pub max_strain: f32,
}

impl TearThreshold {
    /// Builds a tear threshold.
    #[must_use]
    pub fn new(max_strain: f32) -> Self {
        Self { max_strain }
    }
}

/// Relative stretch of a distance constraint, `(current − rest) / rest`.
///
/// A (near) zero rest length is guarded: the denominator is clamped to [`EPS`]
/// so a rope pinned at zero rest still reports a finite, large strain rather
/// than dividing by zero.
#[must_use]
pub fn constraint_strain(rest: f32, current_length: f32) -> f32 {
    let denom = if rest > EPS { rest } else { EPS };
    (current_length - rest) / denom
}

/// Returns `true` when `strain` exceeds the tear threshold.
#[must_use]
pub fn should_tear(strain: f32, threshold: TearThreshold) -> bool {
    strain > threshold.max_strain
}

/// A request to rebuild island connectivity after a constraint tears
/// (design §10).
///
/// The broken constraint's index (into the caller's constraint list) and the
/// two particles it joined are handed to the topology stage, which detaches the
/// edge and re-partitions the affected island into its now-separate components.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TopologyUpdateRequest {
    /// Index of the broken constraint in the source list.
    pub broken_constraint: u32,
    /// First particle of the severed edge.
    pub detach_a: u32,
    /// Second particle of the severed edge.
    pub detach_b: u32,
}

/// Scans distance constraints for over-strained edges, emitting a tear request
/// per broken constraint (design §10).
///
/// Non-distance primitives and constraints referencing an out-of-range particle
/// index are skipped. The returned requests are in source order, so the scan is
/// deterministic and replay-safe.
#[must_use]
pub fn tear_scan(
    primitives: &[ConstraintPrimitive],
    positions: &[Vec3],
    threshold: TearThreshold,
) -> Vec<TopologyUpdateRequest> {
    let mut requests = Vec::new();
    for (index, primitive) in primitives.iter().enumerate() {
        if primitive.kind != ConstraintKind::Distance {
            continue;
        }
        let a = primitive.particles[0] as usize;
        let b = primitive.particles[1] as usize;
        if a >= positions.len() || b >= positions.len() {
            continue;
        }
        let length = positions[a].distance(positions[b]);
        let strain = constraint_strain(primitive.rest_value, length);
        if should_tear(strain, threshold) {
            requests.push(TopologyUpdateRequest {
                broken_constraint: index as u32,
                detach_a: primitive.particles[0],
                detach_b: primitive.particles[1],
            });
        }
    }
    requests
}

/// The collision scheme a fracture chunk is resolved against (design §10).
///
/// Debris is cheap in bulk only if its collision proxy is cheap. The scheme is
/// a per-chunk choice ranging from a single analytic plane (ground shards) to a
/// full convex hull (hero pieces); a scheduler trades fidelity for count using
/// the same ladder `Houdini` and `EmberGen` destruction layers expose.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FractureCollisionScheme {
    /// Collide against a single analytic half-space (e.g. the floor).
    AnalyticPlane,
    /// Collide against a shared signed-distance field of the environment.
    SignedDistanceField,
    /// Collide against a per-chunk convex hull (highest fidelity).
    ConvexHull,
    /// Collide as a bounding-sphere particle proxy (cheapest, bulk debris).
    ParticleProxy,
}

/// Binds a fracture chunk to the mesh renderer that draws its shard.
///
/// Each chunk is a rigid piece of geometry drawn by the mesh-renderer path
/// (design §16); this records which renderer instance and which vertex range of
/// the shared debris buffer belong to the chunk.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ChunkMeshBinding {
    /// Mesh-renderer instance handle drawing this shard.
    pub renderer_id: u32,
    /// First vertex of the shard in the shared debris vertex buffer.
    pub base_vertex: u32,
    /// Vertex count of the shard.
    pub vertex_count: u32,
}

/// The rigid-body approximation of a fracture chunk (design §10).
///
/// A chunk is simulated as a single `XPBD` rigid proxy rather than as its
/// constituent particles: the aggregate inverse mass, center of mass, and a
/// bounding radius are enough for the coarse rigid solve and the
/// [`FractureCollisionScheme::ParticleProxy`] sphere test that keep mass
/// destruction affordable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidApproximation {
    /// Aggregate inverse mass (`0` for an immovable shard).
    pub inv_mass: f32,
    /// Mass-weighted center of mass.
    pub center_of_mass: Vec3,
    /// Radius of the bounding sphere about the center of mass.
    pub bounding_radius: f32,
}

impl RigidApproximation {
    /// An empty (massless, zero-radius) approximation.
    pub const EMPTY: Self = Self {
        inv_mass: 0.0,
        center_of_mass: Vec3::ZERO,
        bounding_radius: 0.0,
    };
}

/// Computes the rigid approximation of a shard from its particles.
///
/// Mass is taken as `1 / inv_mass`, with a (numerically) zero inverse mass
/// modeled as [`STATIC_MASS`] so an anchored particle still weights the center
/// of mass. The chunk's inverse mass is `1 / Σ mass`, its center of mass the
/// mass-weighted position mean, and its bounding radius the farthest particle
/// distance from that center. `positions` and `inv_masses` are zipped, so a
/// length mismatch simply uses the shorter run.
#[must_use]
pub fn rigid_from_particles(positions: &[Vec3], inv_masses: &[f32]) -> RigidApproximation {
    let count = positions.len().min(inv_masses.len());
    if count == 0 {
        return RigidApproximation::EMPTY;
    }
    let mut total_mass = 0.0f32;
    let mut weighted = Vec3::ZERO;
    for (position, &inv_mass) in positions[..count].iter().zip(&inv_masses[..count]) {
        let mass = if inv_mass > EPS {
            1.0 / inv_mass
        } else {
            STATIC_MASS
        };
        total_mass += mass;
        weighted = weighted.add(position.scale(mass));
    }
    let center_of_mass = if total_mass > EPS {
        weighted.scale(1.0 / total_mass)
    } else {
        Vec3::ZERO
    };
    let mut bounding_radius = 0.0f32;
    for position in &positions[..count] {
        let dist = position.distance(center_of_mass);
        if dist > bounding_radius {
            bounding_radius = dist;
        }
    }
    let inv_mass = if total_mass > EPS {
        1.0 / total_mass
    } else {
        0.0
    };
    RigidApproximation {
        inv_mass,
        center_of_mass,
        bounding_radius,
    }
}

/// One shard produced by a fracture: its particle span, mesh binding, rigid
/// proxy, and collision scheme (design §10).
///
/// A fracture pattern turns a body's particles into a set of chunks; each chunk
/// hangs a mesh renderer for its shard and a rigid `XPBD` approximation for its
/// motion, and collides via its [`FractureCollisionScheme`]. This is the data
/// contract the destruction stage emits, not the fracture algorithm itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractureChunk {
    /// Stable identifier of this chunk within the fracture.
    pub chunk_id: u32,
    /// First particle index of the chunk's contiguous span.
    pub first_particle: u32,
    /// Number of particles in the chunk.
    pub particle_count: u32,
    /// Mesh-renderer binding for the shard geometry.
    pub mesh: ChunkMeshBinding,
    /// Rigid-body approximation of the shard.
    pub rigid: RigidApproximation,
    /// Collision scheme the shard resolves against.
    pub collision: FractureCollisionScheme,
}

/// Partitions a body's particles into contiguous fracture chunks (design §10).
///
/// The particles are split into `chunk_count` contiguous, near-equal spans
/// (earlier chunks absorb the remainder), and each span's rigid approximation is
/// computed with [`rigid_from_particles`]. Every chunk is bound to renderer
/// `renderer_base + chunk_id` and uses `collision`. `chunk_count` is clamped to
/// the particle count, and a zero-particle or zero-chunk input yields no chunks,
/// so the split is total and deterministic. `positions` and `inv_masses` are
/// zipped; the shorter run bounds the particle count.
#[must_use]
pub fn plan_fracture(
    positions: &[Vec3],
    inv_masses: &[f32],
    chunk_count: u32,
    renderer_base: u32,
    collision: FractureCollisionScheme,
) -> Vec<FractureChunk> {
    let particle_count = positions.len().min(inv_masses.len());
    if particle_count == 0 || chunk_count == 0 {
        return Vec::new();
    }
    let chunks = (chunk_count as usize).min(particle_count);
    let base = particle_count / chunks;
    let remainder = particle_count % chunks;

    let mut result = Vec::with_capacity(chunks);
    let mut cursor = 0usize;
    for chunk_id in 0..chunks {
        let span = if chunk_id < remainder { base + 1 } else { base };
        let end = cursor + span;
        let rigid = rigid_from_particles(&positions[cursor..end], &inv_masses[cursor..end]);
        result.push(FractureChunk {
            chunk_id: chunk_id as u32,
            first_particle: cursor as u32,
            particle_count: span as u32,
            mesh: ChunkMeshBinding {
                renderer_id: renderer_base + chunk_id as u32,
                base_vertex: cursor as u32,
                vertex_count: span as u32,
            },
            rigid,
            collision,
        });
        cursor = end;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const TEST_EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < TEST_EPS
    }

    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    #[test]
    fn compliance_is_the_inverse_of_stiffness() {
        assert!(approx(Compliance::from_stiffness(2.0).alpha, 0.5));
        assert!(approx(Compliance::from_stiffness(4.0).alpha, 0.25));
        // A non-positive stiffness folds to rigid.
        assert!(Compliance::from_stiffness(0.0).is_rigid());
        assert!(Compliance::from_stiffness(-3.0).is_rigid());
        assert!(Compliance::RIGID.is_rigid());
        // Negative raw compliance clamps to rigid.
        assert!(Compliance::new(-1.0).is_rigid());
    }

    #[test]
    fn compliance_scales_by_inverse_substep_squared() {
        let c = Compliance::new(0.5);
        // dt = 0.5 -> alpha / dt^2 = 0.5 / 0.25 = 2.0.
        assert!(approx(c.scaled_for_substep(0.5), 2.0));
        // A rigid constraint stays rigid at any substep.
        assert!(approx(Compliance::RIGID.scaled_for_substep(0.5), 0.0));
        // Degenerate substep dt -> 0 (no divide by zero).
        assert!(approx(c.scaled_for_substep(0.0), 0.0));
        // Free-function form agrees.
        assert!(approx(effective_compliance(0.5, 0.5), 2.0));
    }

    #[test]
    fn substep_schedule_divides_the_frame() {
        let schedule = SubstepSchedule::new(1.0 / 60.0, 4, 8);
        assert!(approx(schedule.substep_dt(), (1.0 / 60.0) / 4.0));
        // More substeps stiffen a fixed compliance via the 1/dt^2 scaling.
        let soft = Compliance::new(0.01);
        let few = SubstepSchedule::new(1.0 / 60.0, 1, 8).effective_compliance(soft);
        let many = SubstepSchedule::new(1.0 / 60.0, 8, 8).effective_compliance(soft);
        assert!(many > few);
        // Zero substeps is guarded.
        assert!(approx(
            SubstepSchedule::new(1.0 / 60.0, 0, 8).substep_dt(),
            0.0
        ));
    }

    #[test]
    fn distance_projection_satisfies_a_rigid_equal_mass_constraint() {
        let a = Vec3::new(2.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 0.0);
        // rest 1, equal unit inverse mass, rigid (alpha_tilde = 0), lambda 0.
        let corr = project_distance(a, b, 1.0, 1.0, 1.0, 0.0, 0.0);
        assert!(approx(corr.delta_lambda, -0.5));
        let new_a = a.add(corr.delta_a);
        let new_b = b.add(corr.delta_b);
        // One projection fully satisfies a rigid equal-mass distance constraint.
        assert!(approx(new_a.distance(new_b), 1.0));
        // Corrections are symmetric for equal masses.
        assert!(vec_approx(corr.delta_a, corr.delta_b.scale(-1.0)));
    }

    #[test]
    fn distance_projection_pins_infinite_mass_endpoint() {
        let a = Vec3::new(2.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 0.0);
        // b is immovable (inv_mass 0): all correction goes to a.
        let corr = project_distance(a, b, 1.0, 0.0, 1.0, 0.0, 0.0);
        assert!(vec_approx(corr.delta_b, Vec3::ZERO));
        let new_a = a.add(corr.delta_a);
        assert!(approx(new_a.distance(b), 1.0));
    }

    #[test]
    fn distance_projection_guards_degenerate_inputs() {
        // Coincident endpoints -> no correction.
        let zero = project_distance(Vec3::ZERO, Vec3::ZERO, 1.0, 1.0, 1.0, 0.0, 0.0);
        assert_eq!(zero, DistanceCorrection::ZERO);
        // Two immovable endpoints -> zero denominator -> no correction.
        let pinned = project_distance(
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        );
        assert_eq!(pinned, DistanceCorrection::ZERO);
    }

    #[test]
    fn compliant_projection_undershoots_the_rigid_one() {
        let a = Vec3::new(2.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 0.0);
        let rigid = project_distance(a, b, 1.0, 1.0, 1.0, 0.0, 0.0);
        // A positive alpha_tilde softens the correction magnitude.
        let soft = project_distance(a, b, 1.0, 1.0, 1.0, 2.0, 0.0);
        assert!(soft.delta_lambda.abs() < rigid.delta_lambda.abs());
    }

    #[test]
    fn primitive_indices_expose_only_live_particles() {
        let dist = ConstraintPrimitive::distance(3, 7, 1.0, Compliance::RIGID);
        assert_eq!(dist.indices(), &[3, 7]);
        let tet = ConstraintPrimitive::volume(1, 2, 3, 4, 0.5, Compliance::new(0.01));
        assert_eq!(tet.indices(), &[1, 2, 3, 4]);
        let bend = ConstraintPrimitive::bend(0, 1, 2, 3, 0.0, Compliance::RIGID);
        assert_eq!(bend.particle_count, 4);
    }

    #[test]
    fn anisotropic_axis_is_normalized() {
        let aniso = ConstraintPrimitive::anisotropic(
            0,
            1,
            Vec3::new(0.0, 5.0, 0.0),
            1.0,
            Compliance::RIGID,
        );
        assert!(approx(aniso.axis.length(), 1.0));
        assert!(vec_approx(aniso.axis, Vec3::new(0.0, 1.0, 0.0)));
        // A zero axis collapses to isotropic.
        let iso = ConstraintPrimitive::anisotropic(0, 1, Vec3::ZERO, 1.0, Compliance::RIGID);
        assert!(vec_approx(iso.axis, Vec3::ZERO));
    }

    #[test]
    fn only_distance_primitives_lower_to_particle_constraints() {
        let dist = ConstraintPrimitive::distance(2, 5, 1.5, Compliance::new(0.25));
        let lowered = dist.as_particle_constraint().expect("distance lowers");
        assert_eq!(lowered.a, 2);
        assert_eq!(lowered.b, 5);
        assert!(approx(lowered.rest_length, 1.5));
        assert!(approx(lowered.compliance, 0.25));
        assert!(
            ConstraintPrimitive::bend(0, 1, 2, 3, 0.0, Compliance::RIGID)
                .as_particle_constraint()
                .is_none()
        );
        assert!(
            ConstraintPrimitive::volume(0, 1, 2, 3, 1.0, Compliance::RIGID)
                .as_particle_constraint()
                .is_none()
        );
    }

    #[test]
    fn coloring_reuses_stages_and_skips_non_distance() {
        let primitives = [
            ConstraintPrimitive::distance(0, 1, 1.0, Compliance::RIGID),
            ConstraintPrimitive::distance(2, 3, 1.0, Compliance::RIGID),
            ConstraintPrimitive::distance(4, 5, 1.0, Compliance::RIGID),
            // A volume constraint is skipped by the distance coloring.
            ConstraintPrimitive::volume(0, 1, 2, 3, 1.0, Compliance::RIGID),
        ];
        let coloring = color_distance_constraints(&primitives);
        // Three disjoint distance edges pack into a single color.
        assert_eq!(coloring.color_count(), 1);
        assert_eq!(coloring.constraints.len(), 3);
        let batch = coloring.batches[0];
        assert_eq!(batch_constraints(&coloring, batch).len(), 3);
    }

    #[test]
    fn tearing_triggers_past_the_strain_threshold() {
        // rest 1, stretched to 1.5 -> strain 0.5.
        assert!(approx(constraint_strain(1.0, 1.5), 0.5));
        let threshold = TearThreshold::new(0.4);
        assert!(should_tear(constraint_strain(1.0, 1.5), threshold));
        // A gentler threshold keeps the edge intact.
        assert!(!should_tear(
            constraint_strain(1.0, 1.5),
            TearThreshold::new(0.6)
        ));
        // A zero rest length is guarded (finite, large strain).
        assert!(constraint_strain(0.0, 1.0) > 0.0);
    }

    #[test]
    fn tear_scan_emits_requests_in_source_order() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        ];
        let primitives = [
            // 0-1 at rest length 1: not torn.
            ConstraintPrimitive::distance(0, 1, 1.0, Compliance::RIGID),
            // 1-2 stretched from rest 1 to length 2: torn.
            ConstraintPrimitive::distance(1, 2, 1.0, Compliance::RIGID),
            // A bend primitive is ignored by the scan.
            ConstraintPrimitive::bend(0, 1, 2, 0, 0.0, Compliance::RIGID),
            // Out-of-range index is skipped safely.
            ConstraintPrimitive::distance(2, 99, 0.1, Compliance::RIGID),
        ];
        let requests = tear_scan(&primitives, &positions, TearThreshold::new(0.5));
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0],
            TopologyUpdateRequest {
                broken_constraint: 1,
                detach_a: 1,
                detach_b: 2,
            }
        );
    }

    #[test]
    fn solver_selection_escalates_to_vbd() {
        let sel = SolverSelection::standard();
        // Soft, sparse island stays on XPBD.
        assert_eq!(sel.select(10.0, 2.0, false), SolverTier::Xpbd);
        // High stability forces VBD regardless of the metrics.
        assert_eq!(sel.select(10.0, 2.0, true), SolverTier::Vbd);
        // Very stiff island escalates.
        assert_eq!(sel.select(5000.0, 2.0, false), SolverTier::Vbd);
        // Densely-coupled contacts escalate.
        assert_eq!(sel.select(10.0, 20.0, false), SolverTier::Vbd);
    }

    #[test]
    fn convergence_contract_distinguishes_the_tiers() {
        let xpbd = ConvergenceContract::for_tier(SolverTier::Xpbd);
        let vbd = ConvergenceContract::for_tier(SolverTier::Vbd);
        assert!(!xpbd.unconditionally_stable);
        assert!(vbd.unconditionally_stable);
        // VBD trades cost for stability.
        assert!(vbd.cost_multiplier > xpbd.cost_multiplier);
        assert!(xpbd.parallelizable && vbd.parallelizable);
    }

    #[test]
    fn rigid_approximation_averages_mass_and_bounds_radius() {
        let positions = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        let inv_masses = [1.0, 1.0];
        let rigid = rigid_from_particles(&positions, &inv_masses);
        assert!(vec_approx(rigid.center_of_mass, Vec3::new(1.0, 0.0, 0.0)));
        assert!(approx(rigid.bounding_radius, 1.0));
        // Total mass 2 -> inverse mass 0.5.
        assert!(approx(rigid.inv_mass, 0.5));
    }

    #[test]
    fn rigid_approximation_handles_empty_and_static() {
        assert_eq!(rigid_from_particles(&[], &[]), RigidApproximation::EMPTY);
        // A single immovable particle -> near-zero chunk inverse mass.
        let rigid = rigid_from_particles(&[Vec3::new(1.0, 0.0, 0.0)], &[0.0]);
        assert!(rigid.inv_mass < 1.0e-6);
        assert!(vec_approx(rigid.center_of_mass, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn fracture_splits_particles_into_contiguous_chunks() {
        let positions: Vec<Vec3> = (0..4).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
        let inv_masses = vec![1.0f32; 4];
        let chunks = plan_fracture(
            &positions,
            &inv_masses,
            2,
            100,
            FractureCollisionScheme::ParticleProxy,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].first_particle, 0);
        assert_eq!(chunks[0].particle_count, 2);
        assert_eq!(chunks[1].first_particle, 2);
        assert_eq!(chunks[1].particle_count, 2);
        // Renderer ids run from the base, one per chunk.
        assert_eq!(chunks[0].mesh.renderer_id, 100);
        assert_eq!(chunks[1].mesh.renderer_id, 101);
        // Chunk 0 spans particles at x=0 and x=1: center 0.5, radius 0.5.
        assert!(vec_approx(
            chunks[0].rigid.center_of_mass,
            Vec3::new(0.5, 0.0, 0.0)
        ));
        assert!(approx(chunks[0].rigid.bounding_radius, 0.5));
    }

    #[test]
    fn fracture_remainder_front_loads_earlier_chunks() {
        let positions: Vec<Vec3> = (0..5).map(|i| Vec3::new(i as f32, 0.0, 0.0)).collect();
        let inv_masses = vec![1.0f32; 5];
        let chunks = plan_fracture(
            &positions,
            &inv_masses,
            2,
            0,
            FractureCollisionScheme::ConvexHull,
        );
        assert_eq!(chunks.len(), 2);
        // 5 particles into 2 chunks: first gets the remainder (3), second gets 2.
        assert_eq!(chunks[0].particle_count, 3);
        assert_eq!(chunks[1].particle_count, 2);
    }

    #[test]
    fn fracture_clamps_and_guards_edge_counts() {
        let positions = [Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let inv_masses = [1.0f32, 1.0];
        // Zero chunks or zero particles -> nothing.
        assert!(plan_fracture(
            &positions,
            &inv_masses,
            0,
            0,
            FractureCollisionScheme::AnalyticPlane
        )
        .is_empty());
        assert!(plan_fracture(&[], &[], 4, 0, FractureCollisionScheme::AnalyticPlane).is_empty());
        // More chunks than particles clamps to one particle per chunk.
        let chunks = plan_fracture(
            &positions,
            &inv_masses,
            10,
            0,
            FractureCollisionScheme::SignedDistanceField,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].particle_count, 1);
        assert_eq!(chunks[1].particle_count, 1);
    }
}
