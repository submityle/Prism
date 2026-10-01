//! Projective Dynamics global-solve slot (design 8.6 item 17).
//!
//! Guide-strand dynamics in [`super::dynamics`] relaxes its constraints with a
//! local, Gauss-Seidel sweep: each constraint is projected in turn and the
//! particles move immediately. That is cheap but its convergence rate degrades
//! as a strand grows stiffer or longer. Projective Dynamics (`Bouaziz` 2014)
//! keeps the same per-constraint *local* projections but couples them through a
//! single *global* linear solve, so one iteration propagates information across
//! the whole system at once. Because the global matrix is constant for a fixed
//! constraint graph, it is factored **once** and the factorization is reused
//! every iteration and every frame; only the right-hand side changes.
//!
//! This module is the architecture-side, deterministic CPU golden for that
//! solver family. It is intentionally small-scale and self-contained: the
//! production solve runs on the GPU through the frame-graph schedule, while this
//! module pins down the local/global math so it can be golden-tested by hand.
//! Everything is fixed-order `f32` arithmetic, so identical inputs produce
//! bit-identical outputs.
//!
//! The pieces are:
//!
//! 1. **Local step** — every constraint projects the current state onto its
//!    manifold to produce a target. An edge-length constraint projects the edge
//!    vector onto the sphere of its rest length (see [`local_project_edge`]); a
//!    three-point bend constraint targets a rest Laplacian.
//! 2. **Global step** — solve the symmetric positive-definite (`SPD`) system
//!    `(M/h^2 + sum_i w_i S_i^T S_i) x = M/h^2 s_n + sum_i w_i S_i^T p_i`. The
//!    matrix is factored once with a hand-written `Cholesky` decomposition
//!    ([`GlobalSystem::factor`]) and reused via [`GlobalSystem::solve`]; an
//!    independent conjugate-gradient path ([`GlobalSystem::solve_cg`]) is kept
//!    for cross-checking the direct solve.
//! 3. **`ADMM` variant** — the alternating-direction method of multipliers
//!    (`Overby` 2017) adds a dual variable per constraint and alternates the
//!    same local projection and global solve, emitting primal/dual residual
//!    sequences usable as a convergence criterion (see [`solve_admm`]).
//! 4. **Golden outputs** — the end positions plus a per-iteration residual
//!    sequence (the Projective Dynamics objective, which is monotone
//!    non-increasing) from [`solve_projective`].
//!
//! The crate carries no linear-algebra dependency, so the vector type and every
//! matrix routine (matvec, dot product, factorization, substitution) are written
//! out by hand here. This module deliberately shares no types with the other
//! hair modules so it stays disjoint and independently verifiable.

use alloc::{vec, vec::Vec};

/// Vectors shorter than this are treated as having no well-defined direction.
const EPS_LEN: f32 = 1.0e-12;
/// Floor applied to `Cholesky` pivots so a (near-)singular block never yields a
/// non-finite factor; the mass term keeps the real system well above this.
const EPS_PIVOT: f32 = 1.0e-20;
/// Squared-residual target below which the conjugate-gradient loop stops.
const CG_TOL: f32 = 1.0e-20;
/// Denominator guard for the conjugate-gradient step length.
const CG_EPS: f32 = 1.0e-30;

/// A minimal 3-component vector for the solver's math.
///
/// Defined locally so the module needs no linear-algebra dependency; all
/// operations are plain `f32` arithmetic in a fixed order, which is what makes
/// the solver bit-for-bit reproducible.
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
    /// Constructs a vector from its components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The solver math API is specified with named add/sub methods for call-site uniformity; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses a named sub for call-site uniformity, not operator traits."
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

    /// Squared Euclidean length.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Replaces any non-finite component with `0`, so `NaN`/infinity in the
    /// input can never reach the solver.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self::new(
            sanitize_finite(self.x),
            sanitize_finite(self.y),
            sanitize_finite(self.z),
        )
    }
}

/// A distance (edge-length) constraint between two particles.
///
/// The local projection pulls the edge `x_i - x_j` onto the sphere of radius
/// `rest_length`; `weight` is the Projective Dynamics stiffness `w_i` with which
/// the constraint enters the global system (larger means the solve honors the
/// rest length more tightly relative to the inertia term).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeConstraint {
    /// Index of the first endpoint particle.
    pub i: usize,
    /// Index of the second endpoint particle.
    pub j: usize,
    /// Target segment length the local step projects onto.
    pub rest_length: f32,
    /// Projective Dynamics stiffness weight.
    pub weight: f32,
}

impl EdgeConstraint {
    /// Convenience constructor.
    #[must_use]
    pub const fn new(i: usize, j: usize, rest_length: f32, weight: f32) -> Self {
        Self {
            i,
            j,
            rest_length,
            weight,
        }
    }
}

/// A three-point bending constraint around a center particle.
///
/// The constraint operator is the discrete Laplacian
/// `x_j - 0.5 * x_i - 0.5 * x_k` (with `j` the center), and the local step
/// targets the authored `rest_laplacian` vector (zero for a straight rest pose).
/// Because the target does not depend on the current state, the bend term is a
/// linear spring in the global system and is resolved exactly by one solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BendConstraint {
    /// First neighbor particle.
    pub i: usize,
    /// Center particle (the Laplacian is centered here).
    pub j: usize,
    /// Second neighbor particle.
    pub k: usize,
    /// Rest value of the Laplacian the local step targets.
    pub rest_laplacian: Vec3,
    /// Projective Dynamics stiffness weight.
    pub weight: f32,
}

impl BendConstraint {
    /// Convenience constructor.
    #[must_use]
    pub const fn new(i: usize, j: usize, k: usize, rest_laplacian: Vec3, weight: f32) -> Self {
        Self {
            i,
            j,
            k,
            rest_laplacian,
            weight,
        }
    }
}

/// The full set of constraints driving one solve.
///
/// Edges and bends are stored separately but assemble into the same global
/// system. An empty set is valid: the solve then relaxes only toward the
/// inertial target `s_n`, i.e. it leaves the input positions unchanged.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintSet {
    /// Edge-length (distance) constraints.
    pub edges: Vec<EdgeConstraint>,
    /// Three-point bending constraints.
    pub bends: Vec<BendConstraint>,
}

impl ConstraintSet {
    /// An empty constraint set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            bends: Vec::new(),
        }
    }

    /// A constraint set with a single edge-length constraint.
    #[must_use]
    pub fn single_edge(edge: EdgeConstraint) -> Self {
        Self {
            edges: alloc::vec![edge],
            bends: Vec::new(),
        }
    }
}

/// Tunable parameters for one projective solve.
///
/// `mass` is a uniform per-particle mass and `dt` the time step; together they
/// form the inertia weight `mass / dt^2` on the system diagonal, which also
/// regularizes the otherwise rank-deficient constraint Laplacian into an `SPD`
/// matrix. `iterations` is the fixed per-call budget of local/global rounds.
/// All fields are clamped by [`SolverConfig::sanitized`] so no non-finite or
/// non-positive value can destabilize the solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverConfig {
    /// Time step in seconds (sanitized to a finite positive value).
    pub dt: f32,
    /// Fixed number of local/global iterations per call.
    pub iterations: u32,
    /// Uniform particle mass (sanitized to a finite positive value).
    pub mass: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            iterations: 20,
            mass: 1.0,
        }
    }
}

impl SolverConfig {
    /// Convenience constructor.
    #[must_use]
    pub const fn new(dt: f32, iterations: u32, mass: f32) -> Self {
        Self {
            dt,
            iterations,
            mass,
        }
    }

    /// Clamp non-finite/non-positive `dt`/`mass` to safe defaults and cap the
    /// iteration budget, so the returned config always yields a well-posed
    /// `SPD` system and a bounded loop.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            dt: sanitize_pos(self.dt, 1.0 / 60.0),
            iterations: self.iterations.min(100_000),
            mass: sanitize_pos(self.mass, 1.0),
        }
    }

    /// The inertia weight `mass / dt^2` placed on every diagonal entry of the
    /// global matrix. Always finite and positive after [`Self::sanitized`].
    #[must_use]
    pub fn mass_over_h2(self) -> f32 {
        let c = self.sanitized();
        c.mass / (c.dt * c.dt)
    }
}

/// Projects an edge onto its rest length about the edge midpoint.
///
/// Returns the two endpoint positions that are exactly `rest` apart and share
/// the original midpoint, i.e. the closest rigid edge of the given length to the
/// input (the Projective Dynamics / position-based distance projection). A
/// degenerate (near-zero) edge has no direction, so it is split along the
/// canonical `+x` axis instead of producing a `NaN`.
#[must_use]
pub fn local_project_edge(xi: Vec3, xj: Vec3, rest: f32) -> (Vec3, Vec3) {
    let r = sanitize_nonneg(rest);
    let a = xi.sanitized();
    let b = xj.sanitized();
    let mid = a.add(b).scale(0.5);
    let dir = project_to_length(a.sub(b), r);
    let half = dir.scale(0.5);
    (mid.add(half), mid.sub(half))
}

/// Projects a direction vector onto the sphere of radius `rest`.
///
/// A near-zero input has no direction, so it maps to `rest` along `+x` to stay
/// finite and deterministic.
fn project_to_length(v: Vec3, rest: f32) -> Vec3 {
    let len = v.length();
    if len > EPS_LEN {
        v.scale(rest / len)
    } else {
        Vec3::new(rest, 0.0, 0.0)
    }
}

/// How a constraint's local step projects its measured value.
#[derive(Clone, Copy, Debug)]
enum ProjKind {
    /// Project the measured vector onto a sphere of this radius.
    Edge { rest: f32 },
    /// Target a constant vector regardless of the measured value.
    Fixed { target: Vec3 },
}

/// Internal uniform form of a constraint: a weighted linear operator
/// `D x = sum (coeff * x[idx])` plus its local projection rule.
struct Atom {
    /// `(particle index, coefficient)` terms of the operator `D`.
    terms: Vec<(usize, f32)>,
    /// Projective Dynamics stiffness weight.
    weight: f32,
    /// Local projection rule applied to `D x`.
    kind: ProjKind,
}

impl Atom {
    /// Evaluates `D x` for the current state.
    fn eval(&self, x: &[Vec3]) -> Vec3 {
        let mut acc = Vec3::ZERO;
        for &(idx, coeff) in &self.terms {
            acc = acc.add(x[idx].scale(coeff));
        }
        acc
    }

    /// Projects a measured operator value onto this constraint's manifold.
    fn project(&self, measured: Vec3) -> Vec3 {
        match self.kind {
            ProjKind::Edge { rest } => project_to_length(measured, rest),
            ProjKind::Fixed { target } => target,
        }
    }
}

/// Builds the uniform [`Atom`] list from a constraint set, dropping any
/// constraint whose indices are out of range or not distinct.
fn build_atoms(constraints: &ConstraintSet, n: usize) -> Vec<Atom> {
    let mut atoms = Vec::new();
    for e in &constraints.edges {
        if e.i >= n || e.j >= n || e.i == e.j {
            continue;
        }
        atoms.push(Atom {
            terms: alloc::vec![(e.i, 1.0), (e.j, -1.0)],
            weight: sanitize_nonneg(e.weight),
            kind: ProjKind::Edge {
                rest: sanitize_nonneg(e.rest_length),
            },
        });
    }
    for b in &constraints.bends {
        if b.i >= n || b.j >= n || b.k >= n {
            continue;
        }
        if b.i == b.j || b.j == b.k || b.i == b.k {
            continue;
        }
        atoms.push(Atom {
            terms: alloc::vec![(b.i, -0.5), (b.j, 1.0), (b.k, -0.5)],
            weight: sanitize_nonneg(b.weight),
            kind: ProjKind::Fixed {
                target: b.rest_laplacian.sanitized(),
            },
        });
    }
    atoms
}

/// Assembles the dense symmetric global matrix `A = M/h^2 I + sum_i w_i D_i^T D_i`
/// in row-major order.
fn assemble_matrix(atoms: &[Atom], n: usize, mass_over_h2: f32) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for d in 0..n {
        a[d * n + d] += mass_over_h2;
    }
    for atom in atoms {
        let w = atom.weight;
        for &(ia, ca) in &atom.terms {
            for &(ib, cb) in &atom.terms {
                a[ia * n + ib] += w * ca * cb;
            }
        }
    }
    a
}

/// A pre-factored global system `A x = b`.
///
/// [`Self::factor`] assembles `A` for a fixed constraint graph and computes its
/// `Cholesky` factor once; [`Self::solve`] then reuses the factor for any
/// number of right-hand sides, which is exactly the Projective Dynamics reuse
/// pattern (constant matrix, changing projections). The original matrix is kept
/// alongside the factor so the system can be re-applied (see [`Self::mul`]) for
/// verification and for the conjugate-gradient cross-check.
pub struct GlobalSystem {
    /// Dimension (particle count).
    n: usize,
    /// The assembled `SPD` matrix, row-major `n * n`.
    a: Vec<f32>,
    /// Lower-triangular `Cholesky` factor `L` with `A = L L^T`, row-major `n * n`.
    l: Vec<f32>,
}

impl GlobalSystem {
    /// Assembles and factors the global matrix for a constraint set.
    ///
    /// The inertia term `mass / dt^2` on the diagonal keeps the matrix strictly
    /// positive-definite even when the constraint graph alone is rank-deficient,
    /// so the `Cholesky` factorization always succeeds.
    #[must_use]
    pub fn factor(constraints: &ConstraintSet, n: usize, config: SolverConfig) -> Self {
        let atoms = build_atoms(constraints, n);
        let a = assemble_matrix(&atoms, n, config.mass_over_h2());
        let l = cholesky(&a, n);
        Self { n, a, l }
    }

    /// Dimension of the system.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Solves `A x = rhs` with the stored `Cholesky` factor.
    ///
    /// `rhs` is a per-particle vector; the scalar factor is applied to each of
    /// the three coordinates at once via forward/back substitution. A `rhs`
    /// shorter than the dimension is padded with zeros, so the call never
    /// panics on a short slice.
    #[must_use]
    pub fn solve(&self, rhs: &[Vec3]) -> Vec<Vec3> {
        let n = self.n;
        let mut y = vec![Vec3::ZERO; n];
        for i in 0..n {
            let mut t = *rhs.get(i).unwrap_or(&Vec3::ZERO);
            for (k, yk) in y.iter().enumerate().take(i) {
                t = t.sub(yk.scale(self.l[i * n + k]));
            }
            y[i] = t.scale(1.0 / self.l[i * n + i]);
        }
        let mut x = vec![Vec3::ZERO; n];
        for i in (0..n).rev() {
            let mut t = y[i];
            for (k, xk) in x.iter().enumerate().skip(i + 1) {
                t = t.sub(xk.scale(self.l[k * n + i]));
            }
            x[i] = t.scale(1.0 / self.l[i * n + i]);
        }
        x
    }

    /// Applies the (un-factored) matrix: returns `A x`.
    ///
    /// Used to check that a computed solution satisfies the system and to drive
    /// the conjugate-gradient path.
    #[must_use]
    pub fn mul(&self, x: &[Vec3]) -> Vec<Vec3> {
        let n = self.n;
        let mut out = vec![Vec3::ZERO; n];
        for (i, out_i) in out.iter_mut().enumerate() {
            let mut s = Vec3::ZERO;
            for j in 0..n {
                let xj = *x.get(j).unwrap_or(&Vec3::ZERO);
                s = s.add(xj.scale(self.a[i * n + j]));
            }
            *out_i = s;
        }
        out
    }

    /// Solves `A x = rhs` with an independent conjugate-gradient iteration.
    ///
    /// Each coordinate is solved separately on the same scalar matrix. This path
    /// uses no factorization and exists mainly to cross-check [`Self::solve`]:
    /// on an `SPD` system the two agree to numerical tolerance.
    #[must_use]
    pub fn solve_cg(&self, rhs: &[Vec3]) -> Vec<Vec3> {
        let n = self.n;
        let bx: Vec<f32> = (0..n).map(|i| rhs.get(i).map_or(0.0, |v| v.x)).collect();
        let by: Vec<f32> = (0..n).map(|i| rhs.get(i).map_or(0.0, |v| v.y)).collect();
        let bz: Vec<f32> = (0..n).map(|i| rhs.get(i).map_or(0.0, |v| v.z)).collect();
        let sx = cg_scalar(&self.a, n, &bx);
        let sy = cg_scalar(&self.a, n, &by);
        let sz = cg_scalar(&self.a, n, &bz);
        (0..n).map(|i| Vec3::new(sx[i], sy[i], sz[i])).collect()
    }
}

/// In-place-style `Cholesky` factorization: returns the lower-triangular `L`
/// with `A ~= L L^T`, row-major `n * n`.
///
/// Pivots are floored at [`EPS_PIVOT`] so a degenerate block cannot produce a
/// non-finite factor; the real inertia-regularized system stays well clear of
/// that floor.
fn cholesky(a: &[f32], n: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + i] = sum.max(EPS_PIVOT).sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    l
}

/// Scalar dot product.
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Scalar matrix-vector product `A x` for the dense row-major matrix.
fn matvec_scalar(a: &[f32], n: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut s = 0.0f32;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        out[i] = s;
    }
    out
}

/// Conjugate-gradient solve of `A x = b` for one scalar coordinate.
fn cg_scalar(a: &[f32], n: usize, b: &[f32]) -> Vec<f32> {
    let mut x = vec![0.0f32; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs = dot_scalar(&r, &r);
    let max_iter = n * 10 + 50;
    let mut it = 0;
    while it < max_iter {
        if rs <= CG_TOL {
            break;
        }
        let ap = matvec_scalar(a, n, &p);
        let pap = dot_scalar(&p, &ap);
        if pap.abs() <= CG_EPS {
            break;
        }
        let alpha = rs / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot_scalar(&r, &r);
        let beta = rs_new / rs;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs = rs_new;
        it += 1;
    }
    x
}

/// Builds the Projective Dynamics right-hand side for the current state and
/// returns it together with the objective energy at that state.
///
/// The right-hand side is `M/h^2 s_n + sum_i w_i D_i^T p_i`, where `p_i` is the
/// local projection of `D_i x`. The energy is
/// `0.5 (x - s_n)^T (M/h^2) (x - s_n) + sum_i 0.5 w_i ||D_i x - p_i||^2`, the
/// quantity the alternating local/global steps jointly minimize.
fn assemble_local(atoms: &[Atom], x: &[Vec3], sn: &[Vec3], mass_over_h2: f32) -> (Vec<Vec3>, f32) {
    let n = x.len();
    let mut rhs = Vec::with_capacity(n);
    let mut energy = 0.0f32;
    for i in 0..n {
        rhs.push(sn[i].scale(mass_over_h2));
        energy += 0.5 * mass_over_h2 * x[i].sub(sn[i]).length_squared();
    }
    for atom in atoms {
        let measured = atom.eval(x);
        let p = atom.project(measured);
        for &(idx, coeff) in &atom.terms {
            rhs[idx] = rhs[idx].add(p.scale(atom.weight * coeff));
        }
        energy += 0.5 * atom.weight * measured.sub(p).length_squared();
    }
    (rhs, energy)
}

/// Result of a Projective Dynamics solve.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProjectiveSolve {
    /// Final particle positions after the iteration budget.
    pub positions: Vec<Vec3>,
    /// Per-iteration objective energy, recorded before each global solve.
    ///
    /// The sequence has exactly `config.iterations` entries (empty when the
    /// solve early-returns) and is monotone non-increasing, so it doubles as a
    /// convergence trace.
    pub residuals: Vec<f32>,
}

/// Runs the Projective Dynamics local/global iteration.
///
/// `initial` is the starting state and also the inertial target `s_n` (there is
/// no external force in this architecture golden). The global matrix is factored
/// once and reused across all iterations. Each iteration records the objective
/// energy, then performs one global solve; the recorded residual sequence is
/// monotone non-increasing.
///
/// The call early-returns the sanitized input unchanged when there is nothing to
/// do: an empty state or a zero iteration budget. Non-finite inputs are
/// sanitized, so the solve never emits a `NaN`.
#[must_use]
pub fn solve_projective(
    initial: &[Vec3],
    constraints: &ConstraintSet,
    config: SolverConfig,
) -> ProjectiveSolve {
    let cfg = config.sanitized();
    let n = initial.len();
    let positions: Vec<Vec3> = initial.iter().map(|v| v.sanitized()).collect();
    if n == 0 || cfg.iterations == 0 {
        return ProjectiveSolve {
            positions,
            residuals: Vec::new(),
        };
    }
    let sn = positions.clone();
    let atoms = build_atoms(constraints, n);
    let system = GlobalSystem::factor(constraints, n, cfg);
    let mass_over_h2 = cfg.mass_over_h2();
    let mut state = positions;
    let mut residuals = Vec::with_capacity(cfg.iterations as usize);
    let mut it = 0;
    while it < cfg.iterations {
        let (rhs, energy) = assemble_local(&atoms, &state, &sn, mass_over_h2);
        residuals.push(energy);
        state = system.solve(&rhs);
        it += 1;
    }
    ProjectiveSolve {
        positions: state,
        residuals,
    }
}

/// Result of an `ADMM` solve.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdmmSolve {
    /// Final particle positions after the iteration budget.
    pub positions: Vec<Vec3>,
    /// Per-iteration primal residual `||D x - z||` (weighted L2).
    pub primal_residuals: Vec<f32>,
    /// Per-iteration dual residual `||w D^T (z - z_prev)||`-style measure.
    pub dual_residuals: Vec<f32>,
}

/// Runs the alternating-direction method of multipliers (`ADMM`) variant.
///
/// Each constraint gets a split variable `z_c` and a scaled dual `u_c`. One
/// iteration performs the global solve
/// `A x = M/h^2 s_n + sum_c w_c D_c^T (z_c - u_c)`, re-projects
/// `z_c = proj(D_c x + u_c)`, then updates the dual `u_c += D_c x - z_c`. The
/// global matrix is the same constant `SPD` system as Projective Dynamics and is
/// factored once. Primal (`D x - z`) and dual residual sequences are emitted for
/// use as a convergence criterion.
///
/// Early-returns the sanitized input unchanged for an empty state or a zero
/// budget; non-finite inputs are sanitized so no `NaN` escapes.
#[must_use]
pub fn solve_admm(
    initial: &[Vec3],
    constraints: &ConstraintSet,
    config: SolverConfig,
) -> AdmmSolve {
    let cfg = config.sanitized();
    let n = initial.len();
    let positions: Vec<Vec3> = initial.iter().map(|v| v.sanitized()).collect();
    if n == 0 || cfg.iterations == 0 {
        return AdmmSolve {
            positions,
            primal_residuals: Vec::new(),
            dual_residuals: Vec::new(),
        };
    }
    let sn = positions.clone();
    let atoms = build_atoms(constraints, n);
    let system = GlobalSystem::factor(constraints, n, cfg);
    let mass_over_h2 = cfg.mass_over_h2();

    let mut state = positions;
    // Initialize split variables to the current operator values and the duals
    // to zero (the canonical scaled-form warm start).
    let mut z: Vec<Vec3> = atoms.iter().map(|atom| atom.eval(&state)).collect();
    let mut u: Vec<Vec3> = alloc::vec![Vec3::ZERO; atoms.len()];

    let mut primal_residuals = Vec::with_capacity(cfg.iterations as usize);
    let mut dual_residuals = Vec::with_capacity(cfg.iterations as usize);

    let mut it = 0;
    while it < cfg.iterations {
        // Global step: assemble the right-hand side from (z - u).
        let mut rhs = Vec::with_capacity(n);
        for sn_i in &sn {
            rhs.push(sn_i.scale(mass_over_h2));
        }
        for (c, atom) in atoms.iter().enumerate() {
            let target = z[c].sub(u[c]);
            for &(idx, coeff) in &atom.terms {
                rhs[idx] = rhs[idx].add(target.scale(atom.weight * coeff));
            }
        }
        state = system.solve(&rhs);

        // Local (z) step and dual update, accumulating residuals.
        let mut primal_sq = 0.0f32;
        let mut dual_sq = 0.0f32;
        for (c, atom) in atoms.iter().enumerate() {
            let dx = atom.eval(&state);
            let z_prev = z[c];
            let z_new = atom.project(dx.add(u[c]));
            z[c] = z_new;
            u[c] = u[c].add(dx.sub(z_new));

            primal_sq += atom.weight * dx.sub(z_new).length_squared();
            dual_sq += atom.weight * z_new.sub(z_prev).length_squared();
        }
        primal_residuals.push(primal_sq.sqrt());
        dual_residuals.push(dual_sq.sqrt());
        it += 1;
    }

    AdmmSolve {
        positions: state,
        primal_residuals,
        dual_residuals,
    }
}

/// Clamp to a finite value (non-finite -> `0`).
fn sanitize_finite(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Clamp to a finite `>= 0` value (non-finite -> `0`).
fn sanitize_nonneg(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// Clamp to a finite `> 0` value, falling back to `default` otherwise.
fn sanitize_pos(x: f32, default: f32) -> f32 {
    if x.is_finite() && x > 0.0 {
        x
    } else {
        default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T_EPS: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < T_EPS
    }

    fn vclose(a: Vec3, b: Vec3) -> bool {
        a.sub(b).length() < T_EPS
    }

    #[test]
    fn empty_input_is_noop() {
        let out = solve_projective(&[], &ConstraintSet::new(), SolverConfig::default());
        assert!(out.positions.is_empty());
        assert!(out.residuals.is_empty());
        let admm = solve_admm(&[], &ConstraintSet::new(), SolverConfig::default());
        assert!(admm.positions.is_empty());
        assert!(admm.primal_residuals.is_empty());
    }

    #[test]
    fn empty_constraints_leaves_positions() {
        let initial = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(-4.0, 5.0, 0.5)];
        let out = solve_projective(&initial, &ConstraintSet::new(), SolverConfig::default());
        // With no constraints the solve just relaxes toward s_n = initial.
        assert!(vclose(out.positions[0], initial[0]));
        assert!(vclose(out.positions[1], initial[1]));
        assert_eq!(out.residuals.len(), 20);
    }

    #[test]
    fn local_project_edge_sets_rest_length_about_midpoint() {
        let xi = Vec3::new(0.0, 0.0, 0.0);
        let xj = Vec3::new(4.0, 0.0, 0.0);
        let (ni, nj) = local_project_edge(xi, xj, 1.0);
        // Midpoint preserved, length exactly rest.
        let mid = ni.add(nj).scale(0.5);
        assert!(vclose(mid, Vec3::new(2.0, 0.0, 0.0)));
        assert!(close(ni.sub(nj).length(), 1.0));
    }

    #[test]
    fn degenerate_edge_projection_is_finite() {
        let (ni, nj) = local_project_edge(Vec3::ZERO, Vec3::ZERO, 2.0);
        assert!(close(ni.sub(nj).length(), 2.0));
        for v in [ni.x, ni.y, ni.z, nj.x, nj.y, nj.z] {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn single_edge_converges_to_rest_length() {
        let initial = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        // A large but not ill-conditioned stiffness pulls the edge onto its
        // rest length; the inertia term leaves a tiny, bounded residual.
        let cs = ConstraintSet::single_edge(EdgeConstraint::new(0, 1, 1.0, 1.0e3));
        let cfg = SolverConfig::new(1.0, 60, 1.0);
        let out = solve_projective(&initial, &cs, cfg);
        let len = out.positions[0].sub(out.positions[1]).length();
        assert!((len - 1.0).abs() < 2e-3, "len={len}");
        // Center of mass is preserved by the symmetric solve.
        let com = out.positions[0].add(out.positions[1]).scale(0.5);
        assert!(com.sub(Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn residuals_monotone_non_increasing() {
        let initial = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.3, 0.0),
            Vec3::new(4.0, -0.2, 0.1),
        ];
        let cs = ConstraintSet {
            edges: alloc::vec![
                EdgeConstraint::new(0, 1, 1.0, 50.0),
                EdgeConstraint::new(1, 2, 1.0, 50.0),
            ],
            bends: alloc::vec![BendConstraint::new(0, 1, 2, Vec3::ZERO, 10.0)],
        };
        let out = solve_projective(&initial, &cs, SolverConfig::new(1.0, 30, 1.0));
        assert_eq!(out.residuals.len(), 30);
        let mut prev = out.residuals[0];
        for &e in &out.residuals {
            assert!(e <= prev + T_EPS, "e={e} prev={prev}");
            assert!(e.is_finite());
            prev = e;
        }
    }

    #[test]
    fn fixed_iteration_budget() {
        let initial = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let cs = ConstraintSet::single_edge(EdgeConstraint::new(0, 1, 1.0, 10.0));
        for iters in [0u32, 1, 5, 17] {
            let out = solve_projective(&initial, &cs, SolverConfig::new(1.0, iters, 1.0));
            assert_eq!(out.residuals.len(), iters as usize);
        }
    }

    #[test]
    fn prefactored_solve_satisfies_system() {
        let cs = ConstraintSet {
            edges: alloc::vec![
                EdgeConstraint::new(0, 1, 1.0, 3.0),
                EdgeConstraint::new(1, 2, 1.0, 2.0),
            ],
            bends: Vec::new(),
        };
        let sys = GlobalSystem::factor(&cs, 3, SolverConfig::new(1.0, 1, 1.0));
        let rhs = [
            Vec3::new(1.0, 0.5, -2.0),
            Vec3::new(-3.0, 2.0, 1.0),
            Vec3::new(0.0, -1.0, 4.0),
        ];
        let x = sys.solve(&rhs);
        let ax = sys.mul(&x);
        for i in 0..3 {
            assert!(vclose(ax[i], rhs[i]), "row {i}");
        }
    }

    #[test]
    fn cholesky_matches_cg() {
        let cs = ConstraintSet {
            edges: alloc::vec![
                EdgeConstraint::new(0, 1, 1.0, 4.0),
                EdgeConstraint::new(1, 2, 1.0, 4.0),
                EdgeConstraint::new(2, 3, 1.0, 4.0),
            ],
            bends: alloc::vec![BendConstraint::new(0, 1, 2, Vec3::ZERO, 1.5)],
        };
        let sys = GlobalSystem::factor(&cs, 4, SolverConfig::new(1.0, 1, 1.0));
        let rhs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(-1.0, 0.0, 3.0),
            Vec3::new(2.0, -2.0, 1.0),
        ];
        let direct = sys.solve(&rhs);
        let iterative = sys.solve_cg(&rhs);
        for i in 0..4 {
            assert!(vclose(direct[i], iterative[i]), "row {i}");
        }
    }

    #[test]
    fn symmetric_system_solution_correct() {
        // One edge of weight 2 between two particles, mass/h^2 = 1:
        // A = [[3, -2], [-2, 3]]. Solve A x = (1,0,0),(0,0,0) -> x0=0.6, x1=0.4.
        let cs = ConstraintSet::single_edge(EdgeConstraint::new(0, 1, 1.0, 2.0));
        let sys = GlobalSystem::factor(&cs, 2, SolverConfig::new(1.0, 1, 1.0));
        let rhs = [Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO];
        let x = sys.solve(&rhs);
        assert!(vclose(x[0], Vec3::new(0.6, 0.0, 0.0)));
        assert!(vclose(x[1], Vec3::new(0.4, 0.0, 0.0)));
    }

    #[test]
    fn config_sanitizes_nonfinite() {
        let c = SolverConfig::new(f32::NAN, 10, -5.0).sanitized();
        assert!(c.dt.is_finite() && c.dt > 0.0);
        assert!(c.mass.is_finite() && c.mass > 0.0);
        assert!(c.mass_over_h2().is_finite() && c.mass_over_h2() > 0.0);
    }

    #[test]
    fn nonfinite_inputs_sanitized_no_nan() {
        let initial = [
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::new(f32::INFINITY, 2.0, f32::NEG_INFINITY),
        ];
        let cs = ConstraintSet::single_edge(EdgeConstraint::new(0, 1, f32::NAN, f32::INFINITY));
        let out = solve_projective(&initial, &cs, SolverConfig::new(1.0, 10, 1.0));
        for p in &out.positions {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
        for &r in &out.residuals {
            assert!(r.is_finite());
        }
    }

    #[test]
    fn out_of_range_constraints_are_dropped() {
        // Index 5 does not exist for a 2-particle system; must not panic.
        let initial = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let cs = ConstraintSet {
            edges: alloc::vec![
                EdgeConstraint::new(0, 5, 1.0, 10.0),
                EdgeConstraint::new(1, 1, 1.0, 10.0),
            ],
            bends: Vec::new(),
        };
        let out = solve_projective(&initial, &cs, SolverConfig::new(1.0, 5, 1.0));
        // No valid constraints -> positions relax to the inertial target.
        assert!(vclose(out.positions[0], initial[0]));
        assert!(vclose(out.positions[1], initial[1]));
    }

    #[test]
    fn admm_residuals_decrease_and_converge() {
        let initial = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        let cs = ConstraintSet::single_edge(EdgeConstraint::new(0, 1, 1.0, 50.0));
        let out = solve_admm(&initial, &cs, SolverConfig::new(1.0, 60, 1.0));
        assert_eq!(out.primal_residuals.len(), 60);
        assert_eq!(out.dual_residuals.len(), 60);
        for &r in &out.primal_residuals {
            assert!(r.is_finite() && r >= 0.0);
        }
        // Overall convergence: the final primal residual is well below the first.
        let first = out.primal_residuals[0];
        let last = out.primal_residuals[out.primal_residuals.len() - 1];
        assert!(last <= first + T_EPS, "first={first} last={last}");
        assert!(last < 1e-2, "last={last}");
        // And the edge is close to its rest length.
        let len = out.positions[0].sub(out.positions[1]).length();
        assert!((len - 1.0).abs() < 1e-2, "len={len}");
    }

    #[test]
    fn bend_constraint_targets_rest_laplacian() {
        // Three colinear particles with a straight (zero) rest Laplacian and a
        // kink in the middle; the bend term should straighten it.
        let initial = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let cs = ConstraintSet {
            edges: Vec::new(),
            bends: alloc::vec![BendConstraint::new(0, 1, 2, Vec3::ZERO, 1.0e5)],
        };
        let out = solve_projective(&initial, &cs, SolverConfig::new(1.0, 40, 1.0));
        let lap = out.positions[1]
            .sub(out.positions[0].scale(0.5))
            .sub(out.positions[2].scale(0.5));
        assert!(lap.length() < 1e-2, "lap_len={}", lap.length());
    }
}
