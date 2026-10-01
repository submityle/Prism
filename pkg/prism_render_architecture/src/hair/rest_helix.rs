//! Deterministic rest-state construction for curly (`type-3`/`type-4` textured)
//! hair: a helical rest pose plus direction-dependent (anisotropic) bending
//! compliance.
//!
//! Straight-hair rest states are a line of control points with zero rest
//! curvature; curly hair is fundamentally different. A tightly coiled fibre
//! stores a *helical* rest shape, and its bending resistance is **anisotropic**:
//! deforming the coil along the fibre tangent (tightening/loosening the curl)
//! is far cheaper than bending it out of its coil plane. A single scalar bend
//! stiffness cannot represent that, so this module emits two per-segment
//! compliances (tangential vs normal). This is the architecture-side golden the
//! simulator consumes: it feeds the `helix` vertices as the initial strand, the
//! per-segment rest lengths as distance constraints, the per-joint rest
//! `darboux` (discrete curvature) vectors as the Cosserat/bend rest targets, and
//! the two compliance arrays as the anisotropic `XPBD` bend compliances. It
//! pairs with the guide-strand curl of [`crate::hair::interpolation`] and the
//! `Cosserat` twist used by the solver, but it is deliberately **self-contained**
//! (its own [`Vec3`], its own parameters) and shares no types with those
//! modules, so the rest-state math can be golden-tested in isolation.
//!
//! ## Trig-free `helix` construction
//!
//! The repository's `clippy.toml` forbids every `f32` trigonometric function
//! (`sin`/`cos`/`tan`/`asin`/... and the hyperbolic family) for `libm`
//! determinism, so this module advances around the coil's circular cross-section
//! **without any trigonometry**. The transverse offset from the coil axis is
//! kept as a 2D vector `(x, y)` in a fixed orthonormal plane. Each segment
//! rotates that offset by one fixed step using a caller-supplied unit complex
//! number `(cos_step, sin_step)` and a plain complex multiply:
//!
//! ```text
//! (x', y') = (x * cos_step - y * sin_step, x * sin_step + y * cos_step)
//! ```
//!
//! This is just add/sub/mul, so it is bit-reproducible and never calls a
//! transcendental. The step angle lives entirely in the caller-provided
//! `(cos_step, sin_step)` pair (for example `(0.0, 1.0)` is a 90-degree step and
//! `(0.8660254, 0.5)` is a 30-degree step); this module never computes it from an
//! angle, it only *re-normalizes* the pair to unit length so the coil radius is
//! conserved exactly as the offset is stepped around the circle. Advancing along
//! the coil axis by a fixed pitch per segment while rotating the transverse
//! offset traces the `helix`. Squares are written `x * x`; the only
//! floating-point primitives used are `sqrt`/`abs`/`clamp`/`min`/`max`, and no
//! `f32` equality (`==`/`!=`) is ever performed (comparisons use ordering or an
//! epsilon on the absolute difference).

#![forbid(unsafe_code)]

use alloc::vec::Vec;

/// Hard cap on the number of coil segments, so an absurd or hostile
/// `segments` request cannot allocate without bound. Sanitation clamps into
/// `1..=MAX_SEGMENTS`.
pub const MAX_SEGMENTS: usize = 4096;

/// Base (isotropic) bending compliance the two anisotropic compliances are
/// derived from. Compliance is inverse stiffness, so a larger value means a
/// floppier rest bond.
pub const BASE_BEND_COMPLIANCE: f32 = 1.0;

/// Smallest allowed `bend_stiffness_ratio` (tangential-to-normal stiffness
/// ratio). Kept strictly positive so the tangential compliance (which divides
/// by the ratio) stays finite.
pub const RATIO_MIN: f32 = 0.03125;

/// Largest allowed `bend_stiffness_ratio`.
pub const RATIO_MAX: f32 = 32.0;

/// Default `bend_stiffness_ratio` used when the input is non-finite; `1.0` is
/// the isotropic case (tangential compliance equals normal compliance).
pub const RATIO_DEFAULT: f32 = 1.0;

/// Squared-length threshold below which a vector is treated as degenerate (not
/// normalizable). Chosen well above the subnormal range so normalization never
/// divides by an almost-zero length.
const NORMALIZE_EPS_SQ: f32 = 1.0e-24;

/// Fallback coil-axis direction used when the supplied root tangent is zero or
/// non-finite, so the construction always has a well-defined axis.
const FALLBACK_AXIS: Vec3 = Vec3::new(0.0, 1.0, 0.0);

/// A minimal 3-component vector for the rest-state math.
///
/// Defined locally so the module needs no linear-algebra dependency; every
/// operation is plain `f32` arithmetic in a fixed evaluation order, which is
/// what makes the construction bit-for-bit reproducible across runs and targets.
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
        reason = "This internal type exposes named add/sub methods for call-site uniformity with the rest of the hair architecture; operator traits are intentionally omitted."
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

    /// Cross product `self x rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
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

    /// This vector scaled to unit length, or [`Vec3::ZERO`] if it is too short
    /// to normalize or non-finite. Never divides by an almost-zero length.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq.is_finite() && len_sq > NORMALIZE_EPS_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }

    /// This vector with every non-finite component (`NaN`/infinity) replaced by
    /// `0`, so a poisoned input can never reach the rest-state math.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self::new(
            sanitize_finite(self.x),
            sanitize_finite(self.y),
            sanitize_finite(self.z),
        )
    }
}

/// Builds an orthonormal pair `(normal, binormal)` spanning the plane
/// perpendicular to `axis`.
///
/// The returned vectors are mutually perpendicular unit vectors, each
/// perpendicular to `axis`, with `binormal == axis x normal`. The helper axis
/// is chosen as whichever cardinal direction is least aligned with `axis`, so
/// the seeding cross product is well-conditioned. A zero or non-finite `axis`
/// falls back to [`FALLBACK_AXIS`], so the result is always a valid basis and
/// the routine is trig-free and panic-free.
#[must_use]
pub fn build_orthonormal_basis(axis: Vec3) -> (Vec3, Vec3) {
    let a = {
        let n = axis.normalize_or_zero();
        if n.length_squared() > NORMALIZE_EPS_SQ {
            n
        } else {
            FALLBACK_AXIS
        }
    };
    let helper = if a.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let normal = {
        let n = a.cross(helper).normalize_or_zero();
        if n.length_squared() > NORMALIZE_EPS_SQ {
            n
        } else {
            // `a` was parallel to the helper despite the 0.9 guard; use the
            // other cardinal axis, which cannot also be parallel.
            a.cross(Vec3::new(0.0, 0.0, 1.0)).normalize_or_zero()
        }
    };
    let binormal = a.cross(normal);
    (normal, binormal)
}

/// Parameters describing one coil (`helix`) rest shape.
///
/// The angular step around the coil cross-section is supplied as a unit complex
/// number `(rot_cos_step, rot_sin_step)` rather than an angle, so the
/// construction stays trig-free (see the module docs). Every field is
/// range-checked by [`CurlParams::sanitized`] before use; illegal values fall
/// back to safe defaults and never panic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurlParams {
    /// Coil radius (distance of the fibre from the coil axis). Clamped to
    /// `>= 0`; `0` degenerates to straight hair.
    pub radius: f32,
    /// Advance along the coil axis per segment (the `helix` pitch per step).
    /// Any finite value is accepted; non-finite falls back to `0`.
    pub pitch_per_segment: f32,
    /// Number of coil segments. Clamped into `1..=MAX_SEGMENTS`; the vertex
    /// count is `segments + 1`.
    pub segments: usize,
    /// Real part of the per-segment unit-complex rotation step.
    pub rot_cos_step: f32,
    /// Imaginary part of the per-segment unit-complex rotation step.
    pub rot_sin_step: f32,
    /// Ratio of tangential bending stiffness to normal bending stiffness.
    /// Clamped into `RATIO_MIN..=RATIO_MAX`; `1.0` is isotropic.
    pub bend_stiffness_ratio: f32,
}

impl Default for CurlParams {
    fn default() -> Self {
        Self {
            radius: 0.0,
            pitch_per_segment: 0.0,
            segments: 1,
            rot_cos_step: 1.0,
            rot_sin_step: 0.0,
            bend_stiffness_ratio: RATIO_DEFAULT,
        }
    }
}

impl CurlParams {
    /// Constructs coil parameters from explicit fields (unchecked; call
    /// [`CurlParams::sanitized`] before use, which [`build_rest_helix`] does
    /// automatically).
    #[must_use]
    pub const fn new(
        radius: f32,
        pitch_per_segment: f32,
        segments: usize,
        rot_cos_step: f32,
        rot_sin_step: f32,
        bend_stiffness_ratio: f32,
    ) -> Self {
        Self {
            radius,
            pitch_per_segment,
            segments,
            rot_cos_step,
            rot_sin_step,
            bend_stiffness_ratio,
        }
    }

    /// These parameters with every field forced into its legal range.
    ///
    /// `segments` is clamped into `1..=MAX_SEGMENTS`; `radius` is forced
    /// finite and `>= 0`; `pitch_per_segment` is forced finite; the rotation
    /// step is re-normalized to a unit complex number (falling back to the
    /// identity `(1, 0)` if it is degenerate or non-finite); and
    /// `bend_stiffness_ratio` is forced finite and clamped into
    /// `RATIO_MIN..=RATIO_MAX`. The result always produces a finite,
    /// panic-free [`RestHelix`].
    #[must_use]
    pub fn sanitized(self) -> Self {
        let segments = if self.segments == 0 {
            1
        } else if self.segments > MAX_SEGMENTS {
            MAX_SEGMENTS
        } else {
            self.segments
        };

        let radius = if self.radius.is_finite() && self.radius > 0.0 {
            self.radius
        } else {
            0.0
        };

        let pitch_per_segment = sanitize_finite(self.pitch_per_segment);

        let (rot_cos_step, rot_sin_step) =
            sanitize_unit_complex(self.rot_cos_step, self.rot_sin_step);

        let bend_stiffness_ratio = if self.bend_stiffness_ratio.is_finite() {
            self.bend_stiffness_ratio.clamp(RATIO_MIN, RATIO_MAX)
        } else {
            RATIO_DEFAULT
        };

        Self {
            radius,
            pitch_per_segment,
            segments,
            rot_cos_step,
            rot_sin_step,
            bend_stiffness_ratio,
        }
    }
}

/// The deterministic rest state of one curly fibre.
///
/// All arrays are index-aligned to the coil: `positions` has `segments + 1`
/// vertices (vertex `0` is anchored on the coil axis at `root`), `rest_lengths`
/// has one entry per segment, `rest_darboux` has one entry per interior joint
/// (`segments - 1`, empty for a single segment), and the two compliance arrays
/// have one entry per segment. Every value is finite.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RestHelix {
    /// Rest control-point positions along the `helix` (`segments + 1` of them).
    pub positions: Vec<Vec3>,
    /// Rest length of each coil segment (`segments` of them).
    pub rest_lengths: Vec<f32>,
    /// Per-interior-joint discrete curvature (`darboux`) vector: the curvature
    /// binormal `2 * (e_prev x e_cur) / (|e_prev||e_cur| + e_prev . e_cur)`
    /// between consecutive edges. Its magnitude is the discrete bend angle
    /// measure and is constant for an equal-pitch coil. `segments - 1` of them.
    pub rest_darboux: Vec<Vec3>,
    /// Per-segment bending compliance about the coil tangent direction
    /// (tightening/loosening the curl). `segments` of them.
    pub tangential_compliance: Vec<f32>,
    /// Per-segment bending compliance about the coil normal direction (bending
    /// out of the coil plane). `segments` of them.
    pub normal_compliance: Vec<f32>,
}

/// Builds the helical rest state for a curly fibre rooted at `root` growing
/// along `root_tangent`, using the coil parameters `params`.
///
/// The coil axis passes through `root` along the (normalized) tangent. Starting
/// from a transverse offset of `radius` along the first basis direction, each
/// segment advances `pitch_per_segment` along the axis while rotating the
/// transverse offset by the unit-complex step (see the module docs for the
/// trig-free stepping). The per-segment rest lengths, per-joint discrete
/// `darboux` curvature, and the anisotropic tangential/normal compliances are
/// derived from the resulting vertices.
///
/// Every input is sanitized first (`segments == 0` becomes `1`, non-finite
/// values are replaced by safe defaults, a zero/non-finite tangent falls back to
/// [`FALLBACK_AXIS`]), so the routine is total and panic-free and always returns
/// finite data. Running it twice on the same inputs returns identical results.
#[must_use]
pub fn build_rest_helix(root: Vec3, root_tangent: Vec3, params: CurlParams) -> RestHelix {
    let params = params.sanitized();
    let root = root.sanitized();

    let axis = {
        let t = root_tangent.sanitized().normalize_or_zero();
        if t.length_squared() > NORMALIZE_EPS_SQ {
            t
        } else {
            FALLBACK_AXIS
        }
    };
    let (basis_u, basis_v) = build_orthonormal_basis(axis);

    let segments = params.segments;
    let vertex_count = segments + 1;

    // Transverse offset in the (basis_u, basis_v) plane, stepped as a complex
    // number. It starts at (radius, 0) and is conserved in magnitude.
    let mut offset_x = params.radius;
    let mut offset_y = 0.0f32;

    let mut positions = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let along = params.pitch_per_segment * (i as f32);
        let transverse = basis_u.scale(offset_x).add(basis_v.scale(offset_y));
        let vertex = root.add(axis.scale(along)).add(transverse);
        positions.push(vertex.sanitized());

        // Advance the offset by one unit-complex rotation step for the next
        // vertex. Pure add/sub/mul: zero trigonometry.
        let next_x = offset_x * params.rot_cos_step - offset_y * params.rot_sin_step;
        let next_y = offset_x * params.rot_sin_step + offset_y * params.rot_cos_step;
        offset_x = next_x;
        offset_y = next_y;
    }

    let mut rest_lengths = Vec::with_capacity(segments);
    for i in 0..segments {
        let edge = positions[i + 1].sub(positions[i]);
        rest_lengths.push(sanitize_finite(edge.length()));
    }

    let joint_count = segments.saturating_sub(1);
    let mut rest_darboux = Vec::with_capacity(joint_count);
    for i in 0..joint_count {
        let e_prev = positions[i + 1].sub(positions[i]);
        let e_cur = positions[i + 2].sub(positions[i + 1]);
        rest_darboux.push(curvature_binormal(e_prev, e_cur));
    }

    let (tangential, normal) = anisotropic_compliances(params.bend_stiffness_ratio);
    let mut tangential_compliance = Vec::with_capacity(segments);
    let mut normal_compliance = Vec::with_capacity(segments);
    for _ in 0..segments {
        tangential_compliance.push(tangential);
        normal_compliance.push(normal);
    }

    RestHelix {
        positions,
        rest_lengths,
        rest_darboux,
        tangential_compliance,
        normal_compliance,
    }
}

/// Discrete curvature binormal between two consecutive edge vectors.
///
/// Uses the trig-free discrete-elastic-rod form
/// `2 * (e_prev x e_cur) / (|e_prev||e_cur| + e_prev . e_cur)`, which equals
/// `2 * tan(theta / 2)` along the turning axis without ever evaluating a
/// trigonometric function. The denominator is guarded: if the edges are
/// degenerate or exactly reversed (so it collapses toward zero) the curvature
/// is reported as [`Vec3::ZERO`]. The result is always finite.
#[must_use]
fn curvature_binormal(e_prev: Vec3, e_cur: Vec3) -> Vec3 {
    let len_prev = e_prev.length();
    let len_cur = e_cur.length();
    let denom = len_prev * len_cur + e_prev.dot(e_cur);
    if denom.is_finite() && denom.abs() > NORMALIZE_EPS_SQ {
        e_prev.cross(e_cur).scale(2.0 / denom).sanitized()
    } else {
        Vec3::ZERO
    }
}

/// Splits the base bending compliance into tangential and normal parts for a
/// given tangential-to-normal stiffness `ratio`.
///
/// `normal_compliance = BASE_BEND_COMPLIANCE` and
/// `tangential_compliance = BASE_BEND_COMPLIANCE / ratio`, since compliance is
/// inverse stiffness: a larger `ratio` means a stiffer (lower-compliance)
/// tangent. `ratio == 1` yields the isotropic case where both are equal. The
/// caller passes an already-clamped (finite, strictly positive) ratio.
#[must_use]
fn anisotropic_compliances(ratio: f32) -> (f32, f32) {
    let normal = BASE_BEND_COMPLIANCE;
    let tangential = BASE_BEND_COMPLIANCE / ratio;
    (sanitize_finite(tangential), sanitize_finite(normal))
}

/// Re-normalizes a 2D `(cos, sin)` pair to a unit complex number.
///
/// Returns the identity rotation `(1, 0)` when the input is non-finite or too
/// short to normalize, so the stepping always uses a length-preserving
/// (radius-conserving) rotation.
#[must_use]
fn sanitize_unit_complex(cos: f32, sin: f32) -> (f32, f32) {
    if cos.is_finite() && sin.is_finite() {
        let mag_sq = cos * cos + sin * sin;
        if mag_sq.is_finite() && mag_sq > NORMALIZE_EPS_SQ {
            let inv = 1.0 / mag_sq.sqrt();
            return (cos * inv, sin * inv);
        }
    }
    (1.0, 0.0)
}

/// Returns `x` if it is finite, otherwise `0`, so `NaN`/infinity can never flow
/// into the rest-state arrays.
#[must_use]
fn sanitize_finite(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-difference float comparison (no `f32` `==`/`!=`).
    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn vec_approx(a: Vec3, b: Vec3, eps: f32) -> bool {
        approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
    }

    #[test]
    fn vec3_algebra_golden() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-4.0, 0.5, 2.0);
        assert!(vec_approx(a.add(b), Vec3::new(-3.0, 2.5, 5.0), 1.0e-6));
        assert!(vec_approx(a.sub(b), Vec3::new(5.0, 1.5, 1.0), 1.0e-6));
        assert!(vec_approx(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0), 1.0e-6));
        assert!(approx(a.dot(b), 1.0 * -4.0 + 2.0 * 0.5 + 3.0 * 2.0, 1.0e-6));
        // Right-handed cross product of the basis axes.
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert!(vec_approx(x.cross(y), Vec3::new(0.0, 0.0, 1.0), 1.0e-6));
        assert!(approx(Vec3::new(3.0, 4.0, 0.0).length(), 5.0, 1.0e-6));
        assert!(vec_approx(
            Vec3::new(0.0, 0.0, 5.0).normalize_or_zero(),
            Vec3::new(0.0, 0.0, 1.0),
            1.0e-6
        ));
        // Degenerate / non-finite normalization yields zero, not a panic.
        assert!(vec_approx(
            Vec3::ZERO.normalize_or_zero(),
            Vec3::ZERO,
            1.0e-6
        ));
        assert!(vec_approx(
            Vec3::new(f32::NAN, 1.0, f32::INFINITY).sanitized(),
            Vec3::new(0.0, 1.0, 0.0),
            1.0e-6
        ));
    }

    #[test]
    fn orthonormal_basis_is_orthonormal() {
        let axes = [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 2.0, -2.0).normalize_or_zero(),
            // Degenerate axis must still yield a valid basis via the fallback.
            Vec3::ZERO,
        ];
        for axis in axes {
            let used = {
                let n = axis.normalize_or_zero();
                if n.length_squared() > 1.0e-24 {
                    n
                } else {
                    FALLBACK_AXIS
                }
            };
            let (n, b) = build_orthonormal_basis(axis);
            assert!(approx(n.length(), 1.0, 1.0e-5));
            assert!(approx(b.length(), 1.0, 1.0e-5));
            assert!(approx(n.dot(b), 0.0, 1.0e-5));
            assert!(approx(n.dot(used), 0.0, 1.0e-5));
            assert!(approx(b.dot(used), 0.0, 1.0e-5));
        }
    }

    #[test]
    fn complex_step_circle_returns_and_preserves_radius() {
        // A 90-degree step (cos=0, sin=1) with zero pitch keeps every vertex on
        // the circle of `radius` about the root, and four steps return to the
        // start.
        let radius = 3.0;
        let params = CurlParams::new(radius, 0.0, 4, 0.0, 1.0, 1.0);
        let root = Vec3::new(5.0, -1.0, 2.0);
        let helix = build_rest_helix(root, Vec3::new(0.0, 1.0, 0.0), params);
        assert_eq!(helix.positions.len(), 5);
        for p in &helix.positions {
            assert!(approx(p.sub(root).length(), radius, 1.0e-5));
        }
        // Vertex 4 coincides with vertex 0 (full turn, zero pitch).
        assert!(vec_approx(helix.positions[4], helix.positions[0], 1.0e-5));
    }

    #[test]
    fn straight_hair_radius_zero_is_a_line() {
        let pitch = 1.5;
        let params = CurlParams::new(0.0, pitch, 6, 0.0, 1.0, 1.0);
        let root = Vec3::new(0.0, 0.0, 0.0);
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let helix = build_rest_helix(root, axis, params);
        for (i, p) in helix.positions.iter().enumerate() {
            let expected = root.add(axis.scale(pitch * (i as f32)));
            assert!(vec_approx(*p, expected, 1.0e-5));
        }
        for l in &helix.rest_lengths {
            assert!(approx(*l, pitch, 1.0e-5));
        }
        // A straight line has (near) zero discrete curvature at every joint.
        for d in &helix.rest_darboux {
            assert!(approx(d.length(), 0.0, 1.0e-5));
        }
    }

    #[test]
    fn helix_segment_length_matches_analytic() {
        // 30-degree step: cos=0.8660254, sin=0.5.
        let radius = 3.0;
        let pitch = 2.0;
        let c = 0.8660254_f32;
        let s = 0.5_f32;
        let params = CurlParams::new(radius, pitch, 8, c, s, 1.0);
        let helix = build_rest_helix(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), params);
        // Segment vector = axis * pitch + (rotated_offset - offset); the
        // transverse part has length radius * sqrt(2 - 2c) and is perpendicular
        // to the axis, so the segment length is sqrt(pitch^2 + r^2 (2 - 2c)).
        let expected = (pitch * pitch + radius * radius * (2.0 - 2.0 * c)).sqrt();
        for l in &helix.rest_lengths {
            assert!(approx(*l, expected, 1.0e-4));
        }
    }

    #[test]
    fn positions_advance_monotonically_along_axis() {
        let pitch = 0.75;
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let params = CurlParams::new(2.0, pitch, 10, 0.8660254, 0.5, 1.0);
        let root = Vec3::new(1.0, 2.0, 3.0);
        let helix = build_rest_helix(root, axis, params);
        let mut prev = f32::NEG_INFINITY;
        for (i, p) in helix.positions.iter().enumerate() {
            let along = p.sub(root).dot(axis);
            assert!(approx(along, pitch * (i as f32), 1.0e-4));
            assert!(along > prev);
            prev = along;
        }
    }

    #[test]
    fn equal_pitch_darboux_is_constant() {
        // A uniform coil has constant discrete-curvature magnitude at every
        // interior joint (the binormal direction rotates, the magnitude does
        // not).
        let params = CurlParams::new(2.5, 1.0, 12, 0.8660254, 0.5, 1.0);
        let helix = build_rest_helix(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), params);
        assert_eq!(helix.rest_darboux.len(), 11);
        let first = helix.rest_darboux[0].length();
        assert!(first > 1.0e-4);
        for d in &helix.rest_darboux {
            assert!(approx(d.length(), first, 1.0e-4));
        }
    }

    #[test]
    fn anisotropic_compliance_differs_and_tracks_ratio() {
        let make = |ratio: f32| {
            let params = CurlParams::new(1.0, 1.0, 3, 0.8660254, 0.5, ratio);
            build_rest_helix(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), params)
        };
        let stiff_tangent = make(4.0);
        assert_eq!(stiff_tangent.tangential_compliance.len(), 3);
        assert_eq!(stiff_tangent.normal_compliance.len(), 3);
        for i in 0..3 {
            let t = stiff_tangent.tangential_compliance[i];
            let n = stiff_tangent.normal_compliance[i];
            // Stiffer tangent -> lower tangential compliance than normal.
            assert!(t < n);
            // compliance = BASE / ratio, so t * ratio == BASE.
            assert!(approx(t * 4.0, BASE_BEND_COMPLIANCE, 1.0e-5));
            assert!(approx(n, BASE_BEND_COMPLIANCE, 1.0e-5));
        }
        // Changing the ratio changes the tangential compliance.
        let floppy_tangent = make(0.25);
        assert!(floppy_tangent.tangential_compliance[0] > stiff_tangent.tangential_compliance[0]);
        assert!(approx(
            floppy_tangent.tangential_compliance[0],
            BASE_BEND_COMPLIANCE / 0.25,
            1.0e-5
        ));
    }

    #[test]
    fn segments_are_capped() {
        let params = CurlParams::new(1.0, 1.0, MAX_SEGMENTS + 1000, 0.0, 1.0, 1.0);
        let helix = build_rest_helix(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), params);
        assert_eq!(helix.positions.len(), MAX_SEGMENTS + 1);
        assert_eq!(helix.rest_lengths.len(), MAX_SEGMENTS);
        assert_eq!(helix.rest_darboux.len(), MAX_SEGMENTS - 1);
        assert_eq!(helix.tangential_compliance.len(), MAX_SEGMENTS);
        assert_eq!(helix.normal_compliance.len(), MAX_SEGMENTS);
    }

    #[test]
    fn degenerate_params_do_not_panic() {
        // segments = 0 is promoted to 1; zero tangent uses the fallback axis;
        // identity rotation step is used when none is supplied.
        let params = CurlParams::new(1.0, 1.0, 0, 0.0, 0.0, 1.0);
        let helix = build_rest_helix(Vec3::ZERO, Vec3::ZERO, params);
        assert_eq!(helix.positions.len(), 2);
        assert_eq!(helix.rest_lengths.len(), 1);
        assert!(helix.rest_darboux.is_empty());
        for p in &helix.positions {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
    }

    #[test]
    fn non_finite_inputs_sanitize_to_finite() {
        let params = CurlParams::new(
            f32::NAN,
            f32::INFINITY,
            5,
            f32::NAN,
            f32::NAN,
            f32::INFINITY,
        );
        let root = Vec3::new(f32::NAN, f32::INFINITY, 1.0);
        let tangent = Vec3::new(f32::NAN, f32::NAN, f32::NAN);
        let helix = build_rest_helix(root, tangent, params);
        for p in &helix.positions {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
        for l in &helix.rest_lengths {
            assert!(l.is_finite());
        }
        for d in &helix.rest_darboux {
            assert!(d.x.is_finite() && d.y.is_finite() && d.z.is_finite());
        }
        for t in &helix.tangential_compliance {
            assert!(t.is_finite());
        }
        for n in &helix.normal_compliance {
            assert!(n.is_finite());
        }
    }

    #[test]
    fn build_is_deterministic() {
        let params = CurlParams::new(2.0, 1.25, 16, 0.8660254, 0.5, 3.0);
        let root = Vec3::new(0.3, -1.7, 4.2);
        let tangent = Vec3::new(0.2, 0.9, -0.1);
        let a = build_rest_helix(root, tangent, params);
        let b = build_rest_helix(root, tangent, params);
        assert_eq!(a.positions.len(), b.positions.len());
        for i in 0..a.positions.len() {
            // Bit-exact determinism (integer comparison on the bit patterns).
            assert_eq!(a.positions[i].x.to_bits(), b.positions[i].x.to_bits());
            assert_eq!(a.positions[i].y.to_bits(), b.positions[i].y.to_bits());
            assert_eq!(a.positions[i].z.to_bits(), b.positions[i].z.to_bits());
        }
        for i in 0..a.rest_lengths.len() {
            assert_eq!(a.rest_lengths[i].to_bits(), b.rest_lengths[i].to_bits());
        }
        for i in 0..a.rest_darboux.len() {
            assert_eq!(a.rest_darboux[i].x.to_bits(), b.rest_darboux[i].x.to_bits());
            assert_eq!(a.rest_darboux[i].y.to_bits(), b.rest_darboux[i].y.to_bits());
            assert_eq!(a.rest_darboux[i].z.to_bits(), b.rest_darboux[i].z.to_bits());
        }
    }

    #[test]
    fn radius_and_pitch_scaling_is_linear() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let root = Vec3::ZERO;
        let base = CurlParams::new(2.0, 1.0, 8, 0.8660254, 0.5, 1.0);
        let helix = build_rest_helix(root, axis, base);

        // Doubling the radius doubles the transverse (off-axis) displacement.
        let double_r = CurlParams::new(4.0, 1.0, 8, 0.8660254, 0.5, 1.0);
        let helix_r = build_rest_helix(root, axis, double_r);
        for i in 0..helix.positions.len() {
            let off = {
                let rel = helix.positions[i].sub(root);
                rel.sub(axis.scale(rel.dot(axis))).length()
            };
            let off_r = {
                let rel = helix_r.positions[i].sub(root);
                rel.sub(axis.scale(rel.dot(axis))).length()
            };
            assert!(approx(off_r, 2.0 * off, 1.0e-4));
        }

        // Doubling the pitch doubles the along-axis advance at each vertex.
        let double_p = CurlParams::new(2.0, 2.0, 8, 0.8660254, 0.5, 1.0);
        let helix_p = build_rest_helix(root, axis, double_p);
        for i in 0..helix.positions.len() {
            let along = helix.positions[i].sub(root).dot(axis);
            let along_p = helix_p.positions[i].sub(root).dot(axis);
            assert!(approx(along_p, 2.0 * along, 1.0e-4));
        }
    }

    #[test]
    fn curlparams_sanitize_clamps_bounds() {
        let p = CurlParams::new(-5.0, 2.0, 0, 3.0, 4.0, 1000.0).sanitized();
        assert!(approx(p.radius, 0.0, 1.0e-6));
        assert_eq!(p.segments, 1);
        // (3, 4) normalizes to (0.6, 0.8).
        assert!(approx(p.rot_cos_step, 0.6, 1.0e-5));
        assert!(approx(p.rot_sin_step, 0.8, 1.0e-5));
        assert!(approx(
            p.rot_cos_step * p.rot_cos_step + p.rot_sin_step * p.rot_sin_step,
            1.0,
            1.0e-5
        ));
        assert!(approx(p.bend_stiffness_ratio, RATIO_MAX, 1.0e-5));

        let q = CurlParams::new(1.0, 1.0, 2, 0.0, 0.0, -2.0).sanitized();
        // Degenerate rotation step falls back to identity.
        assert!(approx(q.rot_cos_step, 1.0, 1.0e-6));
        assert!(approx(q.rot_sin_step, 0.0, 1.0e-6));
        // Negative ratio clamps up to RATIO_MIN.
        assert!(approx(q.bend_stiffness_ratio, RATIO_MIN, 1.0e-6));
    }
}
