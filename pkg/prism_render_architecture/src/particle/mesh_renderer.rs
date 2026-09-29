//! Per-particle mesh-instance transform assembly for the `Mesh` renderer
//! (design §15): the `CPU` reference that turns a live particle plus a unit
//! source mesh into a `TRS` instance transform, a world-space bounds, and a
//! `LOD` tier.
//!
//! This is the mesh counterpart to the primitives in [`super::renderers`], and
//! it is deliberately *orthogonal* to them. The `Sprite` path there builds a
//! camera-relative [`super::renderers::BillboardBasis`] (a 2D quad that always
//! faces, or axis-locks toward, the camera); `Ribbon` and `Beam` build swept
//! segment strips. None of those describe a full three-dimensional orientation,
//! so a mesh instance cannot reuse them: a mesh is a rigid body that must be
//! oriented in all three axes, scaled per axis, and placed in the world. This
//! module therefore owns its own affine math ([`Mat3`], [`Affine3`]) and mesh
//! orientation modes ([`MeshOrientation`]) rather than re-deriving the sprite
//! billboard basis. It mirrors Unreal `Niagara`'s Mesh Renderer and Unity `VFX
//! Graph`'s mesh output at the algorithm level without reusing their code.
//!
//! Determinism follows the sibling modules: the only non-arithmetic operation
//! is `sqrt` (through [`Vec3`]), and every rotation angle enters as a numeric
//! `(sin, cos)` pair — no transcendental functions are ever called, so the
//! `CPU` reference stays bit-reproducible against a future `GPU` instancing
//! kernel. Where an angle is *derived* (aligning one direction onto another),
//! its sine and cosine come from a cross/dot product, not from `sin`/`cos`.

use super::sort_cull::Aabb;
use super::{Vec3, EPS_LEN_SQ};

/// Absolute tolerance for `f32` comparisons in this module.
///
/// `f32` equality is never tested with `==`/`!=`; callers and tests compare
/// against this tolerance instead. It is looser than the squared-length
/// threshold [`super::EPS_LEN_SQ`] because it applies to already-normalized
/// quantities (basis components, transformed coordinates) rather than to
/// squared magnitudes.
pub const EPS: f32 = 1e-6;

/// A row-major 3x3 matrix, used for the linear (rotation and scale) part of a
/// mesh instance transform.
///
/// Rows are stored as three [`Vec3`]s so a matrix-times-vector is three dot
/// products. A pure-rotation matrix built from an orthonormal basis has its
/// inverse equal to its transpose, which the instance-bounds and inverse paths
/// rely on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    /// First row.
    pub row0: Vec3,
    /// Second row.
    pub row1: Vec3,
    /// Third row.
    pub row2: Vec3,
}

impl Mat3 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self {
        row0: Vec3::new(1.0, 0.0, 0.0),
        row1: Vec3::new(0.0, 1.0, 0.0),
        row2: Vec3::new(0.0, 0.0, 1.0),
    };

    /// Builds a matrix from its three rows.
    #[must_use]
    pub const fn from_rows(row0: Vec3, row1: Vec3, row2: Vec3) -> Self {
        Self { row0, row1, row2 }
    }

    /// Builds a matrix from its three *columns*.
    ///
    /// This is the natural constructor for an orientation basis: the columns
    /// are the world-space images of the mesh's local `+X`/`+Y`/`+Z` axes, so
    /// `from_columns(x, y, z).mul_vec(Vec3::new(1.0, 0.0, 0.0))` returns `x`.
    #[must_use]
    pub const fn from_columns(col0: Vec3, col1: Vec3, col2: Vec3) -> Self {
        Self {
            row0: Vec3::new(col0.x, col1.x, col2.x),
            row1: Vec3::new(col0.y, col1.y, col2.y),
            row2: Vec3::new(col0.z, col1.z, col2.z),
        }
    }

    /// Builds a diagonal (per-axis scale) matrix.
    #[must_use]
    pub const fn from_diagonal(diagonal: Vec3) -> Self {
        Self {
            row0: Vec3::new(diagonal.x, 0.0, 0.0),
            row1: Vec3::new(0.0, diagonal.y, 0.0),
            row2: Vec3::new(0.0, 0.0, diagonal.z),
        }
    }

    /// Transforms a column vector: `self * v`.
    #[must_use]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3::new(self.row0.dot(v), self.row1.dot(v), self.row2.dot(v))
    }

    /// Matrix product `self * rhs` (named to avoid overloading `Mul`).
    #[must_use]
    pub fn mul_mat(self, rhs: Self) -> Self {
        // Transpose `rhs` so its rows are `rhs`'s columns, then each output
        // entry is a dot product of a row of `self` with a column of `rhs`.
        let t = rhs.transpose();
        Self {
            row0: Vec3::new(
                self.row0.dot(t.row0),
                self.row0.dot(t.row1),
                self.row0.dot(t.row2),
            ),
            row1: Vec3::new(
                self.row1.dot(t.row0),
                self.row1.dot(t.row1),
                self.row1.dot(t.row2),
            ),
            row2: Vec3::new(
                self.row2.dot(t.row0),
                self.row2.dot(t.row1),
                self.row2.dot(t.row2),
            ),
        }
    }

    /// The transpose, which is also the inverse for a pure-rotation matrix.
    #[must_use]
    pub fn transpose(self) -> Self {
        Self::from_columns(self.row0, self.row1, self.row2)
    }

    /// The entry-wise absolute value, used for the fast instance-bounds path.
    #[must_use]
    pub fn abs(self) -> Self {
        Self {
            row0: Vec3::new(self.row0.x.abs(), self.row0.y.abs(), self.row0.z.abs()),
            row1: Vec3::new(self.row1.x.abs(), self.row1.y.abs(), self.row1.z.abs()),
            row2: Vec3::new(self.row2.x.abs(), self.row2.y.abs(), self.row2.z.abs()),
        }
    }
}

/// An affine transform in three dimensions: a 3x3 linear part plus a
/// translation, i.e. the 3x4 instance transform a `GPU` instancing kernel
/// consumes.
///
/// Point transforms apply the linear part and then add the translation; vector
/// (direction) transforms skip the translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine3 {
    /// The rotation-and-scale linear part.
    pub linear: Mat3,
    /// The world-space translation (the particle position).
    pub translation: Vec3,
}

impl Affine3 {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        linear: Mat3::IDENTITY,
        translation: Vec3::ZERO,
    };

    /// Builds an affine transform from a linear part and a translation.
    #[must_use]
    pub const fn new(linear: Mat3, translation: Vec3) -> Self {
        Self {
            linear,
            translation,
        }
    }

    /// Assembles the canonical `TRS` instance transform `T * R * S`.
    ///
    /// The rotation `R` and scale `S` compose into the linear part and the
    /// translation `T` is applied last, matching the standard scale-then-rotate-
    /// then-translate order used by mesh-instancing renderers.
    #[must_use]
    pub fn from_trs(translation: Vec3, rotation: Mat3, scale: Mat3) -> Self {
        Self {
            linear: rotation.mul_mat(scale),
            translation,
        }
    }

    /// Transforms a point: `linear * p + translation`.
    #[must_use]
    pub fn transform_point(self, p: Vec3) -> Vec3 {
        self.linear.mul_vec(p).add(self.translation)
    }

    /// Transforms a direction: `linear * v` (translation is ignored).
    #[must_use]
    pub fn transform_vector(self, v: Vec3) -> Vec3 {
        self.linear.mul_vec(v)
    }

    /// Composes two affine transforms: `self * rhs` applies `rhs` first.
    #[must_use]
    pub fn compose(self, rhs: Self) -> Self {
        Self {
            linear: self.linear.mul_mat(rhs.linear),
            translation: self.linear.mul_vec(rhs.translation).add(self.translation),
        }
    }
}

/// Which local mesh axis an [`MeshOrientation::AlignToAxis`] mode locks onto a
/// world direction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LocalAxis {
    /// The mesh's local `+X` axis.
    PlusX,
    /// The mesh's local `+Y` axis.
    PlusY,
    /// The mesh's local `+Z` axis.
    PlusZ,
}

impl LocalAxis {
    /// The unit vector for this local axis.
    #[must_use]
    pub const fn unit(self) -> Vec3 {
        match self {
            LocalAxis::PlusX => Vec3::new(1.0, 0.0, 0.0),
            LocalAxis::PlusY => Vec3::new(0.0, 1.0, 0.0),
            LocalAxis::PlusZ => Vec3::new(0.0, 0.0, 1.0),
        }
    }
}

/// How a mesh instance is oriented per particle (design §15).
///
/// Unlike the sprite billboard modes in [`super::renderers`], every variant
/// here yields a full three-dimensional rotation, not a camera-relative quad
/// basis. All variants are degenerate-safe: a zero direction or axis collapses
/// to the identity rotation rather than producing a `NaN` basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeshOrientation {
    /// No rotation; the mesh keeps its authored local orientation.
    Identity,
    /// The mesh's local `+X` (forward) axis is aligned to `velocity`, with the
    /// remaining axes completed from `up_reference` via cross products.
    VelocityAligned {
        /// The per-particle velocity to align the local `+X` axis to.
        velocity: Vec3,
        /// The up reference used to complete the orthonormal basis.
        up_reference: Vec3,
    },
    /// A rotation of a numeric `(sin, cos)` angle about `axis`, evaluated with
    /// the Rodrigues formula. The angle is supplied as its sine and cosine, so
    /// no trigonometry is computed here.
    FixedRotation {
        /// The rotation axis (normalized internally; zero yields identity).
        axis: Vec3,
        /// The sine of the rotation angle.
        sin: f32,
        /// The cosine of the rotation angle.
        cos: f32,
    },
    /// A minimal rotation locking the mesh's `local_axis` onto the world
    /// direction `target`.
    AlignToAxis {
        /// The local mesh axis to lock.
        local_axis: LocalAxis,
        /// The world direction the local axis is rotated onto.
        target: Vec3,
    },
}

/// Returns a deterministic unit vector perpendicular to `v`, or `+X` when `v`
/// is (numerically) zero. Used to complete a basis when a cross product
/// collapses.
#[must_use]
fn any_perpendicular(v: Vec3) -> Vec3 {
    // Cross with the world axis least aligned with `v` to avoid a near-zero
    // cross product; squared components compared with no transcendentals.
    let ax = v.x * v.x;
    let ay = v.y * v.y;
    let az = v.z * v.z;
    let reference = if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let perp = v.cross(reference).normalize_or_zero();
    if perp.length_squared() > EPS_LEN_SQ {
        perp
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    }
}

/// Builds an orthonormal right-handed rotation whose local `+X` column points
/// along `forward`, completing `+Z` and `+Y` from `up_reference`.
///
/// Degenerate-safe: a zero `forward` falls back to world `+X`, and a
/// `forward`-parallel `up_reference` falls back to a deterministic
/// perpendicular, so the result is always orthonormal and never `NaN`.
#[must_use]
fn basis_from_forward_x(forward: Vec3, up_reference: Vec3) -> Mat3 {
    let mut x_axis = forward.normalize_or_zero();
    if x_axis.length_squared() <= EPS_LEN_SQ {
        x_axis = Vec3::new(1.0, 0.0, 0.0);
    }
    let mut z_axis = x_axis.cross(up_reference).normalize_or_zero();
    if z_axis.length_squared() <= EPS_LEN_SQ {
        z_axis = any_perpendicular(x_axis);
    }
    // `y = z x x` closes a right-handed frame: `x x y = z`.
    let y_axis = z_axis.cross(x_axis);
    Mat3::from_columns(x_axis, y_axis, z_axis)
}

/// The velocity-aligned rotation for a mesh instance.
///
/// The mesh's local `+X` axis is aligned to `velocity`; the other two axes are
/// completed from `up_reference`. A zero `velocity` falls back to world `+X`.
#[must_use]
pub fn velocity_aligned_basis(velocity: Vec3, up_reference: Vec3) -> Mat3 {
    basis_from_forward_x(velocity, up_reference)
}

/// The Rodrigues rotation matrix for a numeric `(sin, cos)` angle about `axis`.
///
/// The angle is *not* computed here: its sine and cosine are supplied directly,
/// keeping the `CPU` reference free of transcendental functions. `axis` is
/// normalized internally; a (numerically) zero axis yields the identity.
#[must_use]
pub fn fixed_rotation_basis(axis: Vec3, sin: f32, cos: f32) -> Mat3 {
    let n = axis.normalize_or_zero();
    if n.length_squared() <= EPS_LEN_SQ {
        return Mat3::IDENTITY;
    }
    let (x, y, z) = (n.x, n.y, n.z);
    let s = sin;
    let c = cos;
    let one_minus_c = 1.0 - c;
    Mat3::from_rows(
        Vec3::new(
            c + x * x * one_minus_c,
            x * y * one_minus_c - z * s,
            x * z * one_minus_c + y * s,
        ),
        Vec3::new(
            y * x * one_minus_c + z * s,
            c + y * y * one_minus_c,
            y * z * one_minus_c - x * s,
        ),
        Vec3::new(
            z * x * one_minus_c - y * s,
            z * y * one_minus_c + x * s,
            c + z * z * one_minus_c,
        ),
    )
}

/// The minimal rotation taking unit direction `from` onto unit direction `to`.
///
/// The sine and cosine of the rotation angle come from the cross and dot
/// products of the (normalized) inputs — never from trigonometry. Degenerate
/// cases are handled explicitly: equal directions give the identity, opposite
/// directions give a 180-degree turn about a deterministic perpendicular, and a
/// zero input gives the identity.
#[must_use]
fn rotation_between(from: Vec3, to: Vec3) -> Mat3 {
    let f = from.normalize_or_zero();
    let t = to.normalize_or_zero();
    if f.length_squared() <= EPS_LEN_SQ || t.length_squared() <= EPS_LEN_SQ {
        return Mat3::IDENTITY;
    }
    let cos = f.dot(t);
    let axis = f.cross(t);
    if axis.length_squared() <= EPS_LEN_SQ {
        // Parallel (already aligned) or antiparallel (needs a half turn).
        if cos >= 0.0 {
            return Mat3::IDENTITY;
        }
        let perp = any_perpendicular(f);
        return fixed_rotation_basis(perp, 0.0, -1.0);
    }
    let sin = axis.length();
    let n = axis.scale(1.0 / sin);
    fixed_rotation_basis(n, sin, cos)
}

/// The rotation locking the mesh's `local_axis` onto world direction `target`.
///
/// A zero `target` yields the identity (nothing to align to).
#[must_use]
pub fn align_to_axis_basis(local_axis: LocalAxis, target: Vec3) -> Mat3 {
    rotation_between(local_axis.unit(), target)
}

/// Resolves a [`MeshOrientation`] into its rotation matrix.
#[must_use]
pub fn orientation_matrix(orientation: MeshOrientation) -> Mat3 {
    match orientation {
        MeshOrientation::Identity => Mat3::IDENTITY,
        MeshOrientation::VelocityAligned {
            velocity,
            up_reference,
        } => velocity_aligned_basis(velocity, up_reference),
        MeshOrientation::FixedRotation { axis, sin, cos } => fixed_rotation_basis(axis, sin, cos),
        MeshOrientation::AlignToAxis { local_axis, target } => {
            align_to_axis_basis(local_axis, target)
        }
    }
}

/// Builds the per-particle scale matrix.
///
/// The effective per-axis scale is `per_axis * uniform * size_over_life`, so a
/// non-uniform base scale, a uniform multiplier, and an over-life size curve
/// sample all fold into one diagonal matrix.
#[must_use]
pub fn scale_matrix(per_axis: Vec3, uniform: f32, size_over_life: f32) -> Mat3 {
    let effective = per_axis.scale(uniform * size_over_life);
    Mat3::from_diagonal(effective)
}

/// Assembles a `TRS` instance transform from an explicit rotation and scale.
///
/// This is `T * R * S` with `T` the world `position`, and is the low-level
/// entry point used by [`instance_affine`].
#[must_use]
pub fn instance_transform(position: Vec3, rotation: Mat3, scale: Mat3) -> Affine3 {
    Affine3::from_trs(position, rotation, scale)
}

/// Assembles the full per-particle mesh instance transform.
///
/// Combines the world `position`, the resolved `orientation` rotation, and the
/// scale built from `per_axis_scale`, `uniform_scale`, and `size_over_life`
/// into a single `TRS` [`Affine3`].
#[must_use]
pub fn instance_affine(
    position: Vec3,
    orientation: MeshOrientation,
    per_axis_scale: Vec3,
    uniform_scale: f32,
    size_over_life: f32,
) -> Affine3 {
    let rotation = orientation_matrix(orientation);
    let scale = scale_matrix(per_axis_scale, uniform_scale, size_over_life);
    instance_transform(position, rotation, scale)
}

/// Transforms a unit-mesh local [`Aabb`] into its world-space bounds under an
/// instance transform.
///
/// Uses the absolute-value linear part: the transformed center is the point
/// image of the local center, and the transformed half-extent is
/// `abs(linear) * local_half_extent`. This is the exact tight axis-aligned
/// bounds of the transformed box and matches transforming all eight corners and
/// taking their component-wise min/max, but with constant work.
#[must_use]
pub fn transform_aabb(xform: Affine3, local: Aabb) -> Aabb {
    let center = local.center();
    let half = local.half_extents();
    let world_center = xform.transform_point(center);
    let world_half = xform.linear.abs().mul_vec(half);
    Aabb {
        min: world_center.sub(world_half),
        max: world_center.add(world_half),
    }
}

/// Selects a mesh `LOD` tier from a screen-coverage fraction.
///
/// `thresholds` lists the *minimum* coverage for each tier in descending order
/// (tier `0` is the highest detail, so its threshold is the largest). The first
/// tier whose threshold `coverage` meets or exceeds is returned; when coverage
/// falls below every threshold the lowest tier (`thresholds.len()`) is
/// returned. `coverage` is clamped to `0..=1` first. An empty `thresholds`
/// slice always selects tier `0`.
#[must_use]
pub fn select_mesh_lod(coverage: f32, thresholds: &[f32]) -> usize {
    let c = coverage.clamp(0.0, 1.0);
    for (tier, &threshold) in thresholds.iter().enumerate() {
        if c >= threshold {
            return tier;
        }
    }
    thresholds.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    const AXIS_X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    const AXIS_Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    const AXIS_Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    fn vec_close(a: Vec3, b: Vec3) -> bool {
        (a.x - b.x).abs() <= EPS && (a.y - b.y).abs() <= EPS && (a.z - b.z).abs() <= EPS
    }

    fn mat_close(a: Mat3, b: Mat3) -> bool {
        vec_close(a.row0, b.row0) && vec_close(a.row1, b.row1) && vec_close(a.row2, b.row2)
    }

    fn is_orthonormal(m: Mat3) -> bool {
        // Columns are the images of the local axes.
        let x = m.mul_vec(AXIS_X);
        let y = m.mul_vec(AXIS_Y);
        let z = m.mul_vec(AXIS_Z);
        let unit = (x.length() - 1.0).abs() <= EPS
            && (y.length() - 1.0).abs() <= EPS
            && (z.length() - 1.0).abs() <= EPS;
        let orthogonal = x.dot(y).abs() <= EPS && y.dot(z).abs() <= EPS && z.dot(x).abs() <= EPS;
        // Right-handed: x cross y == z.
        let right_handed = vec_close(x.cross(y), z);
        unit && orthogonal && right_handed
    }

    #[test]
    fn eps_is_positive() {
        assert!(EPS > 0.0);
    }

    #[test]
    fn identity_is_neutral() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        assert_eq!(Mat3::IDENTITY.mul_vec(v), v);
        assert!(mat_close(
            Mat3::from_diagonal(Vec3::splat(1.0)),
            Mat3::IDENTITY
        ));
        assert_eq!(Affine3::IDENTITY.transform_point(v), v);
        assert!(mat_close(
            Mat3::IDENTITY.mul_mat(Mat3::IDENTITY),
            Mat3::IDENTITY
        ));
    }

    #[test]
    fn from_columns_maps_axes_to_columns() {
        let c0 = Vec3::new(1.0, 2.0, 3.0);
        let c1 = Vec3::new(4.0, 5.0, 6.0);
        let c2 = Vec3::new(7.0, 8.0, 9.0);
        let m = Mat3::from_columns(c0, c1, c2);
        assert!(vec_close(m.mul_vec(AXIS_X), c0));
        assert!(vec_close(m.mul_vec(AXIS_Y), c1));
        assert!(vec_close(m.mul_vec(AXIS_Z), c2));
    }

    #[test]
    fn matrix_product_matches_sequential_application() {
        let a = fixed_rotation_basis(AXIS_Z, 1.0, 0.0);
        let b = Mat3::from_diagonal(Vec3::new(2.0, 3.0, 4.0));
        let v = Vec3::new(1.0, -1.0, 2.0);
        let composed = a.mul_mat(b).mul_vec(v);
        let sequential = a.mul_vec(b.mul_vec(v));
        assert!(vec_close(composed, sequential));
    }

    #[test]
    fn rotation_transpose_is_its_inverse() {
        // 90 degrees about Z as a numeric (sin, cos) pair.
        let r = fixed_rotation_basis(AXIS_Z, 1.0, 0.0);
        assert!(is_orthonormal(r));
        // +X rotates to +Y for a right-handed 90-degree turn about Z.
        assert!(vec_close(r.mul_vec(AXIS_X), AXIS_Y));
        // transpose == inverse: R^T * R == I.
        assert!(mat_close(r.transpose().mul_mat(r), Mat3::IDENTITY));
        assert!(mat_close(r.mul_mat(r.transpose()), Mat3::IDENTITY));
    }

    #[test]
    fn velocity_aligned_points_x_along_velocity() {
        let velocity = Vec3::new(0.0, 5.0, 0.0);
        let basis = velocity_aligned_basis(velocity, AXIS_Z);
        assert!(is_orthonormal(basis));
        assert!(vec_close(basis.mul_vec(AXIS_X), AXIS_Y));
    }

    #[test]
    fn velocity_aligned_zero_velocity_is_safe() {
        // Degenerate: zero velocity falls back to world +X, still orthonormal.
        let basis = velocity_aligned_basis(Vec3::ZERO, AXIS_Y);
        assert!(is_orthonormal(basis));
        assert!(vec_close(basis.mul_vec(AXIS_X), AXIS_X));
    }

    #[test]
    fn fixed_rotation_zero_axis_is_identity() {
        let basis = fixed_rotation_basis(Vec3::ZERO, 1.0, 0.0);
        assert!(mat_close(basis, Mat3::IDENTITY));
    }

    #[test]
    fn align_to_axis_rotates_local_axis_onto_target() {
        // +X onto +Y is the 90-degree turn about Z.
        let basis = align_to_axis_basis(LocalAxis::PlusX, AXIS_Y);
        assert!(is_orthonormal(basis));
        assert!(vec_close(basis.mul_vec(AXIS_X), AXIS_Y));
        // Antiparallel: +X onto -X is a half turn that still maps +X onto -X.
        let flip = align_to_axis_basis(LocalAxis::PlusX, AXIS_X.scale(-1.0));
        assert!(is_orthonormal(flip));
        assert!(vec_close(flip.mul_vec(AXIS_X), AXIS_X.scale(-1.0)));
        // Already aligned yields the identity.
        let same = align_to_axis_basis(LocalAxis::PlusX, AXIS_X);
        assert!(mat_close(same, Mat3::IDENTITY));
    }

    #[test]
    fn scale_matrix_folds_uniform_and_over_life() {
        let m = scale_matrix(Vec3::new(2.0, 3.0, 4.0), 2.0, 0.5);
        // effective = per_axis * (uniform * size_over_life) = per_axis * 1.0.
        assert!(vec_close(m.mul_vec(AXIS_X), Vec3::new(2.0, 0.0, 0.0)));
        assert!(vec_close(m.mul_vec(AXIS_Y), Vec3::new(0.0, 3.0, 0.0)));
        assert!(vec_close(m.mul_vec(AXIS_Z), Vec3::new(0.0, 0.0, 4.0)));
    }

    #[test]
    fn instance_affine_assembles_translate_rotate_scale() {
        let position = Vec3::new(10.0, 20.0, 30.0);
        let orientation = MeshOrientation::FixedRotation {
            axis: AXIS_Z,
            sin: 1.0,
            cos: 0.0,
        };
        let xform = instance_affine(position, orientation, Vec3::new(2.0, 3.0, 4.0), 1.0, 1.0);
        // Local (1,0,0): scale -> (2,0,0), rotate 90 about Z -> (0,2,0), + T.
        let world = xform.transform_point(AXIS_X);
        assert!(vec_close(world, Vec3::new(10.0, 22.0, 30.0)));
        // A direction ignores the translation.
        let dir = xform.transform_vector(AXIS_X);
        assert!(vec_close(dir, Vec3::new(0.0, 2.0, 0.0)));
    }

    #[test]
    fn transform_aabb_matches_eight_corner_bounds() {
        let local = Aabb {
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(3.0, 1.0, 1.0),
        };
        let xform = instance_affine(
            Vec3::new(5.0, -2.0, 1.0),
            MeshOrientation::FixedRotation {
                axis: AXIS_Z,
                sin: 1.0,
                cos: 0.0,
            },
            Vec3::new(2.0, 1.0, 1.0),
            1.0,
            1.0,
        );
        let fast = transform_aabb(xform, local);

        // Reference: transform all eight corners and reduce.
        let mut reduced = Aabb::empty();
        for &sx in &[local.min.x, local.max.x] {
            for &sy in &[local.min.y, local.max.y] {
                for &sz in &[local.min.z, local.max.z] {
                    reduced = reduced.expand(xform.transform_point(Vec3::new(sx, sy, sz)));
                }
            }
        }
        assert!(vec_close(fast.min, reduced.min));
        assert!(vec_close(fast.max, reduced.max));
    }

    #[test]
    fn select_mesh_lod_walks_thresholds() {
        let thresholds = [0.5, 0.2, 0.05];
        assert_eq!(select_mesh_lod(0.8, &thresholds), 0);
        assert_eq!(select_mesh_lod(0.5, &thresholds), 0);
        assert_eq!(select_mesh_lod(0.3, &thresholds), 1);
        assert_eq!(select_mesh_lod(0.1, &thresholds), 2);
        assert_eq!(select_mesh_lod(0.01, &thresholds), 3);
        // Clamped inputs.
        assert_eq!(select_mesh_lod(2.0, &thresholds), 0);
        assert_eq!(select_mesh_lod(-1.0, &thresholds), 3);
        // Empty threshold list always picks the highest tier.
        assert_eq!(select_mesh_lod(0.5, &[]), 0);
    }

    #[test]
    fn affine_compose_matches_sequential_transforms() {
        let outer = Affine3::new(
            fixed_rotation_basis(AXIS_Z, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let inner = Affine3::new(
            Mat3::from_diagonal(Vec3::new(2.0, 2.0, 2.0)),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let p = Vec3::new(1.0, 0.0, 0.0);
        let composed = outer.compose(inner).transform_point(p);
        let sequential = outer.transform_point(inner.transform_point(p));
        assert!(vec_close(composed, sequential));
    }
}
