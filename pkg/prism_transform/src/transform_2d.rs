//! 2D transform variants: `SE(2)` + non-uniform scale, sharing the 3D
//! hierarchy and propagation engine (design doc §11).
//!
//! A 2D game that uses the full 3D [`Transform`](crate::Transform) wastes
//! algebra and invites bugs around the unused Z axis. [`Transform2d`] is the
//! dimensionally-reduced local transform — a [`Vec2`] translation, a scalar
//! rotation angle in radians, and a [`Vec2`] scale — and [`GlobalTransform2d`]
//! is its world-space result, stored as an [`Affine2`] (a 2x2 linear part plus
//! a [`Vec2`] translation) so that hierarchically-accumulated non-uniform scale
//! and rotation compose exactly, including any shear they introduce — the same
//! "store the affine, not the TRS" reasoning as the 3D path.
//!
//! Propagation reuses the exact same [`Hierarchy`] relations and
//! parent-before-child order as 3D; only the per-node algebra changes. See
//! [`propagate_2d`] for the free-function pass and [`TransformGraph2d`] for the
//! owning facade.

use alloc::vec::Vec;

use prism_math::{Mat2, Mat3, Quat, Vec2};

use crate::change::ChangeTicks;
use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};

/// A 2D affine transform: a 2x2 linear map plus a translation.
///
/// Analogous to [`Affine3`](prism_math::Affine3) but in the plane. The linear
/// part absorbs rotation, non-uniform scale, and the shear their composition
/// can produce, which a pure angle+scale pair cannot represent.
#[derive(Clone, Copy, PartialEq)]
pub struct Affine2 {
    /// Linear part, columns stored as a [`Mat2`].
    pub matrix2: Mat2,
    /// Translation part.
    pub translation: Vec2,
}

impl core::fmt::Debug for Affine2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Affine2 {{ matrix2: Mat2[{:?}, {:?}], translation: {:?} }}",
            self.matrix2.x_axis, self.matrix2.y_axis, self.translation
        )
    }
}

impl Default for Affine2 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine2 {
    /// The identity affine (no rotation/scale, zero translation).
    pub const IDENTITY: Self = Self {
        matrix2: Mat2::IDENTITY,
        translation: Vec2::ZERO,
    };

    /// Build from a linear part and translation.
    #[inline]
    pub const fn from_mat2_translation(matrix2: Mat2, translation: Vec2) -> Self {
        Self {
            matrix2,
            translation,
        }
    }

    /// Transform a point: apply the linear part, then translate.
    #[inline]
    pub fn transform_point(self, p: Vec2) -> Vec2 {
        self.matrix2.mul_vec2(p) + self.translation
    }

    /// Transform a direction: apply the linear part only (ignores translation).
    #[inline]
    pub fn transform_vector(self, v: Vec2) -> Vec2 {
        self.matrix2.mul_vec2(v)
    }

    /// Compose two affines: `self` applied after `rhs`
    /// (`(self * rhs).transform_point(p) == self.transform_point(rhs.transform_point(p))`).
    #[inline]
    pub fn mul_affine2(self, rhs: Affine2) -> Affine2 {
        Affine2 {
            matrix2: mat2_mul(self.matrix2, rhs.matrix2),
            translation: self.matrix2.mul_vec2(rhs.translation) + self.translation,
        }
    }
}

impl core::ops::Mul for Affine2 {
    type Output = Affine2;
    #[inline]
    fn mul(self, rhs: Affine2) -> Affine2 {
        self.mul_affine2(rhs)
    }
}

/// Column-wise 2x2 matrix product (`Mat2` has no `Mul` of its own).
#[inline]
fn mat2_mul(a: Mat2, b: Mat2) -> Mat2 {
    Mat2::from_cols(a.mul_vec2(b.x_axis), a.mul_vec2(b.y_axis))
}

/// Build the pure-rotation [`Mat2`] for `angle` radians, reusing the shared
/// [`prism_math`] trig path (via a Z-axis quaternion) so the sine/cosine match
/// the rest of the engine bit-for-bit.
#[inline]
fn rotation_mat2(angle: f32) -> Mat2 {
    // The top-left 2x2 of a Z-axis rotation is exactly the planar rotation.
    let m = Mat3::from_quat(Quat::from_rotation_z(angle));
    Mat2::from_cols(
        Vec2::new(m.x_axis.x, m.x_axis.y),
        Vec2::new(m.y_axis.x, m.y_axis.y),
    )
}

/// A node's local 2D transform relative to its parent (or to the world when it
/// has no parent).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Transform2d {
    /// Position relative to the parent.
    pub translation: Vec2,
    /// Orientation relative to the parent, in radians (counter-clockwise).
    pub rotation: f32,
    /// Non-uniform scale relative to the parent.
    pub scale: Vec2,
}

impl Default for Transform2d {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform2d {
    /// The identity transform (no translation, no rotation, unit scale).
    pub const IDENTITY: Self = Self {
        translation: Vec2::ZERO,
        rotation: 0.0,
        scale: Vec2::ONE,
    };

    /// Translation-only transform from components.
    #[inline]
    pub const fn from_xy(x: f32, y: f32) -> Self {
        Self {
            translation: Vec2::new(x, y),
            rotation: 0.0,
            scale: Vec2::ONE,
        }
    }
    /// Translation-only transform.
    #[inline]
    pub const fn from_translation(translation: Vec2) -> Self {
        Self {
            translation,
            rotation: 0.0,
            scale: Vec2::ONE,
        }
    }
    /// Rotation-only transform (radians).
    #[inline]
    pub const fn from_angle(rotation: f32) -> Self {
        Self {
            translation: Vec2::ZERO,
            rotation,
            scale: Vec2::ONE,
        }
    }
    /// Scale-only transform.
    #[inline]
    pub const fn from_scale(scale: Vec2) -> Self {
        Self {
            translation: Vec2::ZERO,
            rotation: 0.0,
            scale,
        }
    }

    /// Builder: set translation.
    #[inline]
    pub const fn with_translation(mut self, t: Vec2) -> Self {
        self.translation = t;
        self
    }
    /// Builder: set rotation (radians).
    #[inline]
    pub const fn with_angle(mut self, r: f32) -> Self {
        self.rotation = r;
        self
    }
    /// Builder: set scale.
    #[inline]
    pub const fn with_scale(mut self, s: Vec2) -> Self {
        self.scale = s;
        self
    }

    /// Convert this TRS to an [`Affine2`] (scale, then rotate, then translate).
    #[inline]
    pub fn to_affine2(&self) -> Affine2 {
        let r = rotation_mat2(self.rotation);
        let matrix2 = Mat2::from_cols(r.x_axis * self.scale.x, r.y_axis * self.scale.y);
        Affine2 {
            matrix2,
            translation: self.translation,
        }
    }

    /// Transform a point from local space into parent space.
    #[inline]
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        self.to_affine2().transform_point(point)
    }

    /// Compose with a child transform expressed in `self`'s space, yielding the
    /// child's affine in `self`'s parent space.
    #[inline]
    pub fn mul_affine2(&self, child: &Transform2d) -> Affine2 {
        self.to_affine2() * child.to_affine2()
    }
}

/// World-space 2D transform produced by hierarchy propagation, stored as an
/// [`Affine2`] so accumulated non-uniform scale/shear is representable.
#[derive(Clone, Copy, PartialEq)]
pub struct GlobalTransform2d(pub Affine2);

impl core::fmt::Debug for GlobalTransform2d {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GlobalTransform2d({:?})", self.0)
    }
}

impl Default for GlobalTransform2d {
    #[inline]
    fn default() -> Self {
        Self(Affine2::IDENTITY)
    }
}

impl GlobalTransform2d {
    /// The identity world transform.
    pub const IDENTITY: Self = Self(Affine2::IDENTITY);

    /// Build from a local [`Transform2d`] (as if it had no parent).
    #[inline]
    pub fn from_transform(t: &Transform2d) -> Self {
        Self(t.to_affine2())
    }
    /// The underlying affine.
    #[inline]
    pub fn affine(&self) -> Affine2 {
        self.0
    }
    /// World-space translation.
    #[inline]
    pub fn translation(&self) -> Vec2 {
        self.0.translation
    }
    /// Transform a point from local to world space.
    #[inline]
    pub fn transform_point(&self, point: Vec2) -> Vec2 {
        self.0.transform_point(point)
    }
    /// Combine this parent world transform with a child's local transform to
    /// produce the child's world transform.
    #[inline]
    pub fn mul_transform(&self, local: &Transform2d) -> GlobalTransform2d {
        GlobalTransform2d(self.0 * local.to_affine2())
    }
}

impl From<Transform2d> for GlobalTransform2d {
    #[inline]
    fn from(t: Transform2d) -> Self {
        Self::from_transform(&t)
    }
}

/// Run a full 2D propagation pass, writing a world transform for every node.
///
/// `locals[i]` and `globals[i]` are the local and world transforms of node `i`.
/// Every node is computed as `global[node] = global[parent] * local[node]`
/// (roots as `local.to_affine2()`), in a single parent-before-child sweep.
///
/// # Errors
/// - [`HierarchyError::LengthMismatch`] if either buffer length differs from
///   the number of nodes.
/// - [`HierarchyError::Cycle`] if the hierarchy is not a forest.
pub fn propagate_2d(
    hierarchy: &Hierarchy,
    locals: &[Transform2d],
    globals: &mut [GlobalTransform2d],
) -> Result<(), HierarchyError> {
    if locals.len() != hierarchy.len() || globals.len() != hierarchy.len() {
        return Err(HierarchyError::LengthMismatch);
    }
    let order = hierarchy.compute_order()?;
    for &node in &order {
        let local = &locals[node.index()];
        let world = match hierarchy.parent(node) {
            None => GlobalTransform2d::from_transform(local),
            Some(parent) => globals[parent.index()].mul_transform(local),
        };
        globals[node.index()] = world;
    }
    Ok(())
}

/// An ergonomic owner of a 2D transform hierarchy: the forest, every node's
/// authoritative local [`Transform2d`], the cached world
/// [`GlobalTransform2d`]s, and the [`ChangeTicks`] driving propagation, kept in
/// lock-step. The 2D analogue of [`crate::TransformGraph`].
#[derive(Clone, Debug, Default)]
pub struct TransformGraph2d {
    hierarchy: Hierarchy,
    locals: Vec<Transform2d>,
    globals: Vec<GlobalTransform2d>,
    ticks: ChangeTicks,
}

impl TransformGraph2d {
    /// Create an empty graph.
    #[inline]
    pub const fn new() -> Self {
        Self {
            hierarchy: Hierarchy::new(),
            locals: Vec::new(),
            globals: Vec::new(),
            ticks: ChangeTicks::new(),
        }
    }

    /// Number of nodes in the graph.
    #[inline]
    pub fn len(&self) -> usize {
        self.locals.len()
    }

    /// Whether the graph has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.locals.is_empty()
    }

    /// Borrow the underlying [`Hierarchy`].
    #[inline]
    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    /// Borrow the cached world transforms.
    #[inline]
    pub fn globals(&self) -> &[GlobalTransform2d] {
        &self.globals
    }

    /// Spawn a root node carrying `local` and return its id.
    pub fn spawn_root(&mut self, local: Transform2d) -> NodeId {
        let id = self.hierarchy.spawn_root();
        self.locals.push(local);
        self.globals.push(GlobalTransform2d::from_transform(&local));
        self.ticks.push();
        id
    }

    /// Spawn a child of `parent` carrying `local` and return its id.
    ///
    /// # Panics
    /// Panics if `parent` is out of bounds (see [`Hierarchy::spawn_child`]).
    pub fn spawn_child(&mut self, parent: NodeId, local: Transform2d) -> NodeId {
        let id = self.hierarchy.spawn_child(parent);
        self.locals.push(local);
        self.globals.push(GlobalTransform2d::IDENTITY);
        self.ticks.push();
        id
    }

    /// The authoritative local transform of `node`.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn local(&self, node: NodeId) -> Transform2d {
        self.locals[node.index()]
    }

    /// The cached world transform of `node` (valid as of the last
    /// [`TransformGraph2d::propagate`]).
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn global(&self, node: NodeId) -> GlobalTransform2d {
        self.globals[node.index()]
    }

    /// Overwrite `node`'s local transform and mark it changed.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    pub fn set_local(&mut self, node: NodeId, local: Transform2d) {
        self.locals[node.index()] = local;
        self.ticks.mark(node);
    }

    /// Run a full propagation pass, refreshing every world transform, then
    /// close the change epoch.
    ///
    /// # Panics
    /// Panics only if the internal buffers desynchronize from the hierarchy,
    /// which cannot happen through this type's safe API.
    pub fn propagate(&mut self) {
        propagate_2d(&self.hierarchy, &self.locals, &mut self.globals)
            .expect("TransformGraph2d buffers stay in sync with the hierarchy");
        self.ticks.end_pass();
    }
}
