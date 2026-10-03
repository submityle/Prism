//! Big-world coordinates: `f64` world transforms + grid-cell origin rebasing.
//!
//! Enabled by the `f64` feature. This is milestone **M5(A/B)** of the design
//! doc (§9): the two composable large-world precision strategies.
//!
//! A single-precision `f32` coordinate has only ~24 mantissa bits, so at a
//! world distance of ~16 km the spacing between representable positions grows
//! past a millimetre and geometry *jitters*, Z-fights, and destabilises
//! physics. Prism keeps the authoritative world position in `f64` and only
//! narrows to `f32` **after** subtracting a nearby reference point (the camera
//! or a floating origin), so the value handed to the GPU is small and dense.
//!
//! Two strategies, usable together:
//!
//! - **(A) `f64` world coordinates.** [`TransformHp`] stores an `f64`
//!   translation (rotation/scale stay `f32` — they do not benefit from the
//!   extra range) and [`GlobalTransformHp`] accumulates the hierarchy in an
//!   exact [`DAffine3`]. [`GlobalTransformHp::camera_relative`] produces the
//!   small camera-relative `f32` [`Affine3`] for rendering (UE5 LWC form).
//! - **(B) Grid-cell origin rebasing.** [`FloatingOrigin`] tracks a coarse
//!   [`GridCell`] origin that follows the camera; [`FloatingOrigin::render_offset`]
//!   expresses any world point as an `f32` offset from that origin, and the
//!   origin snaps forward (rebases) when the camera drifts past a threshold so
//!   the active region always hugs zero (Star Citizen form).
//!
//! Both reduce to the same guarantee: the number that reaches the GPU is a
//! *difference* against a nearby reference, so its `f32` error scales with the
//! on-screen distance rather than the absolute world distance.

use alloc::vec::Vec;

use prism_math::{Affine3, DAffine3, DVec3, GridCell, GridPosition, Mat3, Quat, Vec3};

use crate::Transform;
use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};

/// A **local** high-precision transform: an `f64` translation with `f32`
/// rotation and scale.
///
/// Rotation and scale are intentionally single precision: they are bounded
/// quantities (a unit quaternion, a scale near 1) whose precision does not
/// decay with world distance, so widening them would only waste memory and
/// bandwidth. Only translation grows without bound and therefore needs `f64`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TransformHp {
    /// Position relative to the parent, in double precision.
    pub translation: DVec3,
    /// Orientation relative to the parent.
    pub rotation: Quat,
    /// Non-uniform scale relative to the parent.
    pub scale: Vec3,
}

impl Default for TransformHp {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl TransformHp {
    /// The identity transform.
    pub const IDENTITY: Self =
        Self { translation: DVec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE };

    /// Translation-only transform.
    #[inline]
    pub const fn from_translation(translation: DVec3) -> Self {
        Self { translation, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }

    /// Rotation-only transform.
    #[inline]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self { translation: DVec3::ZERO, rotation, scale: Vec3::ONE }
    }

    /// Build from a grid position: the cell origin plus local offset becomes
    /// the exact `f64` translation, with the given rotation and scale.
    #[inline]
    pub fn from_grid_position(pos: GridPosition, rotation: Quat, scale: Vec3) -> Self {
        Self { translation: pos.to_dvec3(), rotation, scale }
    }

    /// Widen a single-precision [`Transform`] into high precision.
    ///
    /// Exact: every `f32` value is representable in `f64`, so this never loses
    /// information. Use it to promote near-origin authoring data into the
    /// big-world path.
    #[inline]
    pub fn from_transform(t: &Transform) -> Self {
        Self { translation: t.translation.as_dvec3(), rotation: t.rotation, scale: t.scale }
    }

    /// The exact `f64` affine of this local transform.
    #[inline]
    pub fn to_daffine(&self) -> DAffine3 {
        DAffine3::from_scale_rotation_translation(
            self.scale.as_dvec3(),
            self.rotation.as_dquat(),
            self.translation,
        )
    }

    /// The grid cell that currently contains this translation.
    #[inline]
    pub fn cell(&self) -> GridCell {
        GridCell::from_dvec3(self.translation)
    }
}

/// A **world-space** high-precision transform, stored as an exact
/// [`DAffine3`] so hierarchical accumulation of non-uniform scale/shear and a
/// far-flung `f64` translation both survive.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GlobalTransformHp {
    affine: DAffine3,
}

impl Default for GlobalTransformHp {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl GlobalTransformHp {
    /// The identity world transform.
    pub const IDENTITY: Self = Self { affine: DAffine3::IDENTITY };

    /// Wrap an existing `f64` affine.
    #[inline]
    pub const fn from_affine(affine: DAffine3) -> Self {
        Self { affine }
    }

    /// Build from a local [`TransformHp`] as if it had no parent (a root).
    #[inline]
    pub fn from_transform_hp(t: &TransformHp) -> Self {
        Self { affine: t.to_daffine() }
    }

    /// The underlying exact `f64` affine.
    #[inline]
    pub fn affine(&self) -> DAffine3 {
        self.affine
    }

    /// The exact `f64` world translation.
    #[inline]
    pub fn translation(&self) -> DVec3 {
        self.affine.translation
    }

    /// The grid cell containing this world position.
    #[inline]
    pub fn cell(&self) -> GridCell {
        GridCell::from_dvec3(self.affine.translation)
    }

    /// This world position split into `(cell, f32 offset)`.
    #[inline]
    pub fn grid_position(&self) -> GridPosition {
        GridPosition::from_dvec3(self.affine.translation)
    }

    /// Propagation step: compose this parent world transform with a child's
    /// local transform to produce the child's world transform.
    ///
    /// Mirrors [`crate::GlobalTransform::mul_transform`] exactly, but in `f64`.
    #[inline]
    pub fn mul_transform(&self, local: &TransformHp) -> GlobalTransformHp {
        GlobalTransformHp { affine: self.affine * local.to_daffine() }
    }

    /// The **camera-relative** single-precision affine for GPU upload (UE5 LWC
    /// form, design §9(A)).
    ///
    /// The translation handed to `f32` is `world − camera`, so its absolute
    /// error scales with the camera-relative distance (what is actually on
    /// screen) rather than the absolute world distance. The linear part
    /// (rotation × scale) is narrowed directly, which is safe because it is a
    /// bounded quantity.
    #[inline]
    pub fn camera_relative(&self, camera: DVec3) -> Affine3 {
        let rel = (self.affine.translation - camera).as_vec3();
        Affine3::from_mat3_translation(self.affine.matrix3.as_mat3(), rel)
    }

    /// The camera-relative affine where the reference point is a grid-cell
    /// origin (design §9(B)); combine with a [`FloatingOrigin`] to keep the
    /// active region near zero.
    #[inline]
    pub fn rebased(&self, origin: GridCell) -> Affine3 {
        self.camera_relative(origin.origin())
    }

    /// The linear (rotation × scale) part only, narrowed to `f32`.
    #[inline]
    pub fn linear_f32(&self) -> Mat3 {
        self.affine.matrix3.as_mat3()
    }
}

impl From<TransformHp> for GlobalTransformHp {
    #[inline]
    fn from(t: TransformHp) -> Self {
        Self::from_transform_hp(&t)
    }
}

/// Run a full high-precision propagation pass over `hierarchy`, writing an
/// exact `f64` world transform for every node.
///
/// `locals[i]` / `globals[i]` are node `i`'s local / world transform. The
/// traversal is parent-before-child and the math is identical in structure to
/// the single-precision [`crate::propagation::propagate`] pass — only the
/// precision differs.
///
/// # Errors
/// - [`HierarchyError::LengthMismatch`] if either buffer length differs from
///   the node count.
/// - [`HierarchyError::Cycle`] if the hierarchy is not a forest.
pub fn propagate_hp(
    hierarchy: &Hierarchy,
    locals: &[TransformHp],
    globals: &mut [GlobalTransformHp],
) -> Result<(), HierarchyError> {
    if locals.len() != hierarchy.len() || globals.len() != hierarchy.len() {
        return Err(HierarchyError::LengthMismatch);
    }
    let order = hierarchy.compute_order()?;
    propagate_hp_in_order(hierarchy, &order, locals, globals);
    Ok(())
}

/// Propagate world transforms for the nodes in `order` (parent-before-child).
///
/// Every referenced parent's world transform must already be valid in
/// `globals` — trivially true for the full order from
/// [`Hierarchy::compute_order`].
///
/// # Panics
/// Panics if any id in `order`, or any parent reachable from it, is out of
/// bounds for `locals`/`globals`.
pub fn propagate_hp_in_order(
    hierarchy: &Hierarchy,
    order: &[NodeId],
    locals: &[TransformHp],
    globals: &mut [GlobalTransformHp],
) {
    for &node in order {
        let local = &locals[node.index()];
        let world = match hierarchy.parent(node) {
            None => GlobalTransformHp::from_transform_hp(local),
            Some(parent) => globals[parent.index()].mul_transform(local),
        };
        globals[node.index()] = world;
    }
}

/// Allocate an identity high-precision world buffer sized for `hierarchy`.
#[inline]
pub fn identity_globals_hp(hierarchy: &Hierarchy) -> Vec<GlobalTransformHp> {
    let mut globals = Vec::with_capacity(hierarchy.len());
    globals.resize(hierarchy.len(), GlobalTransformHp::IDENTITY);
    globals
}

/// A floating world origin (design §9(B), Star Citizen origin rebasing).
///
/// The origin is a coarse [`GridCell`] that the active region is expressed
/// relative to. As the camera moves, [`FloatingOrigin::follow`] snaps the
/// origin forward whenever the camera drifts more than `threshold_cells` away
/// on any axis, so [`FloatingOrigin::render_offset`] always returns a small
/// `f32` vector no matter how far the whole scene has travelled from the world
/// origin.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FloatingOrigin {
    origin: GridCell,
    threshold_cells: i32,
}

impl Default for FloatingOrigin {
    #[inline]
    fn default() -> Self {
        Self::new(GridCell::ZERO)
    }
}

impl FloatingOrigin {
    /// A floating origin at `origin` with a one-cell rebase threshold.
    #[inline]
    pub const fn new(origin: GridCell) -> Self {
        Self { origin, threshold_cells: 1 }
    }

    /// A floating origin with an explicit rebase threshold (in cells). The
    /// threshold is clamped to at least 1 so the origin never thrashes on a
    /// camera sitting exactly on a cell boundary.
    #[inline]
    pub const fn with_threshold(origin: GridCell, threshold_cells: i32) -> Self {
        let threshold_cells = if threshold_cells < 1 { 1 } else { threshold_cells };
        Self { origin, threshold_cells }
    }

    /// The current origin cell.
    #[inline]
    pub const fn origin(&self) -> GridCell {
        self.origin
    }

    /// The exact `f64` world position of the origin cell.
    #[inline]
    pub fn origin_world(&self) -> DVec3 {
        self.origin.origin()
    }

    /// Update the origin to follow `camera`, rebasing when the camera has
    /// drifted more than the threshold on any axis. Returns `true` if the
    /// origin moved.
    #[inline]
    pub fn follow(&mut self, camera: DVec3) -> bool {
        let camera_cell = GridCell::from_dvec3(camera);
        let dx = (camera_cell.x as i64 - self.origin.x as i64).unsigned_abs();
        let dy = (camera_cell.y as i64 - self.origin.y as i64).unsigned_abs();
        let dz = (camera_cell.z as i64 - self.origin.z as i64).unsigned_abs();
        let drift = dx.max(dy).max(dz);
        if drift > self.threshold_cells as u64 {
            self.origin = camera_cell;
            true
        } else {
            false
        }
    }

    /// The `f32` render offset of a world point relative to the current origin.
    ///
    /// Error scales with distance-from-origin, so keeping the origin near the
    /// camera keeps nearby geometry sub-millimetre accurate even 100 km from
    /// the world origin.
    #[inline]
    pub fn render_offset(&self, world: DVec3) -> Vec3 {
        (world - self.origin.origin()).as_vec3()
    }
}
