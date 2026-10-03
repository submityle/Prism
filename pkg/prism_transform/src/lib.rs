//! # prism_transform
//!
//! The transform algebra at the base of Prism's spatial hierarchy.
//!
//! - [`Transform`] is the **local** TRS (translation / rotation / scale) a user
//!   writes. Storing TRS (not a matrix) keeps rotation independently
//!   interpolable (`slerp`) and avoids per-frame matrix decomposition.
//! - [`GlobalTransform`] is the **world-space** result, stored as an
//!   [`Affine3`] because hierarchical accumulation of non-uniform scale can
//!   introduce shear that a pure TRS cannot represent.
//!
//! ## Milestone status (per the design-doc roadmap)
//! - **M0 (done):** the standalone algebra — `Transform`, `GlobalTransform`,
//!   composition, inverse, and direction/helper methods, with round-trip and
//!   associativity tests.
//! - **M1 (this update, done):** single-threaded full-pass hierarchy
//!   propagation driven by change detection. The pieces live in dedicated
//!   modules: [`hierarchy`] (the index-based forest and its stable
//!   parent-before-child order), [`propagation`] (the parent-before-child world
//!   sweep that composes `global[node] = global[parent] * local[node]` in
//!   affine space so non-uniform scale survives), and [`change`] (per-node
//!   change ticks so a pass knows what changed and a later pass can skip clean
//!   subtrees). [`TransformGraph`] ties them together. The hierarchy here is a
//!   self-contained computational core; binding it to `prism_ecs` `ChildOf`
//!   relations is deferred to M6.
//! - **M2 (this update, done):** dirty-subtree *incremental* propagation. The
//!   [`dirty`] module adds a [`dirty::DirtyPropagator`] that reads the M1 change
//!   ticks, collects the minimal set of dirty roots (changed nodes with no
//!   changed ancestor), and sweeps only those subtrees in parent-before-child
//!   order, reusing cached globals for untouched nodes. [`TransformGraph::propagate_incremental`]
//!   wires this in and returns [`dirty::DirtyStats`] so a static scene can be
//!   asserted to recompute nothing. Results are identical to the full M1 pass;
//!   only the work differs.
//! - **M3+ (planned):** parallel propagation via `prism_tasks`, fixed-step
//!   interpolation, big-world/deterministic paths, and GPU upload.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::vec::Vec;

use prism_math::{Affine3, Mat4, Quat, Vec3};

pub mod change;
pub mod dirty;
pub mod hierarchy;
pub mod propagation;

use change::ChangeTicks;
use dirty::{DirtyPropagator, DirtyStats};
use hierarchy::{Hierarchy, HierarchyError, NodeId};

/// A node's local transform: translation, rotation, and scale relative to its
/// parent (or to the world when it has no parent).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Transform {
    /// Position relative to the parent.
    pub translation: Vec3,
    /// Orientation relative to the parent.
    pub rotation: Quat,
    /// Non-uniform scale relative to the parent.
    pub scale: Vec3,
}

impl Default for Transform {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// The identity transform (no translation, no rotation, unit scale).
    pub const IDENTITY: Self =
        Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE };

    /// Translation-only transform from components.
    #[inline]
    pub const fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self { translation: Vec3::new(x, y, z), rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
    /// Translation-only transform.
    #[inline]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self { translation, rotation: Quat::IDENTITY, scale: Vec3::ONE }
    }
    /// Rotation-only transform.
    #[inline]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self { translation: Vec3::ZERO, rotation, scale: Vec3::ONE }
    }
    /// Scale-only transform.
    #[inline]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale }
    }

    /// Builder: set translation.
    #[inline]
    pub const fn with_translation(mut self, t: Vec3) -> Self {
        self.translation = t;
        self
    }
    /// Builder: set rotation.
    #[inline]
    pub const fn with_rotation(mut self, r: Quat) -> Self {
        self.rotation = r;
        self
    }
    /// Builder: set scale.
    #[inline]
    pub const fn with_scale(mut self, s: Vec3) -> Self {
        self.scale = s;
        self
    }

    /// Local +X ("right") axis in parent space.
    #[inline]
    pub fn local_x(&self) -> Vec3 {
        self.rotation * Vec3::X
    }
    /// Local +Y ("up") axis in parent space.
    #[inline]
    pub fn local_y(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }
    /// Local +Z axis in parent space.
    #[inline]
    pub fn local_z(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }
    /// Forward direction (-Z) in parent space.
    #[inline]
    pub fn forward(&self) -> Vec3 {
        -self.local_z()
    }

    /// Rotate this transform in place by `delta` (applied after current).
    #[inline]
    pub fn rotate(&mut self, delta: Quat) {
        self.rotation = (delta * self.rotation).normalize();
    }
    /// Translate this transform in place in parent space.
    #[inline]
    pub fn translate(&mut self, delta: Vec3) {
        self.translation += delta;
    }

    /// Convert this TRS to an [`Affine3`].
    #[inline]
    pub fn to_affine(&self) -> Affine3 {
        Affine3::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
    /// Convert this TRS to a [`Mat4`].
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Transform a point from local space into parent space.
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.translation + self.rotation * (self.scale * point)
    }

    /// Compose with a child transform: `self` is the parent, `child` is local
    /// to `self`. The result is expressed in `self`'s parent space.
    ///
    /// Note: when both parent and child carry non-uniform scale the exact
    /// result can contain shear, which a pure TRS cannot represent. This
    /// method composes the TRS components directly (translation/rotation exact,
    /// scale multiplied component-wise); use [`Transform::mul_affine`] when a
    /// lossless result is required.
    #[inline]
    pub fn mul_transform(&self, child: &Transform) -> Transform {
        Transform {
            translation: self.transform_point(child.translation),
            rotation: (self.rotation * child.rotation).normalize(),
            scale: self.scale * child.scale,
        }
    }

    /// Lossless composition via affine math (may contain shear).
    #[inline]
    pub fn mul_affine(&self, child: &Transform) -> Affine3 {
        self.to_affine() * child.to_affine()
    }
}

impl core::ops::Mul for Transform {
    type Output = Transform;
    #[inline]
    fn mul(self, child: Transform) -> Transform {
        self.mul_transform(&child)
    }
}

/// World-space transform produced by hierarchy propagation. Stored as an
/// [`Affine3`] so accumulated non-uniform scale/shear is representable.
///
/// In M0 this is a value type with full algebra; the systems that actually
/// write it from an ECS hierarchy arrive in M1.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GlobalTransform(pub Affine3);

impl Default for GlobalTransform {
    #[inline]
    fn default() -> Self {
        Self(Affine3::IDENTITY)
    }
}

impl GlobalTransform {
    /// The identity world transform.
    pub const IDENTITY: Self = Self(Affine3::IDENTITY);

    /// Build from a local [`Transform`] (as if it had no parent).
    #[inline]
    pub fn from_transform(t: &Transform) -> Self {
        Self(t.to_affine())
    }
    /// The underlying affine.
    #[inline]
    pub fn affine(&self) -> Affine3 {
        self.0
    }
    /// World-space translation.
    #[inline]
    pub fn translation(&self) -> Vec3 {
        self.0.translation
    }
    /// As a [`Mat4`] for GPU upload or legacy interop.
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        self.0.to_mat4()
    }
    /// Transform a point from local to world space.
    #[inline]
    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.0.transform_point3(point)
    }
    /// Inverse world transform.
    #[inline]
    pub fn inverse(&self) -> Self {
        Self(self.0.inverse())
    }

    /// Propagation step: combine this parent world transform with a child's
    /// local transform to produce the child's world transform.
    #[inline]
    pub fn mul_transform(&self, local: &Transform) -> GlobalTransform {
        GlobalTransform(self.0 * local.to_affine())
    }

    /// Recover an approximate local [`Transform`] (assumes no shear).
    #[inline]
    pub fn compute_transform(&self) -> Transform {
        let (scale, rotation, translation) = self.0.to_scale_rotation_translation();
        Transform { translation, rotation, scale }
    }
}

impl From<Transform> for GlobalTransform {
    #[inline]
    fn from(t: Transform) -> Self {
        Self::from_transform(&t)
    }
}

/// An ergonomic owner of a transform hierarchy: the forest, the authoritative
/// local [`Transform`] of every node, the cached world [`GlobalTransform`]s,
/// and the [`ChangeTicks`] that drive propagation, all kept in lock-step.
///
/// This is the M1 facade over [`hierarchy`], [`propagation`], and [`change`].
/// Callers spawn nodes, edit locals (which marks them changed), and call
/// [`TransformGraph::propagate`] to refresh every world transform.
#[derive(Clone, Debug, Default)]
pub struct TransformGraph {
    hierarchy: Hierarchy,
    locals: Vec<Transform>,
    globals: Vec<GlobalTransform>,
    ticks: ChangeTicks,
    dirty: DirtyPropagator,
}

impl TransformGraph {
    /// Create an empty graph.
    #[inline]
    pub const fn new() -> Self {
        Self {
            hierarchy: Hierarchy::new(),
            locals: Vec::new(),
            globals: Vec::new(),
            ticks: ChangeTicks::new(),
            dirty: DirtyPropagator::new(),
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

    /// Borrow the change-tick bookkeeping.
    #[inline]
    pub fn ticks(&self) -> &ChangeTicks {
        &self.ticks
    }

    /// Spawn a root node carrying `local` and return its id. The node's world
    /// transform is seeded from `local` and marked changed for the next pass.
    pub fn spawn_root(&mut self, local: Transform) -> NodeId {
        let id = self.hierarchy.spawn_root();
        self.locals.push(local);
        self.globals.push(GlobalTransform::from_transform(&local));
        self.ticks.push();
        id
    }

    /// Spawn a child of `parent` carrying `local` and return its id.
    ///
    /// # Panics
    /// Panics if `parent` is out of bounds (see [`Hierarchy::spawn_child`]).
    pub fn spawn_child(&mut self, parent: NodeId, local: Transform) -> NodeId {
        let id = self.hierarchy.spawn_child(parent);
        self.locals.push(local);
        self.globals.push(GlobalTransform::IDENTITY);
        self.ticks.push();
        id
    }

    /// The authoritative local transform of `node`.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn local(&self, node: NodeId) -> Transform {
        self.locals[node.index()]
    }

    /// The cached world transform of `node` (valid as of the last
    /// [`TransformGraph::propagate`]).
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn global(&self, node: NodeId) -> GlobalTransform {
        self.globals[node.index()]
    }

    /// Overwrite `node`'s local transform and mark it changed. Never writes to
    /// any world transform — that is the propagation pass's job.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    pub fn set_local(&mut self, node: NodeId, local: Transform) {
        self.locals[node.index()] = local;
        self.ticks.mark(node);
    }

    /// Whether `node`'s local changed since the last pass.
    ///
    /// # Panics
    /// Panics if `node` is out of bounds.
    #[inline]
    pub fn is_changed(&self, node: NodeId) -> bool {
        self.ticks.is_changed(node)
    }

    /// Run a full propagation pass, refreshing every world transform, then
    /// close the change epoch so all nodes read as clean until the next edit.
    ///
    /// # Panics
    /// Panics only if the internal buffers desynchronize from the hierarchy,
    /// which cannot happen through this type's safe API.
    pub fn propagate(&mut self) {
        propagation::propagate(&self.hierarchy, &self.locals, &mut self.globals)
            .expect("TransformGraph buffers stay in sync with the hierarchy");
        self.ticks.end_pass();
    }

    /// Run an **incremental** propagation pass (M2): recompute only the world
    /// transforms inside dirty subtrees, skip clean subtrees entirely, then
    /// close the change epoch so all nodes read as clean until the next edit.
    ///
    /// Which nodes are dirty is decided by the M1 [`ChangeTicks`]: editing a
    /// local via [`TransformGraph::set_local`] (or topology edits that call
    /// [`ChangeTicks::mark`], such as [`TransformGraph::reparent`]) marks the
    /// affected node, and this pass recomputes that node's whole subtree
    /// because every descendant composes through it.
    ///
    /// After any sequence of edits, the resulting world transforms are
    /// identical to those [`TransformGraph::propagate`] would produce; the
    /// difference is only the work done. For a static scene (nothing changed
    /// since the last pass) this performs no world-matrix compositions and
    /// returns [`DirtyStats`] with `recomputed == 0`.
    ///
    /// # Panics
    /// Panics only if the internal buffers desynchronize from the hierarchy,
    /// which cannot happen through this type's safe API.
    pub fn propagate_incremental(&mut self) -> DirtyStats {
        let stats = self
            .dirty
            .propagate(&self.hierarchy, &self.ticks, &self.locals, &mut self.globals)
            .expect("TransformGraph buffers stay in sync with the hierarchy");
        self.ticks.end_pass();
        stats
    }

    /// Re-parent `child` under `new_parent` (or detach to a root with `None`),
    /// keeping its *local* transform unchanged. The world pose generally moves;
    /// call [`TransformGraph::propagate`] to recompute it.
    ///
    /// # Errors
    /// Propagates the errors of [`Hierarchy::set_parent`].
    pub fn reparent(
        &mut self,
        child: NodeId,
        new_parent: Option<NodeId>,
    ) -> Result<(), HierarchyError> {
        self.hierarchy.set_parent(child, new_parent)?;
        self.ticks.mark(child);
        Ok(())
    }

    /// Re-parent `child` under `new_parent` while preserving its **world**
    /// pose: the local transform is recomputed as
    /// `inverse(parent_world) * child_world`.
    ///
    /// This runs a propagation pass first (to read current world poses) and a
    /// second one afterwards (to refresh the moved subtree). The new local is
    /// recovered via [`GlobalTransform::compute_transform`], so the round-trip
    /// is exact only when the resulting local has no shear (e.g. no mix of
    /// non-uniform parent scale with child rotation); otherwise the world pose
    /// is preserved only to within TRS-representable precision, consistent with
    /// the "Local is authoritative" invariant.
    ///
    /// # Errors
    /// - [`HierarchyError::InvalidNode`] if `child` or `new_parent` is out of
    ///   bounds.
    /// - [`HierarchyError::Cycle`] if the edge would form a cycle.
    pub fn reparent_keeping_world(
        &mut self,
        child: NodeId,
        new_parent: Option<NodeId>,
    ) -> Result<(), HierarchyError> {
        if !self.hierarchy.contains(child) {
            return Err(HierarchyError::InvalidNode);
        }
        if let Some(parent) = new_parent {
            if !self.hierarchy.contains(parent) {
                return Err(HierarchyError::InvalidNode);
            }
        }

        self.propagate();
        let child_world = self.globals[child.index()].affine();
        let parent_world = match new_parent {
            Some(parent) => self.globals[parent.index()].affine(),
            None => Affine3::IDENTITY,
        };
        let new_local_affine = parent_world.inverse() * child_world;

        self.hierarchy.set_parent(child, new_parent)?;
        self.locals[child.index()] = GlobalTransform(new_local_affine).compute_transform();
        self.ticks.mark(child);
        self.propagate();
        Ok(())
    }
}

/// Common imports.
pub mod prelude {
    pub use crate::change::{ChangeTicks, Tick};
    pub use crate::dirty::{DirtyPropagator, DirtyStats};
    pub use crate::hierarchy::{Hierarchy, HierarchyError, NodeId};
    pub use crate::propagation::propagate;
    pub use crate::{GlobalTransform, Transform, TransformGraph};
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_m1;

#[cfg(test)]
mod tests_m2;
