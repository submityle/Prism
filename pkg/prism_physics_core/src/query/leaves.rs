//! Leaf-aware queries with source identity and effective collision policies.

use super::{dispatch, overlap, QueryFilter};
use crate::collider::{ColliderHandle, ColliderShape, PhysicsMaterial};
use crate::math::transform::Isometry;
use crate::state::handle::BodyHandle;
use crate::world::PhysicsWorld;
use glam::Vec3;
use prism_physics_geometry::Ray;

/// Stable identity and effective policy for an immutable collision leaf.
/// A source handoff mapping's `body_child_range` resolves `child_index` back
/// to its original attachment, even when several sources share one collider.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ColliderLeaf {
    pub body: BodyHandle,
    /// Owning body collider (the compound handle, if any).
    pub body_collider: ColliderHandle,
    /// Actual primitive/mesh handle, not an AABB approximation.
    pub collider: ColliderHandle,
    pub child_index: Option<u32>,
    pub pose: Isometry,
    pub material: PhysicsMaterial,
    /// Effective same-group opt-in; irrelevant to external-body contacts.
    pub self_collision: bool,
}

/// First surface intersection for one leaf. Cast hits are sorted near-to-far,
/// stably retaining body-slot and flattened-child order for ties.
#[derive(Clone, Copy, Debug)]
pub struct LeafHit {
    pub leaf: ColliderLeaf,
    pub time_of_impact: f32,
    pub point: Vec3,
    pub normal: Vec3,
}

/// Per-leaf surface projection, sorted by unsigned distance, not a projection
/// onto the exterior boundary of a potentially overlapping compound union.
#[derive(Clone, Copy, Debug)]
pub struct LeafProjection {
    pub leaf: ColliderLeaf,
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub is_inside: bool,
}

impl PhysicsWorld {
    /// Enumerates current leaves and effective policies without mutating or
    /// caching body state. Replacement colliders and body generations are
    /// therefore reflected immediately. Direct colliders have no child index.
    #[must_use]
    pub fn collider_leaves(&self, filter: &QueryFilter) -> Vec<ColliderLeaf> {
        let mut leaves = Vec::new();
        for body in self.query_bodies(filter) {
            let material = self.bodies.material(body.handle).unwrap_or_default();
            let self_collision = self
                .collision_exclusions()
                .membership(body.handle)
                .is_some_and(|membership| membership.self_collision);
            if let ColliderShape::Compound { compound, .. } = *body.shape {
                let Some(data) = self.shapes.compound(compound) else {
                    continue;
                };
                for (index, (child, policy)) in
                    data.children().iter().zip(data.policies()).enumerate()
                {
                    leaves.push(ColliderLeaf {
                        body: body.handle,
                        body_collider: body.collider,
                        collider: child.collider,
                        child_index: Some(index as u32),
                        pose: body.pose.mul(&child.local_pose),
                        material: policy.material.unwrap_or(material),
                        self_collision: policy.self_collision.unwrap_or(self_collision),
                    });
                }
            } else {
                leaves.push(ColliderLeaf {
                    body: body.handle,
                    body_collider: body.collider,
                    collider: body.collider,
                    child_index: None,
                    pose: body.pose,
                    material,
                    self_collision,
                });
            }
        }
        leaves
    }

    fn cast_leaves(
        &self,
        filter: &QueryFilter,
        cast: impl Fn(&ColliderShape, &Isometry) -> Option<super::ray::RayShapeHit>,
    ) -> Vec<LeafHit> {
        let mut hits = Vec::new();
        for leaf in self.collider_leaves(filter) {
            let Some(shape) = self.shapes.get(leaf.collider) else {
                continue;
            };
            if let Some(hit) = cast(shape, &leaf.pose) {
                hits.push(LeafHit {
                    leaf,
                    time_of_impact: hit.time_of_impact,
                    point: hit.point,
                    normal: hit.normal,
                });
            }
        }
        hits.sort_by(|a, b| a.time_of_impact.total_cmp(&b.time_of_impact));
        hits
    }

    /// Returns one first intersection per touched leaf (not just per body).
    #[must_use]
    pub fn raycast_leaves(&self, ray: &Ray, filter: &QueryFilter) -> Vec<LeafHit> {
        self.cast_leaves(filter, |shape, pose| {
            dispatch::raycast_shape_in(shape, pose, ray, &self.shapes)
        })
    }

    /// Returns a swept sphere's first contact with each touched leaf.
    #[must_use]
    pub fn spherecast_leaves(&self, ray: &Ray, radius: f32, filter: &QueryFilter) -> Vec<LeafHit> {
        self.cast_leaves(filter, |shape, pose| {
            dispatch::spherecast_shape_in(shape, pose, ray, radius, &self.shapes)
        })
    }

    /// Mirrors `shapecast_all`, retaining the identity and policy of each
    /// target leaf. A compound mover reports its earliest contact per target.
    #[must_use]
    pub fn shapecast_leaves(
        &self,
        mover: &ColliderShape,
        pose: Isometry,
        motion: Vec3,
        filter: &QueryFilter,
    ) -> Vec<LeafHit> {
        self.cast_leaves(filter, |shape, target_pose| {
            dispatch::shapecast_shape_in(mover, &pose, shape, target_pose, motion, &self.shapes)
        })
    }

    /// Projects onto every accepted leaf, returning nearest surfaces first.
    #[must_use]
    pub fn project_point_leaves(&self, point: Vec3, filter: &QueryFilter) -> Vec<LeafProjection> {
        let mut hits = Vec::new();
        for leaf in self.collider_leaves(filter) {
            let Some(shape) = self.shapes.get(leaf.collider) else {
                continue;
            };
            let hit = dispatch::project_point_shape_in(shape, &leaf.pose, point, &self.shapes);
            hits.push(LeafProjection {
                leaf,
                point: hit.point,
                normal: hit.normal,
                distance: hit.distance,
                is_inside: hit.is_inside,
            });
        }
        hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        hits
    }

    /// Exact per-leaf overlaps in stable body/child order. Like other scene
    /// queries, this is geometry-only, not a group-relative simulation filter.
    #[must_use]
    pub fn overlap_shape_leaves(
        &self,
        shape: &ColliderShape,
        pose: Isometry,
        filter: &QueryFilter,
    ) -> Vec<ColliderLeaf> {
        self.collider_leaves(filter)
            .into_iter()
            .filter(|leaf| {
                self.shapes.get(leaf.collider).is_some_and(|target| {
                    overlap::shapes_overlap_in(shape, &pose, target, &leaf.pose, &self.shapes)
                })
            })
            .collect()
    }
}
