//! Immutable compound collision geometry. Children retain their real shapes;
//! the union AABB is used only for broad-phase candidate generation.

use super::{ColliderHandle, ColliderShape, PhysicsMaterial, ShapeRegistry};
use crate::math::transform::Isometry;
use glam::{Mat3, Vec3};

/// Maximum expanded leaf count in one compound. Bounds construction work and
/// query traversal even when callers repeatedly nest previously built unions.
pub const MAX_COMPOUND_CHILDREN: usize = 4096;

/// Index into a registry's immutable compound arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CompoundHandle(pub u32);

/// One child collider and its rigid pose relative to the compound origin.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CompoundChild {
    /// Existing collider in the same registry.
    pub collider: ColliderHandle,
    /// Child-local to compound-local transform (no scale or shear).
    pub local_pose: Isometry,
}

/// Optional overrides for one collision leaf. `None` inherits the owning
/// body's policy. Parallel policy storage separates reusable geometry from
/// collision response metadata. Nesting preserves overrides; an explicit
/// outer override takes precedence over the nested value.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CompoundLeafPolicy {
    /// Contact response override; `None` uses the body's live material.
    pub material: Option<PhysicsMaterial>,
    /// Same-exclusion-group collision opt-in; `None` uses body membership.
    pub self_collision: Option<bool>,
}

/// Validated, flattened collision leaves. Private storage prevents cycles and
/// mutation after validation; nested compounds are expanded at registration.
#[derive(Clone, Debug)]
pub struct CompoundColliderData {
    children: Vec<CompoundChild>,
    policies: Vec<CompoundLeafPolicy>,
    shape: ColliderShape,
}

/// Validated geometry staged without copying or changing the registry. Existing
/// shape handles remain valid because the registry is append-only. Commit can
/// therefore assign arena handles after every fallible operation has succeeded.
pub(crate) struct PreparedCompound {
    children: Vec<CompoundChild>,
    policies: Vec<CompoundLeafPolicy>,
    min: Vec3,
    max: Vec3,
    mass: f32,
    inertia: Vec3,
}

impl CompoundColliderData {
    /// Nonempty, finite collision leaves in deterministic insertion order.
    /// Every child is a bounded non-compound shape in the owning registry.
    #[must_use]
    pub fn children(&self) -> &[CompoundChild] {
        &self.children
    }

    /// One policy per flattened child, in exactly the same stable order.
    #[must_use]
    pub fn policies(&self) -> &[CompoundLeafPolicy] {
        &self.policies
    }
}

/// Failed registration leaves the registry unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompoundColliderError {
    /// A compound needs at least one collision leaf.
    Empty,
    /// A child collider or its backing arena handle does not exist.
    InvalidHandle,
    /// A plane is unbounded and cannot be a finite compound child.
    UnboundedChild,
    /// Invalid dimensions, non-finite geometry/cache, or a non-rigid pose.
    InvalidGeometry,
    /// Expanded leaf count exceeds [`MAX_COMPOUND_CHILDREN`].
    TooManyChildren,
    /// Policy count differs from input child count, or a material is invalid.
    InvalidPolicy,
}

impl ShapeRegistry {
    /// Validates and registers a true compound union, returning a ready-to-use
    /// collider handle. All handles must refer to this registry. Nested unions
    /// are flattened using their validated private leaves; forged future/self
    /// handles cannot form cycles. No arena entry is appended on failure.
    ///
    /// Child transforms compose as `outer * inner`. The cached mass is the sum
    /// of child masses and diagonal inertias, rotated and shifted to this
    /// origin; authored assembly mass properties can be supplied independently.
    pub fn insert_compound(
        &mut self,
        children: &[CompoundChild],
    ) -> Result<ColliderHandle, CompoundColliderError> {
        let prepared = self.prepare_compound(children)?;
        Ok(self.insert_prepared_compound(prepared))
    }

    /// Registers geometry and per-input-child overrides atomically. Each input
    /// child's override applies to all of its expanded leaves.
    pub fn insert_compound_with_policies(
        &mut self,
        children: &[CompoundChild],
        policies: &[CompoundLeafPolicy],
    ) -> Result<ColliderHandle, CompoundColliderError> {
        let prepared = self.prepare_compound_with_policies(children, policies)?;
        Ok(self.insert_prepared_compound(prepared))
    }

    pub(crate) fn prepare_compound(
        &self,
        children: &[CompoundChild],
    ) -> Result<PreparedCompound, CompoundColliderError> {
        if children.len() > MAX_COMPOUND_CHILDREN {
            return Err(CompoundColliderError::TooManyChildren);
        }
        self.prepare_compound_with_policies(
            children,
            &vec![CompoundLeafPolicy::default(); children.len()],
        )
    }

    pub(crate) fn prepare_compound_with_policies(
        &self,
        children: &[CompoundChild],
        policies: &[CompoundLeafPolicy],
    ) -> Result<PreparedCompound, CompoundColliderError> {
        use CompoundColliderError as Error;
        if children.is_empty() {
            return Err(Error::Empty);
        }
        if children.len() > MAX_COMPOUND_CHILDREN {
            return Err(Error::TooManyChildren);
        }
        if policies.len() != children.len()
            || policies.iter().any(|policy| {
                policy.material.is_some_and(|material| {
                    !material.friction.is_finite()
                        || material.friction < 0.0
                        || !material.restitution.is_finite()
                        || !(0.0..=1.0).contains(&material.restitution)
                })
            })
        {
            return Err(Error::InvalidPolicy);
        }
        let mut leaves = Vec::new();
        let mut leaf_policies = Vec::new();
        for (child, policy) in children.iter().zip(policies) {
            validate_pose(&child.local_pose)?;
            let shape = self.get(child.collider).ok_or(Error::InvalidHandle)?;
            if let ColliderShape::Compound { compound, .. } = *shape {
                let data = self.compound(compound).ok_or(Error::InvalidHandle)?;
                if *shape != data.shape {
                    return Err(Error::InvalidGeometry);
                }
                if leaves.len() + data.children.len() > MAX_COMPOUND_CHILDREN {
                    return Err(Error::TooManyChildren);
                }
                for (leaf, inherited) in data.children.iter().zip(&data.policies) {
                    let local_pose = child.local_pose.mul(&leaf.local_pose);
                    validate_pose(&local_pose)?;
                    leaves.push(CompoundChild {
                        collider: leaf.collider,
                        local_pose,
                    });
                    leaf_policies.push(CompoundLeafPolicy {
                        material: policy.material.or(inherited.material),
                        self_collision: policy.self_collision.or(inherited.self_collision),
                    });
                }
            } else {
                if leaves.len() == MAX_COMPOUND_CHILDREN {
                    return Err(Error::TooManyChildren);
                }
                leaves.push(*child);
                leaf_policies.push(*policy);
            }
        }

        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let mut mass = 0.0;
        let mut inertia = Vec3::ZERO;
        for child in &leaves {
            let shape = self.get(child.collider).ok_or(Error::InvalidHandle)?;
            validate_leaf(self, shape)?;
            let (lo, hi) = shape.local_aabb();
            for x in [lo.x, hi.x] {
                for y in [lo.y, hi.y] {
                    for z in [lo.z, hi.z] {
                        let point = child.local_pose.transform_point(Vec3::new(x, y, z));
                        if !point.is_finite() {
                            return Err(Error::InvalidGeometry);
                        }
                        min = min.min(point);
                        max = max.max(point);
                    }
                }
            }
            let props = shape.mass_properties(1.0);
            let m = if props.inv_mass > 0.0 {
                props.inv_mass.recip()
            } else {
                0.0
            };
            let inv = props.inv_inertia;
            let i = Vec3::new(
                if inv.x > 0.0 { inv.x.recip() } else { 0.0 },
                if inv.y > 0.0 { inv.y.recip() } else { 0.0 },
                if inv.z > 0.0 { inv.z.recip() } else { 0.0 },
            );
            let r = Mat3::from_quat(child.local_pose.rotation);
            let rotated = r * Mat3::from_diagonal(i) * r.transpose();
            let p = child.local_pose.translation;
            inertia += Vec3::new(rotated.x_axis.x, rotated.y_axis.y, rotated.z_axis.z)
                + m * Vec3::new(
                    p.y * p.y + p.z * p.z,
                    p.x * p.x + p.z * p.z,
                    p.x * p.x + p.y * p.y,
                );
            mass += m;
        }
        // Bound derived extents too: finite endpoints alone can still overflow
        // the BVH's centre/extents and rotational CCD radius calculations.
        if !(max - min).is_finite()
            || !(min + max).is_finite()
            || !min.abs().max(max.abs()).length().is_finite()
            || !mass.is_finite()
            || !inertia.is_finite()
        {
            return Err(Error::InvalidGeometry);
        }
        Ok(PreparedCompound {
            children: leaves,
            policies: leaf_policies,
            min,
            max,
            mass,
            inertia,
        })
    }

    pub(crate) fn insert_prepared_compound(
        &mut self,
        prepared: PreparedCompound,
    ) -> ColliderHandle {
        let compound = CompoundHandle(self.compounds.len() as u32);
        let shape = ColliderShape::Compound {
            compound,
            local_aabb_min: prepared.min,
            local_aabb_max: prepared.max,
            unit_density_mass: prepared.mass,
            unit_density_inertia: prepared.inertia,
        };
        self.compounds.push(CompoundColliderData {
            children: prepared.children,
            policies: prepared.policies,
            shape,
        });
        self.insert(shape)
    }

    /// Resolves immutable, validated compound leaves.
    #[must_use]
    pub fn compound(&self, handle: CompoundHandle) -> Option<&CompoundColliderData> {
        self.compounds.get(handle.0 as usize)
    }

    /// Returns the canonical compound shape, including its cached bounds.
    #[must_use]
    pub fn compound_shape(&self, handle: CompoundHandle) -> Option<ColliderShape> {
        self.compound(handle).map(|data| data.shape)
    }
}

fn validate_pose(pose: &Isometry) -> Result<(), CompoundColliderError> {
    if !pose.translation.is_finite() || !pose.rotation.is_finite() || !pose.rotation.is_normalized()
    {
        return Err(CompoundColliderError::InvalidGeometry);
    }
    Ok(())
}

fn validate_leaf(
    registry: &ShapeRegistry,
    shape: &ColliderShape,
) -> Result<(), CompoundColliderError> {
    use CompoundColliderError as Error;
    let positive = |v: f32| v.is_finite() && v > 0.0;
    match *shape {
        ColliderShape::Plane { .. } => return Err(Error::UnboundedChild),
        ColliderShape::Compound { .. } => return Err(Error::InvalidGeometry),
        ColliderShape::Sphere { radius } if !positive(radius) => {
            return Err(Error::InvalidGeometry)
        }
        ColliderShape::Cuboid { half_extents }
            if !half_extents.is_finite() || half_extents.min_element() <= 0.0 =>
        {
            return Err(Error::InvalidGeometry)
        }
        ColliderShape::Capsule {
            half_height,
            radius,
        } if !positive(radius) || !half_height.is_finite() || half_height < 0.0 => {
            return Err(Error::InvalidGeometry)
        }
        ColliderShape::ConvexHull { mesh, .. } => {
            let canonical = registry
                .convex_hull_shape(mesh)
                .ok_or(Error::InvalidHandle)?;
            if canonical != *shape {
                return Err(Error::InvalidGeometry);
            }
        }
        ColliderShape::TriangleMesh { mesh, .. } => {
            let canonical = registry.tri_mesh_shape(mesh).ok_or(Error::InvalidHandle)?;
            if canonical != *shape {
                return Err(Error::InvalidGeometry);
            }
        }
        _ => {}
    }
    let (min, max) = shape.local_aabb();
    let props = shape.mass_properties(1.0);
    if !min.is_finite()
        || !max.is_finite()
        || !min.cmple(max).all()
        || !props.inv_mass.is_finite()
        || !props.inv_inertia.is_finite()
    {
        return Err(Error::InvalidGeometry);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn staged_compounds_are_read_only_and_assign_handles_only_at_commit() {
        let mut shapes = ShapeRegistry::default();
        let sphere = shapes.insert(ColliderShape::Sphere { radius: 0.5 });
        let local = Isometry::new(Vec3::Y, Quat::from_rotation_z(0.4));
        let nested = shapes
            .insert_compound(&[CompoundChild {
                collider: sphere,
                local_pose: local,
            }])
            .unwrap();
        let outer = Isometry::new(Vec3::X * 2.0, Quat::from_rotation_y(-0.3));
        let before = shapes.len();
        let first = shapes
            .prepare_compound(&[CompoundChild {
                collider: nested,
                local_pose: outer,
            }])
            .unwrap();
        let second = shapes
            .prepare_compound(&[CompoundChild {
                collider: sphere,
                local_pose: Isometry::from_translation(-Vec3::X),
            }])
            .unwrap();
        assert_eq!(shapes.len(), before);
        assert_eq!(shapes.compounds.len(), 1);

        // A plane and another compound can be committed between preparation
        // and consumption. Staged data contains no speculative arena handle.
        shapes.insert(ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        });
        let second = shapes.insert_prepared_compound(second);
        let first = shapes.insert_prepared_compound(first);
        for (handle, expected_index, expected_pose) in [
            (second, 1, Isometry::from_translation(-Vec3::X)),
            (first, 2, outer.mul(&local)),
        ] {
            let shape = shapes.get(handle).unwrap();
            let ColliderShape::Compound { compound, .. } = *shape else {
                panic!("compound");
            };
            assert_eq!(compound.0, expected_index);
            assert_eq!(shapes.compound_shape(compound), Some(*shape));
            let leaves = shapes.compound(compound).unwrap().children();
            assert_eq!(leaves.len(), 1);
            assert_eq!(leaves[0].collider, sphere);
            assert_eq!(leaves[0].local_pose, expected_pose);
        }
    }
}
