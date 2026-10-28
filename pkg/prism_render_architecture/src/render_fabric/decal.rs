//! Backend-neutral Decal Render Fabric ABI and CPU resolve contract.
//!
//! The records in this module are the semantic boundary shared by the Scene
//! Fabric recorder and the Decal WESL kernel.  They describe an actual
//! deferred-decal operation: projector instances select materials, clustered
//! receiver candidates are read from the G-buffer stream, and resolved
//! receiver material is written back.  They are intentionally not generic
//! placeholder buffers.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

pub const DECAL_INSTANCE_STRIDE: u64 = core::mem::size_of::<DecalInstance>() as u64;
pub const DECAL_MATERIAL_STRIDE: u64 = core::mem::size_of::<DecalMaterial>() as u64;
pub const DECAL_RECEIVER_STRIDE: u64 = core::mem::size_of::<DecalReceiver>() as u64;
pub const DECAL_CLUSTER_RANGE_STRIDE: u64 = core::mem::size_of::<DecalClusterRange>() as u64;
pub const DECAL_CLUSTER_INDEX_STRIDE: u64 = core::mem::size_of::<u32>() as u64;
pub const DECAL_RESOLVED_RECEIVER_STRIDE: u64 =
    core::mem::size_of::<DecalResolvedReceiver>() as u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum DecalBlendMode {
    AlphaOver = 0,
    Additive = 1,
}

impl Default for DecalBlendMode {
    fn default() -> Self {
        Self::AlphaOver
    }
}

/// One projector instance.  The matrix is column-major and maps world space
/// into the unit decal cube `[-0.5, 0.5]^3`.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct DecalInstance {
    pub world_to_decal: [f32; 16],
    pub projection_fade: [f32; 4], // xyz = projection direction, w = angle start
    pub fade_material: [f32; 4],   // x = angle end, y = edge fade, z = material index
}

impl DecalInstance {
    #[must_use]
    pub fn new(
        world_to_decal: [f32; 16],
        projection_dir: [f32; 3],
        angle_fade_start: f32,
        angle_fade_end: f32,
        edge_fade: f32,
        material_index: u32,
    ) -> Self {
        Self {
            world_to_decal,
            projection_fade: [
                projection_dir[0],
                projection_dir[1],
                projection_dir[2],
                angle_fade_start,
            ],
            fade_material: [angle_fade_end, edge_fade, material_index as f32, 0.0],
        }
    }

    #[must_use]
    pub fn from_projector(
        world_to_decal: [f32; 16],
        projection_dir: [f32; 3],
        angle_fade_start: f32,
        angle_fade_end: f32,
        edge_fade: f32,
        material_index: u32,
    ) -> Self {
        Self::new(
            world_to_decal,
            projection_dir,
            angle_fade_start,
            angle_fade_end,
            edge_fade,
            material_index,
        )
    }
}

/// Material values sampled by the decal resolve.  The normal is tangent-space
/// and uses +Z as the flat/no-perturbation direction.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct DecalMaterial {
    pub albedo: [f32; 4],
    pub normal: [f32; 4],
    pub scalar_blend: [f32; 4], // roughness, metallic, opacity, blend mode
}

impl Default for DecalMaterial {
    fn default() -> Self {
        Self {
            albedo: [1.0, 1.0, 1.0, 1.0],
            normal: [0.0, 0.0, 1.0, 0.0],
            scalar_blend: [0.5, 0.0, 1.0, DecalBlendMode::AlphaOver as u32 as f32],
        }
    }
}

/// One receiver/G-buffer record consumed by the decal pass.
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct DecalReceiver {
    pub world_position: [f32; 4],
    pub world_normal: [f32; 4],
    pub albedo: [f32; 4],
    pub scalar: [f32; 4], // roughness, metallic, object/material id, reserved
}

impl Default for DecalReceiver {
    fn default() -> Self {
        Self {
            world_position: [0.0, 0.0, 0.0, 1.0],
            world_normal: [0.0, 0.0, 1.0, 0.0],
            albedo: [0.0, 0.0, 0.0, 1.0],
            scalar: [0.5, 0.0, 0.0, 0.0],
        }
    }
}

/// Candidate decal range for one receiver/cluster.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct DecalClusterRange {
    pub offset: u32,
    pub count: u32,
    pub reserved: [u32; 2],
}

/// Resolved receiver material written by the GPU pass.
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct DecalResolvedReceiver {
    pub albedo: [f32; 4],
    pub normal: [f32; 4],
    pub scalar_coverage: [f32; 4], // roughness, metallic, coverage, reserved
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecalProjectionHit {
    pub fade: f32,
}

/// Projects one receiver through one instance using the same unit-cube,
/// angle-fade and lateral-edge-fade rules as the production shader.
#[must_use]
pub fn project_receiver(
    instance: &DecalInstance,
    receiver: &DecalReceiver,
) -> Option<DecalProjectionHit> {
    let world_to_decal = Mat4::from_cols_array(&instance.world_to_decal);
    let local = world_to_decal.transform_point3(Vec3::from_array([
        receiver.world_position[0],
        receiver.world_position[1],
        receiver.world_position[2],
    ]));
    if !local.is_finite() || local.x.abs() > 0.5 || local.y.abs() > 0.5 || local.z.abs() > 0.5 {
        return None;
    }

    let normal = Vec3::from_array([
        receiver.world_normal[0],
        receiver.world_normal[1],
        receiver.world_normal[2],
    ]);
    let normal = if normal.is_finite() && normal.length_squared() > 1.0e-12 {
        normal.normalize()
    } else {
        Vec3::Z
    };
    let projection_dir = Vec3::from_array([
        instance.projection_fade[0],
        instance.projection_fade[1],
        instance.projection_fade[2],
    ]);
    let projection_dir = if projection_dir.is_finite() && projection_dir.length_squared() > 1.0e-12
    {
        projection_dir.normalize()
    } else {
        Vec3::Z
    };
    let facing = normal.dot(-projection_dir);
    let angle_start = instance.projection_fade[3].clamp(0.0, 1.0);
    let angle_end = instance.fade_material[0].clamp(0.0, 1.0);
    let angle = if facing <= 0.0 {
        0.0
    } else {
        smoothstep(angle_end, angle_start, facing)
    };
    let edge_width = instance.fade_material[1].clamp(0.0, 0.5);
    let edge = if edge_width <= 0.0 {
        1.0
    } else {
        smoothstep(
            0.0,
            edge_width,
            (0.5 - local.x.abs().max(local.y.abs())).max(0.0),
        )
    };
    Some(DecalProjectionHit {
        fade: (angle * edge).clamp(0.0, 1.0),
    })
}

/// Resolves a receiver against one candidate instance/material pair.
#[must_use]
pub fn resolve_decal(
    receiver: &DecalReceiver,
    instance: &DecalInstance,
    material: &DecalMaterial,
) -> DecalResolvedReceiver {
    let base = DecalResolvedReceiver {
        albedo: receiver.albedo,
        normal: receiver.world_normal,
        scalar_coverage: [receiver.scalar[0], receiver.scalar[1], 0.0, 0.0],
    };
    resolve_decal_over(&base, receiver, instance, material)
}

/// Resolves one candidate over an already accumulated receiver result.
///
/// Candidate order is meaningful: later decals are composited over earlier
/// decals, while projection/fade remains evaluated against the original
/// receiver geometry. This is the CPU contract mirrored by the WESL kernel.
#[must_use]
pub fn resolve_decal_over(
    base: &DecalResolvedReceiver,
    receiver: &DecalReceiver,
    instance: &DecalInstance,
    material: &DecalMaterial,
) -> DecalResolvedReceiver {
    let Some(hit) = project_receiver(instance, receiver) else {
        return *base;
    };
    let alpha = (hit.fade * material.scalar_blend[2].clamp(0.0, 1.0)).clamp(0.0, 1.0);
    let additive = material.scalar_blend[3] >= 0.5;
    let blend = |base: f32, decal: f32| {
        if additive {
            (base.max(0.0) + decal.max(0.0) * alpha).max(0.0)
        } else {
            (base.max(0.0) + (decal.max(0.0) - base.max(0.0)) * alpha).max(0.0)
        }
    };
    let receiver_normal = Vec3::from_array([
        receiver.world_normal[0],
        receiver.world_normal[1],
        receiver.world_normal[2],
    ]);
    let receiver_normal = if receiver_normal.length_squared() > 1.0e-12 {
        receiver_normal.normalize()
    } else {
        Vec3::Z
    };
    let base_normal = Vec3::from_array([base.normal[0], base.normal[1], base.normal[2]]);
    let base_normal = if base_normal.length_squared() > 1.0e-12 {
        base_normal.normalize()
    } else {
        receiver_normal
    };
    let tangent_space =
        Vec3::from_array([material.normal[0], material.normal[1], material.normal[2]]);
    let tangent_space = if tangent_space.length_squared() > 1.0e-12 {
        tangent_space.normalize()
    } else {
        Vec3::Z
    };
    let tangent = if receiver_normal.z.abs() < 0.999 {
        receiver_normal.cross(Vec3::Z).normalize()
    } else {
        receiver_normal.cross(Vec3::Y).normalize()
    };
    let bitangent = receiver_normal.cross(tangent);
    let decal_normal = (tangent * tangent_space.x
        + bitangent * tangent_space.y
        + receiver_normal * tangent_space.z)
        .normalize_or_zero();
    let normal = (base_normal + (decal_normal - base_normal) * alpha).normalize_or_zero();
    let normal = if normal.length_squared() > 1.0e-12 {
        normal
    } else {
        base_normal
    };
    DecalResolvedReceiver {
        albedo: [
            blend(base.albedo[0], material.albedo[0]),
            blend(base.albedo[1], material.albedo[1]),
            blend(base.albedo[2], material.albedo[2]),
            base.albedo[3],
        ],
        normal: [normal.x, normal.y, normal.z, 0.0],
        scalar_coverage: [
            blend(base.scalar_coverage[0], material.scalar_blend[0]),
            blend(base.scalar_coverage[1], material.scalar_blend[1]),
            if additive {
                (base.scalar_coverage[2] + alpha).clamp(0.0, 1.0)
            } else {
                (base.scalar_coverage[2] + (1.0 - base.scalar_coverage[2]) * alpha).clamp(0.0, 1.0)
            },
            0.0,
        ],
    }
}

/// Resolves the valid indexed candidates in order, skipping malformed
/// instance/material references without discarding earlier valid candidates.
#[must_use]
pub fn resolve_decal_candidates(
    receiver: &DecalReceiver,
    instances: &[DecalInstance],
    materials: &[DecalMaterial],
    indices: &[u32],
) -> DecalResolvedReceiver {
    let mut resolved = DecalResolvedReceiver {
        albedo: receiver.albedo,
        normal: receiver.world_normal,
        scalar_coverage: [receiver.scalar[0], receiver.scalar[1], 0.0, 0.0],
    };
    for &index in indices {
        let Some(instance) = instances.get(index as usize) else {
            continue;
        };
        let material_index = instance.fade_material[2] as usize;
        let Some(material) = materials.get(material_index) else {
            continue;
        };
        resolved = resolve_decal_over(&resolved, receiver, instance, material);
    }
    resolved
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if edge1 <= edge0 {
        return if value < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance() -> DecalInstance {
        DecalInstance::new(
            Mat4::IDENTITY.to_cols_array(),
            [0.0, 0.0, 1.0],
            0.8,
            0.2,
            0.1,
            0,
        )
    }

    #[test]
    fn abi_strides_are_stable_and_non_placeholder() {
        assert_eq!(DECAL_INSTANCE_STRIDE, 96);
        assert_eq!(DECAL_MATERIAL_STRIDE, 48);
        assert_eq!(DECAL_RECEIVER_STRIDE, 64);
        assert_eq!(DECAL_CLUSTER_RANGE_STRIDE, 16);
        assert_eq!(DECAL_RESOLVED_RECEIVER_STRIDE, 48);
    }

    #[test]
    fn resolve_accepts_front_facing_receiver_and_keeps_flat_normal() {
        let receiver = DecalReceiver {
            world_position: [0.0, 0.0, 0.0, 1.0],
            world_normal: [0.0, 0.0, -1.0, 0.0],
            albedo: [0.2, 0.3, 0.4, 1.0],
            scalar: [0.2, 0.1, 0.0, 0.0],
        };
        let material = DecalMaterial {
            albedo: [1.0, 0.0, 0.0, 1.0],
            normal: [0.0, 0.0, 1.0, 0.0],
            scalar_blend: [0.8, 0.7, 1.0, 0.0],
        };
        let output = resolve_decal(&receiver, &instance(), &material);
        assert!(output.albedo[0] > receiver.albedo[0]);
        assert!((output.normal[2] + 1.0).abs() < 1.0e-5);
        assert!(output.scalar_coverage[2] > 0.0);
    }

    #[test]
    fn miss_and_back_facing_preserve_receiver_and_zero_coverage() {
        let receiver = DecalReceiver::default();
        let material = DecalMaterial::default();
        let mut outside = receiver;
        outside.world_position[0] = 1.0;
        assert_eq!(
            resolve_decal(&outside, &instance(), &material).scalar_coverage[2],
            0.0
        );
        let mut back = receiver;
        back.world_normal = [0.0, 0.0, 1.0, 0.0];
        assert_eq!(
            resolve_decal(&back, &instance(), &material).scalar_coverage[2],
            0.0
        );
    }
}
