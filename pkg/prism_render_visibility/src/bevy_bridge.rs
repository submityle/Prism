//! Bevy ECS bridge for the unified visibility ABI.
//!
//! Bevy keeps per-entity bounds in *local* space ([`bevy_camera::primitives::Aabb`])
//! plus a world [`GlobalTransform`], and keeps camera culling planes in a
//! [`bevy_camera::primitives::Frustum`]. Prism's visibility core instead consumes
//! engine-neutral, GPU-friendly records ([`GpuViewRecord`], [`SceneBounds`],
//! [`InstanceRecord`]) so the frozen ABI never depends on Bevy types.
//!
//! This module is the single, tested conversion seam between the two. It mirrors
//! the convention already used by `prism_render_material::bevy_bridge` (an
//! optional `bevy` feature that *lowers* Bevy data into Prism records), so Bevy
//! can adopt Prism's GPU-driven, two-phase visibility without the two
//! representations drifting apart.
//!
//! All conversions here are allocation-free and side-effect-free; higher layers
//! decide how to batch them (extract systems, change-detection, etc.).

use bevy_camera::primitives::{Aabb, Frustum};
use bevy_math::{Mat4, Vec3, Vec3A};
use bevy_transform::components::GlobalTransform;
use prism_render_architecture::gpu_scene::{
    GeometryHandle, InstanceRecord, SceneBounds, SceneMaterialHandle, SceneTransform,
};

use crate::{GpuViewRecord, ViewFlags, ViewHandle};

/// Parameters that describe a Bevy camera view in the terms Prism needs but
/// that are not recoverable from the [`Frustum`] alone.
#[derive(Clone, Copy, Debug)]
pub struct BevyViewParams {
    /// Stable handle identifying this view across frames (drives history reuse).
    pub handle: ViewHandle,
    /// `clip_from_world` (a.k.a. view-projection) matrix for the view.
    pub clip_from_world: Mat4,
    /// Previous frame's `clip_from_world`; pass the current one on a camera cut.
    pub previous_clip_from_world: Mat4,
    /// Camera world-space position (used for distance-based LOD selection).
    pub world_position: Vec3,
    /// `[x, y, width, height]` in pixels.
    pub viewport: [u32; 4],
    /// Global LOD bias; `1.0` keeps authored screen-error thresholds.
    pub lod_scale: f32,
    /// Render-layer mask ANDed against each instance's layers.
    pub layer_mask: u32,
    /// View behavior flags (reverse-Z, shadow, reflection, offline, cut, ...).
    pub flags: ViewFlags,
    /// Monotonic history epoch; a changed epoch forces a history reset.
    pub history_epoch: u64,
}

impl Default for BevyViewParams {
    fn default() -> Self {
        Self {
            handle: ViewHandle::INVALID,
            clip_from_world: Mat4::IDENTITY,
            previous_clip_from_world: Mat4::IDENTITY,
            world_position: Vec3::ZERO,
            viewport: [0, 0, 0, 0],
            lod_scale: 1.0,
            layer_mask: u32::MAX,
            flags: ViewFlags::default(),
            history_epoch: 0,
        }
    }
}

/// Converts a Bevy [`Frustum`] (six inward-pointing half-spaces, ordered
/// left, right, top, bottom, near, far) into Prism's `[[f32; 4]; 6]` plane set.
///
/// Both engines use the same convention — a point `p` is inside when
/// `normal·p + d > 0` — so the half-space's `Vec4(normal, d)` maps across
/// component-for-component with no sign flip.
#[inline]
pub fn frustum_planes(frustum: &Frustum) -> [[f32; 4]; 6] {
    let mut planes = [[0.0_f32; 4]; 6];
    for (dst, half_space) in planes.iter_mut().zip(frustum.half_spaces.iter()) {
        *dst = half_space.normal_d().to_array();
    }
    planes
}

/// Builds a [`GpuViewRecord`] from a Bevy camera's [`Frustum`] and the
/// supplementary [`BevyViewParams`].
#[inline]
pub fn view_record(frustum: &Frustum, params: &BevyViewParams) -> GpuViewRecord {
    GpuViewRecord {
        handle: params.handle,
        clip_from_world: params.clip_from_world.to_cols_array_2d(),
        previous_clip_from_world: params.previous_clip_from_world.to_cols_array_2d(),
        world_position: params.world_position.to_array(),
        lod_scale: params.lod_scale,
        viewport: params.viewport,
        frustum_planes: frustum_planes(frustum),
        layer_mask: params.layer_mask,
        flags: params.flags,
        history_epoch: params.history_epoch,
    }
}

/// Converts a *local-space* Bevy [`Aabb`] plus its world [`GlobalTransform`]
/// into world-space [`SceneBounds`].
///
/// The returned bounding sphere radius and half-extents are a conservative
/// upper bound of the transformed box: for a linear map `M` with columns
/// `cx, cy, cz` and local half-extents `h`, the world half-extents are
/// `|M| · h` (component-wise absolute value) and the sphere radius is the
/// farthest transformed corner from the center. Both bounds enclose the box,
/// so the culler never rejects a genuinely visible instance.
#[inline]
pub fn scene_bounds(local_aabb: &Aabb, world_from_local: &GlobalTransform) -> SceneBounds {
    let affine = world_from_local.affine();
    let m = affine.matrix3;
    let center = affine.transform_point3a(local_aabb.center);
    let h = local_aabb.half_extents;

    let (cx, cy, cz) = (m.x_axis, m.y_axis, m.z_axis);
    // Tight bounding-sphere radius: the farthest transformed corner from the
    // center. Only four of the eight sign combinations are distinct up to
    // negation, which does not change a length.
    let (a, b, c) = (cx * h.x, cy * h.y, cz * h.z);
    let radius = (a + b + c)
        .length()
        .max((a + b - c).length())
        .max((a - b + c).length())
        .max((a - b - c).length());
    let abs_h = Vec3A::new(
        h.x * cx.x.abs() + h.y * cy.x.abs() + h.z * cz.x.abs(),
        h.x * cx.y.abs() + h.y * cy.y.abs() + h.z * cz.y.abs(),
        h.x * cx.z.abs() + h.y * cy.z.abs() + h.z * cz.z.abs(),
    );

    SceneBounds {
        center: center.into(),
        radius,
        half_extents: abs_h.into(),
        _padding: 0.0,
    }
}

/// Lowers a Bevy world [`GlobalTransform`] into the architecture contract's
/// row-major 3×4 [`SceneTransform`].
#[inline]
pub fn scene_transform(world_from_local: &GlobalTransform) -> SceneTransform {
    let affine = world_from_local.affine();
    let m = affine.matrix3;
    let t = affine.translation;
    SceneTransform {
        rows: [
            [m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x],
            [m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y],
            [m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z],
        ],
    }
}

/// Assembles a complete [`InstanceRecord`] ready to publish into a
/// `CpuRenderScene` from a Bevy renderable's local bounds, world transform, and
/// backend geometry/material handles.
///
/// `previous_transform` defaults to the current transform; callers that track
/// motion vectors should overwrite it with last frame's value.
#[inline]
pub fn instance_record(
    local_aabb: &Aabb,
    world_from_local: &GlobalTransform,
    geometry: GeometryHandle,
    material: SceneMaterialHandle,
    render_layers: u32,
    flags: u32,
) -> InstanceRecord {
    let current = scene_transform(world_from_local);
    InstanceRecord {
        current_transform: current,
        previous_transform: current,
        bounds: scene_bounds(local_aabb, world_from_local),
        geometry,
        material,
        render_layers,
        flags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::{BTreeMap, BTreeSet};
    use bevy_math::{Quat, Vec4};
    use bevy_transform::components::GlobalTransform;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{
        CpuRenderScene, SceneOperation, SceneTransactionBuilder,
    };
    use prism_render_material::{
        GpuSurfaceParameters, MaterialDomain, MaterialFeatureFlags, MaterialRecord,
        MaterialRenderClass,
    };

    use crate::{cull_view, GeometryLod, GeometryLodChain, VisibilityInput};

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle {
            index,
            generation: 1,
        }
    }

    // A frustum that admits points in front of the camera (looking down -Z),
    // built with the same inward-normal / `n·p + d > 0` convention Bevy uses.
    fn forward_frustum() -> Frustum {
        use bevy_shape::{HalfSpace, ViewFrustum};
        Frustum(ViewFrustum {
            half_spaces: [
                // left: x > -10
                HalfSpace::new(Vec4::new(1.0, 0.0, 0.0, 10.0)),
                // right: x < 10
                HalfSpace::new(Vec4::new(-1.0, 0.0, 0.0, 10.0)),
                // top: y < 10
                HalfSpace::new(Vec4::new(0.0, -1.0, 0.0, 10.0)),
                // bottom: y > -10
                HalfSpace::new(Vec4::new(0.0, 1.0, 0.0, 10.0)),
                // near: z < -1  (in front of camera)
                HalfSpace::new(Vec4::new(0.0, 0.0, -1.0, -1.0)),
                // far: z > -100
                HalfSpace::new(Vec4::new(0.0, 0.0, 1.0, 100.0)),
            ],
        })
    }

    #[test]
    fn frustum_planes_match_half_space_components() {
        let frustum = forward_frustum();
        let planes = frustum_planes(&frustum);
        for (plane, half_space) in planes.iter().zip(frustum.half_spaces.iter()) {
            assert_eq!(*plane, half_space.normal_d().to_array());
        }
    }

    #[test]
    fn identity_bounds_round_trip() {
        let aabb = Aabb::from_min_max(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
        let bounds = scene_bounds(&aabb, &GlobalTransform::IDENTITY);
        assert_eq!(bounds.center, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.half_extents, [1.0, 2.0, 3.0]);
        // sqrt(1+4+9) for an axis-aligned unit-column box.
        let expected = (1.0_f32 + 4.0 + 9.0).sqrt();
        assert!((bounds.radius - expected).abs() < 1.0e-5);
    }

    #[test]
    fn translation_moves_center_only() {
        let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
        let xform = GlobalTransform::from_translation(Vec3::new(5.0, -3.0, 2.0));
        let bounds = scene_bounds(&aabb, &xform);
        assert_eq!(bounds.center, [5.0, -3.0, 2.0]);
        assert_eq!(bounds.half_extents, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn uniform_scale_scales_extents_and_radius() {
        let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
        let xform = GlobalTransform::from_scale(Vec3::splat(2.0));
        let bounds = scene_bounds(&aabb, &xform);
        assert_eq!(bounds.half_extents, [2.0, 2.0, 2.0]);
        let expected = 2.0 * (3.0_f32).sqrt();
        assert!((bounds.radius - expected).abs() < 1.0e-5);
    }

    #[test]
    fn rotation_is_conservative_but_bounded() {
        let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
        // 45° about Y: extents grow but the sphere radius is rotation-invariant.
        let xform =
            GlobalTransform::from_rotation(Quat::from_rotation_y(core::f32::consts::FRAC_PI_4));
        let bounds = scene_bounds(&aabb, &xform);
        let expected_radius = (3.0_f32).sqrt();
        assert!((bounds.radius - expected_radius).abs() < 1.0e-5);
        // A unit cube rotated 45° has an x/z footprint half-width of sqrt(2).
        assert!(bounds.half_extents[0] >= 1.0 - 1.0e-5);
        assert!((bounds.half_extents[0] - (2.0_f32).sqrt()).abs() < 1.0e-4);
        assert!((bounds.half_extents[1] - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn scene_transform_matches_affine_rows() {
        let xform = GlobalTransform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let st = scene_transform(&xform);
        assert_eq!(st.rows[0], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(st.rows[1], [0.0, 1.0, 0.0, 2.0]);
        assert_eq!(st.rows[2], [0.0, 0.0, 1.0, 3.0]);
    }

    fn test_material() -> MaterialRecord {
        MaterialRecord {
            handle: handle(1),
            revision: 1,
            domain: MaterialDomain::default(),
            render_domain: prism_render_material::RenderDomain::Surface,
            render_class: MaterialRenderClass::Opaque,
            response_plan: prism_render_material::ShadingPlan::default(),
            features: MaterialFeatureFlags::default(),
            closure_mask: 0,
            surface: GpuSurfaceParameters::default(),
            textures: alloc::vec::Vec::new(),
        }
    }

    fn lod_chain(geometry: GeometryHandle) -> GeometryLodChain {
        GeometryLodChain {
            geometry,
            lods: alloc::vec![GeometryLod {
                level: 0,
                screen_error: 0.0,
                resident: true,
                fallback: true,
            }],
        }
    }

    // Full end-to-end: build a `CpuRenderScene` from Bevy-shaped inputs through
    // the bridge, then run the real `cull_view` and confirm it keeps an instance
    // inside the frustum and rejects one placed behind the camera.
    #[test]
    fn bridged_instance_culls_through_real_visibility_core() {
        let geometry = handle(7);
        let material_handle = handle(1);
        let scene_handle = handle(1);

        let aabb = Aabb::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));

        let mut geometry_map = BTreeMap::new();
        geometry_map.insert(geometry, lod_chain(geometry));
        let mut material_map = BTreeMap::new();
        material_map.insert(material_handle, test_material());

        let view = view_record(
            &forward_frustum(),
            &BevyViewParams {
                handle: handle(42),
                world_position: Vec3::ZERO,
                layer_mask: u32::MAX,
                ..BevyViewParams::default()
            },
        );

        let build_scene = |world_from_local: &GlobalTransform| {
            let mut scene = CpuRenderScene::default();
            let mut tx = SceneTransactionBuilder::new(1, 1);
            tx.push(SceneOperation::Create {
                handle: scene_handle,
                record: instance_record(
                    &aabb,
                    world_from_local,
                    geometry,
                    material_handle,
                    u32::MAX,
                    0,
                ),
            });
            scene.apply(&tx.finish());
            scene
        };

        let previous_lods = BTreeMap::new();
        let occluded = BTreeSet::new();

        // In front of the camera (-Z) → visible.
        let visible_scene = build_scene(&GlobalTransform::from_translation(Vec3::new(
            0.0, 0.0, -5.0,
        )));
        let (work, stats) = cull_view(
            &view,
            VisibilityInput {
                scene: &visible_scene,
                handles: &[scene_handle],
                geometry: &geometry_map,
                materials: &material_map,
                previous_lods: &previous_lods,
                occluded: &occluded,
                capacity: 16,
                previous_history_epoch: None,
            },
        );
        assert_eq!(work.len(), 1, "instance in frustum should be visible");
        assert_eq!(stats.visible_instances, 1);
        assert_eq!(stats.frustum_rejected, 0);

        // Behind the camera (+Z) → frustum-rejected.
        let hidden_scene = build_scene(&GlobalTransform::from_translation(Vec3::new(
            0.0, 0.0, 50.0,
        )));
        let (work, stats) = cull_view(
            &view,
            VisibilityInput {
                scene: &hidden_scene,
                handles: &[scene_handle],
                geometry: &geometry_map,
                materials: &material_map,
                previous_lods: &previous_lods,
                occluded: &occluded,
                capacity: 16,
                previous_history_epoch: None,
            },
        );
        assert_eq!(work.len(), 0, "instance behind camera should be culled");
        assert_eq!(stats.frustum_rejected, 1);
    }
}
