//! Render-world extraction of shadow-casting lights into the shadow GPU ABI.
//!
//! This system runs in [`ExtractSchedule`] alongside `extract_lights` and must
//! mirror that pass's light ordering exactly: a directional light's
//! `light_index` is its position among the visible directional lights, and a
//! point light's `light_index` is its position among the visible point lights
//! (which occupy the front of the punctual light buffer, ahead of the spots).
//! Keeping the same query, the same visibility skip, and the same increment
//! order guarantees these indices address the very light each shadow record
//! modulates in the resolve pass.
//!
//! The heavy lifting is delegated to the CPU golden reference
//! [`prism_render_shading::shadow`]: [`compute_cascade_splits`] /
//! [`compute_cascade_matrices`] build the stabilized directional cascade
//! matrices, and [`allocate_shadow_atlas`] hands out contiguous atlas layer
//! ranges by descending importance.  Extraction only adapts Bevy's ECS light
//! and camera components into those calls and packs the results into the flat
//! [`GpuDirectionalShadow`] / [`GpuPointShadow`] records via their
//! `from_reference` constructors.
//!
//! # Camera fitting
//! Cascade fitting needs a *finite* right-handed perspective inverse
//! view-projection (the reference unprojects the wgpu NDC cube corners into
//! world space).  Bevy's default camera projection is an infinite reverse-`Z`
//! frustum whose far plane is at infinity, so we rebuild a finite perspective
//! from the camera's field of view / aspect and the settings' `max_distance`
//! rather than reusing the camera's own clip matrix.  Only perspective cameras
//! drive directional cascades; point-light cube shadows are camera-independent
//! and are always extracted.

use bevy_camera::visibility::ViewVisibility;
use bevy_camera::{Camera, Projection};
use bevy_ecs::prelude::*;
use bevy_light::{DirectionalLight, PointLight};
use bevy_math::{ops, Mat4};
use bevy_render::Extract;
use bevy_transform::components::GlobalTransform;
use prism_render_shading::{
    allocate_shadow_atlas, compute_cascade_matrices, compute_cascade_splits, AtlasConfig,
    CascadeMatrix, CascadeSplits, DirectionalShadowConfig, PointShadowConfig, ShadowKind,
    ShadowRequest, MAX_CASCADE_COUNT,
};

use super::abi::{GpuDirectionalShadow, GpuPointShadow, GpuShadowGlobals};
use super::resources::{ExtractedShadows, ShadowAtlasConfig};
use super::settings::PrismShadowSettings;

/// Atlas request-id offset separating point-light ids from directional ones so
/// a single [`allocate_shadow_atlas`] call can rank both kinds together while
/// the back-fill can still recover which caster a slot belongs to.  Directional
/// lights are capped far below this, so the two id spaces never overlap.
const POINT_LIGHT_ID_OFFSET: u32 = 1 << 16;

/// Importance assigned to every directional caster.  Directionals (sun/moon)
/// are the dominant lighting contributor, so they must always claim their atlas
/// cascades ahead of any point light when the layer budget is exhausted.  The
/// allocator ranks by [`f32::total_cmp`], under which `+inf` outranks every
/// finite point importance regardless of how intense the point light is, while
/// ties among directionals fall back to ascending `light_id` (insertion order).
const DIRECTIONAL_IMPORTANCE: f32 = f32::INFINITY;

/// Finite right-handed perspective projection into wgpu clip space with `z` in
/// `[0, 1]`, column-major.  This is a byte-for-byte re-derivation of the
/// `prism_render_shading::shadow::math::perspective_rh_01` golden reference (and
/// of the now-deprecated `glam` `perspective_rh`), kept local so the extracted
/// camera fit unprojects NDC corners identically to the CPU cascade math.
/// `fov_y_radians` is the full vertical field of view; `aspect` is width /
/// height.
fn perspective_rh_01(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let h = ops::tan(fov_y_radians * 0.5).recip();
    let w = h / aspect;
    let r = far / (near - far);
    Mat4::from_cols_array(&[
        w, 0.0, 0.0, 0.0, //
        0.0, h, 0.0, 0.0, //
        0.0, 0.0, r, -1.0, //
        0.0, 0.0, r * near, 0.0,
    ])
}

/// A visible, shadow-casting directional light resolved against the camera.
pub(crate) struct DirectionalCaster {
    /// Index of the modulated light among the visible directional lights.
    light_index: u32,
    /// Per-cascade stabilized world -> light-clip matrices.
    matrices: [CascadeMatrix; MAX_CASCADE_COUNT],
    /// The `PSSM` split table these matrices were fit to.
    splits: CascadeSplits,
    /// Bias/blend/filter tunables baked into the emitted record.
    config: DirectionalShadowConfig,
}

/// A visible, shadow-casting point light.
pub(crate) struct PointCaster {
    /// Index of the modulated light among the visible point lights.
    light_index: u32,
    /// World-space emitter position.
    position: [f32; 3],
    /// Far range normalizing the stored cube-face distances.
    range: f32,
    /// Relative priority for atlas allocation (brighter lights win slots).
    importance: f32,
}

/// A finite perspective camera resolved for cascade fitting.
struct CameraFit {
    /// Column-major inverse of the finite `clip_from_world` matrix.
    inverse_view_projection: [f32; 16],
    /// View-space near distance of the finite frustum.
    near: f32,
    /// View-space far distance (the settings' shadow `max_distance`).
    far: f32,
}

/// Extracts every shadow-casting light into [`ExtractedShadows`] for the frame.
pub(crate) fn extract_shadows(
    mut extracted: ResMut<ExtractedShadows>,
    settings: Res<PrismShadowSettings>,
    atlas_config: Res<ShadowAtlasConfig>,
    cameras: Extract<Query<(&Camera, &GlobalTransform, &Projection)>>,
    directionals: Extract<Query<(&DirectionalLight, &GlobalTransform, Option<&ViewVisibility>)>>,
    points: Extract<Query<(&PointLight, &GlobalTransform, Option<&ViewVisibility>)>>,
) {
    let cascade_count = (settings.cascade_count.clamp(1, MAX_CASCADE_COUNT as u32)) as usize;

    // Directional cascades need the active camera's finite frustum; without a
    // camera we still extract point lights (their cube shadows are view
    // independent) but emit no directional records.
    let camera = select_primary_camera(&cameras, settings.max_distance);

    let mut dir_casters: Vec<DirectionalCaster> = Vec::new();
    if let Some(camera) = camera {
        let mut light_index = 0u32;
        for (light, transform, visibility) in &directionals {
            if is_hidden(visibility) {
                continue;
            }
            // Advance the index for every *visible* directional so it stays in
            // lock-step with `extract_lights`, even for non-casters.
            let index = light_index;
            light_index += 1;
            if !light.shadow_maps_enabled {
                continue;
            }
            let forward = transform.forward();
            let light_direction = [forward.x, forward.y, forward.z];
            let splits = compute_cascade_splits(
                camera.near,
                camera.far,
                cascade_count,
                settings.split_lambda,
            );
            let matrices = compute_cascade_matrices(
                &camera.inverse_view_projection,
                light_direction,
                camera.near,
                camera.far,
                &splits,
                atlas_config.resolution,
            );
            dir_casters.push(DirectionalCaster {
                light_index: index,
                matrices,
                splits,
                config: settings.directional,
            });
        }
    }

    let mut point_casters: Vec<PointCaster> = Vec::new();
    let mut point_index = 0u32;
    for (light, transform, visibility) in &points {
        if is_hidden(visibility) {
            continue;
        }
        let index = point_index;
        point_index += 1;
        if !light.shadow_maps_enabled {
            continue;
        }
        let position = transform.translation();
        point_casters.push(PointCaster {
            light_index: index,
            position: [position.x, position.y, position.z],
            range: light.range,
            importance: light.intensity.max(0.0),
        });
    }

    assemble_shadows(
        &dir_casters,
        &point_casters,
        &atlas_config,
        &settings.point,
        &mut extracted,
    );
}

/// Selects the highest-order active perspective camera and builds the finite
/// inverse view-projection cascade fitting requires.  Returns `None` when no
/// active perspective camera exists (orthographic / custom projections do not
/// drive the `PSSM` cascade scheme).
fn select_primary_camera(
    cameras: &Query<(&Camera, &GlobalTransform, &Projection)>,
    max_distance: f32,
) -> Option<CameraFit> {
    let mut best: Option<(isize, &GlobalTransform, &Projection)> = None;
    for (camera, transform, projection) in cameras.iter() {
        if !camera.is_active {
            continue;
        }
        let take = match best {
            Some((order, _, _)) => camera.order >= order,
            None => true,
        };
        if take {
            best = Some((camera.order, transform, projection));
        }
    }

    let (_, transform, projection) = best?;
    let Projection::Perspective(perspective) = projection else {
        return None;
    };

    let near = perspective.near.max(1.0e-4);
    let far = max_distance.max(near + 1.0e-3);
    let aspect = perspective.aspect_ratio.max(1.0e-4);
    let view_from_world = transform.to_matrix().inverse();
    let clip_from_view = perspective_rh_01(perspective.fov, aspect, near, far);
    let clip_from_world = clip_from_view * view_from_world;
    Some(CameraFit {
        inverse_view_projection: clip_from_world.inverse().to_cols_array(),
        near,
        far,
    })
}

/// Ranks the resolved casters, hands each a contiguous atlas layer range, and
/// packs the assigned slots back into [`ExtractedShadows`].
///
/// Split out from [`extract_shadows`] so the allocation / back-fill / header
/// bookkeeping is testable without a render-world `Extract` harness.
pub(crate) fn assemble_shadows(
    dir_casters: &[DirectionalCaster],
    point_casters: &[PointCaster],
    atlas_config: &ShadowAtlasConfig,
    point_config_base: &PointShadowConfig,
    extracted: &mut ExtractedShadows,
) {
    extracted.directionals.clear();
    extracted.points.clear();
    extracted.globals = GpuShadowGlobals::default();

    let mut requests: Vec<ShadowRequest> =
        Vec::with_capacity(dir_casters.len() + point_casters.len());
    for (index, caster) in dir_casters.iter().enumerate() {
        let cascades = (caster.splits.count.clamp(1, MAX_CASCADE_COUNT)) as u32;
        requests.push(ShadowRequest {
            light_id: index as u32,
            kind: ShadowKind::Directional { cascades },
            importance: DIRECTIONAL_IMPORTANCE,
        });
    }
    for (index, caster) in point_casters.iter().enumerate() {
        requests.push(ShadowRequest {
            light_id: POINT_LIGHT_ID_OFFSET + index as u32,
            kind: ShadowKind::Point,
            importance: caster.importance,
        });
    }

    let allocation = allocate_shadow_atlas(
        AtlasConfig::new(atlas_config.max_layers, atlas_config.resolution),
        &requests,
    );

    let texel_uv_size = atlas_config.texel_uv_size();
    let point_config = {
        let mut config = *point_config_base;
        config.texel_uv_size = texel_uv_size;
        config
    };

    for slot in &allocation.slots {
        if slot.light_id >= POINT_LIGHT_ID_OFFSET {
            let caster = &point_casters[(slot.light_id - POINT_LIGHT_ID_OFFSET) as usize];
            extracted.points.push(GpuPointShadow::from_reference(
                caster.position,
                caster.range,
                &point_config,
                slot.base_layer,
                caster.light_index,
            ));
        } else {
            let caster = &dir_casters[slot.light_id as usize];
            extracted.directionals.push(GpuDirectionalShadow::from_reference(
                &caster.matrices,
                &caster.splits,
                &caster.config,
                texel_uv_size,
                slot.base_layer,
                caster.light_index,
            ));
        }
    }

    extracted.globals = GpuShadowGlobals {
        directional_count: extracted.directionals.len() as u32,
        point_count: extracted.points.len() as u32,
        atlas_resolution: atlas_config.resolution,
        _padding: 0,
    };
}

/// A light is contributing unless it carries an explicitly-hidden
/// [`ViewVisibility`]; lights without the component are treated as visible,
/// matching `extract_lights`.
fn is_hidden(visibility: Option<&ViewVisibility>) -> bool {
    visibility.is_some_and(|visibility| !visibility.get())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::ShadowFilter;

    fn directional_config() -> DirectionalShadowConfig {
        DirectionalShadowConfig {
            normal_offset_scale: 2.0,
            const_depth_bias: 0.0005,
            slope_depth_bias: 0.002,
            max_depth_bias: 0.02,
            cascade_blend_fraction: 0.1,
            filter: ShadowFilter::Pcf { radius: 2 },
        }
    }

    fn point_config() -> PointShadowConfig {
        PointShadowConfig {
            const_bias: 0.001,
            slope_bias: 0.002,
            max_bias: 0.02,
            pcf_radius: 1,
            texel_uv_size: [0.0, 0.0],
        }
    }

    /// A finite inverse view-projection for a camera at the origin looking down
    /// `-z`, matching the wgpu `z in [0, 1]` clip the reference expects.
    fn camera_fit(near: f32, far: f32) -> CameraFit {
        let clip_from_world = perspective_rh_01(1.0, 1.5, near, far);
        CameraFit {
            inverse_view_projection: clip_from_world.inverse().to_cols_array(),
            near,
            far,
        }
    }

    fn directional_caster(fit: &CameraFit, light_index: u32) -> DirectionalCaster {
        let splits = compute_cascade_splits(fit.near, fit.far, 4, 0.5);
        let matrices = compute_cascade_matrices(
            &fit.inverse_view_projection,
            [0.3, -1.0, 0.2],
            fit.near,
            fit.far,
            &splits,
            1024,
        );
        DirectionalCaster {
            light_index,
            matrices,
            splits,
            config: directional_config(),
        }
    }

    #[test]
    fn assembles_one_directional_and_one_point() {
        let fit = camera_fit(0.1, 200.0);
        let dir = vec![directional_caster(&fit, 0)];
        let points = vec![PointCaster {
            light_index: 0,
            position: [1.0, 2.0, 3.0],
            range: 25.0,
            importance: 1000.0,
        }];
        let atlas = ShadowAtlasConfig::new(16, 1024);
        let mut extracted = ExtractedShadows::default();

        assemble_shadows(&dir, &points, &atlas, &point_config(), &mut extracted);

        assert_eq!(extracted.directionals.len(), 1);
        assert_eq!(extracted.points.len(), 1);
        assert_eq!(extracted.globals.directional_count, 1);
        assert_eq!(extracted.globals.point_count, 1);
        assert_eq!(extracted.globals.atlas_resolution, 1024);
        // The point record inherits the live atlas texel size, not the base
        // config's placeholder zero.
        assert_eq!(extracted.points[0].texel_uv_size, [1.0 / 1024.0, 1.0 / 1024.0]);
        assert_eq!(extracted.points[0].light_index, 0);
        assert_eq!(extracted.directionals[0].light_index, 0);
        assert_eq!(extracted.directionals[0].enabled, 1);
    }

    #[test]
    fn directional_claims_layers_ahead_of_points_under_pressure() {
        // Budget of six layers: a four-cascade sun (4 layers) plus a point light
        // (6 layers) cannot both fit; the sun must win by importance.
        let fit = camera_fit(0.1, 200.0);
        let dir = vec![directional_caster(&fit, 0)];
        let points = vec![PointCaster {
            light_index: 0,
            position: [0.0, 0.0, 0.0],
            range: 10.0,
            importance: f32::MAX,
        }];
        let atlas = ShadowAtlasConfig::new(6, 512);
        let mut extracted = ExtractedShadows::default();

        assemble_shadows(&dir, &points, &atlas, &point_config(), &mut extracted);

        assert_eq!(extracted.directionals.len(), 1);
        assert_eq!(extracted.points.len(), 0);
        assert_eq!(extracted.globals.directional_count, 1);
        assert_eq!(extracted.globals.point_count, 0);
    }

    #[test]
    fn base_layers_are_distinct_and_indices_survive_reordering() {
        let fit = camera_fit(0.1, 150.0);
        let dir = vec![directional_caster(&fit, 0), directional_caster(&fit, 1)];
        let points = vec![
            PointCaster {
                light_index: 0,
                position: [5.0, 0.0, 0.0],
                range: 20.0,
                importance: 5.0,
            },
            PointCaster {
                light_index: 1,
                position: [-5.0, 0.0, 0.0],
                range: 20.0,
                importance: 500.0,
            },
        ];
        let atlas = ShadowAtlasConfig::new(32, 1024);
        let mut extracted = ExtractedShadows::default();

        assemble_shadows(&dir, &points, &atlas, &point_config(), &mut extracted);

        assert_eq!(extracted.directionals.len(), 2);
        assert_eq!(extracted.points.len(), 2);

        // Every emitted record must reference a distinct base layer.
        let mut layers: Vec<u32> = extracted
            .directionals
            .iter()
            .map(|record| record.base_layer)
            .chain(extracted.points.iter().map(|record| record.base_layer))
            .collect();
        layers.sort_unstable();
        layers.dedup();
        assert_eq!(layers.len(), 4);

        // Point light_index values are preserved even though the brighter light
        // (importance 500) is allocated first.
        let mut point_indices: Vec<u32> =
            extracted.points.iter().map(|record| record.light_index).collect();
        point_indices.sort_unstable();
        assert_eq!(point_indices, vec![0, 1]);
    }

    #[test]
    fn empty_input_produces_empty_frame_with_authoritative_header() {
        let atlas = ShadowAtlasConfig::new(16, 2048);
        let mut extracted = ExtractedShadows::default();
        extracted.directionals.push(GpuDirectionalShadow::default());

        assemble_shadows(&[], &[], &atlas, &point_config(), &mut extracted);

        assert!(extracted.directionals.is_empty());
        assert!(extracted.points.is_empty());
        assert_eq!(extracted.globals.directional_count, 0);
        assert_eq!(extracted.globals.point_count, 0);
        assert_eq!(extracted.globals.atlas_resolution, 2048);
    }
}
