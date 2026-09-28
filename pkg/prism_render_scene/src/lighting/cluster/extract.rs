//! Render-world extraction of the active camera's froxel-grid view.
//!
//! Clustered light culling is view dependent: the froxel bounds are
//! reconstructed from the camera's projection, and the light positions are
//! tested in that camera's view space.  This system runs in [`ExtractSchedule`]
//! alongside `extract_lights`/`extract_shadows`, selects the same primary
//! active perspective camera the shadow pass fits its cascades to, and records
//! the finite view/projection plus render-target size the CPU golden needs.
//!
//! Only perspective cameras drive the froxel grid; orthographic and custom
//! projections leave [`ExtractedClusterView::view`] `None`, in which case the
//! prepare step falls back to a neutral single-cluster grid.

use bevy_camera::{Camera, Projection};
use bevy_ecs::prelude::*;
use bevy_math::{ops, Mat4};
use bevy_render::Extract;
use bevy_transform::components::GlobalTransform;

/// The finite view fit the froxel grid is built against for one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterViewFit {
    /// World -> view (camera looking down `-z`), column-major `[f32; 16]`.
    pub view_from_world: [f32; 16],
    /// Clip-from-view (finite right-handed, `z in [0, 1]`), column-major.
    pub clip_from_view: [f32; 16],
    /// Render-target resolution in pixels.
    pub screen_size: [u32; 2],
    /// Near-plane distance of the froxel depth slicing.
    pub near: f32,
    /// Far-plane distance of the froxel depth slicing.
    pub far: f32,
}

/// The active camera's froxel-grid view, refreshed every frame.
#[derive(Resource, Default)]
pub struct ExtractedClusterView {
    /// The primary perspective camera's fit, or `None` when no active
    /// perspective camera exists this frame.
    pub view: Option<ClusterViewFit>,
}

/// Finite right-handed perspective into wgpu clip space (`z in [0, 1]`), column
/// major.  Byte-for-byte identical to the shadow pass's `perspective_rh_01`, so
/// the froxel bounds unproject exactly as the CPU golden expects.
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

/// Extracts the primary camera's froxel-grid view.
pub(crate) fn extract_cluster_view(
    mut extracted: ResMut<ExtractedClusterView>,
    cameras: Extract<Query<(&Camera, &GlobalTransform, &Projection)>>,
) {
    extracted.view = select_primary_camera(&cameras);
}

/// Selects the highest-order active perspective camera and builds the finite
/// view/projection the froxel grid needs.  Returns `None` when no active
/// perspective camera with a known viewport size exists.
fn select_primary_camera(
    cameras: &Query<(&Camera, &GlobalTransform, &Projection)>,
) -> Option<ClusterViewFit> {
    let mut best: Option<(isize, &Camera, &GlobalTransform, &Projection)> = None;
    for (camera, transform, projection) in cameras.iter() {
        if !camera.is_active {
            continue;
        }
        let take = match best {
            Some((order, ..)) => camera.order >= order,
            None => true,
        };
        if take {
            best = Some((camera.order, camera, transform, projection));
        }
    }

    let (_, camera, transform, projection) = best?;
    let Projection::Perspective(perspective) = projection else {
        return None;
    };
    let screen = camera.physical_viewport_size()?;

    let near = perspective.near.max(1.0e-4);
    let far = perspective.far.max(near + 1.0e-3);
    let aspect = perspective.aspect_ratio.max(1.0e-4);
    let view_from_world = transform.to_matrix().inverse();
    let clip_from_view = perspective_rh_01(perspective.fov, aspect, near, far);

    Some(ClusterViewFit {
        view_from_world: view_from_world.to_cols_array(),
        clip_from_view: clip_from_view.to_cols_array(),
        screen_size: [screen.x.max(1), screen.y.max(1)],
        near,
        far,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perspective_matches_the_shadow_pass_derivation() {
        // A 1:1, 90-degree perspective: the horizontal/vertical scales are 1.
        let m = perspective_rh_01(core::f32::consts::FRAC_PI_2, 1.0, 1.0, 100.0);
        let cols = m.to_cols_array();
        assert!((cols[0] - 1.0).abs() < 1.0e-5, "w scale");
        assert!((cols[5] - 1.0).abs() < 1.0e-5, "h scale");
        // Right-handed `z in [0, 1]`: the [2][3] entry is -1 (perspective divide).
        assert!((cols[11] + 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn default_extracted_view_is_absent() {
        assert!(ExtractedClusterView::default().view.is_none());
    }
}
