use prism_math::Mat4;
use prism_render_architecture::gpu_scene::{InstanceRecord, SceneBounds, SceneTransform};
use prism_render_visibility::{GpuViewRecord, ViewHandle};
use prism_transform::{ecs::LocalBounds, GlobalTransform};

use super::{PrismCamera, PrismFrustum, PrismRenderable};

/// Convert a native affine to the scene ABI's row-major 3×4 representation.
pub fn scene_transform(global: &GlobalTransform) -> SceneTransform {
    let affine = global.affine();
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

/// Conservative world AABB and enclosing sphere, including shear and negative
/// or nonuniform scales. The sphere encloses the transformed AABB's corners.
pub fn scene_bounds(local: &LocalBounds, global: &GlobalTransform) -> SceneBounds {
    let affine = global.affine();
    let center = affine.transform_point3(local.0.center());
    let half = local.0.half_extents();
    let m = affine.matrix3;
    let extent = m.x_axis.abs() * half.x + m.y_axis.abs() * half.y + m.z_axis.abs() * half.z;
    SceneBounds {
        center: center.to_array(),
        radius: extent.length(),
        half_extents: extent.to_array(),
        _padding: 0.0,
    }
}

pub(super) fn instance_record(
    renderable: &PrismRenderable,
    local: &LocalBounds,
    global: &GlobalTransform,
) -> InstanceRecord {
    let transform = scene_transform(global);
    InstanceRecord {
        current_transform: transform,
        previous_transform: transform,
        bounds: scene_bounds(local, global),
        geometry: renderable.geometry,
        material: renderable.material,
        render_layers: renderable.render_layers,
        flags: renderable.flags,
    }
}

fn columns(m: Mat4) -> [[f32; 4]; 4] {
    [
        m.x_axis.to_array(),
        m.y_axis.to_array(),
        m.z_axis.to_array(),
        m.w_axis.to_array(),
    ]
}

pub(super) fn view_record(
    handle: ViewHandle,
    camera: &PrismCamera,
    frustum: &PrismFrustum,
    global: &GlobalTransform,
    previous: Mat4,
) -> GpuViewRecord {
    GpuViewRecord {
        handle,
        clip_from_world: columns(camera.clip_from_world),
        previous_clip_from_world: columns(previous),
        world_position: global.translation().to_array(),
        lod_scale: camera.lod_scale,
        viewport: camera.viewport,
        layer_mask: camera.layer_mask,
        flags: camera.flags,
        history_epoch: camera.history_epoch,
        frustum_planes: frustum.0.planes.map(|plane| {
            let length = plane.normal.length();
            let divisor = if length > 0.0 { length } else { 1.0 };
            let n = plane.normal / divisor;
            [n.x, n.y, n.z, plane.d / divisor]
        }),
    }
}
