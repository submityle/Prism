//! The CPU reference cull the GPU result is checked against.

use alloc::vec::Vec;

use crate::camera::OrthoCamera;
use crate::frustum::{aabb_inside, Aabb, Frustum};

/// The ascending indices of every instance whose AABB survives `frustum`.
///
/// This is the ground truth for the GPU pass: it runs the same positive-vertex
/// test ([`aabb_inside`]) over the same instance array, in ascending id order,
/// so the GPU's atomically appended list must contain exactly these indices.
#[must_use]
pub fn cull_cpu(frustum: &Frustum, instances: &[Aabb]) -> Vec<u32> {
    instances
        .iter()
        .enumerate()
        .filter(|(_, aabb)| aabb_inside(frustum, aabb))
        .map(|(index, _)| index as u32)
        .collect()
}

/// The exact pixel an instance centred on `center` paints at, or `None` when the
/// centre falls outside the camera's view.
///
/// The vertex shader places the quad symmetrically about the centre, so the
/// centre texel is inside the quad and shows the instance's own color. Because
/// the scene's survivors all share one Z coordinate, this projection is
/// injective and no two survivors compete for the same pixel.
#[must_use]
pub fn projected_center_pixel(camera: &OrthoCamera, center: [f32; 3]) -> Option<[u32; 2]> {
    camera.pixel_index([center[0], center[1]])
}
