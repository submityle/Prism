//! The CPU reference cull the GPU result is checked against.

use alloc::vec::Vec;

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
