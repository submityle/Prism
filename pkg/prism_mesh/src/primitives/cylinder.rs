//! The cylinder primitive.

use core::f32::consts::TAU;

use super::finish;
use crate::mesh::Mesh;
use crate::vertex::Vertex;

/// Builds a cylinder of `radius` and `height`, centred on the origin with its
/// axis along `+Y`, tessellated by `radial_segments` around the circumference.
///
/// The result is a single mesh holding the side wall plus both caps. Side
/// normals point radially outward and cap normals are `±Y`; UVs lie within
/// `[0, 1]` and every triangle winds counter-clockwise seen from outside.
/// Counts follow the closed forms `4 * radial_segments + 6` vertices and
/// `12 * radial_segments` indices.
///
/// # Panics
///
/// Panics if `radial_segments < 3`.
#[expect(
    clippy::disallowed_methods,
    reason = "primitive generation is authoring-time CPU code that needs real sin/cos; prism_mesh must not depend on a backend or Bevy crate, and Prism's deterministic libm-free facade (prism_math::ops) is not public yet"
)]
#[must_use]
pub fn cylinder(radius: f32, height: f32, radial_segments: u32) -> Mesh {
    assert!(
        radial_segments >= 3,
        "a cylinder needs at least 3 radial segments"
    );
    let segments = radial_segments;
    let half_height = height * 0.5;
    let mut vertices = Vec::with_capacity((4 * segments + 6) as usize);
    let mut indices = Vec::with_capacity((12 * segments) as usize);

    // Side wall: a top ring then a bottom ring, `segments + 1` columns each so
    // the seam vertex has its own UV. Ring vertex `j` is `2 * j` (top) and
    // `2 * j + 1` (bottom).
    for segment in 0..=segments {
        let theta = TAU * segment as f32 / segments as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let normal = [cos_theta, 0.0, sin_theta];
        let u = segment as f32 / segments as f32;
        vertices.push(Vertex::new(
            [radius * cos_theta, half_height, radius * sin_theta],
            normal,
            [u, 0.0],
        ));
        vertices.push(Vertex::new(
            [radius * cos_theta, -half_height, radius * sin_theta],
            normal,
            [u, 1.0],
        ));
    }
    for segment in 0..segments {
        let top = 2 * segment;
        let top_next = 2 * (segment + 1);
        let bottom = top + 1;
        let bottom_next = top_next + 1;
        indices.extend_from_slice(&[top, bottom_next, bottom, top, top_next, bottom_next]);
    }

    // Top cap: a centre vertex plus a ring, wound (centre, next, current).
    let top_center = u32::try_from(vertices.len()).expect("cylinder vertex count fits u32");
    vertices.push(Vertex::new(
        [0.0, half_height, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5],
    ));
    let top_ring = u32::try_from(vertices.len()).expect("cylinder vertex count fits u32");
    for segment in 0..=segments {
        let theta = TAU * segment as f32 / segments as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let x = radius * cos_theta;
        let z = radius * sin_theta;
        vertices.push(Vertex::new(
            [x, half_height, z],
            [0.0, 1.0, 0.0],
            [x / radius * 0.5 + 0.5, z / radius * 0.5 + 0.5],
        ));
    }
    for segment in 0..segments {
        indices.extend_from_slice(&[top_center, top_ring + segment + 1, top_ring + segment]);
    }

    // Bottom cap: same construction, wound (centre, current, next).
    let bottom_center = u32::try_from(vertices.len()).expect("cylinder vertex count fits u32");
    vertices.push(Vertex::new(
        [0.0, -half_height, 0.0],
        [0.0, -1.0, 0.0],
        [0.5, 0.5],
    ));
    let bottom_ring = u32::try_from(vertices.len()).expect("cylinder vertex count fits u32");
    for segment in 0..=segments {
        let theta = TAU * segment as f32 / segments as f32;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let x = radius * cos_theta;
        let z = radius * sin_theta;
        vertices.push(Vertex::new(
            [x, -half_height, z],
            [0.0, -1.0, 0.0],
            [x / radius * 0.5 + 0.5, z / radius * 0.5 + 0.5],
        ));
    }
    for segment in 0..segments {
        indices.extend_from_slice(&[
            bottom_center,
            bottom_ring + segment,
            bottom_ring + segment + 1,
        ]);
    }

    finish(&vertices, indices)
}
