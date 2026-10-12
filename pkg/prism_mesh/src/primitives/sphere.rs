//! The UV sphere primitive.

use core::f32::consts::{PI, TAU};

use super::finish;
use crate::mesh::Mesh;
use crate::vertex::Vertex;

/// Builds a UV sphere of `radius`, tessellated into `sectors` longitude steps
/// and `stacks` latitude bands.
///
/// Vertices lie exactly on the sphere, so each vertex's `position` doubles as
/// its unit normal. UVs run from `(0, 0)` at the north pole to `(1, 1)` at the
/// south, with `sectors + 1` columns so the seam has its own vertices. Every
/// triangle winds counter-clockwise seen from outside; the degenerate triangles
/// at the poles are kept so the counts follow the closed forms
/// `(stacks + 1) * (sectors + 1)` vertices and `stacks * sectors * 6` indices.
///
/// # Panics
///
/// Panics if `sectors < 3` or `stacks < 2`.
#[expect(
    clippy::disallowed_methods,
    reason = "primitive generation is authoring-time CPU code that needs real sin/cos; prism_mesh must not depend on a backend or Bevy crate, and Prism's deterministic libm-free facade (prism_math::ops) is not public yet"
)]
#[must_use]
pub fn uv_sphere(radius: f32, sectors: u32, stacks: u32) -> Mesh {
    assert!(sectors >= 3, "a UV sphere needs at least 3 sectors");
    assert!(stacks >= 2, "a UV sphere needs at least 2 stacks");

    let columns = sectors + 1;
    let mut vertices = Vec::with_capacity(((stacks + 1) * columns) as usize);
    for stack in 0..=stacks {
        let phi = PI * stack as f32 / stacks as f32; // 0 at the north (+Y) pole.
        let (sin_phi, cos_phi) = phi.sin_cos();
        let v = stack as f32 / stacks as f32;
        for sector in 0..=sectors {
            let theta = TAU * sector as f32 / sectors as f32;
            let (sin_theta, cos_theta) = theta.sin_cos();
            let normal = [sin_phi * cos_theta, cos_phi, sin_phi * sin_theta];
            vertices.push(Vertex::new(
                [normal[0] * radius, normal[1] * radius, normal[2] * radius],
                normal,
                [sector as f32 / sectors as f32, v],
            ));
        }
    }

    let mut indices = Vec::with_capacity((stacks * sectors * 6) as usize);
    for stack in 0..stacks {
        for sector in 0..sectors {
            let a = stack * columns + sector;
            let b = a + columns;
            // (a, a+1, b+1) + (a, b+1, b) wind counter-clockwise from outside.
            indices.extend_from_slice(&[a, a + 1, b + 1, a, b + 1, b]);
        }
    }
    finish(&vertices, indices)
}
