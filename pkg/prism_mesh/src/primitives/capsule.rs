use alloc::vec::Vec;

use super::common::{build_mesh, positive, subdivisions};
use super::PrimitiveError;

/// Builds a Y-axis capsule from two hemispheres and a cylindrical section.
#[must_use = "the generated mesh must be retained"]
pub fn capsule(
    radius: f32,
    half_length: f32,
    sectors: u32,
    hemisphere_rings: u32,
) -> Result<crate::Mesh, PrimitiveError> {
    let radius = positive("radius", radius)?;
    let half_length = if half_length.is_finite() && half_length >= 0.0 {
        half_length
    } else {
        return Err(PrimitiveError::InvalidDimension {
            name: "half_length",
            value: half_length,
        });
    };
    let sectors = subdivisions("sectors", sectors, 3)?;
    let hemisphere_rings = subdivisions("hemisphere_rings", hemisphere_rings, 1)?;
    let ring_count = hemisphere_rings * 2 + 2;
    let stride = sectors + 1;
    let mut positions = Vec::with_capacity((ring_count * stride) as usize);
    let mut normals = Vec::with_capacity(positions.capacity());
    let mut uvs = Vec::with_capacity(positions.capacity());
    for ring in 0..ring_count {
        let (y, ring_radius, normal_y) = if ring <= hemisphere_rings {
            let angle = core::f32::consts::FRAC_PI_2 * ring as f32 / hemisphere_rings as f32;
            (
                half_length + angle.cos() * radius,
                angle.sin() * radius,
                angle.cos(),
            )
        } else if ring == hemisphere_rings + 1 {
            (-half_length, radius, 0.0)
        } else {
            let bottom_ring = ring - (hemisphere_rings + 1);
            let angle = core::f32::consts::FRAC_PI_2 * bottom_ring as f32 / hemisphere_rings as f32;
            (
                -half_length - angle.sin() * radius,
                angle.cos() * radius,
                -angle.sin(),
            )
        };
        let v = ring as f32 / (ring_count - 1) as f32;
        for sector in 0..=sectors {
            let u = sector as f32 / sectors as f32;
            let theta = core::f32::consts::TAU * u;
            let radial = [theta.cos(), 0.0, theta.sin()];
            let normal = [
                radial[0] * (ring_radius / radius),
                normal_y,
                radial[2] * (ring_radius / radius),
            ];
            positions.push([radial[0] * ring_radius, y, radial[2] * ring_radius]);
            normals.push(normal);
            uvs.push([u, v]);
        }
    }
    let mut indices = Vec::with_capacity(((ring_count - 1) * sectors * 6) as usize);
    for ring in 0..ring_count - 1 {
        for sector in 0..sectors {
            let a = ring * stride + sector;
            let b = a + stride;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    build_mesh(positions, normals, uvs, indices)
}
