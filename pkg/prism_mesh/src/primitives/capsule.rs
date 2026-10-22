use super::common::{build_mesh, generated_counts, positive, reserve, subdivisions};
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
    let ring_count = hemisphere_rings
        .checked_mul(2)
        .and_then(|count| count.checked_add(2))
        .ok_or(PrimitiveError::CountOverflow { kind: "vertex" })?;
    let stride = sectors
        .checked_add(1)
        .ok_or(PrimitiveError::CountOverflow { kind: "vertex" })?;
    let vertices = u64::from(ring_count)
        .checked_mul(u64::from(stride))
        .ok_or(PrimitiveError::CountOverflow { kind: "vertex" })?;
    let indices = u64::from(ring_count - 1)
        .checked_mul(u64::from(sectors))
        .and_then(|count| count.checked_mul(6))
        .ok_or(PrimitiveError::CountOverflow { kind: "index" })?;
    let (vertex_count, index_count) = generated_counts(vertices, indices)?;
    let mut positions = reserve("vertex", vertex_count)?;
    let mut normals = reserve("vertex", vertex_count)?;
    let mut uvs = reserve("vertex", vertex_count)?;
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
    let mut indices = reserve("index", index_count)?;
    for ring in 0..ring_count - 1 {
        for sector in 0..sectors {
            let a = ring * stride + sector;
            let b = a + stride;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    build_mesh(positions, normals, uvs, indices)
}
