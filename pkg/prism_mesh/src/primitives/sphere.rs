use super::common::{build_mesh, generated_counts, positive, reserve, subdivisions};
use super::PrimitiveError;

/// Builds a UV sphere with duplicated seam vertices and outward normals.
#[must_use = "the generated mesh must be retained"]
pub fn uv_sphere(radius: f32, sectors: u32, rings: u32) -> Result<crate::Mesh, PrimitiveError> {
    let radius = positive("radius", radius)?;
    let sectors = subdivisions("sectors", sectors, 3)?;
    let rings = subdivisions("rings", rings, 2)?;
    let stride = sectors
        .checked_add(1)
        .ok_or(PrimitiveError::CountOverflow { kind: "vertex" })?;
    let ring_count = rings
        .checked_add(1)
        .ok_or(PrimitiveError::CountOverflow { kind: "vertex" })?;
    let vertices = u64::from(ring_count)
        .checked_mul(u64::from(stride))
        .ok_or(PrimitiveError::CountOverflow { kind: "vertex" })?;
    let indices = u64::from(rings)
        .checked_mul(u64::from(sectors))
        .and_then(|count| count.checked_mul(6))
        .ok_or(PrimitiveError::CountOverflow { kind: "index" })?;
    let (vertex_count, index_count) = generated_counts(vertices, indices)?;
    let mut positions = reserve("vertex", vertex_count)?;
    let mut normals = reserve("vertex", vertex_count)?;
    let mut uvs = reserve("vertex", vertex_count)?;
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let phi = core::f32::consts::PI * v;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for sector in 0..=sectors {
            let u = sector as f32 / sectors as f32;
            let theta = core::f32::consts::TAU * u;
            let normal = [sin_phi * theta.cos(), cos_phi, sin_phi * theta.sin()];
            positions.push([normal[0] * radius, normal[1] * radius, normal[2] * radius]);
            normals.push(normal);
            uvs.push([u, v]);
        }
    }
    let mut indices = reserve("index", index_count)?;
    for ring in 0..rings {
        for sector in 0..sectors {
            let a = ring * stride + sector;
            let b = a + stride;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    build_mesh(positions, normals, uvs, indices)
}
