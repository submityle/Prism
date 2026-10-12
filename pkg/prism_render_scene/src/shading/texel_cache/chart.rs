//! Per-face object-space parameterization for shade-cache pages.
//!
//! A shade page is not addressed through the material VT UV domain.  Each
//! triangle receives an independent local chart, which is the smallest
//! executable form of the Ptex-style seam contract from §14.6.  The chart is
//! built from the same [`RenderShadingGeometry`] rows consumed by the shading
//! resolve, so it cannot silently use a different UV source.

use crate::RenderShadingGeometry;

const UV_DEGENERATE_EPSILON: f32 = 1.0e-12;

/// A validated, per-face UV chart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceChart {
    /// Stable face/primitive identity within the owning geometry.
    pub face: u16,
    /// The three source UVs in primitive winding order.
    pub uv: [[f32; 2]; 3],
    /// Lower corner of the chart's local UV bounds.
    pub uv_min: [f32; 2],
    /// Upper corner of the chart's local UV bounds.
    pub uv_max: [f32; 2],
    /// Number of texels in the independently allocated face tile.
    pub texel_extent: [u32; 2],
}

/// Failure to derive a safe face chart from render-world geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaceChartError {
    /// The requested primitive does not exist or references a missing vertex.
    InvalidFace,
    /// At least one UV is non-finite.
    NonFiniteUv,
    /// The UV triangle has no invertible affine parameterization.
    DegenerateUv,
    /// The requested texel density is not finite or positive.
    InvalidTexelDensity,
    /// The requested chart would exceed the configured extent.
    ExtentOverflow,
}

/// Builds a face-local chart from the canonical shading geometry table.
///
/// `texels_per_uv` is the desired resolution for one unit of local UV span.
/// The result is independently bounded by `max_extent` on each axis.  No
/// global UV packing or material-VT page-table identity is introduced here;
/// callers may allocate the resulting tile under the shade page's `face`
/// identity and add gutter texels separately.
pub fn build_face_chart(
    geometry: &RenderShadingGeometry,
    face: u32,
    texels_per_uv: f32,
    max_extent: u32,
) -> Result<FaceChart, FaceChartError> {
    if !texels_per_uv.is_finite() || texels_per_uv <= 0.0 {
        return Err(FaceChartError::InvalidTexelDensity);
    }
    if max_extent == 0 {
        return Err(FaceChartError::ExtentOverflow);
    }
    let primitive = geometry
        .primitives
        .get(usize::try_from(face).map_err(|_| FaceChartError::InvalidFace)?)
        .ok_or(FaceChartError::InvalidFace)?;
    let mut uv = [[0.0; 2]; 3];
    for (out, &index) in uv.iter_mut().zip(primitive.indices.iter()) {
        let vertex = geometry
            .vertices
            .get(usize::try_from(index).map_err(|_| FaceChartError::InvalidFace)?)
            .ok_or(FaceChartError::InvalidFace)?;
        *out = vertex.uv;
        if !is_finite_uv(*out) {
            return Err(FaceChartError::NonFiniteUv);
        }
    }
    let determinant = uv_determinant(uv);
    if !determinant.is_finite() || determinant.abs() <= UV_DEGENERATE_EPSILON {
        return Err(FaceChartError::DegenerateUv);
    }

    let uv_min = [
        uv.iter()
            .map(|value| value[0])
            .fold(f32::INFINITY, f32::min),
        uv.iter()
            .map(|value| value[1])
            .fold(f32::INFINITY, f32::min),
    ];
    let uv_max = [
        uv.iter()
            .map(|value| value[0])
            .fold(f32::NEG_INFINITY, f32::max),
        uv.iter()
            .map(|value| value[1])
            .fold(f32::NEG_INFINITY, f32::max),
    ];
    let mut texel_extent = [0; 2];
    for axis in 0..2 {
        let scaled = f64::from(uv_max[axis] - uv_min[axis]) * f64::from(texels_per_uv);
        if !scaled.is_finite() || scaled > f64::from(max_extent) {
            return Err(FaceChartError::ExtentOverflow);
        }
        let extent = scaled.ceil().max(1.0);
        texel_extent[axis] =
            u32::try_from(extent as u64).map_err(|_| FaceChartError::ExtentOverflow)?;
    }

    Ok(FaceChart {
        face: u16::try_from(face).map_err(|_| FaceChartError::InvalidFace)?,
        uv,
        uv_min,
        uv_max,
        texel_extent,
    })
}

impl FaceChart {
    /// Converts a UV in this face's affine domain to a local texel coordinate.
    ///
    /// Coordinates are clamped to the chart bounds.  This is the explicit
    /// per-face rule used for gutter generation; it never crosses into a
    /// neighbouring material-VT chart implicitly.
    #[must_use]
    pub fn texel_for_uv(&self, uv: [f32; 2]) -> Option<[u32; 2]> {
        if !is_finite_uv(uv) {
            return None;
        }
        let mut texel = [0; 2];
        for axis in 0..2 {
            let span = self.uv_max[axis] - self.uv_min[axis];
            if !span.is_finite() || span <= 0.0 {
                return None;
            }
            let normalized = ((uv[axis] - self.uv_min[axis]) / span).clamp(0.0, 1.0);
            let coordinate = normalized * self.texel_extent[axis] as f32;
            texel[axis] = (coordinate.floor() as u32).min(self.texel_extent[axis] - 1);
        }
        Some(texel)
    }

    /// Reconstructs the UV at barycentric coordinates in the source face.
    #[must_use]
    pub fn uv_at_barycentrics(&self, barycentrics: [f32; 3]) -> [f32; 2] {
        [
            self.uv[0][0] * barycentrics[0]
                + self.uv[1][0] * barycentrics[1]
                + self.uv[2][0] * barycentrics[2],
            self.uv[0][1] * barycentrics[0]
                + self.uv[1][1] * barycentrics[1]
                + self.uv[2][1] * barycentrics[2],
        ]
    }
}

fn is_finite_uv(uv: [f32; 2]) -> bool {
    uv[0].is_finite() && uv[1].is_finite()
}

fn uv_determinant(uv: [[f32; 2]; 3]) -> f32 {
    (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[1][1] - uv[0][1]) * (uv[2][0] - uv[0][0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RenderShadingPrimitive, RenderShadingVertex};

    fn geometry(uv: [[f32; 2]; 3]) -> RenderShadingGeometry {
        RenderShadingGeometry {
            vertices: uv
                .into_iter()
                .map(|uv| RenderShadingVertex {
                    uv,
                    ..Default::default()
                })
                .collect(),
            primitives: vec![RenderShadingPrimitive {
                indices: [0, 1, 2],
                flags: 0,
            }],
            flags: 0,
        }
    }

    #[test]
    fn face_chart_is_local_and_maps_triangle_uv_to_tile_edges() {
        let chart = build_face_chart(
            &geometry([[2.0, -1.0], [3.0, -1.0], [2.0, 1.0]]),
            0,
            4.0,
            64,
        )
        .unwrap();
        assert_eq!(chart.face, 0);
        assert_eq!(chart.texel_extent, [4, 8]);
        assert_eq!(chart.texel_for_uv([2.0, -1.0]), Some([0, 0]));
        assert_eq!(chart.texel_for_uv([3.0, 1.0]), Some([3, 7]));
        assert_eq!(chart.uv_at_barycentrics([0.25, 0.25, 0.5]), [2.25, 0.0]);
    }

    #[test]
    fn invalid_geometry_never_produces_a_chart() {
        assert_eq!(
            build_face_chart(&geometry([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), 0, 4.0, 64),
            Err(FaceChartError::DegenerateUv)
        );
        assert_eq!(
            build_face_chart(&geometry([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 2, 4.0, 64),
            Err(FaceChartError::InvalidFace)
        );
        assert_eq!(
            build_face_chart(&geometry([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 0, 4.0, 2),
            Err(FaceChartError::ExtentOverflow)
        );
        assert_eq!(
            build_face_chart(&geometry([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), 0, 1.0, 0),
            Err(FaceChartError::ExtentOverflow)
        );
    }

    #[test]
    fn non_finite_uv_is_rejected_before_extent_math() {
        assert_eq!(
            build_face_chart(
                &geometry([[0.0, 0.0], [f32::NAN, 0.0], [0.0, 1.0]]),
                0,
                4.0,
                64,
            ),
            Err(FaceChartError::NonFiniteUv)
        );
    }
}
