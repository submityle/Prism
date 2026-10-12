//! ABI and job construction for the real texture-space irradiance pass.
//!
//! The pass stores view-independent incoming irradiance, not a material
//! response.  Material detail remains a sampling-time concern as specified by
//! §14.3; the job therefore consumes canonical render geometry plus a light
//! and carries the shade page's generation/epoch proof.

use bytemuck::{Pod, Zeroable};

use crate::RenderShadingGeometry;

use super::super::chart::{FaceChart, FaceChartError};

/// A single directional-light texture-space shade job.
///
/// This is the storage-buffer layout consumed by `shade.wgsl`. The trailing
/// fields are deliberately explicit `vec4`-sized rows so the Rust and WGSL
/// layouts stay stable across backends.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ShadeJob {
    /// Number of output texels on the x/y axes.
    pub extent: [u32; 2],
    /// Atlas destination origin in texels.
    pub atlas_origin: [u32; 2],
    /// Indexed shade page-table entry used for the write proof.
    pub page_index: u32,
    /// Content generation expected by the page table.
    pub generation: u32,
    /// Material/light epoch expected by the page table.
    pub epoch: u32,
    pub _proof_padding: u32,
    /// Three indices into the canonical `RenderShadingVertex` table.
    pub vertex_indices: [u32; 3],
    pub _vertex_padding: u32,
    /// Local chart UV bounds.
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// Direction from the surface toward the directional light.
    pub light_direction: [f32; 3],
    pub _light_direction_padding: f32,
    /// RGB incident radiance supplied by the directional light.
    pub light_radiance: [f32; 3],
    pub _light_radiance_padding: f32,
}

/// Failure while converting canonical geometry and a validated face chart to
/// a GPU shade job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeJobError {
    /// The chart's face does not exist in the geometry.
    InvalidFace,
    /// A chart extent cannot be represented by the job ABI.
    InvalidExtent,
    /// The chart does not describe the UVs in the canonical geometry row.
    InvalidChart,
    /// The light contains a NaN or infinity.
    NonFiniteLight,
    /// A directional light must have a non-zero direction.
    InvalidLightDirection,
}

/// Failure while deriving a shade job from canonical geometry and density.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeJobFromDensityError {
    /// The per-face UV chart could not be built.
    Chart(FaceChartError),
    /// The validated chart or light could not become a GPU job.
    Job(ShadeJobError),
}

impl ShadeJob {
    /// Builds a real shade job from the same geometry rows used by resolve.
    pub fn from_face(
        geometry: &RenderShadingGeometry,
        chart: FaceChart,
        atlas_origin: [u32; 2],
        page_index: u32,
        generation: u32,
        epoch: u32,
        light_direction: [f32; 3],
        light_radiance: [f32; 3],
    ) -> Result<Self, ShadeJobError> {
        let primitive = geometry
            .primitives
            .get(usize::from(chart.face))
            .ok_or(ShadeJobError::InvalidFace)?;
        if chart.texel_extent.contains(&0) {
            return Err(ShadeJobError::InvalidExtent);
        }
        if chart
            .uv
            .into_iter()
            .flatten()
            .chain(chart.uv_min)
            .chain(chart.uv_max)
            .any(|value| !value.is_finite())
            || chart
                .uv_min
                .into_iter()
                .zip(chart.uv_max)
                .any(|(min, max)| min >= max)
        {
            return Err(ShadeJobError::InvalidChart);
        }
        let mut source_uv = [[0.0; 2]; 3];
        for (slot, &index) in source_uv.iter_mut().zip(primitive.indices.iter()) {
            *slot = geometry
                .vertices
                .get(usize::try_from(index).map_err(|_| ShadeJobError::InvalidFace)?)
                .ok_or(ShadeJobError::InvalidFace)?
                .uv;
        }
        if chart.uv != source_uv {
            return Err(ShadeJobError::InvalidChart);
        }
        if light_direction
            .into_iter()
            .chain(light_radiance)
            .any(|value| !value.is_finite())
        {
            return Err(ShadeJobError::NonFiniteLight);
        }
        let direction_length_squared = light_direction
            .into_iter()
            .map(|value| value * value)
            .sum::<f32>();
        if direction_length_squared <= 1.0e-12 {
            return Err(ShadeJobError::InvalidLightDirection);
        }
        Ok(Self {
            extent: chart.texel_extent,
            atlas_origin,
            page_index,
            generation,
            epoch,
            _proof_padding: 0,
            vertex_indices: primitive.indices,
            _vertex_padding: 0,
            uv_min: chart.uv_min,
            uv_max: chart.uv_max,
            light_direction,
            _light_direction_padding: 0.0,
            light_radiance,
            _light_radiance_padding: 0.0,
        })
    }

    /// Validates the independently derived chart before job construction.
    pub fn from_face_with_density(
        geometry: &RenderShadingGeometry,
        face: u32,
        texels_per_uv: f32,
        max_extent: u32,
        atlas_origin: [u32; 2],
        page_index: u32,
        generation: u32,
        epoch: u32,
        light_direction: [f32; 3],
        light_radiance: [f32; 3],
    ) -> Result<Self, ShadeJobFromDensityError> {
        let chart =
            super::super::chart::build_face_chart(geometry, face, texels_per_uv, max_extent)
                .map_err(ShadeJobFromDensityError::Chart)?;
        Self::from_face(
            geometry,
            chart,
            atlas_origin,
            page_index,
            generation,
            epoch,
            light_direction,
            light_radiance,
        )
        .map_err(ShadeJobFromDensityError::Job)
    }
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::*;
    use crate::{RenderShadingPrimitive, RenderShadingVertex};

    fn geometry() -> RenderShadingGeometry {
        let vertex = |uv: [f32; 2]| RenderShadingVertex {
            uv,
            normal: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        RenderShadingGeometry {
            vertices: vec![vertex([0.0, 0.0]), vertex([1.0, 0.0]), vertex([0.0, 1.0])],
            primitives: vec![RenderShadingPrimitive {
                indices: [0, 1, 2],
                flags: 0,
            }],
            flags: 0,
        }
    }

    #[test]
    fn shade_job_layout_is_the_explicit_wgsl_abi() {
        assert_eq!(size_of::<ShadeJob>(), 96);
    }

    #[test]
    fn shade_job_rejects_invalid_chart_and_light_inputs() {
        let geometry = geometry();
        let chart = super::super::super::chart::build_face_chart(&geometry, 0, 4.0, 4).unwrap();
        let mut mismatched = chart;
        mismatched.uv[1][0] = 0.5;
        assert_eq!(
            ShadeJob::from_face(
                &geometry,
                mismatched,
                [0, 0],
                0,
                1,
                1,
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ),
            Err(ShadeJobError::InvalidChart)
        );
        assert_eq!(
            ShadeJob::from_face(
                &geometry,
                chart,
                [0, 0],
                0,
                1,
                1,
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ),
            Err(ShadeJobError::InvalidLightDirection)
        );
        assert_eq!(
            ShadeJob::from_face_with_density(
                &geometry,
                0,
                4.0,
                4,
                [0, 0],
                0,
                1,
                1,
                [0.0, 0.0, 1.0],
                [f32::NAN, 1.0, 1.0],
            ),
            Err(ShadeJobFromDensityError::Job(ShadeJobError::NonFiniteLight))
        );
    }
}
