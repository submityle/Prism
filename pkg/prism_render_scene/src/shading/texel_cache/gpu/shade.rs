//! ABI and job construction for the real texture-space irradiance pass.
//!
//! The pass stores view-independent incoming irradiance, not a material
//! response.  Material detail remains a sampling-time concern as specified by
//! §14.3; the job therefore consumes canonical render geometry plus a light
//! and carries the shade page's generation/epoch proof.

use bytemuck::{Pod, Zeroable};

use crate::{RenderShadingGeometry, RenderShadingGeometryHeader};

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
    /// Three geometry-local indices from the primitive row.
    pub vertex_indices: [u32; 3],
    /// Absolute base of this geometry in the packed render-world vertex table.
    pub vertex_base: u32,
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
    /// The requested logical page does not intersect the chart at its mip.
    InvalidPage,
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
    /// Resolves the primitive-local indices against this job's packed vertex
    /// base. Dispatch validation and the shader use this same checked rule.
    #[must_use]
    pub fn global_vertex_indices(self) -> Option<[u32; 3]> {
        let [a, b, c] = self.vertex_indices;
        Some([
            self.vertex_base.checked_add(a)?,
            self.vertex_base.checked_add(b)?,
            self.vertex_base.checked_add(c)?,
        ])
    }

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
        Self::from_face_with_vertex_base(
            geometry,
            chart,
            0,
            atlas_origin,
            page_index,
            generation,
            epoch,
            light_direction,
            light_radiance,
        )
    }

    /// Builds a shade job whose primitive-local indices resolve through an
    /// absolute base in the packed render-world vertex table.
    pub fn from_face_with_vertex_base(
        geometry: &RenderShadingGeometry,
        chart: FaceChart,
        vertex_base: u32,
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
            vertex_base,
            uv_min: chart.uv_min,
            uv_max: chart.uv_max,
            light_direction,
            _light_direction_padding: 0.0,
            light_radiance,
            _light_radiance_padding: 0.0,
        })
    }

    /// Builds one page-local job from a chart and logical page coordinate.
    ///
    /// The writer ABI stores the UV rectangle covered by the job, not merely
    /// the chart's full extent. This is essential when a chart spans more
    /// than one page: every dispatch must shade only its own page rectangle,
    /// otherwise each page overwrites the same full chart into a different
    /// atlas allocation.
    pub fn from_page_with_vertex_base(
        geometry: &RenderShadingGeometry,
        chart: FaceChart,
        mip: u8,
        page_x: u16,
        page_y: u16,
        page_size: u32,
        vertex_base: u32,
        atlas_origin: [u32; 2],
        page_index: u32,
        generation: u32,
        epoch: u32,
        light_direction: [f32; 3],
        light_radiance: [f32; 3],
    ) -> Result<Self, ShadeJobError> {
        if page_size == 0 || mip >= 32 {
            return Err(ShadeJobError::InvalidPage);
        }

        let shift = u32::from(mip);
        let chart_width = chart.texel_extent[0].checked_shr(shift).unwrap_or(0).max(1);
        let chart_height = chart.texel_extent[1].checked_shr(shift).unwrap_or(0).max(1);
        let origin_x = u32::from(page_x)
            .checked_mul(page_size)
            .ok_or(ShadeJobError::InvalidPage)?;
        let origin_y = u32::from(page_y)
            .checked_mul(page_size)
            .ok_or(ShadeJobError::InvalidPage)?;
        if origin_x >= chart_width || origin_y >= chart_height {
            return Err(ShadeJobError::InvalidPage);
        }

        let extent = [
            page_size.min(chart_width - origin_x),
            page_size.min(chart_height - origin_y),
        ];
        let uv_span = [
            chart.uv_max[0] - chart.uv_min[0],
            chart.uv_max[1] - chart.uv_min[1],
        ];
        let page_uv_min = [
            chart.uv_min[0] + uv_span[0] * origin_x as f32 / chart_width as f32,
            chart.uv_min[1] + uv_span[1] * origin_y as f32 / chart_height as f32,
        ];
        let page_uv_max = [
            chart.uv_min[0] + uv_span[0] * (origin_x + extent[0]) as f32 / chart_width as f32,
            chart.uv_min[1] + uv_span[1] * (origin_y + extent[1]) as f32 / chart_height as f32,
        ];

        let mut job = Self::from_face_with_vertex_base(
            geometry,
            chart,
            vertex_base,
            atlas_origin,
            page_index,
            generation,
            epoch,
            light_direction,
            light_radiance,
        )?;
        job.extent = extent;
        job.uv_min = page_uv_min;
        job.uv_max = page_uv_max;
        Ok(job)
    }

    /// Builds a shade job directly from the production geometry header.
    /// Primitive indices stay local; only the header's packed vertex offset is
    /// copied into the explicit ABI field.
    pub fn from_face_with_header(
        geometry: &RenderShadingGeometry,
        header: RenderShadingGeometryHeader,
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
        if primitive
            .indices
            .into_iter()
            .any(|index| header.global_vertex_index(index).is_none())
        {
            return Err(ShadeJobError::InvalidFace);
        }
        Self::from_face_with_vertex_base(
            geometry,
            chart,
            header.vertex_offset,
            atlas_origin,
            page_index,
            generation,
            epoch,
            light_direction,
            light_radiance,
        )
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
        Self::from_face_with_density_and_vertex_base(
            geometry,
            face,
            texels_per_uv,
            max_extent,
            0,
            atlas_origin,
            page_index,
            generation,
            epoch,
            light_direction,
            light_radiance,
        )
    }

    /// Density-based variant for a geometry stored in packed global rows.
    pub fn from_face_with_density_and_vertex_base(
        geometry: &RenderShadingGeometry,
        face: u32,
        texels_per_uv: f32,
        max_extent: u32,
        vertex_base: u32,
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
        Self::from_face_with_vertex_base(
            geometry,
            chart,
            vertex_base,
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
    use core::mem::{offset_of, size_of};

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
        assert_eq!(offset_of!(ShadeJob, vertex_indices), 32);
        assert_eq!(offset_of!(ShadeJob, vertex_base), 44);
        assert_eq!(offset_of!(ShadeJob, uv_min), 48);
    }

    #[test]
    fn shade_job_keeps_primitive_indices_local_and_carries_the_packed_base() {
        let geometry = geometry();
        let chart = super::super::super::chart::build_face_chart(&geometry, 0, 4.0, 4).unwrap();
        let job = ShadeJob::from_face_with_header(
            &geometry,
            RenderShadingGeometryHeader {
                vertex_offset: 19,
                vertex_count: 3,
                ..Default::default()
            },
            chart,
            [0, 0],
            0,
            1,
            1,
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(job.vertex_indices, [0, 1, 2]);
        assert_eq!(job.vertex_base, 19);
        assert_eq!(job.global_vertex_indices(), Some([19, 20, 21]));
    }

    #[test]
    fn shade_job_rejects_a_header_that_cannot_cover_the_primitive() {
        let geometry = geometry();
        let chart = super::super::super::chart::build_face_chart(&geometry, 0, 4.0, 4).unwrap();
        assert_eq!(
            ShadeJob::from_face_with_header(
                &geometry,
                RenderShadingGeometryHeader {
                    vertex_offset: 19,
                    vertex_count: 2,
                    ..Default::default()
                },
                chart,
                [0, 0],
                0,
                1,
                1,
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ),
            Err(ShadeJobError::InvalidFace)
        );
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

    #[test]
    fn page_job_carries_only_its_local_extent_and_uv_rectangle() {
        let geometry = geometry();
        let chart = super::super::super::chart::build_face_chart(&geometry, 0, 4.0, 4).unwrap();
        let job = ShadeJob::from_page_with_vertex_base(
            &geometry,
            chart,
            0,
            1,
            0,
            2,
            0,
            [8, 16],
            3,
            1,
            1,
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(job.extent, [2, 2]);
        assert_eq!(job.atlas_origin, [8, 16]);
        assert_eq!(job.uv_min, [0.5, 0.0]);
        assert_eq!(job.uv_max, [1.0, 0.5]);
    }

    #[test]
    fn page_job_rejects_a_page_outside_the_chart_mip() {
        let geometry = geometry();
        let chart = super::super::super::chart::build_face_chart(&geometry, 0, 4.0, 4).unwrap();
        assert_eq!(
            ShadeJob::from_page_with_vertex_base(
                &geometry,
                chart,
                0,
                2,
                0,
                2,
                0,
                [0, 0],
                0,
                1,
                1,
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ),
            Err(ShadeJobError::InvalidPage)
        );
    }
}
