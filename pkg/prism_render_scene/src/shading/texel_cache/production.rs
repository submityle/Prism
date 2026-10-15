//! Render-world input seam for the production texel-space shade writer.
//!
//! Explicit [`ShadeDispatchInputs`] remains the compatibility API. Visibility
//! feedback can also be lowered into the same input shape by the production
//! resource, so the page table, chart metadata, writer jobs, and resolve
//! consumer share one validated path.

use std::collections::BTreeMap;

use bevy_ecs::prelude::*;

use crate::{
    geometry::{RenderShadingGeometryRegistry, RenderShadingVertex},
    lighting::ExtractedLights,
};

use super::{
    build_face_chart, FaceChart, GpuShadeChartInfo, GpuShadeResolvePage, ShadeJob, ShadeJobError,
    ShadePageKey, ShadePageTableError, ShadePageTableRecord, ShadePageTableSnapshot,
};

/// One explicit page/job request supplied by the real shade-page producer.
///
/// `geometry_index` is a retained render-world geometry handle, `page_key`
/// identifies an already allocated shade page, and `directional_light` indexes
/// the directional-light rows extracted from the main world.  None of these
/// fields is synthesized by the dispatch integration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadeDispatchJobInput {
    /// Stable render-world geometry slot containing the face.
    pub geometry_index: u32,
    /// Validated face-local chart for the requested triangle.
    pub chart: FaceChart,
    /// Page-table identity that receives this job's output.
    pub page_key: ShadePageKey,
    /// Index into [`ExtractedLights::directionals`].
    pub directional_light: u32,
}

/// Explicit per-frame inputs for the production shade writer.
///
/// The default value is intentionally empty.  The input is copied from the
/// main world during extraction, so an object/material/page producer can own
/// page allocation there without making the render node inspect incomplete
/// scene state.  The page records must describe published shade pages; this
/// integration does not publish them merely because a compute pass was
/// recorded.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct ShadeDispatchInputs {
    /// Material/light epoch shared by every page record in this frame.
    pub epoch: u64,
    /// Logical shade page size in texels. This is explicit producer metadata;
    /// the resolve consumer never infers it from atlas coordinates.
    pub page_size: u32,
    /// Published page records used to construct the indexed GPU page table.
    pub pages: Vec<ShadePageTableRecord>,
    /// Face/chart/light requests to turn into GPU jobs.
    pub jobs: Vec<ShadeDispatchJobInput>,
}

impl ShadeDispatchInputs {
    /// Clears the frame input, returning the production path to no-input.
    pub fn clear(&mut self) {
        self.epoch = 0;
        self.page_size = 0;
        self.pages.clear();
        self.jobs.clear();
    }

    /// Whether this frame has no externally supplied production work.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty() && self.jobs.is_empty()
    }
}

/// CPU batch ready for the GPU [`super::ShadeDispatch`] resource.
#[derive(Debug)]
pub(crate) struct ShadeDispatchBatch {
    pub jobs: Vec<ShadeJob>,
    pub page_table: ShadePageTableSnapshot,
    pub resolve_pages: Vec<GpuShadeResolvePage>,
    pub chart_infos: Vec<GpuShadeChartInfo>,
    pub vertices: Vec<RenderShadingVertex>,
}

/// Failure to lower explicit production inputs against live render-world data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ShadeDispatchBuildError {
    /// No producer supplied a complete frame input.
    NoInput,
    /// The page table itself did not satisfy its publication invariants.
    PageTable(ShadePageTableError),
    /// A request names a page that is not in the supplied table.
    MissingPage,
    /// A published page has no explicit chart provenance in this frame.
    MissingChartProvenance,
    /// Two jobs publish conflicting chart provenance for the same page/chart.
    ConflictingChartProvenance,
    /// The page size is missing or cannot address a logical page.
    InvalidPageSize,
    /// A page and its chart name different faces.
    PageFaceMismatch,
    /// A request names a geometry slot that is not live in the render world.
    MissingGeometry,
    /// A request names a directional light that was not extracted this frame.
    MissingDirectionalLight,
    /// The packed vertex table cannot be addressed by the GPU ABI.
    VertexOffsetOverflow,
    /// The canonical geometry/chart/light validation rejected the request.
    Job(ShadeJobError),
}

impl From<ShadePageTableError> for ShadeDispatchBuildError {
    fn from(error: ShadePageTableError) -> Self {
        Self::PageTable(error)
    }
}

/// Builds a real GPU batch from the render-world geometry and light tables.
///
/// The packed vertex order matches [`crate::RenderShadingGeometryBuffers`]:
/// ascending geometry slot order, with each geometry's vertices appended in
/// source order.  Consequently `vertex_base` in every job is the same base
/// used by the resolve-side geometry buffers.
pub(crate) fn build_shade_dispatch_batch(
    inputs: &ShadeDispatchInputs,
    geometry: &RenderShadingGeometryRegistry,
    lights: &ExtractedLights,
) -> Result<ShadeDispatchBatch, ShadeDispatchBuildError> {
    build_shade_dispatch_batch_inner(inputs, geometry, lights, false)
}

/// Builds a batch from the automatic visibility producer. Every page still
/// validates its own key/handle epoch proof, but one feedback frame may carry
/// pages from more than one material revision.
pub(crate) fn build_shade_dispatch_batch_mixed_epochs(
    inputs: &ShadeDispatchInputs,
    geometry: &RenderShadingGeometryRegistry,
    lights: &ExtractedLights,
) -> Result<ShadeDispatchBatch, ShadeDispatchBuildError> {
    build_shade_dispatch_batch_inner(inputs, geometry, lights, true)
}

fn build_shade_dispatch_batch_inner(
    inputs: &ShadeDispatchInputs,
    geometry: &RenderShadingGeometryRegistry,
    lights: &ExtractedLights,
    allow_mixed_epochs: bool,
) -> Result<ShadeDispatchBatch, ShadeDispatchBuildError> {
    if inputs.jobs.is_empty() || inputs.pages.is_empty() {
        return Err(ShadeDispatchBuildError::NoInput);
    }
    if inputs.page_size == 0 {
        return Err(ShadeDispatchBuildError::InvalidPageSize);
    }

    let page_table = if allow_mixed_epochs {
        ShadePageTableSnapshot::try_from_records_mixed_epochs(inputs.pages.clone())?
    } else {
        ShadePageTableSnapshot::try_from_records(inputs.epoch, inputs.pages.clone())?
    };

    // Build the resolve metadata only from explicit job provenance. A page
    // without a chart is rejected; no UV, face, or atlas-derived guess is
    // allowed to become sampleable.
    let mut page_provenance = BTreeMap::<ShadePageKey, FaceChart>::new();
    for request in &inputs.jobs {
        if request.page_key.face != request.chart.face {
            return Err(ShadeDispatchBuildError::PageFaceMismatch);
        }
        if page_provenance
            .insert(request.page_key, request.chart)
            .is_some()
        {
            return Err(ShadeDispatchBuildError::ConflictingChartProvenance);
        }
    }
    for record in page_table.records() {
        if !page_provenance.contains_key(&record.key) {
            return Err(ShadeDispatchBuildError::MissingChartProvenance);
        }
    }

    let mut chart_by_identity =
        BTreeMap::<(u32, u32, u64, u16, super::ShadeChannel), FaceChart>::new();
    for record in page_table.records() {
        let chart = *page_provenance
            .get(&record.key)
            .ok_or(ShadeDispatchBuildError::MissingChartProvenance)?;
        let identity = (
            record.key.object_id,
            record.key.material_id,
            record.key.material_epoch,
            record.key.face,
            record.key.channel,
        );
        if let Some(previous) = chart_by_identity.insert(identity, chart)
            && previous != chart
        {
            return Err(ShadeDispatchBuildError::ConflictingChartProvenance);
        }
    }

    let resolve_pages = page_table
        .records()
        .iter()
        .map(|record| {
            let chart = *page_provenance
                .get(&record.key)
                .ok_or(ShadeDispatchBuildError::MissingChartProvenance)?;
            let page_index = page_table
                .page_index(record.key)
                .ok_or(ShadeDispatchBuildError::MissingPage)?;
            let entry = *page_table
                .gpu_entries()
                .get(
                    usize::try_from(page_index)
                        .map_err(|_| ShadeDispatchBuildError::MissingPage)?,
                )
                .ok_or(ShadeDispatchBuildError::MissingPage)?;
            Ok(GpuShadeResolvePage::from_entry_at_mip(
                entry,
                chart,
                inputs.page_size,
                record.key.channel as u8 as u32,
                record.key.mip,
            ))
        })
        .collect::<Result<Vec<_>, ShadeDispatchBuildError>>()?;

    let chart_infos = chart_by_identity
        .into_iter()
        .map(
            |((object_id, material_id, material_epoch, face, channel), chart)| GpuShadeChartInfo {
                object_id,
                material_id,
                material_epoch_low: material_epoch as u32,
                material_epoch_high: (material_epoch >> 32) as u32,
                face_channel: u32::from(face) | ((channel as u8 as u32) << 24),
                chart_extent: chart.texel_extent,
                page_size: inputs.page_size,
                uv_min: chart.uv_min,
                uv_max: chart.uv_max,
            },
        )
        .collect::<Vec<_>>();

    let mut vertices = Vec::new();
    let mut vertex_bases = BTreeMap::new();
    for entry in geometry.entries_for_upload() {
        let base = u32::try_from(vertices.len())
            .map_err(|_| ShadeDispatchBuildError::VertexOffsetOverflow)?;
        let next_len = vertices
            .len()
            .checked_add(entry.geometry.vertices.len())
            .ok_or(ShadeDispatchBuildError::VertexOffsetOverflow)?;
        u32::try_from(next_len).map_err(|_| ShadeDispatchBuildError::VertexOffsetOverflow)?;
        vertex_bases.insert(entry.handle.index, base);
        vertices.extend(entry.geometry.vertices.iter().copied());
    }

    let mut jobs = Vec::with_capacity(inputs.jobs.len());
    for request in &inputs.jobs {
        let page_index = page_table
            .page_index(request.page_key)
            .ok_or(ShadeDispatchBuildError::MissingPage)?;
        let page_entry = page_table
            .gpu_entries()
            .get(usize::try_from(page_index).map_err(|_| ShadeDispatchBuildError::MissingPage)?)
            .ok_or(ShadeDispatchBuildError::MissingPage)?;
        let geometry_entry = geometry
            .get(request.geometry_index)
            .ok_or(ShadeDispatchBuildError::MissingGeometry)?;
        let vertex_base = *vertex_bases
            .get(&request.geometry_index)
            .ok_or(ShadeDispatchBuildError::MissingGeometry)?;
        let light = lights
            .directionals
            .get(
                usize::try_from(request.directional_light)
                    .map_err(|_| ShadeDispatchBuildError::MissingDirectionalLight)?,
            )
            .ok_or(ShadeDispatchBuildError::MissingDirectionalLight)?;
        let generation = page_entry.generation;
        let epoch = page_entry.epoch;
        let light_radiance = [
            light.illuminance[0] * light.visibility,
            light.illuminance[1] * light.visibility,
            light.illuminance[2] * light.visibility,
        ];
        jobs.push(
            ShadeJob::from_page_with_vertex_base(
                &geometry_entry.geometry,
                request.chart,
                request.page_key.mip,
                request.page_key.page_x,
                request.page_key.page_y,
                inputs.page_size,
                vertex_base,
                [page_entry.atlas_x, page_entry.atlas_y],
                page_index,
                generation,
                epoch,
                light.direction_to_light,
                light_radiance,
            )
            .map_err(ShadeDispatchBuildError::Job)?,
        );
    }

    Ok(ShadeDispatchBatch {
        jobs,
        page_table,
        resolve_pages,
        chart_infos,
        vertices,
    })
}

/// Builds a chart from the canonical geometry before a caller submits a job
/// input.  This helper keeps chart creation on the same geometry contract as
/// the production batch builder.
pub fn build_shade_dispatch_chart(
    geometry: &crate::RenderShadingGeometry,
    face: u32,
    texels_per_uv: f32,
    max_extent: u32,
) -> Result<FaceChart, super::FaceChartError> {
    build_face_chart(geometry, face, texels_per_uv, max_extent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuDirectionalLight, RenderShadingGeometry, RenderShadingPrimitive, RenderShadingVertex,
    };
    use prism_render_architecture::abi::GenerationalHandle;

    fn geometry() -> RenderShadingGeometry {
        RenderShadingGeometry {
            vertices: vec![
                RenderShadingVertex {
                    uv: [0.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    ..Default::default()
                },
                RenderShadingVertex {
                    uv: [1.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    ..Default::default()
                },
                RenderShadingVertex {
                    uv: [0.0, 1.0],
                    normal: [0.0, 0.0, 1.0],
                    ..Default::default()
                },
            ],
            primitives: vec![RenderShadingPrimitive {
                indices: [0, 1, 2],
                flags: 0,
            }],
            flags: 0,
        }
    }

    fn inputs() -> ShadeDispatchInputs {
        let key = ShadePageKey {
            object_id: 11,
            material_id: 7,
            material_epoch: 3,
            face: 0,
            mip: 0,
            channel: super::super::ShadeChannel::LitDiffuse,
            page_x: 0,
            page_y: 0,
        };
        let page = ShadePageTableRecord {
            key,
            handle: super::super::ShadePageHandle {
                slot: 2,
                generation: 4,
                epoch: 3,
            },
            atlas_x: 0,
            atlas_y: 0,
        };
        let chart = build_face_chart(&geometry(), 0, 4.0, 64).unwrap();
        ShadeDispatchInputs {
            epoch: 3,
            page_size: 16,
            pages: vec![page],
            jobs: vec![ShadeDispatchJobInput {
                geometry_index: 5,
                chart,
                page_key: key,
                directional_light: 0,
            }],
        }
    }

    #[test]
    fn batch_uses_live_geometry_and_extracted_directional_light() {
        let mut registry = RenderShadingGeometryRegistry::default();
        registry.upsert(
            GenerationalHandle {
                index: 5,
                generation: 2,
            },
            1,
            geometry(),
        );
        let mut lights = ExtractedLights::default();
        lights.directionals.push(GpuDirectionalLight {
            direction_to_light: [0.0, 0.0, 1.0],
            visibility: 0.5,
            illuminance: [2.0, 4.0, 6.0],
            _padding: 0.0,
        });

        let batch = build_shade_dispatch_batch(&inputs(), &registry, &lights).unwrap();
        assert_eq!(batch.jobs.len(), 1);
        assert_eq!(batch.jobs[0].vertex_base, 0);
        assert_eq!(batch.jobs[0].extent, [4, 4]);
        assert_eq!(batch.jobs[0].uv_min, [0.0, 0.0]);
        assert_eq!(batch.jobs[0].uv_max, [1.0, 1.0]);
        assert_eq!(batch.jobs[0].light_radiance, [1.0, 2.0, 3.0]);
        assert_eq!(batch.page_table.gpu_entries()[0].generation, 4);
    }

    #[test]
    fn batch_rejects_a_published_page_without_chart_provenance() {
        let mut input = inputs();
        let mut second_key = input.pages[0].key;
        second_key.page_x = 1;
        input.pages.push(ShadePageTableRecord {
            key: second_key,
            handle: super::super::ShadePageHandle {
                slot: 3,
                generation: 4,
                epoch: 3,
            },
            atlas_x: 16,
            atlas_y: 0,
        });

        let registry = RenderShadingGeometryRegistry::default();
        let lights = ExtractedLights::default();
        assert!(matches!(
            build_shade_dispatch_batch(&input, &registry, &lights),
            Err(ShadeDispatchBuildError::MissingChartProvenance)
        ));
    }

    #[test]
    fn batch_rejects_conflicting_chart_provenance_for_one_page() {
        let mut input = inputs();
        let mut conflicting = input.jobs[0].chart;
        conflicting.uv_max[0] = 0.75;
        input.jobs.push(ShadeDispatchJobInput {
            geometry_index: input.jobs[0].geometry_index,
            chart: conflicting,
            page_key: input.jobs[0].page_key,
            directional_light: input.jobs[0].directional_light,
        });

        let registry = RenderShadingGeometryRegistry::default();
        let lights = ExtractedLights::default();
        assert!(matches!(
            build_shade_dispatch_batch(&input, &registry, &lights),
            Err(ShadeDispatchBuildError::ConflictingChartProvenance)
        ));
    }

    #[test]
    fn missing_real_light_is_disabled_at_the_input_boundary() {
        let mut registry = RenderShadingGeometryRegistry::default();
        registry.upsert(
            GenerationalHandle {
                index: 5,
                generation: 2,
            },
            1,
            geometry(),
        );
        assert!(matches!(
            build_shade_dispatch_batch(&inputs(), &registry, &ExtractedLights::default()),
            Err(ShadeDispatchBuildError::MissingDirectionalLight)
        ));
    }
}
