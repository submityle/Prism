//! Visibility-raster to material-VT feedback bridge.
//!
//! The producer in this module reads the same integer attachments that the
//! visibility raster writes.  It copies those attachments into a mappable
//! buffer, starts the map only after submission, and consumes the result on a
//! later frame.  No CPU visibility helper, full-screen policy, or synthetic
//! sample is used as a replacement for a missing raster producer.

use std::sync::{mpsc, mpsc::Receiver, Mutex};

use bevy_ecs::prelude::*;
use bevy_math::{Mat4, Vec3};
use bevy_render::{
    render_resource::{
        Buffer, BufferAsyncError, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
        Extent3d, MapMode, Origin3d, TexelCopyBufferInfo, TexelCopyBufferLayout,
        TexelCopyTextureInfo, TextureAspect,
    },
    renderer::{PendingCommandBuffers, RenderDevice},
    view::{ExtractedView, RetainedViewEntity},
};
use prism_render_architecture::gpu_scene::InstanceRecord;
use prism_render_architecture::texture_streaming::TextureSemantic as StreamingTextureSemantic;
use prism_render_material::{
    GpuMaterialTexture, MaterialVtFormat, MaterialVtSourceSnapshot, NO_VIRTUAL_TEXTURE,
    TextureSemantic as MaterialTextureSemantic,
};
use prism_render_visibility::GpuViewRecord;

use crate::{
    geometry::{RenderShadingGeometry, RenderShadingGeometryRegistry},
    material::{
        runtime::RenderMaterialRegistry,
        vt_source::MaterialVtSourceRegistry,
    },
    scene::RenderGpuScene,
    shading::{PrismShadingSettings, ViewVisibilityBuffer},
    visibility::runtime::UnifiedVisibilityState,
};

use super::{
    aggregate_material_vt_demands, MaterialVtDemandMaterial, MaterialVtDemandRegistry,
    MaterialVtVisibilitySample,
};

const VISIBILITY_TEXEL_BYTES: u64 = 16;
const COPY_ROW_ALIGNMENT: u64 = 256;
/// Matches the bounded material-VT producer's physical page budget.
const MATERIAL_VT_PAGE_BYTE_COST: u64 = 70_000;

#[derive(Resource, Default)]
pub(crate) struct MaterialVtFeedbackReadback {
    inner: Mutex<ReadbackInner>,
}

#[derive(Default)]
struct ReadbackInner {
    state: Option<ReadbackState>,
    next_frame: u64,
}

enum ReadbackState {
    CopyQueued(PendingReadback),
    Mapping(InFlightReadback),
}

struct PendingReadback {
    buffer: Buffer,
    views: Vec<PendingView>,
    frame: u64,
}

struct PendingView {
    retained: RetainedViewEntity,
    ids_offset: u64,
    metadata_offset: u64,
    row_pitch: u64,
    size: bevy_math::UVec2,
    view: GpuViewRecord,
}

struct InFlightReadback {
    pending: PendingReadback,
    receiver: Receiver<Result<(), BufferAsyncError>>,
}

/// Copies the actual visibility-raster IDs and barycentric metadata.
pub(crate) fn request_material_vt_feedback_readback(
    settings: Res<PrismShadingSettings>,
    state: Res<UnifiedVisibilityState>,
    device: Res<RenderDevice>,
    mut pending: ResMut<PendingCommandBuffers>,
    readback: Res<MaterialVtFeedbackReadback>,
    views: Query<(&ExtractedView, &ViewVisibilityBuffer)>,
) {
    if !settings.enable_visibility_buffer {
        return;
    }

    let mut inner = readback.inner.lock().unwrap();
    if inner.state.is_some() {
        return;
    }

    let mut offset = 0_u64;
    let mut captures = Vec::new();
    for (extracted, visibility) in &views {
        let Some(view) = state
            .views
            .iter()
            .find(|view| state.retained_view(view.handle) == Some(extracted.retained_view_entity))
            .copied()
        else {
            continue;
        };
        let (_, _, size) = visibility.feedback_sources();
        if size.x == 0 || size.y == 0 {
            continue;
        }
        let row_pitch = aligned_row_pitch(size.x);
        let bytes = row_pitch.saturating_mul(u64::from(size.y));
        let ids_offset = offset;
        let metadata_offset = ids_offset.saturating_add(bytes);
        offset = metadata_offset.saturating_add(bytes);
        captures.push(PendingView {
            retained: extracted.retained_view_entity,
            ids_offset,
            metadata_offset,
            row_pitch,
            size,
            view,
        });
    }
    if captures.is_empty() {
        return;
    }

    let target = device.create_buffer(&BufferDescriptor {
        label: Some("prism material VT visibility feedback readback"),
        size: offset.max(COPY_ROW_ALIGNMENT),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("prism material VT visibility feedback readback"),
    });
    for capture in &captures {
        let Some((_, visibility)) = views
            .iter()
            .find(|(view, _)| view.retained_view_entity == capture.retained)
        else {
            continue;
        };
        let (ids, metadata, size) = visibility.feedback_sources();
        debug_assert_eq!(size, capture.size);
        let extent = Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        };
        let layout = |offset| TexelCopyBufferLayout {
            offset,
            bytes_per_row: Some(u32::try_from(capture.row_pitch).unwrap_or(u32::MAX)),
            rows_per_image: Some(size.y),
        };
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: ids,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &target,
                layout: layout(capture.ids_offset),
            },
            extent,
        );
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: metadata,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &target,
                layout: layout(capture.metadata_offset),
            },
            extent,
        );
    }
    pending.push_encoder(encoder, "prism material VT visibility feedback readback");
    let frame = inner.next_frame.max(1);
    inner.next_frame = frame.saturating_add(1);
    inner.state = Some(ReadbackState::CopyQueued(PendingReadback {
        buffer: target,
        views: captures,
        frame,
    }));
}

/// Starts mapping only after the render graph has submitted the copy.
pub(crate) fn map_submitted_material_vt_feedback(
    device: Res<RenderDevice>,
    readback: Res<MaterialVtFeedbackReadback>,
) {
    let mut inner = readback.inner.lock().unwrap();
    let Some(ReadbackState::CopyQueued(pending)) = inner.state.take() else {
        return;
    };
    let (sender, receiver) = mpsc::sync_channel(1);
    device.map_buffer(&pending.buffer.slice(..), MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    inner.state = Some(ReadbackState::Mapping(InFlightReadback { pending, receiver }));
}

/// Consumes one complete multi-view feedback frame and writes only validated
/// source-backed page demands to the registry.
pub(crate) fn collect_material_vt_feedback(
    settings: Res<PrismShadingSettings>,
    readback: Res<MaterialVtFeedbackReadback>,
    mut demands: ResMut<MaterialVtDemandRegistry>,
    scene: Res<RenderGpuScene>,
    geometries: Res<RenderShadingGeometryRegistry>,
    materials: Res<RenderMaterialRegistry>,
    sources: Res<MaterialVtSourceRegistry>,
    views: Query<&ExtractedView>,
) {
    let mut inner = readback.inner.lock().unwrap();
    if !settings.enable_visibility_buffer {
        inner.state = None;
        demands.clear();
        return;
    }
    let Some(ReadbackState::Mapping(in_flight)) = inner.state.take() else {
        return;
    };
    match in_flight.receiver.try_recv() {
        Ok(Ok(())) => {}
        Err(mpsc::TryRecvError::Empty) => {
            inner.state = Some(ReadbackState::Mapping(in_flight));
            return;
        }
        Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => {
            demands.clear();
            return;
        }
    }

    let pending = &in_flight.pending;
    let mapped = pending.buffer.slice(..).get_mapped_range().unwrap();
    let mut page_demands = Vec::new();
    for capture in &pending.views {
        if !views.iter().any(|view| view.retained_view_entity == capture.retained) {
            continue;
        }
        decode_view_feedback(
            &mapped,
            capture,
            pending.frame,
            &scene,
            &geometries,
            &materials,
            &sources,
            &mut page_demands,
        );
    }
    drop(mapped);
    pending.buffer.unmap();

    if demands.replace(page_demands).is_err() {
        // A semantic conflict is an invalid producer frame, not permission to
        // keep or synthesize a partial demand set.
        demands.clear();
    }
}

fn decode_view_feedback(
    bytes: &[u8],
    capture: &PendingView,
    frame: u64,
    scene: &RenderGpuScene,
    geometries: &RenderShadingGeometryRegistry,
    materials: &RenderMaterialRegistry,
    sources: &MaterialVtSourceRegistry,
    output: &mut Vec<prism_render_architecture::texture_streaming::PageDemand>,
) {
    let row_words = usize::try_from(capture.row_pitch / 4).unwrap_or(0);
    let ids_words = bytemuck::cast_slice::<u8, u32>(&bytes[capture.ids_offset as usize..]);
    let metadata_words = bytemuck::cast_slice::<u8, u32>(&bytes[capture.metadata_offset as usize..]);
    for y in 0..capture.size.y as usize {
        for x in 0..capture.size.x as usize {
            let word = y.saturating_mul(row_words).saturating_add(x.saturating_mul(4));
            let Some(ids) = ids_words.get(word..word.saturating_add(4)) else {
                continue;
            };
            let Some(metadata) = metadata_words.get(word..word.saturating_add(4)) else {
                continue;
            };
            if ids[0] == u32::MAX || metadata[0] == u32::MAX {
                continue;
            }
            let Some((generation, Some(instance))) = scene.mirror().record_at(ids[0]) else {
                continue;
            };
            if generation != ids[1] {
                continue;
            }
            let Some(geometry) = geometries.get(instance.geometry.index) else {
                continue;
            };
            if geometry.handle.generation != instance.geometry.generation {
                continue;
            }
            let Some(material) = materials.registry.record_at(metadata[0]) else {
                continue;
            };
            if material.handle.generation != metadata[1] || material.handle != instance.material {
                continue;
            }
            let Some(sample) = decode_visibility_sample(
                ids,
                metadata,
                instance,
                &geometry.geometry,
                &capture.view,
            ) else {
                continue;
            };

            for texture in &material.textures {
                let Some(semantic) = texture_semantic(texture.semantic) else {
                    continue;
                };
                if texture.virtual_texture_id == NO_VIRTUAL_TEXTURE {
                    continue;
                }
                let Some(source) = source_for_texture(sources, texture) else {
                    continue;
                };
                let input = MaterialVtDemandMaterial {
                    material_id: material.handle.index,
                    texture_id: texture.virtual_texture_id,
                    semantic,
                    source,
                    layer: 0,
                    page_byte_cost: MATERIAL_VT_PAGE_BYTE_COST,
                };
                if let Ok(demands) = aggregate_material_vt_demands(
                    &input,
                    [sample],
                    |_| None,
                    frame,
                ) {
                    output.extend(demands);
                }
            }
        }
    }
}

fn source_for_texture<'a>(
    sources: &'a MaterialVtSourceRegistry,
    texture: &GpuMaterialTexture,
) -> Option<&'a MaterialVtSourceSnapshot> {
    sources.entries().find_map(|entry| {
        let snapshot = entry.state().snapshot()?;
        let descriptor = snapshot.descriptor();
        (descriptor.texture_id() == texture.virtual_texture_id
            && descriptor.width() == texture.virtual_texture_width
            && descriptor.height() == texture.virtual_texture_height
            && descriptor.page_size() == texture.virtual_texture_page_size
            && descriptor.format() == MaterialVtFormat::Rgba8Unorm)
            .then_some(snapshot)
    })
}

fn texture_semantic(value: u32) -> Option<StreamingTextureSemantic> {
    Some(match value {
        value if value == MaterialTextureSemantic::BaseColor as u32 => {
            StreamingTextureSemantic::Color
        }
        value if value == MaterialTextureSemantic::Emissive as u32 => StreamingTextureSemantic::Hdr,
        value if value == MaterialTextureSemantic::MetallicRoughness as u32 => {
            StreamingTextureSemantic::RoughnessMetalAo
        }
        value if value == MaterialTextureSemantic::Normal as u32 => StreamingTextureSemantic::Normal,
        value if value == MaterialTextureSemantic::Occlusion as u32 => {
            StreamingTextureSemantic::RoughnessMetalAo
        }
        value if value == MaterialTextureSemantic::ClearCoat as u32 => StreamingTextureSemantic::Color,
        value if value == MaterialTextureSemantic::ClearCoatRoughness as u32 => {
            StreamingTextureSemantic::RoughnessMetalAo
        }
        value if value == MaterialTextureSemantic::ClearCoatNormal as u32 => {
            StreamingTextureSemantic::Normal
        }
        value if value == MaterialTextureSemantic::FaceShadowSdf as u32 => StreamingTextureSemantic::Mask,
        value if value == MaterialTextureSemantic::MeasuredBrdf as u32 => StreamingTextureSemantic::Hdr,
        value if value == MaterialTextureSemantic::IlluminationMask as u32 => {
            StreamingTextureSemantic::Mask
        }
        _ => return None,
    })
}

fn decode_visibility_sample(
    ids: &[u32],
    metadata: &[u32],
    instance: &InstanceRecord,
    geometry: &RenderShadingGeometry,
    view: &GpuViewRecord,
) -> Option<MaterialVtVisibilitySample> {
    if geometry.flags & crate::SHADING_GEOMETRY_FLAG_MISSING_UV != 0 {
        return None;
    }
    let primitive = *geometry.primitives.get(ids[2] as usize)?;
    let a = *geometry.vertices.get(primitive.indices[0] as usize)?;
    let b = *geometry.vertices.get(primitive.indices[1] as usize)?;
    let c = *geometry.vertices.get(primitive.indices[2] as usize)?;
    let bary_x = (metadata[3] & 0xffff) as f32 / 65_535.0;
    let bary_y = (metadata[3] >> 16) as f32 / 65_535.0;
    let weights = [bary_x, bary_y, (1.0 - bary_x - bary_y).max(0.0)];
    let uv = add3(add2(scale2(a.uv, weights[0]), scale2(b.uv, weights[1])), scale2(c.uv, weights[2]));
    if !uv.into_iter().all(f32::is_finite) {
        return None;
    }

    let pa = screen_position(view, transform_point(instance, a.position))?;
    let pb = screen_position(view, transform_point(instance, b.position))?;
    let pc = screen_position(view, transform_point(instance, c.position))?;
    let determinant = (pb[0] - pa[0]) * (pc[1] - pa[1])
        - (pb[1] - pa[1]) * (pc[0] - pa[0]);
    if !determinant.is_finite() || determinant.abs() <= f32::EPSILON {
        return None;
    }
    let dl1dx = (pc[1] - pa[1]) / determinant;
    let dl1dy = -(pc[0] - pa[0]) / determinant;
    let dl2dx = -(pb[1] - pa[1]) / determinant;
    let dl2dy = (pb[0] - pa[0]) / determinant;
    let uv_b = sub2(b.uv, a.uv);
    let uv_c = sub2(c.uv, a.uv);
    let uv_ddx = add2(scale2(uv_b, dl1dx), scale2(uv_c, dl2dx));
    let uv_ddy = add2(scale2(uv_b, dl1dy), scale2(uv_c, dl2dy));
    if !uv_ddx.into_iter().chain(uv_ddy).all(f32::is_finite) {
        return None;
    }

    Some(MaterialVtVisibilitySample {
        object_id: ids[0],
        face_id: ids[2],
        material_id: metadata[0],
        uv,
        uv_ddx,
        uv_ddy,
        // One unit is one actual covered raster pixel.  It is intentionally
        // not a guessed full-screen weight; duplicate page requests are
        // merged by maximum importance in the typed aggregator.
        screen_importance: 1,
    })
}

fn transform_point(instance: &InstanceRecord, point: [f32; 3]) -> Vec3 {
    let rows = instance.current_transform.rows;
    Vec3::new(
        rows[0][0] * point[0] + rows[0][1] * point[1] + rows[0][2] * point[2] + rows[0][3],
        rows[1][0] * point[0] + rows[1][1] * point[1] + rows[1][2] * point[2] + rows[1][3],
        rows[2][0] * point[0] + rows[2][1] * point[1] + rows[2][2] * point[2] + rows[2][3],
    )
}

fn screen_position(view: &GpuViewRecord, world: Vec3) -> Option<[f32; 2]> {
    let clip = Mat4::from_cols_array_2d(&view.clip_from_world) * world.extend(1.0);
    if !clip.w.is_finite() || clip.w.abs() <= f32::EPSILON {
        return None;
    }
    let ndc = Vec3::new(clip.x, clip.y, clip.z) / clip.w;
    let viewport = view.viewport;
    let result = [
        viewport[0] as f32 + (ndc.x * 0.5 + 0.5) * viewport[2] as f32,
        viewport[1] as f32 + (0.5 - ndc.y * 0.5) * viewport[3] as f32,
    ];
    result.into_iter().all(f32::is_finite).then_some(result)
}

fn aligned_row_pitch(width: u32) -> u64 {
    let bytes = u64::from(width).saturating_mul(VISIBILITY_TEXEL_BYTES);
    bytes.div_ceil(COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT
}

fn scale2(value: [f32; 2], scale: f32) -> [f32; 2] {
    [value[0] * scale, value[1] * scale]
}

fn add2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add3(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    add2(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{RenderShadingPrimitive, RenderShadingVertex};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{GeometryHandle, SceneMaterialHandle, SceneTransform};

    fn instance() -> InstanceRecord {
        InstanceRecord {
            current_transform: SceneTransform::IDENTITY,
            geometry: GeometryHandle { index: 2, generation: 1 },
            material: SceneMaterialHandle { index: 7, generation: 1 },
            ..Default::default()
        }
    }

    fn view() -> GpuViewRecord {
        GpuViewRecord {
            handle: GenerationalHandle {
                index: 0,
                generation: 0,
            },
            clip_from_world: Mat4::IDENTITY.to_cols_array_2d(),
            previous_clip_from_world: Mat4::IDENTITY.to_cols_array_2d(),
            world_position: [0.0, 0.0, 0.0],
            lod_scale: 1.0,
            viewport: [0, 0, 100, 100],
            frustum_planes: [[0.0, 0.0, 0.0, 1.0]; 6],
            layer_mask: 1,
            flags: Default::default(),
            history_epoch: 0,
        }
    }

    fn triangle(flags: u32) -> RenderShadingGeometry {
        RenderShadingGeometry {
            vertices: vec![
                RenderShadingVertex { position: [0.0, 0.0, 0.0], uv: [0.0, 0.0], flags, ..Default::default() },
                RenderShadingVertex { position: [1.0, 0.0, 0.0], uv: [1.0, 0.0], flags, ..Default::default() },
                RenderShadingVertex { position: [0.0, 1.0, 0.0], uv: [0.0, 1.0], flags, ..Default::default() },
            ],
            primitives: vec![RenderShadingPrimitive { indices: [0, 1, 2], flags }],
            flags,
        }
    }

    #[test]
    fn raster_metadata_reconstructs_uv_and_screen_gradient() {
        let metadata_bary = (16_384_u32) | (16_384_u32 << 16);
        let sample = decode_visibility_sample(
            &[3, 1, 0, 0],
            &[7, 1, 0, metadata_bary],
            &instance(),
            &triangle(0),
            &view(),
        )
        .expect("valid raster metadata should decode");
        assert!((sample.uv[0] - 0.25).abs() < 0.001);
        assert!((sample.uv[1] - 0.5).abs() < 0.001);
        assert!((sample.uv_ddx[0] - 0.02).abs() < 0.001);
        assert!(sample.uv_ddx[1].abs() < 0.001);
        assert_eq!(sample.screen_importance, 1);
    }

    #[test]
    fn missing_uv_is_not_replaced_by_a_zero_uv_sample() {
        assert!(decode_visibility_sample(
            &[3, 1, 0, 0],
            &[7, 1, 0, 0],
            &instance(),
            &triangle(crate::SHADING_GEOMETRY_FLAG_MISSING_UV),
            &view(),
        )
        .is_none());
    }

    #[test]
    fn copy_row_pitch_is_webgpu_aligned() {
        assert_eq!(aligned_row_pitch(1), 256);
        assert_eq!(aligned_row_pitch(16), 256);
        assert_eq!(aligned_row_pitch(17), 512);
    }
}
