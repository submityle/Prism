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
    GpuMaterialTexture, Illumination, MaterialVtFormat, MaterialVtSourceSnapshot,
    TextureSemantic as MaterialTextureSemantic, NO_VIRTUAL_TEXTURE,
};
use prism_render_visibility::{
    barycentric_derivatives, interpolate_vec2, ClipVertex, GpuViewRecord, Viewport,
};

use crate::{
    geometry::{RenderShadingGeometry, RenderShadingGeometryRegistry},
    material::{runtime::RenderMaterialRegistry, vt_source::MaterialVtSourceRegistry},
    scene::RenderGpuScene,
    shading::{
        texel_cache::{
            build_shade_dispatch_chart, shade_page_request_from_visibility, ShadeDemandConfig,
            ShadeDemandRecord, ShadeDemandRegistry, ShadeVisibilitySample,
        },
        PrismShadingSettings, ViewVisibilityBuffer,
    },
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
const SHADE_PAGE_BYTE_COST: u64 = 70_000;
const SHADE_TEXELS_PER_UV: f32 = 128.0;
const SHADE_MAX_CHART_EXTENT: u32 = 1024;

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
    inner.state = Some(ReadbackState::Mapping(InFlightReadback {
        pending,
        receiver,
    }));
}

/// Consumes one complete multi-view feedback frame and writes only validated
/// source-backed page demands to the registry.
pub(crate) fn collect_material_vt_feedback(
    settings: Res<PrismShadingSettings>,
    readback: Res<MaterialVtFeedbackReadback>,
    mut demands: ResMut<MaterialVtDemandRegistry>,
    mut shade_demands: ResMut<ShadeDemandRegistry>,
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
        shade_demands.clear();
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
            shade_demands.clear();
            return;
        }
    }

    let pending = &in_flight.pending;
    let mapped = pending.buffer.slice(..).get_mapped_range().unwrap();
    let mut page_demands = Vec::new();
    let mut shade_page_requests = Vec::new();
    for capture in &pending.views {
        if !views
            .iter()
            .any(|view| view.retained_view_entity == capture.retained)
        {
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
            &mut shade_page_requests,
        );
    }
    drop(mapped);
    pending.buffer.unmap();

    if demands.replace(page_demands).is_err() {
        // A semantic conflict is an invalid producer frame, not permission to
        // keep or synthesize a partial demand set.
        demands.clear();
    }
    shade_demands.replace_records(shade_page_requests);
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
    shade_output: &mut Vec<ShadeDemandRecord>,
) {
    let row_words = usize::try_from(capture.row_pitch / 4).unwrap_or(0);
    let ids_words = bytemuck::cast_slice::<u8, u32>(&bytes[capture.ids_offset as usize..]);
    let metadata_words =
        bytemuck::cast_slice::<u8, u32>(&bytes[capture.metadata_offset as usize..]);
    for y in 0..capture.size.y as usize {
        for x in 0..capture.size.x as usize {
            let word = y
                .saturating_mul(row_words)
                .saturating_add(x.saturating_mul(4));
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

            let shade_channel = match material.illumination {
                Illumination::Lit => Some(crate::shading::texel_cache::ShadeChannel::LitDiffuse),
                Illumination::Stylized => {
                    Some(crate::shading::texel_cache::ShadeChannel::StylizedDiffuse)
                }
                Illumination::Unlit | Illumination::Custom => None,
            };
            if let Some(channel) = shade_channel
                && let Ok(chart) = build_shade_dispatch_chart(
                    &geometry.geometry,
                    ids[2],
                    SHADE_TEXELS_PER_UV,
                    SHADE_MAX_CHART_EXTENT,
                )
                && let Some(request) = shade_page_request_from_visibility(
                    ShadeVisibilitySample {
                        object_id: sample.object_id,
                        material_id: sample.material_id,
                        face_id: sample.face_id,
                        uv: sample.uv,
                        uv_ddx: sample.uv_ddx,
                        uv_ddy: sample.uv_ddy,
                        screen_importance: sample.screen_importance,
                    },
                    ShadeDemandConfig {
                        chart_extent: chart.texel_extent,
                        page_size: 128,
                        page_byte_cost: SHADE_PAGE_BYTE_COST,
                        frame,
                        channel,
                        max_mip: 8,
                    },
                    u64::from(material.revision),
                )
            {
                shade_output.push(ShadeDemandRecord {
                    request,
                    geometry_index: instance.geometry.index,
                    chart,
                });
            }

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
                if let Ok(demands) =
                    aggregate_material_vt_demands(&input, [sample], |_| None, frame)
                {
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
        value if value == MaterialTextureSemantic::Normal as u32 => {
            StreamingTextureSemantic::Normal
        }
        value if value == MaterialTextureSemantic::Occlusion as u32 => {
            StreamingTextureSemantic::RoughnessMetalAo
        }
        value if value == MaterialTextureSemantic::ClearCoat as u32 => {
            StreamingTextureSemantic::Color
        }
        value if value == MaterialTextureSemantic::ClearCoatRoughness as u32 => {
            StreamingTextureSemantic::RoughnessMetalAo
        }
        value if value == MaterialTextureSemantic::ClearCoatNormal as u32 => {
            StreamingTextureSemantic::Normal
        }
        value if value == MaterialTextureSemantic::FaceShadowSdf as u32 => {
            StreamingTextureSemantic::Mask
        }
        value if value == MaterialTextureSemantic::MeasuredBrdf as u32 => {
            StreamingTextureSemantic::Hdr
        }
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
    let bary_x = f32::from((metadata[3] & 0xffff) as u16) / 65_535.0;
    let bary_y = f32::from((metadata[3] >> 16) as u16) / 65_535.0;
    let weights = [bary_x, bary_y, (1.0 - bary_x - bary_y).max(0.0)];
    if !weights.into_iter().all(f32::is_finite) {
        return None;
    }

    // The raster stores perspective-correct barycentrics. Recover the pixel
    // position from those weights, then use the shared DAIS solver so both the
    // interpolated UV and its gradients follow the rasterizer's perspective
    // correction. Treating these weights as affine (the old implementation)
    // over- or under-requests pages on triangles spanning depth.
    let world = [
        transform_point(instance, a.position),
        transform_point(instance, b.position),
        transform_point(instance, c.position),
    ];
    let clip = [
        clip_vertex(view, world[0]),
        clip_vertex(view, world[1]),
        clip_vertex(view, world[2]),
    ];
    let screen = [
        screen_position(view, world[0])?,
        screen_position(view, world[1])?,
        screen_position(view, world[2])?,
    ];
    let affine_scale = [
        weights[0] * clip[0].position[3],
        weights[1] * clip[1].position[3],
        weights[2] * clip[2].position[3],
    ];
    let affine_sum = affine_scale.into_iter().sum::<f32>();
    if !affine_sum.is_finite() || affine_sum.abs() <= f32::EPSILON {
        return None;
    }
    let pixel = [
        (affine_scale[0] * screen[0][0]
            + affine_scale[1] * screen[1][0]
            + affine_scale[2] * screen[2][0])
            / affine_sum
            - view.viewport[0] as f32,
        (affine_scale[0] * screen[0][1]
            + affine_scale[1] * screen[1][1]
            + affine_scale[2] * screen[2][1])
            / affine_sum
            - view.viewport[1] as f32,
    ];
    let bary = barycentric_derivatives(
        &clip,
        Viewport::new(view.viewport[2], view.viewport[3]),
        pixel,
    )?;
    let uv = interpolate_vec2(&bary, [a.uv, b.uv, c.uv]);
    if !uv.value.into_iter().all(f32::is_finite)
        || !uv.ddx.into_iter().chain(uv.ddy).all(f32::is_finite)
    {
        return None;
    }
    let uv_ddx = uv.ddx;
    let uv_ddy = uv.ddy;

    Some(MaterialVtVisibilitySample {
        object_id: ids[0],
        face_id: ids[2],
        material_id: metadata[0],
        uv: uv.value,
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

fn clip_vertex(view: &GpuViewRecord, world: Vec3) -> ClipVertex {
    let clip = Mat4::from_cols_array_2d(&view.clip_from_world) * world.extend(1.0);
    ClipVertex::new([clip.x, clip.y, clip.z, clip.w])
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{RenderShadingPrimitive, RenderShadingVertex};
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{
        GeometryHandle, SceneMaterialHandle, SceneTransform,
    };

    fn instance() -> InstanceRecord {
        InstanceRecord {
            current_transform: SceneTransform::IDENTITY,
            geometry: GeometryHandle {
                index: 2,
                generation: 1,
            },
            material: SceneMaterialHandle {
                index: 7,
                generation: 1,
            },
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

    fn perspective_triangle(flags: u32) -> RenderShadingGeometry {
        RenderShadingGeometry {
            vertices: vec![
                RenderShadingVertex {
                    position: [-0.5, -0.5, 0.5],
                    uv: [0.0, 0.0],
                    flags,
                    ..Default::default()
                },
                RenderShadingVertex {
                    position: [0.5, -0.5, -0.5],
                    uv: [1.0, 0.0],
                    flags,
                    ..Default::default()
                },
                RenderShadingVertex {
                    position: [0.0, 0.5, -0.5],
                    uv: [0.0, 1.0],
                    flags,
                    ..Default::default()
                },
            ],
            primitives: vec![RenderShadingPrimitive {
                indices: [0, 1, 2],
                flags,
            }],
            flags,
        }
    }

    fn triangle(flags: u32) -> RenderShadingGeometry {
        RenderShadingGeometry {
            vertices: vec![
                RenderShadingVertex {
                    position: [0.0, 0.0, 0.0],
                    uv: [0.0, 0.0],
                    flags,
                    ..Default::default()
                },
                RenderShadingVertex {
                    position: [1.0, 0.0, 0.0],
                    uv: [1.0, 0.0],
                    flags,
                    ..Default::default()
                },
                RenderShadingVertex {
                    position: [0.0, 1.0, 0.0],
                    uv: [0.0, 1.0],
                    flags,
                    ..Default::default()
                },
            ],
            primitives: vec![RenderShadingPrimitive {
                indices: [0, 1, 2],
                flags,
            }],
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
    fn perspective_feedback_uses_perspective_correct_uv_and_gradients() {
        let metadata_bary = 16_384_u32 | (16_384_u32 << 16);
        let mut perspective_view = view();
        perspective_view.clip_from_world = Mat4::from_cols_array_2d(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ])
        .to_cols_array_2d();
        let geometry = perspective_triangle(0);
        let sample = decode_visibility_sample(
            &[3, 1, 0, 0],
            &[7, 1, 0, metadata_bary],
            &instance(),
            &geometry,
            &perspective_view,
        )
        .expect("perspective-visible sample should decode");

        let bary_x = f32::from(16_384_u16) / 65_535.0;
        let bary_y = f32::from(16_384_u16) / 65_535.0;
        let weights = [bary_x, bary_y, 1.0 - bary_x - bary_y];
        let expected_uv = [weights[1], weights[2]];
        assert!((sample.uv[0] - expected_uv[0]).abs() < 0.001);
        assert!((sample.uv[1] - expected_uv[1]).abs() < 0.001);

        let affine_gradient_x = 0.034_285_713_f32;
        assert!(
            (sample.uv_ddx[0] - affine_gradient_x).abs() > 0.005,
            "perspective-correct gradient {:?} unexpectedly matches affine gradient",
            sample.uv_ddx,
        );

        let clip = geometry.vertices[..3]
            .iter()
            .map(|vertex| {
                clip_vertex(
                    &perspective_view,
                    transform_point(&instance(), vertex.position),
                )
            })
            .collect::<Vec<_>>();
        let clip: [ClipVertex; 3] = clip.try_into().expect("triangle has three vertices");
        let screen = geometry.vertices[..3]
            .iter()
            .map(|vertex| {
                screen_position(
                    &perspective_view,
                    transform_point(&instance(), vertex.position),
                )
            })
            .collect::<Option<Vec<_>>>()
            .expect("triangle projects to the screen");
        let affine_weights = [
            weights[0] * clip[0].position[3],
            weights[1] * clip[1].position[3],
            weights[2] * clip[2].position[3],
        ];
        let denominator = affine_weights.into_iter().sum::<f32>();
        let pixel = [
            (affine_weights[0] * screen[0][0]
                + affine_weights[1] * screen[1][0]
                + affine_weights[2] * screen[2][0])
                / denominator,
            (affine_weights[0] * screen[0][1]
                + affine_weights[1] * screen[1][1]
                + affine_weights[2] * screen[2][1])
                / denominator,
        ];
        let uv_at = |pixel| {
            let bary = barycentric_derivatives(&clip, Viewport::new(100, 100), pixel)
                .expect("nondegenerate perspective triangle");
            interpolate_vec2(
                &bary,
                [
                    geometry.vertices[0].uv,
                    geometry.vertices[1].uv,
                    geometry.vertices[2].uv,
                ],
            )
            .value
        };
        let h = 0.01;
        let plus = uv_at([pixel[0] + h, pixel[1]]);
        let minus = uv_at([pixel[0] - h, pixel[1]]);
        let finite_difference = [
            (plus[0] - minus[0]) / (2.0 * h),
            (plus[1] - minus[1]) / (2.0 * h),
        ];
        for (actual, expected) in sample.uv_ddx.into_iter().zip(finite_difference) {
            assert!((actual - expected).abs() < 0.001);
        }
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
