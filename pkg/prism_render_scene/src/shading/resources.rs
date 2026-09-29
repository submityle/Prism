use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        BindGroup, Buffer, BufferDescriptor, BufferUsages, TextureDescriptor, TextureDimension,
        TextureFormat, TextureUsages,
    },
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
    view::Msaa,
};
use prism_render_shading::{ShadingWorkItem, MAX_SHADING_CLASSES};

use super::classification_gpu::{GpuShadingDispatchArgs, GpuShadingPixelClass};

pub(crate) const VISIBILITY_ID_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;

/// Linear HDR radiance target written by the shading-resolve compute pass
/// (`shading_resolve.wesl`) and later sampled by the composite/tonemap step.
/// `Rgba16Float` matches the `texture_storage_2d<rgba16float, write>` binding
/// in the shader and keeps enough range/precision for pre-exposure radiance.
pub(crate) const SCENE_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Per-pixel screen-space motion (`cur_uv - prev_uv`) written by the resolve
/// pass and consumed by the SSR temporal accumulation (and, later, TAA).
/// `Rg16Float` matches the `texture_storage_2d<rg16float, write>` binding in
/// `shading_resolve.wesl`; two half-float channels are ample for a UV-space
/// delta that is almost always a small fraction of the frame.
pub(crate) const MOTION_VECTOR_FORMAT: TextureFormat = TextureFormat::Rg16Float;

#[derive(Component)]
pub(crate) struct ViewVisibilityBuffer {
    ids: CachedTexture,
    metadata: CachedTexture,
    /// Linear HDR radiance the shading-resolve pass stores into and the
    /// composite step samples from.
    scene_color: CachedTexture,
    /// SSR energy-conservation export: the IBL specular radiance the resolve
    /// folded into `scene_color` per pixel, so the SSR composite can subtract
    /// it before substituting the screen-space reflection.
    ssr_env_specular: CachedTexture,
    /// SSR energy-conservation export: the split-sum environment-BRDF weight
    /// `(f0 * dfg.x + dfg.y) * occlusion` the resolve scaled that specular by,
    /// reused by the composite to weight the SSR reflection identically.
    ssr_spec_weight: CachedTexture,
    /// Per-pixel screen-space motion vector (`cur_uv - prev_uv`) the resolve
    /// pass writes for every covered pixel; consumed by the SSR temporal
    /// reprojection to follow object + camera motion instead of ghosting.
    motion_vectors: CachedTexture,
    /// Screen-space GI export: the pre-albedo diffuse *irradiance* the resolve
    /// evaluated per pixel. The SSGI hemisphere gather blends missed rays
    /// toward this so indirect light degrades to the IBL/SH ambient instead of
    /// to black.
    ssgi_ambient: CachedTexture,
    /// Screen-space GI export: the Lambertian albedo (`base_color *
    /// (1 - metallic)`) the SSGI composite multiplies the gathered pre-albedo
    /// radiance by before blending it over the IBL diffuse under confidence.
    ssgi_albedo: CachedTexture,
    pub(crate) size: bevy_math::UVec2,
}

impl ViewVisibilityBuffer {
    pub(crate) fn attachments(
        &self,
    ) -> (
        &bevy_render::render_resource::TextureView,
        &bevy_render::render_resource::TextureView,
    ) {
        (&self.ids.default_view, &self.metadata.default_view)
    }

    /// Storage/sampling view of the linear HDR radiance target.
    pub(crate) fn scene_color_view(&self) -> &bevy_render::render_resource::TextureView {
        &self.scene_color.default_view
    }

    /// The `scene_color` GPU texture itself, for a full-surface
    /// `copy_texture_to_texture` (an opt-in ping-pong post-process writes its
    /// result back in place here).
    pub(crate) fn scene_color_texture(&self) -> &bevy_render::render_resource::Texture {
        &self.scene_color.texture
    }

    /// Storage/sampling view of the SSR IBL-specular export.
    pub(crate) fn ssr_env_specular_view(&self) -> &bevy_render::render_resource::TextureView {
        &self.ssr_env_specular.default_view
    }

    /// Storage/sampling view of the SSR environment-BRDF weight export.
    pub(crate) fn ssr_spec_weight_view(&self) -> &bevy_render::render_resource::TextureView {
        &self.ssr_spec_weight.default_view
    }

    /// Storage/sampling view of the per-pixel motion-vector G-buffer.
    pub(crate) fn motion_vectors_view(&self) -> &bevy_render::render_resource::TextureView {
        &self.motion_vectors.default_view
    }

    /// Storage/sampling view of the SSGI pre-albedo ambient-irradiance export.
    pub(crate) fn ssgi_ambient_view(&self) -> &bevy_render::render_resource::TextureView {
        &self.ssgi_ambient.default_view
    }

    /// Storage/sampling view of the SSGI Lambertian-albedo export.
    pub(crate) fn ssgi_albedo_view(&self) -> &bevy_render::render_resource::TextureView {
        &self.ssgi_albedo.default_view
    }
}

#[derive(Component)]
pub(crate) struct ViewShadingBuffers {
    pub(crate) pixel_classes: Buffer,
    pub(crate) work_items: Buffer,
    pub(crate) class_counts: Buffer,
    pub(crate) class_offsets: Buffer,
    pub(crate) class_cursors: Buffer,
    pub(crate) dispatch_args: Buffer,
    pub(crate) diagnostics: Buffer,
    pub(crate) input_bind_group: Option<BindGroup>,
    pub(crate) output_bind_group: Option<BindGroup>,
    pub(crate) size: bevy_math::UVec2,
    pub(crate) capacity: u32,
}

impl ViewShadingBuffers {
    fn new(device: &RenderDevice, size: bevy_math::UVec2) -> Self {
        let capacity = size.x.saturating_mul(size.y).max(1);
        Self {
            pixel_classes: storage_buffer::<GpuShadingPixelClass>(
                device,
                "prism pixel classes",
                capacity,
                false,
            ),
            work_items: storage_buffer::<ShadingWorkItem>(
                device,
                "prism shading work",
                capacity,
                false,
            ),
            class_counts: storage_buffer::<u32>(
                device,
                "prism class counts",
                MAX_SHADING_CLASSES as u32,
                false,
            ),
            class_offsets: storage_buffer::<u32>(
                device,
                "prism class offsets",
                MAX_SHADING_CLASSES as u32,
                false,
            ),
            class_cursors: storage_buffer::<u32>(
                device,
                "prism class cursors",
                MAX_SHADING_CLASSES as u32,
                false,
            ),
            dispatch_args: storage_buffer::<GpuShadingDispatchArgs>(
                device,
                "prism shading dispatch",
                MAX_SHADING_CLASSES as u32,
                true,
            ),
            diagnostics: storage_buffer::<u32>(device, "prism classification diagnostics", 4, true),
            input_bind_group: None,
            output_bind_group: None,
            size,
            capacity,
        }
    }

    pub(crate) fn clear_transient_state(
        &self,
        encoder: &mut bevy_render::render_resource::CommandEncoder,
    ) {
        encoder.clear_buffer(&self.class_counts, 0, None);
        encoder.clear_buffer(&self.class_offsets, 0, None);
        encoder.clear_buffer(&self.class_cursors, 0, None);
        encoder.clear_buffer(&self.dispatch_args, 0, None);
        encoder.clear_buffer(&self.diagnostics, 0, None);
    }
}

fn storage_buffer<T>(
    device: &RenderDevice,
    label: &'static str,
    elements: u32,
    indirect: bool,
) -> Buffer {
    let mut usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    if indirect {
        usage |= BufferUsages::INDIRECT | BufferUsages::COPY_SRC;
    }
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: (elements.max(1) as u64).saturating_mul(size_of::<T>() as u64),
        usage,
        mapped_at_creation: false,
    })
}

pub(crate) fn prepare_visibility_buffers(
    mut commands: Commands,
    settings: Res<super::runtime::PrismShadingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
    )>,
) {
    for (entity, camera, msaa, existing) in &views {
        if !settings.enable_visibility_buffer || msaa.is_some_and(|value| value.samples() != 1) {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewVisibilityBuffer>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|buffer| buffer.size == size) {
            continue;
        }
        let ids = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism visibility IDs"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: VISIBILITY_ID_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
        );
        let metadata = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism visibility metadata"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: VISIBILITY_ID_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
        );
        let scene_color = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism scene color (HDR)"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                // STORAGE_BINDING: written by the resolve compute pass.
                // TEXTURE_BINDING: sampled by the later composite/tonemap step.
                // COPY_DST: an opt-in ping-pong post-process (motion blur)
                // copies its full-resolution result back over scene_color so the
                // downstream composite reads the processed image.
                usage: TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            },
        );
        // SSR energy-conservation exports written alongside `scene_color` by
        // the resolve pass. Allocated unconditionally so the resolve bind
        // group is always valid (the pass runs even when SSR is disabled); the
        // SSR composite reads them only when reflections are active.
        let ssr_env_specular = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSR env specular export"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let ssr_spec_weight = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSR spec weight export"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        // Motion-vector G-buffer, written every covered pixel by the resolve
        // pass. STORAGE_BINDING: written by the resolve compute pass;
        // TEXTURE_BINDING: sampled by the SSR temporal reprojection (and TAA).
        let motion_vectors = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism motion vectors"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: MOTION_VECTOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        // Screen-space GI exports written alongside `scene_color` by the
        // resolve pass. STORAGE_BINDING: written by the resolve compute pass;
        // TEXTURE_BINDING: sampled by the SSGI trace (ambient) and composite
        // (albedo). Allocated unconditionally so the resolve bind group is
        // always valid; the SSGI pass reads them only when GI is active.
        let ssgi_ambient = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSGI ambient export"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        let ssgi_albedo = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism SSGI albedo export"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SCENE_COLOR_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        commands.entity(entity).insert(ViewVisibilityBuffer {
            ids,
            metadata,
            scene_color,
            ssr_env_specular,
            ssr_spec_weight,
            motion_vectors,
            ssgi_ambient,
            ssgi_albedo,
            size,
        });
    }
}

pub(crate) fn prepare_shading_buffers(
    mut commands: Commands,
    settings: Res<super::runtime::PrismShadingSettings>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewShadingBuffers>,
    )>,
) {
    for (entity, camera, msaa, existing) in &views {
        if !settings.enable_visibility_buffer || msaa.is_some_and(|value| value.samples() != 1) {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewShadingBuffers>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewShadingBuffers>();
            }
            continue;
        };
        if existing.is_some_and(|buffers| buffers.size == size) {
            continue;
        }
        commands
            .entity(entity)
            .insert(ViewShadingBuffers::new(&device, size));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_shader::{Shader, ShaderCache, ShaderCacheSource};

    fn load_source(
        _: &(),
        source: ShaderCacheSource,
        _: &bevy_shader::ValidateShader,
    ) -> Result<String, bevy_shader::ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("visibility raster shader is WESL"),
        }
    }

    #[test]
    fn formats_cover_the_two_target_visibility_abi() {
        assert_eq!(VISIBILITY_ID_FORMAT, TextureFormat::Rgba32Uint);
        assert_eq!(VISIBILITY_ID_FORMAT.block_copy_size(None), Some(16));
        assert_eq!(2 * 16, size_of::<prism_render_shading::VisibilityPixel>());
    }

    #[test]
    fn visibility_raster_wesl_compiles_with_barycentric_contract() {
        let shader_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d56_4953_5241_5354_4552_0001),
        };
        let source = include_str!("../shaders/visibility_raster.wesl");
        let source = source[source.find("struct GpuSceneInstance").unwrap()..]
            .replace(
                "bevy_render::utils::decompress_vertex_position",
                "decompress_vertex_position",
            )
            .replace(
                "@builtin(barycentric) barycentrics: vec3<f32>",
                "@location(5) barycentrics: vec3<f32>",
            );
        let stubs = r#"
fn affine3_to_square(value: mat3x4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(value[0]), vec4<f32>(value[1]), vec4<f32>(value[2]),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}
fn position_world_to_clip(position: vec3<f32>) -> vec4<f32> { return vec4(position, 1.0); }
fn decompress_vertex_position(position: vec4<f32>, center: vec3<f32>, half_extents: vec3<f32>) -> vec3<f32> {
    return position.xyz;
}
"#;
        let mut cache = ShaderCache::new((), load_source);
        cache.set_shader(
            shader_id,
            Shader::from_wesl(
                format!("{stubs}{source}"),
                "shaders/prism_visibility_raster.wesl",
            ),
        );
        cache
            .get(0, shader_id, &[])
            .unwrap_or_else(|error| panic!("visibility raster shader failed: {error}"));
    }
}
