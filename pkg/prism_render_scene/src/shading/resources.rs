use bevy_ecs::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        BindGroup, Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, Texture,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    },
    renderer::RenderDevice,
    view::Msaa,
};
use prism_render_driver as rhi;
use prism_render_shading::ShadingWorkItem;

use crate::gpu::rhi::{SceneRhi, SceneRhiTexture, SceneRhiTextureView};

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

/// Native integer attachments retain their original shader ABI. The curated
/// RHI format set does not represent Rgba32Uint, so these are not RHI imports.
/// Like the RHI exports below, they are allocated once per resident view.
struct VisibilityIdTexture {
    default_view: TextureView,
    texture: Texture,
}

impl VisibilityIdTexture {
    fn new(device: &RenderDevice, size: bevy_math::UVec2, label: &'static str) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: VISIBILITY_ID_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self {
            default_view: texture.create_view(&Default::default()),
            texture,
        }
    }
}

/// Owns both table entries for the exact texture consumed by native passes.
/// No parallel/shadow allocation is made for the RHI graph. Native command
/// buffers retain their handles after view replacement; table IDs are valid
/// only while this owner remains resident on the view.
struct VisibilityTexture {
    default_view: SceneRhiTextureView,
    texture: SceneRhiTexture,
}

impl VisibilityTexture {
    fn new(
        rhi: &SceneRhi,
        size: bevy_math::UVec2,
        label: &'static str,
        format: rhi::TextureFormat,
        usage: rhi::TextureUsages,
    ) -> Self {
        let mut descriptor =
            rhi::TextureDescriptor::new_2d(rhi::Extent3d::new_2d(size.x, size.y), format, usage);
        descriptor.label = Some(label.into());
        let texture = rhi
            .create_texture(&descriptor)
            .expect("visibility texture descriptor must be supported by the host renderer");
        let default_view = texture
            .create_view(&rhi::TextureViewDescriptor::default())
            .expect("visibility texture default view must match its allocation");
        Self {
            default_view,
            texture,
        }
    }
}

#[derive(Component)]
pub(crate) struct ViewVisibilityBuffer {
    ids: VisibilityIdTexture,
    metadata: VisibilityIdTexture,
    /// Linear HDR radiance the shading-resolve pass stores into and the
    /// composite step samples from.
    scene_color: VisibilityTexture,
    /// SSR energy-conservation export: the IBL specular radiance the resolve
    /// folded into `scene_color` per pixel, so the SSR composite can subtract
    /// it before substituting the screen-space reflection.
    ssr_env_specular: VisibilityTexture,
    /// SSR energy-conservation export: the split-sum environment-BRDF weight
    /// `(f0 * dfg.x + dfg.y) * occlusion` the resolve scaled that specular by,
    /// reused by the composite to weight the SSR reflection identically.
    ssr_spec_weight: VisibilityTexture,
    /// Per-pixel screen-space motion vector (`cur_uv - prev_uv`) the resolve
    /// pass writes for every covered pixel; consumed by the SSR temporal
    /// reprojection to follow object + camera motion instead of ghosting.
    motion_vectors: VisibilityTexture,
    /// Screen-space GI export: the pre-albedo diffuse *irradiance* the resolve
    /// evaluated per pixel. The SSGI hemisphere gather blends missed rays
    /// toward this so indirect light degrades to the IBL/SH ambient instead of
    /// to black.
    ssgi_ambient: VisibilityTexture,
    /// Screen-space GI export: the Lambertian albedo (`base_color *
    /// (1 - metallic)`) the SSGI composite multiplies the gathered pre-albedo
    /// radiance by before blending it over the IBL diffuse under confidence.
    ssgi_albedo: VisibilityTexture,
    /// World-space ReSTIR direct export: the raw cosine-weighted punctual
    /// *irradiance* the resolve evaluated per pixel (no 1/pi, Fresnel or
    /// albedo), in the same units as the ReSTIR resolve's `gi_out`. The
    /// world_restir composite subtracts the clustered punctual diffuse this
    /// reconstructs before folding in the reservoir estimate, so the
    /// substitution never double-counts the direct lighting the cache reused.
    world_restir_direct: VisibilityTexture,
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

    /// The actual raster attachments used by the visibility pass.  The
    /// feedback bridge copies these textures after the render graph has
    /// recorded the raster work; it never treats the CPU `VisibilityBuffer`
    /// helper as a producer.
    pub(crate) fn feedback_sources(
        &self,
    ) -> (
        &bevy_render::render_resource::Texture,
        &bevy_render::render_resource::Texture,
        bevy_math::UVec2,
    ) {
        (&self.ids.texture, &self.metadata.texture, self.size)
    }

    /// Storage/sampling view of the linear HDR radiance target.
    pub(crate) fn scene_color_view(&self) -> &bevy_render::render_resource::TextureView {
        self.scene_color.default_view.view()
    }

    /// The `scene_color` GPU texture itself, for a full-surface
    /// `copy_texture_to_texture` (an opt-in ping-pong post-process writes its
    /// result back in place here).
    pub(crate) fn scene_color_texture(&self) -> &bevy_render::render_resource::Texture {
        self.scene_color_rhi().texture()
    }

    /// The owner of the HDR resource exported to the native composition pass.
    /// Imports must use this lease's table and keep the view resident through
    /// recording; its ID is not a native Bevy texture ID.
    pub(crate) fn scene_color_rhi(&self) -> &SceneRhiTexture {
        &self.scene_color.texture
    }

    /// Storage/sampling view of the SSR IBL-specular export.
    pub(crate) fn ssr_env_specular_view(&self) -> &bevy_render::render_resource::TextureView {
        self.ssr_env_specular.default_view.view()
    }

    /// Storage/sampling view of the SSR environment-BRDF weight export.
    pub(crate) fn ssr_spec_weight_view(&self) -> &bevy_render::render_resource::TextureView {
        self.ssr_spec_weight.default_view.view()
    }

    /// Storage/sampling view of the per-pixel motion-vector G-buffer.
    pub(crate) fn motion_vectors_view(&self) -> &bevy_render::render_resource::TextureView {
        self.motion_vectors.default_view.view()
    }

    /// Storage/sampling view of the SSGI pre-albedo ambient-irradiance export.
    pub(crate) fn ssgi_ambient_view(&self) -> &bevy_render::render_resource::TextureView {
        self.ssgi_ambient.default_view.view()
    }

    /// Storage/sampling view of the SSGI Lambertian-albedo export.
    pub(crate) fn ssgi_albedo_view(&self) -> &bevy_render::render_resource::TextureView {
        self.ssgi_albedo.default_view.view()
    }

    /// Storage/sampling view of the world-space ReSTIR direct-light export
    /// (raw cosine-weighted punctual irradiance).
    pub(crate) fn world_restir_direct_view(&self) -> &bevy_render::render_resource::TextureView {
        self.world_restir_direct.default_view.view()
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
    pub(crate) buckets: Buffer,
    pub(crate) input_bind_group: Option<BindGroup>,
    pub(crate) output_bind_group: Option<BindGroup>,
    pub(crate) size: bevy_math::UVec2,
    pub(crate) capacity: u32,
    pub(crate) bucket_count: u32,
    pub(crate) bucket_table_version: u32,
}

impl ViewShadingBuffers {
    fn new(
        device: &RenderDevice,
        size: bevy_math::UVec2,
        bucket_table: &prism_render_shading::ResponseBucketTable,
        bucket_table_version: u32,
    ) -> Self {
        let capacity = size.x.saturating_mul(size.y).max(1);
        let bucket_count = bucket_table.len() as u32;
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
            class_counts: storage_buffer::<u32>(device, "prism class counts", bucket_count, false),
            class_offsets: storage_buffer::<u32>(
                device,
                "prism class offsets",
                bucket_count,
                false,
            ),
            class_cursors: storage_buffer::<u32>(
                device,
                "prism class cursors",
                bucket_count,
                false,
            ),
            dispatch_args: storage_buffer::<GpuShadingDispatchArgs>(
                device,
                "prism shading dispatch",
                bucket_count,
                true,
            ),
            diagnostics: storage_buffer::<u32>(device, "prism classification diagnostics", 4, true),
            buckets: device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("prism response bucket table"),
                contents: bytemuck::cast_slice(
                    &bucket_table
                        .buckets()
                        .iter()
                        .map(|bucket| super::classification_gpu::GpuShadingBucket {
                            execution_family: bucket.execution_family as u32,
                            program_id: bucket.program_id.0,
                            render_phase: bucket.render_phase as u32,
                            reserved: 0,
                        })
                        .collect::<Vec<_>>(),
                ),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
            input_bind_group: None,
            output_bind_group: None,
            size,
            capacity,
            bucket_count,
            bucket_table_version,
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
    rhi: Res<SceneRhi>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &ExtractedCamera,
        Option<&Msaa>,
        Option<&ViewVisibilityBuffer>,
    )>,
) {
    for (entity, camera, msaa, existing) in &views {
        let size = camera
            .physical_viewport_size
            .filter(|size| size.x != 0 && size.y != 0);
        if !settings.enable_visibility_buffer
            || msaa.is_some_and(|value| value.samples() != 1)
            || size.is_none()
        {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewVisibilityBuffer>();
            }
            continue;
        }
        let size = size.expect("nonzero viewport checked above");
        if existing.is_some_and(|buffer| {
            buffer.size == size
                && std::sync::Arc::ptr_eq(buffer.scene_color_rhi().device(), rhi.device())
        }) {
            continue;
        }

        // Keep the per-view allocation stable between frames, including views
        // with equal extents. Replacing the host bridge also invalidates this
        // owner, so table-local IDs never cross a device recovery.
        let storage = rhi::TextureUsages::STORAGE_BINDING | rhi::TextureUsages::TEXTURE_BINDING;
        let allocate =
            |label, format, usage| VisibilityTexture::new(&rhi, size, label, format, usage);
        commands.entity(entity).insert(ViewVisibilityBuffer {
            ids: VisibilityIdTexture::new(&device, size, "prism visibility IDs"),
            metadata: VisibilityIdTexture::new(&device, size, "prism visibility metadata"),
            // COPY_DST receives ping-pong post-processing results. Consumers
            // sample the same allocation; this is not an RHI mirror.
            scene_color: allocate(
                "prism scene color (HDR)",
                rhi::TextureFormat::Rgba16Float,
                storage | rhi::TextureUsages::COPY_DST,
            ),
            // Resolve always writes these exports, even when their optional
            // downstream consumers are disabled.
            ssr_env_specular: allocate(
                "prism SSR env specular export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            ssr_spec_weight: allocate(
                "prism SSR spec weight export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            // Transparent water writes its MRT over opaque resolve motion.
            motion_vectors: allocate(
                "prism motion vectors",
                rhi::TextureFormat::Rg16Float,
                storage | rhi::TextureUsages::RENDER_ATTACHMENT,
            ),
            ssgi_ambient: allocate(
                "prism SSGI ambient export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            ssgi_albedo: allocate(
                "prism SSGI albedo export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            world_restir_direct: allocate(
                "prism world-space ReSTIR direct export",
                rhi::TextureFormat::Rgba16Float,
                storage,
            ),
            size,
        });
    }
}

pub(crate) fn prepare_shading_buffers(
    mut commands: Commands,
    settings: Res<super::runtime::PrismShadingSettings>,
    materials: Res<super::super::material::runtime::RenderMaterialRegistry>,
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
        let bucket_table_version = materials.registry.execution_table_version();
        if existing.is_some_and(|buffers| {
            buffers.size == size && buffers.bucket_table_version == bucket_table_version
        }) {
            continue;
        }
        let bucket_table = match materials.registry.execution_tables() {
            Some(tables) => {
                match prism_render_shading::ResponseBucketTable::from_execution_tables(tables) {
                    Ok(table) => table,
                    Err(error) => {
                        bevy_log::error!("invalid material response bucket table: {error:?}");
                        commands.entity(entity).remove::<ViewShadingBuffers>();
                        continue;
                    }
                }
            }
            None => prism_render_shading::ResponseBucketTable::builtin(),
        };
        commands.entity(entity).insert(ViewShadingBuffers::new(
            &device,
            size,
            &bucket_table,
            bucket_table_version,
        ));
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
                "@location(6) barycentrics: vec3<f32>",
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
                format!("enable wgpu_binding_array;\n{stubs}{source}"),
                "shaders/prism_visibility_raster.wesl",
            ),
        );
        cache
            .get(0, shader_id, &[])
            .unwrap_or_else(|error| panic!("visibility raster shader failed: {error}"));
    }
}
