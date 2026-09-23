use bevy_ecs::prelude::*;
use bevy_image::ToExtents;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
    renderer::RenderDevice,
    texture::{CachedTexture, TextureCache},
};

pub(crate) const VISIBILITY_ID_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;
pub(crate) const VISIBILITY_BARYCENTRIC_FORMAT: TextureFormat = TextureFormat::Rg16Unorm;

#[derive(Component)]
pub(crate) struct ViewVisibilityBuffer {
    ids: CachedTexture,
    barycentrics: CachedTexture,
    pub(crate) size: bevy_math::UVec2,
}

impl ViewVisibilityBuffer {
    pub(crate) fn texture_views(
        &self,
    ) -> (
        &bevy_render::render_resource::TextureView,
        &bevy_render::render_resource::TextureView,
    ) {
        (&self.ids.default_view, &self.barycentrics.default_view)
    }
}

pub(crate) fn prepare_visibility_buffers(
    mut commands: Commands,
    settings: Res<super::runtime::PrismShadingSettings>,
    mut texture_cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, Option<&ViewVisibilityBuffer>)>,
) {
    for (entity, camera, existing) in &views {
        if !settings.enable_visibility_buffer {
            if existing.is_some() {
                commands.entity(entity).remove::<ViewVisibilityBuffer>();
            }
            continue;
        }
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        if existing.is_some_and(|buffer| buffer.size == size) {
            let _ = existing.map(ViewVisibilityBuffer::texture_views);
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
        let barycentrics = texture_cache.get(
            &device,
            TextureDescriptor {
                label: Some("prism visibility barycentrics"),
                size: size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: VISIBILITY_BARYCENTRIC_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
        );
        commands.entity(entity).insert(ViewVisibilityBuffer {
            ids,
            barycentrics,
            size,
        });
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
    fn formats_cover_full_generational_identity_and_compact_barycentrics() {
        assert_eq!(VISIBILITY_ID_FORMAT, TextureFormat::Rgba32Uint);
        assert_eq!(VISIBILITY_BARYCENTRIC_FORMAT, TextureFormat::Rg16Unorm);
        assert_eq!(VISIBILITY_ID_FORMAT.block_copy_size(None), Some(16));
        assert_eq!(VISIBILITY_BARYCENTRIC_FORMAT.block_copy_size(None), Some(4));
    }

    #[test]
    fn visibility_raster_wesl_compiles_with_barycentric_contract() {
        let shader_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d56_4953_5241_5354_4552_0001),
        };
        let source = include_str!("../shaders/visibility_raster.wesl");
        let source = source[source.find("struct GpuSceneInstance").unwrap()..]
            .replace("bevy_render::utils::decompress_vertex_position", "decompress_vertex_position")
            .replace("@builtin(barycentric) barycentrics: vec3<f32>", "@location(5) barycentrics: vec3<f32>");
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
            Shader::from_wesl(format!("{stubs}{source}"), "shaders/prism_visibility_raster.wesl"),
        );
        cache
            .get(0, shader_id, &[])
            .unwrap_or_else(|error| panic!("visibility raster shader failed: {error}"));
    }
}
