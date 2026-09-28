//! Reads Bevy [`EnvironmentMapLight`] cube-maps into an SH radiance probe.
//!
//! [`EnvironmentMapLight`] stores its surroundings as two cube-map images: a
//! pre-convolved `diffuse_map` and a mipmapped radiance `specular_map`.  The
//! CPU golden probe in `prism_render_shading` is a *radiance* SH vector that is
//! convolved on demand into irradiance (diffuse) and reconstructed along the
//! reflection vector (specular), so the raw radiance environment -- the
//! `specular_map` base mip -- is the correct source to project.
//!
//! Only CPU-decodable, square cube-maps (six 2D array layers) are supported.
//! Compressed (BCn/ASTC) or otherwise unreadable images return [`None`] so the
//! caller can fall back to the constant ambient term instead of guessing.
//!
//! Projection is O(6 * size^2) per unique image, so results are memoised by
//! [`AssetId`] in [`EnvironmentProbeCache`]: a probe is only re-projected when
//! its backing image changes, not every frame.

use bevy_asset::AssetId;
use bevy_color::ColorToComponents;
use bevy_ecs::prelude::Resource;
use bevy_image::Image;
use bevy_platform::collections::HashMap;
use bevy_render::render_resource::{TextureDimension, TextureViewDimension};
use prism_render_shading::{project_cubemap_to_sh, CubemapFaces, SphericalHarmonicsL2};

/// Largest cube-map edge we will project on the CPU.  Beyond this the per-frame
/// (well, per-change) cost of a full spherical integration is not worth the
/// negligible extra low-frequency accuracy, so oversized probes are declined
/// and the caller keeps the ambient fallback.
const MAX_PROJECTION_SIZE: u32 = 256;

/// Returns `true` when `image` is a square cube-map laid out as six 2D array
/// layers, i.e. something [`Image::get_color_at_3d`] can address per face.
fn is_square_cubemap(image: &Image) -> bool {
    let descriptor = &image.texture_descriptor;
    let size = descriptor.size;
    let is_cube = matches!(
        image
            .texture_view_descriptor
            .as_ref()
            .and_then(|view| view.dimension),
        Some(TextureViewDimension::Cube) | Some(TextureViewDimension::CubeArray)
    ) || size.depth_or_array_layers == 6;
    descriptor.dimension == TextureDimension::D2
        && size.width == size.height
        && size.width > 0
        && size.depth_or_array_layers >= 6
        && is_cube
}

/// Copies the six faces of a CPU-readable cube-map into backend-neutral
/// [`CubemapFaces`] holding linear-RGB radiance.
///
/// Returns [`None`] if the image is not a square cube-map, exceeds
/// [`MAX_PROJECTION_SIZE`], has no CPU-side data, or uses a format
/// [`Image::get_color_at_3d`] cannot decode (e.g. block-compressed).
pub fn cubemap_faces_from_image(image: &Image) -> Option<CubemapFaces> {
    if !is_square_cubemap(image) {
        return None;
    }
    image.data.as_ref()?;
    let size = image.texture_descriptor.size.width;
    if size > MAX_PROJECTION_SIZE {
        return None;
    }
    let texel_count = (size as usize) * (size as usize);
    let mut faces: [Vec<[f32; 3]>; 6] = core::array::from_fn(|_| Vec::with_capacity(texel_count));
    for (face, buffer) in faces.iter_mut().enumerate() {
        for y in 0..size {
            for x in 0..size {
                // A single undecodable texel means the whole format is
                // unreadable, so bail to the ambient fallback rather than
                // stitching a partly-black probe.
                let color = image.get_color_at_3d(x, y, face as u32).ok()?;
                let [r, g, b, _] = color.to_linear().to_f32_array();
                buffer.push([r, g, b]);
            }
        }
    }
    CubemapFaces::new(size, faces)
}

/// Projects an [`EnvironmentMapLight`] radiance cube-map into an SH probe,
/// returning [`None`] when the image cannot be read on the CPU.
pub fn project_image_to_sh(image: &Image) -> Option<SphericalHarmonicsL2> {
    cubemap_faces_from_image(image).map(|faces| project_cubemap_to_sh(&faces))
}

/// Memoises projected environment probes by their backing [`AssetId`].
///
/// Projection is comparatively expensive, and environment maps rarely change,
/// so the render world keeps one cache and only re-projects when an image is
/// seen for the first time.  Entries store the *unscaled* radiance probe; the
/// per-light `intensity` multiplier is applied by the caller after lookup.
#[derive(Resource, Default)]
pub struct EnvironmentProbeCache {
    entries: HashMap<AssetId<Image>, Option<SphericalHarmonicsL2>>,
}

impl EnvironmentProbeCache {
    /// Returns the cached probe for `id`, projecting `image` on first use.
    ///
    /// The result is `None` for images that cannot be projected; that verdict
    /// is cached too so an unreadable map is not retried every frame.
    pub fn get_or_project(
        &mut self,
        id: AssetId<Image>,
        image: &Image,
    ) -> Option<SphericalHarmonicsL2> {
        *self
            .entries
            .entry(id)
            .or_insert_with(|| project_image_to_sh(image))
    }

    /// Drops the cached probe for `id`, forcing a re-projection next lookup.
    /// Call this when an environment image is modified or removed.
    pub fn invalidate(&mut self, id: AssetId<Image>) {
        self.entries.remove(&id);
    }

    /// Number of cached verdicts, exposed for tests and diagnostics.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no verdicts.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::RenderAssetUsages;
    use bevy_image::Image;
    use bevy_render::render_resource::{Extent3d, TextureDimension, TextureFormat};

    /// Builds a six-layer square RGBA-f32 cube-map whose faces are constant
    /// colours, so the projection has a known DC term.
    fn constant_cubemap(size: u32, face_colors: [[f32; 3]; 6]) -> Image {
        let mut data = Vec::with_capacity((size * size * 6 * 4) as usize * 4);
        for color in face_colors {
            for _ in 0..(size * size) {
                for channel in color {
                    data.extend_from_slice(&channel.to_ne_bytes());
                }
                data.extend_from_slice(&1.0f32.to_ne_bytes());
            }
        }
        Image::new(
            Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            data,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::default(),
        )
    }

    #[test]
    fn projects_uniform_cubemap_to_matching_dc_term() {
        let image = constant_cubemap(8, [[0.5, 0.25, 0.75]; 6]);
        let probe = project_image_to_sh(&image).expect("readable cubemap");
        let reference = SphericalHarmonicsL2::from_constant([0.5, 0.25, 0.75]);
        for channel in 0..3 {
            assert!(
                (probe.coefficients[0][channel] - reference.coefficients[0][channel]).abs()
                    < 2.0e-2,
                "channel {channel}: {} vs {}",
                probe.coefficients[0][channel],
                reference.coefficients[0][channel]
            );
        }
    }

    #[test]
    fn rejects_plain_2d_texture() {
        let image = Image::new_fill(
            Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0u8; 16],
            TextureFormat::Rgba32Float,
            RenderAssetUsages::default(),
        );
        assert!(project_image_to_sh(&image).is_none());
    }

    #[test]
    fn cache_projects_once_and_reuses() {
        let image = constant_cubemap(4, [[1.0, 1.0, 1.0]; 6]);
        let id = AssetId::<Image>::default();
        let mut cache = EnvironmentProbeCache::default();
        assert!(cache.is_empty());
        let first = cache.get_or_project(id, &image);
        let second = cache.get_or_project(id, &image);
        assert_eq!(first, second);
        assert_eq!(cache.len(), 1);
        assert!(first.is_some());
        cache.invalidate(id);
        assert!(cache.is_empty());
    }
}
