//! The GPU form of an [`Image`]: texture, view, and sampler.
//!
//! [`GpuImage`] is what a material binding reaches for — the sampled texture and
//! the sampler it declared — and it is prepared and released by the render-asset
//! pipeline (`register_render_assets::<I, GpuImage>`), so an image created or
//! changed in the simulation reaches the GPU without the renderer knowing
//! anything about the image's authoring side.

use prism_asset::AssetId;
use prism_image::Image;
use prism_render_driver::{
    Extent3d, ImageDataLayout, RenderDevice, RenderQueue, SamplerDescriptor, SamplerId,
    TextureDescriptor, TextureFormat, TextureId, TextureUsages, TextureViewDescriptor,
    TextureViewId, TextureWrite,
};

use crate::render_asset::{PrepareAssetError, RenderAsset};

/// The usages an image texture is created with.
///
/// It is sampled by a shader and uploaded into, which is all an image needs; a
/// render target or a storage image would be a different kind of texture.
pub(crate) const IMAGE_TEXTURE_USAGES: TextureUsages =
    TextureUsages::TEXTURE_BINDING.union(TextureUsages::COPY_DST);

/// An image on the GPU.
pub struct GpuImage {
    /// The texture holding the texels.
    pub texture: TextureId,
    /// The texture's default view, what a binding binds.
    pub view: TextureViewId,
    /// The sampler the image declared.
    pub sampler: SamplerId,
    /// The texture's extents.
    pub extent: Extent3d,
    /// The texture's format.
    pub format: TextureFormat,
    /// The sampler descriptor the current sampler was created from, so a
    /// re-prepare knows whether it can keep it.
    sampler_descriptor: SamplerDescriptor,
}

impl GpuImage {
    /// The byte length of one texel row, as uploaded and as read back.
    #[must_use]
    pub const fn bytes_per_row(&self) -> u32 {
        self.extent.width * self.format.bytes_per_texel()
    }
}

/// Creates, reuses, or releases an image's texture/view/sampler and uploads it.
///
/// The ownership rule from [`RenderAsset::prepare`] lives here: a texture from
/// `previous` that already has the right extent and format is reused (and
/// re-uploaded into), while a replacement is fully created and uploaded before
/// the old objects are released. That preserves the last-known-good GPU form
/// until the new form is ready.
pub fn prepare_image(
    device: &dyn RenderDevice,
    queue: &dyn RenderQueue,
    source: &Image,
    previous: Option<&GpuImage>,
) -> GpuImage {
    let extent = source.extent();
    let format = source.format();
    let reusable =
        previous.is_some_and(|previous| previous.extent == extent && previous.format == format);

    let (texture, view, old_texture) = if reusable {
        let previous = previous.expect("checked above");
        (previous.texture, previous.view, None)
    } else {
        let texture = device.create_texture(&TextureDescriptor::new_2d(
            extent,
            format,
            IMAGE_TEXTURE_USAGES,
        ));
        let view = device.create_texture_view(texture, &TextureViewDescriptor::default());
        (
            texture,
            view,
            previous.map(|previous| (previous.texture, previous.view)),
        )
    };

    let desired_sampler = source.sampler();
    let (sampler, old_sampler) = match previous {
        Some(previous) if previous.sampler_descriptor == *desired_sampler => {
            (previous.sampler, None)
        }
        Some(previous) => (
            device.create_sampler(desired_sampler),
            Some(previous.sampler),
        ),
        None => (device.create_sampler(desired_sampler), None),
    };

    queue.write_texture(
        TextureWrite {
            texture,
            mip_level: 0,
            origin: [0, 0, 0],
            size: extent,
        },
        source.data(),
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(source.bytes_per_row()),
            rows_per_image: Some(extent.height),
        },
    );

    if let Some((old_texture, old_view)) = old_texture {
        device.destroy_texture_view(old_view);
        device.destroy_texture(old_texture);
    }
    if let Some(old_sampler) = old_sampler {
        device.destroy_sampler(old_sampler);
    }

    GpuImage {
        texture,
        view,
        sampler,
        extent,
        format,
        sampler_descriptor: desired_sampler.clone(),
    }
}

impl RenderAsset for GpuImage {
    type SourceAsset = Image;

    fn prepare(
        source: &Image,
        id: AssetId<Image>,
        device: &dyn RenderDevice,
        queue: &dyn RenderQueue,
        previous: Option<&Self>,
    ) -> Result<Self, PrepareAssetError<Image>> {
        // `Image::new` already validated the data; re-validating here would
        // duplicate the check, so an image that exists is uploadable.
        if source.extent().width == 0 || source.extent().height == 0 {
            return Err(PrepareAssetError::Unpreparable { id });
        }
        Ok(prepare_image(device, queue, source, previous))
    }

    fn unload(prepared: &Self, device: &dyn RenderDevice) {
        device.destroy_sampler(prepared.sampler);
        device.destroy_texture_view(prepared.view);
        device.destroy_texture(prepared.texture);
    }
}
