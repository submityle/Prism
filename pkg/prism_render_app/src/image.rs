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

/// Owns resources created during one image replacement until the upload commits.
///
/// The RHI currently exposes fire-and-forget creation and upload methods, so a
/// backend failure is reported as an unwind rather than a typed `Result`. This
/// guard still gives successful resource creations transactional cleanup on
/// unwind and, more importantly, keeps the previous representation untouched
/// until the replacement upload has returned.
struct ImageResourceTransaction<'a> {
    device: &'a dyn RenderDevice,
    texture: Option<TextureId>,
    view: Option<TextureViewId>,
    sampler: Option<SamplerId>,
    committed: bool,
}

impl<'a> ImageResourceTransaction<'a> {
    fn new(device: &'a dyn RenderDevice) -> Self {
        Self {
            device,
            texture: None,
            view: None,
            sampler: None,
            committed: false,
        }
    }

    fn commit(&mut self) {
        self.committed = true;
    }

    fn create_texture(&mut self, descriptor: &TextureDescriptor) -> TextureId {
        let texture = self.device.create_texture(descriptor);
        self.texture = Some(texture);
        texture
    }

    fn create_view(&mut self, texture: TextureId) -> TextureViewId {
        let view = self
            .device
            .create_texture_view(texture, &TextureViewDescriptor::default());
        self.view = Some(view);
        view
    }

    fn create_sampler(&mut self, descriptor: &SamplerDescriptor) -> SamplerId {
        let sampler = self.device.create_sampler(descriptor);
        self.sampler = Some(sampler);
        sampler
    }
}

impl Drop for ImageResourceTransaction<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Some(sampler) = self.sampler.take() {
            self.device.destroy_sampler(sampler);
        }
        if let Some(view) = self.view.take() {
            self.device.destroy_texture_view(view);
        }
        if let Some(texture) = self.texture.take() {
            self.device.destroy_texture(texture);
        }
    }
}

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
/// The ownership rule from [`RenderAsset::prepare`] lives here: every prepare
/// that has a previous representation builds a private texture/view first, so
/// uploading new texels never mutates the last-known-good representation. The
/// old texture/view are released only after the new upload returns successfully;
/// an unchanged sampler descriptor may be shared because it is immutable.
pub fn prepare_image(
    device: &dyn RenderDevice,
    queue: &dyn RenderQueue,
    source: &Image,
    previous: Option<&GpuImage>,
) -> GpuImage {
    let extent = source.extent();
    let format = source.format();
    let mut transaction = ImageResourceTransaction::new(device);
    let texture = transaction.create_texture(&TextureDescriptor::new_2d(
        extent,
        format,
        IMAGE_TEXTURE_USAGES,
    ));
    let view = transaction.create_view(texture);

    let desired_sampler = source.sampler();
    let (sampler, reuses_previous_sampler) = match previous {
        Some(previous) if previous.sampler_descriptor == *desired_sampler => {
            (previous.sampler, true)
        }
        _ => {
            let sampler = transaction.create_sampler(desired_sampler);
            (sampler, false)
        }
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

    transaction.commit();
    if let Some(previous) = previous {
        device.destroy_texture_view(previous.view);
        device.destroy_texture(previous.texture);
        if !reuses_previous_sampler {
            device.destroy_sampler(previous.sampler);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_driver::{DeviceDescriptor, Instance, RequestAdapterOptions};
    use prism_render_driver_null::NullInstance;

    #[test]
    fn an_aborted_transaction_reclaims_real_null_resources() {
        let options = RequestAdapterOptions::default();
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&options)
            .expect("the null backend always has an adapter");
        let (device, _) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("the null device always comes up");

        let mut transaction = ImageResourceTransaction::new(&device);
        let texture = transaction.create_texture(&TextureDescriptor::new_2d(
            Extent3d::new_2d(4, 3),
            TextureFormat::Rgba8Unorm,
            IMAGE_TEXTURE_USAGES,
        ));
        transaction.create_view(texture);
        transaction.create_sampler(&SamplerDescriptor::default());
        assert_eq!(device.live_resource_count(), 3);

        drop(transaction);
        assert_eq!(device.live_resource_count(), 0);
    }
}
