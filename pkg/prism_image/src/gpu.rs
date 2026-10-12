//! Uploading a CPU [`Image`] into an RHI texture.

use prism_render_driver::{
    Extent3d, RenderDevice, RenderQueue, TextureDescriptor, TextureDimension, TextureFormat,
    TextureId, TextureUsages, TextureViewDescriptor, TextureViewId, TextureWrite,
};

use crate::image::Image;

/// The texel usages [`upload`] requests for the destination texture.
///
/// - `TEXTURE_BINDING` so the result can be sampled by a material.
/// - `COPY_DST` so the upload itself can write it.
/// - `COPY_SRC` so golden-image read-back and later mip generation can copy it
///   back out.
///
/// The set is fixed rather than configurable so an uploaded image behaves the
/// same everywhere; a caller that needs a narrower texture can create it
/// directly through the RHI.
pub const UPLOAD_USAGE: TextureUsages = TextureUsages::TEXTURE_BINDING
    .union(TextureUsages::COPY_DST)
    .union(TextureUsages::COPY_SRC);

/// An uploaded image: the texture it lives in, a whole-texture view of it, and
/// the metadata a material needs to bind it.
///
/// The view is the `D2`, all-aspect, whole-mip view a shader samples; the
/// format and size mirror what the texture was created with so a binding site
/// does not have to re-derive them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GpuImage {
    /// The uploaded texture.
    pub texture: TextureId,
    /// A default view covering the whole texture.
    pub view: TextureViewId,
    /// The RHI format the texture was created with.
    pub format: TextureFormat,
    /// The uploaded extent in texels.
    pub size: Extent3d,
}

impl GpuImage {
    /// Releases the view and the texture back to `device`.
    ///
    /// Dropping a `GpuImage` does not release anything on its own; the RHI's
    /// ids are plain values. Call this when the image is no longer needed, or
    /// hand the ids to a deferred-reclaim queue once the frames that bound them
    /// have retired.
    pub fn destroy<D: RenderDevice>(&self, device: &D) {
        device.destroy_texture_view(self.view);
        device.destroy_texture(self.texture);
    }
}

/// Uploads `image` into a fresh GPU texture and returns its handles.
///
/// The destination is a single-mip, single-sample 2D texture in
/// `image.format().texture_format()` with [`UPLOAD_USAGE`]. The bytes are
/// written through `queue` using the image's own [`data_layout`], so the upload
/// is ordered with respect to the next submission on that queue.
///
/// [`data_layout`]: Image::data_layout
#[must_use]
pub fn upload<D: RenderDevice, Q: RenderQueue>(image: &Image, device: &D, queue: &Q) -> GpuImage {
    let size = image.extent();
    let format = image.format().texture_format();
    let descriptor = TextureDescriptor {
        label: Some(String::from("prism_image upload")),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: UPLOAD_USAGE,
    };
    let texture = device.create_texture(&descriptor);
    let view = device.create_texture_view(texture, &TextureViewDescriptor::default());
    queue.write_texture(
        TextureWrite {
            texture,
            mip_level: 0,
            origin: [0, 0, 0],
            size,
        },
        image.data(),
        image.data_layout(),
    );
    GpuImage {
        texture,
        view,
        format,
        size,
    }
}
