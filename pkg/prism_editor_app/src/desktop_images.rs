//! Native thumbnail texture residency. The viewport owns image rendering.
use crate::{DesktopContentImage, DesktopGpu};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use prism_editor_content::thumbnail::Thumbnail;
use prism_editor_content::thumbnail::ThumbnailKey;
use prism_editor_viewport::image::{
    ImageError, ImageHandle, ImageLimits, ImageQuad, ImageRenderer,
};

/// Bounded per-device thumbnail residency and physical image preparation.
/// Recreate after device loss/resume; CPU page pixels stay with the UI.
pub struct DesktopContentImages {
    renderer: ImageRenderer,
    resident: BTreeMap<ThumbnailKey, (ImageHandle, Arc<Thumbnail>)>,
    uploads: u64,
    #[cfg(feature = "runtime-preview")]
    preview: Option<(Arc<prism_editor_ui::FrameGraphPreviewImage>, ImageHandle)>,
}

impl DesktopContentImages {
    /// Shares the native UI/viewport GPU and color target format.
    pub fn new(gpu: &DesktopGpu) -> Result<Self, ImageError> {
        Ok(Self {
            renderer: ImageRenderer::new(
                &gpu.device,
                &gpu.queue,
                gpu.format,
                ImageLimits::default(),
            )?,
            resident: BTreeMap::new(),
            uploads: 0,
            #[cfg(feature = "runtime-preview")]
            preview: None,
        })
    }

    /// Synchronize only visible images. Unchanged keys retain textures; removed
    /// and superseded keys release residency before new allocations. An error
    /// clears preparation so callers can never encode an old source revision.
    /// Call before encoding, and submit before preparing another frame.
    pub fn prepare(
        &mut self,
        images: &[DesktopContentImage],
        scale: f32,
        extent: [u32; 2],
    ) -> Result<(), ImageError> {
        #[cfg(feature = "runtime-preview")]
        {
            self.prepare_with_preview(images, None, scale, extent)
        }
        #[cfg(not(feature = "runtime-preview"))]
        {
            let result = self.prepare_inner(images, scale, extent);
            if result.is_err() {
                self.renderer.clear();
                self.resident.clear();
            }
            result
        }
    }

    /// Prepare content and the actual active frame-graph image in the same
    /// bounded compositor. Display image identity controls uploads; geometry
    /// changes reuse its texture. Hidden images release residency.
    #[cfg(feature = "runtime-preview")]
    pub fn prepare_with_preview(
        &mut self,
        images: &[DesktopContentImage],
        preview: Option<&crate::DesktopFrameGraphImage>,
        scale: f32,
        extent: [u32; 2],
    ) -> Result<(), ImageError> {
        let result = self.prepare_preview_inner(images, preview, scale, extent);
        if result.is_err() {
            self.renderer.clear();
            self.resident.clear();
            self.preview = None;
        }
        result
    }

    #[cfg(feature = "runtime-preview")]
    fn prepare_preview_inner(
        &mut self,
        images: &[DesktopContentImage],
        preview: Option<&crate::DesktopFrameGraphImage>,
        scale: f32,
        extent: [u32; 2],
    ) -> Result<(), ImageError> {
        if preview.is_none()
            && let Some((_, handle)) = self.preview.take()
        {
            self.renderer.remove(&handle)?;
        }
        let mut quads = self.prepare_quads(images, scale, extent)?;
        if let Some(preview) = preview {
            let previous = self.preview.as_ref().map(|(image, _)| image.clone());
            let handle = preview
                .view
                .sync_image(&mut self.renderer, &mut self.preview)?;
            if self.preview.as_ref().is_some_and(|(image, _)| {
                previous.as_ref().is_none_or(|old| !Arc::ptr_eq(old, image))
            }) {
                self.uploads = self.uploads.saturating_add(1);
            }
            if let Some(handle) = handle {
                let logical =
                    |r: prism_ui_layout::Rect| [r.left(), r.top(), r.size.width, r.size.height];
                quads.push(
                    preview
                        .view
                        .image_quad(handle, logical(preview.rect), logical(preview.clip), scale)
                        .map_err(|_| ImageError::InvalidGeometry)?,
                );
            }
        }
        self.renderer.prepare(extent, &quads)
    }

    #[cfg(not(feature = "runtime-preview"))]
    fn prepare_inner(
        &mut self,
        images: &[DesktopContentImage],
        scale: f32,
        extent: [u32; 2],
    ) -> Result<(), ImageError> {
        let quads = self.prepare_quads(images, scale, extent)?;
        self.renderer.prepare(extent, &quads)
    }

    fn prepare_quads(
        &mut self,
        images: &[DesktopContentImage],
        scale: f32,
        extent: [u32; 2],
    ) -> Result<Vec<ImageQuad>, ImageError> {
        if !scale.is_finite() || !(0.125..=16.0).contains(&scale) {
            return Err(ImageError::InvalidGeometry);
        }
        let visible = |image: &&DesktopContentImage| {
            image.opacity > 0.0
                && image.rect.size.width > 0.0
                && image.rect.size.height > 0.0
                && image.rect.left().max(image.clip.left()).max(0.0)
                    < image
                        .rect
                        .right()
                        .min(image.clip.right())
                        .min(extent[0] as f32 / scale)
                && image.rect.top().max(image.clip.top()).max(0.0)
                    < image
                        .rect
                        .bottom()
                        .min(image.clip.bottom())
                        .min(extent[1] as f32 / scale)
        };
        let wanted: BTreeSet<_> = images
            .iter()
            .filter(visible)
            .map(|image| image.request.clone())
            .collect();
        let removed: Vec<_> = self
            .resident
            .keys()
            .filter(|key| !wanted.contains(*key))
            .cloned()
            .collect();
        for key in removed {
            if let Some((handle, _)) = self.resident.remove(&key) {
                self.renderer.remove(&handle)?;
            }
        }
        let physical = |r: prism_ui_layout::Rect| {
            [
                r.left() * scale,
                r.top() * scale,
                r.size.width * scale,
                r.size.height * scale,
            ]
        };
        let mut quads = Vec::new();
        for image in images.iter().filter(visible) {
            // A service epoch reset can replace pixels under the same key.
            // Never reuse an old device texture after that CPU publication.
            if self
                .resident
                .get(&image.request)
                .is_some_and(|(_, pixels)| !Arc::ptr_eq(pixels, &image.image))
                && let Some((handle, _)) = self.resident.remove(&image.request)
            {
                self.renderer.remove(&handle)?;
            }
            if !self.resident.contains_key(&image.request) {
                let handle = self.renderer.upload(
                    image.image.width(),
                    image.image.height(),
                    image.image.rgba8(),
                )?;
                self.resident
                    .insert(image.request.clone(), (handle, image.image.clone()));
                self.uploads = self.uploads.saturating_add(1);
            }
            quads.push(ImageQuad {
                image: self.resident[&image.request].0.clone(),
                rect: physical(image.rect),
                clip: physical(image.clip),
                opacity: image.opacity,
            });
        }
        Ok(quads)
    }

    /// Load-composite into the caller target; no intermediate target or readback.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
    ) -> Result<(), ImageError> {
        self.renderer.encode(encoder, target)
    }

    /// Successful texture uploads since this device instance was created.
    #[must_use]
    pub const fn texture_uploads(&self) -> u64 {
        self.uploads
    }

    /// Physical geometry writes; identical frames keep this unchanged.
    #[must_use]
    pub fn geometry_uploads(&self) -> u64 {
        self.renderer.geometry_uploads()
    }

    /// Retained texture count and logical RGBA8 bytes.
    #[must_use]
    pub fn usage(&self) -> (usize, u64) {
        self.renderer.usage()
    }
}
