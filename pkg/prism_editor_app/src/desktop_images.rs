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
        let result = self.prepare_inner(images, scale, extent);
        if result.is_err() {
            self.renderer.clear();
            self.resident.clear();
        }
        result
    }

    fn prepare_inner(
        &mut self,
        images: &[DesktopContentImage],
        scale: f32,
        extent: [u32; 2],
    ) -> Result<(), ImageError> {
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
        self.renderer.prepare(extent, &quads)
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
