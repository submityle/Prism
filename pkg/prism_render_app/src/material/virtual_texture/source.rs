//! Immutable source snapshots. This observes the existing image extraction;
//! it never drains a second asset-event reader or steals generic prepare work.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use prism_asset::AssetId;
use prism_ecs::prelude::Resource;
use prism_image::Image;
use prism_render_driver::TextureFormat;
use prism_render_material::{
    MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceLayout, MaterialVtSourceSnapshot,
    NO_VIRTUAL_TEXTURE, PORTABLE_TEXTURE_ATLAS,
};

use super::NativeVtError;
use crate::{image::GpuImage, render_asset::ExtractedAssets};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeVtSourceRequest {
    pub texture_id: u32,
    pub page_size: u32,
}

pub struct NativeVtSource {
    pub(super) image: AssetId<Image>,
    pub(super) revision: u64,
    payload: NativeVtPayload,
}

enum NativeVtPayload {
    Cpu(MaterialVtSourceSnapshot),
    #[cfg(feature = "resident-page-streaming")]
    Resident {
        asset: Arc<prism_asset_container::GpuResidentManifestAsset>,
        layout: MaterialVtSourceLayout,
        offset: u64,
    },
}

impl core::fmt::Debug for NativeVtSource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NativeVtSource")
            .field("image", &self.image)
            .field("revision", &self.revision)
            .field("layout", self.layout())
            .field("cpu_backed", &self.cpu_snapshot().is_some())
            .finish()
    }
}

impl NativeVtSource {
    pub fn image(&self) -> AssetId<Image> {
        self.image
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn layout(&self) -> &MaterialVtSourceLayout {
        match &self.payload {
            NativeVtPayload::Cpu(snapshot) => snapshot.layout(),
            #[cfg(feature = "resident-page-streaming")]
            NativeVtPayload::Resident { layout, .. } => layout,
        }
    }

    pub fn cpu_snapshot(&self) -> Option<&MaterialVtSourceSnapshot> {
        match &self.payload {
            NativeVtPayload::Cpu(snapshot) => Some(snapshot),
            #[cfg(feature = "resident-page-streaming")]
            NativeVtPayload::Resident { .. } => None,
        }
    }

    #[cfg(feature = "resident-page-streaming")]
    pub(super) fn resident(
        &self,
    ) -> Option<(&prism_asset_container::GpuResidentManifestAsset, u64)> {
        match &self.payload {
            NativeVtPayload::Resident { asset, offset, .. } => Some((asset, *offset)),
            NativeVtPayload::Cpu(_) => None,
        }
    }
}

#[derive(Default, Resource)]
pub struct NativeVtSources {
    sources: BTreeMap<u32, Arc<NativeVtSource>>,
    images: BTreeMap<AssetId<Image>, u32>,
    revision: u64,
}

impl NativeVtSources {
    pub fn get(&self, texture: u32) -> Option<&Arc<NativeVtSource>> {
        self.sources.get(&texture)
    }

    /// Replaces both texture-ID and image-ID aliases. An older immutable
    /// snapshot remains readable but can no longer qualify a publication.
    pub fn insert(
        &mut self,
        image: AssetId<Image>,
        snapshot: MaterialVtSourceSnapshot,
    ) -> Result<Arc<NativeVtSource>, NativeVtError> {
        self.insert_payload(image, NativeVtPayload::Cpu(snapshot))
    }

    /// Registers an immutable Manifest-authorized mip payload on its original
    /// device. Only metadata is materialized on the CPU. `offset` is relative
    /// to the asset, not its chunk; the complete mip chain must fit the entry.
    #[cfg(feature = "resident-page-streaming")]
    pub fn insert_resident(
        &mut self,
        image: AssetId<Image>,
        asset: Arc<prism_asset_container::GpuResidentManifestAsset>,
        descriptor: MaterialVtSourceDescriptor,
        offset: u64,
    ) -> Result<Arc<NativeVtSource>, NativeVtError> {
        let layout = MaterialVtSourceLayout::new(descriptor).map_err(NativeVtError::Source)?;
        if offset
            .checked_add(layout.byte_len())
            .is_none_or(|end| end > asset.entry().raw_len)
        {
            return Err(NativeVtError::ImageMipLayout);
        }
        self.insert_payload(
            image,
            NativeVtPayload::Resident {
                asset,
                layout,
                offset,
            },
        )
    }

    fn insert_payload(
        &mut self,
        image: AssetId<Image>,
        payload: NativeVtPayload,
    ) -> Result<Arc<NativeVtSource>, NativeVtError> {
        let texture = match &payload {
            NativeVtPayload::Cpu(snapshot) => snapshot.descriptor().texture_id(),
            #[cfg(feature = "resident-page-streaming")]
            NativeVtPayload::Resident { layout, .. } => layout.descriptor().texture_id(),
        };
        if matches!(texture, NO_VIRTUAL_TEXTURE | PORTABLE_TEXTURE_ATLAS) {
            return Err(NativeVtError::ReservedTextureId);
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(NativeVtError::IdentityExhausted)?;
        let source = Arc::new(NativeVtSource {
            image,
            revision,
            payload,
        });
        self.remove_image(image);
        if let Some(previous) = self.sources.insert(texture, Arc::clone(&source)) {
            self.images.remove(&previous.image);
        }
        self.images.insert(image, texture);
        self.revision = revision;
        Ok(source)
    }

    pub fn remove_image(&mut self, image: AssetId<Image>) {
        if let Some(texture) = self.images.remove(&image) {
            self.sources.remove(&texture);
        }
    }

    pub(super) fn is_current(&self, source: &Arc<NativeVtSource>) -> bool {
        self.get(source.layout().descriptor().texture_id())
            .is_some_and(|current| Arc::ptr_eq(current, source))
    }

    /// Captures a complete, uncompressed mip chain, rejecting image/VT mip
    /// geometry disagreements instead of silently reinterpreting odd extents.
    pub fn capture_image(
        &mut self,
        image_id: AssetId<Image>,
        image: &Image,
        request: NativeVtSourceRequest,
    ) -> Result<Arc<NativeVtSource>, NativeVtError> {
        let result = self.capture_image_inner(image_id, image, request);
        // A failed hot replacement must invalidate the old publication token,
        // but an identical retained extraction must not churn its identity.
        if result.is_err() {
            self.remove_image(image_id);
        }
        result
    }

    fn capture_image_inner(
        &mut self,
        image_id: AssetId<Image>,
        image: &Image,
        request: NativeVtSourceRequest,
    ) -> Result<Arc<NativeVtSource>, NativeVtError> {
        if matches!(
            request.texture_id,
            NO_VIRTUAL_TEXTURE | PORTABLE_TEXTURE_ATLAS
        ) {
            return Err(NativeVtError::ReservedTextureId);
        }
        let extent = image.extent();
        let descriptor = MaterialVtSourceDescriptor::new(
            request.texture_id,
            extent.width,
            extent.height,
            u16::try_from(extent.depth_or_array_layers)
                .map_err(|_| NativeVtError::ImageMipLayout)?,
            request.page_size,
            u8::try_from(image.mip_level_count()).map_err(|_| NativeVtError::ImageMipLayout)?,
            source_format(image.format())?,
        )
        .map_err(NativeVtError::Source)?;
        if let Some(current) = self.sources.get(&request.texture_id)
            && current.image == image_id
            && let Some(snapshot) = current.cpu_snapshot()
            && snapshot.descriptor() == descriptor
            && snapshot.mips().iter().all(|mip| {
                let start = mip.offset_bytes as usize;
                let end = start + mip.level_stride_bytes as usize;
                image.mip_data(u32::from(mip.mip)) == Some(&snapshot.bytes()[start..end])
                    && image.mip_extent(u32::from(mip.mip)).is_some_and(|extent| {
                        extent.width == mip.width && extent.height == mip.height
                    })
            })
        {
            return Ok(Arc::clone(current));
        }
        let mut bytes = Vec::new();
        for mip in 0..image.mip_level_count() {
            let level = image.mip_data(mip).ok_or(NativeVtError::ImageMipLayout)?;
            bytes
                .try_reserve(level.len())
                .map_err(|_| NativeVtError::AllocationFailed)?;
            bytes.extend_from_slice(level);
        }
        let snapshot =
            MaterialVtSourceSnapshot::new(descriptor, bytes).map_err(NativeVtError::Source)?;
        for mip in snapshot.mips() {
            let extent = image
                .mip_extent(u32::from(mip.mip))
                .ok_or(NativeVtError::ImageMipLayout)?;
            if extent.width != mip.width || extent.height != mip.height {
                return Err(NativeVtError::ImageMipLayout);
            }
        }
        self.insert(image_id, snapshot)
    }

    /// Call once after image extraction, before `prepare_assets::<I, GpuImage>`
    /// consumes the batch. Removed entries win, matching generic preparation.
    /// Only the last extracted snapshot per image is captured, in final-event
    /// order. Errors describe those final snapshots, never superseded events
    /// or removed images. Reobserving a retained batch therefore cannot churn
    /// an unchanged final snapshot's publication token.
    /// `requests` is the complete current set, not a delta: omission revokes a
    /// source even without extraction events. A request edit requires another
    /// capture/extraction of that image before the new request can be staged.
    pub fn observe_images(
        &mut self,
        extracted: &ExtractedAssets<GpuImage>,
        requests: &BTreeMap<AssetId<Image>, NativeVtSourceRequest>,
    ) -> Vec<(AssetId<Image>, NativeVtError)> {
        let mut errors = Vec::new();
        let removed: BTreeSet<_> = extracted.removed().iter().copied().collect();
        let latest: BTreeMap<_, _> = extracted
            .extracted()
            .iter()
            .enumerate()
            .map(|(index, (id, _))| (*id, index))
            .collect();
        for (index, (id, image)) in extracted.extracted().iter().enumerate() {
            if latest.get(id) != Some(&index) || removed.contains(id) {
                continue;
            }
            if let Some(request) = requests.get(id) {
                if let Err(error) = self.capture_image(*id, image, *request) {
                    errors.push((*id, error));
                }
            } else if self
                .images
                .get(id)
                .and_then(|texture| self.sources.get(texture))
                .is_some_and(|source| source.cpu_snapshot().is_some())
            {
                self.remove_image(*id);
            }
        }
        for id in &removed {
            self.remove_image(*id);
        }
        // A request edit also invalidates an outstanding job, even when the
        // image itself did not produce another extraction event. The caller
        // must supply that image again to capture the new request.
        self.sources.retain(|_, source| {
            // Explicit resident registrations have their own lifetime. CPU
            // extraction requests must not silently revoke them on quiet frames.
            if source.cpu_snapshot().is_none() {
                return true;
            }
            requests.get(&source.image).is_some_and(|request| {
                let descriptor = source.layout().descriptor();
                request.texture_id == descriptor.texture_id()
                    && request.page_size == descriptor.page_size()
            })
        });
        self.images.retain(|image, texture| {
            self.sources
                .get(texture)
                .is_some_and(|source| source.image == *image)
        });
        errors
    }
}

fn source_format(format: TextureFormat) -> Result<MaterialVtFormat, NativeVtError> {
    Ok(match format {
        TextureFormat::R8Unorm => MaterialVtFormat::R8Unorm,
        TextureFormat::Rg8Unorm => MaterialVtFormat::Rg8Unorm,
        TextureFormat::Rgba8Unorm => MaterialVtFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => MaterialVtFormat::Rgba8Srgb,
        TextureFormat::Rgba16Float => MaterialVtFormat::Rgba16Float,
        // The RHI has no float32-filterability capability. The shipping VT
        // binding uses a filtering sampler, so do not advertise RGBA32F.
        _ => return Err(NativeVtError::UnsupportedFormat),
    })
}
