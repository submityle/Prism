//! CPU staging and Bevy queue upload for material-VT resources.

use alloc::vec::Vec;
use core::mem::size_of;
use core::ops::Range;

use bevy_render::{
    render_resource::{
        Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    },
    renderer::RenderQueue,
};
use prism_render_architecture::texture_streaming::{SlotPlacement, TexturePageKey};
use prism_render_material::{
    build_material_vt_atlas_tile, MaterialVtAtlasTile, MaterialVtSourceSnapshot,
};

use crate::shading::texel_cache::gpu::page_table::MaterialVirtualTexturePageTableSnapshot;

use super::{
    abi::MaterialVirtualTextureGpuConfig,
    atlas::{MaterialVirtualTextureAtlasConfig, MaterialVirtualTextureAtlasError},
    resources::MaterialVirtualTextureGpuBundle,
};

/// A single queued atlas write, with its exact destination and row layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureAtlasUpload {
    /// Virtual page carried through for diagnostics and publication matching.
    pub key: TexturePageKey,
    /// Physical slot selected by the material-VT residency producer.
    pub slot: u32,
    /// Array-layer/tile origin in the atlas.
    pub destination: SlotPlacement,
    /// Stored tile edge in texels.
    pub extent_px: u32,
    /// Copy row pitch in bytes.
    pub bytes_per_row: u32,
    /// Owned tile bytes, including gutters and row padding.
    pub bytes: Vec<u8>,
}

impl MaterialVirtualTextureAtlasUpload {
    fn from_tile(
        config: MaterialVirtualTextureAtlasConfig,
        tile: &MaterialVtAtlasTile,
    ) -> Result<Self, MaterialVirtualTextureAtlasAtlasError> {
        config.validate_tile(tile)?;
        let geometry = config
            .geometry()
            .map_err(MaterialVirtualTextureAtlasAtlasError::Atlas)?;
        Ok(Self {
            key: tile.key,
            slot: tile.slot,
            destination: geometry.slot_placement(tile.slot),
            extent_px: geometry.tile_extent_px(),
            bytes_per_row: tile.bytes_per_row,
            bytes: tile.staging_bytes.clone(),
        })
    }

    /// Number of bytes copied by this atlas upload.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Number of block/texel rows in this uncompressed atlas upload.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.extent_px
    }
}

/// Error while preparing material-VT queue writes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaterialVirtualTextureAtlasAtlasError {
    /// Atlas geometry or slot validation failed.
    Atlas(MaterialVirtualTextureAtlasError),
    /// The source page could not be extracted or materialized.
    Source(prism_render_material::MaterialVtAtlasUploadError),
}

impl From<MaterialVirtualTextureAtlasError> for MaterialVirtualTextureAtlasAtlasError {
    fn from(value: MaterialVirtualTextureAtlasError) -> Self {
        Self::Atlas(value)
    }
}

/// All CPU data for one atomic material-VT GPU update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureStaging {
    page_table_words: Vec<u32>,
    config: MaterialVirtualTextureGpuConfig,
    atlas_uploads: Vec<MaterialVirtualTextureAtlasUpload>,
}

impl MaterialVirtualTextureStaging {
    /// Starts an update from the existing validated material-VT snapshot.
    pub fn new(
        atlas_config: MaterialVirtualTextureAtlasConfig,
        snapshot: &MaterialVirtualTexturePageTableSnapshot,
        config: MaterialVirtualTextureGpuConfig,
    ) -> Result<Self, MaterialVirtualTextureStagingError> {
        if config.entry_count != u32::try_from(snapshot.len()).unwrap_or(u32::MAX) {
            return Err(MaterialVirtualTextureStagingError::EntryCountMismatch);
        }
        atlas_config
            .geometry()
            .map_err(MaterialVirtualTextureStagingError::Atlas)?;
        Ok(Self {
            page_table_words: snapshot.gpu_words().to_vec(),
            config,
            atlas_uploads: Vec::new(),
        })
    }

    /// Adds an already guttered source tile to this update.
    pub fn push_tile(
        &mut self,
        atlas_config: MaterialVirtualTextureAtlasConfig,
        tile: &MaterialVtAtlasTile,
    ) -> Result<(), MaterialVirtualTextureStagingError> {
        let upload = MaterialVirtualTextureAtlasUpload::from_tile(atlas_config, tile)
            .map_err(MaterialVirtualTextureStagingError::AtlasUpload)?;
        if self
            .atlas_uploads
            .iter()
            .any(|existing| existing.key == upload.key)
        {
            return Err(MaterialVirtualTextureStagingError::DuplicatePageKey);
        }
        if self
            .atlas_uploads
            .iter()
            .any(|existing| existing.slot == upload.slot)
        {
            return Err(MaterialVirtualTextureStagingError::DuplicatePhysicalSlot);
        }
        if self.atlas_uploads.len() >= self.config.entry_count as usize {
            return Err(MaterialVirtualTextureStagingError::TooManyAtlasUploads);
        }
        self.atlas_uploads.push(upload);
        Ok(())
    }

    /// Builds and adds a guttered tile from the immutable material-VT source.
    ///
    /// The page's borrowed [`MaterialVtPage::rows`] are walked before the
    /// existing atlas builder is called. This keeps the source-row contract at
    /// the GPU boundary while retaining the builder's neighbor/gutter policy.
    pub fn push_source_page(
        &mut self,
        atlas_config: MaterialVirtualTextureAtlasConfig,
        source: &MaterialVtSourceSnapshot,
        key: TexturePageKey,
        slot: u32,
        resident_neighbors: &std::collections::BTreeSet<TexturePageKey>,
    ) -> Result<(), MaterialVirtualTextureStagingError> {
        let page = source
            .page(key)
            .map_err(|_| MaterialVirtualTextureStagingError::SourcePage)?;
        let mut row_bytes = 0usize;
        for row in page.rows() {
            row_bytes = row_bytes
                .checked_add(row.bytes.len())
                .ok_or(MaterialVirtualTextureStagingError::SourcePage)?;
        }
        if row_bytes == 0 {
            return Err(MaterialVirtualTextureStagingError::SourcePage);
        }
        let source_config = atlas_config
            .source_tile_config()
            .map_err(|error| MaterialVirtualTextureStagingError::Source(error))?;
        let tile =
            build_material_vt_atlas_tile(source, key, slot, resident_neighbors, source_config)
                .map_err(MaterialVirtualTextureStagingError::Source)?;
        self.push_tile(atlas_config, &tile)
    }

    /// Flat page-table words to be written to the dedicated storage buffer.
    #[must_use]
    pub fn page_table_words(&self) -> &[u32] {
        &self.page_table_words
    }

    /// Config to be written to the dedicated config buffer.
    #[must_use]
    pub const fn config(&self) -> MaterialVirtualTextureGpuConfig {
        self.config
    }

    /// Atlas writes in deterministic caller order.
    #[must_use]
    pub fn atlas_uploads(&self) -> &[MaterialVirtualTextureAtlasUpload] {
        &self.atlas_uploads
    }
}

/// Failure while assembling one material-VT update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaterialVirtualTextureStagingError {
    /// Atlas configuration is invalid.
    Atlas(MaterialVirtualTextureAtlasError),
    /// A tile does not match the atlas layout.
    AtlasUpload(MaterialVirtualTextureAtlasAtlasError),
    /// The config entry count disagrees with the validated snapshot.
    EntryCountMismatch,
    /// Two uploads address the same virtual page.
    DuplicatePageKey,
    /// Two uploads claim the same physical atlas slot.
    DuplicatePhysicalSlot,
    /// More atlas uploads were supplied than page-table entries.
    TooManyAtlasUploads,
    /// The source page has no readable rows.
    SourcePage,
    /// The source-to-atlas conversion failed.
    Source(prism_render_material::MaterialVtAtlasUploadError),
}

/// Exact queue-write ranges returned by one upload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureUploadRanges {
    /// Resource slice receiving this complete update.
    pub resource_slot: u8,
    /// Page-table byte range written in the dedicated table buffer.
    pub page_table: Range<u64>,
    /// Config byte range written in the config buffer.
    pub config: Range<u64>,
    /// Atlas writes, including their destination and byte lengths.
    pub atlas: Vec<MaterialVirtualTextureAtlasUploadRange>,
}

/// Range metadata for one atlas write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVirtualTextureAtlasUploadRange {
    /// Virtual page identity.
    pub key: TexturePageKey,
    /// Physical atlas slot.
    pub slot: u32,
    /// Destination placement.
    pub destination: SlotPlacement,
    /// Number of uploaded bytes.
    pub byte_len: u64,
}

impl MaterialVirtualTextureGpuBundle {
    /// Validates a staging payload and computes the exact queue-write ranges
    /// without recording any GPU writes. Callers that need a recoverable
    /// lifecycle reservation use this before [`Self::upload_staging`].
    pub fn plan_upload(
        &self,
        staging: &MaterialVirtualTextureStaging,
    ) -> Result<MaterialVirtualTextureUploadRanges, MaterialVirtualTextureUploadError> {
        let capacity = self.sizes().page_table_entry_capacity;
        if staging.config.entry_count > capacity {
            return Err(MaterialVirtualTextureUploadError::PageTableTooLarge {
                entries: staging.config.entry_count,
                capacity,
            });
        }
        if staging.config.enabled != 0
            && staging.atlas_uploads.len() != staging.config.entry_count as usize
        {
            return Err(MaterialVirtualTextureUploadError::IncompleteAtlasUploads {
                expected: staging.config.entry_count,
                actual: staging.atlas_uploads.len() as u32,
            });
        }
        let atlas = staging
            .atlas_uploads
            .iter()
            .map(|upload| MaterialVirtualTextureAtlasUploadRange {
                key: upload.key,
                slot: upload.slot,
                destination: upload.destination,
                byte_len: upload.bytes.len() as u64,
            })
            .collect();
        Ok(MaterialVirtualTextureUploadRanges {
            resource_slot: self.staging_slot(),
            page_table: 0..(staging.page_table_words.len() * size_of::<u32>()) as u64,
            config: 0..MaterialVirtualTextureGpuConfig::byte_size(),
            atlas,
        })
    }

    /// Enqueues all writes in `staging` against this bundle's independent GPU resources.
    pub fn upload_staging(
        &self,
        queue: &RenderQueue,
        staging: &MaterialVirtualTextureStaging,
    ) -> Result<MaterialVirtualTextureUploadRanges, MaterialVirtualTextureUploadError> {
        let ranges = self.plan_upload(staging)?;
        if !staging.page_table_words.is_empty() {
            queue.write_buffer(
                self.staging_page_table_buffer(),
                0,
                bytemuck::cast_slice(&staging.page_table_words),
            );
        }
        queue.write_buffer(
            self.staging_config_buffer(),
            0,
            bytemuck::bytes_of(&staging.config),
        );
        for upload in &staging.atlas_uploads {
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: self.staging_atlas_texture(),
                    mip_level: 0,
                    origin: Origin3d {
                        x: upload.destination.origin_x_px,
                        y: upload.destination.origin_y_px,
                        z: upload.destination.layer,
                    },
                    aspect: TextureAspect::All,
                },
                &upload.bytes,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(upload.bytes_per_row),
                    rows_per_image: Some(upload.rows()),
                },
                Extent3d {
                    width: upload.extent_px,
                    height: upload.extent_px,
                    depth_or_array_layers: 1,
                },
            );
        }
        Ok(ranges)
    }
}

/// Error while writing a CPU staging update to a device-owned bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialVirtualTextureUploadError {
    /// The bundle's dedicated page-table allocation is too small.
    PageTableTooLarge { entries: u32, capacity: u32 },
    /// An enabled table was not accompanied by one atlas tile per entry.
    IncompleteAtlasUploads { expected: u32, actual: u32 },
}
