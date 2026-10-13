//! Portable, non-bindless material texture rows.
//!
//! A deferred resolve cannot bind one image/sampler pair per material.  This
//! module defines the small, honest portable slice used when binding arrays are
//! unavailable: authored RGBA8-unorm images whose base mip fits one normalized
//! page are copied into a fixed-size `texture_2d_array` layer.  The material
//! row carries the layer through the existing index/generation pair; the GPU
//! side additionally publishes the generation only after the tile upload was
//! queued.
//!
//! Larger or differently formatted sources are deliberately rejected here.
//! They remain eligible for the virtual-texture producer when its contract
//! accepts them, otherwise the row is marked unavailable and diagnostics can
//! report the capability failure.  No source is silently converted to white.

use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_render_material::{
    build_material_vt_atlas_tile, MaterialVtAtlasUploadError, MaterialVtFormat,
    MaterialVtSourceSnapshot,
};

use super::virtual_texture::{MaterialVirtualTextureAtlasConfig, MaterialVirtualTextureAtlasError};

/// Logical texel edge of one portable normalized row.
pub const PORTABLE_TEXTURE_PAGE_SIZE: u32 = 128;
/// One clamp gutter on each side of a portable row.
pub const PORTABLE_TEXTURE_GUTTER: u32 = 1;
/// Maximum number of layers in the portable array, including the three
/// reserved fallback layers used by the shared texture-slot heap.
pub const MAX_PORTABLE_TEXTURE_LAYERS: u32 = 259;

/// The physical edge of each portable array layer.
pub const fn portable_texture_extent() -> u32 {
    PORTABLE_TEXTURE_PAGE_SIZE + PORTABLE_TEXTURE_GUTTER * 2
}

/// Why an authored source cannot be represented by the portable direct-row
/// path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortableTextureError {
    /// The source format is not the normalized format owned by this atlas.
    UnsupportedFormat(MaterialVtFormat),
    /// The source has more than one array layer.
    UnsupportedLayers(u16),
    /// The source's base mip is larger than one portable row.
    TooLarge { width: u32, height: u32 },
    /// The slot cannot be addressed by the configured layer count.
    SlotOutOfRange { slot: u32, capacity: u32 },
    /// The source page could not be converted into an atlas tile.
    Tile(MaterialVtAtlasUploadError),
    /// The atlas geometry is invalid.
    Atlas(MaterialVirtualTextureAtlasError),
}

/// A validated upload for one portable texture layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PortableTextureUpload {
    /// Destination array layer.
    pub layer: u32,
    /// Copy row pitch, aligned for WebGPU texture writes.
    pub bytes_per_row: u32,
    /// Full guttered tile bytes.
    pub bytes: Vec<u8>,
}

/// Returns whether the source can be represented without decoding or lossy
/// format guessing by the portable normalized atlas.
pub fn supports_portable_texture(source: &MaterialVtSourceSnapshot) -> bool {
    let descriptor = source.descriptor();
    descriptor.format() == MaterialVtFormat::Rgba8Unorm
        && descriptor.layers() == 1
        && descriptor.width() <= PORTABLE_TEXTURE_PAGE_SIZE
        && descriptor.height() <= PORTABLE_TEXTURE_PAGE_SIZE
}

/// Builds the real CPU tile copied into a portable array layer.
pub fn build_portable_texture_upload(
    source: &MaterialVtSourceSnapshot,
    slot: u32,
    capacity: u32,
) -> Result<PortableTextureUpload, PortableTextureError> {
    let descriptor = source.descriptor();
    if descriptor.format() != MaterialVtFormat::Rgba8Unorm {
        return Err(PortableTextureError::UnsupportedFormat(descriptor.format()));
    }
    if descriptor.layers() != 1 {
        return Err(PortableTextureError::UnsupportedLayers(descriptor.layers()));
    }
    if descriptor.width() > PORTABLE_TEXTURE_PAGE_SIZE
        || descriptor.height() > PORTABLE_TEXTURE_PAGE_SIZE
    {
        return Err(PortableTextureError::TooLarge {
            width: descriptor.width(),
            height: descriptor.height(),
        });
    }
    if slot >= capacity {
        return Err(PortableTextureError::SlotOutOfRange { slot, capacity });
    }
    let atlas = MaterialVirtualTextureAtlasConfig::new(
        PORTABLE_TEXTURE_PAGE_SIZE,
        PORTABLE_TEXTURE_GUTTER,
        1,
        capacity,
        MaterialVtFormat::Rgba8Unorm,
    )
    .map_err(PortableTextureError::Atlas)?;
    let key = TexturePageKey {
        texture: descriptor.texture_id(),
        mip: 0,
        layer: 0,
        x: 0,
        y: 0,
    };
    let tile = build_material_vt_atlas_tile(
        source,
        key,
        slot,
        &std::collections::BTreeSet::new(),
        atlas
            .source_tile_config()
            .map_err(PortableTextureError::Tile)?,
    )
    .map_err(PortableTextureError::Tile)?;
    Ok(PortableTextureUpload {
        layer: slot,
        bytes_per_row: tile.bytes_per_row,
        bytes: tile.staging_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_material::{MaterialVtSourceDescriptor, MaterialVtSourceSnapshot};

    fn source(width: u32, height: u32, format: MaterialVtFormat) -> MaterialVtSourceSnapshot {
        let descriptor = MaterialVtSourceDescriptor::new(
            7,
            width,
            height,
            1,
            PORTABLE_TEXTURE_PAGE_SIZE,
            width.max(height).ilog2() as u8 + 1,
            format,
        )
        .unwrap();
        let mut bytes = Vec::new();
        for mip in 0..u32::from(descriptor.mip_count()) {
            let w = (width >> mip).max(1);
            let h = (height >> mip).max(1);
            bytes.extend(std::iter::repeat_n(
                17,
                (w * h) as usize * format.bytes_per_texel() as usize,
            ));
        }
        MaterialVtSourceSnapshot::new(descriptor, bytes).unwrap()
    }

    #[test]
    fn portable_upload_is_a_real_guttered_rgba8_tile() {
        let snapshot = source(4, 2, MaterialVtFormat::Rgba8Unorm);
        let upload = build_portable_texture_upload(&snapshot, 3, MAX_PORTABLE_TEXTURE_LAYERS)
            .expect("small normalized source is portable");
        assert_eq!(upload.layer, 3);
        let expected_row_pitch = (portable_texture_extent() * 4).next_multiple_of(256);
        assert_eq!(upload.bytes_per_row, expected_row_pitch);
        assert_eq!(
            upload.bytes.len(),
            expected_row_pitch as usize * portable_texture_extent() as usize
        );
    }

    #[test]
    fn portable_capability_rejects_large_and_non_normalized_sources() {
        assert!(!supports_portable_texture(&source(
            256,
            128,
            MaterialVtFormat::Rgba8Unorm
        )));
        assert!(!supports_portable_texture(&source(
            4,
            4,
            MaterialVtFormat::Rgba8Srgb
        )));
    }
}
