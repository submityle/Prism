//! Portable, non-bindless material texture rows.
//!
//! A deferred resolve cannot bind one image/sampler pair per material.  This
//! module defines the small, honest portable slice used when binding arrays are
//! unavailable: authored sources are represented by the first complete authored
//! mip that fits one normalized page, converted to RGBA8-unorm, and copied into
//! a fixed `texture_2d_array` layer. The material row carries the layer through
//! the existing index/generation pair; the GPU side additionally publishes the
//! generation only after the tile upload has completed.
//!
//! This is deliberately a mip fallback, not a synthetic color: a large source
//! uses a real lower-resolution authored mip. Multi-layer sources and
//! unpublished slots remain fail-closed; other source formats and page
//! geometries use explicit channel conversion and edge padding. The fallback
//! is bounded to one tile per source, so non-bindless devices do not turn a
//! missing page into an unbounded upload or per-pixel indirection.

use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_render_material::{
    build_material_vt_atlas_tile, MaterialVtAtlasTile, MaterialVtAtlasUploadError, MaterialVtError,
    MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
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

/// One source-backed authored mip used by the portable non-bindless path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PortableTextureFallback {
    /// Exact source page uploaded into the portable tile.
    pub key: TexturePageKey,
    /// Width of the authored mip in texels.
    pub width: u32,
    /// Height of the authored mip in texels.
    pub height: u32,
}

/// Why an authored source cannot be represented by the portable direct-row
/// path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortableTextureError {
    /// The source has more than one array layer.
    UnsupportedLayers(u16),
    /// No complete authored mip fits the portable row contract.
    TooLarge { width: u32, height: u32 },
    /// The slot cannot be addressed by the configured layer count.
    SlotOutOfRange { slot: u32, capacity: u32 },
    /// The source page could not be converted into an atlas tile.
    Tile(MaterialVtAtlasUploadError),
    /// The atlas geometry is invalid.
    Atlas(MaterialVirtualTextureAtlasError),
    /// The normalized source snapshot could not be constructed.
    Source(MaterialVtError),
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

/// Finds the first complete authored mip that fits one portable tile.
///
/// The search is deterministic and monotonically coarsens the source. The
/// source snapshot has already validated the full mip chain, so selecting this
/// page never invents texels or requires a runtime resample.
#[must_use]
pub fn portable_texture_fallback(
    source: &MaterialVtSourceSnapshot,
) -> Option<PortableTextureFallback> {
    let descriptor = source.descriptor();
    if descriptor.layers() != 1 {
        return None;
    }
    // One portable layer represents one complete source page. For sources
    // whose logical page is smaller than the normalized layer, keep that
    // invariant by selecting a mip that fits the source page as well.
    let max_fallback_extent = descriptor.page_size().min(PORTABLE_TEXTURE_PAGE_SIZE);
    for mip in 0..u32::from(descriptor.mip_count()) {
        let width = (descriptor.width() >> mip).max(1);
        let height = (descriptor.height() >> mip).max(1);
        if width <= max_fallback_extent && height <= max_fallback_extent {
            return Some(PortableTextureFallback {
                key: TexturePageKey {
                    texture: descriptor.texture_id(),
                    mip: u8::try_from(mip).ok()?,
                    layer: 0,
                    x: 0,
                    y: 0,
                },
                width,
                height,
            });
        }
    }
    None
}

/// Returns whether the source can be represented by the portable normalized
/// atlas. Conversion is explicit and deterministic: normalized byte formats
/// preserve their channel meaning, sRGB RGB is decoded to linear UNORM8, and
/// float channels are clamped/quantized to the portable atlas range.
pub fn supports_portable_texture(source: &MaterialVtSourceSnapshot) -> bool {
    portable_texture_fallback(source).is_some()
}

/// Builds one normalized RGBA8 tile from the selected authored source mip.
///
/// The returned tile has the portable atlas layout even when the source uses a
/// different format or logical page size. The selected source page is padded
/// by edge extension before the standard gutter builder runs, so every texel
/// in the fallback remains source-backed and no semantic white texel is
/// synthesized for a compatible source.
pub(crate) fn build_portable_texture_tile(
    source: &MaterialVtSourceSnapshot,
    slot: u32,
    capacity: u32,
) -> Result<MaterialVtAtlasTile, PortableTextureError> {
    let descriptor = source.descriptor();
    if descriptor.layers() != 1 {
        return Err(PortableTextureError::UnsupportedLayers(descriptor.layers()));
    }
    if slot >= capacity {
        return Err(PortableTextureError::SlotOutOfRange { slot, capacity });
    }
    let fallback = portable_texture_fallback(source).ok_or(PortableTextureError::TooLarge {
        width: descriptor.width(),
        height: descriptor.height(),
    })?;
    let page = source
        .page(fallback.key)
        .map_err(PortableTextureError::Source)?;
    let metadata = page.metadata();
    let width = usize::try_from(metadata.extent_width)
        .map_err(|_| PortableTextureError::Source(MaterialVtError::PageOffsetOverflow))?;
    let height = usize::try_from(metadata.extent_height)
        .map_err(|_| PortableTextureError::Source(MaterialVtError::PageOffsetOverflow))?;
    let source_bpp = usize::try_from(descriptor.format().bytes_per_texel())
        .map_err(|_| PortableTextureError::Source(MaterialVtError::PayloadSizeOverflow))?;
    if width == 0 || height == 0 {
        return Err(PortableTextureError::Source(
            MaterialVtError::PageOffsetOverflow,
        ));
    }

    let normalized_extent = usize::try_from(PORTABLE_TEXTURE_PAGE_SIZE)
        .map_err(|_| PortableTextureError::Source(MaterialVtError::PayloadSizeOverflow))?;
    let mip_count = normalized_extent.ilog2() as usize + 1;
    let mut bytes = Vec::new();
    let total_bytes = (0..mip_count).try_fold(0usize, |total, mip| {
        let extent = (normalized_extent >> mip).max(1);
        let level_bytes = extent
            .checked_mul(extent)
            .and_then(|value| value.checked_mul(4))
            .ok_or(PortableTextureError::Source(
                MaterialVtError::PayloadSizeOverflow,
            ))?;
        total
            .checked_add(level_bytes)
            .ok_or(PortableTextureError::Source(
                MaterialVtError::PayloadSizeOverflow,
            ))
    })?;
    bytes
        .try_reserve_exact(total_bytes)
        .map_err(|_| PortableTextureError::Source(MaterialVtError::AllocationFailed))?;
    bytes.resize(total_bytes, 0);

    let level_zero_len = normalized_extent
        .checked_mul(normalized_extent)
        .and_then(|value| value.checked_mul(4))
        .ok_or(PortableTextureError::Source(
            MaterialVtError::PayloadSizeOverflow,
        ))?;
    for y in 0..normalized_extent {
        let source_y = y.min(height - 1);
        let row = page
            .row(
                u32::try_from(source_y).map_err(|_| {
                    PortableTextureError::Source(MaterialVtError::PageOffsetOverflow)
                })?,
            )
            .ok_or(PortableTextureError::Source(
                MaterialVtError::PageOffsetOverflow,
            ))?;
        for x in 0..normalized_extent {
            let source_x = x.min(width - 1);
            let source_offset =
                source_x
                    .checked_mul(source_bpp)
                    .ok_or(PortableTextureError::Source(
                        MaterialVtError::PageOffsetOverflow,
                    ))?;
            let source_end =
                source_offset
                    .checked_add(source_bpp)
                    .ok_or(PortableTextureError::Source(
                        MaterialVtError::PageOffsetOverflow,
                    ))?;
            let source_texel =
                row.bytes
                    .get(source_offset..source_end)
                    .ok_or(PortableTextureError::Source(
                        MaterialVtError::PayloadLengthMismatch {
                            expected: u64::from(metadata.row_bytes),
                            actual: u64::try_from(row.bytes.len()).unwrap_or(u64::MAX),
                        },
                    ))?;
            let destination = y
                .checked_mul(normalized_extent)
                .and_then(|value| value.checked_add(x))
                .and_then(|value| value.checked_mul(4))
                .ok_or(PortableTextureError::Source(
                    MaterialVtError::PayloadSizeOverflow,
                ))?;
            convert_texel_to_rgba8(
                descriptor.format(),
                source_texel,
                &mut bytes[destination..destination + 4],
            );
        }
    }
    debug_assert_eq!(level_zero_len, normalized_extent * normalized_extent * 4);

    let normalized_descriptor = MaterialVtSourceDescriptor::new(
        descriptor.texture_id(),
        PORTABLE_TEXTURE_PAGE_SIZE,
        PORTABLE_TEXTURE_PAGE_SIZE,
        1,
        PORTABLE_TEXTURE_PAGE_SIZE,
        u8::try_from(mip_count)
            .map_err(|_| PortableTextureError::Source(MaterialVtError::PayloadSizeOverflow))?,
        MaterialVtFormat::Rgba8Unorm,
    )
    .map_err(PortableTextureError::Source)?;
    let normalized = MaterialVtSourceSnapshot::new(normalized_descriptor, bytes)
        .map_err(PortableTextureError::Source)?;
    let normalized_key = TexturePageKey {
        texture: descriptor.texture_id(),
        mip: 0,
        layer: 0,
        x: 0,
        y: 0,
    };
    let atlas = MaterialVirtualTextureAtlasConfig::new(
        PORTABLE_TEXTURE_PAGE_SIZE,
        PORTABLE_TEXTURE_GUTTER,
        1,
        capacity,
        MaterialVtFormat::Rgba8Unorm,
    )
    .map_err(PortableTextureError::Atlas)?;
    let mut tile = build_material_vt_atlas_tile(
        &normalized,
        normalized_key,
        slot,
        &std::collections::BTreeSet::new(),
        atlas
            .source_tile_config()
            .map_err(PortableTextureError::Tile)?,
    )
    .map_err(PortableTextureError::Tile)?;
    tile.key = fallback.key;
    Ok(tile)
}

/// Builds the real CPU tile copied into a portable array layer.
pub fn build_portable_texture_upload(
    source: &MaterialVtSourceSnapshot,
    slot: u32,
    capacity: u32,
) -> Result<PortableTextureUpload, PortableTextureError> {
    let tile = build_portable_texture_tile(source, slot, capacity)?;
    Ok(PortableTextureUpload {
        layer: slot,
        bytes_per_row: tile.bytes_per_row,
        bytes: tile.staging_bytes,
    })
}

fn convert_texel_to_rgba8(format: MaterialVtFormat, source: &[u8], destination: &mut [u8]) {
    debug_assert_eq!(destination.len(), 4);
    match format {
        MaterialVtFormat::R8Unorm => {
            destination.copy_from_slice(&[source[0], 0, 0, u8::MAX]);
        }
        MaterialVtFormat::Rg8Unorm => {
            destination.copy_from_slice(&[source[0], source[1], 0, u8::MAX]);
        }
        MaterialVtFormat::Rgba8Unorm => destination.copy_from_slice(source),
        MaterialVtFormat::Rgba8Srgb => {
            destination[0] = quantize_unorm8(srgb_to_linear(f32::from(source[0]) / 255.0));
            destination[1] = quantize_unorm8(srgb_to_linear(f32::from(source[1]) / 255.0));
            destination[2] = quantize_unorm8(srgb_to_linear(f32::from(source[2]) / 255.0));
            destination[3] = source[3];
        }
        MaterialVtFormat::Rgba16Float => {
            for channel in 0..4 {
                let offset = channel * 2;
                let bits = u16::from_le_bytes([source[offset], source[offset + 1]]);
                destination[channel] = quantize_unorm8(f16_to_f32(bits));
            }
        }
        MaterialVtFormat::Rgba32Float => {
            for channel in 0..4 {
                let offset = channel * 4;
                destination[channel] = quantize_unorm8(f32::from_le_bytes([
                    source[offset],
                    source[offset + 1],
                    source[offset + 2],
                    source[offset + 3],
                ]));
            }
        }
    }
}

fn quantize_unorm8(value: f32) -> u8 {
    if value.is_nan() {
        return 0;
    }
    (value.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut fraction = fraction;
                let mut exponent = 0u32;
                while fraction & 0x0400 == 0 {
                    fraction <<= 1;
                    exponent += 1;
                }
                let fraction = (fraction & 0x03ff) << 13;
                sign | ((127 - 14 - exponent) << 23) | fraction
            }
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        exponent => sign | ((exponent + (127 - 15)) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
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
    fn large_source_uses_a_real_authored_mip_for_non_bindless_fallback() {
        let snapshot = source(256, 128, MaterialVtFormat::Rgba8Unorm);
        let fallback = portable_texture_fallback(&snapshot).expect("authored mip fits");
        assert_eq!(fallback.key.mip, 1);
        assert_eq!((fallback.width, fallback.height), (128, 64));

        let upload = build_portable_texture_upload(&snapshot, 3, MAX_PORTABLE_TEXTURE_LAYERS)
            .expect("large source has a source-backed fallback tile");
        assert!(!upload.bytes.is_empty());
    }

    #[test]
    fn portable_fallback_normalizes_incompatible_page_geometry() {
        let original = source(4, 4, MaterialVtFormat::Rgba8Unorm);
        assert!(supports_portable_texture(&original));
        let descriptor =
            MaterialVtSourceDescriptor::new(7, 4, 4, 1, 4, 3, MaterialVtFormat::Rgba8Unorm)
                .unwrap();
        let incompatible =
            MaterialVtSourceSnapshot::new(descriptor, vec![17; (16 + 4 + 1) * 4]).unwrap();
        assert!(supports_portable_texture(&incompatible));
        let fallback = portable_texture_fallback(&incompatible).expect("coarsest source page");
        assert_eq!(fallback.key.mip, 0);
        assert_eq!(fallback.width, 4);
        let upload = build_portable_texture_upload(&incompatible, 3, MAX_PORTABLE_TEXTURE_LAYERS)
            .expect("incompatible page geometry is normalized");
        assert!(!upload.bytes.is_empty());
    }

    #[test]
    fn portable_fallback_normalizes_srgb_and_float_sources() {
        assert!(supports_portable_texture(&source(
            4,
            4,
            MaterialVtFormat::Rgba8Srgb
        )));
        assert!(supports_portable_texture(&source(
            4,
            4,
            MaterialVtFormat::Rgba16Float
        )));
        assert!(supports_portable_texture(&source(
            4,
            4,
            MaterialVtFormat::Rgba32Float
        )));
    }
}
