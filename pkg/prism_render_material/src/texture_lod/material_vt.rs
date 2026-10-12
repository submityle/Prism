//! Immutable CPU source and page extraction for material virtual textures.
//!
//! This module is deliberately a source contract, not a renderer. A
//! [`MaterialVtSourceSnapshot`] owns one immutable byte snapshot containing a
//! complete mip chain. It derives deterministic mip metadata and exposes each
//! virtual page as row views into that snapshot. No texels are generated,
//! padded, decoded, or uploaded here; a backend can turn the rows into an
//! atlas upload after applying its own gutter/encoding policy.
//!
//! The byte layout is mip-major, then array-layer-major, then row-major:
//!
//! ```text
//! mip 0: layer 0 rows, layer 1 rows, ...
//! mip 1: layer 0 rows, layer 1 rows, ...
//! ...
//! ```
//!
//! Every mip uses `max(1, width >> mip)` by `max(1, height >> mip)` texels.
//! The descriptor requires the complete pyramid implied by
//! [`VirtualTexture::max_mip`](super::VirtualTexture::max_mip), so its page
//! keys and the existing material LOD/page-lookup contracts describe the same
//! address space.

use alloc::{sync::Arc, vec::Vec};
use core::fmt;

use prism_render_architecture::texture_streaming::TexturePageKey;

use super::{PageRequest, VirtualTexture};
use crate::{GpuMaterialTexture, NO_VIRTUAL_TEXTURE};

/// Uncompressed source formats understood by the CPU page contract.
///
/// The snapshot stores the source bytes verbatim. Multi-byte formats use
/// little-endian component bytes, but this module never reinterprets them.
/// Block-compressed formats and GPU-native opaque handles are intentionally not
/// accepted: a caller must decode or otherwise materialize such data before
/// constructing a snapshot.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum MaterialVtFormat {
    /// One unsigned normalized byte per texel.
    R8Unorm = 0,
    /// Two unsigned normalized bytes per texel.
    Rg8Unorm = 1,
    /// Four unsigned normalized bytes per texel.
    Rgba8Unorm = 2,
    /// Four sRGB bytes per texel.
    Rgba8Srgb = 3,
    /// Four IEEE-754 binary16 components per texel.
    Rgba16Float = 4,
    /// Four IEEE-754 binary32 components per texel.
    Rgba32Float = 5,
}

impl MaterialVtFormat {
    /// Number of source bytes occupied by one texel.
    #[must_use]
    pub const fn bytes_per_texel(self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm => 2,
            Self::Rgba8Unorm | Self::Rgba8Srgb => 4,
            Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
        }
    }

    /// Returns whether the format carries an sRGB transfer function.
    #[must_use]
    pub const fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8Srgb)
    }
}

/// Validated dimensions and byte-layout identity for one material VT source.
///
/// `mip_count` must equal the complete pyramid count implied by the largest
/// base dimension. `layers` is the array-layer count represented in the source
/// byte payload and in [`TexturePageKey::layer`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialVtSourceDescriptor {
    texture_id: u32,
    width: u32,
    height: u32,
    layers: u16,
    page_size: u32,
    mip_count: u8,
    format: MaterialVtFormat,
}

impl MaterialVtSourceDescriptor {
    /// Validates and creates a source descriptor.
    ///
    /// The source must contain every mip down to a one-texel largest axis. A
    /// truncated chain is rejected because [`VirtualTexture`] and the material
    /// page lookup contract would otherwise be able to request pages absent
    /// from the source.
    pub fn new(
        texture_id: u32,
        width: u32,
        height: u32,
        layers: u16,
        page_size: u32,
        mip_count: u8,
        format: MaterialVtFormat,
    ) -> Result<Self, MaterialVtError> {
        if texture_id == NO_VIRTUAL_TEXTURE {
            return Err(MaterialVtError::ReservedTextureId);
        }
        if width == 0 || height == 0 {
            return Err(MaterialVtError::ZeroDimension);
        }
        if layers == 0 {
            return Err(MaterialVtError::ZeroLayerCount);
        }
        if page_size == 0 {
            return Err(MaterialVtError::ZeroPageSize);
        }

        let max_axis = width.max(height);
        let expected_mip_count = max_axis.ilog2() + 1;
        if u32::from(mip_count) != expected_mip_count {
            return Err(MaterialVtError::MipCountMismatch {
                expected: expected_mip_count,
                actual: mip_count,
            });
        }

        for mip in 0..expected_mip_count {
            let mip_width = mip_dimension(width, mip);
            let mip_height = mip_dimension(height, mip);
            let pages_x = mip_width.div_ceil(page_size);
            let pages_y = mip_height.div_ceil(page_size);
            if pages_x > u32::from(u16::MAX) + 1 || pages_y > u32::from(u16::MAX) + 1 {
                return Err(MaterialVtError::PageGridTooLarge {
                    mip: mip as u8,
                    pages_x,
                    pages_y,
                });
            }
            // The row pitch is exposed as a `u32` copy-friendly value. Reject
            // an unrepresentable source instead of truncating it.
            if u64::from(mip_width) * u64::from(format.bytes_per_texel()) > u64::from(u32::MAX) {
                return Err(MaterialVtError::RowPitchOverflow { mip: mip as u8 });
            }
        }

        Ok(Self {
            texture_id,
            width,
            height,
            layers,
            page_size,
            mip_count,
            format,
        })
    }

    /// Stable texture identity copied into [`TexturePageKey::texture`] and the
    /// VT-enabled [`GpuMaterialTexture::virtual_texture_id`] field.
    #[must_use]
    pub const fn texture_id(self) -> u32 {
        self.texture_id
    }

    /// Base mip width in texels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Base mip height in texels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Number of array layers/faces in the source.
    #[must_use]
    pub const fn layers(self) -> u16 {
        self.layers
    }

    /// Square virtual page edge in texels.
    #[must_use]
    pub const fn page_size(self) -> u32 {
        self.page_size
    }

    /// Number of mip levels in the complete source chain.
    #[must_use]
    pub const fn mip_count(self) -> u8 {
        self.mip_count
    }

    /// Source texel format.
    #[must_use]
    pub const fn format(self) -> MaterialVtFormat {
        self.format
    }
}

/// One deterministic mip-level layout inside a source snapshot.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialVtMipMetadata {
    /// Mip index, where zero is finest.
    pub mip: u8,
    /// Mip width in texels.
    pub width: u32,
    /// Mip height in texels.
    pub height: u32,
    /// Number of pages along X.
    pub pages_x: u32,
    /// Number of pages along Y.
    pub pages_y: u32,
    /// Tight source row pitch in bytes.
    pub row_pitch_bytes: u32,
    /// Bytes between the first byte of one layer and the next at this mip.
    pub layer_stride_bytes: u64,
    /// Bytes between the first byte of this mip and the next mip.
    pub level_stride_bytes: u64,
    /// Absolute byte offset of the first layer in this mip.
    pub offset_bytes: u64,
}

/// Metadata for one virtual page, without borrowing its payload.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialVtPageMetadata {
    /// Shared streaming key for this page.
    pub key: TexturePageKey,
    /// Mip dimensions containing this page.
    pub mip_width: u32,
    /// Mip height containing this page.
    pub mip_height: u32,
    /// Top-left texel origin in the mip.
    pub origin_x: u32,
    /// Top-left texel origin in the mip.
    pub origin_y: u32,
    /// Actual width of the page at the right edge.
    pub extent_width: u32,
    /// Actual height of the page at the bottom edge.
    pub extent_height: u32,
    /// Tight source row pitch for the containing mip.
    pub source_row_pitch_bytes: u32,
    /// Bytes in one extracted page row.
    pub row_bytes: u32,
    /// Absolute byte offset of the first extracted row.
    pub first_row_offset_bytes: u64,
    /// Number of source rows in this page.
    pub row_count: u32,
    /// Bytes a caller receives when concatenating the extracted rows.
    pub extracted_byte_len: u64,
}

/// One borrowed source row of a virtual page.
///
/// The row is borrowed from the immutable [`MaterialVtSourceSnapshot`]. Its
/// lifetime cannot outlive that snapshot, and the `bytes` slice contains only
/// source texels: no clamp edge, gutter, compression, or synthesized data is
/// added.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVtPageRow<'a> {
    /// Zero-based row within the extracted page.
    pub row: u32,
    /// Absolute byte offset in the source snapshot.
    pub source_offset_bytes: u64,
    /// Tight source bytes for this row's page extent.
    pub bytes: &'a [u8],
}

/// A page view borrowing bytes from an immutable source snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVtPage<'a> {
    metadata: MaterialVtPageMetadata,
    source: &'a [u8],
}

impl<'a> MaterialVtPage<'a> {
    /// Page metadata used by an upload planner.
    #[must_use]
    pub const fn metadata(self) -> MaterialVtPageMetadata {
        self.metadata
    }

    /// Returns one extracted row, or `None` outside the page's row range.
    #[must_use]
    pub fn row(self, row: u32) -> Option<MaterialVtPageRow<'a>> {
        if row >= self.metadata.row_count {
            return None;
        }
        let row_pitch = usize::try_from(self.metadata.source_row_pitch_bytes).ok()?;
        let row_bytes = usize::try_from(self.metadata.row_bytes).ok()?;
        let row_index = usize::try_from(row).ok()?;
        let start = row_index.checked_mul(row_pitch)?;
        let end = start.checked_add(row_bytes)?;
        let bytes = self.source.get(start..end)?;
        Some(MaterialVtPageRow {
            row,
            source_offset_bytes: self
                .metadata
                .first_row_offset_bytes
                .checked_add(u64::from(row) * u64::from(self.metadata.source_row_pitch_bytes))?,
            bytes,
        })
    }

    /// Iterates the page's rows in ascending source order.
    pub fn rows(self) -> MaterialVtPageRows<'a> {
        MaterialVtPageRows {
            page: self,
            next: 0,
        }
    }
}

/// Iterator over the borrowed rows of a [`MaterialVtPage`].
#[derive(Clone, Copy, Debug)]
pub struct MaterialVtPageRows<'a> {
    page: MaterialVtPage<'a>,
    next: u32,
}

impl<'a> Iterator for MaterialVtPageRows<'a> {
    type Item = MaterialVtPageRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let row = self.page.row(self.next)?;
        self.next = self.next.saturating_add(1);
        Some(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.page.metadata.row_count.saturating_sub(self.next);
        let remaining = usize::try_from(remaining).unwrap_or(usize::MAX);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for MaterialVtPageRows<'_> {}

/// CPU-side metadata paired with the existing GPU material texture-row ABI.
///
/// `gpu` is the only value that belongs in the existing flat material texture
/// row heap. The remaining fields let a source/page producer validate and
/// schedule pages without recovering information from the packed GPU row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVtRow {
    /// ABI row to publish in the material texture heap.
    pub gpu: GpuMaterialTexture,
    /// Source format required to interpret page bytes.
    pub format: MaterialVtFormat,
    /// Number of array layers in the source.
    pub layers: u16,
    /// Number of complete mip levels in the source.
    pub mip_count: u8,
    /// Total number of virtual pages across all mips and layers.
    pub page_count: u64,
}

/// Immutable material VT source snapshot and its deterministic extraction
/// metadata.
///
/// Cloning a snapshot clones the `Arc` handle, not mutable access to its bytes.
/// All [`MaterialVtPage`] views borrow the snapshot for the duration of the
/// view, so a producer may safely retain a page view only while retaining the
/// source snapshot as well.
#[derive(Clone, Debug)]
pub struct MaterialVtSourceSnapshot {
    descriptor: MaterialVtSourceDescriptor,
    bytes: Arc<[u8]>,
    mips: Vec<MaterialVtMipMetadata>,
    total_pages: u64,
}

impl MaterialVtSourceSnapshot {
    /// Validates an immutable, complete mip-chain payload and builds its
    /// deterministic metadata.
    pub fn new(
        descriptor: MaterialVtSourceDescriptor,
        bytes: impl Into<Arc<[u8]>>,
    ) -> Result<Self, MaterialVtError> {
        let bytes = bytes.into();
        let bpp = u64::from(descriptor.format.bytes_per_texel());
        let mut offset = 0u64;
        let mut total_pages = 0u64;
        let mut mips = Vec::with_capacity(usize::from(descriptor.mip_count));

        for mip in 0..u32::from(descriptor.mip_count) {
            let width = mip_dimension(descriptor.width, mip);
            let height = mip_dimension(descriptor.height, mip);
            let row_pitch = u64::from(width)
                .checked_mul(bpp)
                .ok_or(MaterialVtError::PayloadSizeOverflow)?;
            let layer_stride = row_pitch
                .checked_mul(u64::from(height))
                .ok_or(MaterialVtError::PayloadSizeOverflow)?;
            let level_stride = layer_stride
                .checked_mul(u64::from(descriptor.layers))
                .ok_or(MaterialVtError::PayloadSizeOverflow)?;
            let pages_x = width.div_ceil(descriptor.page_size);
            let pages_y = height.div_ceil(descriptor.page_size);
            let page_count = u64::from(pages_x)
                .checked_mul(u64::from(pages_y))
                .and_then(|v| v.checked_mul(u64::from(descriptor.layers)))
                .ok_or(MaterialVtError::PageCountOverflow)?;
            total_pages = total_pages
                .checked_add(page_count)
                .ok_or(MaterialVtError::PageCountOverflow)?;
            let end = offset
                .checked_add(level_stride)
                .ok_or(MaterialVtError::PayloadSizeOverflow)?;

            mips.push(MaterialVtMipMetadata {
                mip: mip as u8,
                width,
                height,
                pages_x,
                pages_y,
                row_pitch_bytes: u32::try_from(row_pitch).expect("descriptor validates row pitch"),
                layer_stride_bytes: layer_stride,
                level_stride_bytes: level_stride,
                offset_bytes: offset,
            });
            offset = end;
        }

        let actual =
            u64::try_from(bytes.len()).map_err(|_| MaterialVtError::PayloadSizeOverflow)?;
        if actual != offset {
            return Err(MaterialVtError::PayloadLengthMismatch {
                expected: offset,
                actual,
            });
        }

        Ok(Self {
            descriptor,
            bytes,
            mips,
            total_pages,
        })
    }

    /// Source descriptor shared by all page keys and row metadata.
    #[must_use]
    pub const fn descriptor(&self) -> MaterialVtSourceDescriptor {
        self.descriptor
    }

    /// The immutable source bytes, in the descriptor's mip-major layout.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Deterministic mip metadata, ordered from finest to coarsest.
    #[must_use]
    pub fn mips(&self) -> &[MaterialVtMipMetadata] {
        &self.mips
    }

    /// Number of virtual pages across all mips and layers.
    #[must_use]
    pub const fn page_count(&self) -> u64 {
        self.total_pages
    }

    /// The same LOD/page geometry used by material `texture_lod` requests.
    #[must_use]
    pub fn virtual_texture(&self) -> VirtualTexture {
        VirtualTexture::new(
            self.descriptor.width,
            self.descriptor.height,
            self.descriptor.page_size,
        )
    }

    /// Creates the ABI row and retains the source-side format/count metadata.
    #[must_use]
    pub fn texture_row(
        &self,
        bindless_index: u32,
        generation: u32,
        semantic: u32,
        sampler_index: u32,
    ) -> MaterialVtRow {
        MaterialVtRow {
            gpu: GpuMaterialTexture::with_virtual_texture(
                bindless_index,
                generation,
                semantic,
                sampler_index,
                self.descriptor.texture_id,
                self.descriptor.width,
                self.descriptor.height,
                self.descriptor.page_size,
            ),
            format: self.descriptor.format,
            layers: self.descriptor.layers,
            mip_count: self.descriptor.mip_count,
            page_count: self.total_pages,
        }
    }

    /// Converts an existing texture-Lod [`PageRequest`] into the shared page
    /// key without truncating an unrepresentable coordinate.
    pub fn key_for_request(
        &self,
        layer: u16,
        request: PageRequest,
    ) -> Result<TexturePageKey, MaterialVtError> {
        let mip = u8::try_from(request.mip)
            .map_err(|_| MaterialVtError::MipOutOfRange { mip: request.mip })?;
        let x = u16::try_from(request.page_x).map_err(|_| MaterialVtError::PageXOutOfRange {
            mip,
            page_x: request.page_x,
        })?;
        let y = u16::try_from(request.page_y).map_err(|_| MaterialVtError::PageYOutOfRange {
            mip,
            page_y: request.page_y,
        })?;
        let key = TexturePageKey {
            texture: self.descriptor.texture_id,
            mip,
            layer,
            x,
            y,
        };
        self.validate_key(key)?;
        Ok(key)
    }

    /// Returns deterministic page keys in the same order as the streaming
    /// table: mip, layer, X, then Y.
    pub fn page_keys(&self) -> MaterialVtPageKeyIter<'_> {
        MaterialVtPageKeyIter::new(self)
    }

    /// Returns deterministic page metadata in page-key order.
    pub fn page_metadata_iter(&self) -> impl Iterator<Item = MaterialVtPageMetadata> + '_ {
        self.page_keys()
            .map(|key| self.page_metadata(key).expect("validated page key"))
    }

    /// Extracts metadata for one page without copying its bytes.
    pub fn page_metadata(
        &self,
        key: TexturePageKey,
    ) -> Result<MaterialVtPageMetadata, MaterialVtError> {
        let mip = self.validate_key(key)?;
        let bpp = u64::from(self.descriptor.format.bytes_per_texel());
        let origin_x = u32::from(key.x)
            .checked_mul(self.descriptor.page_size)
            .ok_or(MaterialVtError::PageOffsetOverflow)?;
        let origin_y = u32::from(key.y)
            .checked_mul(self.descriptor.page_size)
            .ok_or(MaterialVtError::PageOffsetOverflow)?;
        let extent_width = self.descriptor.page_size.min(mip.width - origin_x);
        let extent_height = self.descriptor.page_size.min(mip.height - origin_y);
        let row_bytes = u64::from(extent_width)
            .checked_mul(bpp)
            .ok_or(MaterialVtError::PageOffsetOverflow)?;
        let layer_offset = mip
            .offset_bytes
            .checked_add(u64::from(key.layer) * mip.layer_stride_bytes)
            .ok_or(MaterialVtError::PageOffsetOverflow)?;
        let first_row = layer_offset
            .checked_add(u64::from(origin_y) * u64::from(mip.row_pitch_bytes))
            .and_then(|v| v.checked_add(u64::from(origin_x) * bpp))
            .ok_or(MaterialVtError::PageOffsetOverflow)?;
        let extracted_byte_len = row_bytes
            .checked_mul(u64::from(extent_height))
            .ok_or(MaterialVtError::PageOffsetOverflow)?;

        Ok(MaterialVtPageMetadata {
            key,
            mip_width: mip.width,
            mip_height: mip.height,
            origin_x,
            origin_y,
            extent_width,
            extent_height,
            source_row_pitch_bytes: mip.row_pitch_bytes,
            row_bytes: u32::try_from(row_bytes).map_err(|_| MaterialVtError::PageOffsetOverflow)?,
            first_row_offset_bytes: first_row,
            row_count: extent_height,
            extracted_byte_len,
        })
    }

    /// Borrows one page's source rows. The returned view cannot outlive `self`.
    pub fn page(&self, key: TexturePageKey) -> Result<MaterialVtPage<'_>, MaterialVtError> {
        let metadata = self.page_metadata(key)?;
        let start = usize::try_from(metadata.first_row_offset_bytes)
            .map_err(|_| MaterialVtError::PayloadSizeOverflow)?;
        let row_pitch = usize::try_from(metadata.source_row_pitch_bytes)
            .map_err(|_| MaterialVtError::PayloadSizeOverflow)?;
        let rows = usize::try_from(metadata.row_count)
            .map_err(|_| MaterialVtError::PayloadSizeOverflow)?;
        let last_row = rows
            .checked_sub(1)
            .ok_or(MaterialVtError::PageOffsetOverflow)?;
        let end = start
            .checked_add(
                last_row
                    .checked_mul(row_pitch)
                    .ok_or(MaterialVtError::PageOffsetOverflow)?,
            )
            .and_then(|v| v.checked_add(usize::try_from(metadata.row_bytes).ok()?))
            .ok_or(MaterialVtError::PageOffsetOverflow)?;
        let source = self
            .bytes
            .get(start..end)
            .ok_or(MaterialVtError::PayloadLengthMismatch {
                expected: end as u64,
                actual: self.bytes.len() as u64,
            })?;
        Ok(MaterialVtPage { metadata, source })
    }

    fn validate_key(&self, key: TexturePageKey) -> Result<&MaterialVtMipMetadata, MaterialVtError> {
        if key.texture != self.descriptor.texture_id {
            return Err(MaterialVtError::TextureIdMismatch {
                expected: self.descriptor.texture_id,
                actual: key.texture,
            });
        }
        if key.layer >= self.descriptor.layers {
            return Err(MaterialVtError::LayerOutOfRange { layer: key.layer });
        }
        let mip = self
            .mips
            .get(usize::from(key.mip))
            .ok_or(MaterialVtError::MipOutOfRange {
                mip: u32::from(key.mip),
            })?;
        if u32::from(key.x) >= mip.pages_x {
            return Err(MaterialVtError::PageXOutOfRange {
                mip: key.mip,
                page_x: u32::from(key.x),
            });
        }
        if u32::from(key.y) >= mip.pages_y {
            return Err(MaterialVtError::PageYOutOfRange {
                mip: key.mip,
                page_y: u32::from(key.y),
            });
        }
        Ok(mip)
    }
}

/// Deterministic page-key iterator for a source snapshot.
#[derive(Clone, Debug)]
pub struct MaterialVtPageKeyIter<'a> {
    source: &'a MaterialVtSourceSnapshot,
    mip: usize,
    layer: u16,
    x: u32,
    y: u32,
}

impl<'a> MaterialVtPageKeyIter<'a> {
    fn new(source: &'a MaterialVtSourceSnapshot) -> Self {
        Self {
            source,
            mip: 0,
            layer: 0,
            x: 0,
            y: 0,
        }
    }
}

impl Iterator for MaterialVtPageKeyIter<'_> {
    type Item = TexturePageKey;

    fn next(&mut self) -> Option<Self::Item> {
        let metadata = *self.source.mips.get(self.mip)?;
        let key = TexturePageKey {
            texture: self.source.descriptor.texture_id,
            mip: metadata.mip,
            layer: self.layer,
            x: u16::try_from(self.x).expect("descriptor validates page-key range"),
            y: u16::try_from(self.y).expect("descriptor validates page-key range"),
        };

        self.y += 1;
        if self.y >= metadata.pages_y {
            self.y = 0;
            self.x += 1;
            if self.x >= metadata.pages_x {
                self.x = 0;
                self.layer += 1;
                if self.layer >= self.source.descriptor.layers {
                    self.layer = 0;
                    self.mip = self.mip.saturating_add(1);
                }
            }
        }
        Some(key)
    }
}

/// Failure while validating a source snapshot or addressing a page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialVtError {
    /// Base width or height was zero.
    ZeroDimension,
    /// The texture identity is reserved as the non-VT ABI sentinel.
    ReservedTextureId,
    /// The source had no array layer.
    ZeroLayerCount,
    /// The page edge was zero.
    ZeroPageSize,
    /// The supplied chain did not contain the complete implied mip pyramid.
    MipCountMismatch {
        /// Required complete mip count.
        expected: u32,
        /// Supplied mip count.
        actual: u8,
    },
    /// A mip's page grid cannot be represented by the shared `u16` key fields.
    PageGridTooLarge {
        /// Mip whose grid overflowed.
        mip: u8,
        /// Page columns requested.
        pages_x: u32,
        /// Page rows requested.
        pages_y: u32,
    },
    /// A tight row pitch cannot be represented by row metadata.
    RowPitchOverflow { mip: u8 },
    /// Total source byte arithmetic overflowed.
    PayloadSizeOverflow,
    /// Total virtual page arithmetic overflowed.
    PageCountOverflow,
    /// The immutable payload length did not match the descriptor's exact layout.
    PayloadLengthMismatch { expected: u64, actual: u64 },
    /// A page key belongs to another source identity.
    TextureIdMismatch { expected: u32, actual: u32 },
    /// A layer is outside the descriptor.
    LayerOutOfRange { layer: u16 },
    /// A mip is outside the descriptor.
    MipOutOfRange { mip: u32 },
    /// A page column is outside the mip grid.
    PageXOutOfRange { mip: u8, page_x: u32 },
    /// A page row is outside the mip grid.
    PageYOutOfRange { mip: u8, page_y: u32 },
    /// Page origin/length arithmetic overflowed.
    PageOffsetOverflow,
}

impl fmt::Display for MaterialVtError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => formatter.write_str("material VT dimensions must be nonzero"),
            Self::ReservedTextureId => {
                formatter.write_str("material VT texture id is reserved for non-VT rows")
            }
            Self::ZeroLayerCount => formatter.write_str("material VT layer count must be nonzero"),
            Self::ZeroPageSize => formatter.write_str("material VT page size must be nonzero"),
            Self::MipCountMismatch { expected, actual } => write!(
                formatter,
                "material VT requires {expected} complete mips, got {actual}"
            ),
            Self::PageGridTooLarge {
                mip,
                pages_x,
                pages_y,
            } => write!(
                formatter,
                "material VT mip {mip} grid {pages_x}x{pages_y} exceeds key range"
            ),
            Self::RowPitchOverflow { mip } => {
                write!(formatter, "material VT mip {mip} row pitch overflows u32")
            }
            Self::PayloadSizeOverflow => formatter.write_str("material VT payload size overflows"),
            Self::PageCountOverflow => formatter.write_str("material VT page count overflows"),
            Self::PayloadLengthMismatch { expected, actual } => write!(
                formatter,
                "material VT payload length mismatch: expected {expected}, got {actual}"
            ),
            Self::TextureIdMismatch { expected, actual } => {
                write!(
                    formatter,
                    "material VT texture id mismatch: expected {expected}, got {actual}"
                )
            }
            Self::LayerOutOfRange { layer } => {
                write!(formatter, "material VT layer {layer} is out of range")
            }
            Self::MipOutOfRange { mip } => {
                write!(formatter, "material VT mip {mip} is out of range")
            }
            Self::PageXOutOfRange { mip, page_x } => {
                write!(
                    formatter,
                    "material VT mip {mip} page x {page_x} is out of range"
                )
            }
            Self::PageYOutOfRange { mip, page_y } => {
                write!(
                    formatter,
                    "material VT mip {mip} page y {page_y} is out of range"
                )
            }
            Self::PageOffsetOverflow => formatter.write_str("material VT page offset overflows"),
        }
    }
}

impl std::error::Error for MaterialVtError {}

fn mip_dimension(dimension: u32, mip: u32) -> u32 {
    let shifted = dimension.checked_shr(mip).unwrap_or(0);
    if shifted == 0 {
        1
    } else {
        shifted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(
        width: u32,
        height: u32,
        layers: u16,
        page_size: u32,
        mip_count: u8,
    ) -> MaterialVtSourceDescriptor {
        MaterialVtSourceDescriptor::new(
            41,
            width,
            height,
            layers,
            page_size,
            mip_count,
            MaterialVtFormat::Rgba8Unorm,
        )
        .unwrap()
    }

    fn payload(descriptor: MaterialVtSourceDescriptor) -> Vec<u8> {
        let mut bytes = Vec::new();
        for mip in 0..u32::from(descriptor.mip_count()) {
            let width = mip_dimension(descriptor.width(), mip);
            let height = mip_dimension(descriptor.height(), mip);
            let level_bytes = usize::try_from(width).unwrap()
                * usize::try_from(height).unwrap()
                * usize::from(descriptor.layers())
                * usize::try_from(descriptor.format().bytes_per_texel()).unwrap();
            let base = bytes.len() as u8;
            bytes.extend((0..level_bytes).map(|offset| base.wrapping_add(offset as u8)));
        }
        bytes
    }

    #[test]
    fn descriptor_rejects_incomplete_chain_and_zero_inputs() {
        assert_eq!(
            MaterialVtSourceDescriptor::new(1, 0, 4, 1, 2, 3, MaterialVtFormat::R8Unorm),
            Err(MaterialVtError::ZeroDimension)
        );
        assert_eq!(
            MaterialVtSourceDescriptor::new(1, 4, 4, 1, 2, 2, MaterialVtFormat::R8Unorm),
            Err(MaterialVtError::MipCountMismatch {
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            MaterialVtSourceDescriptor::new(
                NO_VIRTUAL_TEXTURE,
                4,
                4,
                1,
                2,
                3,
                MaterialVtFormat::R8Unorm
            ),
            Err(MaterialVtError::ReservedTextureId)
        );
    }

    #[test]
    fn snapshot_has_exact_deterministic_mip_and_key_order() {
        let descriptor = descriptor(4, 3, 2, 2, 3);
        let snapshot = MaterialVtSourceSnapshot::new(descriptor, payload(descriptor)).unwrap();
        assert_eq!(snapshot.mips()[0].width, 4);
        assert_eq!(snapshot.mips()[0].height, 3);
        assert_eq!(snapshot.mips()[0].pages_x, 2);
        assert_eq!(snapshot.mips()[0].pages_y, 2);
        assert_eq!(snapshot.mips()[1].width, 2);
        assert_eq!(snapshot.mips()[1].height, 1);
        assert_eq!(snapshot.mips()[2].width, 1);
        assert_eq!(snapshot.page_count(), 12);

        let keys: Vec<_> = snapshot.page_keys().collect();
        assert_eq!(keys.len(), 12);
        assert!(keys.windows(2).all(|window| window[0] < window[1]));
        assert_eq!(
            keys[0],
            TexturePageKey {
                texture: 41,
                mip: 0,
                layer: 0,
                x: 0,
                y: 0
            }
        );
        assert_eq!(
            keys[3],
            TexturePageKey {
                texture: 41,
                mip: 0,
                layer: 0,
                x: 1,
                y: 1
            }
        );
        assert_eq!(keys[4].layer, 1);
        assert_eq!(keys[8].mip, 1);
        assert_eq!(keys[11].mip, 2);
    }

    #[test]
    fn edge_page_exposes_only_real_source_rows() {
        let descriptor = descriptor(4, 3, 1, 2, 3);
        let snapshot = MaterialVtSourceSnapshot::new(descriptor, payload(descriptor)).unwrap();
        let key = TexturePageKey {
            texture: 41,
            mip: 0,
            layer: 0,
            x: 1,
            y: 1,
        };
        let page = snapshot.page(key).unwrap();
        assert_eq!(page.metadata().extent_width, 2);
        assert_eq!(page.metadata().extent_height, 1);
        assert_eq!(page.metadata().row_bytes, 8);
        let rows: Vec<_> = page.rows().collect();
        assert_eq!(rows.len(), 1);
        // Base mip starts at zero, row 2 starts at 32, and x=2 starts at 8.
        assert_eq!(rows[0].source_offset_bytes, 40);
        assert_eq!(rows[0].bytes, &[40, 41, 42, 43, 44, 45, 46, 47]);
        assert_eq!(page.row(1), None);
    }

    #[test]
    fn request_key_matches_existing_virtual_texture_addressing() {
        let descriptor = descriptor(256, 256, 2, 128, 9);
        let snapshot = MaterialVtSourceSnapshot::new(descriptor, payload(descriptor)).unwrap();
        let request = snapshot.virtual_texture().page_at([1.0, 1.0], 0);
        let key = snapshot.key_for_request(1, request).unwrap();
        assert_eq!(
            key,
            TexturePageKey {
                texture: 41,
                mip: 0,
                layer: 1,
                x: 1,
                y: 1
            }
        );
        assert!(snapshot.page(key).is_ok());
    }

    #[test]
    fn texture_row_is_the_existing_abi_row_with_source_metadata() {
        let descriptor = descriptor(4, 4, 1, 2, 3);
        let snapshot = MaterialVtSourceSnapshot::new(descriptor, payload(descriptor)).unwrap();
        let row = snapshot.texture_row(9, 12, 3, 7);
        assert_eq!(row.gpu.virtual_texture_id, 41);
        assert_eq!(row.gpu.virtual_texture_width, 4);
        assert_eq!(row.gpu.virtual_texture_height, 4);
        assert_eq!(row.gpu.virtual_texture_page_size, 2);
        assert_eq!(row.gpu.index, 9);
        assert_eq!(row.format, MaterialVtFormat::Rgba8Unorm);
        assert_eq!(row.layers, 1);
        assert_eq!(row.mip_count, 3);
        assert_eq!(row.page_count, 6);
    }

    #[test]
    fn page_rows_include_layer_and_mip_offsets_without_copying() {
        let descriptor = descriptor(4, 3, 2, 2, 3);
        let snapshot = MaterialVtSourceSnapshot::new(descriptor, payload(descriptor)).unwrap();
        let key = TexturePageKey {
            texture: 41,
            mip: 1,
            layer: 1,
            x: 0,
            y: 0,
        };
        let page = snapshot.page(key).unwrap();
        let row = page.row(0).unwrap();
        // mip 0 occupies 96 bytes; mip 1 has an 8-byte layer stride.
        assert_eq!(row.source_offset_bytes, 104);
        assert_eq!(row.bytes, &snapshot.bytes()[104..112]);
        assert_eq!(row.bytes.as_ptr(), snapshot.bytes()[104..].as_ptr());
    }

    #[test]
    fn source_formats_have_explicit_texel_widths() {
        assert_eq!(MaterialVtFormat::R8Unorm.bytes_per_texel(), 1);
        assert_eq!(MaterialVtFormat::Rg8Unorm.bytes_per_texel(), 2);
        assert_eq!(MaterialVtFormat::Rgba8Unorm.bytes_per_texel(), 4);
        assert_eq!(MaterialVtFormat::Rgba8Srgb.bytes_per_texel(), 4);
        assert_eq!(MaterialVtFormat::Rgba16Float.bytes_per_texel(), 8);
        assert_eq!(MaterialVtFormat::Rgba32Float.bytes_per_texel(), 16);
        assert!(MaterialVtFormat::Rgba8Srgb.is_srgb());
        assert!(!MaterialVtFormat::Rgba8Unorm.is_srgb());
    }

    #[test]
    fn page_key_rejects_wrong_identity_layer_and_grid_coordinates() {
        let descriptor = descriptor(4, 3, 1, 2, 3);
        let snapshot = MaterialVtSourceSnapshot::new(descriptor, payload(descriptor)).unwrap();
        assert!(matches!(
            snapshot.page_metadata(TexturePageKey {
                texture: 42,
                mip: 0,
                layer: 0,
                x: 0,
                y: 0,
            }),
            Err(MaterialVtError::TextureIdMismatch { .. })
        ));
        assert!(matches!(
            snapshot.page_metadata(TexturePageKey {
                texture: 41,
                mip: 0,
                layer: 1,
                x: 0,
                y: 0,
            }),
            Err(MaterialVtError::LayerOutOfRange { layer: 1 })
        ));
        assert!(matches!(
            snapshot.page_metadata(TexturePageKey {
                texture: 41,
                mip: 0,
                layer: 0,
                x: 2,
                y: 0,
            }),
            Err(MaterialVtError::PageXOutOfRange { .. })
        ));
    }

    #[test]
    fn payload_length_is_an_explicit_contract_error() {
        let descriptor = descriptor(2, 2, 1, 2, 2);
        let mut bytes = payload(descriptor);
        bytes.pop();
        assert!(matches!(
            MaterialVtSourceSnapshot::new(descriptor, bytes),
            Err(MaterialVtError::PayloadLengthMismatch {
                expected: 20,
                actual: 19
            })
        ));
    }
}
