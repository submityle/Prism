//! Strict CPU-side preparation for uploading an [`Image`] to a real texture.
//!
//! `prism_image` owns image bytes, but it does not own GPU resources. This
//! module therefore produces a renderer-consumable staging layout and only
//! performs a queue write when the caller supplies a real [`TextureId`]. The
//! caller remains responsible for creating that texture with the plan's
//! extent and format.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use prism_render_driver::{
    Extent3d, ImageDataLayout, RenderQueue, TextureFormat, TextureId, TextureWrite,
};

use crate::Image;

/// Required row alignment for a GPU texture-copy staging layout.
///
/// The CPU-side [`Image`] remains tightly packed. The upload plan pads each
/// row to this boundary so a backend can consume it through a standard
/// texture-copy upload path without guessing or repacking the bytes.
pub const IMAGE_UPLOAD_ROW_ALIGNMENT: u32 = 256;

/// Why an image could not be represented as a bounded upload plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageUploadPlanError {
    /// The unpadded row pitch cannot be represented by the RHI's `u32` layout.
    RowPitchOverflow,
    /// The padded staging allocation cannot be represented by this process.
    StagingSizeOverflow,
}

impl core::fmt::Display for ImageUploadPlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RowPitchOverflow => {
                f.write_str("image upload row pitch overflows the RHI layout")
            }
            Self::StagingSizeOverflow => {
                f.write_str("image upload staging size overflows the process address space")
            }
        }
    }
}

impl std::error::Error for ImageUploadPlanError {}

/// A validated, renderer-agnostic staging layout for one image upload.
///
/// The plan owns padded bytes. Its `data` length is exactly
/// `layout.bytes_per_row * layout.rows_per_image * extent.depth_or_array_layers`;
/// its unpadded image content is copied row-for-row from [`Image::data`]. No
/// GPU object is created or represented by this type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageUploadPlan {
    extent: Extent3d,
    format: TextureFormat,
    layout: ImageDataLayout,
    data: Vec<u8>,
}

impl ImageUploadPlan {
    /// Builds a staging plan from an already validated image.
    pub fn for_image(image: &Image) -> Result<Self, ImageUploadPlanError> {
        let extent = image.extent();
        let tight_row = u64::from(extent.width)
            .checked_mul(u64::from(image.format().bytes_per_texel()))
            .ok_or(ImageUploadPlanError::RowPitchOverflow)?;
        let aligned_row = tight_row
            .checked_add(u64::from(IMAGE_UPLOAD_ROW_ALIGNMENT - 1))
            .map(|value| value / u64::from(IMAGE_UPLOAD_ROW_ALIGNMENT))
            .and_then(|value| value.checked_mul(u64::from(IMAGE_UPLOAD_ROW_ALIGNMENT)))
            .ok_or(ImageUploadPlanError::RowPitchOverflow)?;
        let bytes_per_row =
            u32::try_from(aligned_row).map_err(|_| ImageUploadPlanError::RowPitchOverflow)?;

        let row_count = u64::from(extent.height)
            .checked_mul(u64::from(extent.depth_or_array_layers))
            .ok_or(ImageUploadPlanError::StagingSizeOverflow)?;
        let staging_len = aligned_row
            .checked_mul(row_count)
            .and_then(|size| usize::try_from(size).ok())
            .ok_or(ImageUploadPlanError::StagingSizeOverflow)?;

        let tight_row =
            usize::try_from(tight_row).map_err(|_| ImageUploadPlanError::StagingSizeOverflow)?;
        let padded_row =
            usize::try_from(aligned_row).map_err(|_| ImageUploadPlanError::StagingSizeOverflow)?;
        let height = usize::try_from(extent.height)
            .map_err(|_| ImageUploadPlanError::StagingSizeOverflow)?;
        let layers = usize::try_from(extent.depth_or_array_layers)
            .map_err(|_| ImageUploadPlanError::StagingSizeOverflow)?;
        let mut data = vec![0; staging_len];
        for layer in 0..layers {
            for row in 0..height {
                let row_index = layer
                    .checked_mul(height)
                    .and_then(|base| base.checked_add(row))
                    .ok_or(ImageUploadPlanError::StagingSizeOverflow)?;
                let source =
                    checked_row_range(row_index, tight_row, tight_row, image.data().len())?;
                let target = checked_row_range(row_index, padded_row, tight_row, data.len())?;
                data[target].copy_from_slice(&image.data()[source]);
            }
        }

        Ok(Self {
            extent,
            format: image.format(),
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(extent.height),
            },
            data,
        })
    }

    /// The destination texture size required by this plan.
    #[must_use]
    pub const fn extent(&self) -> Extent3d {
        self.extent
    }

    /// The destination texture format required by this plan.
    #[must_use]
    pub const fn format(&self) -> TextureFormat {
        self.format
    }

    /// The exact RHI layout for [`Self::data`].
    #[must_use]
    pub const fn layout(&self) -> ImageDataLayout {
        self.layout
    }

    /// The padded staging bytes, including row padding.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The number of bytes allocated in the staging payload.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the staging payload is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Uploads this plan to a caller-owned real texture through a real queue.
    ///
    /// `texture` must have been created by the renderer with [`Self::extent`]
    /// and [`Self::format`]. This method intentionally cannot create, inspect,
    /// or fake that GPU object because resource ownership belongs to the render
    /// app/device boundary.
    pub fn write_to(&self, queue: &dyn RenderQueue, texture: TextureId) {
        queue.write_texture(
            TextureWrite {
                texture,
                mip_level: 0,
                origin: [0, 0, 0],
                size: self.extent,
            },
            &self.data,
            self.layout,
        );
    }
}

/// Computes one row range without relying on unchecked `usize` arithmetic.
pub(crate) fn checked_row_range(
    row_index: usize,
    row_pitch: usize,
    row_bytes: usize,
    buffer_len: usize,
) -> Result<Range<usize>, ImageUploadPlanError> {
    let start = row_index
        .checked_mul(row_pitch)
        .ok_or(ImageUploadPlanError::StagingSizeOverflow)?;
    let end = start
        .checked_add(row_bytes)
        .ok_or(ImageUploadPlanError::StagingSizeOverflow)?;
    if end > buffer_len {
        return Err(ImageUploadPlanError::StagingSizeOverflow);
    }
    Ok(start..end)
}
