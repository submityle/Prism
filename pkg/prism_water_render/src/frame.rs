//! Shared offscreen render-target plumbing for the water renderer.
//!
//! Both the procedural [`crate::surface`] demo and the spectral
//! [`crate::ocean`] renderer rasterise into an offscreen `Rgba8Unorm` colour
//! target with a `Depth32Float` depth buffer, then copy the result back to a
//! tightly packed [`RenderedFrame`]. That texture-creation, read-back and
//! row-unpadding logic is identical between them, so it lives here once rather
//! than being duplicated per renderer.

use alloc::vec::Vec;
use std::io;
use std::path::Path;

use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoder, CommandEncoderDescriptor, Extent3d, MapMode,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

use crate::context::GpuContext;
use crate::png;

/// Row alignment, in bytes, required for `copy_texture_to_buffer` destinations.
pub const ROW_ALIGNMENT: u32 = 256;

/// The colour format of the offscreen render target and the read-back image.
pub const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// The depth format used for hidden-surface removal.
pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// A rendered frame read back to host memory as tightly packed `RGBA8`.
#[derive(Clone, Debug)]
pub struct RenderedFrame {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row-major `RGBA8` pixels, top row first (`width * height * 4` bytes).
    pub rgba: Vec<u8>,
}

impl RenderedFrame {
    /// Encodes the frame as a `PNG` and writes it to `path`.
    ///
    /// # Errors
    ///
    /// Returns any `io` error raised while creating or writing the file.
    pub fn save_png(&self, path: &Path) -> io::Result<()> {
        let bytes = png::encode_rgba8(self.width, self.height, &self.rgba);
        std::fs::write(path, bytes)
    }

    /// Counts the number of distinct `RGBA` pixel values in the frame.
    ///
    /// A flat clear produces `1`; any rasterised, shaded geometry produces many.
    /// Tests use this as device-independent evidence that the pipeline actually
    /// drew and shaded the surface rather than clearing to a single colour.
    #[must_use]
    pub fn distinct_colors(&self) -> usize {
        let mut seen = Vec::new();
        for px in self.rgba.chunks_exact(4) {
            let key = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
            if let Err(idx) = seen.binary_search(&key) {
                seen.insert(idx, key);
            }
        }
        seen.len()
    }

    /// Counts pixels where the blue channel dominates red and is appreciably
    /// lit, a device-independent witness that the frame reads as water.
    #[must_use]
    pub fn bluish_pixels(&self) -> usize {
        self.rgba
            .chunks_exact(4)
            .filter(|px| px[2] > px[0] && px[2] > 40)
            .count()
    }
}

/// Rasterises an offscreen frame of `width` x `height` on `ctx`.
///
/// Creates the colour and depth targets and the read-back buffer, lets `record`
/// encode one render pass against those views, then copies the colour target
/// back to host memory. The caller owns the pipeline, bind groups and clear
/// colour, so this helper stays renderer-agnostic.
///
/// # Panics
///
/// Panics if the device fails to map the read-back buffer, which indicates a
/// lost or broken adapter rather than a recoverable per-frame condition.
#[must_use]
pub(crate) fn render_offscreen<F>(
    ctx: &GpuContext,
    width: u32,
    height: u32,
    record: F,
) -> RenderedFrame
where
    F: FnOnce(&mut CommandEncoder, &TextureView, &TextureView),
{
    let device = ctx.device();
    let queue = ctx.queue();

    let width = width.max(1);
    let height = height.max(1);

    let extent = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let color_tex = device.create_texture(&TextureDescriptor {
        label: Some("water-frame-color"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth_tex = device.create_texture(&TextureDescriptor {
        label: Some("water-frame-depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&TextureViewDescriptor::default());
    let depth_view = depth_tex.create_view(&TextureViewDescriptor::default());

    let bytes_per_row = align_up(width * 4, ROW_ALIGNMENT);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("water-frame-readback"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("water-frame-encoder"),
    });

    record(&mut encoder, &color_view, &depth_view);

    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &color_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        extent,
    );

    queue.submit([encoder.finish()]);

    readback.slice(..).map_async(MapMode::Read, |_| {});
    ctx.wait();

    let rgba = unpack_rows(&readback, width, height, bytes_per_row);
    readback.unmap();

    RenderedFrame {
        width,
        height,
        rgba,
    }
}

/// Copies a mapped, row-padded staging buffer into a tight `RGBA8` vector.
fn unpack_rows(readback: &wgpu::Buffer, width: u32, height: u32, bytes_per_row: u32) -> Vec<u8> {
    let view = readback
        .slice(..)
        .get_mapped_range()
        .expect("read-back buffer should map after the device completes the copy");
    let row_bytes = (width * 4) as usize;
    let padded = bytes_per_row as usize;
    let mut out = Vec::with_capacity(row_bytes * height as usize);
    for row in 0..height as usize {
        let start = row * padded;
        out.extend_from_slice(&view[start..start + row_bytes]);
    }
    drop(view);
    out
}

/// Rounds `value` up to the next multiple of `align` (a power of two).
fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}
