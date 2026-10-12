//! Reading GPU results back to the CPU, and turning rendered pixels into a
//! number the test can assert on.

use alloc::vec::Vec;

use prism_render_driver::BufferId;
use prism_render_driver_wgpu::WgpuQueue;

/// A backend hook that copies a buffer's contents back to the CPU.
///
/// The RHI itself is fire-and-forget, so readback is deliberately a backend
/// capability rather than an RHI trait method: it exists for tests, golden
/// images and observability, never on the hot path. The wgpu backend provides
/// the only implementation for now.
pub trait ReadBuffer {
    /// Copies `size` bytes from `buffer` to the CPU, blocking until the GPU has
    /// caught up.
    fn read_buffer(&self, buffer: BufferId, size: u64) -> Vec<u8>;
}

impl ReadBuffer for WgpuQueue {
    fn read_buffer(&self, buffer: BufferId, size: u64) -> Vec<u8> {
        WgpuQueue::read_buffer(self, buffer, size)
    }
}

/// Counts the pixels whose red channel is above half intensity.
///
/// The cull draw paints axis-aligned white quads on a black clear, so this is
/// the rasterized coverage: an exact multiple of the per-instance quad area
/// whenever the survivors land in distinct, non-overlapping cells.
#[must_use]
pub fn count_lit_pixels(rgba: &[u8]) -> usize {
    rgba.chunks_exact(4).filter(|pixel| pixel[0] > 127).count()
}

/// The RGBA bytes of pixel `(x, y)` in a tightly packed `width x height` image.
///
/// # Panics
///
/// Panics when the pixel lies outside the image or the buffer is too short.
#[must_use]
pub fn pixel_at(rgba: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    assert!(
        x < width,
        "pixel column {x} is outside a {width}-wide image"
    );
    let offset = ((y * width + x) * 4) as usize;
    rgba[offset..offset + 4]
        .try_into()
        .expect("a tightly packed RGBA image has four bytes per pixel")
}
