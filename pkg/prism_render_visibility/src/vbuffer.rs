//! Visibility buffer (`VBuffer`) primitives for deferred material resolve.
//!
//! A visibility buffer stores, per screen pixel, the identity of the surface
//! that is visible there: an `instance_id` plus a `primitive_id` (triangle
//! index). Shading is deferred to a later pass that reconstructs every surface
//! attribute analytically from this minimal payload, following Deferred
//! Attribute Interpolation Shading (DAIS, Schied & Dachsbacher 2015).
//!
//! The identity is packed into a single `u64` so the buffer is a flat
//! `Vec<u64>` that maps one-to-one onto a GPU `R32G32_UINT` (or `R64_UINT`)
//! render target. The sentinel `u64::MAX` marks an uncovered pixel.

use alloc::vec;
use alloc::vec::Vec;

/// Sentinel stored for a pixel that no triangle covers.
///
/// `instance_id == u32::MAX && primitive_id == u32::MAX` can never be a real
/// sample because a scene never allocates `u32::MAX` instances, so the packed
/// `u64::MAX` is an unambiguous "empty" marker.
pub const EMPTY_SAMPLE: u64 = u64::MAX;

/// Identity of the surface visible at one pixel.
///
/// This is the entire per-pixel payload of the visibility buffer. All shading
/// inputs (position, normal, `UV`, tangent frame, material) are reconstructed
/// later from `instance_id` + `primitive_id` plus the frame's transforms and
/// vertex streams.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct VisibilitySample {
    /// Index of the instance (TLAS leaf) that owns the visible triangle.
    pub instance_id: u32,
    /// Index of the triangle within that instance's geometry.
    pub primitive_id: u32,
}

impl VisibilitySample {
    /// Builds a sample from an instance and primitive index.
    #[must_use]
    pub const fn new(instance_id: u32, primitive_id: u32) -> Self {
        Self {
            instance_id,
            primitive_id,
        }
    }

    /// Packs the sample into a single `u64` (`instance_id` in the high 32 bits,
    /// `primitive_id` in the low 32 bits).
    ///
    /// Packing instance in the high bits means a plain integer sort of the
    /// packed words groups pixels by instance first, which is the natural order
    /// for a deferred material dispatch.
    #[must_use]
    pub const fn pack(self) -> u64 {
        ((self.instance_id as u64) << 32) | (self.primitive_id as u64)
    }

    /// Unpacks a `u64` previously produced by [`VisibilitySample::pack`].
    ///
    /// Returns `None` for the [`EMPTY_SAMPLE`] sentinel.
    #[must_use]
    pub const fn unpack(packed: u64) -> Option<Self> {
        if packed == EMPTY_SAMPLE {
            return None;
        }
        Some(Self {
            instance_id: (packed >> 32) as u32,
            primitive_id: (packed & 0xFFFF_FFFF) as u32,
        })
    }
}

/// Flat, row-major visibility buffer of packed [`VisibilitySample`] words.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisibilityBuffer {
    width: u32,
    height: u32,
    samples: Vec<u64>,
}

impl VisibilityBuffer {
    /// Allocates a fully-uncovered buffer of the given dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `width * height` overflows `usize`, which only happens on
    /// absurd resolutions that could never fit in memory anyway.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width as usize)
            .checked_mul(height as usize)
            .expect("visibility buffer dimensions overflow usize");
        Self {
            width,
            height,
            samples: vec![EMPTY_SAMPLE; len],
        }
    }

    /// Width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Total pixel count.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer has zero pixels.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Raw packed words, row-major. Mirrors a GPU readback.
    #[must_use]
    pub fn samples(&self) -> &[u64] {
        &self.samples
    }

    /// Converts a pixel coordinate to a linear index, or `None` if out of range.
    #[must_use]
    pub const fn linear_index(&self, x: u32, y: u32) -> Option<usize> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some((y as usize) * (self.width as usize) + (x as usize))
    }

    /// Reads the sample at a pixel, or `None` if uncovered / out of range.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Option<VisibilitySample> {
        let idx = self.linear_index(x, y)?;
        VisibilitySample::unpack(self.samples[idx])
    }

    /// Writes a sample at a pixel. Out-of-range writes are ignored.
    pub fn set(&mut self, x: u32, y: u32, sample: VisibilitySample) {
        if let Some(idx) = self.linear_index(x, y) {
            self.samples[idx] = sample.pack();
        }
    }

    /// Clears a pixel back to uncovered. Out-of-range writes are ignored.
    pub fn clear_pixel(&mut self, x: u32, y: u32) {
        if let Some(idx) = self.linear_index(x, y) {
            self.samples[idx] = EMPTY_SAMPLE;
        }
    }

    /// Whether a pixel holds a real sample (in range and not the sentinel).
    #[must_use]
    pub fn is_covered(&self, x: u32, y: u32) -> bool {
        self.linear_index(x, y)
            .is_some_and(|idx| self.samples[idx] != EMPTY_SAMPLE)
    }

    /// Resets every pixel to uncovered without reallocating.
    pub fn clear(&mut self) {
        self.samples.fill(EMPTY_SAMPLE);
    }

    /// Counts covered pixels. O(n); intended for diagnostics and tests.
    #[must_use]
    pub fn covered_count(&self) -> usize {
        self.samples.iter().filter(|&&s| s != EMPTY_SAMPLE).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trips() {
        let s = VisibilitySample::new(7, 123_456);
        let packed = s.pack();
        assert_eq!(VisibilitySample::unpack(packed), Some(s));
    }

    #[test]
    fn pack_orders_instance_major() {
        let a = VisibilitySample::new(1, 999).pack();
        let b = VisibilitySample::new(2, 0).pack();
        // Instance 1 (any primitive) always sorts before instance 2.
        assert!(a < b);
    }

    #[test]
    fn empty_sentinel_unpacks_to_none() {
        assert_eq!(VisibilitySample::unpack(EMPTY_SAMPLE), None);
    }

    #[test]
    fn max_real_sample_is_distinguishable_from_empty() {
        // Largest realistic sample must still differ from the sentinel.
        let s = VisibilitySample::new(u32::MAX - 1, u32::MAX);
        assert_ne!(s.pack(), EMPTY_SAMPLE);
        assert_eq!(VisibilitySample::unpack(s.pack()), Some(s));
    }

    #[test]
    fn new_buffer_is_fully_uncovered() {
        let vb = VisibilityBuffer::new(4, 3);
        assert_eq!(vb.len(), 12);
        assert_eq!(vb.covered_count(), 0);
        assert!(!vb.is_covered(0, 0));
        assert_eq!(vb.get(2, 2), None);
    }

    #[test]
    fn set_get_and_clear_pixel() {
        let mut vb = VisibilityBuffer::new(8, 8);
        let s = VisibilitySample::new(3, 17);
        vb.set(5, 6, s);
        assert!(vb.is_covered(5, 6));
        assert_eq!(vb.get(5, 6), Some(s));
        assert_eq!(vb.covered_count(), 1);

        vb.clear_pixel(5, 6);
        assert!(!vb.is_covered(5, 6));
        assert_eq!(vb.get(5, 6), None);
        assert_eq!(vb.covered_count(), 0);
    }

    #[test]
    fn linear_index_matches_row_major_layout() {
        let vb = VisibilityBuffer::new(10, 4);
        assert_eq!(vb.linear_index(0, 0), Some(0));
        assert_eq!(vb.linear_index(9, 0), Some(9));
        assert_eq!(vb.linear_index(0, 1), Some(10));
        assert_eq!(vb.linear_index(3, 2), Some(23));
    }

    #[test]
    fn out_of_range_access_is_safe() {
        let mut vb = VisibilityBuffer::new(4, 4);
        assert_eq!(vb.linear_index(4, 0), None);
        assert_eq!(vb.linear_index(0, 4), None);
        assert_eq!(vb.get(4, 4), None);
        assert!(!vb.is_covered(100, 100));
        // Out-of-range writes must not panic and must not touch any pixel.
        vb.set(99, 99, VisibilitySample::new(1, 1));
        assert_eq!(vb.covered_count(), 0);
    }

    #[test]
    fn clear_resets_all_pixels() {
        let mut vb = VisibilityBuffer::new(3, 3);
        vb.set(0, 0, VisibilitySample::new(1, 2));
        vb.set(2, 2, VisibilitySample::new(3, 4));
        assert_eq!(vb.covered_count(), 2);
        vb.clear();
        assert_eq!(vb.covered_count(), 0);
    }
}
