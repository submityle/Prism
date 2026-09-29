//! Per-particle *trail history ring buffer*: capturing and ageing an ordered
//! run of centerline samples for each trail (design §15).
//!
//! This module is the data-source *upstream* of
//! [`super::ribbon_geometry`], and is deliberately kept separate from it.
//! `ribbon_geometry` performs the geometric expansion of an already-ordered
//! centerline into camera-facing triangle-strip corners; it consumes points, it
//! never decides *which* points exist. Here we own the orthogonal concern: the
//! fixed-capacity per-trail ring buffer that records where a particle has been
//! (`position`), how wide the ribbon should be there (`width`), and how old each
//! recorded sample is (`age`), plus the capture/expiry policy that governs when
//! a new sample is appended and when an old one falls out of the trail.
//!
//! The buffer is laid out as one contiguous flat point array of
//! `trail_count * max_points_per_trail` slots, so a `GPU` compute kernel can
//! address `trail * capacity + local` without a per-trail base table. Element
//! stride reuses the shared `std430` constants from [`super::gpu_layout`]: a
//! `position` padded to a `vec4` plus a `vec2` carrying `width`/`age`. Nothing
//! here re-derives strides or attribute widths.
//!
//! Only squared-distance arithmetic and integer ring math are used — no `sqrt`
//! and no transcendental functions — so this `CPU` reference stays bit-for-bit
//! reproducible against a future `GPU` capture kernel. Reads follow a
//! Structure-of-Arrays (`SoA`) friendly, oldest-to-newest ordering.

use alloc::vec::Vec;

use crate::particle::gpu_layout::{storage_bytes, VEC2_STRIDE, VEC4_STRIDE};

/// Byte stride of one recorded trail point in the flat `std430` point buffer.
///
/// The `position` occupies a `vec4` slot (16 bytes, the natural alignment a
/// `vec3` is padded up to) and `width`/`age` share a trailing `vec2` (8 bytes).
pub const TRAIL_POINT_STRIDE: usize = VEC4_STRIDE + VEC2_STRIDE;

/// One recorded sample along a particle's trail (design §15).
///
/// `position` is the world-space centerline point, `width` the ribbon
/// half-extent request sampled from the size-over-life curve at capture time,
/// and `age` the seconds elapsed since the sample was recorded (advanced by the
/// simulation and compared against the capture policy's `max_age`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrailPoint {
    /// World-space centerline position of this sample.
    pub position: [f32; 3],
    /// Ribbon width request recorded at capture time.
    pub width: f32,
    /// Seconds elapsed since this sample was captured.
    pub age: f32,
}

impl TrailPoint {
    /// Creates a trail point from its position, width, and age.
    #[must_use]
    pub const fn new(position: [f32; 3], width: f32, age: f32) -> Self {
        Self {
            position,
            width,
            age,
        }
    }
}

/// The fixed-capacity per-trail ring-buffer geometry shared by the `CPU`
/// reference and the future `GPU` capture kernel (design §15).
///
/// `max_points_per_trail` is the ring capacity of a single trail;
/// `trail_count` is how many independent trails share the flat point buffer.
/// All addressing is pure integer arithmetic and saturates rather than
/// wrapping, so a degenerate configuration can never alias or under-allocate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TrailBuffer {
    /// Ring capacity of a single trail (points retained before overwrite).
    pub max_points_per_trail: u32,
    /// Number of independent trails sharing the flat point buffer.
    pub trail_count: u32,
}

impl TrailBuffer {
    /// Creates a trail buffer from its per-trail capacity and trail count.
    #[must_use]
    pub const fn new(max_points_per_trail: u32, trail_count: u32) -> Self {
        Self {
            max_points_per_trail,
            trail_count,
        }
    }

    /// Ring capacity of a single trail (its `max_points_per_trail`).
    #[must_use]
    pub const fn ring_capacity(self) -> u32 {
        self.max_points_per_trail
    }

    /// Total number of point slots across every trail
    /// (`max_points_per_trail * trail_count`).
    ///
    /// The product saturates at [`u32::MAX`] rather than wrapping.
    #[must_use]
    pub fn total_point_capacity(self) -> u32 {
        self.max_points_per_trail.saturating_mul(self.trail_count)
    }

    /// Total byte size of the flat `std430` point buffer.
    ///
    /// Uses [`TRAIL_POINT_STRIDE`] and the shared clamp-to-one-element rule from
    /// [`storage_bytes`], so an empty buffer still yields a valid, non-empty
    /// `GPU` storage binding.
    #[must_use]
    pub fn point_buffer_bytes(self) -> usize {
        let count = usize::try_from(self.total_point_capacity()).unwrap_or(usize::MAX);
        storage_bytes(TRAIL_POINT_STRIDE, count)
    }

    /// Flat-buffer index of point `local` within trail `trail`.
    ///
    /// Returns [`None`] when `trail` is out of range or `local` is beyond the
    /// ring capacity, guarding every caller against an out-of-bounds slot.
    #[must_use]
    pub fn point_global_index(self, trail: u32, local: u32) -> Option<u32> {
        if trail >= self.trail_count || local >= self.max_points_per_trail {
            None
        } else {
            Some(
                trail
                    .saturating_mul(self.max_points_per_trail)
                    .saturating_add(local),
            )
        }
    }

    /// Flat-buffer index of the ring *write head* for `trail` after
    /// `count_written` samples have been captured.
    ///
    /// The local ring slot is `count_written % ring_capacity`; that slot is then
    /// resolved to a global index via [`Self::point_global_index`]. Returns
    /// [`None`] when the ring capacity is zero or `trail` is out of range.
    #[must_use]
    pub fn head_index(self, trail: u32, count_written: u32) -> Option<u32> {
        let capacity = self.max_points_per_trail;
        if capacity == 0 {
            return None;
        }
        let local = count_written % capacity;
        self.point_global_index(trail, local)
    }
}

/// When a new sample is captured into a trail and when an old one expires
/// (design §15).
///
/// `min_distance` is the minimum world-space travel between consecutive
/// captured samples; `max_age` is the lifetime after which a sample is dropped
/// from the trail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrailCapturePolicy {
    /// Minimum distance a particle must move before a new sample is captured.
    pub min_distance: f32,
    /// Sample lifetime, in seconds, before it is considered expired.
    pub max_age: f32,
}

impl TrailCapturePolicy {
    /// Creates a capture policy from its distance threshold and maximum age.
    #[must_use]
    pub const fn new(min_distance: f32, max_age: f32) -> Self {
        Self {
            min_distance,
            max_age,
        }
    }

    /// Whether the particle has moved far enough from `last_pos` to capture a
    /// new sample at `new_pos`.
    ///
    /// The comparison is on squared distance against `min_distance * min_distance`
    /// so no `sqrt` is needed. Capture happens once travel reaches the threshold.
    #[must_use]
    pub fn should_capture(self, last_pos: [f32; 3], new_pos: [f32; 3]) -> bool {
        let dx = new_pos[0] - last_pos[0];
        let dy = new_pos[1] - last_pos[1];
        let dz = new_pos[2] - last_pos[2];
        let distance_squared = dx * dx + dy * dy + dz * dz;
        let threshold_squared = self.min_distance * self.min_distance;
        distance_squared >= threshold_squared
    }

    /// Whether a sample of the given `age` has outlived `max_age`.
    #[must_use]
    pub fn is_expired(self, age: f32) -> bool {
        age > self.max_age
    }

    /// Number of active ribbon segments a trail with `written` captured samples
    /// spans, clamped to the ring `capacity`.
    ///
    /// A trail holding `n` points forms `n - 1` segments (and zero for a single
    /// point or an empty trail), so this returns
    /// `min(written, capacity).saturating_sub(1)`.
    #[must_use]
    pub fn active_segment_count(self, written: u32, capacity: u32) -> u32 {
        written.min(capacity).saturating_sub(1)
    }
}

/// Local ring-slot indices of a trail from oldest to newest sample.
///
/// `head` is the ring write head (`count % capacity`), `count` the number of
/// samples captured so far, and `capacity` the ring capacity. Before the ring
/// fills (`count < capacity`) the samples occupy `0..count` in capture order;
/// once full, iteration starts at `head` (the oldest, next-to-overwrite slot)
/// and wraps around the ring. A zero capacity yields an empty ordering.
#[must_use]
pub fn iter_ordered_indices(head: u32, count: u32, capacity: u32) -> Vec<usize> {
    if capacity == 0 {
        return Vec::new();
    }
    if count < capacity {
        (0..count)
            .map(|i| usize::try_from(i).unwrap_or(usize::MAX))
            .collect()
    } else {
        (0..capacity)
            .map(|i| {
                let slot = (u64::from(head) + u64::from(i)) % u64::from(capacity);
                usize::try_from(slot).unwrap_or(usize::MAX)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the float assertions in this module's tests.
    const CMP_EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn trail_point_stores_fields() {
        let point = TrailPoint::new([1.0, 2.0, 3.0], 0.5, 0.25);
        assert_eq!(point.position, [1.0, 2.0, 3.0]);
        assert!(approx_eq(point.width, 0.5));
        assert!(approx_eq(point.age, 0.25));
    }

    #[test]
    fn stride_packs_vec4_plus_vec2() {
        assert_eq!(TRAIL_POINT_STRIDE, VEC4_STRIDE + VEC2_STRIDE);
        assert_eq!(TRAIL_POINT_STRIDE, 24);
    }

    #[test]
    fn capacities_multiply_and_saturate() {
        let buffer = TrailBuffer::new(4, 3);
        assert_eq!(buffer.ring_capacity(), 4);
        assert_eq!(buffer.total_point_capacity(), 12);

        let huge = TrailBuffer::new(u32::MAX, 2);
        assert_eq!(huge.total_point_capacity(), u32::MAX);
    }

    #[test]
    fn point_buffer_bytes_uses_stride_and_clamps_empty() {
        let buffer = TrailBuffer::new(4, 3);
        assert_eq!(buffer.point_buffer_bytes(), TRAIL_POINT_STRIDE * 12);

        let empty = TrailBuffer::new(0, 0);
        assert_eq!(empty.point_buffer_bytes(), TRAIL_POINT_STRIDE);
    }

    #[test]
    fn global_index_guards_out_of_bounds() {
        let buffer = TrailBuffer::new(4, 3);
        assert_eq!(buffer.point_global_index(0, 0), Some(0));
        assert_eq!(buffer.point_global_index(2, 3), Some(11));
        assert_eq!(buffer.point_global_index(3, 0), None);
        assert_eq!(buffer.point_global_index(0, 4), None);
    }

    #[test]
    fn head_index_is_ring_write_head() {
        let buffer = TrailBuffer::new(4, 3);
        assert_eq!(buffer.head_index(1, 0), Some(4));
        assert_eq!(buffer.head_index(1, 6), Some(6));
        assert_eq!(buffer.head_index(3, 0), None);

        let empty = TrailBuffer::new(0, 2);
        assert_eq!(empty.head_index(0, 0), None);
    }

    #[test]
    fn ordered_indices_before_ring_fills() {
        assert_eq!(iter_ordered_indices(0, 3, 4), Vec::from([0, 1, 2]));
    }

    #[test]
    fn ordered_indices_exactly_full() {
        assert_eq!(iter_ordered_indices(0, 4, 4), Vec::from([0, 1, 2, 3]));
    }

    #[test]
    fn ordered_indices_wrap_when_full() {
        assert_eq!(iter_ordered_indices(2, 6, 4), Vec::from([2, 3, 0, 1]));
    }

    #[test]
    fn ordered_indices_empty_capacity() {
        assert!(iter_ordered_indices(0, 0, 0).is_empty());
    }

    #[test]
    fn distance_threshold_uses_squared_distance() {
        let policy = TrailCapturePolicy::new(2.0, 5.0);
        assert!(!policy.should_capture([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]));
        assert!(policy.should_capture([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]));
        assert!(policy.should_capture([0.0, 0.0, 0.0], [3.0, 0.0, 0.0]));
    }

    #[test]
    fn expiry_compares_against_max_age() {
        let policy = TrailCapturePolicy::new(1.0, 5.0);
        assert!(!policy.is_expired(4.0));
        assert!(!policy.is_expired(5.0));
        assert!(policy.is_expired(6.0));
    }

    #[test]
    fn active_segment_count_clamps_and_saturates() {
        let policy = TrailCapturePolicy::new(1.0, 5.0);
        assert_eq!(policy.active_segment_count(3, 4), 2);
        assert_eq!(policy.active_segment_count(10, 4), 3);
        assert_eq!(policy.active_segment_count(1, 4), 0);
        assert_eq!(policy.active_segment_count(0, 4), 0);
    }
}
