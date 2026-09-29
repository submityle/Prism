//! `GPU`-written indirect **compute-dispatch** argument buffers (design §9, §15).
//!
//! When a particle compute stage's launch size is decided *on the device* — how
//! many particles survived, how many events a frame appended — the host cannot
//! know the workgroup count at record time. The stage instead reads its
//! `x, y, z` workgroup counts from an on-device argument buffer that an earlier
//! compute pass wrote, and dispatches with `vkCmdDispatchIndirect` /
//! `dispatchWorkgroupsIndirect`. This module is the deterministic `CPU`
//! reference for the **layout and byte offsets** of that buffer: the
//! [`DispatchIndirectCommand`] `std430` record, the per-command
//! [`IndirectDispatchBuffer::command_offset`] into a packed array of commands,
//! and the [`CounterToDispatch`] rule a fill shader follows to turn an append
//! counter into a launch size.
//!
//! **Orthogonality.** This layer is deliberately narrow and does not overlap its
//! siblings:
//! - [`super::indirect_draw`] packs already-decided *draw* parameters (and its
//!   own `DispatchIndirectArgs`) from an alive count; this module instead owns
//!   the *addressing* of a multi-command dispatch buffer — capacity, stride, and
//!   bounds-checked offsets — that the `GPU` fills.
//! - [`super::gpu_dispatch`] owns the host-side pass order and the *direct*
//!   `CPU`-known workgroup derivation; this module owns only the record layout
//!   the `GPU` writes and reads back indirectly.
//!
//! Everything is pure integer arithmetic and panic-free: a zero workgroup size
//! yields an empty launch rather than dividing by zero, an out-of-range command
//! index yields `None` rather than an aliasing offset, and the workgroup product
//! is computed in `u64` so it can never wrap.

use crate::particle::gpu_layout::U32_STRIDE;

/// Number of `u32` dimension words in one dispatch record: `x`, `y`, `z`.
pub const DISPATCH_DIMENSIONS: usize = 3;

/// `std430` byte stride of one [`DispatchIndirectCommand`]: three tightly
/// packed `u32` workgroup counts, so `3 * 4 = 12` bytes. This matches the
/// `vkCmdDispatchIndirect` / `WebGPU` `dispatchWorkgroupsIndirect` record and is
/// the step between consecutive commands in an [`IndirectDispatchBuffer`].
pub const STRIDE: usize = DISPATCH_DIMENSIONS * U32_STRIDE;

/// The `GPU`-written argument record for one indirect compute dispatch.
///
/// The three fields are the workgroup counts along each axis, laid out in the
/// exact `std430` order the driver reads. A `GPU` fill kernel writes this record
/// (typically from an append counter via [`CounterToDispatch`]) before the
/// dependent stage dispatches from it.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DispatchIndirectCommand {
    /// Workgroups launched along `x` (the linear domain axis for 1-D kernels).
    pub x: u32,
    /// Workgroups launched along `y` (1 for the linear kernels).
    pub y: u32,
    /// Workgroups launched along `z` (1 for the linear kernels).
    pub z: u32,
}

impl DispatchIndirectCommand {
    /// A record with explicit workgroup counts along each axis.
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// The 1-D launch that covers `element_count` domain elements at
    /// `workgroup_size` threads per group: `x = ceil(elements / size)` with
    /// `y = z = 1`.
    ///
    /// A zero `workgroup_size` is guarded to an empty launch (`x = 0`) instead
    /// of dividing by zero — a degenerate stage simply does no work.
    #[must_use]
    pub const fn from_element_count(element_count: u32, workgroup_size: u32) -> Self {
        let x = if workgroup_size == 0 {
            0
        } else {
            element_count.div_ceil(workgroup_size)
        };
        Self { x, y: 1, z: 1 }
    }

    /// Total workgroups this record launches, `x * y * z`, computed in `u64` so
    /// the product can never overflow a `u32`.
    #[must_use]
    pub const fn total_workgroups(&self) -> u64 {
        (self.x as u64) * (self.y as u64) * (self.z as u64)
    }

    /// Returns `true` when any axis is zero, i.e. the dispatch launches no
    /// workgroups at all. A device treats such a record as a no-op.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.x == 0 || self.y == 0 || self.z == 0
    }

    /// The three dimension words in `std430` / indirect-buffer order.
    #[must_use]
    pub const fn as_words(&self) -> [u32; DISPATCH_DIMENSIONS] {
        [self.x, self.y, self.z]
    }
}

/// A packed, `GPU`-writable array of [`DispatchIndirectCommand`] records.
///
/// Several device-driven stages can share one buffer, each owning a slot at a
/// fixed index; the fill pass writes slot `i` and the dependent stage dispatches
/// from byte offset `i * STRIDE`. This type owns only the *addressing* contract
/// — the capacity, the total byte size, and the bounds-checked offset — not the
/// `wgpu` buffer itself.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IndirectDispatchBuffer {
    /// Number of dispatch command slots the buffer holds.
    pub command_capacity: usize,
}

impl IndirectDispatchBuffer {
    /// A buffer sized for `command_capacity` dispatch records.
    #[must_use]
    pub const fn new(command_capacity: usize) -> Self {
        Self { command_capacity }
    }

    /// Total buffer size in bytes: `capacity * STRIDE`.
    #[must_use]
    pub const fn buffer_bytes(&self) -> usize {
        self.command_capacity * STRIDE
    }

    /// Byte offset of the dispatch record at `index`, or `None` when `index` is
    /// out of range. Guarding the upper bound keeps a stage from dispatching
    /// from a neighboring slot's memory.
    #[must_use]
    pub const fn command_offset(&self, index: usize) -> Option<usize> {
        if index < self.command_capacity {
            Some(index * STRIDE)
        } else {
            None
        }
    }

    /// Clamp a workgroup count to a device's per-dimension dispatch limit (for
    /// example `65535` on many desktop `GPU`s). A fill kernel applies the same
    /// clamp so an oversized domain never exceeds the hardware bound.
    #[must_use]
    pub const fn max_dimension_clamp(count: u32, max_dimension: u32) -> u32 {
        if count < max_dimension {
            count
        } else {
            max_dimension
        }
    }
}

/// The contract for turning a `GPU` append counter into a dispatch launch size.
///
/// Many stages append their live element count into an atomic counter during an
/// earlier pass; a tiny fill kernel then reads that counter and writes a
/// [`DispatchIndirectCommand`] for the dependent stage. This type carries the
/// `workgroup_size` that fill kernel divides by and documents the exact
/// `ceil`-division it performs, so the `CPU` reference and the `WESL` shader
/// agree.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CounterToDispatch {
    /// Threads per workgroup the dependent stage launches with.
    pub workgroup_size: u32,
}

impl CounterToDispatch {
    /// A counter-to-dispatch rule for a stage using `workgroup_size` threads per
    /// group.
    #[must_use]
    pub const fn new(workgroup_size: u32) -> Self {
        Self { workgroup_size }
    }

    /// The record that clears the buffer to a no-op launch before the counter is
    /// resolved: `(0, 1, 1)`. Zeroing only `x` (rather than all three) keeps the
    /// `y`/`z` axes at their valid 1-D value so a partially written record still
    /// reads as an empty — not malformed — launch.
    #[must_use]
    pub const fn clear_command(&self) -> DispatchIndirectCommand {
        DispatchIndirectCommand::new(0, 1, 1)
    }

    /// The launch size for an already-resolved counter value, using the same
    /// `ceil`-division the fill kernel performs. This mirrors what the `GPU`
    /// writes; see [`indirect_from_counter_note`](Self::indirect_from_counter_note)
    /// for the shader-side description.
    #[must_use]
    pub const fn dispatch_for_count(&self, counter: u32) -> DispatchIndirectCommand {
        DispatchIndirectCommand::from_element_count(counter, self.workgroup_size)
    }

    /// Prose describing how the fill shader derives the launch from the counter,
    /// kept beside the arithmetic so the `WESL` twin stays in sync.
    #[must_use]
    pub const fn indirect_from_counter_note(&self) -> &'static str {
        "GPU fill kernel: read the append counter `n`, compute \
         `x = (n + workgroup_size - 1) / workgroup_size` (ceil-division, guarded \
         so a zero workgroup size writes 0), set `y = z = 1`, and store the three \
         words as a DispatchIndirectCommand for the dependent stage to dispatch \
         from with dispatchWorkgroupsIndirect."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_is_three_u32_words() {
        assert_eq!(U32_STRIDE, 4);
        assert_eq!(DISPATCH_DIMENSIONS, 3);
        assert_eq!(STRIDE, 12);
    }

    #[test]
    fn from_element_count_div_ceil_boundaries() {
        // Zero elements -> no workgroups.
        assert_eq!(
            DispatchIndirectCommand::from_element_count(0, 64),
            DispatchIndirectCommand::new(0, 1, 1)
        );
        // Exact multiple.
        assert_eq!(DispatchIndirectCommand::from_element_count(256, 64).x, 4);
        // One over a multiple rounds up.
        assert_eq!(DispatchIndirectCommand::from_element_count(257, 64).x, 5);
        // One under a multiple rounds up to the same group.
        assert_eq!(DispatchIndirectCommand::from_element_count(255, 64).x, 4);
        // Fewer than one group still needs one.
        assert_eq!(DispatchIndirectCommand::from_element_count(1, 64).x, 1);
    }

    #[test]
    fn from_element_count_guards_zero_workgroup_size() {
        let c = DispatchIndirectCommand::from_element_count(1_000, 0);
        assert_eq!(c, DispatchIndirectCommand::new(0, 1, 1));
    }

    #[test]
    fn from_element_count_sets_y_and_z_to_one() {
        let c = DispatchIndirectCommand::from_element_count(100, 64);
        assert_eq!(c.y, 1);
        assert_eq!(c.z, 1);
        assert_eq!(c.as_words(), [2, 1, 1]);
    }

    #[test]
    fn total_workgroups_uses_u64_without_overflow() {
        let c = DispatchIndirectCommand::new(u32::MAX, u32::MAX, 1);
        // Would overflow a u32; the u64 product is exact.
        assert_eq!(c.total_workgroups(), (u32::MAX as u64) * (u32::MAX as u64));
        assert_eq!(DispatchIndirectCommand::new(5, 1, 1).total_workgroups(), 5);
    }

    #[test]
    fn is_empty_detects_any_zero_axis() {
        assert!(DispatchIndirectCommand::new(0, 1, 1).is_empty());
        assert!(DispatchIndirectCommand::new(4, 0, 1).is_empty());
        assert!(DispatchIndirectCommand::new(4, 1, 0).is_empty());
        assert!(!DispatchIndirectCommand::new(4, 1, 1).is_empty());
        assert!(DispatchIndirectCommand::default().is_empty());
    }

    #[test]
    fn buffer_bytes_scales_with_capacity() {
        assert_eq!(IndirectDispatchBuffer::new(0).buffer_bytes(), 0);
        assert_eq!(IndirectDispatchBuffer::new(1).buffer_bytes(), STRIDE);
        assert_eq!(IndirectDispatchBuffer::new(8).buffer_bytes(), 96);
    }

    #[test]
    fn command_offset_is_strided_and_bounds_checked() {
        let buffer = IndirectDispatchBuffer::new(3);
        assert_eq!(buffer.command_offset(0), Some(0));
        assert_eq!(buffer.command_offset(1), Some(STRIDE));
        assert_eq!(buffer.command_offset(2), Some(2 * STRIDE));
        // First out-of-range index is guarded.
        assert_eq!(buffer.command_offset(3), None);
        assert_eq!(buffer.command_offset(999), None);
    }

    #[test]
    fn empty_buffer_offsets_are_always_none() {
        let buffer = IndirectDispatchBuffer::new(0);
        assert_eq!(buffer.command_offset(0), None);
    }

    #[test]
    fn max_dimension_clamp_caps_at_device_limit() {
        // Under the limit passes through.
        assert_eq!(
            IndirectDispatchBuffer::max_dimension_clamp(100, 65_535),
            100
        );
        // Over the limit clamps down.
        assert_eq!(
            IndirectDispatchBuffer::max_dimension_clamp(1_000_000, 65_535),
            65_535
        );
        // Exactly at the limit clamps to the limit.
        assert_eq!(
            IndirectDispatchBuffer::max_dimension_clamp(65_535, 65_535),
            65_535
        );
    }

    #[test]
    fn clear_command_is_a_no_op_launch() {
        let rule = CounterToDispatch::new(64);
        let cleared = rule.clear_command();
        assert_eq!(cleared, DispatchIndirectCommand::new(0, 1, 1));
        assert!(cleared.is_empty());
        assert_eq!(cleared.total_workgroups(), 0);
    }

    #[test]
    fn dispatch_for_count_matches_element_derivation() {
        let rule = CounterToDispatch::new(64);
        assert_eq!(
            rule.dispatch_for_count(257),
            DispatchIndirectCommand::from_element_count(257, 64)
        );
        assert_eq!(
            rule.dispatch_for_count(0),
            DispatchIndirectCommand::new(0, 1, 1)
        );
    }

    #[test]
    fn counter_note_is_non_empty_prose() {
        let rule = CounterToDispatch::new(256);
        assert!(rule.indirect_from_counter_note().contains("ceil-division"));
    }
}
