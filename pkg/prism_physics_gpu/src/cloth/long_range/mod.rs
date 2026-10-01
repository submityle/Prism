//! `GPU` colour-batched cloth long-range-attachment (LRA) leashes.
//!
//! A long-range attachment is a *one-sided* distance constraint tying one cloth
//! particle to a fixed world-space anchor: the particle moves freely while it
//! stays within `max_distance` of the anchor, but once it drifts farther it is
//! pulled back onto the leash sphere. Seeding every particle with a geodesic
//! leash to a nearby kinematic attachment removes the long-wavelength stretch
//! that pure local distance constraints need many iterations to resolve, so a
//! garment stops sagging like rubber under fast motion without an expensive
//! global solve (Kim et al. 2012; the classic `NvCloth` technique).
//!
//! The sequential Gauss-Seidel golden lives in [`prism_physics_core`] as
//! `project_long_range`; this module runs the exact same projection on the
//! `GPU` as a sequence of *graph-coloured* batches so the hardware stays
//! race-free without an intra-pass barrier.
//!
//! # Layout
//!
//! - [`coloring`] — the single-endpoint greedy colourer that partitions the
//!   leashes into independent sets (one race-free pass each) so two leashes on
//!   the *same* particle land in different colours and keep their sequential
//!   order.
//! - [`cpu`] — the [`cpu_cloth_long_range`] golden twin (delegating the
//!   per-leash arithmetic to [`prism_physics_core`]'s `project_long_range`) plus
//!   an independent brute-force anchor.
//! - [`gpu`] — the real-device [`GpuClothLongRange`] pipeline.
//!
//! # Correctness model
//!
//! A leash writes exactly one particle, so within one colour class the writes
//! touch disjoint slots and a parallel pass matches the sequential sweep
//! exactly; across colours the device runs the classes in the same ascending
//! order as the twin. The only divergence is the per-projection
//! fused-multiply-add and division/square-root rounding, so parity is verified
//! within a tight tolerance rather than bit-for-bit — the same model the rest
//! of the solver uses.
//!
//! # Provenance
//!
//! The one-sided long-range-attachment leash is a published position-based
//! dynamics technique (Kim et al., "Long Range Attachments"); greedy graph
//! colouring of the constraint conflict graph is standard batched-PBD practice.
//! No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

pub mod coloring;
pub mod cpu;
pub mod gpu;

pub use coloring::{colour_long_range, LongRangeColoring};
pub use cpu::cpu_cloth_long_range;
pub use gpu::GpuClothLongRange;

/// One one-sided long-range-attachment leash, over a raw particle index and a
/// fixed world-space anchor.
///
/// This is the `GPU` crate's public re-statement of
/// [`prism_physics_core`]'s `LongRangeConstraint`, which stores a crate-private
/// particle handle that callers outside the engine cannot build. The field
/// layout is also the exact bytes uploaded to the device (std430: `anchor`
/// occupies the first three floats, then the two scalars, the index, and two
/// pad words to the 32-byte stride), so it is `Pod`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClothLongRangeConstraint {
    /// Fixed world-space anchor the leash is measured from.
    pub anchor: [f32; 3],
    /// Maximum allowed distance from `anchor`; the leash only acts when the
    /// particle is farther than this.
    pub max_distance: f32,
    /// `XPBD` compliance (`0` is a perfectly rigid leash).
    pub compliance: f32,
    /// The leashed particle index.
    pub particle: u32,
    /// Padding to the std430 32-byte stride; always zero.
    _pad0: u32,
    /// Padding to the std430 32-byte stride; always zero.
    _pad1: u32,
}

impl ClothLongRangeConstraint {
    /// Creates a one-sided leash tying `particle` to `anchor` with the given
    /// `max_distance` (clamped non-negative) and `compliance` (clamped
    /// non-negative; `0` is a rigid leash), matching
    /// [`prism_physics_core`]'s `LongRangeConstraint::new`.
    #[must_use]
    pub fn new(particle: u32, anchor: Vec3, max_distance: f32, compliance: f32) -> Self {
        ClothLongRangeConstraint {
            anchor: anchor.to_array(),
            max_distance: max_distance.max(0.0),
            compliance: compliance.max(0.0),
            particle,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Returns the anchor as a [`Vec3`].
    #[must_use]
    pub fn anchor(&self) -> Vec3 {
        Vec3::from_array(self.anchor)
    }
}
