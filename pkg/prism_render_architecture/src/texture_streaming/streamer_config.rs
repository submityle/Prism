//! Configuration for the cross-frame virtual-texture streamer.
//!
//! The [`VirtualTextureStreamer`](super::streamer::VirtualTextureStreamer) is a
//! stateful driver layered over the stateless per-frame primitives in this
//! module (`decode_feedback` -> residency -> `schedule` -> pool -> page table ->
//! atlas). All of its cross-frame policy knobs live here so the driver body
//! stays focused on wiring. Every field is integer-valued, so a given config and
//! feedback stream produce a bit-identical streaming schedule on any target.
//!
//! The policy has three cooperating anti-thrash mechanisms, each tunable here:
//!
//! * A load **debounce** ([`min_demand_frames`](StreamerConfig::min_demand_frames)):
//!   a page must be requested for that many consecutive frames before it is
//!   uploaded, so a page a view only grazes for a single frame never costs a
//!   round trip. A hard cut can bypass the debounce through
//!   [`high_priority_bypass`](StreamerConfig::high_priority_bypass).
//! * A retention **decay** ([`decay_shift`](StreamerConfig::decay_shift) and
//!   [`decay_min`](StreamerConfig::decay_min)): a resident page that stops being
//!   requested loses priority gradually rather than instantly, so a page that
//!   briefly leaves the view lingers resident instead of being dropped and
//!   immediately reloaded.
//! * An eviction **protection window**
//!   ([`min_resident_frames`](StreamerConfig::min_resident_frames) plus
//!   [`protection_bonus`](StreamerConfig::protection_bonus)): a freshly uploaded
//!   page is given a scheduling bonus for a few frames so it is not evicted the
//!   instant a competing demand appears, amortizing the upload it just paid for.

use super::atlas::AtlasGeometry;
use super::feedback::SemanticWeights;

/// Cross-frame policy for the virtual-texture streamer.
///
/// Construct with [`StreamerConfig::new`] (which supplies production-sensible
/// defaults for everything but the byte budget) and refine through the chained
/// `with_*` setters. [`StreamerConfig::sanitized`] clamps the handful of fields
/// whose degenerate values would stall the loop, so a driver can accept an
/// arbitrary caller config without defensively re-checking each knob.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StreamerConfig {
    /// Per-[`TextureSemantic`](super::TextureSemantic) priority weights applied
    /// when a decoded demand is scored.
    pub semantic_weights: SemanticWeights,
    /// Hard physical-pool byte budget handed to
    /// [`schedule`](super::scheduler::schedule) every frame. The resident set
    /// never exceeds it.
    pub byte_budget: u64,
    /// Consecutive frames a page must be requested before it is eligible to
    /// upload. `1` disables the debounce (load on first sight). `0` is treated
    /// as `1` by [`sanitized`](Self::sanitized).
    pub min_demand_frames: u32,
    /// Priority at or above which a page bypasses the debounce and may upload on
    /// its first requested frame, modelling a hard cut or teleport. `None`
    /// disables the bypass entirely.
    pub high_priority_bypass: Option<u64>,
    /// Right-shift applied to a non-requested page's priority each frame to
    /// decay it (`priority >> decay_shift` is the proportional step). A larger
    /// shift decays more slowly. Clamped to `0..64` by
    /// [`sanitized`](Self::sanitized).
    pub decay_shift: u32,
    /// Minimum absolute decay step subtracted each frame on top of the
    /// proportional step, guaranteeing a non-requested page's priority reaches
    /// zero in bounded time even once the proportional step rounds to zero. `0`
    /// is treated as `1` by [`sanitized`](Self::sanitized).
    pub decay_min: u64,
    /// Number of frames after upload during which a resident page receives
    /// [`protection_bonus`](Self::protection_bonus) in the scheduler, shielding
    /// it from immediate eviction. `0` disables the protection window.
    pub min_resident_frames: u64,
    /// Priority bonus added to a page still inside its protection window. A
    /// value near one mip level of urgency keeps a freshly loaded page above
    /// cold decayed pages without overriding genuinely higher live demand.
    pub protection_bonus: u64,
    /// Optional physical atlas geometry. When set, each frame resolves its
    /// uploads into an [`AtlasCopyPlan`](super::atlas::AtlasCopyPlan); when
    /// `None`, the frame reports no atlas plan and the caller lays copies out
    /// itself.
    pub atlas: Option<AtlasGeometry>,
    /// Optional per-frame upload-bandwidth budget in staging bytes. When set,
    /// the streamer uploads at most this many bytes of newly admitted pages per
    /// frame, draining any remaining admitted-but-not-yet-uploaded pages over
    /// later frames in priority order; a page is not published into the `GPU`
    /// page table until its upload completes. `None` uploads every admitted page
    /// the frame it is seated, matching an unbounded upload path.
    pub upload_budget_bytes: Option<u64>,
    /// Optional mip-tail residency floor. When `Some(floor)`, the streamer forces
    /// the `floor`-level page covering every demanded page to stay resident, so
    /// [`GpuPageTable::resolve`](super::indirection::GpuPageTable::resolve) can
    /// always fall back to at least that page and never returns a hole. `None`
    /// pins no tail, leaving coarse-page residency entirely to demand.
    pub mip_tail_floor: Option<u8>,
}

impl StreamerConfig {
    /// Builds a config for `byte_budget` physical bytes with default policy.
    ///
    /// Defaults: a two-frame load debounce, no hard-cut bypass, a `>> 3`
    /// (~12.5% per frame) retention decay with a one-unit floor, a two-frame
    /// eviction protection window worth one mip level of urgency, default
    /// semantic weights, and no atlas geometry.
    #[must_use]
    pub fn new(byte_budget: u64) -> Self {
        Self {
            semantic_weights: SemanticWeights::DEFAULT,
            byte_budget,
            min_demand_frames: 2,
            high_priority_bypass: None,
            decay_shift: 3,
            decay_min: 1,
            min_resident_frames: 2,
            protection_bonus: super::feedback::MIP_URGENCY,
            atlas: None,
            upload_budget_bytes: None,
            mip_tail_floor: None,
        }
    }

    /// Overrides the per-semantic priority weights.
    #[must_use]
    pub const fn with_semantic_weights(mut self, weights: SemanticWeights) -> Self {
        self.semantic_weights = weights;
        self
    }

    /// Sets the load debounce in consecutive requested frames.
    #[must_use]
    pub const fn with_min_demand_frames(mut self, frames: u32) -> Self {
        self.min_demand_frames = frames;
        self
    }

    /// Sets the priority threshold that bypasses the load debounce, or clears it
    /// with `None`.
    #[must_use]
    pub const fn with_high_priority_bypass(mut self, threshold: Option<u64>) -> Self {
        self.high_priority_bypass = threshold;
        self
    }

    /// Sets the retention decay shift and absolute floor step.
    #[must_use]
    pub const fn with_decay(mut self, shift: u32, min_step: u64) -> Self {
        self.decay_shift = shift;
        self.decay_min = min_step;
        self
    }

    /// Sets the eviction protection window length and its priority bonus.
    #[must_use]
    pub const fn with_protection(mut self, frames: u64, bonus: u64) -> Self {
        self.min_resident_frames = frames;
        self.protection_bonus = bonus;
        self
    }

    /// Attaches physical atlas geometry so frames resolve atlas copies.
    #[must_use]
    pub const fn with_atlas(mut self, geometry: AtlasGeometry) -> Self {
        self.atlas = Some(geometry);
        self
    }

    /// Caps the staging bytes uploaded per frame, or clears the cap with `None`.
    ///
    /// A `Some(0)` budget still admits one page per frame so the loop cannot
    /// deadlock: the highest-priority pending page is always uploaded even when
    /// its cost alone exceeds the budget.
    #[must_use]
    pub const fn with_upload_budget(mut self, staging_bytes: Option<u64>) -> Self {
        self.upload_budget_bytes = staging_bytes;
        self
    }

    /// Pins the mip-tail residency floor to `floor_mip`, or clears it with
    /// `None`.
    ///
    /// With a floor set, the covering page at `floor_mip` of every demanded page
    /// is forced resident at top priority, guaranteeing the resolver always has
    /// a fallback. `None` restores pure demand-driven residency and is
    /// behaviourally identical to leaving the floor unset.
    #[must_use]
    pub const fn with_mip_tail_floor(mut self, floor_mip: Option<u8>) -> Self {
        self.mip_tail_floor = floor_mip;
        self
    }

    /// Returns a copy with degenerate knobs clamped to values that keep the
    /// streaming loop live.
    ///
    /// `min_demand_frames` and `decay_min` are raised to `1` so a page can
    /// always eventually load and always eventually decays to zero;
    /// `decay_shift` is capped at `63` so the proportional step never
    /// invokes undefined shift behaviour. All other fields pass through, since
    /// zero is a meaningful disable for them.
    #[must_use]
    pub const fn sanitized(mut self) -> Self {
        if self.min_demand_frames == 0 {
            self.min_demand_frames = 1;
        }
        if self.decay_min == 0 {
            self.decay_min = 1;
        }
        if self.decay_shift > 63 {
            self.decay_shift = 63;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sets_documented_defaults() {
        let cfg = StreamerConfig::new(1 << 20);
        assert_eq!(cfg.byte_budget, 1 << 20);
        assert_eq!(cfg.min_demand_frames, 2);
        assert_eq!(cfg.high_priority_bypass, None);
        assert_eq!(cfg.decay_shift, 3);
        assert_eq!(cfg.decay_min, 1);
        assert_eq!(cfg.min_resident_frames, 2);
        assert_eq!(cfg.protection_bonus, super::super::feedback::MIP_URGENCY);
        assert_eq!(cfg.atlas, None);
        assert_eq!(cfg.upload_budget_bytes, None);
        assert_eq!(cfg.mip_tail_floor, None);
        assert_eq!(cfg.semantic_weights, SemanticWeights::DEFAULT);
    }

    #[test]
    fn chained_setters_apply() {
        let cfg = StreamerConfig::new(100)
            .with_min_demand_frames(4)
            .with_high_priority_bypass(Some(500_000))
            .with_decay(5, 8)
            .with_protection(6, 42);
        assert_eq!(cfg.min_demand_frames, 4);
        assert_eq!(cfg.high_priority_bypass, Some(500_000));
        assert_eq!(cfg.decay_shift, 5);
        assert_eq!(cfg.decay_min, 8);
        assert_eq!(cfg.min_resident_frames, 6);
        assert_eq!(cfg.protection_bonus, 42);
    }

    #[test]
    fn upload_budget_setter_round_trips() {
        let cfg = StreamerConfig::new(100).with_upload_budget(Some(4096));
        assert_eq!(cfg.upload_budget_bytes, Some(4096));
        let cleared = cfg.with_upload_budget(None);
        assert_eq!(cleared.upload_budget_bytes, None);
    }

    #[test]
    fn mip_tail_floor_setter_round_trips() {
        let cfg = StreamerConfig::new(100).with_mip_tail_floor(Some(3));
        assert_eq!(cfg.mip_tail_floor, Some(3));
        let cleared = cfg.with_mip_tail_floor(None);
        assert_eq!(cleared.mip_tail_floor, None);
    }

    #[test]
    fn sanitized_raises_degenerate_knobs() {
        let cfg = StreamerConfig::new(100)
            .with_min_demand_frames(0)
            .with_decay(99, 0)
            .sanitized();
        assert_eq!(cfg.min_demand_frames, 1);
        assert_eq!(cfg.decay_min, 1);
        assert_eq!(cfg.decay_shift, 63);
    }

    #[test]
    fn sanitized_preserves_meaningful_zeroes() {
        let cfg = StreamerConfig::new(100)
            .with_protection(0, 0)
            .with_high_priority_bypass(None)
            .sanitized();
        assert_eq!(cfg.min_resident_frames, 0);
        assert_eq!(cfg.protection_bonus, 0);
        assert_eq!(cfg.high_priority_bypass, None);
    }
}
