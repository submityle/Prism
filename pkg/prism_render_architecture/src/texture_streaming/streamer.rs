//! Stateful cross-frame virtual-texture streaming orchestrator.
//!
//! The other modules in [`texture_streaming`](super) are deliberately
//! stateless: each resolves exactly one stage of a single frame
//! (`decode_feedback` -> [`TextureResidencyTable`] -> [`schedule`] ->
//! [`PhysicalPagePool`] -> [`GpuPageTable`] -> `plan_atlas_copies`). This module
//! owns the missing piece: a long-lived driver that chains those stages every
//! frame and carries the cross-frame state the per-frame primitives cannot — the
//! physical pool, the serialized page table, and the per-page history that
//! powers anti-thrash policy.
//!
//! # Per-frame pipeline
//!
//! Each call to [`VirtualTextureStreamer::stream_feedback`] (or the
//! pre-decoded [`stream_demands`](VirtualTextureStreamer::stream_demands)) runs,
//! in order:
//!
//! 1. decode every texture's `GPU` min-mip feedback grid into deduplicated
//!    [`PageDemand`]s;
//! 2. decay every tracked page's retained priority, then overwrite the priority
//!    of pages requested this frame with their freshly scored demand;
//! 3. assemble a [`TextureResidencyTable`] from the pages currently resident in
//!    the pool plus the debounce-eligible load candidates;
//! 4. run the byte-budget [`schedule`] and apply its plan to the
//!    [`PhysicalPagePool`];
//! 5. fold residency changes back into the per-page history;
//! 6. rebuild the [`GpuPageTable`] and, when atlas geometry is configured, the
//!    [`AtlasCopyPlan`](super::atlas::AtlasCopyPlan);
//! 7. drop fully cold pages so the history stays bounded.
//!
//! # Anti-thrash policy
//!
//! A naive per-frame greedy scheduler thrashes: a page that flickers in and out
//! of a view, or that is contested near the budget edge, is loaded and evicted
//! repeatedly, wasting bandwidth. Three integer mechanisms (all tuned through
//! [`StreamerConfig`]) damp that:
//!
//! * **Load debounce** withholds a page's upload until it has been requested for
//!   several consecutive frames, so a one-frame graze never pays a round trip; a
//!   configurable priority threshold lets a hard cut bypass it.
//! * **Retention decay** lowers an unrequested page's priority gradually instead
//!   of dropping it to zero, so a page that briefly leaves the view lingers
//!   resident rather than being dropped and immediately reloaded.
//! * **Eviction protection** gives a freshly uploaded page a scheduling bonus
//!   for a few frames, so it is not evicted the instant a competing demand
//!   appears before it has been used.
//!
//! The driver holds no `GPU` handle. The device-side twin records the copies in
//! each frame's [`StreamerFrame::uploads`] / atlas plan and uploads
//! [`GpuPageTable::words`]; everything here is computed deterministically on the
//! `CPU`.

use super::feedback::PageDemand;
use super::feedback_decode::{decode_feedback, FeedbackTextureDesc};
use super::indirection::{GpuPageTable, PageResolution};
use super::pool::{PageUpload, PhysicalPagePool};
use super::residency::TextureResidencyTable;
use super::scheduler::schedule;
use super::streamer_config::StreamerConfig;
use super::streamer_frame::StreamerFrame;
use super::{atlas::plan_atlas_copies, mip_tail::mip_tail_covers, TexturePageKey};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

/// Priority stamped on a mip-tail floor page so it outranks ordinary demands
/// and the scheduler always admits (and never evicts) it under budget. Half of
/// [`u64::MAX`] leaves ample headroom for the protection bonus added later
/// without risking saturation.
const MIP_TAIL_PRIORITY: u64 = u64::MAX >> 1;

/// One texture's feedback grid for a single frame.
///
/// Pairs the immutable [`FeedbackTextureDesc`] describing a streamable texture
/// with the row-major min-mip grid the `GPU` wrote for it this frame.
#[derive(Clone, Copy, Debug)]
pub struct FeedbackInput<'a> {
    /// Description of the texture whose grid this is.
    pub desc: FeedbackTextureDesc,
    /// Row-major min-mip feedback grid, `desc.grid_len()` bytes.
    pub grid: &'a [u8],
}

impl<'a> FeedbackInput<'a> {
    /// Pairs a texture description with its feedback grid.
    #[must_use]
    pub const fn new(desc: FeedbackTextureDesc, grid: &'a [u8]) -> Self {
        Self { desc, grid }
    }
}

/// Per-page cross-frame history the driver maintains outside the stateless
/// per-frame primitives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PageTrack {
    /// Retained streaming priority: set to the demand score on a requested
    /// frame, decayed on an unrequested one.
    effective_priority: u64,
    /// Physical byte cost of the page, from its most recent demand.
    byte_cost: u64,
    /// Last frame the page was requested, or `0` if never.
    last_demand_frame: u64,
    /// Consecutive frames the page has been requested up to and including
    /// `last_demand_frame`; drives the load debounce.
    demand_streak: u32,
    /// Frame the page became resident, or `None` while not resident; drives the
    /// eviction protection window.
    resident_since: Option<u64>,
}

/// Long-lived driver that streams virtual-texture pages across frames.
///
/// Construct with [`VirtualTextureStreamer::new`], then call
/// [`stream_feedback`](Self::stream_feedback) once per frame with the frame's
/// `GPU` feedback grids. Between frames, [`page_table`](Self::page_table)
/// exposes the serialized indirection buffer and [`pool`](Self::pool) the
/// physical residency, both ready for the device-side twin to consume.
#[derive(Clone, Debug)]
pub struct VirtualTextureStreamer {
    config: StreamerConfig,
    pool: PhysicalPagePool,
    page_table: GpuPageTable,
    tracks: BTreeMap<TexturePageKey, PageTrack>,
    /// Pages seated in the pool whose staging upload has not yet completed.
    /// Non-empty only when an upload budget throttles a frame; these pages are
    /// withheld from the published page table until drained.
    pending: BTreeSet<TexturePageKey>,
    frame: u64,
}

impl VirtualTextureStreamer {
    /// Builds a streamer with `config` and a physical pool of `pool_capacity`
    /// tile slots.
    ///
    /// `config` is passed through [`StreamerConfig::sanitized`] so degenerate
    /// knobs cannot stall the loop. The caller should size `pool_capacity` to at
    /// least `config.byte_budget` divided by the per-page byte cost; a pool
    /// smaller than the byte budget simply drops the loads it cannot seat
    /// (never fatal), capping residency at the slot count.
    #[must_use]
    pub fn new(config: StreamerConfig, pool_capacity: u32) -> Self {
        Self {
            config: config.sanitized(),
            pool: PhysicalPagePool::new(pool_capacity),
            page_table: GpuPageTable::new(),
            tracks: BTreeMap::new(),
            pending: BTreeSet::new(),
            frame: 0,
        }
    }

    /// The active configuration (after sanitization).
    #[must_use]
    pub const fn config(&self) -> &StreamerConfig {
        &self.config
    }

    /// The physical page pool and its current residency.
    #[must_use]
    pub const fn pool(&self) -> &PhysicalPagePool {
        &self.pool
    }

    /// The serialized `GPU` page table reflecting the current residency.
    #[must_use]
    pub const fn page_table(&self) -> &GpuPageTable {
        &self.page_table
    }

    /// Resolves `key` against the current page table using the configured
    /// [`mip_tail_floor`](StreamerConfig::mip_tail_floor) as the coarsest
    /// fallback, so a caller samples with the exact floor this streamer pinned.
    ///
    /// When a mip-tail floor is configured, every demanded page's covering floor
    /// page is forced resident (see the per-frame pipeline), so this returns
    /// [`Some`] for any page demanded since the covers converged — the floor is
    /// the guaranteed fallback and the result never holes. With no floor
    /// configured the coarsest probe is `key.mip` itself, so this resolves only
    /// the exact page and returns [`None`] until it streams in, matching plain
    /// demand-driven residency. This threads the pinned floor automatically so a
    /// caller cannot pass a coarser bound than was actually guaranteed resident.
    #[must_use]
    pub fn resolve(&self, key: TexturePageKey) -> Option<PageResolution> {
        let coarsest = self.config.mip_tail_floor.unwrap_or(key.mip);
        self.page_table.resolve(key, coarsest)
    }

    /// The number of frames streamed so far.
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// The number of pages currently tracked (resident plus still-warm history).
    #[must_use]
    pub fn tracked_pages(&self) -> usize {
        self.tracks.len()
    }

    /// Total physical bytes resident right now, measured from the pool.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        let mut total = 0u64;
        for (key, _slot) in self.pool.iter() {
            if let Some(track) = self.tracks.get(&key) {
                total = total.saturating_add(track.byte_cost);
            }
        }
        total
    }

    /// Streams one frame from raw `GPU` feedback grids.
    ///
    /// Decodes every input into [`PageDemand`]s (a page not yet resident scores
    /// against a `None` resident mip, so missing detail outranks a refinement),
    /// then advances the pipeline. Returns the frame's resolved work.
    pub fn stream_feedback(&mut self, feedback: &[FeedbackInput<'_>]) -> StreamerFrame {
        let next_frame = self.frame.saturating_add(1);
        let mut demands: Vec<PageDemand> = Vec::new();
        {
            let pool = &self.pool;
            for input in feedback {
                let decoded = decode_feedback(
                    &input.desc,
                    input.grid,
                    |key| {
                        if pool.contains(key) {
                            Some(key.mip)
                        } else {
                            None
                        }
                    },
                    next_frame,
                );
                demands.extend(decoded);
            }
        }
        self.step(&demands)
    }

    /// Streams one frame from already-decoded demands.
    ///
    /// Equivalent to [`stream_feedback`](Self::stream_feedback) for callers that
    /// decode feedback themselves or synthesize demand directly.
    pub fn stream_demands(&mut self, demands: &[PageDemand]) -> StreamerFrame {
        self.step(demands)
    }

    /// The shared per-frame pipeline behind both public entry points.
    fn step(&mut self, demands: &[PageDemand]) -> StreamerFrame {
        self.frame = self.frame.saturating_add(1);
        let frame = self.frame;
        let weights = self.config.semantic_weights;

        // (1) Decay every tracked page. Pages requested this frame are
        // overwritten in (2); decaying them first is harmless.
        for track in self.tracks.values_mut() {
            let proportional = track.effective_priority >> self.config.decay_shift;
            let step = proportional.max(self.config.decay_min);
            track.effective_priority = track.effective_priority.saturating_sub(step);
        }

        // (2) Fold in this frame's demands: refresh recency, advance the demand
        // streak, adopt the latest byte cost, and set the retained priority to
        // the freshly scored demand.
        for demand in demands {
            let priority = demand.priority(&weights);
            let track = self.tracks.entry(demand.key).or_insert(PageTrack {
                effective_priority: 0,
                byte_cost: demand.byte_cost,
                last_demand_frame: 0,
                demand_streak: 0,
                resident_since: None,
            });
            let consecutive = track.last_demand_frame.saturating_add(1) == frame;
            track.demand_streak = if consecutive {
                track.demand_streak.saturating_add(1)
            } else {
                1
            };
            track.last_demand_frame = frame;
            track.byte_cost = demand.byte_cost;
            track.effective_priority = priority;
        }

        // (2b) Mip-tail residency floor: when configured, force the covering
        // floor-mip page of every page demanded this frame to be resident at top
        // priority. This is a correctness floor, not a demand: it bypasses the
        // load debounce and outranks ordinary pages so the scheduler never
        // evicts it under budget, guaranteeing `GpuPageTable::resolve` can always
        // fall back to at least the floor mip and never returns a hole. A cover
        // page that coincides with a real demand simply has its priority lifted;
        // one never demanded is minted here with a representative byte cost (the
        // largest demanded this frame, a safe upper bound for budgeting). With no
        // floor configured this stage is skipped and the schedule is unchanged.
        if let Some(floor) = self.config.mip_tail_floor {
            let cover_byte_cost = demands.iter().map(|demand| demand.byte_cost).max();
            if let Some(byte_cost) = cover_byte_cost {
                for cover in mip_tail_covers(demands.iter().map(|demand| demand.key), floor) {
                    let track = self.tracks.entry(cover).or_insert(PageTrack {
                        effective_priority: 0,
                        byte_cost,
                        last_demand_frame: 0,
                        demand_streak: 0,
                        resident_since: None,
                    });
                    track.last_demand_frame = frame;
                    track.demand_streak = track.demand_streak.max(self.config.min_demand_frames);
                    track.effective_priority = track.effective_priority.max(MIP_TAIL_PRIORITY);
                }
            }
        }

        // (3) Assemble this frame's scheduler input: every resident page is a
        // candidate (so the scheduler may keep or evict it), and every requested
        // non-resident page that clears the debounce is a load candidate.
        let mut table = TextureResidencyTable::new();
        let mut deferred_loads = 0usize;
        let mut demanded_pages = 0usize;
        for (key, track) in &self.tracks {
            if track.last_demand_frame == frame {
                demanded_pages += 1;
            }
            if let Some(since) = track.resident_since {
                let mut priority = track.effective_priority;
                if frame.saturating_sub(since) < self.config.min_resident_frames {
                    priority = priority.saturating_add(self.config.protection_bonus);
                }
                let recency = track.last_demand_frame.max(since);
                table.request(*key, priority, track.byte_cost, recency);
                table.mark_resident(*key);
            } else if track.last_demand_frame == frame {
                let bypass = self
                    .config
                    .high_priority_bypass
                    .is_some_and(|threshold| track.effective_priority >= threshold);
                if bypass || track.demand_streak >= self.config.min_demand_frames {
                    table.request(*key, track.effective_priority, track.byte_cost, frame);
                } else {
                    deferred_loads += 1;
                }
            }
        }

        // (4) Schedule within the hard byte budget and seat the admitted set in
        // the physical pool. `apply_plan` returns the pages newly seated this
        // frame; their staging upload is resolved in stage (5b).
        let plan = schedule(&table, self.config.byte_budget);
        let admitted = self.pool.apply_plan(&plan);

        // (5a) Reflect residency changes back into the history and the pending
        // upload set. A page leaving the pool can never owe an upload; a page
        // entering it owes one until stage (5b) drains it.
        for key in &plan.evicts {
            if let Some(track) = self.tracks.get_mut(key) {
                track.resident_since = None;
            }
            self.pending.remove(key);
        }
        for upload in &admitted {
            if let Some(track) = self.tracks.get_mut(&upload.key) {
                track.resident_since = Some(frame);
            }
            self.pending.insert(upload.key);
        }

        // (5b) Resolve this frame's actual uploads from the pending backlog,
        // honouring the optional per-frame upload-bandwidth budget. Without a
        // budget the whole backlog uploads at once; with one, the
        // highest-priority pending pages upload until the staging budget is
        // spent and the rest carry forward.
        let (uploads, uploaded_bytes) = self.drain_pending_uploads();

        // (6) Rebuild the GPU page table from the uploaded resident set only
        // (pages still owing an upload are withheld so a shader never samples a
        // slot whose tile data has not landed yet), plus the optional atlas plan
        // over the pages uploaded this frame.
        self.page_table = self.rebuild_page_table();
        let atlas = self
            .config
            .atlas
            .as_ref()
            .map(|geometry| plan_atlas_copies(geometry, &uploads));

        // (7) Drop fully cold pages so the history stays bounded: keep anything
        // resident, still carrying retained priority, or requested this frame.
        self.tracks.retain(|_, track| {
            track.resident_since.is_some()
                || track.effective_priority > 0
                || track.last_demand_frame == frame
        });

        // (8) Measure truthful post-apply telemetry from the pool.
        let resident_bytes = self.resident_bytes();
        let resident_count = self.pool.resident_count() as usize;
        let pending_uploads = self.pending.len();

        StreamerFrame {
            plan,
            uploads,
            atlas,
            resident_bytes,
            resident_count,
            demanded_pages,
            deferred_loads,
            uploaded_bytes,
            pending_uploads,
        }
    }

    /// Retained streaming priority of a tracked page, or `0` if untracked.
    fn effective_priority_of(&self, key: TexturePageKey) -> u64 {
        self.tracks
            .get(&key)
            .map_or(0, |track| track.effective_priority)
    }

    /// Latest physical byte cost of a tracked page, or `0` if untracked.
    fn byte_cost_of(&self, key: TexturePageKey) -> u64 {
        self.tracks.get(&key).map_or(0, |track| track.byte_cost)
    }

    /// Uploads pending pages subject to the optional upload-bandwidth budget,
    /// returning the frame's upload list (key-ascending) and the staging bytes
    /// it consumed. Uploaded pages are removed from the pending set.
    fn drain_pending_uploads(&mut self) -> (Vec<PageUpload>, u64) {
        if self.pending.is_empty() {
            return (Vec::new(), 0);
        }
        match self.config.upload_budget_bytes {
            // No budget: the whole backlog uploads this frame, key-ascending.
            None => {
                let drained = core::mem::take(&mut self.pending);
                let mut uploads = Vec::with_capacity(drained.len());
                let mut bytes = 0u64;
                for key in drained {
                    if let Some(slot) = self.pool.slot_of(key) {
                        uploads.push(PageUpload { key, slot });
                        bytes = bytes.saturating_add(self.byte_cost_of(key));
                    }
                }
                (uploads, bytes)
            }
            // Budgeted: pack the highest-priority pending pages that fit, always
            // admitting the single top page so a page larger than the whole
            // budget cannot deadlock the backlog. Lower-priority pages that
            // still fit the remaining budget are packed opportunistically.
            Some(budget) => {
                let mut ranked: Vec<(u64, u64, TexturePageKey)> = self
                    .pending
                    .iter()
                    .map(|&key| (self.effective_priority_of(key), self.byte_cost_of(key), key))
                    .collect();
                ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
                let mut selected: Vec<(TexturePageKey, u32)> = Vec::new();
                let mut bytes = 0u64;
                for (_priority, cost, key) in ranked {
                    let would = bytes.saturating_add(cost);
                    if !selected.is_empty() && would > budget {
                        continue;
                    }
                    if let Some(slot) = self.pool.slot_of(key) {
                        selected.push((key, slot));
                        bytes = would;
                    }
                }
                for (key, _slot) in &selected {
                    self.pending.remove(key);
                }
                let mut uploads: Vec<PageUpload> = selected
                    .into_iter()
                    .map(|(key, slot)| PageUpload { key, slot })
                    .collect();
                uploads.sort_unstable_by_key(|upload| upload.key);
                (uploads, bytes)
            }
        }
    }

    /// Rebuilds the published page table from the pool, withholding pages that
    /// are seated but still owe an upload.
    fn rebuild_page_table(&self) -> GpuPageTable {
        if self.pending.is_empty() {
            return GpuPageTable::from_pool(&self.pool);
        }
        let bindings: Vec<(TexturePageKey, u32)> = self
            .pool
            .iter()
            .filter(|(key, _slot)| !self.pending.contains(key))
            .collect();
        GpuPageTable::from_bindings(&bindings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_streaming::atlas::{AtlasGeometry, AtlasTileFormat};
    use crate::texture_streaming::TextureSemantic;

    const PAGE_BYTES: u64 = 1_000;

    fn key(x: u16) -> TexturePageKey {
        TexturePageKey {
            texture: 1,
            mip: 0,
            layer: 0,
            x,
            y: 0,
        }
    }

    fn demand(k: TexturePageKey, importance: u16) -> PageDemand {
        PageDemand {
            key: k,
            semantic: TextureSemantic::Color,
            desired_mip: 0,
            resident_mip: None,
            screen_importance: importance,
            byte_cost: PAGE_BYTES,
            frame: 0,
        }
    }

    #[test]
    fn single_frame_graze_is_debounced_not_loaded() {
        // Default two-frame debounce: a page requested on just one frame must
        // not trigger an upload.
        let mut streamer = VirtualTextureStreamer::new(StreamerConfig::new(100 * PAGE_BYTES), 64);
        let f = streamer.stream_demands(&[demand(key(0), 500)]);
        assert_eq!(f.loaded(), 0);
        assert_eq!(f.deferred_loads, 1);
        assert_eq!(f.demanded_pages, 1);
        assert_eq!(streamer.resident_bytes(), 0);
    }

    #[test]
    fn debounce_satisfied_loads_after_min_frames() {
        let mut streamer = VirtualTextureStreamer::new(StreamerConfig::new(100 * PAGE_BYTES), 64);
        let f1 = streamer.stream_demands(&[demand(key(0), 500)]);
        assert_eq!(f1.loaded(), 0, "first frame debounced");
        let f2 = streamer.stream_demands(&[demand(key(0), 500)]);
        assert_eq!(f2.loaded(), 1, "second consecutive frame uploads");
        assert_eq!(f2.uploads[0].key, key(0));
        assert!(streamer.pool().contains(key(0)));
    }

    #[test]
    fn broken_streak_restarts_the_debounce() {
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES).with_min_demand_frames(3),
            64,
        );
        streamer.stream_demands(&[demand(key(0), 500)]); // streak 1
        streamer.stream_demands(&[]); // gap: streak resets
        let f = streamer.stream_demands(&[demand(key(0), 500)]); // streak 1 again
        assert_eq!(f.loaded(), 0, "a gap restarts the debounce");
    }

    #[test]
    fn high_priority_bypass_loads_on_first_frame() {
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES)
                .with_min_demand_frames(10)
                .with_high_priority_bypass(Some(1)),
            64,
        );
        let f = streamer.stream_demands(&[demand(key(0), 500)]);
        assert_eq!(f.loaded(), 1, "a hard cut bypasses the debounce");
    }

    #[test]
    fn stable_demand_converges_to_steady_state() {
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES).with_min_demand_frames(1),
            64,
        );
        let demands = [demand(key(0), 500), demand(key(1), 400), demand(key(2), 300)];
        let f1 = streamer.stream_demands(&demands);
        assert_eq!(f1.loaded(), 3, "all three fit and load immediately");
        let f2 = streamer.stream_demands(&demands);
        assert!(f2.is_steady(), "unchanged demand produces no further work");
        let f3 = streamer.stream_demands(&demands);
        assert!(f3.is_steady());
        assert_eq!(streamer.pool().resident_count(), 3);
    }

    #[test]
    fn oscillating_demand_does_not_thrash() {
        // Budget holds both pages. A is always visible; B flickers every other
        // frame. Retention decay must keep B resident so it is never reloaded.
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(2 * PAGE_BYTES).with_min_demand_frames(1),
            4,
        );
        let both = [demand(key(0), 500), demand(key(1), 400)];
        let a_only = [demand(key(0), 500)];
        let f1 = streamer.stream_demands(&both);
        assert_eq!(f1.loaded(), 2, "both load on first sight");
        for i in 0..8 {
            let f = if i % 2 == 0 {
                streamer.stream_demands(&a_only)
            } else {
                streamer.stream_demands(&both)
            };
            assert!(
                f.is_steady(),
                "oscillation must not thrash the resident set (frame {i})"
            );
        }
        assert!(streamer.pool().contains(key(1)));
    }

    #[test]
    fn budget_pressure_evicts_lowest_priority() {
        // Budget holds one page; the higher-importance page wins each frame.
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(PAGE_BYTES)
                .with_min_demand_frames(1)
                .with_protection(0, 0),
            4,
        );
        let f1 = streamer.stream_demands(&[demand(key(0), 10), demand(key(1), 900)]);
        assert_eq!(f1.loaded(), 1);
        assert_eq!(f1.uploads[0].key, key(1), "higher importance loads");
        assert!(f1.plan.evicts.is_empty());
        // Now key(0) is the important one; it must evict the resident key(1).
        let f2 = streamer.stream_demands(&[demand(key(0), 900), demand(key(1), 10)]);
        assert_eq!(f2.loaded(), 1, "the newly important page loads");
        assert_eq!(f2.uploads[0].key, key(0));
        assert_eq!(f2.plan.evicts, alloc::vec![key(1)]);
        assert!(streamer.pool().contains(key(0)));
        assert!(!streamer.pool().contains(key(1)));
    }

    #[test]
    fn protection_window_shields_a_fresh_page() {
        // A huge protection bonus keeps a just-loaded page resident even against
        // a higher-importance competitor, until the window expires.
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(PAGE_BYTES)
                .with_min_demand_frames(1)
                .with_protection(3, 100_000_000),
            4,
        );
        streamer.stream_demands(&[demand(key(0), 10)]); // load A, resident_since = 1
        // Frame 2: B is far more important, but A is still protected.
        let f2 = streamer.stream_demands(&[demand(key(0), 10), demand(key(1), 1000)]);
        assert!(f2.plan.evicts.is_empty(), "protected page is not evicted");
        assert_eq!(f2.loaded(), 0, "budget is full, so B cannot load yet");
        assert!(streamer.pool().contains(key(0)));
    }

    #[test]
    fn retention_decay_delays_then_allows_eviction() {
        // Isolate decay (no protection). A loads at high priority, then is
        // abandoned; a steady competitor B should only win once A's priority has
        // decayed below B's.
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(PAGE_BYTES)
                .with_min_demand_frames(1)
                .with_protection(0, 0),
            4,
        );
        streamer.stream_demands(&[demand(key(0), 1000)]); // frame 1: load A
        assert!(streamer.pool().contains(key(0)));
        // Frames 2..: A abandoned, B steadily requested at a lower base score.
        let b = [demand(key(1), 0)];
        let f2 = streamer.stream_demands(&b);
        assert!(f2.is_steady(), "A retained: decay not yet below B");
        let f3 = streamer.stream_demands(&b);
        assert!(f3.is_steady(), "A still retained");
        let f4 = streamer.stream_demands(&b);
        assert_eq!(f4.plan.evicts, alloc::vec![key(0)], "A finally decays out");
        assert_eq!(f4.uploads[0].key, key(1));
        assert!(streamer.pool().contains(key(1)));
    }

    #[test]
    fn page_table_matches_pool_after_frames() {
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES).with_min_demand_frames(1),
            64,
        );
        for _ in 0..3 {
            streamer.stream_demands(&[demand(key(0), 500), demand(key(5), 300), demand(key(2), 400)]);
        }
        let rebuilt = GpuPageTable::from_pool(streamer.pool());
        assert_eq!(streamer.page_table().words(), rebuilt.words());
        for (resident, slot) in streamer.pool().iter() {
            assert_eq!(streamer.page_table().lookup(resident), Some(slot));
        }
    }

    #[test]
    fn atlas_plan_tracks_uploads_when_configured() {
        let format = AtlasTileFormat {
            block_extent_px: 1,
            bytes_per_block: 4,
        };
        let geometry = AtlasGeometry::new(128, 8, format).expect("valid geometry");
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES)
                .with_min_demand_frames(1)
                .with_atlas(geometry),
            64,
        );
        let f = streamer.stream_demands(&[demand(key(0), 500), demand(key(1), 400)]);
        let plan = f.atlas.expect("atlas configured");
        assert_eq!(plan.copies.len(), f.uploads.len());
        for (copy, upload) in plan.copies.iter().zip(&f.uploads) {
            assert_eq!(copy.key, upload.key);
            assert_eq!(copy.slot, upload.slot);
        }
    }

    #[test]
    fn no_atlas_plan_when_unconfigured() {
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES).with_min_demand_frames(1),
            64,
        );
        let f = streamer.stream_demands(&[demand(key(0), 500)]);
        assert!(f.atlas.is_none());
    }

    #[test]
    fn cold_pages_are_garbage_collected() {
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES)
                .with_min_demand_frames(5)
                .with_decay(0, u64::MAX),
            64,
        );
        // A one-frame graze with an immediate full decay must leave no residue.
        streamer.stream_demands(&[demand(key(0), 1)]);
        streamer.stream_demands(&[]);
        assert_eq!(streamer.tracked_pages(), 0, "cold page forgotten");
    }

    #[test]
    fn identical_input_streams_are_deterministic() {
        let run = || {
            let mut streamer = VirtualTextureStreamer::new(
                StreamerConfig::new(2 * PAGE_BYTES).with_min_demand_frames(1),
                4,
            );
            let demands = [demand(key(0), 500), demand(key(1), 400), demand(key(2), 300)];
            let mut frames = Vec::new();
            for _ in 0..6 {
                frames.push(streamer.stream_demands(&demands));
            }
            frames
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn feedback_decode_path_drives_the_loop() {
        // Exercise the stream_feedback decode wiring end to end.
        let desc = FeedbackTextureDesc {
            texture: 1,
            layer: 0,
            semantic: TextureSemantic::Color,
            base_mip: 0,
            mip_count: 4,
            pages_x: 2,
            pages_y: 2,
            page_byte_cost: PAGE_BYTES,
            screen_importance: 500,
        };
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES).with_min_demand_frames(1),
            64,
        );
        // Cell (0,0) wants mip 0; the rest are unrequested.
        let grid = [0u8, super::super::NOT_REQUESTED, super::super::NOT_REQUESTED, super::super::NOT_REQUESTED];
        let f = streamer.stream_feedback(&[FeedbackInput::new(desc, &grid)]);
        assert_eq!(f.demanded_pages, 1);
        assert_eq!(f.loaded(), 1);
        assert_eq!(f.uploads[0].key.mip, 0);
    }

    #[test]
    fn no_upload_budget_uploads_whole_backlog_at_once() {
        // Default (unbudgeted) path: every admitted page uploads immediately and
        // nothing is left pending, preserving the original contract.
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES).with_min_demand_frames(1),
            64,
        );
        let f = streamer.stream_demands(&[demand(key(0), 300), demand(key(1), 500), demand(key(2), 400)]);
        assert_eq!(f.loaded(), 3);
        assert_eq!(f.pending_uploads, 0);
        assert_eq!(f.uploaded_bytes, 3 * PAGE_BYTES);
        assert!(f.is_steady() == false || f.plan.evicts.is_empty());
    }

    #[test]
    fn upload_budget_throttles_and_drains_in_priority_order() {
        // A one-page-per-frame upload budget: all three pages seat in the pool
        // immediately, but only the highest-priority page's data lands per frame,
        // and the rest drain over subsequent frames in priority order.
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES)
                .with_min_demand_frames(1)
                .with_protection(0, 0)
                .with_upload_budget(Some(PAGE_BYTES)),
            64,
        );
        let demands = [demand(key(0), 300), demand(key(1), 500), demand(key(2), 400)];

        let f1 = streamer.stream_demands(&demands);
        assert_eq!(f1.loaded(), 1, "only one page uploads under the budget");
        assert_eq!(f1.uploads[0].key, key(1), "highest importance uploads first");
        assert_eq!(f1.uploaded_bytes, PAGE_BYTES);
        assert_eq!(f1.pending_uploads, 2);
        assert!(!f1.is_steady(), "a backlog is not steady");
        // All three are seated in the pool, but only the uploaded one is published.
        assert!(streamer.pool().contains(key(0)));
        assert!(streamer.pool().contains(key(1)));
        assert!(streamer.pool().contains(key(2)));
        assert_eq!(streamer.page_table().lookup(key(1)), streamer.pool().slot_of(key(1)));
        assert_eq!(streamer.page_table().lookup(key(0)), None, "pending page withheld");
        assert_eq!(streamer.page_table().lookup(key(2)), None, "pending page withheld");

        let f2 = streamer.stream_demands(&demands);
        assert_eq!(f2.loaded(), 1);
        assert_eq!(f2.uploads[0].key, key(2), "next-highest drains second");
        assert_eq!(f2.pending_uploads, 1);
        assert!(f2.plan.loads.is_empty(), "no new seats: pool already holds all three");

        let f3 = streamer.stream_demands(&demands);
        assert_eq!(f3.loaded(), 1);
        assert_eq!(f3.uploads[0].key, key(0), "lowest drains last");
        assert_eq!(f3.pending_uploads, 0);

        let f4 = streamer.stream_demands(&demands);
        assert!(f4.is_steady(), "backlog cleared and resident set stable");
        assert_eq!(streamer.page_table().words(), GpuPageTable::from_pool(streamer.pool()).words());
    }

    #[test]
    fn zero_upload_budget_still_makes_progress() {
        // A zero budget must not deadlock: the single top page always uploads.
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES)
                .with_min_demand_frames(1)
                .with_protection(0, 0)
                .with_upload_budget(Some(0)),
            64,
        );
        let f = streamer.stream_demands(&[demand(key(0), 300), demand(key(1), 500)]);
        assert_eq!(f.loaded(), 1, "one page uploads even against a zero budget");
        assert_eq!(f.uploads[0].key, key(1));
        assert_eq!(f.uploaded_bytes, PAGE_BYTES);
        assert_eq!(f.pending_uploads, 1);
    }

    #[test]
    fn evicting_a_pending_page_cancels_its_upload() {
        // A page seated but not yet uploaded that gets evicted before its turn
        // must simply drop out of the backlog, never emitting a stale upload.
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(2 * PAGE_BYTES)
                .with_min_demand_frames(1)
                .with_protection(0, 0)
                .with_upload_budget(Some(PAGE_BYTES)),
            4,
        );
        // Frame 1: seat two pages (byte budget = 2 pages), upload only the top.
        let f1 = streamer.stream_demands(&[demand(key(0), 300), demand(key(1), 500)]);
        assert_eq!(f1.loaded(), 1);
        assert_eq!(f1.uploads[0].key, key(1));
        assert_eq!(f1.pending_uploads, 1, "key(0) is seated but pending");
        // Frame 2: a far more important third page forces key(0) (pending, lowest
        // priority) out of the pool before it ever uploaded.
        let f2 = streamer.stream_demands(&[demand(key(1), 500), demand(key(2), 900)]);
        assert!(f2.plan.evicts.contains(&key(0)), "the pending page is evicted");
        assert!(!streamer.pool().contains(key(0)));
        assert!(f2.uploads.iter().all(|u| u.key != key(0)), "no stale upload for an evicted page");
    }

    fn page(mip: u8, x: u16) -> TexturePageKey {
        TexturePageKey {
            texture: 1,
            mip,
            layer: 0,
            x,
            y: 0,
        }
    }

    #[test]
    fn mip_tail_floor_pins_uncovered_page_before_fine_page_streams() {
        // A floor at mip 3 under a default two-frame debounce: the fine page is
        // still debounced on frame 1, but its covering floor page must already
        // be resident so the resolver has a fallback instead of a hole.
        let floor = 3u8;
        let fine = page(0, 5);
        let cover = page(floor, 5 >> 3); // (mip 3, x 0)
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES).with_mip_tail_floor(Some(floor)),
            64,
        );
        let f = streamer.stream_demands(&[demand(fine, 500)]);
        // The fine page itself is debounced on its first sighting.
        assert!(!streamer.pool().contains(fine), "fine page is still debounced");
        assert_eq!(f.deferred_loads, 1);
        // The covering floor page, though never demanded, is forced resident.
        assert!(streamer.pool().contains(cover), "floor cover is pinned resident");
        // The resolver therefore never holes: it falls back to the floor cover.
        let resolved = streamer
            .page_table()
            .resolve(fine, floor)
            .expect("floor guarantees a resident cover");
        assert_eq!(resolved.key, cover);
        assert_eq!(resolved.mip_bias, floor);
    }

    #[test]
    fn streamer_resolve_threads_configured_floor() {
        // With a floor, streamer.resolve() uses it automatically and never holes
        // once the covers converge; without a floor it only resolves the exact
        // page and holes until that page itself streams in.
        let floor = 3u8;
        let fine = page(0, 5);
        let cover = page(floor, 5 >> 3);
        let mut pinned = VirtualTextureStreamer::new(
            StreamerConfig::new(100 * PAGE_BYTES).with_mip_tail_floor(Some(floor)),
            64,
        );
        pinned.stream_demands(&[demand(fine, 500)]);
        let r = pinned.resolve(fine).expect("floor guarantees a cover");
        assert_eq!(r.key, cover);
        assert_eq!(r.mip_bias, floor);

        let mut unpinned =
            VirtualTextureStreamer::new(StreamerConfig::new(100 * PAGE_BYTES), 64);
        unpinned.stream_demands(&[demand(fine, 500)]);
        // No floor and the fine page is still debounced: resolve finds nothing.
        assert!(unpinned.resolve(fine).is_none());
    }

    #[test]
    fn mip_tail_floor_cover_survives_byte_contention() {
        // One-page byte budget with the finest pages churning: the floor cover
        // outranks them and stays resident, so the resolver always succeeds.
        let floor = 2u8;
        let fine = page(0, 9);
        let cover = page(floor, 9 >> 2); // (mip 2, x 2)
        let mut streamer = VirtualTextureStreamer::new(
            StreamerConfig::new(PAGE_BYTES)
                .with_min_demand_frames(1)
                .with_mip_tail_floor(Some(floor)),
            64,
        );
        for _ in 0..4 {
            streamer.stream_demands(&[demand(fine, 900)]);
        }
        assert!(streamer.pool().contains(cover), "cover held under contention");
        assert!(streamer.page_table().resolve(fine, floor).is_some());
    }

    #[test]
    fn mip_tail_floor_none_matches_unset_floor() {
        // `with_mip_tail_floor(None)` must be byte-identical to never setting it.
        let mut with_none = VirtualTextureStreamer::new(
            StreamerConfig::new(2 * PAGE_BYTES)
                .with_min_demand_frames(1)
                .with_mip_tail_floor(None),
            64,
        );
        let mut unset = VirtualTextureStreamer::new(
            StreamerConfig::new(2 * PAGE_BYTES).with_min_demand_frames(1),
            64,
        );
        for frame in 0..6u16 {
            let d = [demand(key(frame % 3), 400 + frame)];
            let a = with_none.stream_demands(&d);
            let b = unset.stream_demands(&d);
            assert_eq!(a.loaded(), b.loaded());
            assert_eq!(a.evicted(), b.evicted());
            assert_eq!(a.resident_count, b.resident_count);
            assert_eq!(a.resident_bytes, b.resident_bytes);
            assert_eq!(with_none.page_table().words(), unset.page_table().words());
        }
    }
}
