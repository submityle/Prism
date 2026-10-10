//! The automatic barrier solver: tracks the current synchronization state of
//! every buffer and texture subresource and emits the minimal [`Barrier`] set
//! needed to make each new use correct.
//!
//! Explicit APIs (Vulkan/D3D12/Metal) require the application to insert memory
//! and layout barriers by hand; getting them wrong causes corruption or
//! GPU hangs that are brutal to debug. Prism follows the approach proven by
//! engines like O3DE/Granite and by wgpu's internal `hub` trackers: the user
//! declares *how* each resource is used in each pass, and a tracker derives the
//! transitions. This keeps the ergonomic win of implicit synchronization while
//! preserving the "escape hatch" of hand-authored barriers for the rare case a
//! heuristic is too conservative.
//!
//! What the solver gets right:
//! - **Hazard classification.** Read-after-read with a matching layout needs no
//!   barrier; the tracker instead *merges* the reads so a later writer waits on
//!   the union of all readers. Write-after-read, read-after-write, and
//!   write-after-write always emit a barrier.
//! - **Subresource granularity.** Textures are tracked per (mip, layer) so a
//!   transition of one mip does not over-synchronize the rest. Uniform ranges
//!   coalesce back into a single barrier.
//! - **Layout transitions.** Texture barriers carry before/after
//!   [`TextureLayout`], the main reason texture (vs. buffer) barriers exist.
//! - **Split barriers.** [`split_barrier`] turns an immediate barrier into a
//!   `Begin`/`End` pair so the backend can overlap the transition with
//!   unrelated work.
//! - **Batching + dedup.** [`optimize_batch`] drops no-op/redundant barriers
//!   from a recorded batch before submission.
//!
//! Single-threaded (`&mut self`); the backend wraps it per queue. `no_std`,
//! no `unsafe`, deterministic (`BTreeMap`-keyed).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::sync::{
    merge_read_buffer_states, merge_read_texture_states, Barrier, BarrierKind, BufferBarrier,
    BufferState, SubresourceRange, TextureBarrier, TextureState,
};

/// Per-texture tracking metadata: dimensions plus the current state of every
/// subresource.
struct TextureTracking {
    mip_levels: u32,
    array_layers: u32,
    /// Flat `mip_levels * array_layers` grid, row-major by mip then layer:
    /// `state[mip * array_layers + layer]`.
    states: Vec<TextureState>,
}

impl TextureTracking {
    fn new(mip_levels: u32, array_layers: u32, initial: TextureState) -> Self {
        let mips = mip_levels.max(1);
        let layers = array_layers.max(1);
        let count = (mips as usize) * (layers as usize);
        Self {
            mip_levels: mips,
            array_layers: layers,
            states: alloc::vec![initial; count],
        }
    }

    #[inline]
    fn idx(&self, mip: u32, layer: u32) -> usize {
        (mip as usize) * (self.array_layers as usize) + (layer as usize)
    }

    /// Resolves a possibly-`MAX` subresource range to concrete inclusive-exclusive
    /// mip/layer bounds clamped to the texture's dimensions.
    fn resolve(&self, range: SubresourceRange) -> (u32, u32, u32, u32) {
        let mip_start = range.base_mip_level.min(self.mip_levels);
        let mip_end = range
            .base_mip_level
            .saturating_add(range.mip_level_count)
            .min(self.mip_levels);
        let layer_start = range.base_array_layer.min(self.array_layers);
        let layer_end = range
            .base_array_layer
            .saturating_add(range.array_layer_count)
            .min(self.array_layers);
        (mip_start, mip_end, layer_start, layer_end)
    }
}

/// Tracks GPU resource state and solves for the barriers each use requires.
///
/// Generic over the backend's buffer handle `B` and texture handle `T` so the
/// solver never depends on concrete RHI id types; both must be `Ord + Copy`
/// (used as deterministic map keys).
pub struct StateTracker<B: Ord + Copy, T: Ord + Copy> {
    buffers: BTreeMap<B, BufferState>,
    textures: BTreeMap<T, TextureTracking>,
}

impl<B: Ord + Copy, T: Ord + Copy> StateTracker<B, T> {
    /// Creates an empty tracker.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            textures: BTreeMap::new(),
        }
    }

    /// Registers a buffer with its initial state (typically
    /// [`BufferState::initial`]). Re-registering resets the tracked state.
    pub fn register_buffer(&mut self, handle: B, initial: BufferState) {
        self.buffers.insert(handle, initial);
    }

    /// Registers a texture with its mip/layer counts and initial state
    /// (typically [`TextureState::initial`], i.e. `Undefined`). Dimensions of
    /// zero are treated as one.
    pub fn register_texture(
        &mut self,
        handle: T,
        mip_levels: u32,
        array_layers: u32,
        initial: TextureState,
    ) {
        self.textures.insert(
            handle,
            TextureTracking::new(mip_levels, array_layers, initial),
        );
    }

    /// Forgets a buffer (e.g. on destruction) so a recycled handle starts
    /// fresh.
    pub fn forget_buffer(&mut self, handle: B) {
        self.buffers.remove(&handle);
    }

    /// Forgets a texture.
    pub fn forget_texture(&mut self, handle: T) {
        self.textures.remove(&handle);
    }

    /// The current tracked state of a buffer, if registered.
    #[must_use]
    pub fn buffer_state(&self, handle: B) -> Option<BufferState> {
        self.buffers.get(&handle).copied()
    }

    /// The current tracked state of a single texture subresource, if
    /// registered and in range.
    #[must_use]
    pub fn texture_state(&self, handle: T, mip: u32, layer: u32) -> Option<TextureState> {
        let t = self.textures.get(&handle)?;
        if mip >= t.mip_levels || layer >= t.array_layers {
            return None;
        }
        t.states.get(t.idx(mip, layer)).copied()
    }

    /// Declares a use of `handle` in `next` and returns the barrier required to
    /// make that use correct, or `None` if none is needed (read-after-read with
    /// no write). Auto-registers an unknown buffer at [`BufferState::initial`]
    /// before transitioning.
    ///
    /// On a read-after-read the tracked state becomes the *union* of the reads
    /// so a subsequent writer is ordered after all of them.
    pub fn use_buffer(&mut self, handle: B, next: BufferState) -> Option<Barrier<B, T>> {
        let current = *self
            .buffers
            .entry(handle)
            .or_insert_with(BufferState::initial);

        // Read-after-read with no hazard: merge, no barrier.
        if !current.is_write()
            && !next.is_write()
            && let Some(merged) = merge_read_buffer_states(current, next)
        {
            self.buffers.insert(handle, merged);
            return None;
        }

        // Hazard (W->R, R->W, W->W, or cross-queue): emit a barrier and adopt
        // the new state.
        self.buffers.insert(handle, next);
        if current == next && !current.is_write() {
            return None;
        }
        Some(Barrier::Buffer(BufferBarrier {
            buffer: handle,
            before: current,
            after: next,
            kind: BarrierKind::Immediate,
        }))
    }

    /// Declares a use of a texture subresource range in `next` and returns the
    /// barriers required. Subresources already in a compatible read state are
    /// merged and skipped; the rest transition. Barriers over a uniform
    /// before-state range are coalesced into one. Auto-registers an unknown
    /// texture as a single-subresource `Undefined` texture.
    pub fn use_texture(
        &mut self,
        handle: T,
        range: SubresourceRange,
        next: TextureState,
    ) -> Vec<Barrier<B, T>> {
        let t = self
            .textures
            .entry(handle)
            .or_insert_with(|| TextureTracking::new(1, 1, TextureState::initial()));

        let (mip_start, mip_end, layer_start, layer_end) = t.resolve(range);
        let mut out: Vec<Barrier<B, T>> = Vec::new();

        // Fast path: the whole requested range shares one before-state.
        // Emit a single coalesced barrier (or none for a pure read-merge).
        let mut uniform_before: Option<TextureState> = None;
        let mut is_uniform = true;
        for mip in mip_start..mip_end {
            for layer in layer_start..layer_end {
                let s = t.states[t.idx(mip, layer)];
                match uniform_before {
                    None => uniform_before = Some(s),
                    Some(u) if u == s => {}
                    Some(_) => {
                        is_uniform = false;
                    }
                }
                if !is_uniform {
                    break;
                }
            }
            if !is_uniform {
                break;
            }
        }

        if is_uniform {
            if let Some(before) = uniform_before {
                if let Some(bar) = Self::transition_one(
                    handle,
                    before,
                    next,
                    SubresourceRange {
                        base_mip_level: mip_start,
                        mip_level_count: mip_end.saturating_sub(mip_start),
                        base_array_layer: layer_start,
                        array_layer_count: layer_end.saturating_sub(layer_start),
                    },
                ) {
                    out.push(bar);
                }
                let adopted = Self::adopt(before, next);
                for mip in mip_start..mip_end {
                    for layer in layer_start..layer_end {
                        let i = t.idx(mip, layer);
                        t.states[i] = adopted;
                    }
                }
            }
            return out;
        }

        // Mixed before-states: transition each subresource individually.
        for mip in mip_start..mip_end {
            for layer in layer_start..layer_end {
                let i = t.idx(mip, layer);
                let before = t.states[i];
                if let Some(bar) =
                    Self::transition_one(handle, before, next, SubresourceRange::single(mip, layer))
                {
                    out.push(bar);
                }
                t.states[i] = Self::adopt(before, next);
            }
        }
        out
    }

    /// Computes the barrier (if any) for a single uniform before→after
    /// transition over `range`.
    fn transition_one(
        handle: T,
        before: TextureState,
        next: TextureState,
        range: SubresourceRange,
    ) -> Option<Barrier<B, T>> {
        // Read-after-read with the same layout: no hazard.
        if !before.is_write()
            && !next.is_write()
            && before.layout == next.layout
            && merge_read_texture_states(before, next).is_some()
        {
            return None;
        }
        // Identical read state: nothing to do.
        if before == next && !before.is_write() {
            return None;
        }
        Some(Barrier::Texture(TextureBarrier {
            texture: handle,
            range,
            before,
            after: next,
            kind: BarrierKind::Immediate,
        }))
    }

    /// The state a subresource adopts after a use: the merged read union when
    /// both sides are compatible reads, otherwise simply `next`.
    fn adopt(before: TextureState, next: TextureState) -> TextureState {
        if !before.is_write()
            && !next.is_write()
            && let Some(m) = merge_read_texture_states(before, next)
        {
            return m;
        }
        next
    }
}

impl<B: Ord + Copy, T: Ord + Copy> Default for StateTracker<B, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Splits an immediate barrier into a `Begin`/`End` pair for a split-barrier
/// submission. Record the `Begin` right after the resource's last use and the
/// `End` right before its next use; the GPU overlaps the transition with work
/// recorded in between. Returns the pair `(begin, end)`.
#[must_use]
pub fn split_barrier<B: Copy, T: Copy>(barrier: Barrier<B, T>) -> (Barrier<B, T>, Barrier<B, T>) {
    match barrier {
        Barrier::Buffer(b) => (
            Barrier::Buffer(BufferBarrier {
                kind: BarrierKind::Begin,
                ..b
            }),
            Barrier::Buffer(BufferBarrier {
                kind: BarrierKind::End,
                ..b
            }),
        ),
        Barrier::Texture(t) => (
            Barrier::Texture(TextureBarrier {
                kind: BarrierKind::Begin,
                ..t
            }),
            Barrier::Texture(TextureBarrier {
                kind: BarrierKind::End,
                ..t
            }),
        ),
    }
}

/// Removes no-op barriers (identical, non-writing before/after on the same
/// resource) from a recorded batch in place. A cheap final pass before handing
/// the batch to the backend; the per-use solver already avoids most redundancy,
/// but merged passes can still accumulate trivial entries.
pub fn optimize_batch<B: Copy + PartialEq, T: Copy + PartialEq>(batch: &mut Vec<Barrier<B, T>>) {
    batch.retain(|b| match b {
        Barrier::Buffer(bb) => bb.before != bb.after || bb.before.is_write(),
        Barrier::Texture(tb) => {
            tb.before != tb.after || tb.before.is_write() || tb.before.layout != tb.after.layout
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::PipelineStages;

    type Tracker = StateTracker<u32, u32>;

    #[test]
    fn write_after_write_emits_barrier() {
        let mut t = Tracker::new();
        t.register_buffer(1, BufferState::initial());
        // First write: initial(no access) -> storage_write is a transition.
        let b1 = t.use_buffer(
            1,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        assert!(b1.is_some());
        // Second write: W->W hazard.
        let b2 = t.use_buffer(
            1,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        assert!(b2.is_some(), "write-after-write must barrier");
    }

    #[test]
    fn read_after_read_merges_without_barrier() {
        let mut t = Tracker::new();
        t.register_buffer(1, BufferState::uniform(PipelineStages::VERTEX_SHADER));
        let b = t.use_buffer(
            1,
            BufferState::storage_read(PipelineStages::FRAGMENT_SHADER),
        );
        assert!(b.is_none(), "read-after-read needs no barrier");
        let merged = t.buffer_state(1).unwrap();
        assert!(merged.stages.contains(PipelineStages::VERTEX_SHADER));
        assert!(merged.stages.contains(PipelineStages::FRAGMENT_SHADER));
    }

    #[test]
    fn write_after_read_emits_barrier() {
        let mut t = Tracker::new();
        t.register_buffer(1, BufferState::vertex());
        let b = t.use_buffer(
            1,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        assert!(b.is_some(), "write-after-read must barrier");
    }

    #[test]
    fn texture_layout_transition_barrier() {
        let mut t = Tracker::new();
        t.register_texture(10, 1, 1, TextureState::initial());
        let bars = t.use_texture(10, SubresourceRange::all(), TextureState::color_target());
        assert_eq!(bars.len(), 1);
        if let Barrier::Texture(tb) = bars[0] {
            assert_eq!(tb.before.layout, crate::sync::TextureLayout::Undefined);
            assert_eq!(tb.after.layout, crate::sync::TextureLayout::ColorAttachment);
        } else {
            panic!("expected texture barrier");
        }
        // Transition to shader-read then re-read: second read needs no barrier.
        let b2 = t.use_texture(
            10,
            SubresourceRange::all(),
            TextureState::shader_read(PipelineStages::FRAGMENT_SHADER),
        );
        assert_eq!(b2.len(), 1, "color->shader_read is a layout change");
        let b3 = t.use_texture(
            10,
            SubresourceRange::all(),
            TextureState::shader_read(PipelineStages::VERTEX_SHADER),
        );
        assert!(b3.is_empty(), "read->read same layout: no barrier");
    }

    #[test]
    fn subresource_granularity() {
        let mut t = Tracker::new();
        t.register_texture(10, 3, 1, TextureState::initial());
        // Transition only mip 0 to color target.
        let b = t.use_texture(
            10,
            SubresourceRange::single(0, 0),
            TextureState::color_target(),
        );
        assert_eq!(b.len(), 1);
        // mip 1 is still Undefined.
        assert_eq!(
            t.texture_state(10, 1, 0).unwrap().layout,
            crate::sync::TextureLayout::Undefined
        );
        assert_eq!(
            t.texture_state(10, 0, 0).unwrap().layout,
            crate::sync::TextureLayout::ColorAttachment
        );
    }

    #[test]
    fn mixed_before_states_transition_individually() {
        let mut t = Tracker::new();
        t.register_texture(10, 2, 1, TextureState::initial());
        // Put mip 0 in color, leave mip 1 undefined.
        t.use_texture(
            10,
            SubresourceRange::single(0, 0),
            TextureState::color_target(),
        );
        // Now transition the whole texture to shader_read: mip0 (color) and
        // mip1 (undefined) have different before-states -> 2 barriers.
        let bars = t.use_texture(
            10,
            SubresourceRange::all(),
            TextureState::shader_read(PipelineStages::FRAGMENT_SHADER),
        );
        assert_eq!(bars.len(), 2);
    }

    #[test]
    fn split_barrier_produces_begin_end() {
        let b: Barrier<u32, u32> = Barrier::Buffer(BufferBarrier {
            buffer: 1,
            before: BufferState::copy_dst(),
            after: BufferState::vertex(),
            kind: BarrierKind::Immediate,
        });
        let (begin, end) = split_barrier(b);
        assert_eq!(begin.kind(), BarrierKind::Begin);
        assert_eq!(end.kind(), BarrierKind::End);
    }

    #[test]
    fn optimize_removes_noops() {
        let mut batch: Vec<Barrier<u32, u32>> = alloc::vec![
            Barrier::Buffer(BufferBarrier {
                buffer: 1,
                before: BufferState::vertex(),
                after: BufferState::vertex(),
                kind: BarrierKind::Immediate,
            }),
            Barrier::Buffer(BufferBarrier {
                buffer: 2,
                before: BufferState::copy_dst(),
                after: BufferState::vertex(),
                kind: BarrierKind::Immediate,
            }),
        ];
        optimize_batch(&mut batch);
        assert_eq!(batch.len(), 1, "the read->read no-op is dropped");
    }

    #[test]
    fn auto_registers_unknown_resources() {
        let mut t = Tracker::new();
        // Never registered: use_buffer should auto-register at initial().
        let b = t.use_buffer(99, BufferState::copy_dst());
        assert!(b.is_some());
        assert!(t.buffer_state(99).unwrap().is_write());
    }
}
