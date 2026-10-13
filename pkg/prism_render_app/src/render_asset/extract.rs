//! Bringing a frame's changed assets across the world boundary.
//!
//! The main world owns the authoritative asset arena; the render world gets a
//! *copy* of what changed this frame, because preparing needs the asset's
//! contents and the render world cannot borrow the simulation's world. Bevy
//! spells the copy `ExtractedAssets<A>` and this is the Prism equivalent, filled
//! by an extract step the renderer appends to the sub-app.

use alloc::vec::Vec;

use prism_asset::{AssetEvent, AssetId, Assets};
use prism_ecs::prelude::Resource;
use prism_ecs::world::World;

use super::RenderAsset;

/// The assets a frame carried into the render world.
///
/// `extracted` holds an owned copy of every asset added or modified since the
/// last extract, in event order, because a prepare needs the contents and the
/// main world's arena is not reachable from here. `removed` holds the ids whose
/// source went away, which is how the prepared table learns to release them.
/// Both clear every frame after the prepare stage consumed them.
#[derive(Resource)]
pub struct ExtractedAssets<A: RenderAsset> {
    extracted: Vec<(AssetId<A::SourceAsset>, A::SourceAsset)>,
    removed: Vec<AssetId<A::SourceAsset>>,
}

impl<A: RenderAsset> Default for ExtractedAssets<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: RenderAsset> ExtractedAssets<A> {
    /// An empty frame's worth of assets.
    #[must_use]
    pub fn new() -> Self {
        Self {
            extracted: Vec::new(),
            removed: Vec::new(),
        }
    }

    /// Records an owned copy of an added/modified asset.
    pub fn push_extracted(&mut self, id: AssetId<A::SourceAsset>, asset: A::SourceAsset) {
        self.extracted.push((id, asset));
    }

    /// Records that an asset's source is gone.
    pub fn push_removed(&mut self, id: AssetId<A::SourceAsset>) {
        self.removed.push(id);
    }

    /// The assets added/modified this frame, in event order.
    #[must_use]
    pub fn extracted(&self) -> &[(AssetId<A::SourceAsset>, A::SourceAsset)] {
        &self.extracted
    }

    /// The ids whose source was removed this frame.
    #[must_use]
    pub fn removed(&self) -> &[AssetId<A::SourceAsset>] {
        &self.removed
    }

    /// Takes both lists, leaving the frame empty.
    pub fn take(
        &mut self,
    ) -> (
        Vec<(AssetId<A::SourceAsset>, A::SourceAsset)>,
        Vec<AssetId<A::SourceAsset>>,
    ) {
        (
            core::mem::take(&mut self.extracted),
            core::mem::take(&mut self.removed),
        )
    }

    /// Whether the frame carried nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.extracted.is_empty() && self.removed.is_empty()
    }

    /// The total number of changed assets this frame.
    #[must_use]
    pub fn len(&self) -> usize {
        self.extracted.len() + self.removed.len()
    }
}

/// Drains the main world's changed assets into the render world's frame.
///
/// The extract step for one render-asset family: it reads (and empties) the main
/// world arena's event queue and copies every added/modified asset into the
/// render world's [`ExtractedAssets`]. Removed ids cross as ids only — there is
/// nothing left to copy, and the prepare stage needs the id to release the GPU
/// form.
///
/// `Clone` is what makes the copy possible without taking ownership away from
/// the simulation: the main world keeps its asset, and the render world prepares
/// from its own copy. Nothing here mutates simulation state beyond draining the
/// arena's own event queue, which is the queue's contract.
///
/// Entries **accumulate** until the prepare stage consumes them
/// ([`take`](ExtractedAssets::take)). A frame whose prepare did not run therefore
/// keeps its assets for the next one instead of losing them — the events are
/// already drained from the simulation's queue, so dropping them here would lose
/// the change for good.
pub fn extract_assets<A: RenderAsset>(main: &mut World, render: &mut World) {
    let events: Vec<AssetEvent<A::SourceAsset>> =
        match main.get_resource_mut::<Assets<A::SourceAsset>>() {
            Some(assets) => assets.drain_events(),
            // No arena in the main world: nothing of this family exists yet.
            None => return,
        };
    if events.is_empty() {
        return;
    }

    if render.get_resource::<ExtractedAssets<A>>().is_none() {
        render.insert_resource(ExtractedAssets::<A>::new());
    }
    let mut frame = core::mem::take(render.resource_mut::<ExtractedAssets<A>>());
    for event in events {
        let id = event.id();
        match event {
            AssetEvent::Added { .. } | AssetEvent::Modified { .. } => {
                let copied = main
                    .get_resource::<Assets<A::SourceAsset>>()
                    .and_then(|assets| assets.get(id))
                    .cloned();
                if let Some(asset) = copied {
                    frame.push_extracted(id, asset);
                }
            }
            AssetEvent::Removed { .. } => frame.push_removed(id),
            // Unused/Failed/LoadedWithDependencies do not change the GPU form.
            AssetEvent::Unused { .. }
            | AssetEvent::Failed { .. }
            | AssetEvent::LoadedWithDependencies { .. } => {}
        }
    }
    *render.resource_mut::<ExtractedAssets<A>>() = frame;
}
