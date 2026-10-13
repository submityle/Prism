//! The prepared table: [`RenderAssets`].

use alloc::collections::BTreeMap;

use prism_asset::AssetId;
use prism_ecs::prelude::Resource;

use super::RenderAsset;

/// Every prepared GPU representation the render world currently holds.
///
/// Keyed by the source asset's [`AssetId`], so a phase item, a bind group, or a
/// scene instance can name an asset by the same identity the simulation uses.
/// The table owns the representations: removing an entry is what drops a GPU
/// form, which is why [`sync_assets`](super::sync_assets) is the only writer.
#[derive(Resource)]
pub struct RenderAssets<A: RenderAsset> {
    prepared: BTreeMap<AssetId<A::SourceAsset>, A>,
}

impl<A: RenderAsset> Default for RenderAssets<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: RenderAsset> RenderAssets<A> {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prepared: BTreeMap::new(),
        }
    }

    /// The representation of `id`, if it is prepared.
    #[must_use]
    pub fn get(&self, id: AssetId<A::SourceAsset>) -> Option<&A> {
        self.prepared.get(&id)
    }

    /// Mutable access to the representation of `id`.
    pub fn get_mut(&mut self, id: AssetId<A::SourceAsset>) -> Option<&mut A> {
        self.prepared.get_mut(&id)
    }

    /// Whether `id` has a prepared representation.
    #[must_use]
    pub fn contains(&self, id: AssetId<A::SourceAsset>) -> bool {
        self.prepared.contains_key(&id)
    }

    /// Inserts/replaces the representation of `id`, returning the old one.
    pub fn insert(&mut self, id: AssetId<A::SourceAsset>, value: A) -> Option<A> {
        self.prepared.insert(id, value)
    }

    /// Takes the representation of `id` out of the table.
    pub fn remove(&mut self, id: AssetId<A::SourceAsset>) -> Option<A> {
        self.prepared.remove(&id)
    }

    /// Every prepared representation, in id order.
    pub fn iter(&self) -> impl Iterator<Item = (AssetId<A::SourceAsset>, &A)> {
        self.prepared.iter().map(|(id, value)| (*id, value))
    }

    /// Every prepared representation, mutably, in id order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (AssetId<A::SourceAsset>, &mut A)> {
        self.prepared.iter_mut().map(|(id, value)| (*id, value))
    }

    /// The number of prepared assets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.prepared.len()
    }

    /// Whether nothing is prepared.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.prepared.is_empty()
    }
}
