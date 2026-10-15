//! Client-side voice bookkeeping for the Prism ECS audio bridge.
//!
//! The runtime allocates handles on its audio thread and the command ring is
//! intentionally one-way. [`VoiceMirror`] therefore replays the accepted
//! voice commands against a private [`VoicePool`]
//! so ECS systems can associate a stable handle with each entity without
//! adding a response ring to the real-time path. [`LiveVoices`] owns the
//! entity-to-handle mapping and [`PendingStops`] retains stops that could not
//! enter a full command ring.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use prism_audio_core::voice::{Importance, VoiceHandle, VoiceInfo, VoicePool, VoiceRequest};
use prism_ecs::prelude::{Entity, Resource};

/// Deterministic client-side mirror of the runtime's voice pool.
#[derive(Resource)]
pub struct VoiceMirror {
    pool: VoicePool,
}

impl VoiceMirror {
    /// Builds a mirror with the same total and physical capacities as the
    /// runtime pool.
    #[must_use]
    #[inline]
    pub(crate) fn new(capacity: usize, max_physical: usize) -> Self {
        Self {
            pool: VoicePool::new(capacity, max_physical),
        }
    }

    /// Replays an accepted spawn and returns the handle the runtime will mint.
    #[inline]
    pub(crate) fn allocate(&mut self, request: VoiceRequest) -> Option<VoiceHandle> {
        self.pool.allocate(request)
    }

    /// Replays an accepted stop and reports whether the handle was live.
    #[inline]
    pub(crate) fn release(&mut self, handle: VoiceHandle) -> bool {
        self.pool.release(handle)
    }

    /// Replays an accepted importance update.
    #[inline]
    pub(crate) fn set_importance(&mut self, handle: VoiceHandle, importance: Importance) -> bool {
        self.pool.set_importance(handle, importance)
    }

    /// Replays an accepted physical-voice budget update.
    #[inline]
    pub(crate) fn set_max_physical(&mut self, max_physical: usize) {
        self.pool.set_max_physical(max_physical);
    }

    /// Returns the number of mirrored physical and virtual voices.
    #[must_use]
    #[inline]
    pub fn active_count(&self) -> usize {
        self.pool.active_count()
    }

    /// Returns the number of mirrored physical voices.
    #[must_use]
    #[inline]
    pub fn physical_count(&self) -> usize {
        self.pool.physical_count()
    }

    /// Inspects a generation-checked predicted voice without mutating the pool.
    pub fn get(&self, handle: VoiceHandle) -> Option<VoiceInfo> {
        self.pool.get(handle)
    }
}

/// Authoritative entity-to-live-voice mapping.
#[derive(Resource, Debug, Default)]
pub struct LiveVoices {
    pub(crate) map: BTreeMap<Entity, VoiceHandle>,
}

impl LiveVoices {
    /// Records a handle for an entity and returns a previous handle, if any.
    #[inline]
    pub(crate) fn insert(&mut self, entity: Entity, handle: VoiceHandle) -> Option<VoiceHandle> {
        self.map.insert(entity, handle)
    }

    /// Removes and returns an entity's live handle.
    #[inline]
    pub(crate) fn remove(&mut self, entity: Entity) -> Option<VoiceHandle> {
        self.map.remove(&entity)
    }

    /// Reads an entity's live handle.
    #[must_use]
    #[inline]
    pub fn get(&self, entity: Entity) -> Option<VoiceHandle> {
        self.map.get(&entity).copied()
    }

    /// Reports whether an entity has a live handle.
    #[must_use]
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.map.contains_key(&entity)
    }

    /// Returns the number of mapped entities.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Reports whether the mapping is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Handles waiting for a retry after a full command ring rejected a stop.
#[derive(Resource, Debug, Default)]
pub struct PendingStops {
    handles: Vec<VoiceHandle>,
}

impl PendingStops {
    /// Adds a handle to the retry queue.
    #[inline]
    pub(crate) fn push(&mut self, handle: VoiceHandle) {
        if !self.handles.contains(&handle) {
            self.handles.push(handle);
        }
    }

    /// Takes all queued handles while preserving their enqueue order.
    #[must_use]
    #[inline]
    pub(crate) fn take_all(&mut self) -> Vec<VoiceHandle> {
        core::mem::take(&mut self.handles)
    }

    /// Returns the number of queued handles.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Reports whether no handles are queued.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }
}

/// Entities whose [`AudioEmitter`](crate::AudioEmitter) was removed.
///
/// Prism ECS exposes lifecycle observers rather than Bevy's
/// `RemovedComponents` query. The plugin records the entity at the removal
/// boundary and the ordered audio system drains this resource on the next
/// update, allowing the voice stop path to remain centralized.
#[derive(Resource, Debug, Default)]
pub struct RemovedAudioEmitters {
    entities: BTreeSet<Entity>,
}

impl RemovedAudioEmitters {
    /// Records one removed emitter entity.
    #[inline]
    pub(crate) fn push(&mut self, entity: Entity) {
        self.entities.insert(entity);
    }

    /// Takes deduplicated pending removals in stable entity order.
    #[must_use]
    #[inline]
    pub(crate) fn take_all(&mut self) -> BTreeSet<Entity> {
        core::mem::take(&mut self.entities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_core::voice::{VirtualBehavior, VoiceGroup};

    #[test]
    fn mirror_tracks_pool_operations() {
        let mut mirror = VoiceMirror::new(2, 1);
        let request = VoiceRequest {
            group: VoiceGroup(1),
            priority: 0,
            importance: 1.0,
            behavior: VirtualBehavior::ContinueVirtual,
        };
        let handle = mirror.allocate(request).expect("first slot");
        assert_eq!(mirror.active_count(), 1);
        assert_eq!(mirror.physical_count(), 1);
        assert!(mirror.set_importance(handle, 0.5));
        assert!(mirror.release(handle));
        assert_eq!(mirror.active_count(), 0);
    }

    #[test]
    fn pending_stops_take_is_ordered_and_empty() {
        let mut pending = PendingStops::default();
        let mut mirror = VoiceMirror::new(2, 2);
        let request = VoiceRequest::new(VoiceGroup(0), 1.0);
        let first = mirror.allocate(request).expect("first slot");
        let second = mirror.allocate(request).expect("second slot");
        pending.push(first);
        pending.push(second);
        assert_eq!(pending.take_all(), vec![first, second]);
        assert!(pending.is_empty());
    }
}
