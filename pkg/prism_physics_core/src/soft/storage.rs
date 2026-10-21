//! Deterministic ownership storage for world-owned soft bodies.
//!
//! A [`SoftBodyStorage`] owns the mutable [`SoftBody`] state that a
//! [`crate::world::PhysicsWorld`] advances. The storage deliberately has no
//! gameplay meaning: a caller may use a slot for cloth, rope, hair, a
//! volumetric body, or any other particle/constraint authoring.

use super::body::SoftBody;

/// A generational handle into [`SoftBodyStorage`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftBodyHandle {
    index: u32,
    generation: u32,
}

impl SoftBodyHandle {
    /// A sentinel handle that never refers to a live soft body.
    pub const INVALID: SoftBodyHandle = SoftBodyHandle {
        index: u32::MAX,
        generation: u32::MAX,
    };

    const fn new(index: u32, generation: u32) -> SoftBodyHandle {
        SoftBodyHandle { index, generation }
    }

    /// Returns the storage slot index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Returns the generation captured by this handle.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl Default for SoftBodyHandle {
    fn default() -> Self {
        Self::INVALID
    }
}

/// Deterministic generational storage for world-owned soft bodies.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftBodyStorage {
    bodies: Vec<Option<SoftBody>>,
    generations: Vec<u32>,
    free_list: Vec<u32>,
    live: usize,
}

impl SoftBodyStorage {
    /// Creates an empty storage.
    #[must_use]
    pub fn new() -> SoftBodyStorage {
        Self::default()
    }

    /// Inserts a soft body and returns its stable handle.
    pub fn insert(&mut self, body: SoftBody) -> SoftBodyHandle {
        self.live += 1;
        if let Some(index) = self.free_list.pop() {
            let slot = index as usize;
            self.bodies[slot] = Some(body);
            SoftBodyHandle::new(index, self.generations[slot])
        } else {
            let index = self.bodies.len() as u32;
            self.bodies.push(Some(body));
            self.generations.push(0);
            SoftBodyHandle::new(index, 0)
        }
    }

    /// Removes a soft body, invalidating its handle.
    pub fn remove(&mut self, handle: SoftBodyHandle) -> bool {
        if !self.contains(handle) {
            return false;
        }
        let slot = handle.index as usize;
        self.bodies[slot] = None;
        self.generations[slot] = self.generations[slot].wrapping_add(1);
        self.free_list.push(handle.index);
        self.live -= 1;
        true
    }

    /// Returns whether `handle` refers to a live soft body.
    #[must_use]
    pub fn contains(&self, handle: SoftBodyHandle) -> bool {
        let slot = handle.index as usize;
        slot < self.bodies.len()
            && self.bodies[slot].is_some()
            && self.generations[slot] == handle.generation
    }

    /// Returns the number of live soft bodies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Returns whether no soft bodies are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Returns a shared soft body reference when `handle` is live.
    #[must_use]
    pub fn get(&self, handle: SoftBodyHandle) -> Option<&SoftBody> {
        self.contains(handle)
            .then(|| self.bodies[handle.index as usize].as_ref().unwrap())
    }

    /// Returns a mutable soft body reference when `handle` is live.
    pub fn get_mut(&mut self, handle: SoftBodyHandle) -> Option<&mut SoftBody> {
        if !self.contains(handle) {
            return None;
        }
        self.bodies[handle.index as usize].as_mut()
    }

    /// Returns live handles in deterministic ascending slot order.
    pub fn handles(&self) -> impl Iterator<Item = SoftBodyHandle> + '_ {
        self.bodies.iter().enumerate().filter_map(|(slot, body)| {
            body.as_ref()
                .map(|_| SoftBodyHandle::new(slot as u32, self.generations[slot]))
        })
    }

    /// Returns the live handle occupying `slot`, if any.
    #[must_use]
    pub fn handle_at_slot(&self, slot: usize) -> Option<SoftBodyHandle> {
        self.bodies.get(slot)?.as_ref()?;
        Some(SoftBodyHandle::new(
            slot as u32,
            *self.generations.get(slot)?,
        ))
    }

    /// Temporarily takes a live body without changing its handle validity.
    pub(crate) fn take(&mut self, handle: SoftBodyHandle) -> Option<SoftBody> {
        if !self.contains(handle) {
            return None;
        }
        self.bodies[handle.index as usize].take()
    }

    /// Puts a body back into the slot returned by [`Self::take`].
    pub(crate) fn put(&mut self, handle: SoftBodyHandle, body: SoftBody) {
        if self.contains_slot(handle) {
            debug_assert!(self.bodies[handle.index as usize].is_none());
            self.bodies[handle.index as usize] = Some(body);
        }
    }

    fn contains_slot(&self, handle: SoftBodyHandle) -> bool {
        let slot = handle.index as usize;
        slot < self.bodies.len() && self.generations[slot] == handle.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_are_deterministic_and_stale_after_remove() {
        let mut storage = SoftBodyStorage::new();
        let first = storage.insert(SoftBody::with_default_config());
        let second = storage.insert(SoftBody::with_default_config());
        assert_eq!(storage.handles().collect::<Vec<_>>(), vec![first, second]);
        assert!(storage.remove(first));
        assert!(!storage.contains(first));
        let replacement = storage.insert(SoftBody::with_default_config());
        assert_eq!(replacement.index(), first.index());
        assert_ne!(replacement.generation(), first.generation());
        assert_eq!(storage.len(), 2);
    }
}
