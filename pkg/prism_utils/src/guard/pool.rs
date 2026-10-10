//! A generational slab that detects stale and double frees ([`GuardedPool`]).
//!
//! Each slot carries a generation counter that is bumped whenever the slot is
//! recycled, so a [`GuardHandle`] captured before a free no longer matches and
//! is reported as [`GuardError::UseAfterFree`] rather than silently aliasing a
//! different value. Freeing an already-free slot is [`GuardError::DoubleFree`];
//! a handle whose index was never issued is [`GuardError::DanglingHandle`].
//! Freed slots are threaded onto a free list for `O(1)` reuse. The whole module
//! is safe code: validity is tracked with an `enum` state and a generation
//! counter, not with raw pointers.

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use super::GuardError;

/// A generational handle into a [`GuardedPool`]: a `Copy`, domain-tagged
/// `(index, generation)` pair.
///
/// All trait impls are hand-written so they never require the stored type `T`
/// or the domain marker `D` to implement anything (both are compile-time tags
/// held in `PhantomData<fn() -> (T, D)>`). A handle is only valid for the exact
/// slot *generation* it was issued for, which is what makes stale access
/// detectable.
pub struct GuardHandle<T, D = ()> {
    index: u32,
    generation: u32,
    _marker: PhantomData<fn() -> (T, D)>,
}

impl<T, D> GuardHandle<T, D> {
    /// The slot index this handle refers to.
    #[must_use]
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The slot generation this handle was issued for.
    #[must_use]
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    #[inline]
    const fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    /// Forge a handle from raw parts. Test-only, used to exercise the
    /// dangling-handle path with an index that was never issued by a pool.
    #[cfg(test)]
    pub(crate) const fn from_raw_for_test(index: u32, generation: u32) -> Self {
        Self::new(index, generation)
    }
}

impl<T, D> Clone for GuardHandle<T, D> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, D> Copy for GuardHandle<T, D> {}

impl<T, D> PartialEq for GuardHandle<T, D> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T, D> Eq for GuardHandle<T, D> {}

impl<T, D> core::hash::Hash for GuardHandle<T, D> {
    #[inline]
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T, D> core::fmt::Debug for GuardHandle<T, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GuardHandle(#{}, gen {})", self.index, self.generation)
    }
}

/// The occupancy of a single pool slot.
enum SlotState<T> {
    /// A live value.
    Live(T),
    /// A free slot, linked into the free list by the next free index.
    Free { next_free: Option<u32> },
}

struct Slot<T> {
    generation: u32,
    state: SlotState<T>,
}

/// A generational slab that hardens value lifetime: it detects
/// use-after-free, double free, and dangling handles (design doc §24.3), all in
/// safe code.
///
/// Values are inserted with [`insert`](Self::insert), accessed through the
/// handle, and reclaimed with [`try_remove`](Self::try_remove). A freed slot is
/// reused on the next [`insert`](Self::insert) with a bumped generation, so
/// handles to the old occupant are rejected.
pub struct GuardedPool<T, D = ()> {
    slots: Vec<Slot<T>>,
    free_head: Option<u32>,
    live: usize,
    _marker: PhantomData<fn() -> D>,
}

impl<T, D> GuardedPool<T, D> {
    /// Create an empty pool.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            live: 0,
            _marker: PhantomData,
        }
    }

    /// Create an empty pool with room for `cap` slots.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            slots: Vec::with_capacity(cap),
            free_head: None,
            live: 0,
            _marker: PhantomData,
        }
    }

    /// Number of live values.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Whether the pool holds no live values.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Number of allocated slots (live plus free).
    #[must_use]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Compute the next generation for a recycled slot, skipping 0 so that a
    /// freshly default handle (generation 0) never aliases a live slot.
    #[inline]
    const fn bump(generation: u32) -> u32 {
        match generation.wrapping_add(1) {
            0 => 1,
            g => g,
        }
    }

    /// Insert `value`, returning a fresh handle. Reuses a free slot when one is
    /// available, otherwise grows the slab.
    ///
    /// # Panics
    /// Panics if the pool would exceed `u32::MAX` slots.
    pub fn insert(&mut self, value: T) -> GuardHandle<T, D> {
        if let Some(index) = self.free_head {
            let slot = &mut self.slots[index as usize];
            let next_free = match slot.state {
                SlotState::Free { next_free } => next_free,
                SlotState::Live(_) => {
                    // The free list only ever links free slots; a live slot
                    // here would mean the list was corrupted. Rebuild defensively
                    // by treating this slot as the list tail instead of trusting
                    // a bogus link.
                    None
                }
            };
            let generation = Self::bump(slot.generation);
            slot.generation = generation;
            slot.state = SlotState::Live(value);
            self.free_head = next_free;
            self.live += 1;
            return GuardHandle::new(index, generation);
        }

        let index = u32::try_from(self.slots.len())
            .expect("guarded pool capacity exceeded (> u32::MAX slots)");
        let generation = 1;
        self.slots.push(Slot {
            generation,
            state: SlotState::Live(value),
        });
        self.live += 1;
        GuardHandle::new(index, generation)
    }

    /// Classify a handle against the current slot table without mutating.
    fn classify(&self, handle: GuardHandle<T, D>) -> Result<(), GuardError> {
        let Some(slot) = self.slots.get(handle.index as usize) else {
            return Err(GuardError::DanglingHandle);
        };
        match &slot.state {
            SlotState::Live(_) if slot.generation == handle.generation => Ok(()),
            SlotState::Free { .. } if slot.generation == handle.generation => {
                Err(GuardError::DoubleFree)
            }
            _ => Err(GuardError::UseAfterFree),
        }
    }

    /// Borrow the value behind `handle`.
    ///
    /// # Errors
    /// - [`GuardError::DanglingHandle`] if the slot was never allocated.
    /// - [`GuardError::UseAfterFree`] if the slot has been freed or recycled.
    pub fn try_get(&self, handle: GuardHandle<T, D>) -> Result<&T, GuardError> {
        let slot = self
            .slots
            .get(handle.index as usize)
            .ok_or(GuardError::DanglingHandle)?;
        match &slot.state {
            SlotState::Live(value) if slot.generation == handle.generation => Ok(value),
            _ => Err(GuardError::UseAfterFree),
        }
    }

    /// Mutably borrow the value behind `handle`.
    ///
    /// # Errors
    /// Same as [`try_get`](Self::try_get).
    pub fn try_get_mut(&mut self, handle: GuardHandle<T, D>) -> Result<&mut T, GuardError> {
        let slot = self
            .slots
            .get_mut(handle.index as usize)
            .ok_or(GuardError::DanglingHandle)?;
        match &mut slot.state {
            SlotState::Live(value) if slot.generation == handle.generation => Ok(value),
            _ => Err(GuardError::UseAfterFree),
        }
    }

    /// Remove and return the value behind `handle`, freeing its slot.
    ///
    /// # Errors
    /// - [`GuardError::DanglingHandle`] if the slot was never allocated.
    /// - [`GuardError::DoubleFree`] if the slot is already free (and the
    ///   generation still matches, i.e. this exact handle freed it before).
    /// - [`GuardError::UseAfterFree`] if the slot has since been recycled.
    pub fn try_remove(&mut self, handle: GuardHandle<T, D>) -> Result<T, GuardError> {
        // Classify first with an immutable borrow so the error cases do not
        // disturb the slot or the free list.
        self.classify(handle)?;

        let index = handle.index as usize;
        let next_free = self.free_head;
        let slot = &mut self.slots[index];
        let previous = core::mem::replace(&mut slot.state, SlotState::Free { next_free });
        match previous {
            SlotState::Live(value) => {
                self.free_head = Some(handle.index);
                self.live -= 1;
                Ok(value)
            }
            SlotState::Free {
                next_free: original,
            } => {
                // `classify` already proved the slot was live, so this branch is
                // unreachable in practice. Restore the slot rather than panic,
                // keeping the pool consistent even under an impossible state.
                slot.state = SlotState::Free {
                    next_free: original,
                };
                Err(GuardError::UseAfterFree)
            }
        }
    }

    /// Borrow the value behind `handle`, panicking on any violation.
    ///
    /// # Panics
    /// Panics with the [`GuardError`] text if the handle is dangling or stale.
    #[must_use]
    pub fn get(&self, handle: GuardHandle<T, D>) -> &T {
        match self.try_get(handle) {
            Ok(value) => value,
            Err(err) => panic!("GuardedPool::get: {err}"),
        }
    }

    /// Mutably borrow the value behind `handle`, panicking on any violation.
    ///
    /// # Panics
    /// Panics with the [`GuardError`] text if the handle is dangling or stale.
    #[must_use]
    pub fn get_mut(&mut self, handle: GuardHandle<T, D>) -> &mut T {
        match self.try_get_mut(handle) {
            Ok(value) => value,
            Err(err) => panic!("GuardedPool::get_mut: {err}"),
        }
    }

    /// Remove and return the value behind `handle`, panicking on any violation.
    ///
    /// # Panics
    /// Panics with the [`GuardError`] text on a dangling handle, double free, or
    /// use-after-free.
    pub fn remove(&mut self, handle: GuardHandle<T, D>) -> T {
        match self.try_remove(handle) {
            Ok(value) => value,
            Err(err) => panic!("GuardedPool::remove: {err}"),
        }
    }

    /// Whether `handle` currently refers to a live value.
    #[must_use]
    pub fn contains(&self, handle: GuardHandle<T, D>) -> bool {
        self.try_get(handle).is_ok()
    }

    /// Remove every value and reset the pool to empty.
    ///
    /// Slot generations are not preserved; every previously issued handle is
    /// invalidated.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = None;
        self.live = 0;
    }

    /// Iterate over `(handle, &value)` for every live value, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (GuardHandle<T, D>, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            if let SlotState::Live(value) = &slot.state {
                let index = u32::try_from(i).unwrap_or(u32::MAX);
                Some((GuardHandle::new(index, slot.generation), value))
            } else {
                None
            }
        })
    }

    /// Iterate over `&value` for every live value, in slot order.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|slot| match &slot.state {
            SlotState::Live(value) => Some(value),
            SlotState::Free { .. } => None,
        })
    }
}

impl<T, D> Default for GuardedPool<T, D> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, D> core::fmt::Debug for GuardedPool<T, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GuardedPool")
            .field("live", &self.live)
            .field("capacity", &self.slots.len())
            .finish()
    }
}
