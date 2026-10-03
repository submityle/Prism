//! Generational entity identifiers and the allocator that hands them out.
//!
//! An [`Entity`] is a lightweight 64-bit handle: a 32-bit slot `index` plus a
//! non-zero 32-bit `generation`. When a slot is recycled its generation is
//! bumped, so a stale [`Entity`] that refers to a freed slot is detected by a
//! generation mismatch and resolves to "not found" rather than silently
//! aliasing a different entity. This is the safety foundation that world
//! streaming (design §13) and rollback (design §14) rely on.

use alloc::vec::Vec;
use core::num::NonZeroU32;
use core::sync::atomic::{AtomicI64, Ordering};

use crate::archetype::ArchetypeId;

/// A lightweight, copyable handle to an entity.
///
/// The handle is a `(index, generation)` pair. `index` selects a slot in the
/// [`Entities`] metadata table; `generation` disambiguates reuse of that slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Entity {
    index: u32,
    generation: NonZeroU32,
}

impl Entity {
    /// The 32-bit slot index of this entity.
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The non-zero generation of this entity.
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation.get()
    }

    /// Pack the handle into a single `u64` (`generation << 32 | index`).
    ///
    /// Useful for stable serialization and as a dense map key.
    #[inline]
    pub const fn to_bits(self) -> u64 {
        ((self.generation.get() as u64) << 32) | (self.index as u64)
    }

    /// Reconstruct an [`Entity`] previously produced by [`Entity::to_bits`].
    ///
    /// Returns `None` if the high 32 bits (the generation) are zero, which can
    /// never be produced by a live handle.
    #[inline]
    pub const fn from_bits(bits: u64) -> Option<Self> {
        match NonZeroU32::new((bits >> 32) as u32) {
            Some(generation) => Some(Self {
                index: bits as u32,
                generation,
            }),
            None => None,
        }
    }
}

impl core::fmt::Debug for Entity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Entity({}v{})", self.index, self.generation.get())
    }
}

/// Where an allocated entity's component data currently lives.
///
/// For M0 the location is `{archetype, row}`; the chunk index is folded in when
/// chunked storage lands (design §5.3 / M2). An entity that is allocated but
/// not yet inserted into any archetype uses [`EntityLocation::EMPTY`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EntityLocation {
    /// The archetype the entity currently belongs to.
    pub archetype_id: ArchetypeId,
    /// The row within the archetype's table.
    pub row: u32,
}

impl EntityLocation {
    /// Sentinel location for an entity that is allocated but not yet placed in
    /// any archetype table.
    pub const EMPTY: Self = Self {
        archetype_id: ArchetypeId::INVALID,
        row: u32::MAX,
    };

    /// Whether this location is the [`EMPTY`](Self::EMPTY) sentinel.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.archetype_id == ArchetypeId::INVALID
    }
}

/// Per-slot metadata tracked by the [`Entities`] allocator.
#[derive(Clone, Copy)]
struct EntityMeta {
    /// Current generation of the slot. Live entities carry this generation.
    generation: NonZeroU32,
    /// Current storage location (or [`EntityLocation::EMPTY`]).
    location: EntityLocation,
    /// Whether the slot is currently allocated (vs. sitting on the free list).
    alive: bool,
}

impl EntityMeta {
    const INITIAL_GENERATION: NonZeroU32 = NonZeroU32::new(1).unwrap();
}

/// The entity allocator and metadata table.
///
/// Maintains a free-list of recycled slots plus a dense `Vec` of per-slot
/// metadata. Allocation prefers recycled slots (bumping their generation) and
/// only grows the table when the free list is empty.
///
/// The allocator also supports lock-free *reservation* ([`Entities::reserve_entity`]):
/// deferred [`Commands`](crate::command::Commands) hand out a valid [`Entity`]
/// handle immediately from a shared borrow, and a later [`Entities::flush`]
/// materialises every reserved handle into real metadata at a synchronization
/// point. This is the foundation the command buffer builds on.
pub struct Entities {
    meta: Vec<EntityMeta>,
    free: Vec<u32>,
    /// Number of currently-live entities.
    len: u32,
    /// Count of handles reserved via [`Entities::reserve_entity`] since the last
    /// [`Entities::flush`]. Counts up monotonically; reset to zero on flush.
    reserved: AtomicI64,
}

impl Default for Entities {
    fn default() -> Self {
        Self::new()
    }
}

impl Entities {
    /// Create an empty allocator.
    pub fn new() -> Self {
        Self {
            meta: Vec::new(),
            free: Vec::new(),
            len: 0,
            reserved: AtomicI64::new(0),
        }
    }

    /// Number of currently-live entities.
    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether there are no live entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Highest slot index ever allocated plus one (i.e. the metadata table
    /// length). Useful for sizing dense side tables.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.meta.len() as u32
    }

    /// Allocate a fresh entity handle with an [`EMPTY`](EntityLocation::EMPTY)
    /// location. The caller is responsible for subsequently recording a real
    /// location via [`Entities::set_location`].
    pub fn alloc(&mut self) -> Entity {
        debug_assert_eq!(
            *self.reserved.get_mut(),
            0,
            "alloc() called with outstanding reservations; flush() first"
        );
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let meta = &mut self.meta[index as usize];
            debug_assert!(!meta.alive, "free-list slot must be dead");
            meta.alive = true;
            meta.location = EntityLocation::EMPTY;
            Entity {
                index,
                generation: meta.generation,
            }
        } else {
            let index = self.meta.len() as u32;
            assert!(index != u32::MAX, "exhausted entity index space");
            self.meta.push(EntityMeta {
                generation: EntityMeta::INITIAL_GENERATION,
                location: EntityLocation::EMPTY,
                alive: true,
            });
            Entity {
                index,
                generation: EntityMeta::INITIAL_GENERATION,
            }
        }
    }

    /// Whether `entity` refers to a currently-live slot with a matching
    /// generation.
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.meta
            .get(entity.index as usize)
            .is_some_and(|m| m.alive && m.generation == entity.generation)
    }

    /// Resolve the current storage location of a live entity.
    ///
    /// Returns `None` for a stale handle (generation mismatch) or a freed slot.
    #[inline]
    pub fn location(&self, entity: Entity) -> Option<EntityLocation> {
        let meta = self.meta.get(entity.index as usize)?;
        if meta.alive && meta.generation == entity.generation {
            Some(meta.location)
        } else {
            None
        }
    }

    /// Record a new storage location for a live entity.
    ///
    /// # Panics
    /// Panics in debug builds if the entity is not live.
    #[inline]
    pub fn set_location(&mut self, entity: Entity, location: EntityLocation) {
        let meta = &mut self.meta[entity.index as usize];
        debug_assert!(meta.alive && meta.generation == entity.generation);
        meta.location = location;
    }

    /// Free a live entity, pushing its slot onto the free list and bumping the
    /// slot generation so stale handles no longer resolve.
    ///
    /// Returns the entity's last known location, or `None` if the handle was
    /// already stale/dead (in which case nothing is changed).
    pub fn free(&mut self, entity: Entity) -> Option<EntityLocation> {
        debug_assert_eq!(
            *self.reserved.get_mut(),
            0,
            "free() called with outstanding reservations; flush() first"
        );
        let meta = self.meta.get_mut(entity.index as usize)?;
        if !meta.alive || meta.generation != entity.generation {
            return None;
        }
        let location = meta.location;
        meta.alive = false;
        // Bump generation so recycled handles differ. Wrap past u32::MAX back to
        // 1 (0 is reserved to keep `NonZeroU32` valid); collisions after a full
        // 4-billion-cycle wrap are acceptable and standard for generational
        // indices.
        let next = meta.generation.get().wrapping_add(1);
        meta.generation = NonZeroU32::new(next).unwrap_or(EntityMeta::INITIAL_GENERATION);
        meta.location = EntityLocation::EMPTY;
        self.free.push(entity.index);
        self.len -= 1;
        Some(location)
    }

    /// Reserve a fresh [`Entity`] handle from a *shared* borrow, without
    /// mutating the metadata table.
    ///
    /// Reserved handles are valid, unique, and carry the correct generation,
    /// but their slot metadata is only materialised by a later
    /// [`Entities::flush`]. This is what lets [`Commands`](crate::command::Commands)
    /// hand out an [`Entity`] immediately while deferring the structural change.
    ///
    /// Reservations draw from the free list first (newest-freed first, matching
    /// [`Entities::alloc`]) and then from brand-new indices beyond the current
    /// table length. The reservation counter is lock-free, so this is safe to
    /// call concurrently from multiple command buffers.
    pub fn reserve_entity(&self) -> Entity {
        // `n` is this reservation's 0-based ordinal since the last flush.
        let n = self.reserved.fetch_add(1, Ordering::Relaxed);
        debug_assert!(n >= 0, "reservation counter overflow");
        let n = n as usize;
        let free_len = self.free.len();
        if n < free_len {
            // Draw from the free list, newest first (mirrors `alloc`'s `pop`).
            let index = self.free[free_len - 1 - n];
            let generation = self.meta[index as usize].generation;
            Entity { index, generation }
        } else {
            // Brand-new index beyond the current table.
            let ordinal = (n - free_len) as u32;
            let index = self.meta.len() as u32 + ordinal;
            assert!(index != u32::MAX, "exhausted entity index space");
            Entity {
                index,
                generation: EntityMeta::INITIAL_GENERATION,
            }
        }
    }

    /// Whether there are reserved-but-not-yet-materialised handles outstanding.
    #[inline]
    pub fn needs_flush(&mut self) -> bool {
        *self.reserved.get_mut() != 0
    }

    /// Materialise every handle produced by [`Entities::reserve_entity`] since
    /// the last flush, giving each a live metadata slot with an
    /// [`EntityLocation::EMPTY`] location.
    ///
    /// This must run at a synchronization point (it takes `&mut self`) before
    /// any `alloc`/`free`, and before the reserved entities are read.
    pub fn flush(&mut self) {
        let reserved = core::mem::replace(self.reserved.get_mut(), 0);
        if reserved <= 0 {
            return;
        }
        let reserved = reserved as usize;
        let free_len = self.free.len();
        let consumed_free = reserved.min(free_len);
        let new_count = reserved - consumed_free;

        // Materialise free-list slots consumed by reservation (the newest
        // `consumed_free` entries, which `reserve_entity` drew from the end).
        let keep = free_len - consumed_free;
        for &index in &self.free[keep..] {
            let meta = &mut self.meta[index as usize];
            debug_assert!(!meta.alive, "reserved free slot must be dead");
            meta.alive = true;
            meta.location = EntityLocation::EMPTY;
        }
        self.free.truncate(keep);

        // Materialise brand-new indices appended beyond the old table length.
        for _ in 0..new_count {
            self.meta.push(EntityMeta {
                generation: EntityMeta::INITIAL_GENERATION,
                location: EntityLocation::EMPTY,
                alive: true,
            });
        }
        self.len += reserved as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arch(n: u32) -> EntityLocation {
        EntityLocation {
            archetype_id: ArchetypeId::new(n),
            row: 0,
        }
    }

    #[test]
    fn alloc_is_unique_and_live() {
        let mut e = Entities::new();
        let a = e.alloc();
        let b = e.alloc();
        assert_ne!(a, b);
        assert_eq!(e.len(), 2);
        assert!(e.contains(a));
        assert!(e.contains(b));
        assert_eq!(e.location(a), Some(EntityLocation::EMPTY));
    }

    #[test]
    fn free_recycles_slot_and_bumps_generation() {
        let mut e = Entities::new();
        let a = e.alloc();
        assert_eq!(a.index(), 0);
        assert_eq!(a.generation(), 1);
        e.set_location(a, arch(3));
        assert_eq!(e.free(a), Some(arch(3)));
        assert!(!e.contains(a));
        assert_eq!(e.len(), 0);

        // Reallocating reuses slot 0 but with a higher generation.
        let b = e.alloc();
        assert_eq!(b.index(), 0);
        assert_eq!(b.generation(), 2);
        // The stale handle `a` must no longer resolve.
        assert!(!e.contains(a));
        assert_eq!(e.location(a), None);
        assert!(e.contains(b));
    }

    #[test]
    fn double_free_is_noop() {
        let mut e = Entities::new();
        let a = e.alloc();
        assert!(e.free(a).is_some());
        assert_eq!(e.free(a), None);
    }

    #[test]
    fn reserve_then_flush_materialises_new_indices() {
        let mut e = Entities::new();
        let a = e.reserve_entity();
        let b = e.reserve_entity();
        assert_ne!(a, b);
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);
        // Not live until flushed.
        assert!(!e.contains(a));
        assert!(e.needs_flush());
        e.flush();
        assert!(!e.needs_flush());
        assert!(e.contains(a));
        assert!(e.contains(b));
        assert_eq!(e.len(), 2);
        assert_eq!(e.location(a), Some(EntityLocation::EMPTY));
    }

    #[test]
    fn reserve_draws_from_free_list_first() {
        let mut e = Entities::new();
        let a = e.alloc();
        let b = e.alloc();
        assert!(e.free(a).is_some());
        assert!(e.free(b).is_some());
        // Two slots free; reserving two should recycle both with bumped gens.
        let r0 = e.reserve_entity();
        let r1 = e.reserve_entity();
        let r2 = e.reserve_entity();
        // r0/r1 recycle freed slots; r2 is a brand-new index.
        assert!(r0.index() < 2);
        assert!(r1.index() < 2);
        assert_ne!(r0.index(), r1.index());
        assert_eq!(r2.index(), 2);
        assert_eq!(r0.generation(), 2);
        e.flush();
        assert!(e.contains(r0));
        assert!(e.contains(r1));
        assert!(e.contains(r2));
        assert_eq!(e.len(), 3);
    }

    #[test]
    fn bits_roundtrip() {
        let mut e = Entities::new();
        let a = e.alloc();
        let bits = a.to_bits();
        assert_eq!(Entity::from_bits(bits), Some(a));
        assert_eq!(Entity::from_bits(0), None);
    }
}
