//! Bounded bindless texture-slot allocator for the unified material ABI.
//!
//! Every texture a lowered [`StandardMaterial`] references is assigned a stable
//! *bindless slot*: an index into the `texture_2d<f32>` binding array that the
//! shading passes sample. Slots are handed out by [`BindlessTextureHeap`], which
//! this module owns end to end so the render world never leaks raw indices.
//!
//! The heap replaces the earlier naive monotonic counter with real lifetime
//! management:
//!
//! * A small block of **reserved slots** ([`WHITE_SLOT`], [`FLAT_NORMAL_SLOT`],
//!   [`BLACK_SLOT`]) is always present so every semantic has a correct fallback
//!   even before any real image uploads. Missing base-color/metallic-roughness/
//!   occlusion/emissive maps resolve to opaque white; missing normal maps
//!   resolve to the flat tangent-space normal `(0.5, 0.5, 1.0)`.
//! * Real images are reference counted. [`acquire`](BindlessTextureHeap::acquire)
//!   reuses the slot of an already-resident image and bumps its refcount;
//!   [`release`](BindlessTextureHeap::release) drops a reference and only returns
//!   the slot to the free list once the last material stops using it.
//! * Freed slots are reused via a free list, and their **generation is bumped on
//!   release** so any lingering GPU material row that still points at the old
//!   `(index, generation)` pair is detected as stale rather than silently
//!   sampling a different texture.
//! * The heap is **bounded**. When capacity is exhausted `acquire` degrades
//!   gracefully to [`WHITE_SLOT`] and increments an overflow counter instead of
//!   growing without limit or panicking.
//!
//! Slot generations mirror the [`GenerationalHandle`] discipline used by the GPU
//! scene allocator so consumers can treat texture slots and scene handles with
//! the same staleness rules.

use bevy_asset::AssetId;
use bevy_image::Image;
use bevy_platform::collections::HashMap;
use prism_render_material::TextureSemantic;

/// Reserved slot holding an opaque white (`1, 1, 1, 1`) texel. Fallback for
/// base color, metallic-roughness, occlusion, and emissive maps.
pub const WHITE_SLOT: u32 = 0;
/// Reserved slot holding the flat tangent-space normal (`0.5, 0.5, 1.0`).
/// Fallback for normal and clear-coat-normal maps.
pub const FLAT_NORMAL_SLOT: u32 = 1;
/// Reserved slot holding an opaque black (`0, 0, 0, 1`) texel. Available for
/// additive/emissive-off fallbacks that must contribute nothing.
pub const BLACK_SLOT: u32 = 2;
/// First slot index available for dynamically uploaded images. All lower
/// indices are permanently reserved for the fallback textures above.
pub const FIRST_DYNAMIC_SLOT: u32 = 3;
/// Generation stamped on the reserved fallback slots. They never move, so their
/// generation is fixed and always valid.
pub const RESERVED_GENERATION: u32 = 1;
/// Default heap capacity (total slot count, reserved slots included). Sized for
/// a large open-world texture set while staying within typical
/// `maxSamplersPerShaderStage` / binding-array limits after real device caps are
/// wired in.
pub const DEFAULT_CAPACITY: u32 = 1 << 16;

/// A stable bindless texture slot: an index into the sampled-texture binding
/// array plus the generation that was live when the slot was handed out.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindlessSlot {
    /// Index into the `texture_2d<f32>` binding array.
    pub index: u32,
    /// Generation of the slot at acquisition time. A GPU material row is only
    /// allowed to sample the slot while this matches the heap's current
    /// generation for `index`.
    pub generation: u32,
}

impl BindlessSlot {
    /// The reserved white fallback slot.
    pub const WHITE: Self = Self {
        index: WHITE_SLOT,
        generation: RESERVED_GENERATION,
    };
    /// The reserved flat-normal fallback slot.
    pub const FLAT_NORMAL: Self = Self {
        index: FLAT_NORMAL_SLOT,
        generation: RESERVED_GENERATION,
    };
    /// The reserved black fallback slot.
    pub const BLACK: Self = Self {
        index: BLACK_SLOT,
        generation: RESERVED_GENERATION,
    };
}

/// Reference-counted residency record for one live image.
#[derive(Clone, Copy, Debug)]
struct HeapEntry {
    slot: u32,
    generation: u32,
    refcount: u32,
}

/// Snapshot of heap occupancy for diagnostics and soak tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BindlessHeapStats {
    /// Total addressable slots, reserved slots included.
    pub capacity: u32,
    /// Number of distinct images currently resident (refcount >= 1).
    pub live_images: u32,
    /// Dynamic slots sitting on the free list awaiting reuse.
    pub free_slots: u32,
    /// Highest dynamic slot index ever allocated plus one (the high-water mark).
    pub high_water: u32,
    /// Count of `acquire` calls that hit the capacity ceiling and fell back to
    /// [`WHITE_SLOT`] since construction.
    pub overflow: u32,
}

/// Bounded, reference-counted allocator of bindless texture slots.
pub struct BindlessTextureHeap {
    capacity: u32,
    entries: HashMap<AssetId<Image>, HeapEntry>,
    /// Current generation per slot index, including reserved slots. Grows as the
    /// high-water mark advances.
    slot_generations: Vec<u32>,
    free: Vec<u32>,
    next: u32,
    overflow: u32,
}

impl Default for BindlessTextureHeap {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

impl BindlessTextureHeap {
    /// Creates a heap with room for `capacity` total slots (reserved slots
    /// included). `capacity` must leave at least one dynamic slot.
    pub fn new(capacity: u32) -> Self {
        assert!(
            capacity > FIRST_DYNAMIC_SLOT,
            "bindless texture heap needs {FIRST_DYNAMIC_SLOT} reserved slots plus room for at least one image"
        );
        Self {
            capacity,
            entries: HashMap::default(),
            slot_generations: vec![RESERVED_GENERATION; FIRST_DYNAMIC_SLOT as usize],
            free: Vec::new(),
            next: FIRST_DYNAMIC_SLOT,
            overflow: 0,
        }
    }

    /// The fallback slot for a texture semantic when the material omits that map
    /// or the heap is out of capacity. Normal-style maps fall back to the flat
    /// tangent-space normal; everything else falls back to opaque white.
    pub const fn default_slot(semantic: TextureSemantic) -> BindlessSlot {
        match semantic {
            TextureSemantic::Normal | TextureSemantic::ClearCoatNormal => BindlessSlot::FLAT_NORMAL,
            _ => BindlessSlot::WHITE,
        }
    }

    /// Total addressable slots, reserved slots included.
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Highest dynamic slot index ever allocated plus one (the high-water mark).
    /// The bindless upload path only needs to materialise views for slots below
    /// this bound; everything above is guaranteed to still be a reserved default.
    pub const fn high_water(&self) -> u32 {
        self.next
    }

    /// Iterates every currently resident `(slot_index, image)` pair so the
    /// render-world upload system can place each live image's `TextureView` and
    /// `Sampler` at its assigned bindless slot. Reserved slots (0/1/2) are never
    /// yielded because they are owned by the default fallback textures, not by
    /// any asset. Iteration order is unspecified.
    pub fn iter_slots(&self) -> impl Iterator<Item = (u32, AssetId<Image>)> + '_ {
        self.entries
            .iter()
            .map(|(image, entry)| (entry.slot, *image))
    }

    /// Rebuilds the heap with a new total `capacity`, discarding all residency.
    /// Used once at render startup to clamp the slot space to the device's
    /// bindless binding-array limit so `acquire` can never hand out an index the
    /// bound `binding_array` cannot address. Panics if any image is still
    /// resident, because resizing under live references would silently
    /// invalidate GPU material rows.
    pub fn reset_with_capacity(&mut self, capacity: u32) {
        assert!(
            self.entries.is_empty(),
            "bindless texture heap must be empty before its capacity is reconfigured"
        );
        *self = Self::new(capacity);
    }

    /// Acquires the bindless slot for `image`, uploading a fresh slot on first
    /// use and reference counting subsequent uses. Returns [`BindlessSlot::WHITE`]
    /// and records an overflow when capacity is exhausted, so callers always get
    /// a sampleable slot.
    pub fn acquire(&mut self, image: AssetId<Image>) -> BindlessSlot {
        if let Some(entry) = self.entries.get_mut(&image) {
            entry.refcount += 1;
            return BindlessSlot {
                index: entry.slot,
                generation: entry.generation,
            };
        }
        let Some(slot) = self.alloc_slot() else {
            self.overflow += 1;
            return BindlessSlot::WHITE;
        };
        let generation = self.slot_generations[slot as usize];
        self.entries.insert(
            image,
            HeapEntry {
                slot,
                generation,
                refcount: 1,
            },
        );
        BindlessSlot {
            index: slot,
            generation,
        }
    }

    /// Drops one reference to `image`. When the last reference is released the
    /// slot is returned to the free list and its generation is bumped so stale
    /// GPU references no longer validate. Returns `true` when the slot became
    /// free, `false` when the image was still referenced or unknown.
    pub fn release(&mut self, image: AssetId<Image>) -> bool {
        let Some(entry) = self.entries.get_mut(&image) else {
            return false;
        };
        entry.refcount -= 1;
        if entry.refcount > 0 {
            return false;
        }
        let slot = entry.slot;
        self.entries.remove(&image);
        self.bump_generation(slot);
        self.free.push(slot);
        true
    }

    /// Returns the currently resident slot for `image` without changing its
    /// reference count.
    pub fn slot_of(&self, image: AssetId<Image>) -> Option<BindlessSlot> {
        self.entries.get(&image).map(|entry| BindlessSlot {
            index: entry.slot,
            generation: entry.generation,
        })
    }

    /// Returns the live reference count for `image`, or zero when it is not
    /// resident.
    pub fn refcount(&self, image: AssetId<Image>) -> u32 {
        self.entries.get(&image).map_or(0, |entry| entry.refcount)
    }

    /// Current occupancy snapshot for diagnostics and tests.
    pub fn stats(&self) -> BindlessHeapStats {
        BindlessHeapStats {
            capacity: self.capacity,
            live_images: self.entries.len() as u32,
            free_slots: self.free.len() as u32,
            high_water: self.next,
            overflow: self.overflow,
        }
    }

    /// Pops a free slot or extends the high-water mark, honoring the capacity
    /// ceiling. Returns `None` when no dynamic slot is available.
    fn alloc_slot(&mut self) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        if self.next >= self.capacity {
            return None;
        }
        let slot = self.next;
        self.next += 1;
        self.slot_generations.push(RESERVED_GENERATION);
        Some(slot)
    }

    /// Advances a slot's generation, skipping the wrap-around value `0` which is
    /// reserved as "never allocated" in some consumers.
    fn bump_generation(&mut self, slot: u32) {
        let generation = &mut self.slot_generations[slot as usize];
        *generation = generation.wrapping_add(1);
        if *generation == 0 {
            *generation = RESERVED_GENERATION;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::uuid::Uuid;

    fn image(tag: u128) -> AssetId<Image> {
        AssetId::Uuid {
            uuid: Uuid::from_u128(tag),
        }
    }

    #[test]
    fn reserved_slots_are_stable_and_generation_valid() {
        assert_eq!(BindlessSlot::WHITE.index, WHITE_SLOT);
        assert_eq!(BindlessSlot::FLAT_NORMAL.index, FLAT_NORMAL_SLOT);
        assert_eq!(BindlessSlot::BLACK.index, BLACK_SLOT);
        assert_eq!(BindlessSlot::WHITE.generation, RESERVED_GENERATION);
    }

    #[test]
    fn default_slot_maps_normal_maps_to_flat_normal() {
        assert_eq!(
            BindlessTextureHeap::default_slot(TextureSemantic::Normal),
            BindlessSlot::FLAT_NORMAL
        );
        assert_eq!(
            BindlessTextureHeap::default_slot(TextureSemantic::ClearCoatNormal),
            BindlessSlot::FLAT_NORMAL
        );
        assert_eq!(
            BindlessTextureHeap::default_slot(TextureSemantic::BaseColor),
            BindlessSlot::WHITE
        );
        assert_eq!(
            BindlessTextureHeap::default_slot(TextureSemantic::Occlusion),
            BindlessSlot::WHITE
        );
    }

    #[test]
    fn first_acquire_allocates_from_the_dynamic_range() {
        let mut heap = BindlessTextureHeap::new(16);
        let slot = heap.acquire(image(1));
        assert_eq!(slot.index, FIRST_DYNAMIC_SLOT);
        assert_eq!(slot.generation, RESERVED_GENERATION);
        let stats = heap.stats();
        assert_eq!(stats.live_images, 1);
        assert_eq!(stats.high_water, FIRST_DYNAMIC_SLOT + 1);
        assert_eq!(stats.free_slots, 0);
    }

    #[test]
    fn repeated_acquire_reuses_slot_and_reference_counts() {
        let mut heap = BindlessTextureHeap::new(16);
        let first = heap.acquire(image(7));
        let second = heap.acquire(image(7));
        assert_eq!(first, second);
        assert_eq!(heap.refcount(image(7)), 2);
        assert_eq!(heap.stats().live_images, 1);

        // One release keeps it resident; the second frees it.
        assert!(!heap.release(image(7)));
        assert_eq!(heap.refcount(image(7)), 1);
        assert_eq!(heap.slot_of(image(7)), Some(first));
        assert!(heap.release(image(7)));
        assert_eq!(heap.refcount(image(7)), 0);
        assert_eq!(heap.slot_of(image(7)), None);
    }

    #[test]
    fn released_slot_is_reused_with_a_bumped_generation() {
        let mut heap = BindlessTextureHeap::new(16);
        let first = heap.acquire(image(1));
        assert!(heap.release(image(1)));
        assert_eq!(heap.stats().free_slots, 1);

        let reused = heap.acquire(image(2));
        assert_eq!(reused.index, first.index, "the freed slot must be reused");
        assert_eq!(
            reused.generation,
            first.generation + 1,
            "reuse must bump the slot generation so stale references are detectable"
        );
        assert_eq!(heap.stats().free_slots, 0);
    }

    #[test]
    fn distinct_images_get_distinct_slots() {
        let mut heap = BindlessTextureHeap::new(16);
        let a = heap.acquire(image(10));
        let b = heap.acquire(image(20));
        let c = heap.acquire(image(30));
        assert_ne!(a.index, b.index);
        assert_ne!(b.index, c.index);
        assert_ne!(a.index, c.index);
        assert!(a.index >= FIRST_DYNAMIC_SLOT);
        assert_eq!(heap.stats().live_images, 3);
    }

    #[test]
    fn capacity_exhaustion_falls_back_to_white_and_counts_overflow() {
        // Capacity 4 => reserved 0,1,2 + exactly one dynamic slot (index 3).
        let mut heap = BindlessTextureHeap::new(FIRST_DYNAMIC_SLOT + 1);
        let ok = heap.acquire(image(1));
        assert_eq!(ok.index, FIRST_DYNAMIC_SLOT);

        let overflow = heap.acquire(image(2));
        assert_eq!(overflow, BindlessSlot::WHITE);
        assert_eq!(heap.stats().overflow, 1);
        // The overflowed image is not tracked, so releasing it is a no-op.
        assert!(!heap.release(image(2)));
        assert_eq!(heap.refcount(image(2)), 0);

        // Freeing the one real slot lets the next acquire succeed again.
        assert!(heap.release(image(1)));
        let recovered = heap.acquire(image(3));
        assert_eq!(recovered.index, FIRST_DYNAMIC_SLOT);
        assert_eq!(heap.stats().overflow, 1);
    }

    #[test]
    fn releasing_unknown_image_is_a_noop() {
        let mut heap = BindlessTextureHeap::new(16);
        assert!(!heap.release(image(999)));
        assert_eq!(heap.stats().live_images, 0);
    }

    #[test]
    fn generation_skips_zero_on_wrap() {
        let mut heap = BindlessTextureHeap::new(8);
        // Force the slot generation to u32::MAX so the next release wraps.
        heap.acquire(image(1));
        heap.slot_generations[FIRST_DYNAMIC_SLOT as usize] = u32::MAX;
        assert!(heap.release(image(1)));
        let reused = heap.acquire(image(2));
        assert_eq!(reused.generation, RESERVED_GENERATION);
    }
}
