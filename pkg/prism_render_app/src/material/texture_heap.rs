//! Bounded, reference-counted allocator of bindless texture slots.
//!
//! Every texture a material references is assigned a stable *bindless slot*: an
//! index into the `texture_2d<f32>` binding array the shading passes sample. The
//! ABI's texture rows carry that index (`GpuMaterialTexture::index`), and the
//! shader reads `material_texture_heap[record.index]`, so a slot has to stay
//! valid for as long as a material row can point at it — and a *reused* slot has
//! to be detectably different from the one a stale row still names.
//!
//! The heap owns that lifetime end to end, so no raw index leaks into a
//! material:
//!
//! * a small block of **reserved slots** ([`WHITE_SLOT`], [`FLAT_NORMAL_SLOT`],
//!   [`BLACK_SLOT`]) is always present, so every texture semantic has a correct
//!   fallback before any image is uploaded — a missing normal map resolves to the
//!   flat tangent-space normal, not to whatever happens to be in slot 0;
//! * real images are **reference counted**: acquiring an already-resident image
//!   reuses its slot, and releasing only frees it once the last material stops
//!   using it;
//! * a freed slot's **generation is bumped**, so a lingering row that still names
//!   the old `(index, generation)` pair is detected as stale rather than silently
//!   sampling a different texture;
//! * the heap is **bounded**: at capacity, acquire degrades to [`WHITE_SLOT`] and
//!   counts the overflow instead of growing without limit or panicking.
//!
//! Slot generations mirror the generational-handle discipline the scene handles
//! use, so consumers can apply one staleness rule to both.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use prism_asset::AssetId;
use prism_image::Image;

/// Reserved slot holding an opaque white (`1, 1, 1, 1`) texel. Fallback for
/// base colour, metallic-roughness, occlusion, and emissive maps.
pub const WHITE_SLOT: u32 = 0;
/// Reserved slot holding the flat tangent-space normal (`0.5, 0.5, 1.0`).
/// Fallback for normal and clear-coat-normal maps.
pub const FLAT_NORMAL_SLOT: u32 = 1;
/// Reserved slot holding an opaque black (`0, 0, 0, 1`) texel, for fallbacks that
/// must contribute nothing.
pub const BLACK_SLOT: u32 = 2;
/// The first slot available to a real image; every lower index is permanently a
/// fallback.
pub const FIRST_DYNAMIC_SLOT: u32 = 3;
/// The generation stamped on the reserved slots. They never move, so it is fixed.
pub const RESERVED_GENERATION: u32 = 1;
/// The default capacity (reserved slots included).
pub const DEFAULT_CAPACITY: u32 = 1 << 16;

/// Which reserved slot a semantic resolves to when its map is missing or the
/// heap is full.
///
/// The *mapping* from a texture semantic
/// ([`prism_render_material::TextureSemantic`], which lives with the material
/// ABI's Bevy bridge) to one of these is the material upload path's business:
/// the renderer only has to provide the three fallback textures and know which
/// slot holds which.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextureFallback {
    /// Opaque white: base colour, metallic-roughness, occlusion, emissive.
    White,
    /// The flat tangent-space normal `(0.5, 0.5, 1.0)`: normal and clear-coat
    /// normal maps.
    FlatNormal,
    /// Opaque black, for a fallback that must contribute nothing.
    Black,
}

/// A stable bindless texture slot: an index into the sampled-texture binding
/// array, plus the generation that was live when it was handed out.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindlessSlot {
    /// Index into the `texture_2d<f32>` binding array.
    pub index: u32,
    /// The slot's generation when it was acquired. A material row may sample the
    /// slot only while this matches the heap's current generation for `index`.
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

/// Residency record for one live image.
#[derive(Clone, Copy, Debug)]
struct HeapEntry {
    slot: u32,
    generation: u32,
    refcount: u32,
}

/// Occupancy snapshot, for diagnostics and soak tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TextureHeapStats {
    /// Total addressable slots, reserved ones included.
    pub capacity: u32,
    /// Distinct images currently resident (refcount ≥ 1).
    pub live_images: u32,
    /// Dynamic slots sitting on the free list.
    pub free_slots: u32,
    /// One past the highest dynamic slot ever allocated.
    pub high_water: u32,
    /// Acquisitions that hit the capacity ceiling and fell back to white.
    pub overflow: u32,
}

/// Bounded, reference-counted allocator of bindless texture slots.
pub struct TextureSlotHeap {
    capacity: u32,
    entries: BTreeMap<AssetId<Image>, HeapEntry>,
    /// The current generation per slot index, including the reserved slots.
    slot_generations: Vec<u32>,
    free: Vec<u32>,
    next: u32,
    overflow: u32,
}

impl Default for TextureSlotHeap {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

impl TextureSlotHeap {
    /// A heap with room for `capacity` total slots (reserved slots included).
    ///
    /// # Panics
    /// Panics unless the capacity leaves room for at least one real image.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        assert!(
            capacity > FIRST_DYNAMIC_SLOT,
            "a bindless texture heap needs the reserved slots plus room for at least one image"
        );
        Self {
            capacity,
            entries: BTreeMap::new(),
            slot_generations: vec![RESERVED_GENERATION; FIRST_DYNAMIC_SLOT as usize],
            free: Vec::new(),
            next: FIRST_DYNAMIC_SLOT,
            overflow: 0,
        }
    }

    /// The reserved slot a fallback resolves to.
    ///
    /// A missing map resolves to the slot its *meaning* calls for, not to
    /// whichever slot happens to be free: a missing normal map must be the flat
    /// tangent-space normal, not white (which would be a wrong, unit-length
    /// normal).
    #[must_use]
    pub const fn fallback_slot(fallback: TextureFallback) -> BindlessSlot {
        match fallback {
            TextureFallback::White => BindlessSlot::WHITE,
            TextureFallback::FlatNormal => BindlessSlot::FLAT_NORMAL,
            TextureFallback::Black => BindlessSlot::BLACK,
        }
    }

    /// Total addressable slots, reserved ones included.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// One past the highest dynamic slot ever allocated.
    ///
    /// The upload path only has to materialize views below this bound; above it
    /// every slot is guaranteed to still be a reserved default.
    #[must_use]
    pub const fn high_water(&self) -> u32 {
        self.next
    }

    /// Every resident `(slot, image)` pair, so the upload path can place each
    /// live image's view and sampler at its assigned slot.
    ///
    /// The reserved slots are never yielded: they belong to the fallback
    /// textures, not to any asset. Order is by image identity, which makes a
    /// frame's upload order deterministic.
    pub fn iter_slots(&self) -> impl Iterator<Item = (u32, AssetId<Image>)> + '_ {
        self.entries
            .iter()
            .map(|(image, entry)| (entry.slot, *image))
    }

    /// Rebuilds the heap with a new capacity, discarding all residency.
    ///
    /// Called once while wiring the device's binding-array limit in, so acquire
    /// can never hand out an index the bound array cannot address.
    ///
    /// # Panics
    /// Panics if any image is still resident: resizing under live references
    /// would silently invalidate the rows that name them.
    pub fn reset_with_capacity(&mut self, capacity: u32) {
        assert!(
            self.entries.is_empty(),
            "the texture heap must be empty before its capacity is reconfigured"
        );
        *self = Self::new(capacity);
    }

    /// The slot for `image`, uploading a fresh one on first use and counting
    /// subsequent uses as references.
    ///
    /// At capacity this returns [`BindlessSlot::WHITE`] and counts an overflow, so
    /// a caller always gets a sampleable slot rather than a panic or an
    /// out-of-range index.
    pub fn acquire(&mut self, image: AssetId<Image>) -> BindlessSlot {
        if let Some(entry) = self.entries.get_mut(&image) {
            entry.refcount += 1;
            return BindlessSlot {
                index: entry.slot,
                generation: entry.generation,
            };
        }
        let Some(slot) = self.allocate_slot() else {
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

    /// Drops one reference to `image`.
    ///
    /// Returns `true` when that released the slot (it goes back to the free list
    /// and its generation is bumped, so stale rows stop validating). Returns
    /// `false` when the image still has references or was never resident.
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

    /// The image resident at `slot`, if any.
    ///
    /// The inverse of [`slot_of`](Self::slot_of), for consumers that walk slots
    /// (the bindless upload path places a view per occupied slot).
    #[must_use]
    pub fn image_at(&self, slot: u32) -> Option<AssetId<Image>> {
        self.entries
            .iter()
            .find_map(|(image, entry)| (entry.slot == slot).then_some(*image))
    }

    /// The resident slot for `image`, without changing its reference count.
    #[must_use]
    pub fn slot_of(&self, image: AssetId<Image>) -> Option<BindlessSlot> {
        self.entries.get(&image).map(|entry| BindlessSlot {
            index: entry.slot,
            generation: entry.generation,
        })
    }

    /// The live reference count for `image`, or zero when it is not resident.
    #[must_use]
    pub fn refcount(&self, image: AssetId<Image>) -> u32 {
        self.entries.get(&image).map_or(0, |entry| entry.refcount)
    }

    /// The current generation of `index`, for validating a row that names a slot.
    #[must_use]
    pub fn generation_of(&self, index: u32) -> Option<u32> {
        self.slot_generations.get(index as usize).copied()
    }

    /// Whether a row naming `(index, generation)` may still sample that slot.
    #[must_use]
    pub fn validates(&self, slot: BindlessSlot) -> bool {
        self.generation_of(slot.index) == Some(slot.generation)
    }

    /// The occupancy snapshot.
    #[must_use]
    pub fn stats(&self) -> TextureHeapStats {
        TextureHeapStats {
            capacity: self.capacity,
            live_images: self.entries.len() as u32,
            free_slots: self.free.len() as u32,
            high_water: self.next,
            overflow: self.overflow,
        }
    }

    /// Takes a dynamic slot, reusing a freed one before advancing the high-water
    /// mark.
    fn allocate_slot(&mut self) -> Option<u32> {
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

    /// Advances a slot's generation when it is freed.
    fn bump_generation(&mut self, slot: u32) {
        let generation = &mut self.slot_generations[slot as usize];
        *generation = generation.wrapping_add(1).max(1);
    }
}

#[cfg(test)]
mod tests {
    use prism_asset::{AssetId, AssetIndex};

    use super::{TextureSlotHeap, FIRST_DYNAMIC_SLOT, RESERVED_GENERATION};

    /// The generation skips zero on wrap, so `0` stays available as "never handed
    /// out" for consumers that reserve it. The test pokes the slot's generation
    /// directly because no public operation can fast-forward it to the wrap.
    #[test]
    fn generation_wraps_to_the_reserved_value_not_zero() {
        let mut heap = TextureSlotHeap::new(8);
        let image = AssetId::<prism_image::Image>::new(AssetIndex::from_parts(1, 1));
        heap.acquire(image);
        heap.slot_generations[FIRST_DYNAMIC_SLOT as usize] = u32::MAX;
        assert!(heap.release(image));
        let other = AssetId::<prism_image::Image>::new(AssetIndex::from_parts(2, 1));
        let reused = heap.acquire(other);
        assert_eq!(
            reused.generation, RESERVED_GENERATION,
            "zero is never a live generation"
        );
    }
}
