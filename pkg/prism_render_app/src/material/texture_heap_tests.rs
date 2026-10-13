//! Behavioural tests for the bindless texture slot heap.

use crate::material::texture_heap::{
    BindlessSlot, TextureFallback, TextureSlotHeap, BLACK_SLOT, FIRST_DYNAMIC_SLOT,
    FLAT_NORMAL_SLOT, RESERVED_GENERATION, WHITE_SLOT,
};
use prism_asset::{AssetId, AssetIndex};
use prism_image::Image;

fn image(tag: u32) -> AssetId<Image> {
    AssetId::new(AssetIndex::from_parts(tag, 1))
}

#[test]
fn reserved_slots_are_stable_and_generation_valid() {
    assert_eq!(BindlessSlot::WHITE.index, WHITE_SLOT);
    assert_eq!(BindlessSlot::FLAT_NORMAL.index, FLAT_NORMAL_SLOT);
    assert_eq!(BindlessSlot::BLACK.index, BLACK_SLOT);
    assert_eq!(BindlessSlot::WHITE.generation, RESERVED_GENERATION);
}

/// Each fallback resolves to its own reserved slot — a missing normal map must
/// be the flat tangent-space normal, not white (which is not a unit-length
/// normal at all).
#[test]
fn every_fallback_resolves_to_its_own_reserved_slot() {
    assert_eq!(
        TextureSlotHeap::fallback_slot(TextureFallback::FlatNormal),
        BindlessSlot::FLAT_NORMAL
    );
    assert_eq!(
        TextureSlotHeap::fallback_slot(TextureFallback::White),
        BindlessSlot::WHITE
    );
    assert_eq!(
        TextureSlotHeap::fallback_slot(TextureFallback::Black),
        BindlessSlot::BLACK
    );
    assert_ne!(
        TextureSlotHeap::fallback_slot(TextureFallback::FlatNormal),
        TextureSlotHeap::fallback_slot(TextureFallback::White)
    );
}

#[test]
fn the_first_acquire_allocates_from_the_dynamic_range() {
    let mut heap = TextureSlotHeap::new(16);
    let slot = heap.acquire(image(1));
    assert_eq!(slot.index, FIRST_DYNAMIC_SLOT);
    assert_eq!(slot.generation, RESERVED_GENERATION);
    let stats = heap.stats();
    assert_eq!(stats.live_images, 1);
    assert_eq!(stats.high_water, FIRST_DYNAMIC_SLOT + 1);
    assert_eq!(stats.free_slots, 0);
}

#[test]
fn repeated_acquires_reuse_the_slot_and_reference_count() {
    let mut heap = TextureSlotHeap::new(16);
    let first = heap.acquire(image(7));
    let second = heap.acquire(image(7));
    assert_eq!(first, second, "the same image keeps its slot");
    assert_eq!(heap.refcount(image(7)), 2);
    assert_eq!(heap.stats().live_images, 1);

    assert!(!heap.release(image(7)), "one reference is still held");
    assert_eq!(heap.refcount(image(7)), 1);
    assert_eq!(heap.slot_of(image(7)), Some(first));
    assert!(heap.release(image(7)), "the last reference frees the slot");
    assert_eq!(heap.refcount(image(7)), 0);
    assert_eq!(heap.slot_of(image(7)), None);
}

/// Reuse has to be *detectable*: a row that still names the old `(index,
/// generation)` pair must stop validating once the slot is handed out again.
#[test]
fn a_reused_slot_bumps_its_generation_and_invalidates_the_old_row() {
    let mut heap = TextureSlotHeap::new(16);
    let first = heap.acquire(image(1));
    assert!(heap.validates(first));
    assert!(heap.release(image(1)));
    assert_eq!(heap.stats().free_slots, 1);
    assert!(
        !heap.validates(first),
        "the freed slot no longer validates the old row"
    );
    assert_eq!(heap.slot_of(image(1)), None);

    let reused = heap.acquire(image(2));
    assert_eq!(reused.index, first.index, "the freed slot is reused");
    assert_eq!(
        reused.generation,
        first.generation + 1,
        "reuse bumps the generation so stale references are detectable"
    );
    assert!(heap.validates(reused));
    assert_eq!(heap.stats().free_slots, 0);
}

#[test]
fn distinct_images_get_distinct_slots() {
    let mut heap = TextureSlotHeap::new(16);
    let a = heap.acquire(image(10));
    let b = heap.acquire(image(20));
    let c = heap.acquire(image(30));
    assert_ne!(a.index, b.index);
    assert_ne!(b.index, c.index);
    assert_ne!(a.index, c.index);
    assert!(a.index >= FIRST_DYNAMIC_SLOT);
    assert_eq!(heap.stats().live_images, 3);
    assert_eq!(heap.high_water(), FIRST_DYNAMIC_SLOT + 3);
}

/// At capacity the heap degrades to the white fallback instead of panicking or
/// handing out an index the bound array cannot address.
#[test]
fn capacity_exhaustion_falls_back_to_white_and_counts_the_overflow() {
    // Reserved 0/1/2 plus exactly one dynamic slot.
    let mut heap = TextureSlotHeap::new(FIRST_DYNAMIC_SLOT + 1);
    let ok = heap.acquire(image(1));
    assert_eq!(ok.index, FIRST_DYNAMIC_SLOT);

    let overflow = heap.acquire(image(2));
    assert_eq!(overflow, BindlessSlot::WHITE);
    assert_eq!(heap.stats().overflow, 1);
    assert!(
        !heap.release(image(2)),
        "the overflowed image is not tracked"
    );
    assert_eq!(heap.refcount(image(2)), 0);

    assert!(heap.release(image(1)));
    let recovered = heap.acquire(image(3));
    assert_eq!(
        recovered.index, FIRST_DYNAMIC_SLOT,
        "freeing a slot lets the next acquire succeed"
    );
    assert_eq!(heap.stats().overflow, 1, "the count is cumulative");
}

#[test]
fn releasing_an_unknown_image_is_a_noop() {
    let mut heap = TextureSlotHeap::new(16);
    assert!(!heap.release(image(999)));
    assert_eq!(heap.stats().live_images, 0);
}

/// The upload path walks exactly the resident images, so every live slot gets a
/// view placed and the reserved fallbacks are left to their own textures.
#[test]
fn iter_slots_yields_every_resident_image_in_a_deterministic_order() {
    let mut heap = TextureSlotHeap::new(16);
    heap.acquire(image(30));
    heap.acquire(image(10));
    heap.acquire(image(20));
    let slots: Vec<(u32, AssetId<Image>)> = heap.iter_slots().collect();
    assert_eq!(slots.len(), 3);
    assert!(slots.iter().all(|(slot, _)| *slot >= FIRST_DYNAMIC_SLOT));
    let images: Vec<u32> = slots
        .iter()
        .map(|(_, image)| image.index().index())
        .collect();
    assert_eq!(images, vec![10, 20, 30], "ordered by image identity");
    // Slots follow *acquisition* order (30 → 3, 10 → 4, 20 → 5), so iterating by
    // image identity yields them out of order — which is what makes the order
    // deterministic rather than merely "the order the map happens to produce".
    let indices: Vec<u32> = slots.iter().map(|(slot, _)| *slot).collect();
    let expected: Vec<u32> = [image(10), image(20), image(30)]
        .into_iter()
        .map(|image| heap.slot_of(image).expect("resident").index)
        .collect();
    assert_eq!(indices, expected);
}

/// Reconfiguring the capacity (to the device's binding-array limit) is only safe
/// while nothing is resident: a resize under live rows would silently invalidate
/// the indices they name.
#[test]
fn resetting_the_capacity_requires_an_empty_heap() {
    let mut heap = TextureSlotHeap::new(16);
    heap.acquire(image(1));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        heap.reset_with_capacity(8);
    }));
    assert!(result.is_err(), "resizing under live references panics");

    let mut heap = TextureSlotHeap::new(16);
    heap.acquire(image(1));
    heap.release(image(1));
    heap.reset_with_capacity(FIRST_DYNAMIC_SLOT + 32);
    assert_eq!(heap.capacity(), FIRST_DYNAMIC_SLOT + 32);
    assert_eq!(
        heap.high_water(),
        FIRST_DYNAMIC_SLOT,
        "residency is discarded"
    );
    assert_eq!(heap.stats().live_images, 0);
}
