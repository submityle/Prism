//! CPU core of the virtual-shadow-map **page-table bridge**: it turns the
//! GPU page-mark pass's resident-window request bitmap (read back one frame
//! late) into the golden driver's request set, and turns the resulting
//! virtual-page residency back into the flat `window_slot_count`-entry
//! virtual->physical page table the `vsm_sample.wesl` shader indexes.
//!
//! Everything here is pure and CPU-testable; it is the byte-for-byte twin of
//! the GPU `vsm_page_mark.wesl` slot addressing, reusing the golden
//! [`slot_to_page_key`] inverse so the request set decoded from the bitmap is
//! exactly the set the golden `generate_page_requests` front end would have
//! produced for the same receivers.

use bevy_math::Vec2;
use prism_render_shading::{
    slot_to_page_key, ClipmapConfig, Residency, ShadowPageKey, VirtualPageTable,
};

use super::super::abi::VSM_PAGE_UNMAPPED;

/// Scans a resident-window request bitmap and returns the flat window-slot
/// index of every marked (non-zero) slot, in ascending order.
///
/// The GPU page-mark pass marks a slot by `atomicOr`-ing a non-zero bit into
/// its `u32`; a zero word is an unmarked slot. Ascending order keeps the
/// decoded key set deterministic regardless of GPU marking order.
pub(crate) fn marked_slots(bitmap: &[u32]) -> impl Iterator<Item = u32> + '_ {
    bitmap
        .iter()
        .enumerate()
        .filter_map(|(slot, &mark)| (mark != 0).then_some(slot as u32))
}

/// Decodes a marked resident-window request bitmap into the requested
/// [`ShadowPageKey`]s the golden driver consumes.
///
/// Each marked slot is inverted through [`slot_to_page_key`] (decode window
/// slot -> rebuild that level's camera-snapped clipmap window -> absolute world
/// page -> key). Slots that fall outside the addressable clipmap (a stale bit
/// from a since-retuned layout) invert to `None` and are dropped, so the result
/// only ever contains keys the current clipmap can back.
pub(crate) fn request_keys(
    clipmap: &ClipmapConfig,
    camera_light_space: Vec2,
    light: u32,
    bitmap: &[u32],
) -> Vec<ShadowPageKey> {
    marked_slots(bitmap)
        .filter_map(|slot| slot_to_page_key(clipmap, camera_light_space, light, slot))
        .collect()
}

/// Builds the flat `slot_count`-entry virtual->physical page table the
/// `vsm_sample` shader indexes by window slot.
///
/// For every window slot, the entry is the physical atlas page backing that
/// slot's clipmap page when the page is resident in `table`, otherwise
/// [`VSM_PAGE_UNMAPPED`]. The driver promotes freshly rendered pages to
/// resident before returning, so both reused and this-frame-rendered pages
/// resolve to their backing physical page here; only genuinely unbacked slots
/// stay unmapped.
///
/// `camera_light_space` and `clipmap` must match the ones the request bitmap
/// was marked against, so slot -> key inversion lands on the same pages the
/// driver made resident.
pub(crate) fn build_page_table(
    clipmap: &ClipmapConfig,
    camera_light_space: Vec2,
    light: u32,
    slot_count: u32,
    table: &VirtualPageTable,
) -> Vec<u32> {
    (0..slot_count)
        .map(|slot| {
            slot_to_page_key(clipmap, camera_light_space, light, slot)
                .and_then(|key| table.get(&key))
                .and_then(Residency::physical_page)
                .unwrap_or(VSM_PAGE_UNMAPPED)
        })
        .collect()
}

/// Forward window-slot encoder mirror used by tests to assert the decode is a
/// true inverse of the golden slot addressing.
#[cfg(test)]
pub(crate) fn forward_slot(
    clipmap: &ClipmapConfig,
    camera_light_space: Vec2,
    key: ShadowPageKey,
) -> Option<u32> {
    let level = key.level;
    let window = clipmap.build_level(level, camera_light_space);
    let page = bevy_math::IVec2::new(
        i32::from(key.x) - clipmap.page_coord_bias,
        i32::from(key.y) - clipmap.page_coord_bias,
    );
    let local = window.local_page(page)?;
    Some(prism_render_shading::window_slot(clipmap, level, local))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::UVec2;
    use prism_render_shading::{
        window_slot, window_slot_count, VirtualShadowMap, VirtualShadowSettings,
    };

    fn test_clipmap() -> ClipmapConfig {
        let settings = VirtualShadowSettings {
            physical_pages: 64,
            ..Default::default()
        };
        ClipmapConfig::from_settings(&settings, 8, 0.1, 10.0, 32_768)
    }

    #[test]
    fn marked_slots_reports_every_nonzero_slot_ascending() {
        let bitmap = [0u32, 1, 0, 7, 0, 0, 3];
        let slots: Vec<u32> = marked_slots(&bitmap).collect();
        assert_eq!(slots, vec![1, 3, 6]);
    }

    #[test]
    fn request_keys_invert_the_marked_slots_to_addressable_pages() {
        let clipmap = test_clipmap();
        let camera = Vec2::new(3.25, -1.5);
        let light = 0;
        let level = 2u16;
        let local = UVec2::new(3, 5);
        let slot = window_slot(&clipmap, level, local);
        let mut bitmap = vec![0u32; window_slot_count(&clipmap) as usize];
        bitmap[slot as usize] = 1;

        let keys = request_keys(&clipmap, camera, light, &bitmap);
        assert_eq!(keys.len(), 1);
        assert_eq!(forward_slot(&clipmap, camera, keys[0]), Some(slot));
    }

    #[test]
    fn page_table_maps_resident_slots_and_leaves_the_rest_unmapped() {
        let clipmap = test_clipmap();
        let camera = Vec2::ZERO;
        let light = 0;
        let slot_count = window_slot_count(&clipmap);

        let settings = VirtualShadowSettings {
            physical_pages: 64,
            ..Default::default()
        };
        let mut vsm = VirtualShadowMap::from_settings(&settings, 8, 0.1, 10.0, 32_768);

        let mut bitmap = vec![0u32; slot_count as usize];
        let slot_a = window_slot(&clipmap, 0, UVec2::new(4, 4));
        let slot_b = window_slot(&clipmap, 1, UVec2::new(2, 6));
        bitmap[slot_a as usize] = 1;
        bitmap[slot_b as usize] = 1;
        let keys = request_keys(&clipmap, camera, light, &bitmap);
        assert_eq!(keys.len(), 2);

        let result = vsm.drive_frame_with_requests(light, camera, false, &keys, &[]);
        assert_eq!(result.to_render.len(), 2);

        let page_table = build_page_table(&clipmap, camera, light, slot_count, vsm.table());
        assert_eq!(page_table.len(), slot_count as usize);
        assert_ne!(page_table[slot_a as usize], VSM_PAGE_UNMAPPED);
        assert_ne!(page_table[slot_b as usize], VSM_PAGE_UNMAPPED);
        assert_ne!(page_table[slot_a as usize], page_table[slot_b as usize]);
        let empty_slot = window_slot(&clipmap, 3, UVec2::new(7, 7));
        assert_eq!(page_table[empty_slot as usize], VSM_PAGE_UNMAPPED);
    }
}
