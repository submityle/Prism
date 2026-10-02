//! Window-slot codec: the pure CPU twin of the resident-window slot addressing
//! the GPU page-mark pass (`shaders/vsm_page_mark.wesl`) writes.
//!
//! The page-mark compute pass marks requested pages in a flat per-level,
//! camera-snapped resident-window bitmap using
//!
//! ```text
//! slot = level * edge*edge + local_y * edge + local_x
//! ```
//!
//! where `edge = pages_per_level_edge` and `(local_x, local_y)` are the page's
//! coordinates inside that level's camera-snapped window (its `origin_page` is
//! [`ClipmapConfig::build_level`]'s, i.e. `floor(camera / page_world_size) -
//! edge/2`). The readback bridge reads that bitmap back and must invert the
//! encoding to recover which absolute [`ShadowPageKey`] each marked slot names,
//! so it can drive residency and fill the page table. These functions are that
//! inverse (and its forward direction), kept pure and golden-tested so the CPU
//! decode stays byte-exact with the shader's encode.

use bevy_math::{IVec2, UVec2, Vec2};

use super::clipmap::ClipmapConfig;
use super::ShadowPageKey;

/// Pages along one edge of a level's resident window (always `>= 1`).
#[inline]
fn window_edge(clipmap: &ClipmapConfig) -> u32 {
    u32::from(clipmap.pages_per_level_edge.max(1))
}

/// Number of window slots one level occupies in the flat bitmap: `edge * edge`.
#[inline]
pub fn window_slots_per_level(clipmap: &ClipmapConfig) -> u32 {
    let edge = window_edge(clipmap);
    edge * edge
}

/// Total number of window slots across every level: `levels * edge * edge`.
///
/// This is the length of the request bitmap the page-mark pass writes and the
/// bridge reads back; twin of the device-side `window_slot_count`.
#[inline]
pub fn window_slot_count(clipmap: &ClipmapConfig) -> u32 {
    u32::from(clipmap.level_count()) * window_slots_per_level(clipmap)
}

/// Encodes a `(level, window-local page)` pair into its flat window slot,
/// exactly as the page-mark shader does. `local` must lie inside the level's
/// `edge x edge` window.
#[inline]
pub fn window_slot(clipmap: &ClipmapConfig, level: u16, local: UVec2) -> u32 {
    let edge = window_edge(clipmap);
    let level = u32::from(level.min(clipmap.level_count() - 1));
    level * edge * edge + local.y * edge + local.x
}

/// Decodes a flat window slot back into its `(level, window-local page)` pair,
/// or `None` when the slot is out of range for this clipmap.
#[inline]
pub fn decode_window_slot(clipmap: &ClipmapConfig, slot: u32) -> Option<(u16, UVec2)> {
    if slot >= window_slot_count(clipmap) {
        return None;
    }
    let edge = window_edge(clipmap);
    let per_level = edge * edge;
    let level = slot / per_level;
    let rem = slot % per_level;
    let local = UVec2::new(rem % edge, rem / edge);
    Some((level as u16, local))
}

/// Resolves a marked window slot to the absolute [`ShadowPageKey`] it names for
/// `light`, given the camera position (in the light's 2D space) that snapped
/// every level's resident window this frame.
///
/// This is the inverse of the GPU page-mark encode: decode the slot to its
/// `(level, local)`, rebuild the level's camera-snapped window with the same
/// [`ClipmapConfig::build_level`] the shader mirrors, offset `local` by the
/// window `origin_page` to get the absolute world page, then bias it into a key
/// with [`ClipmapConfig::page_key`]. Returns `None` for an out-of-range slot or
/// a page whose biased coordinate overflows the key.
pub fn slot_to_page_key(
    clipmap: &ClipmapConfig,
    camera_light_space: Vec2,
    light: u32,
    slot: u32,
) -> Option<ShadowPageKey> {
    let (level, local) = decode_window_slot(clipmap, slot)?;
    let window = clipmap.build_level(level, camera_light_space);
    let page = window.origin_page + IVec2::new(local.x as i32, local.y as i32);
    clipmap.page_key(light, level, page)
}

#[cfg(test)]
mod tests {
    use super::super::request::{generate_page_requests, Receiver};
    use super::*;

    fn config() -> ClipmapConfig {
        ClipmapConfig {
            levels: 4,
            page_size: 128,
            pages_per_level_edge: 8,
            level0_texel_world_size: 0.1,
            level0_max_distance: 10.0,
            page_coord_bias: 32_768,
        }
    }

    #[test]
    fn slot_count_is_levels_times_edge_squared() {
        let c = config();
        assert_eq!(window_slots_per_level(&c), 64);
        assert_eq!(window_slot_count(&c), 4 * 64);
    }

    #[test]
    fn window_slot_round_trips_through_decode() {
        let c = config();
        for slot in 0..window_slot_count(&c) {
            let (level, local) = decode_window_slot(&c, slot).expect("in range");
            assert!(local.x < 8 && local.y < 8);
            assert_eq!(window_slot(&c, level, local), slot);
        }
    }

    #[test]
    fn out_of_range_slot_decodes_to_none() {
        let c = config();
        assert!(decode_window_slot(&c, window_slot_count(&c)).is_none());
        assert!(slot_to_page_key(&c, Vec2::ZERO, 0, window_slot_count(&c)).is_none());
    }

    /// The core inverse property the bridge relies on: every page the golden
    /// request generator asks for maps to a slot inside its level's snapped
    /// window, and decoding that slot recovers the exact same key. This ties the
    /// slot codec to the same window snapping the page-mark shader uses.
    #[test]
    fn requested_keys_round_trip_through_their_window_slot() {
        let c = config();
        let camera = Vec2::new(3.0, -5.0);
        let receivers = [
            Receiver {
                light_space_xy: camera,
                view_distance: 1.0,
                filter_radius_texels: 0.0,
            },
            Receiver {
                light_space_xy: camera + Vec2::new(40.0, 0.0),
                view_distance: 1.0,
                filter_radius_texels: 0.0,
            },
            Receiver {
                light_space_xy: camera + Vec2::new(0.0, 30.0),
                view_distance: 120.0,
                filter_radius_texels: 1.0,
            },
            Receiver {
                light_space_xy: camera - Vec2::new(25.0, 15.0),
                view_distance: 400.0,
                filter_radius_texels: 0.0,
            },
        ];
        let requests = generate_page_requests(&c, 0, &receivers);
        assert!(!requests.keys.is_empty());
        for key in &requests.keys {
            // Rebuild the same snapped window and locate the key inside it.
            let window = c.build_level(key.level, camera);
            let abs_page = IVec2::new(
                i32::from(key.x) - c.page_coord_bias,
                i32::from(key.y) - c.page_coord_bias,
            );
            let local = window
                .local_page(abs_page)
                .expect("requested key lies in its snapped window");
            let slot = window_slot(&c, key.level, local);
            // Decoding that slot must recover the identical key.
            assert_eq!(slot_to_page_key(&c, camera, 0, slot).as_ref(), Some(key));
        }
    }
}
