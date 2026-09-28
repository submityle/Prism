//! Caster-movement page invalidation and the camera-static reuse rule.
//!
//! A resident page's cached depth stays valid until something *inside* it
//! moves.  When a shadow *caster* moves, only the pages its world-space bounds
//! sweep over (at every clip level) need their depth re-rendered; the rest of
//! the working set is reused untouched.  This module turns a caster's light-
//! space bounding rectangle into that set of invalidated [`ShadowPageKey`]s and
//! records which clip levels were affected.
//!
//! Crucially, **camera translation does not invalidate any page** — the whole
//! point of the clipmap's world-space page addressing (see `clipmap.rs`) is
//! that a static caster keeps the same page identity as the camera pans, so its
//! depth is reused.  [`camera_move_invalidates_pages`] encodes that guarantee.
//!
//! Caster invalidation is a *scene* change, so [`Invalidation::history_mask`]
//! maps it onto the architecture [`InvalidationMask::SCENE`] bit, aligning this
//! per-light page set with the engine-wide history-invalidation concept.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use bevy_math::{IVec2, Vec2};

use crate::shadow::virtual_sm::clipmap::ClipmapConfig;
use crate::shadow::virtual_sm::page_table::{key_from_order, page_order};
use prism_render_architecture::history::InvalidationMask;
use prism_render_architecture::virtual_shadow::ShadowPageKey;

/// A shadow caster's world-space movement, expressed as the light-space
/// axis-aligned rectangle its bounds swept this frame (the union of its old and
/// new footprints).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CasterMovement {
    /// Minimum corner of the swept rectangle in the light clipmap plane.
    pub light_space_min: Vec2,
    /// Maximum corner of the swept rectangle in the light clipmap plane.
    pub light_space_max: Vec2,
}

impl CasterMovement {
    /// Normalises the corners so `min <= max` on each axis.
    fn normalized(&self) -> (Vec2, Vec2) {
        (
            self.light_space_min.min(self.light_space_max),
            self.light_space_min.max(self.light_space_max),
        )
    }
}

/// The invalidation produced for one light this frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Invalidation {
    /// Light these invalidations belong to.
    pub light: u32,
    /// Pages whose depth must be re-rendered, in deterministic key order.
    pub pages: Vec<ShadowPageKey>,
    /// Bit `l` is set when clip level `l` had at least one page invalidated.
    pub level_mask: u32,
}

impl Invalidation {
    /// Whether any page was invalidated this frame.
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Whether clip `level` had any page invalidated.
    pub fn level_dirty(&self, level: u16) -> bool {
        level < 32 && (self.level_mask & (1u32 << level)) != 0
    }

    /// Maps this per-light page invalidation onto the engine-wide history
    /// invalidation concept: a moved caster is a scene change, so a non-empty
    /// invalidation reports [`InvalidationMask::SCENE`] (and nothing otherwise).
    pub fn history_mask(&self) -> InvalidationMask {
        if self.pages.is_empty() {
            InvalidationMask::default()
        } else {
            InvalidationMask::SCENE
        }
    }
}

/// Whether translating the camera invalidates cached shadow pages.
///
/// Always `false`: clipmap pages are addressed in world space, so a static
/// caster keeps its page identity (and cached depth) as the camera moves. This
/// constant documents — and lets tests pin — the reuse guarantee.
pub const fn camera_move_invalidates_pages() -> bool {
    false
}

/// Computes the pages invalidated for `light` by `casters` across `levels`.
///
/// For every caster rectangle and every requested clip level, the rectangle's
/// covered page range is added to the invalidation set and the level's bit is
/// marked. Pages outside the representable key range are skipped.
pub fn invalidate_casters(
    clipmap: &ClipmapConfig,
    light: u32,
    casters: &[CasterMovement],
    levels: &[u16],
) -> Invalidation {
    let mut pages = BTreeSet::new();
    let mut level_mask = 0u32;
    for caster in casters {
        let (min, max) = caster.normalized();
        for &level in levels {
            let min_page = clipmap.world_page_coords(level, min);
            let max_page = clipmap.world_page_coords(level, max);
            let mut dirtied = false;
            for y in min_page.y..=max_page.y {
                for x in min_page.x..=max_page.x {
                    if let Some(key) = clipmap.page_key(light, level, IVec2::new(x, y)) {
                        pages.insert(page_order(&key));
                        dirtied = true;
                    }
                }
            }
            if dirtied && level < 32 {
                level_mask |= 1u32 << level;
            }
        }
    }
    let pages = pages.into_iter().map(key_from_order).collect();
    Invalidation {
        light,
        pages,
        level_mask,
    }
}

#[cfg(test)]
mod tests {
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

    /// A caster contained in one page invalidates exactly that page and marks
    /// only its level.
    #[test]
    fn small_caster_invalidates_one_page() {
        let c = config();
        // page_world_size(0) = 12.8; a tiny rect near the origin sits in one page.
        let caster = CasterMovement {
            light_space_min: Vec2::new(1.0, 1.0),
            light_space_max: Vec2::new(2.0, 2.0),
        };
        let inv = invalidate_casters(&c, 3, &[caster], &[0]);
        assert_eq!(inv.pages.len(), 1);
        assert_eq!(inv.light, 3);
        assert!(inv.level_dirty(0));
        assert!(!inv.level_dirty(1));
        assert_eq!(inv.history_mask(), InvalidationMask::SCENE);
    }

    /// A rectangle spanning several pages invalidates the whole covered range.
    #[test]
    fn wide_caster_covers_a_page_rectangle() {
        let c = config();
        let pws = c.page_world_size(0); // 12.8
        // Span ~2 pages in x and ~2 in y => a 3x3-or-so covered block.
        let caster = CasterMovement {
            light_space_min: Vec2::new(0.5, 0.5),
            light_space_max: Vec2::new(pws * 2.0 + 0.5, pws * 2.0 + 0.5),
        };
        let inv = invalidate_casters(&c, 0, &[caster], &[0]);
        let min_page = c.world_page_coords(0, caster.light_space_min);
        let max_page = c.world_page_coords(0, caster.light_space_max);
        let expected =
            ((max_page.x - min_page.x + 1) * (max_page.y - min_page.y + 1)) as usize;
        assert_eq!(inv.pages.len(), expected);
    }

    /// The same caster invalidates independent page sets on each requested
    /// level and marks each level's bit.
    #[test]
    fn multiple_levels_are_marked_independently() {
        let c = config();
        let caster = CasterMovement {
            light_space_min: Vec2::new(1.0, 1.0),
            light_space_max: Vec2::new(2.0, 2.0),
        };
        let inv = invalidate_casters(&c, 0, &[caster], &[0, 2]);
        assert!(inv.level_dirty(0));
        assert!(inv.level_dirty(2));
        assert!(!inv.level_dirty(1));
        // Each level contributes its own page key (distinct level field).
        let levels: BTreeSet<u16> = inv.pages.iter().map(|k| k.level).collect();
        assert_eq!(levels, BTreeSet::from([0, 2]));
    }

    /// No casters means nothing is invalidated, and the history mask stays
    /// empty (the camera-static reuse path).
    #[test]
    fn no_casters_reuses_everything() {
        let c = config();
        let inv = invalidate_casters(&c, 0, &[], &[0, 1, 2, 3]);
        assert!(inv.is_empty());
        assert_eq!(inv.level_mask, 0);
        assert_eq!(inv.history_mask(), InvalidationMask::default());
        // Camera motion alone must never invalidate a page.
        assert!(!camera_move_invalidates_pages());
    }
}
