//! Receiver-driven page-request generation, de-duplication and filter-footprint
//! expansion.
//!
//! Before any shadow depth is rendered, the virtual shadow map has to know
//! *which* pages this frame actually needs.  That set is driven by the shadow
//! *receivers* — the visible surfaces that will sample the shadow map.  Each
//! receiver, projected into the light's clipmap plane, selects a clip level
//! from its view distance and lands on one page; the soft-shadow filter kernel
//! (PCF/PCSS) then widens that footprint, because a sample near a page edge
//! reads texels from the neighbouring pages too.
//!
//! This module turns a slice of receivers into the de-duplicated, deterministic
//! set of [`ShadowPageKey`]s the allocator must make resident.  Generating the
//! request set on the CPU mirrors the GPU page-request pass that marks pages in
//! the feedback buffer, so the two stay in step.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use bevy_math::{ops, IVec2, Vec2};

use crate::shadow::virtual_sm::clipmap::ClipmapConfig;
use crate::shadow::virtual_sm::page_table::{key_from_order, page_order};
use prism_render_architecture::virtual_shadow::ShadowPageKey;

/// One shadow receiver, already projected into the light's clipmap plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Receiver {
    /// Receiver position in the light-space clipmap plane (world units).
    pub light_space_xy: Vec2,
    /// Positive view-space distance from the camera, which selects the clip
    /// level (finer levels for nearer receivers).
    pub view_distance: f32,
    /// Soft-shadow filter kernel half-width in shadow texels; widens the page
    /// footprint so edge samples still find resident neighbours.
    pub filter_radius_texels: f32,
}

/// The page-request set produced for one light this frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PageRequestSet {
    /// Unique requested pages in deterministic key order.
    pub keys: Vec<ShadowPageKey>,
    /// Total footprint cells visited before de-duplication (a churn metric).
    pub raw_count: usize,
}

/// Number of extra page rings a filter kernel of `filter_radius_texels` adds
/// around a receiver's page, given `page_size` texels per page.
///
/// A kernel of radius `R` texels can spill up to `R` texels past the sampled
/// texel; expressed in pages that is `ceil(R / page_size)` rings on every side
/// (`0` for a point sample), a conservative bound that never under-requests.
pub fn filter_page_radius(filter_radius_texels: f32, page_size: u16) -> i32 {
    let radius = filter_radius_texels.max(0.0);
    if radius <= 0.0 {
        return 0;
    }
    let page = f32::from(page_size.max(1));
    ops::ceil(radius / page) as i32
}

/// Generates the de-duplicated page-request set for `light` from `receivers`.
///
/// Each receiver selects a clip level from its view distance, maps to its
/// absolute world page, and expands into a square footprint sized by its filter
/// radius.  Footprint cells outside the representable key range are skipped.
pub fn generate_page_requests(
    clipmap: &ClipmapConfig,
    light: u32,
    receivers: &[Receiver],
) -> PageRequestSet {
    let mut unique = BTreeSet::new();
    let mut raw_count = 0usize;
    for receiver in receivers {
        let level = clipmap.select_level(receiver.view_distance);
        let center = clipmap.world_page_coords(level, receiver.light_space_xy);
        let rings = filter_page_radius(receiver.filter_radius_texels, clipmap.page_size);
        for dy in -rings..=rings {
            for dx in -rings..=rings {
                raw_count += 1;
                let page = center + IVec2::new(dx, dy);
                if let Some(key) = clipmap.page_key(light, level, page) {
                    unique.insert(page_order(&key));
                }
            }
        }
    }
    let keys = unique.into_iter().map(key_from_order).collect();
    PageRequestSet { keys, raw_count }
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

    /// A point sample (zero filter radius) requests exactly one page.
    #[test]
    fn point_sample_requests_a_single_page() {
        let c = config();
        let set = generate_page_requests(
            &c,
            0,
            &[Receiver {
                light_space_xy: Vec2::new(3.0, 4.0),
                view_distance: 1.0,
                filter_radius_texels: 0.0,
            }],
        );
        assert_eq!(set.keys.len(), 1);
        assert_eq!(set.raw_count, 1);
    }

    /// The filter radius maps to page rings: `page_size` texels -> one ring ->
    /// a 3x3 footprint.
    #[test]
    fn filter_radius_expands_to_page_rings() {
        assert_eq!(filter_page_radius(0.0, 128), 0);
        assert_eq!(filter_page_radius(1.0, 128), 1); // any spill -> at least one ring
        assert_eq!(filter_page_radius(128.0, 128), 1);
        assert_eq!(filter_page_radius(200.0, 128), 2);

        let c = config();
        let set = generate_page_requests(
            &c,
            0,
            &[Receiver {
                light_space_xy: Vec2::new(0.0, 0.0),
                view_distance: 1.0,
                filter_radius_texels: 130.0, // two rings
            }],
        );
        // Two rings -> 5x5 footprint.
        assert_eq!(set.keys.len(), 25);
    }

    /// Overlapping receivers de-duplicate: the unique set is smaller than the
    /// raw footprint count.
    #[test]
    fn overlapping_receivers_are_deduplicated() {
        let c = config();
        let receiver = Receiver {
            light_space_xy: Vec2::new(0.0, 0.0),
            view_distance: 1.0,
            filter_radius_texels: 128.0, // one ring, 3x3
        };
        let set = generate_page_requests(&c, 7, &[receiver, receiver, receiver]);
        assert_eq!(set.raw_count, 27); // 3 receivers * 9 cells
        assert_eq!(set.keys.len(), 9); // all identical footprints collapse
                                       // Every key belongs to the requested light and is sorted/unique.
        assert!(set.keys.iter().all(|k| k.light == 7));
        let mut sorted = set.keys.clone();
        sorted.sort_by_key(page_order);
        assert_eq!(set.keys, sorted);
    }

    /// Receivers at different view distances land on different clip levels.
    #[test]
    fn distance_splits_receivers_across_levels() {
        let c = config();
        let near = Receiver {
            light_space_xy: Vec2::new(0.0, 0.0),
            view_distance: 1.0, // level 0
            filter_radius_texels: 0.0,
        };
        let far = Receiver {
            light_space_xy: Vec2::new(0.0, 0.0),
            view_distance: 1000.0, // coarsest level
            filter_radius_texels: 0.0,
        };
        let set = generate_page_requests(&c, 0, &[near, far]);
        let levels: BTreeSet<u16> = set.keys.iter().map(|k| k.level).collect();
        assert!(levels.contains(&0));
        assert!(levels.contains(&(c.level_count() - 1)));
        assert_eq!(levels.len(), 2);
    }
}
