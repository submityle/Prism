//! Shadow-atlas layer budgeting and allocation.
//!
//! Every shadow-casting light needs one or more layers of the shared shadow
//! atlas (`shadow.wesl` binds it as a single `texture_2d_array<f32>`, addressed
//! by `layer`):
//!
//! * a **directional** light needs one layer per active cascade,
//! * a **point** light needs six layers (one per cube-map face),
//! * a **spot** light needs a single projected layer.
//!
//! The atlas has a fixed layer budget, so when the scene asks for more shadow
//! layers than fit, some lights must go shadowless this frame.  Following the
//! importance-sorted budgeting Unreal uses for its shadow atlas, this module
//! admits lights in descending importance order and drops the ones that no
//! longer fit.  Admission is *best effort*: a light whose block is larger than
//! the remaining budget is dropped, but smaller lower-importance lights can
//! still fill the tail of the atlas.
//!
//! The output is a deterministic assignment of contiguous layer ranges that the
//! GPU depth pass rasterizes into and the resolve pass samples from, keeping the
//! `layer` indices consistent with the CPU `ShadowDepthSampler` twin.

use crate::shadow::cascade::MAX_CASCADE_COUNT;
use alloc::vec::Vec;

/// The shadow-map topology a light needs, which fixes how many contiguous atlas
/// layers it consumes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShadowKind {
    /// Cascaded directional shadow: one layer per active cascade.
    Directional {
        /// Number of active cascades (clamped to `[1, MAX_CASCADE_COUNT]`).
        cascades: u32,
    },
    /// Omnidirectional point shadow: six cube-map faces.
    Point,
    /// Spot light: a single projected shadow map.
    Spot,
}

/// Number of atlas layers a point light's cube map occupies.
pub const POINT_LAYER_COUNT: u32 = 6;

impl ShadowKind {
    /// The number of contiguous atlas layers this shadow needs.
    pub fn layer_count(self) -> u32 {
        match self {
            ShadowKind::Directional { cascades } => cascades.clamp(1, MAX_CASCADE_COUNT as u32),
            ShadowKind::Point => POINT_LAYER_COUNT,
            ShadowKind::Spot => 1,
        }
    }
}

/// One light's request for atlas space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowRequest {
    /// Stable light identifier (e.g. index into the light buffer).
    pub light_id: u32,
    /// The shadow-map topology this light needs.
    pub kind: ShadowKind,
    /// Relative importance; higher-importance lights are admitted first when the
    /// atlas cannot hold every request.  Ties break by ascending `light_id` so
    /// the allocation is fully deterministic.
    pub importance: f32,
}

/// Fixed atlas budget: a square shadow map of `resolution` texels per side,
/// replicated across at most `max_layers` array layers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasConfig {
    /// Total number of array layers available in the atlas texture.
    pub max_layers: u32,
    /// Edge resolution (in texels) of each square atlas layer.
    pub resolution: u32,
}

impl AtlasConfig {
    /// Builds a config, clamping `resolution` to at least one texel.
    pub fn new(max_layers: u32, resolution: u32) -> Self {
        Self {
            max_layers,
            resolution: resolution.max(1),
        }
    }
}

/// The contiguous layer range assigned to one admitted light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasSlot {
    /// The light this slot belongs to.
    pub light_id: u32,
    /// The shadow topology, which determines `layer_count`.
    pub kind: ShadowKind,
    /// First atlas array layer of this light's block.
    pub base_layer: u32,
    /// Number of layers in this light's block.
    pub layer_count: u32,
}

impl AtlasSlot {
    /// The global atlas layer for sub-index `index` (a cascade for a directional
    /// light, or a cube face for a point light), clamped into the slot's range.
    pub fn layer(&self, index: u32) -> u32 {
        self.base_layer + index.min(self.layer_count.saturating_sub(1))
    }
}

/// The result of budgeting a frame's shadow requests against the atlas.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AtlasAllocation {
    /// Admitted lights and their contiguous layer ranges, in descending
    /// importance order (so `base_layer` is monotonically increasing).
    pub slots: Vec<AtlasSlot>,
    /// `light_id`s that did not fit the budget this frame, in the order they
    /// were considered (descending importance).
    pub dropped: Vec<u32>,
    /// The number of layers actually consumed by the admitted slots.
    pub used_layers: u32,
    /// The per-layer edge resolution carried through from the config.
    pub resolution: u32,
}

/// Budgets `requests` against `config`, returning the layer assignment.
///
/// Lights are considered in descending `importance` (ties by ascending
/// `light_id`).  Each light that fits the remaining budget is assigned the next
/// contiguous layer range; each that does not is dropped, and consideration
/// continues so smaller lower-importance lights may still fill the atlas tail.
pub fn allocate_shadow_atlas(config: AtlasConfig, requests: &[ShadowRequest]) -> AtlasAllocation {
    let resolution = config.resolution.max(1);
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by(|&a, &b| {
        let ra = &requests[a];
        let rb = &requests[b];
        rb.importance
            .total_cmp(&ra.importance)
            .then(ra.light_id.cmp(&rb.light_id))
    });

    let mut slots = Vec::new();
    let mut dropped = Vec::new();
    let mut next_layer = 0u32;
    for &index in &order {
        let request = requests[index];
        let count = request.kind.layer_count();
        let remaining = config.max_layers.saturating_sub(next_layer);
        if count == 0 || count > remaining {
            dropped.push(request.light_id);
            continue;
        }
        slots.push(AtlasSlot {
            light_id: request.light_id,
            kind: request.kind,
            base_layer: next_layer,
            layer_count: count,
        });
        next_layer += count;
    }

    AtlasAllocation {
        slots,
        dropped,
        used_layers: next_layer,
        resolution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_layer_counts_are_topology_sized() {
        assert_eq!(ShadowKind::Directional { cascades: 3 }.layer_count(), 3);
        assert_eq!(ShadowKind::Point.layer_count(), POINT_LAYER_COUNT);
        assert_eq!(ShadowKind::Spot.layer_count(), 1);
    }

    #[test]
    fn directional_cascades_clamp_into_range() {
        assert_eq!(ShadowKind::Directional { cascades: 0 }.layer_count(), 1);
        assert_eq!(
            ShadowKind::Directional { cascades: 99 }.layer_count(),
            MAX_CASCADE_COUNT as u32
        );
    }

    #[test]
    fn single_directional_gets_a_contiguous_block() {
        let config = AtlasConfig::new(16, 2048);
        let requests = [ShadowRequest {
            light_id: 7,
            kind: ShadowKind::Directional { cascades: 4 },
            importance: 1.0,
        }];
        let alloc = allocate_shadow_atlas(config, &requests);
        assert_eq!(alloc.slots.len(), 1);
        assert_eq!(alloc.slots[0].base_layer, 0);
        assert_eq!(alloc.slots[0].layer_count, 4);
        assert_eq!(alloc.slots[0].layer(2), 2);
        assert_eq!(alloc.slots[0].layer(9), 3); // clamped to last cascade
        assert_eq!(alloc.used_layers, 4);
        assert_eq!(alloc.resolution, 2048);
        assert!(alloc.dropped.is_empty());
    }

    #[test]
    fn point_light_takes_six_contiguous_faces_after_a_directional() {
        let config = AtlasConfig::new(16, 1024);
        let requests = [
            ShadowRequest {
                light_id: 1,
                kind: ShadowKind::Directional { cascades: 2 },
                importance: 2.0,
            },
            ShadowRequest {
                light_id: 2,
                kind: ShadowKind::Point,
                importance: 1.0,
            },
        ];
        let alloc = allocate_shadow_atlas(config, &requests);
        assert_eq!(alloc.slots[0].light_id, 1);
        assert_eq!(alloc.slots[0].base_layer, 0);
        assert_eq!(alloc.slots[1].light_id, 2);
        assert_eq!(alloc.slots[1].base_layer, 2);
        assert_eq!(alloc.slots[1].layer_count, 6);
        assert_eq!(alloc.used_layers, 8);
    }

    #[test]
    fn over_budget_drops_lowest_importance_first() {
        let config = AtlasConfig::new(6, 512);
        let requests = [
            ShadowRequest {
                light_id: 10,
                kind: ShadowKind::Point, // 6 layers, fills the whole budget
                importance: 5.0,
            },
            ShadowRequest {
                light_id: 20,
                kind: ShadowKind::Spot, // 1 layer, no room left
                importance: 1.0,
            },
        ];
        let alloc = allocate_shadow_atlas(config, &requests);
        assert_eq!(alloc.slots.len(), 1);
        assert_eq!(alloc.slots[0].light_id, 10);
        assert_eq!(alloc.dropped, [20]);
        assert_eq!(alloc.used_layers, 6);
    }

    #[test]
    fn admission_is_best_effort_across_a_too_big_light() {
        // Budget of 5 layers: the top-importance point light (6) cannot fit, but
        // the two lower-importance spots still should.
        let config = AtlasConfig::new(5, 256);
        let requests = [
            ShadowRequest {
                light_id: 1,
                kind: ShadowKind::Point,
                importance: 9.0,
            },
            ShadowRequest {
                light_id: 2,
                kind: ShadowKind::Spot,
                importance: 5.0,
            },
            ShadowRequest {
                light_id: 3,
                kind: ShadowKind::Directional { cascades: 3 },
                importance: 4.0,
            },
        ];
        let alloc = allocate_shadow_atlas(config, &requests);
        assert_eq!(alloc.dropped, [1]);
        assert_eq!(alloc.slots.len(), 2);
        assert_eq!(alloc.slots[0].light_id, 2);
        assert_eq!(alloc.slots[0].base_layer, 0);
        assert_eq!(alloc.slots[1].light_id, 3);
        assert_eq!(alloc.slots[1].base_layer, 1);
        assert_eq!(alloc.used_layers, 4);
    }

    #[test]
    fn equal_importance_breaks_ties_by_light_id() {
        let config = AtlasConfig::new(2, 128);
        let requests = [
            ShadowRequest {
                light_id: 42,
                kind: ShadowKind::Spot,
                importance: 1.0,
            },
            ShadowRequest {
                light_id: 7,
                kind: ShadowKind::Spot,
                importance: 1.0,
            },
        ];
        let alloc = allocate_shadow_atlas(config, &requests);
        // Lower light_id wins the earlier (base 0) slot on an importance tie.
        assert_eq!(alloc.slots[0].light_id, 7);
        assert_eq!(alloc.slots[0].base_layer, 0);
        assert_eq!(alloc.slots[1].light_id, 42);
        assert_eq!(alloc.slots[1].base_layer, 1);
    }

    #[test]
    fn empty_requests_and_zero_budget_are_well_defined() {
        let empty = allocate_shadow_atlas(AtlasConfig::new(8, 1024), &[]);
        assert!(empty.slots.is_empty());
        assert_eq!(empty.used_layers, 0);
        assert_eq!(empty.resolution, 1024);

        let starved = allocate_shadow_atlas(
            AtlasConfig::new(0, 1024),
            &[ShadowRequest {
                light_id: 1,
                kind: ShadowKind::Spot,
                importance: 1.0,
            }],
        );
        assert!(starved.slots.is_empty());
        assert_eq!(starved.dropped, [1]);
        assert_eq!(starved.used_layers, 0);
    }

    #[test]
    fn resolution_is_clamped_to_at_least_one() {
        let alloc = allocate_shadow_atlas(AtlasConfig::new(4, 0), &[]);
        assert_eq!(alloc.resolution, 1);
    }
}
