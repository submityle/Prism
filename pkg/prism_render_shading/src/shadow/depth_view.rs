//! Per-layer draw plan for the shadow-map depth rasterization pass.
//!
//! [`allocate_shadow_atlas`](crate::shadow::allocate_shadow_atlas) decides
//! *which* atlas layers each admitted light owns; this module turns that layout
//! into the concrete list of depth-pass draws that *fill* those layers.  Each
//! [`ShadowDepthDraw`] carries the global atlas `layer` to render into and the
//! [`ShadowDepthView`] the GPU pass binds as its per-view uniform: a
//! world -> light-clip matrix plus, for point lights, the light position and
//! inverse range used to store range-normalized distance.
//!
//! The produced views are the CPU twin of the `ShadowDepthView` uniform in
//! `shadow_depth.wesl`; the field order and packing there mirror
//! [`ShadowDepthView`] here so an uploaded buffer and this reference stay
//! byte-consistent.  Directional cascade matrices come from
//! [`compute_cascade_matrices`](crate::shadow::compute_cascade_matrices); point
//! cube-face matrices come from
//! [`cube_face_view_projections`](crate::shadow::cube_face_view_projections);
//! spot matrices are supplied by the caller.

use alloc::vec::Vec;

use crate::shadow::atlas::{AtlasAllocation, ShadowKind};
use crate::shadow::cascade::MAX_CASCADE_COUNT;
use crate::shadow::csm::CascadeMatrix;
use crate::shadow::math::Mat4;
use crate::shadow::point::cube_face_view_projections;

/// How the depth pass stores its `.r` channel for a given shadow view, matching
/// the `SHADOW_DEPTH_MODE_*` constants in `shadow_depth.wesl`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowDepthMode {
    /// Store the wgpu NDC depth `z in [0, 1]` (directional cascades and spot
    /// cones, which are sampled through their projection matrix).
    Ndc,
    /// Store the light-range-normalized linear distance `dist / range in
    /// [0, 1]` (point-light cube faces, sampled by direction).
    Distance,
}

impl ShadowDepthMode {
    /// The `u32` discriminant uploaded in `ShadowDepthView.params.x`.
    ///
    /// `0` = [`ShadowDepthMode::Ndc`], `1` = [`ShadowDepthMode::Distance`],
    /// matching `SHADOW_DEPTH_MODE_NDC` / `SHADOW_DEPTH_MODE_DISTANCE`.
    pub fn as_u32(self) -> u32 {
        match self {
            ShadowDepthMode::Ndc => 0,
            ShadowDepthMode::Distance => 1,
        }
    }
}

/// The per-view constants the depth pass binds for one atlas layer, mirroring
/// the `ShadowDepthView` uniform in `shadow_depth.wesl`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowDepthView {
    /// Column-major world -> light-clip matrix (wgpu clip: `z` in `[0, 1]`).
    pub view_projection: Mat4,
    /// `xyz`: light world position; `w`: `1 / range`.  Only read in
    /// [`ShadowDepthMode::Distance`]; zeroed otherwise.
    pub light_position: [f32; 4],
    /// Whether this layer stores NDC depth or normalized distance.
    pub mode: ShadowDepthMode,
}

/// The per-light matrices the plan needs to fill a slot's layers, supplied by
/// the caller for each admitted `light_id`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "This enum is a short-lived, by-value carrier for one light's \
depth-view geometry; only a handful exist per frame, so boxing the \
directional cascade array would add pointless heap allocations."
)]
pub enum ShadowViewGeometry {
    /// Cascaded directional light: one matrix per cascade (inactive cascade
    /// slots are padded with [`CascadeMatrix::identity`]).
    Directional {
        /// The stabilized cascade matrices, cascade `0` nearest the camera.
        cascades: [CascadeMatrix; MAX_CASCADE_COUNT],
    },
    /// Omnidirectional point light: cube-face matrices are derived internally
    /// from the light position and range so they stay matched to the sampler.
    Point {
        /// World-space position of the light.
        position: [f32; 3],
        /// Near clip of each cube-face frustum.
        near: f32,
        /// Far range; also normalizes the stored distance (`w = 1 / far`).
        far: f32,
    },
    /// Spot light: a single caller-supplied world -> light-clip matrix.
    Spot {
        /// Column-major world -> light-clip matrix for the spot cone.
        view_projection: Mat4,
    },
}

/// One depth-pass draw: render the scene into atlas `layer` using `view`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowDepthDraw {
    /// The light this layer belongs to.
    pub light_id: u32,
    /// Global atlas array layer to rasterize into.
    pub layer: u32,
    /// The per-view constants bound for this layer.
    pub view: ShadowDepthView,
}

/// Expands an [`AtlasAllocation`] into the flat list of depth-pass draws that
/// fill every admitted layer.
///
/// `geometry_for` supplies the matrices for each admitted `light_id`.  A slot
/// whose supplied geometry does not match its [`ShadowKind`] is skipped (its
/// layers are left unfilled) so a caller bug degrades to a shadowless light
/// rather than a mismatched projection.  Draws are emitted in slot order
/// (descending importance), each slot's layers in ascending sub-index, so the
/// output `layer` values are the same globals the resolve pass samples.
pub fn plan_shadow_depth_draws(
    allocation: &AtlasAllocation,
    mut geometry_for: impl FnMut(u32) -> ShadowViewGeometry,
) -> Vec<ShadowDepthDraw> {
    let mut draws = Vec::new();
    for slot in &allocation.slots {
        let geometry = geometry_for(slot.light_id);
        match (slot.kind, geometry) {
            (ShadowKind::Directional { .. }, ShadowViewGeometry::Directional { cascades }) => {
                for sub in 0..slot.layer_count {
                    let cascade = cascades[(sub as usize).min(MAX_CASCADE_COUNT - 1)];
                    draws.push(ShadowDepthDraw {
                        light_id: slot.light_id,
                        layer: slot.layer(sub),
                        view: ShadowDepthView {
                            view_projection: cascade.view_projection,
                            light_position: [0.0, 0.0, 0.0, 0.0],
                            mode: ShadowDepthMode::Ndc,
                        },
                    });
                }
            }
            (ShadowKind::Point, ShadowViewGeometry::Point { position, near, far }) => {
                let faces = cube_face_view_projections(position, near, far);
                let inv_range = far.max(1.0e-4).recip();
                for (sub, view_projection) in faces.iter().enumerate() {
                    draws.push(ShadowDepthDraw {
                        light_id: slot.light_id,
                        layer: slot.layer(sub as u32),
                        view: ShadowDepthView {
                            view_projection: *view_projection,
                            light_position: [position[0], position[1], position[2], inv_range],
                            mode: ShadowDepthMode::Distance,
                        },
                    });
                }
            }
            (ShadowKind::Spot, ShadowViewGeometry::Spot { view_projection }) => {
                draws.push(ShadowDepthDraw {
                    light_id: slot.light_id,
                    layer: slot.base_layer,
                    view: ShadowDepthView {
                        view_projection,
                        light_position: [0.0, 0.0, 0.0, 0.0],
                        mode: ShadowDepthMode::Ndc,
                    },
                });
            }
            // Geometry / kind mismatch: leave the light shadowless this frame.
            _ => {}
        }
    }
    draws
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shadow::atlas::{allocate_shadow_atlas, AtlasConfig, ShadowRequest, POINT_LAYER_COUNT};

    fn directional_geometry() -> ShadowViewGeometry {
        let mut cascades = [CascadeMatrix::identity(); MAX_CASCADE_COUNT];
        // Tag each cascade's matrix so the plan's ordering is observable.
        for (index, cascade) in cascades.iter_mut().enumerate() {
            cascade.view_projection[12] = index as f32;
        }
        ShadowViewGeometry::Directional { cascades }
    }

    /// A three-cascade directional slot yields three NDC draws whose matrices
    /// follow the cascade order and whose layers are the slot's globals.
    #[test]
    fn directional_slot_expands_per_cascade() {
        let alloc = allocate_shadow_atlas(
            AtlasConfig::new(8, 1024),
            &[ShadowRequest {
                light_id: 7,
                kind: ShadowKind::Directional { cascades: 3 },
                importance: 1.0,
            }],
        );
        let draws = plan_shadow_depth_draws(&alloc, |_| directional_geometry());
        assert_eq!(draws.len(), 3);
        for (sub, draw) in draws.iter().enumerate() {
            assert_eq!(draw.light_id, 7);
            assert_eq!(draw.layer, sub as u32);
            assert_eq!(draw.view.mode, ShadowDepthMode::Ndc);
            assert_eq!(draw.view.view_projection[12], sub as f32);
            assert_eq!(draw.view.light_position, [0.0, 0.0, 0.0, 0.0]);
        }
    }

    /// A point slot yields six distance draws carrying the light position and
    /// inverse range, one per cube face in `+X, -X, +Y, -Y, +Z, -Z` order.
    #[test]
    fn point_slot_expands_per_cube_face() {
        let alloc = allocate_shadow_atlas(
            AtlasConfig::new(8, 512),
            &[ShadowRequest {
                light_id: 3,
                kind: ShadowKind::Point,
                importance: 1.0,
            }],
        );
        let position = [2.0, -1.0, 4.0];
        let far = 50.0;
        let draws = plan_shadow_depth_draws(&alloc, |_| ShadowViewGeometry::Point {
            position,
            near: 0.1,
            far,
        });
        assert_eq!(draws.len(), POINT_LAYER_COUNT as usize);
        let faces = cube_face_view_projections(position, 0.1, far);
        for (sub, draw) in draws.iter().enumerate() {
            assert_eq!(draw.light_id, 3);
            assert_eq!(draw.layer, sub as u32);
            assert_eq!(draw.view.mode, ShadowDepthMode::Distance);
            assert_eq!(draw.view.view_projection, faces[sub]);
            assert_eq!(
                draw.view.light_position,
                [position[0], position[1], position[2], (1.0_f32 / far)]
            );
        }
    }

    /// A spot slot yields a single NDC draw at its base layer with the supplied
    /// matrix.
    #[test]
    fn spot_slot_expands_to_single_layer() {
        let alloc = allocate_shadow_atlas(
            AtlasConfig::new(8, 256),
            &[ShadowRequest {
                light_id: 11,
                kind: ShadowKind::Spot,
                importance: 1.0,
            }],
        );
        let mut view_projection = CascadeMatrix::identity().view_projection;
        view_projection[13] = 9.0;
        let draws =
            plan_shadow_depth_draws(&alloc, |_| ShadowViewGeometry::Spot { view_projection });
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].light_id, 11);
        assert_eq!(draws[0].layer, alloc.slots[0].base_layer);
        assert_eq!(draws[0].view.mode, ShadowDepthMode::Ndc);
        assert_eq!(draws[0].view.view_projection[13], 9.0);
    }

    /// Layers stay global across several admitted lights: the second slot's
    /// draws continue numbering where the first left off.
    #[test]
    fn multiple_slots_keep_global_layers() {
        let alloc = allocate_shadow_atlas(
            AtlasConfig::new(16, 512),
            &[
                ShadowRequest {
                    light_id: 1,
                    kind: ShadowKind::Directional { cascades: 4 },
                    importance: 2.0,
                },
                ShadowRequest {
                    light_id: 2,
                    kind: ShadowKind::Point,
                    importance: 1.0,
                },
            ],
        );
        let draws = plan_shadow_depth_draws(&alloc, |light_id| match light_id {
            1 => directional_geometry(),
            _ => ShadowViewGeometry::Point {
                position: [0.0, 0.0, 0.0],
                near: 0.1,
                far: 20.0,
            },
        });
        assert_eq!(draws.len(), 4 + POINT_LAYER_COUNT as usize);
        // Directional occupies layers 0..4, the point light 4..10, contiguous.
        let layers: Vec<u32> = draws.iter().map(|draw| draw.layer).collect();
        assert_eq!(layers, (0..10).collect::<Vec<_>>());
    }

    /// A slot whose supplied geometry does not match its kind is skipped rather
    /// than emitting a mismatched projection.
    #[test]
    fn mismatched_geometry_is_skipped() {
        let alloc = allocate_shadow_atlas(
            AtlasConfig::new(8, 256),
            &[ShadowRequest {
                light_id: 5,
                kind: ShadowKind::Point,
                importance: 1.0,
            }],
        );
        // Hand the point slot directional geometry: no draws should be emitted.
        let draws = plan_shadow_depth_draws(&alloc, |_| directional_geometry());
        assert!(draws.is_empty());
    }
}
