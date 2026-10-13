//! CPU material-VT demand aggregation.
//!
//! The visibility renderer currently exposes an instance/primitive identity,
//! while the material-VT producer consumes [`TexturePageKey`]s.  This module is
//! the typed seam between those two contracts.  Callers provide the material's
//! real immutable source snapshot and an interpolated visibility sample with
//! its UV footprint; this module performs the same LOD/page expansion as the
//! material sampler and returns deterministic streaming demands.
//!
//! No page is invented when an upstream value is absent.  An empty sample
//! iterator produces an empty result, and malformed source/sample identities
//! are rejected explicitly.

use std::collections::BTreeMap;

use bevy_ecs::prelude::Resource;

use prism_render_architecture::texture_streaming::{PageDemand, TexturePageKey, TextureSemantic};
use prism_render_material::MaterialVtSourceSnapshot;
use prism_render_visibility::texture_lod;

/// Material-side identity and source data for one virtualized texture channel.
///
/// The source is an immutable snapshot captured from the main-world image
/// path.  Keeping the snapshot in this input makes the demand path incapable
/// of producing a page key for a texture whose source metadata is unavailable.
#[derive(Clone, Copy, Debug)]
pub struct MaterialVtDemandMaterial<'a> {
    /// Stable material row identity supplied by the material registry.
    pub material_id: u32,
    /// Stable virtual-texture identity. Must match `source.descriptor()`.
    pub texture_id: u32,
    /// Streaming semantic used by the scheduler's fixed-point priority.
    pub semantic: TextureSemantic,
    /// Immutable source snapshot containing dimensions and page layout.
    pub source: &'a MaterialVtSourceSnapshot,
    /// Array layer/cube face selected by the material sample.
    pub layer: u16,
    /// Physical bytes occupied by one resident material-VT tile.
    ///
    /// This is deliberately supplied by the physical pool configuration; the
    /// source snapshot only knows source texel bytes, not atlas tile padding.
    pub page_byte_cost: u64,
}

/// One real visibility/material sample from which page demand can be derived.
///
/// `uv` is the interpolated UV at the visible face sample. `uv_ddx` and
/// `uv_ddy` are screen-space UV derivatives, such as those reconstructed from
/// a visibility-buffer triangle. `object_id`, `face_id`, and `material_id`
/// preserve the upstream provenance and allow the seam to reject accidental
/// cross-material input; page aggregation itself is keyed by the shared page
/// identity because that is the streaming ABI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialVtVisibilitySample {
    /// Stable object/instance identity from visibility.
    pub object_id: u32,
    /// Triangle/face identity from visibility.
    pub face_id: u32,
    /// Material identity associated with this visible face.
    pub material_id: u32,
    /// Interpolated texture coordinate at the visible sample.
    pub uv: [f32; 2],
    /// Screen-space derivative of `uv` in the X direction.
    pub uv_ddx: [f32; 2],
    /// Screen-space derivative of `uv` in the Y direction.
    pub uv_ddy: [f32; 2],
    /// Fixed-point projected importance. Values above the streaming range are
    /// saturated by [`PageDemand`].
    pub screen_importance: u16,
}

/// Why a typed material/visibility sample could not produce demand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialVtDemandError {
    /// The material input's texture identity disagrees with its source.
    TextureIdMismatch { expected: u32, actual: u32 },
    /// The requested array layer/face does not exist in the source.
    LayerOutOfRange { layer: u16, layers: u16 },
    /// A zero page cost cannot be scheduled by the streaming budget.
    ZeroPageByteCost,
    /// The visibility sample belongs to another material row.
    MaterialIdMismatch { expected: u32, actual: u32 },
    /// UV or derivative input contains NaN or infinity.
    NonFiniteUv,
    /// The source's page key could not represent the derived request.
    InvalidPage,
    /// One physical page was reported with incompatible channel semantics.
    ConflictingSemantic {
        key: TexturePageKey,
        first: TextureSemantic,
        second: TextureSemantic,
    },
}

#[derive(Clone, Copy, Debug)]
struct DemandCandidate {
    semantic: TextureSemantic,
    screen_importance: u16,
}

/// Aggregates visibility samples into deterministic material-VT page demand.
///
/// Each sample expands into the two pages needed by the material sampler's
/// trilinear lookup.  Duplicate pages are merged by maximum screen
/// importance.  The result is sorted by [`TexturePageKey`] through a
/// `BTreeMap`, so input traversal order cannot alter the output.  The
/// residency callback is consulted only after this key-sorted merge; it must
/// return the finest resident mip for the exact key, or `None` when absent.
pub fn aggregate_material_vt_demands(
    material: &MaterialVtDemandMaterial<'_>,
    samples: impl IntoIterator<Item = MaterialVtVisibilitySample>,
    mut resident_mip: impl FnMut(TexturePageKey) -> Option<u8>,
    frame: u64,
) -> Result<Vec<PageDemand>, MaterialVtDemandError> {
    let descriptor = material.source.descriptor();
    if material.texture_id != descriptor.texture_id() {
        return Err(MaterialVtDemandError::TextureIdMismatch {
            expected: descriptor.texture_id(),
            actual: material.texture_id,
        });
    }
    if material.layer >= descriptor.layers() {
        return Err(MaterialVtDemandError::LayerOutOfRange {
            layer: material.layer,
            layers: descriptor.layers(),
        });
    }
    if material.page_byte_cost == 0 {
        return Err(MaterialVtDemandError::ZeroPageByteCost);
    }

    let virtual_texture = material.source.virtual_texture();
    let mut merged = BTreeMap::<TexturePageKey, DemandCandidate>::new();
    for sample in samples {
        if sample.material_id != material.material_id {
            return Err(MaterialVtDemandError::MaterialIdMismatch {
                expected: material.material_id,
                actual: sample.material_id,
            });
        }
        if !sample
            .uv
            .into_iter()
            .chain(sample.uv_ddx)
            .chain(sample.uv_ddy)
            .all(f32::is_finite)
        {
            return Err(MaterialVtDemandError::NonFiniteUv);
        }

        let lod = texture_lod(
            sample.uv_ddx,
            sample.uv_ddy,
            descriptor.width(),
            descriptor.height(),
        );
        let pages = virtual_texture.residency(sample.uv, lod);
        for request in pages {
            let key = material
                .source
                .key_for_request(material.layer, request)
                .map_err(|_| MaterialVtDemandError::InvalidPage)?;
            match merged.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(DemandCandidate {
                        semantic: material.semantic,
                        screen_importance: sample.screen_importance,
                    });
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let candidate = entry.get_mut();
                    if candidate.semantic != material.semantic {
                        return Err(MaterialVtDemandError::ConflictingSemantic {
                            key,
                            first: candidate.semantic,
                            second: material.semantic,
                        });
                    }
                    candidate.screen_importance =
                        candidate.screen_importance.max(sample.screen_importance);
                }
            }
        }
    }

    Ok(merged
        .into_iter()
        .map(|(key, candidate)| PageDemand {
            key,
            semantic: candidate.semantic,
            desired_mip: key.mip,
            resident_mip: resident_mip(key),
            screen_importance: candidate.screen_importance,
            byte_cost: material.page_byte_cost,
            frame,
        })
        .collect())
}

/// Render-world input buffer for real material-VT visibility feedback.
///
/// The buffer is intentionally a small, typed hand-off rather than an
/// implicit "everything is visible" policy.  A visibility/readback producer
/// replaces it once per frame; the material-VT producer consumes the exact
/// page keys and priorities on its next prepare pass.  An empty replacement is
/// meaningful: it says that this frame had no VT samples and allows the
/// streamer to apply its retention/eviction policy.
#[derive(Resource, Clone, Debug, Default)]
pub struct MaterialVtDemandRegistry {
    demands: Vec<PageDemand>,
    revision: u64,
    has_feedback: bool,
}

impl MaterialVtDemandRegistry {
    /// Replaces the current frame's validated demands.
    pub fn replace(
        &mut self,
        demands: impl IntoIterator<Item = PageDemand>,
    ) -> Result<(), MaterialVtDemandError> {
        let mut merged = BTreeMap::<TexturePageKey, PageDemand>::new();
        for demand in demands {
            match merged.entry(demand.key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(demand);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let current = entry.get_mut();
                    if current.semantic != demand.semantic {
                        return Err(MaterialVtDemandError::ConflictingSemantic {
                            key: demand.key,
                            first: current.semantic,
                            second: demand.semantic,
                        });
                    }
                    current.desired_mip = current.desired_mip.min(demand.desired_mip);
                    current.resident_mip = match (current.resident_mip, demand.resident_mip) {
                        (None, _) | (_, None) => None,
                        (Some(first), Some(second)) => Some(first.min(second)),
                    };
                    current.screen_importance =
                        current.screen_importance.max(demand.screen_importance);
                    current.byte_cost = current.byte_cost.max(demand.byte_cost);
                    current.frame = current.frame.max(demand.frame);
                }
            }
        }
        self.demands = merged.into_values().collect();
        self.revision = self.revision.saturating_add(1);
        self.has_feedback = true;
        Ok(())
    }

    /// Clears feedback and returns to the bounded deterministic bootstrap
    /// policy. This is for startup/teardown; an actual empty frame should use
    /// [`replace`](Self::replace) with an empty iterator.
    pub fn clear(&mut self) {
        self.demands.clear();
        self.revision = self.revision.saturating_add(1);
        self.has_feedback = false;
    }

    /// Current demand set, sorted by the shared page key.
    #[must_use]
    pub fn demands(&self) -> &[PageDemand] {
        &self.demands
    }

    /// Monotonic change marker consumed by the producer.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Whether the current value came from a real feedback frame.
    #[must_use]
    pub const fn has_feedback(&self) -> bool {
        self.has_feedback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::texture_streaming::TextureSemantic;
    use prism_render_material::{
        MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
    };

    fn source() -> MaterialVtSourceSnapshot {
        let descriptor =
            MaterialVtSourceDescriptor::new(7, 256, 256, 1, 128, 9, MaterialVtFormat::Rgba8Unorm)
                .expect("valid descriptor");
        let mut bytes = Vec::new();
        for mip in 0..9_u32 {
            let width = (256_u32 >> mip).max(1);
            bytes.extend(std::iter::repeat_n(0_u8, (width * width * 4) as usize));
        }
        MaterialVtSourceSnapshot::new(descriptor, bytes).expect("valid source")
    }

    fn material(source: &MaterialVtSourceSnapshot) -> MaterialVtDemandMaterial<'_> {
        MaterialVtDemandMaterial {
            material_id: 11,
            texture_id: 7,
            semantic: TextureSemantic::Color,
            source,
            layer: 0,
            page_byte_cost: 70_000,
        }
    }

    fn sample(material_id: u32, uv: [f32; 2], importance: u16) -> MaterialVtVisibilitySample {
        MaterialVtVisibilitySample {
            object_id: 3,
            face_id: 9,
            material_id,
            uv,
            uv_ddx: [0.25, 0.0],
            uv_ddy: [0.0, 0.25],
            screen_importance: importance,
        }
    }

    #[test]
    fn empty_visibility_is_an_empty_real_feedback_result() {
        let source = source();
        let result = aggregate_material_vt_demands(&material(&source), [], |_| None, 4)
            .expect("empty input is valid");
        assert!(result.is_empty());
    }

    #[test]
    fn expands_uv_footprint_to_sorted_trilinear_pages_and_merges_importance() {
        let source = source();
        let input = material(&source);
        let a = sample(input.material_id, [0.25, 0.25], 200);
        let mut b = sample(input.material_id, [0.25, 0.25], 800);
        b.object_id = 99;
        b.face_id = 100;
        let result = aggregate_material_vt_demands(&input, [b, a], |_| None, 12).unwrap();

        assert_eq!(result.len(), 2, "fine and coarse trilinear pages");
        assert!(result.windows(2).all(|pair| pair[0].key < pair[1].key));
        assert!(result.iter().all(|demand| demand.key.texture == 7));
        assert!(result.iter().all(|demand| demand.screen_importance == 800));
        assert!(result.iter().all(|demand| demand.resident_mip.is_none()));
        assert!(result.iter().all(|demand| demand.frame == 12));
    }

    #[test]
    fn rejects_cross_material_and_non_finite_visibility_input() {
        let source = source();
        let input = material(&source);
        assert_eq!(
            aggregate_material_vt_demands(&input, [sample(12, [0.0, 0.0], 1)], |_| None, 1),
            Err(MaterialVtDemandError::MaterialIdMismatch {
                expected: 11,
                actual: 12,
            })
        );

        let mut invalid = sample(11, [0.0, 0.0], 1);
        invalid.uv_ddx[0] = f32::NAN;
        assert_eq!(
            aggregate_material_vt_demands(&input, [invalid], |_| None, 1),
            Err(MaterialVtDemandError::NonFiniteUv)
        );
    }

    #[test]
    fn resident_lookup_is_keyed_by_exact_demand_page() {
        let source = source();
        let input = material(&source);
        let result = aggregate_material_vt_demands(
            &input,
            [sample(input.material_id, [0.25, 0.25], 100)],
            |key| Some(key.mip.saturating_add(1)),
            3,
        )
        .unwrap();
        assert!(result
            .iter()
            .all(|demand| { demand.resident_mip == Some(demand.key.mip.saturating_add(1)) }));
    }

    #[test]
    fn registry_replaces_and_deduplicates_real_demands() {
        let key = TexturePageKey {
            texture: 7,
            mip: 0,
            layer: 0,
            x: 1,
            y: 2,
        };
        let demand = PageDemand {
            key,
            semantic: TextureSemantic::Color,
            desired_mip: 0,
            resident_mip: None,
            screen_importance: 100,
            byte_cost: 1,
            frame: 1,
        };
        let mut registry = MaterialVtDemandRegistry::default();
        assert!(!registry.has_feedback());
        registry.replace([demand, demand]).unwrap();
        assert!(registry.has_feedback());
        assert_eq!(registry.demands(), &[demand]);
        let revision = registry.revision();
        registry.replace([]).unwrap();
        assert!(registry.has_feedback());
        assert!(registry.demands().is_empty());
        assert!(registry.revision() > revision);
        registry.clear();
        assert!(!registry.has_feedback());
    }

    #[test]
    fn registry_merges_duplicate_pages_deterministically_and_rejects_semantic_conflicts() {
        let key = TexturePageKey {
            texture: 7,
            mip: 0,
            layer: 0,
            x: 1,
            y: 2,
        };
        let mut registry = MaterialVtDemandRegistry::default();
        let first = PageDemand {
            key,
            semantic: TextureSemantic::Color,
            desired_mip: 3,
            resident_mip: Some(2),
            screen_importance: 100,
            byte_cost: 4,
            frame: 8,
        };
        let second = PageDemand {
            desired_mip: 1,
            resident_mip: None,
            screen_importance: 900,
            byte_cost: 8,
            frame: 9,
            ..first
        };
        registry.replace([second, first]).unwrap();
        assert_eq!(
            registry.demands(),
            &[PageDemand {
                desired_mip: 1,
                resident_mip: None,
                screen_importance: 900,
                byte_cost: 8,
                frame: 9,
                ..first
            }]
        );

        let conflicting = PageDemand {
            semantic: TextureSemantic::Normal,
            ..first
        };
        assert_eq!(
            registry.replace([first, conflicting]),
            Err(MaterialVtDemandError::ConflictingSemantic {
                key,
                first: TextureSemantic::Color,
                second: TextureSemantic::Normal,
            })
        );
    }
}
