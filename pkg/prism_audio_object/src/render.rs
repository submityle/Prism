//! The unified object-audio render entry point.
//!
//! [`render_scene`] (and the lower-level [`render_objects`]) tie the crate
//! together into the end-to-end control-rate reduction described by the
//! design:
//!
//! 1. Snapshot the scene's current objects (callers advance the scene first).
//! 2. Ask the hardware [`crate::budget::ObjectBudget`] how many discrete
//!    clusters may be presented (`target_clusters`).
//! 3. Reduce the objects to that many representatives with the
//!    energy-preserving [`crate::clustering`] fallback.
//! 4. Fold / pan / encode those representatives into the requested
//!    [`OutputFormat`].
//!
//! When the budget is zero the renderer cannot present discrete objects, so the
//! clustering step yields nothing and every original object is folded directly
//! into the target instead (into the bed for [`OutputFormat::Bed`], or encoded
//! per object for the binaural and Ambisonic paths). In that case
//! [`RenderOutput::clusters`] is empty and the payload is built from the
//! unreduced objects.
//!
//! The outputs are per-source *gain* data (control rate), not a per-sample
//! mix: multiply each source's mono signal by its row/parameters and sum.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Composes design section 44.2's bed, objects, metadata, budget, clustering,
//! downmix, and Ambisonics stages into one entry point. Built on
//! [`crate::clustering`], [`crate::fold`], [`crate::pan`], and
//! [`crate::ambisonics`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::ambisonics;
use crate::bed::BedLayout;
use crate::budget::ObjectBudget;
use crate::clustering::{self, ClusteredObject};
use crate::fold::{self, BinauralDirection};
use crate::object::AudioObject;
use crate::scene::ObjectScene;

/// The delivery target the renderer should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum OutputFormat {
    /// Downmix sources onto the scene's channel bed (per-source gain rows).
    Bed,
    /// Reduce sources to binaural direction parameters.
    Binaural,
    /// Encode sources into an `AmbiX` transport of the given order.
    Ambisonic {
        /// Ambisonic order (clamped to [`crate::ambisonics::MAX_ORDER`]).
        order: usize,
    },
}

/// The format-specific result of a render.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderPayload {
    /// One per-bed-channel gain row per source (width [`BedLayout::channel_count`]).
    Bed(Vec<Vec<Sample>>),
    /// One binaural direction parameter set per source.
    Binaural(Vec<BinauralDirection>),
    /// One `AmbiX` coefficient vector per source, with the encoded order.
    Ambisonic {
        /// Encoded Ambisonic order.
        order: usize,
        /// Per-source `AmbiX` coefficient rows (ACN order, SN3D).
        rows: Vec<Vec<Sample>>,
    },
}

/// The full result of a render: the representative clusters (empty when the
/// budget forced a full fold) and the format-specific payload.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderOutput {
    /// The clustered representatives used, or empty when every object was
    /// folded directly (zero budget).
    pub clusters: Vec<ClusteredObject>,
    /// The format-specific gain data.
    pub payload: RenderPayload,
}

/// Renders a flat object list for `bed` under `budget` into `format`.
///
/// This is the scene-independent core used by [`render_scene`]; it performs
/// the budget -> clustering -> fold/encode pipeline described at the module
/// level.
#[must_use]
pub fn render_objects(
    objects: &[AudioObject],
    bed: BedLayout,
    budget: ObjectBudget,
    format: OutputFormat,
) -> RenderOutput {
    let target = budget.target_clusters(objects.len());
    let clusters = clustering::cluster_objects(objects, target);
    // A zero target (zero budget) yields no clusters; fold the raw objects.
    let folded = clusters.is_empty() && !objects.is_empty();

    let payload = match format {
        OutputFormat::Bed => {
            let rows = if folded {
                fold::fold_objects_to_bed(objects, bed)
            } else {
                fold::fold_clusters_to_bed(&clusters, bed)
            };
            RenderPayload::Bed(rows)
        }
        OutputFormat::Binaural => {
            let dirs = if folded {
                fold::fold_objects_to_binaural(objects)
            } else {
                fold::fold_clusters_to_binaural(&clusters)
            };
            RenderPayload::Binaural(dirs)
        }
        OutputFormat::Ambisonic { order } => {
            let order = order.min(ambisonics::MAX_ORDER);
            let rows = if folded {
                objects
                    .iter()
                    .map(|o| ambisonics::encode_object(o, order))
                    .collect()
            } else {
                ambisonics::encode_clusters(&clusters, order)
            };
            RenderPayload::Ambisonic { order, rows }
        }
    };

    RenderOutput { clusters, payload }
}

/// Renders `scene`'s current objects under `budget` into `format`, using the
/// scene's own bed layout.
///
/// Callers should [`ObjectScene::advance_to`] the desired time first so the
/// snapshot reflects the current metadata.
#[must_use]
pub fn render_scene(
    scene: &ObjectScene,
    budget: ObjectBudget,
    format: OutputFormat,
) -> RenderOutput {
    let objects = scene.current_objects();
    render_objects(&objects, scene.bed(), budget, format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ambisonics::channel_count;
    use crate::bed::direction_from_angles;
    use crate::clustering::total_energy;
    use crate::object::{AudioObject, ObjectId};
    use bevy_math::ops;

    const EPS: Sample = 1e-3;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn obj(id: u32, az: Sample, el: Sample, gain: Sample) -> AudioObject {
        AudioObject::new(ObjectId(id), direction_from_angles(az, el), gain)
    }

    #[test]
    fn within_budget_passes_objects_through_as_clusters() {
        let objects = [obj(0, 0.0, 0.0, 1.0), obj(1, 90.0, 0.0, 0.5)];
        let out = render_objects(
            &objects,
            BedLayout::Surround7_1_4,
            ObjectBudget::new(8),
            OutputFormat::Bed,
        );
        assert_eq!(out.clusters.len(), 2);
        match out.payload {
            RenderPayload::Bed(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), BedLayout::Surround7_1_4.channel_count());
            }
            _ => panic!("expected bed payload"),
        }
    }

    #[test]
    fn over_budget_reduces_to_budget_count() {
        let objects = [
            obj(0, 0.0, 0.0, 1.0),
            obj(1, 5.0, 0.0, 1.0),
            obj(2, 90.0, 0.0, 1.0),
            obj(3, 95.0, 0.0, 1.0),
        ];
        let out = render_objects(
            &objects,
            BedLayout::Surround7_1_4,
            ObjectBudget::new(2),
            OutputFormat::Bed,
        );
        assert_eq!(out.clusters.len(), 2);
        // Clustering is energy preserving.
        let input: Sample = objects.iter().map(AudioObject::energy).sum();
        assert!(close(total_energy(&out.clusters), input));
    }

    #[test]
    fn zero_budget_folds_all_objects_into_bed() {
        let objects = [obj(0, 0.0, 0.0, 1.0), obj(1, 90.0, 0.0, 1.0)];
        let out = render_objects(
            &objects,
            BedLayout::Surround7_1_4,
            ObjectBudget::new(0),
            OutputFormat::Bed,
        );
        assert!(out.clusters.is_empty());
        match out.payload {
            RenderPayload::Bed(rows) => assert_eq!(rows.len(), 2),
            _ => panic!("expected bed payload"),
        }
    }

    #[test]
    fn binaural_payload_has_one_entry_per_cluster() {
        let objects = [obj(0, -30.0, 0.0, 1.0), obj(1, 30.0, 0.0, 1.0)];
        let out = render_objects(
            &objects,
            BedLayout::Stereo,
            ObjectBudget::new(4),
            OutputFormat::Binaural,
        );
        match out.payload {
            RenderPayload::Binaural(dirs) => {
                assert_eq!(dirs.len(), 2);
                // One leans left (negative azimuth), one right (positive).
                let has_left = dirs.iter().any(|d| d.azimuth < -1.0);
                let has_right = dirs.iter().any(|d| d.azimuth > 1.0);
                assert!(has_left && has_right);
            }
            _ => panic!("expected binaural payload"),
        }
    }

    #[test]
    fn ambisonic_payload_rows_match_channel_count_and_order() {
        let objects = [obj(0, 0.0, 0.0, 0.5), obj(1, 120.0, 10.0, 0.5)];
        let out = render_objects(
            &objects,
            BedLayout::Surround7_1_4,
            ObjectBudget::new(4),
            OutputFormat::Ambisonic { order: 2 },
        );
        match out.payload {
            RenderPayload::Ambisonic { order, rows } => {
                assert_eq!(order, 2);
                assert_eq!(rows.len(), 2);
                for r in &rows {
                    assert_eq!(r.len(), channel_count(2));
                }
            }
            _ => panic!("expected ambisonic payload"),
        }
    }

    #[test]
    fn ambisonic_order_is_clamped() {
        let objects = [obj(0, 0.0, 0.0, 1.0)];
        let out = render_objects(
            &objects,
            BedLayout::Stereo,
            ObjectBudget::new(4),
            OutputFormat::Ambisonic { order: 99 },
        );
        match out.payload {
            RenderPayload::Ambisonic { order, rows } => {
                assert_eq!(order, ambisonics::MAX_ORDER);
                assert_eq!(rows[0].len(), channel_count(ambisonics::MAX_ORDER));
            }
            _ => panic!("expected ambisonic payload"),
        }
    }

    #[test]
    fn empty_scene_renders_empty_payload() {
        let objects: [AudioObject; 0] = [];
        let out = render_objects(
            &objects,
            BedLayout::Stereo,
            ObjectBudget::new(4),
            OutputFormat::Bed,
        );
        assert!(out.clusters.is_empty());
        match out.payload {
            RenderPayload::Bed(rows) => assert!(rows.is_empty()),
            _ => panic!("expected bed payload"),
        }
    }

    #[test]
    fn render_scene_uses_scene_bed() {
        let mut scene = ObjectScene::new(BedLayout::Surround5_1_4);
        scene.add_object(obj(0, 0.0, 0.0, 1.0));
        let out = render_scene(&scene, ObjectBudget::new(4), OutputFormat::Bed);
        match out.payload {
            RenderPayload::Bed(rows) => {
                assert_eq!(rows[0].len(), BedLayout::Surround5_1_4.channel_count());
            }
            _ => panic!("expected bed payload"),
        }    }
}
