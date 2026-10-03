//! Atmos-style object-budget split: keep the most important objects discrete
//! and fold the overflow into the channel bed.
//!
//! Platform object renderers expose a finite discrete-object budget
//! ([`crate::budget::ObjectBudget`]). When a scene presents more objects than
//! the budget allows, the classic Dolby-Atmos delivery model keeps the most
//! important sources as *discrete* objects and folds the remaining *overflow*
//! sources down into the channel [`crate::bed::BedLayout`]. This complements
//! [`crate::clustering`], which instead merges every object into a smaller set
//! of representatives: splitting preserves full per-object fidelity for the
//! top-ranked sources (position, spread, snap) at the cost of flattening the
//! long tail onto the bed, whereas clustering spreads the reduction evenly.
//!
//! [`split_to_budget`] partitions a flat object list into:
//!
//! * the top-`max_objects` sources by importance, kept as discrete
//!   [`AudioObject`]s (importance-descending order), and
//! * one summed per-bed-channel gain row for every remaining (overflow)
//!   source, folded through the constant-power panner of [`crate::pan`].
//!
//! Importance ranks `priority` first, then acoustic `energy`, with the object
//! `id` as a deterministic final tiebreak, so the split is bit-reproducible
//! for a given input regardless of platform.
//!
//! The overflow row is a control-rate *gain* row (not a per-sample mix): add it
//! to the bed bus by multiplying each overflow source's mono signal by its
//! panned gains and summing, exactly as [`crate::fold::sum_bed_matrix`]
//! documents. Because the panner is constant power, each overflow source
//! deposits its own `gain^2` of energy; the summed row is a gain matrix, so its
//! channel sum-of-squares reflects inter-source gain overlap rather than a
//! naive energy sum (uncorrelated source signals still sum to the per-source
//! energy total acoustically).
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the object-bed budget limit of design section 33 (and the
//! hardware object budget of section 44.2): the "keep top-K discrete plus fold
//! the rest to the bed" delivery path. Built on [`crate::budget`],
//! [`crate::fold`], and [`crate::pan`]; parallels [`crate::clustering`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::bed::BedLayout;
use crate::budget::ObjectBudget;
use crate::fold;
use crate::object::AudioObject;

/// The result of partitioning a scene's objects against a hardware budget.
///
/// `objects` holds the retained discrete sources in importance-descending
/// order (highest `priority`, then `energy`, then lowest `id`). `bed_overflow`
/// is the summed per-bed-channel gain row for every source that did not fit the
/// budget; its length is always the bed's [`BedLayout::channel_count`], and it
/// is an all-zero row when nothing overflowed.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BudgetSplit {
    /// The retained discrete objects, importance-descending.
    pub objects: Vec<AudioObject>,
    /// Summed per-bed-channel gain row for the folded overflow sources.
    pub bed_overflow: Vec<Sample>,
}

impl BudgetSplit {
    /// Returns the number of retained discrete objects.
    #[must_use]
    pub fn discrete_count(&self) -> usize {
        self.objects.len()
    }

    /// Returns whether any sources were folded into the bed overflow (true
    /// when the overflow row carries non-zero gain on any channel).
    #[must_use]
    pub fn has_overflow(&self) -> bool {
        self.bed_overflow.iter().any(|&g| g != 0.0)
    }
}

/// Compares two objects by rendering importance, strongest first.
///
/// Ordering is `priority` descending, then acoustic `energy` descending, then
/// `id` ascending. [`Sample::total_cmp`] gives a deterministic total order over
/// all float values (including non-finite ones) so the sort never depends on
/// platform floating-point comparison quirks.
fn more_important(a: &AudioObject, b: &AudioObject) -> core::cmp::Ordering {
    b.priority
        .total_cmp(&a.priority)
        .then_with(|| b.energy().total_cmp(&a.energy()))
        .then_with(|| a.id.get().cmp(&b.id.get()))
}

/// Partitions `objects` against `budget`, keeping the most important sources
/// discrete and folding the overflow onto `bed`.
///
/// When every object fits the budget, all are retained discrete and
/// `bed_overflow` is an all-zero row of width [`BedLayout::channel_count`].
/// When the budget is zero, nothing is kept discrete and every source is folded
/// into `bed_overflow`. Otherwise the top [`ObjectBudget::max_objects`] sources
/// by [`more_important`] are retained and the rest are folded.
///
/// The result is deterministic for a given input.
///
/// ```
/// use bevy_math::Vec3;
/// use prism_audio_object::bed::BedLayout;
/// use prism_audio_object::budget::ObjectBudget;
/// use prism_audio_object::budget_split::split_to_budget;
/// use prism_audio_object::object::{AudioObject, ObjectId};
///
/// let mut loud = AudioObject::new(ObjectId(0), Vec3::new(1.0, 0.0, 0.0), 1.0);
/// loud.priority = 5.0;
/// let quiet = AudioObject::new(ObjectId(1), Vec3::new(-1.0, 0.0, 0.0), 0.1);
/// let split = split_to_budget(&[loud, quiet], BedLayout::Stereo, ObjectBudget::new(1));
///
/// // The louder, higher-priority object stays discrete; the other folds down.
/// assert_eq!(split.discrete_count(), 1);
/// assert_eq!(split.objects[0].id, ObjectId(0));
/// assert_eq!(split.bed_overflow.len(), BedLayout::Stereo.channel_count());
/// assert!(split.has_overflow());
/// ```
#[must_use]
pub fn split_to_budget(
    objects: &[AudioObject],
    bed: BedLayout,
    budget: ObjectBudget,
) -> BudgetSplit {
    let channels = bed.channel_count();

    // Fast path: everything fits, nothing folds.
    if budget.fits(objects.len()) {
        return BudgetSplit {
            objects: objects.to_vec(),
            bed_overflow: fold::sum_bed_matrix(&[], channels),
        };
    }

    // Rank by importance without disturbing the caller's slice.
    let mut ranked: Vec<usize> = (0..objects.len()).collect();
    ranked.sort_by(|&a, &b| more_important(&objects[a], &objects[b]));

    let keep = budget.max_objects();
    let (discrete_idx, overflow_idx) = ranked.split_at(keep.min(ranked.len()));

    let retained: Vec<AudioObject> = discrete_idx.iter().map(|&i| objects[i]).collect();

    let overflow: Vec<AudioObject> = overflow_idx.iter().map(|&i| objects[i]).collect();
    let overflow_rows = fold::fold_objects_to_bed(&overflow, bed);
    let bed_overflow = fold::sum_bed_matrix(&overflow_rows, channels);

    BudgetSplit {
        objects: retained,
        bed_overflow,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bed::direction_from_angles;
    use crate::object::ObjectId;
    use bevy_math::ops;

    const EPS: Sample = 1e-4;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn obj(id: u32, az: Sample, gain: Sample, priority: Sample) -> AudioObject {
        let mut o = AudioObject::new(ObjectId(id), direction_from_angles(az, 0.0), gain);
        o.priority = priority;
        o
    }

    fn sum_sq(row: &[Sample]) -> Sample {
        row.iter().map(|&g| g * g).sum()
    }

    #[test]
    fn within_budget_keeps_all_and_zero_overflow() {
        let objects = [obj(0, 0.0, 1.0, 1.0), obj(1, 90.0, 0.5, 1.0)];
        let split = split_to_budget(&objects, BedLayout::Surround7_1_4, ObjectBudget::new(8));
        assert_eq!(split.discrete_count(), 2);
        assert_eq!(
            split.bed_overflow.len(),
            BedLayout::Surround7_1_4.channel_count()
        );
        assert!(!split.has_overflow());
        assert!(close(sum_sq(&split.bed_overflow), 0.0));
    }

    #[test]
    fn exactly_at_budget_folds_nothing() {
        let objects = [obj(0, 0.0, 1.0, 1.0), obj(1, 90.0, 1.0, 1.0)];
        let split = split_to_budget(&objects, BedLayout::Stereo, ObjectBudget::new(2));
        assert_eq!(split.discrete_count(), 2);
        assert!(!split.has_overflow());
    }

    #[test]
    fn over_budget_retains_most_important_by_priority() {
        // Four equal-gain objects; priorities pick the discrete survivors.
        let objects = [
            obj(0, 0.0, 1.0, 1.0),
            obj(1, 30.0, 1.0, 9.0),
            obj(2, 60.0, 1.0, 2.0),
            obj(3, 90.0, 1.0, 7.0),
        ];
        let split = split_to_budget(&objects, BedLayout::Surround7_1_4, ObjectBudget::new(2));
        assert_eq!(split.discrete_count(), 2);
        // Highest priority first: id 1 (9.0) then id 3 (7.0).
        assert_eq!(split.objects[0].id, ObjectId(1));
        assert_eq!(split.objects[1].id, ObjectId(3));
        assert!(split.has_overflow());
        assert_eq!(
            split.bed_overflow.len(),
            BedLayout::Surround7_1_4.channel_count()
        );
    }

    #[test]
    fn energy_breaks_priority_ties() {
        // Equal priority; louder objects (more energy) win the discrete slots.
        let objects = [
            obj(0, 0.0, 0.2, 1.0),
            obj(1, 30.0, 0.9, 1.0),
            obj(2, 60.0, 0.5, 1.0),
        ];
        let split = split_to_budget(&objects, BedLayout::Stereo, ObjectBudget::new(1));
        assert_eq!(split.discrete_count(), 1);
        assert_eq!(split.objects[0].id, ObjectId(1));
    }

    #[test]
    fn id_breaks_full_ties_deterministically() {
        // Identical priority and gain: lowest id is retained.
        let objects = [obj(7, 0.0, 1.0, 1.0), obj(3, 10.0, 1.0, 1.0)];
        let split = split_to_budget(&objects, BedLayout::Stereo, ObjectBudget::new(1));
        assert_eq!(split.objects[0].id, ObjectId(3));
    }

    #[test]
    fn zero_budget_folds_everything() {
        let objects = [obj(0, 0.0, 1.0, 1.0), obj(1, 90.0, 1.0, 1.0)];
        let split = split_to_budget(&objects, BedLayout::Surround7_1_4, ObjectBudget::new(0));
        assert_eq!(split.discrete_count(), 0);
        assert!(split.has_overflow());
        assert!(sum_sq(&split.bed_overflow) > 0.0);
        assert_eq!(
            split.bed_overflow.len(),
            BedLayout::Surround7_1_4.channel_count()
        );
    }

    #[test]
    fn overflow_row_matches_direct_fold_of_overflow_sources() {
        // The documented construction: the overflow row equals summing the fold
        // rows of exactly the non-retained (overflow) sources.
        let objects = [
            obj(0, 0.0, 1.0, 5.0),
            obj(1, 45.0, 1.0, 4.0),
            obj(2, 90.0, 1.0, 3.0),
            obj(3, 135.0, 1.0, 2.0),
        ];
        let bed = BedLayout::Surround5_1_4;
        let split = split_to_budget(&objects, bed, ObjectBudget::new(2));
        // Retained are ids 0 and 1 (priorities 5, 4); overflow are ids 2 and 3.
        let overflow = [objects[2], objects[3]];
        let expected =
            fold::sum_bed_matrix(&fold::fold_objects_to_bed(&overflow, bed), bed.channel_count());
        assert_eq!(split.bed_overflow.len(), expected.len());
        for (a, b) in split.bed_overflow.iter().zip(expected.iter()) {
            assert!(close(*a, *b));
        }
    }

    #[test]
    fn single_overflow_source_preserves_its_energy_in_bed() {
        // One retained, one overflow: a single folded constant-power source
        // deposits exactly gain^2 energy (no inter-source gain overlap).
        let objects = [obj(0, 0.0, 1.0, 9.0), obj(1, 90.0, 0.5, 1.0)];
        let split = split_to_budget(&objects, BedLayout::Surround7_1_4, ObjectBudget::new(1));
        assert_eq!(split.discrete_count(), 1);
        assert_eq!(split.objects[0].id, ObjectId(0));
        assert!(close(sum_sq(&split.bed_overflow), 0.25));
    }

    #[test]
    fn empty_scene_yields_empty_discrete_and_zero_overflow() {
        let objects: [AudioObject; 0] = [];
        let split = split_to_budget(&objects, BedLayout::Stereo, ObjectBudget::new(4));
        assert_eq!(split.discrete_count(), 0);
        assert!(!split.has_overflow());
        assert_eq!(split.bed_overflow.len(), BedLayout::Stereo.channel_count());
    }

    #[test]
    fn split_is_deterministic() {
        let objects = [
            obj(5, 0.0, 0.7, 3.0),
            obj(2, 45.0, 0.7, 3.0),
            obj(9, 90.0, 0.4, 5.0),
            obj(1, 135.0, 0.9, 1.0),
        ];
        let a = split_to_budget(&objects, BedLayout::Surround7_1_4, ObjectBudget::new(2));
        let b = split_to_budget(&objects, BedLayout::Surround7_1_4, ObjectBudget::new(2));
        assert_eq!(a, b);
    }
}
