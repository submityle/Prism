//! Deterministic de-duplication and merging of near-coincident impacts.
//!
//! Physics solvers routinely report several contact impulses for what a
//! listener hears as a single strike (multiple manifold points, sub-step
//! re-solves). Rendering each as its own excitation produces a smeared "voice
//! storm". Before the real-time stage, and off the audio thread, this module
//! collapses impacts that belong to the same contact and fall within a short
//! time window into one: their energies add (the physically correct way to
//! combine simultaneous excitations), the strongest sub-impact donates the
//! contact point, and the earliest offset is kept so the merged strike still
//! lands sample-accurately. The procedure is a stable sort followed by a linear
//! sweep, so it is fully deterministic and replayable.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the budget/merge stage of design section 47.1; consumes
//! [`crate::contact::event::ImpactEvent`] and feeds the ordered stream drained
//! by [`crate::contact::bus::ContactEventBus`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::contact::event::{ContactPoint, ImpactEvent};
use prism_audio_core::math::Sample;

/// Configuration for the impact merge pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MergeConfig {
    /// Two impacts on the same contact within this many samples are merged.
    pub window_samples: u32,
}

impl Default for MergeConfig {
    #[inline]
    fn default() -> Self {
        // ~1 ms at 48 kHz: tighter than a perceptible double-strike.
        Self { window_samples: 48 }
    }
}

/// Combines energy from `src` into `dst`, keeping the strongest contact point.
///
/// Energies add in the linear impulse domain (simultaneous excitations sum);
/// the earliest sample offset is retained so the merged strike stays
/// sample-accurate.
#[inline]
fn absorb(dst: &mut ImpactEvent, src: &ImpactEvent) {
    if src.impulse > dst.impulse {
        dst.point = src.point;
    }
    dst.impulse += src.impulse;
    dst.normal += src.normal;
    dst.tangential += src.tangential;
    dst.sample_offset = dst.sample_offset.min(src.sample_offset);
}

/// Merges near-coincident impacts in place.
///
/// The slice is first stably sorted by `(contact, sample_offset)` so the sweep
/// is order-independent of the physics emit order, then adjacent impacts on the
/// same contact within [`MergeConfig::window_samples`] are folded together. The
/// vector is truncated to the merged count. Returns the number of surviving
/// impacts.
pub fn merge_impacts(impacts: &mut Vec<ImpactEvent>, config: MergeConfig) -> usize {
    if impacts.len() <= 1 {
        return impacts.len();
    }
    impacts.sort_by(|a, b| {
        a.contact
            .cmp(&b.contact)
            .then(a.sample_offset.cmp(&b.sample_offset))
    });

    let mut write = 0usize;
    for read in 1..impacts.len() {
        let current = impacts[read];
        let head = impacts[write];
        let same_contact = head.contact == current.contact;
        let within_window = current.sample_offset.saturating_sub(head.sample_offset)
            <= config.window_samples;
        if same_contact && within_window {
            let mut merged = head;
            absorb(&mut merged, &current);
            impacts[write] = merged;
        } else {
            write += 1;
            impacts[write] = current;
        }
    }
    let count = write + 1;
    impacts.truncate(count);
    count
}

/// Returns a single impact equivalent to a dense cluster of distant strikes.
///
/// Far-field crowds (many bodies rattling at a distance) are collapsed to one
/// "group impact" so they cost a single voice. The combined impulse is the sum
/// (energy-preserving), and the contact point is the energy-weighted mean,
/// giving a representative strike timbre. Returns [`None`] for an empty slice.
#[must_use]
pub fn cluster_to_group(impacts: &[ImpactEvent]) -> Option<ImpactEvent> {
    let first = impacts.first()?;
    let mut acc = *first;
    acc.impulse = 0.0;
    acc.normal = 0.0;
    acc.tangential = 0.0;
    let mut weighted_point = 0.0;
    let mut total = 0.0;
    let mut earliest = u32::MAX;
    for e in impacts {
        acc.impulse += e.impulse;
        acc.normal += e.normal;
        acc.tangential += e.tangential;
        weighted_point += e.point.value() * e.impulse;
        total += e.impulse;
        earliest = earliest.min(e.sample_offset);
    }
    let point: Sample = if total > Sample::MIN_POSITIVE {
        weighted_point / total
    } else {
        first.point.value()
    };
    acc.point = ContactPoint::new(point);
    acc.sample_offset = earliest;
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::MaterialPairId;

    fn impact(contact: u64, impulse: Sample, offset: u32) -> ImpactEvent {
        ImpactEvent::new(
            crate::contact::event::ContactId(contact),
            MaterialPairId::new(0, 0),
            impulse,
            impulse,
            0.0,
            ContactPoint::new(0.5),
            offset,
            512,
        )
    }

    #[test]
    fn coincident_impacts_merge_and_sum() {
        let mut v = alloc::vec![impact(1, 2.0, 10), impact(1, 3.0, 20)];
        let n = merge_impacts(&mut v, MergeConfig { window_samples: 48 });
        assert_eq!(n, 1);
        assert!((v[0].impulse - 5.0).abs() < 1e-6);
        assert_eq!(v[0].sample_offset, 10);
    }

    #[test]
    fn distant_in_time_not_merged() {
        let mut v = alloc::vec![impact(1, 2.0, 10), impact(1, 3.0, 400)];
        let n = merge_impacts(&mut v, MergeConfig { window_samples: 48 });
        assert_eq!(n, 2);
    }

    #[test]
    fn different_contacts_not_merged() {
        let mut v = alloc::vec![impact(1, 2.0, 10), impact(2, 3.0, 11)];
        let n = merge_impacts(&mut v, MergeConfig { window_samples: 48 });
        assert_eq!(n, 2);
    }

    #[test]
    fn merge_is_order_independent() {
        let mut a = alloc::vec![impact(1, 2.0, 10), impact(1, 3.0, 20)];
        let mut b = alloc::vec![impact(1, 3.0, 20), impact(1, 2.0, 10)];
        merge_impacts(&mut a, MergeConfig::default());
        merge_impacts(&mut b, MergeConfig::default());
        assert_eq!(a, b);
    }

    #[test]
    fn strongest_point_wins() {
        let mut strong = impact(1, 5.0, 30);
        strong.point = ContactPoint::new(0.9);
        let mut v = alloc::vec![impact(1, 1.0, 10), strong];
        merge_impacts(&mut v, MergeConfig::default());
        assert!((v[0].point.value() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn cluster_group_sums_energy() {
        let v = alloc::vec![impact(1, 1.0, 10), impact(2, 2.0, 20), impact(3, 1.0, 30)];
        let g = cluster_to_group(&v).unwrap();
        assert!((g.impulse - 4.0).abs() < 1e-6);
        assert_eq!(g.sample_offset, 10);
    }

    #[test]
    fn cluster_empty_is_none() {
        assert!(cluster_to_group(&[]).is_none());
    }
}
