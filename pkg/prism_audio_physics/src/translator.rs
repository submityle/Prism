//! The stateful core that turns per-block contact facts into audio drive.
//!
//! [`ContactAudioTranslator`] is the heart of the bridge. Each block the host
//! calls [`ContactAudioTranslator::begin_block`], feeds every contact manifold
//! through [`ContactAudioTranslator::ingest`] tagged with its lifecycle phase,
//! and finally [`ContactAudioTranslator::drain`]s the accumulated procedural
//! events. A started contact becomes a sample-accurate
//! [`prism_audio_procedural::contact::ImpactEvent`], a persisting contact a
//! [`prism_audio_procedural::contact::SustainEvent`], and an ended contact a
//! [`prism_audio_procedural::contact::SeparationEvent`] (after which the contact
//! is forgotten). On drain the impacts are merged, far-field clustered, and
//! truncated to the per-block budget so the voice count stays bounded.
//!
//! This stage runs at block start, off the real-time inner loop, so the modest
//! allocation it performs is acceptable; the real-time path only consumes the
//! already-built event vectors.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the orchestration of design section 47.1: it wires
//! [`crate::kinematics`], [`crate::impulse`], [`crate::roughness`],
//! [`crate::offset`], [`crate::material`], and [`crate::cluster`] onto the
//! [`prism_audio_procedural::contact`] event stream feeding
//! `prism_audio_core::scheduler::EventScheduler`.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use alloc::collections::BTreeMap;
use core::mem;

use bevy_math::Vec3;

use prism_audio_procedural::contact::{
    merge_impacts, ContactId, ImpactEvent, SeparationEvent, SustainEvent,
};

use crate::cluster::cluster_far_impacts;
use crate::config::TranslatorConfig;
use crate::contact_id::{procedural_id, ContactKey};
use crate::contact_input::{ContactManifoldView, ContactPhase};
use crate::impulse::{reduced_mass, ImpulseEstimate};
use crate::kinematics::{decompose, relative_velocity};
use crate::material::MaterialResolver;
use crate::offset::BlockClock;
use crate::roughness::{strike_position, surface_roughness};
use crate::body::BodyAudioState;

/// The procedural event batch produced for one audio block.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContactDrive {
    /// Discrete impacts, merged, clustered, and budget-capped.
    pub impacts: Vec<ImpactEvent>,
    /// Continuous sustain samples for persisting contacts.
    pub sustains: Vec<SustainEvent>,
    /// Separation events for contacts that broke this block.
    pub separations: Vec<SeparationEvent>,
}

/// Stateful translator from physics contact facts to procedural drive.
#[derive(Clone, Debug)]
pub struct ContactAudioTranslator {
    config: TranslatorConfig,
    resolver: MaterialResolver,
    /// Persistent contacts, so a manifold keeps its id across blocks.
    contacts: BTreeMap<ContactKey, ContactId>,
    clock: BlockClock,
    impacts: Vec<ImpactEvent>,
    /// Representative world position of each contact id emitted this block.
    positions: BTreeMap<ContactId, Vec3>,
    sustains: Vec<SustainEvent>,
    separations: Vec<SeparationEvent>,
}

impl ContactAudioTranslator {
    /// Builds a translator from its configuration and material resolver.
    #[inline]
    #[must_use]
    pub fn new(config: TranslatorConfig, resolver: MaterialResolver) -> Self {
        Self {
            config,
            resolver,
            contacts: BTreeMap::new(),
            clock: BlockClock::new(48_000, 0),
            impacts: Vec::new(),
            positions: BTreeMap::new(),
            sustains: Vec::new(),
            separations: Vec::new(),
        }
    }

    /// Returns the number of contacts currently tracked across blocks.
    #[inline]
    #[must_use]
    pub fn tracked_contacts(&self) -> usize {
        self.contacts.len()
    }

    /// Resets the per-block event buffers and installs the block clock.
    ///
    /// The persistent contact map survives across blocks; only the per-block
    /// impact/sustain/separation buffers and position map are cleared.
    #[inline]
    pub fn begin_block(&mut self, clock: BlockClock) {
        self.clock = clock;
        self.impacts.clear();
        self.positions.clear();
        self.sustains.clear();
        self.separations.clear();
    }

    /// Resolves (and remembers) the contact id for a manifold view.
    #[inline]
    fn contact_id_for(&mut self, view: &ContactManifoldView) -> (ContactKey, ContactId) {
        let key = ContactKey::new(view.body_a, view.body_b);
        let id = *self.contacts.entry(key).or_insert_with(|| procedural_id(key));
        (key, id)
    }

    /// Ingests one contact manifold, emitting the event for its phase.
    ///
    /// `a` and `b` are the kinematic snapshots of the two bodies named by the
    /// view; `phase` selects impact (`Started`), sustain (`Persisting`), or
    /// separation (`Ended`); and `substep_fraction` positions a started impact
    /// on the right sample inside the block.
    pub fn ingest(
        &mut self,
        view: &ContactManifoldView,
        a: &BodyAudioState,
        b: &BodyAudioState,
        phase: ContactPhase,
        substep_fraction: f32,
    ) {
        let pair = self.resolver.resolve(a.material(), b.material());
        let (key, contact) = self.contact_id_for(view);

        // Representative contact point and its overlap depth.
        let (point, penetration) = match view.representative_point() {
            Some(rep) => (rep.world_point, rep.penetration),
            None => ((a.center_of_mass() + b.center_of_mass()) * 0.5, 0.0),
        };

        let v_rel = relative_velocity(a, b, point);
        let split = decompose(v_rel, view.normal);

        match phase {
            ContactPhase::Started => {
                let m = reduced_mass(a.inverse_mass(), b.inverse_mass());
                let est = ImpulseEstimate::from_split(m, split, self.config.restitution);
                let strike = strike_position(
                    point,
                    a.center_of_mass(),
                    self.config.default_body_extent,
                );
                let offset = self.clock.sample_offset(substep_fraction);
                let impact = ImpactEvent::new(
                    contact,
                    pair,
                    est.magnitude,
                    est.normal,
                    est.tangential,
                    strike,
                    offset,
                    self.clock.block_frames,
                );
                self.impacts.push(impact);
                self.positions.insert(contact, point);
            }
            ContactPhase::Persisting => {
                let roughness = surface_roughness(pair, self.config.default_roughness);
                let sustain = SustainEvent::new(
                    contact,
                    pair,
                    split.tangential_speed,
                    penetration,
                    roughness,
                );
                self.sustains.push(sustain);
            }
            ContactPhase::Ended => {
                self.separations.push(SeparationEvent { contact });
                self.contacts.remove(&key);
            }
        }
    }

    /// Drains the accumulated events, applying merge, cluster, and budget.
    ///
    /// Impacts are first merged by
    /// [`prism_audio_procedural::contact::merge_impacts`], then far-field
    /// clustered by [`crate::cluster::cluster_far_impacts`], then sorted by
    /// descending impulse and truncated to
    /// [`TranslatorConfig::max_impacts_per_block`] so the loudest strikes
    /// survive the budget. The per-block buffers are emptied.
    pub fn drain(&mut self) -> ContactDrive {
        let mut impacts = mem::take(&mut self.impacts);
        let sustains = mem::take(&mut self.sustains);
        let separations = mem::take(&mut self.separations);
        let positions_map = mem::take(&mut self.positions);

        merge_impacts(&mut impacts, self.config.merge);

        // Rebuild an index-aligned position slice from the per-contact map; a
        // contact with no recorded position falls back to the listener so it is
        // treated as near and never spuriously clustered.
        let positions: Vec<Vec3> = impacts
            .iter()
            .map(|e| {
                positions_map
                    .get(&e.contact)
                    .copied()
                    .unwrap_or(self.config.cluster.listener)
            })
            .collect();

        cluster_far_impacts(&mut impacts, &positions, &self.config.cluster);

        if impacts.len() > self.config.max_impacts_per_block {
            // Keep the loudest strikes; stable sort for a deterministic choice.
            impacts.sort_by(|x, y| {
                y.impulse
                    .partial_cmp(&x.impulse)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            impacts.truncate(self.config.max_impacts_per_block);
        }

        ContactDrive {
            impacts,
            sustains,
            separations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::BodyAudioId;
    use crate::contact_input::ContactPointView;
    use crate::material::AudioMaterialId;

    fn body(id: u64, linear: Vec3, inv_mass: f32, mat: u32) -> BodyAudioState {
        BodyAudioState::new(
            BodyAudioId(id),
            linear,
            Vec3::ZERO,
            Vec3::ZERO,
            inv_mass,
            AudioMaterialId(mat),
        )
    }

    fn manifold(a: u64, b: u64, point: Vec3, penetration: f32) -> ContactManifoldView {
        ContactManifoldView::new(
            BodyAudioId(a),
            BodyAudioId(b),
            Vec3::Y,
            alloc::vec![ContactPointView::new(point, penetration)],
        )
    }

    fn translator() -> ContactAudioTranslator {
        ContactAudioTranslator::new(TranslatorConfig::default(), MaterialResolver::new())
    }

    #[test]
    fn single_impact_end_to_end() {
        let mut t = translator();
        t.begin_block(BlockClock::new(48_000, 128));
        // Body a drops onto body b along +y (normal points a->b = +y).
        let a = body(1, Vec3::new(0.0, 2.0, 0.0), 1.0, 10);
        let b = body(2, Vec3::ZERO, 0.0, 20);
        let view = manifold(1, 2, Vec3::new(0.0, 0.0, 0.0), 0.01);
        t.ingest(&view, &a, &b, ContactPhase::Started, 0.5);
        let drive = t.drain();
        assert_eq!(drive.impacts.len(), 1);
        assert!(drive.impacts[0].impulse > 0.0);
        assert_eq!(drive.impacts[0].sample_offset, 64);
        assert_eq!(t.tracked_contacts(), 1);
    }

    #[test]
    fn machine_gun_impacts_merge() {
        let mut t = translator();
        t.begin_block(BlockClock::new(48_000, 512));
        let a = body(1, Vec3::new(0.0, 2.0, 0.0), 1.0, 10);
        let b = body(2, Vec3::ZERO, 0.0, 20);
        // Same contact, several sub-step impulses within the merge window.
        for _ in 0..4 {
            let view = manifold(1, 2, Vec3::ZERO, 0.01);
            t.ingest(&view, &a, &b, ContactPhase::Started, 0.0);
        }
        let drive = t.drain();
        assert_eq!(drive.impacts.len(), 1);
    }

    #[test]
    fn sustain_across_blocks_keeps_contact() {
        let mut t = translator();
        let a = body(1, Vec3::new(3.0, 0.0, 0.0), 1.0, 10);
        let b = body(2, Vec3::ZERO, 0.0, 20);

        t.begin_block(BlockClock::new(48_000, 128));
        let view = manifold(1, 2, Vec3::ZERO, 0.02);
        t.ingest(&view, &a, &b, ContactPhase::Started, 0.0);
        let _ = t.drain();

        t.begin_block(BlockClock::new(48_000, 128));
        let view = manifold(1, 2, Vec3::ZERO, 0.02);
        t.ingest(&view, &a, &b, ContactPhase::Persisting, 0.0);
        let drive = t.drain();
        assert_eq!(drive.sustains.len(), 1);
        // Tangential motion of 3 m/s should show up in the sustain.
        assert!((drive.sustains[0].tangential_speed - 3.0).abs() < 1e-6);
        assert_eq!(t.tracked_contacts(), 1);
    }

    #[test]
    fn separation_forgets_contact() {
        let mut t = translator();
        let a = body(1, Vec3::new(0.0, 2.0, 0.0), 1.0, 10);
        let b = body(2, Vec3::ZERO, 0.0, 20);

        t.begin_block(BlockClock::new(48_000, 128));
        t.ingest(&manifold(1, 2, Vec3::ZERO, 0.01), &a, &b, ContactPhase::Started, 0.0);
        let _ = t.drain();
        assert_eq!(t.tracked_contacts(), 1);

        t.begin_block(BlockClock::new(48_000, 128));
        t.ingest(&manifold(1, 2, Vec3::ZERO, 0.0), &a, &b, ContactPhase::Ended, 0.0);
        let drive = t.drain();
        assert_eq!(drive.separations.len(), 1);
        assert_eq!(t.tracked_contacts(), 0);
    }

    #[test]
    fn budget_truncates_to_loudest() {
        let mut config = TranslatorConfig {
            max_impacts_per_block: 2,
            ..Default::default()
        };
        // Disable clustering so each distinct contact survives to the budget.
        config.cluster.min_cluster_size = usize::MAX;
        let mut t = ContactAudioTranslator::new(config, MaterialResolver::new());
        t.begin_block(BlockClock::new(48_000, 512));
        // Five distinct contacts with increasing speed (louder impacts).
        for i in 0..5u64 {
            let a = body(i * 2 + 1, Vec3::new(0.0, 1.0 + i as f32, 0.0), 1.0, 10);
            let b = body(i * 2 + 2, Vec3::ZERO, 0.0, 20);
            let view = manifold(i * 2 + 1, i * 2 + 2, Vec3::new(i as f32 * 100.0, 0.0, 0.0), 0.01);
            t.ingest(&view, &a, &b, ContactPhase::Started, 0.0);
        }
        let drive = t.drain();
        assert_eq!(drive.impacts.len(), 2);
        // The two surviving impacts are the loudest two.
        assert!(drive.impacts[0].impulse >= drive.impacts[1].impulse);
        assert!(drive.impacts[1].impulse > 0.0);
    }

    #[test]
    fn static_pair_makes_no_impulse() {
        let mut t = translator();
        t.begin_block(BlockClock::new(48_000, 128));
        let a = body(1, Vec3::new(0.0, 2.0, 0.0), 0.0, 10);
        let b = body(2, Vec3::ZERO, 0.0, 20);
        t.ingest(&manifold(1, 2, Vec3::ZERO, 0.01), &a, &b, ContactPhase::Started, 0.0);
        let drive = t.drain();
        assert_eq!(drive.impacts.len(), 1);
        assert!(drive.impacts[0].impulse.abs() < 1e-6);
    }
}
