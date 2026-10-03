//! The bounded, deterministic contact event bus (`ContactEventSource`).
//!
//! This is the hand-off point between the physics step and the audio engine.
//! The solver pushes raw [`ContactEvent`]s for the current block; the bus
//! enforces a per-block impact budget (replacing the weakest queued impact when
//! full, so loud strikes are never starved by a flood of taps), then at block
//! finalisation merges near-coincident impacts and orders them by sample offset
//! so the real-time stage can walk them in time order. Continuous sustain
//! samples and separations are collected alongside. Every buffer is
//! preallocated to its budget, so ingestion and finalisation never allocate and
//! the whole bus can be driven from a command-ring drain without touching the
//! heap on the hot path.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the contact event bus of design section 47.1; orders events for
//! sample-accurate dispatch (section 8) and feeds the modal
//! ([`crate::modal`]) and continuous ([`crate::continuous`]) stages.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::contact::event::{
    ContactEvent, ContactId, ImpactEvent, SeparationEvent, SustainEvent,
};
use crate::contact::merge::{merge_impacts, MergeConfig};

/// A bounded collector for one block's worth of contact events.
#[derive(Clone, Debug)]
pub struct ContactEventBus {
    impacts: Vec<ImpactEvent>,
    sustains: Vec<SustainEvent>,
    separations: Vec<SeparationEvent>,
    impact_budget: usize,
    sustain_budget: usize,
    merge_config: MergeConfig,
    finalized: bool,
}

impl ContactEventBus {
    /// Creates a bus sized for `impact_budget` impacts and `sustain_budget`
    /// simultaneous sustained contacts per block.
    ///
    /// Both budgets are forced to at least one, and all storage is preallocated
    /// so steady-state ingestion performs no allocation.
    #[must_use]
    pub fn new(impact_budget: usize, sustain_budget: usize, merge_config: MergeConfig) -> Self {
        let impact_budget = impact_budget.max(1);
        let sustain_budget = sustain_budget.max(1);
        Self {
            impacts: Vec::with_capacity(impact_budget),
            sustains: Vec::with_capacity(sustain_budget),
            separations: Vec::with_capacity(sustain_budget),
            impact_budget,
            sustain_budget,
            merge_config,
            finalized: false,
        }
    }

    /// Clears the bus for a new block without releasing its capacity.
    #[inline]
    pub fn begin_block(&mut self) {
        self.impacts.clear();
        self.sustains.clear();
        self.separations.clear();
        self.finalized = false;
    }

    /// Ingests one contact event, honouring the per-kind budgets.
    pub fn ingest(&mut self, event: ContactEvent) {
        match event {
            ContactEvent::Impact(e) => self.ingest_impact(e),
            ContactEvent::Sustain(e) => self.ingest_sustain(e),
            ContactEvent::Separation(e) => self.ingest_separation(e),
        }
    }

    /// Pushes an impact, replacing the weakest queued impact when over budget.
    fn ingest_impact(&mut self, event: ImpactEvent) {
        self.finalized = false;
        if self.impacts.len() < self.impact_budget {
            self.impacts.push(event);
            return;
        }
        // Over budget: evict the weakest impact if the newcomer is louder.
        let mut weakest = 0usize;
        for (i, e) in self.impacts.iter().enumerate() {
            if e.impulse < self.impacts[weakest].impulse {
                weakest = i;
            }
        }
        if event.impulse > self.impacts[weakest].impulse {
            self.impacts[weakest] = event;
        }
    }

    /// Pushes or coalesces a sustain sample; a contact already present this
    /// block has its latest sample kept (the newest physics state wins).
    fn ingest_sustain(&mut self, event: SustainEvent) {
        if let Some(slot) = self.sustains.iter_mut().find(|s| s.contact == event.contact) {
            *slot = event;
            return;
        }
        if self.sustains.len() < self.sustain_budget {
            self.sustains.push(event);
        }
    }

    /// Records a separation, de-duplicating repeated reports for one contact.
    fn ingest_separation(&mut self, event: SeparationEvent) {
        if self.separations.iter().any(|s| s.contact == event.contact) {
            return;
        }
        if self.separations.len() < self.separations.capacity() {
            self.separations.push(event);
        }
    }

    /// Finalises the block: merges near-coincident impacts and orders them by
    /// sample offset. Returns the number of surviving impacts. Idempotent
    /// within a block.
    pub fn finalize_block(&mut self) -> usize {
        if !self.finalized {
            merge_impacts(&mut self.impacts, self.merge_config);
            self.impacts.sort_by_key(|a| a.sample_offset);
            self.finalized = true;
        }
        self.impacts.len()
    }

    /// Returns the finalised, time-ordered impacts for this block.
    #[inline]
    #[must_use]
    pub fn impacts(&self) -> &[ImpactEvent] {
        &self.impacts
    }

    /// Returns the sustained-contact samples for this block.
    #[inline]
    #[must_use]
    pub fn sustains(&self) -> &[SustainEvent] {
        &self.sustains
    }

    /// Returns the separations reported this block.
    #[inline]
    #[must_use]
    pub fn separations(&self) -> &[SeparationEvent] {
        &self.separations
    }

    /// Returns `true` if `contact` separated this block.
    #[inline]
    #[must_use]
    pub fn separated(&self, contact: ContactId) -> bool {
        self.separations.iter().any(|s| s.contact == contact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::event::ContactPoint;
    use crate::material::MaterialPairId;

    fn impact(contact: u64, impulse: f32, offset: u32) -> ContactEvent {
        ContactEvent::Impact(ImpactEvent::new(
            ContactId(contact),
            MaterialPairId::new(0, 0),
            impulse,
            impulse,
            0.0,
            ContactPoint::new(0.5),
            offset,
            512,
        ))
    }

    #[test]
    fn finalize_orders_by_offset() {
        let mut bus = ContactEventBus::new(8, 4, MergeConfig::default());
        bus.begin_block();
        bus.ingest(impact(1, 1.0, 300));
        bus.ingest(impact(2, 1.0, 100));
        bus.ingest(impact(3, 1.0, 200));
        bus.finalize_block();
        let offs: Vec<u32> = bus.impacts().iter().map(|e| e.sample_offset).collect();
        assert_eq!(offs, alloc::vec![100, 200, 300]);
    }

    #[test]
    fn budget_keeps_strongest() {
        let mut bus = ContactEventBus::new(2, 4, MergeConfig::default());
        bus.begin_block();
        bus.ingest(impact(1, 1.0, 10));
        bus.ingest(impact(2, 2.0, 20));
        bus.ingest(impact(3, 5.0, 30)); // should evict the weakest (1.0)
        bus.finalize_block();
        let max = bus
            .impacts()
            .iter()
            .map(|e| e.impulse)
            .fold(0.0f32, f32::max);
        assert!((max - 5.0).abs() < 1e-6);
        assert_eq!(bus.impacts().len(), 2);
    }

    #[test]
    fn sustain_latest_wins() {
        let mut bus = ContactEventBus::new(4, 4, MergeConfig::default());
        bus.begin_block();
        bus.ingest(ContactEvent::Sustain(SustainEvent::new(
            ContactId(1),
            MaterialPairId::new(0, 0),
            1.0,
            1.0,
            0.5,
        )));
        bus.ingest(ContactEvent::Sustain(SustainEvent::new(
            ContactId(1),
            MaterialPairId::new(0, 0),
            3.0,
            1.0,
            0.5,
        )));
        assert_eq!(bus.sustains().len(), 1);
        assert!((bus.sustains()[0].tangential_speed - 3.0).abs() < 1e-6);
    }

    #[test]
    fn separation_dedup() {
        let mut bus = ContactEventBus::new(4, 4, MergeConfig::default());
        bus.begin_block();
        bus.ingest(ContactEvent::Separation(SeparationEvent {
            contact: ContactId(7),
        }));
        bus.ingest(ContactEvent::Separation(SeparationEvent {
            contact: ContactId(7),
        }));
        assert_eq!(bus.separations().len(), 1);
        assert!(bus.separated(ContactId(7)));
    }
}
