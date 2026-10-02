//! Named control buses: aggregation and broadcast of scalar control signals.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the control-bus requirement of design section 12. A control bus
//! is a named scalar (such as "tension" or "underwater amount") that many
//! parameters subscribe to; it carries a base value and a smoothed resolved
//! value. The modulation matrix (see `super::matrix`) writes bus targets each
//! control tick, and output smoothing uses `prism_audio_core::param`.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::param::{Ramp, Smoothed};
use prism_audio_core::Sample;

/// Stable identifier for a bus within a single [`ControlBusBank`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BusId(pub usize);

/// A single named control signal.
///
/// The bus holds a `base` value (for example the direct output of an RTPC
/// mapping) and a smoothed resolved value. Each control tick a caller sets a
/// new target with [`ControlBus::set_target`] and then advances the smoother
/// with [`ControlBus::advance`]; subscribers read [`ControlBus::value`].
#[derive(Debug, Clone)]
pub struct ControlBus {
    base: Sample,
    value: Smoothed,
    ramp: Ramp,
}

impl ControlBus {
    /// Creates a bus settled at `initial` with immediate smoothing.
    #[must_use]
    pub fn new(initial: Sample) -> Self {
        Self {
            base: initial,
            value: Smoothed::new(initial),
            ramp: Ramp::Immediate,
        }
    }

    /// Returns the base value (the pre-modulation input level).
    #[inline]
    #[must_use]
    pub fn base(&self) -> Sample {
        self.base
    }

    /// Sets the base value fed into the modulation accumulation.
    #[inline]
    pub fn set_base(&mut self, base: Sample) {
        self.base = base;
    }

    /// Returns the current smoothed, resolved value.
    #[inline]
    #[must_use]
    pub fn value(&self) -> Sample {
        self.value.current()
    }

    /// Sets the smoothing ramp applied when the resolved target changes.
    #[inline]
    pub fn set_smoothing(&mut self, ramp: Ramp) {
        self.ramp = ramp;
    }

    /// Sets the resolved target the bus should glide toward.
    #[inline]
    pub fn set_target(&mut self, target: Sample) {
        self.value.set_target(target, self.ramp);
    }

    /// Advances the output smoother by `frames` samples.
    #[inline]
    pub fn advance(&mut self, frames: u32) {
        for _ in 0..frames {
            self.value.next_sample();
        }
    }

    /// Resets the resolved value to the base value instantly.
    #[inline]
    pub fn reset(&mut self) {
        self.value = Smoothed::new(self.base);
    }
}

/// A fixed collection of named control buses addressed by [`BusId`].
///
/// Bus creation allocates and runs off the audio thread; lookups and value
/// reads are allocation-free and safe on the real-time thread.
#[derive(Debug, Clone, Default)]
pub struct ControlBusBank {
    buses: Vec<ControlBus>,
    names: Vec<String>,
}

impl ControlBusBank {
    /// Creates an empty bank.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buses: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Adds a named bus initialized to `initial` and returns its id.
    pub fn add_bus(&mut self, name: &str, initial: Sample) -> BusId {
        let id = BusId(self.buses.len());
        self.buses.push(ControlBus::new(initial));
        self.names.push(String::from(name));
        id
    }

    /// Returns the number of buses.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.buses.len()
    }

    /// Returns `true` if the bank holds no buses.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buses.is_empty()
    }

    /// Finds a bus id by name, if present.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<BusId> {
        self.names.iter().position(|n| n == name).map(BusId)
    }

    /// Returns an immutable reference to a bus.
    #[inline]
    #[must_use]
    pub fn bus(&self, id: BusId) -> Option<&ControlBus> {
        self.buses.get(id.0)
    }

    /// Returns a mutable reference to a bus.
    #[inline]
    pub fn bus_mut(&mut self, id: BusId) -> Option<&mut ControlBus> {
        self.buses.get_mut(id.0)
    }

    /// Returns the current value of a bus, or `0` if the id is unknown.
    #[inline]
    #[must_use]
    pub fn value(&self, id: BusId) -> Sample {
        self.buses.get(id.0).map_or(0.0, ControlBus::value)
    }

    /// Advances every bus smoother by `frames` samples.
    pub fn advance_all(&mut self, frames: u32) {
        for b in &mut self.buses {
            b.advance(frames);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-6;

    #[test]
    fn add_and_lookup() {
        let mut bank = ControlBusBank::new();
        let tension = bank.add_bus("tension", 0.0);
        let underwater = bank.add_bus("underwater", 1.0);
        assert_eq!(bank.find("tension"), Some(tension));
        assert_eq!(bank.find("underwater"), Some(underwater));
        assert_eq!(bank.find("missing"), None);
        assert_eq!(bank.len(), 2);
    }

    #[test]
    fn immediate_target_applies_at_once() {
        let mut bus = ControlBus::new(0.0);
        bus.set_target(0.5);
        bus.advance(1);
        assert!((bus.value() - 0.5).abs() < EPS);
    }

    #[test]
    fn linear_smoothing_glides() {
        let mut bus = ControlBus::new(0.0);
        bus.set_smoothing(Ramp::Linear { samples: 4 });
        bus.set_target(1.0);
        bus.advance(2);
        let mid = bus.value();
        assert!(mid > 0.0 && mid < 1.0, "mid={mid}");
        bus.advance(2);
        assert!((bus.value() - 1.0).abs() < EPS);
    }

    #[test]
    fn reset_restores_base() {
        let mut bus = ControlBus::new(0.25);
        bus.set_target(0.9);
        bus.advance(1);
        bus.reset();
        assert!((bus.value() - 0.25).abs() < EPS);
    }

    #[test]
    fn bank_value_defaults_zero_for_unknown() {
        let bank = ControlBusBank::new();
        assert!(bank.value(BusId(99)).abs() < EPS);
        assert!(bank.is_empty());
    }
}
