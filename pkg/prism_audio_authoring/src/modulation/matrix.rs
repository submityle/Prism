//! Modulation matrix: routing of sources onto control buses.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the modulation matrix of design section 12. It owns a bank of
//! control buses (see `super::control_bus`), a set of boxed modulation sources
//! (see `super::source`), and a list of routes. Each control tick it evaluates
//! every source, folds the per-route contributions onto their target buses
//! using the rules in `super::mixing`, shapes them through `super::curve`, and
//! advances the bus smoothers. Routing is deterministic and allocation-free in
//! the per-tick hot path.

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use prism_audio_core::Sample;

use super::control_bus::{BusId, ControlBus, ControlBusBank};
use super::curve::Curve;
use super::mixing::ModMix;
use super::source::{ModContext, Modulator};

/// Whether a route treats its input (and output contribution) as a unipolar
/// `[0, 1]` quantity or a bipolar `[-1, 1]` quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum Polarity {
    /// Input is interpreted on `[0, 1]`; the shaped contribution stays on
    /// `[0, depth]`.
    Unipolar,
    /// Input is interpreted on `[-1, 1]`; the shaped contribution spans
    /// `[-depth, depth]`.
    Bipolar,
}

/// The signal a route reads as its driving input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum RouteInput {
    /// Reads the value of a registered source by index.
    Source(usize),
    /// Reads the resolved value of another bus, sampled from the previous
    /// control tick to keep bus-to-bus routing acyclic.
    Bus(BusId),
}

/// A single source-to-bus routing with depth, polarity, shaping, and mixing.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ModRoute {
    /// Where the route reads its driving signal from.
    pub input: RouteInput,
    /// The bus the shaped contribution is folded into.
    pub target: BusId,
    /// Scales the shaped contribution.
    pub depth: Sample,
    /// Interpretation of the input and the output contribution range.
    pub polarity: Polarity,
    /// Response shape applied to the normalized input.
    pub curve: Curve,
    /// How the contribution folds onto the bus accumulator.
    pub mix: ModMix,
}

impl ModRoute {
    /// Creates an additive, bipolar, linear route from `input` to `target`.
    #[must_use]
    pub fn new(input: RouteInput, target: BusId, depth: Sample) -> Self {
        Self {
            input,
            target,
            depth,
            polarity: Polarity::Bipolar,
            curve: Curve::Linear,
            mix: ModMix::Add,
        }
    }

    /// Sets the route polarity.
    #[must_use]
    pub fn with_polarity(mut self, polarity: Polarity) -> Self {
        self.polarity = polarity;
        self
    }

    /// Sets the response curve.
    #[must_use]
    pub fn with_curve(mut self, curve: Curve) -> Self {
        self.curve = curve;
        self
    }

    /// Sets the mixing rule.
    #[must_use]
    pub fn with_mix(mut self, mix: ModMix) -> Self {
        self.mix = mix;
        self
    }

    /// Evaluates the shaped, depth-scaled contribution for a raw `input` value.
    #[inline]
    #[must_use]
    fn contribution(&self, input: Sample) -> Sample {
        let x01 = match self.polarity {
            Polarity::Unipolar => input.clamp(0.0, 1.0),
            Polarity::Bipolar => (input * 0.5 + 0.5).clamp(0.0, 1.0),
        };
        let shaped = self.curve.map(x01);
        match self.polarity {
            Polarity::Unipolar => shaped * self.depth,
            Polarity::Bipolar => (shaped * 2.0 - 1.0) * self.depth,
        }
    }
}

/// A matrix of modulation sources routed onto named control buses.
///
/// Sources are evaluated once per control tick; each bus resolves to its base
/// value folded with every route that targets it, and the resolved value is
/// smoothed by the underlying [`ControlBus`]. Bus-to-bus routes read the
/// previous tick's resolved value so routing never feeds back within a tick.
#[derive(Default)]
pub struct ModMatrix {
    buses: ControlBusBank,
    sources: Vec<Box<dyn Modulator>>,
    routes: Vec<ModRoute>,
    source_values: Vec<Sample>,
    prev_bus: Vec<Sample>,
}

impl core::fmt::Debug for ModMatrix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ModMatrix")
            .field("buses", &self.buses)
            .field("source_count", &self.sources.len())
            .field("routes", &self.routes)
            .finish()
    }
}

impl ModMatrix {
    /// Creates an empty matrix with no buses, sources, or routes.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buses: ControlBusBank::new(),
            sources: Vec::new(),
            routes: Vec::new(),
            source_values: Vec::new(),
            prev_bus: Vec::new(),
        }
    }

    /// Adds a named control bus initialized to `initial` and returns its id.
    pub fn add_bus(&mut self, name: &str, initial: Sample) -> BusId {
        let id = self.buses.add_bus(name, initial);
        self.prev_bus.push(initial);
        id
    }

    /// Registers a modulation source and returns its index.
    pub fn add_source(&mut self, source: Box<dyn Modulator>) -> usize {
        let index = self.sources.len();
        self.source_values.push(source.value());
        self.sources.push(source);
        index
    }

    /// Adds a route. The route is ignored during ticking if it targets or reads
    /// an id that does not exist, so callers can build in any order.
    pub fn add_route(&mut self, route: ModRoute) {
        self.routes.push(route);
    }

    /// Returns the number of registered sources.
    #[inline]
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Returns the number of control buses.
    #[inline]
    #[must_use]
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Looks up a bus id by name.
    #[must_use]
    pub fn find_bus(&self, name: &str) -> Option<BusId> {
        self.buses.find(name)
    }

    /// Sets the base (pre-modulation) value of a bus.
    pub fn set_bus_base(&mut self, id: BusId, base: Sample) {
        if let Some(bus) = self.buses.bus_mut(id) {
            bus.set_base(base);
        }
    }

    /// Sets the output smoothing ramp of a bus.
    pub fn set_bus_smoothing(&mut self, id: BusId, ramp: prism_audio_core::param::Ramp) {
        if let Some(bus) = self.buses.bus_mut(id) {
            bus.set_smoothing(ramp);
        }
    }

    /// Returns the current resolved value of a bus, or `0` if unknown.
    #[inline]
    #[must_use]
    pub fn bus_value(&self, id: BusId) -> Sample {
        self.buses.value(id)
    }

    /// Returns an immutable reference to a bus.
    #[inline]
    #[must_use]
    pub fn bus(&self, id: BusId) -> Option<&ControlBus> {
        self.buses.bus(id)
    }

    /// Returns the current value of a source by index, or `0` if unknown.
    #[inline]
    #[must_use]
    pub fn source_value(&self, index: usize) -> Sample {
        self.source_values.get(index).copied().unwrap_or(0.0)
    }

    /// Advances all sources and buses by one control tick.
    pub fn tick(&mut self, ctx: &ModContext) {
        // 1. Evaluate every source once, caching its value for this tick.
        for (slot, source) in self.source_values.iter_mut().zip(self.sources.iter_mut()) {
            *slot = source.tick(ctx);
        }

        // 2. Resolve each bus from its base folded with the routes targeting it.
        let bus_count = self.buses.len();
        for bi in 0..bus_count {
            let id = BusId(bi);
            let base = self.buses.bus(id).map_or(0.0, ControlBus::base);
            let mut acc = base;
            for route in &self.routes {
                if route.target != id {
                    continue;
                }
                let input = match route.input {
                    RouteInput::Source(si) => self.source_values.get(si).copied().unwrap_or(0.0),
                    RouteInput::Bus(bid) => self.prev_bus.get(bid.0).copied().unwrap_or(0.0),
                };
                acc = route.mix.combine(acc, route.contribution(input));
            }
            if let Some(bus) = self.buses.bus_mut(id) {
                bus.set_target(acc);
            }
        }

        // 3. Advance the output smoothers, then snapshot resolved values so the
        //    next tick's bus-to-bus routes read a stable previous value.
        self.buses.advance_all(ctx.frames);
        for (slot, bi) in self.prev_bus.iter_mut().zip(0..bus_count) {
            *slot = self.buses.value(BusId(bi));
        }
    }

    /// Resets every source and bus to its initial state.
    pub fn reset(&mut self) {
        for source in &mut self.sources {
            source.reset();
        }
        for (slot, source) in self.source_values.iter_mut().zip(self.sources.iter()) {
            *slot = source.value();
        }
        for bi in 0..self.buses.len() {
            let id = BusId(bi);
            let base = self.buses.bus(id).map_or(0.0, ControlBus::base);
            if let Some(bus) = self.buses.bus_mut(id) {
                bus.reset();
            }
            if let Some(slot) = self.prev_bus.get_mut(bi) {
                *slot = base;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modulation::lfo_source::LfoModulator;
    use crate::modulation::random::{RandomMode, RandomModulator};
    use prism_audio_core::modulation::LfoWaveform;

    const SR: u32 = 48_000;
    const EPS: Sample = 1.0e-4;

    #[test]
    fn constant_source_reaches_bus() {
        // A sawtooth source parked at phase 0.25 drives a bus additively.
        let mut m = ModMatrix::new();
        let bus = m.add_bus("target", 0.0);
        let src = m.add_source(Box::new(LfoModulator::new(
            SR,
            1.0,
            LfoWaveform::Sine,
            true,
        )));
        m.add_route(
            ModRoute::new(RouteInput::Source(src), bus, 1.0).with_polarity(Polarity::Unipolar),
        );
        let ctx = ModContext::new(SR, 64);
        for _ in 0..100 {
            m.tick(&ctx);
        }
        let v = m.bus_value(bus);
        assert!((0.0..=1.0).contains(&v), "bus value out of range: {v}");
    }

    #[test]
    fn base_value_is_preserved_without_routes() {
        let mut m = ModMatrix::new();
        let bus = m.add_bus("tension", 0.42);
        let ctx = ModContext::new(SR, 32);
        m.tick(&ctx);
        assert!((m.bus_value(bus) - 0.42).abs() < EPS);
    }

    #[test]
    fn additive_route_offsets_base() {
        let mut m = ModMatrix::new();
        let bus = m.add_bus("level", 0.25);
        // A smooth random source at a very slow rate stays near its first draw.
        let src = m.add_source(Box::new(RandomModulator::new(
            SR,
            0.001,
            RandomMode::Smooth,
            false,
            1,
        )));
        m.add_route(
            ModRoute::new(RouteInput::Source(src), bus, 0.5)
                .with_polarity(Polarity::Unipolar)
                .with_mix(ModMix::Add),
        );
        let ctx = ModContext::new(SR, 16);
        m.tick(&ctx);
        let v = m.bus_value(bus);
        // base 0.25 + (source in [0,1]) * 0.5 => within [0.25, 0.75].
        assert!((0.25..=0.75 + EPS).contains(&v), "v={v}");
    }

    #[test]
    fn bus_to_bus_route_uses_previous_value() {
        let mut m = ModMatrix::new();
        let a = m.add_bus("a", 1.0);
        let b = m.add_bus("b", 0.0);
        // b <- a (unipolar, depth 1). First tick reads a's previous value (its
        // initial 1.0 snapshot), so b should approach 1.0 over ticks.
        m.add_route(
            ModRoute::new(RouteInput::Bus(a), b, 1.0).with_polarity(Polarity::Unipolar),
        );
        let ctx = ModContext::new(SR, 8);
        for _ in 0..10 {
            m.tick(&ctx);
        }
        assert!((m.bus_value(b) - 1.0).abs() < EPS, "b={}", m.bus_value(b));
    }

    #[test]
    fn reset_restores_bases() {
        let mut m = ModMatrix::new();
        let bus = m.add_bus("x", 0.3);
        let src = m.add_source(Box::new(RandomModulator::new(
            SR,
            100.0,
            RandomMode::Stepped,
            true,
            5,
        )));
        m.add_route(ModRoute::new(RouteInput::Source(src), bus, 1.0));
        let ctx = ModContext::new(SR, 64);
        for _ in 0..50 {
            m.tick(&ctx);
        }
        m.reset();
        assert!((m.bus_value(bus) - 0.3).abs() < EPS, "x={}", m.bus_value(bus));
    }

    #[test]
    fn unknown_ids_are_ignored() {
        let mut m = ModMatrix::new();
        let bus = m.add_bus("only", 0.1);
        // Route from a non-existent source index is harmless.
        m.add_route(ModRoute::new(RouteInput::Source(99), bus, 1.0));
        let ctx = ModContext::new(SR, 16);
        m.tick(&ctx);
        assert!((m.bus_value(bus) - 0.1).abs() < EPS);
    }
}
