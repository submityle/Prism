//! Runtime *property binder* — the `CPU` reference resolver for authored
//! parameter bindings (design §8.3 *interaction* / *signal* categories, §30
//! parameter system).
//!
//! [`super::authoring`] owns the *static* half of the parameter contract: it
//! stores the exposed-parameter table ([`super::authoring::ExposedParam`]) and
//! the [`super::authoring::BindingSource`] enum that records *where* each
//! parameter's value comes from — a baked constant, a `gameplay` channel, a
//! material-instance channel, a timeline track, or a named curve. That module
//! is deliberately declarative: it says *that* parameter `n` is driven by
//! `gameplay` channel `c`, but it never reads a live channel or evaluates a
//! curve. Its documentation marks runtime binding as *pending the `GPU`
//! backend*.
//!
//! This module is the orthogonal *runtime* half: given a live
//! [`BinderContext`] holding the current `gameplay` / material / timeline
//! channel arrays, the authored curves, and the global signals
//! (`time` / `dt` / `frame_index` / a deterministic random stream), it turns a
//! [`BindingSource`] into a concrete [`ParamValue`]. It is a pure `CPU`
//! reference: a future `GPU` kernel resolves the same bindings on-device, and
//! this module is the bit-checkable oracle those results are validated against.
//!
//! The split is strict: authoring never resolves and the binder never mutates
//! the authored table. The two compose at a higher layer that walks the exposed
//! table and calls [`BinderContext::resolve_checked`] once per parameter.
//!
//! # Design category coverage
//!
//! * *Interaction* (§8.3): the `UserParam` / property-binder `DataInterface` —
//!   [`BindingSource::Gameplay`], [`BindingSource::Material`], and
//!   [`BindingSource::Timeline`] channels resolved against live arrays.
//! * *Signal* (§8.3): the `Global` signals — [`GlobalSignal::Time`],
//!   [`GlobalSignal::DeltaTime`], [`GlobalSignal::FrameIndex`], and a
//!   decorrelated [`GlobalSignal::Random`] stream.
//!
//! # Determinism
//!
//! Every routine uses only ordinary `f32` arithmetic plus `abs` (for the
//! tolerance guards). No transcendental function is called, and the single
//! reciprocal needed to convert a per-frame delta into a per-second rate is a
//! scalar `1.0 / dt` fed into [`super::Vec3::scale`] behind an [`EPS`] guard
//! (there is no vector division). The random stream reuses the shared hash
//! `RNG` from [`super::determinism`], so a seed plus a frame index reproduces
//! every draw. Nothing here can panic on hostile input: channel look-ups use
//! `get` rather than indexing, and every fallible path returns a diagnosable
//! [`BindError`] instead of unwinding.

use alloc::vec::Vec;

use super::authoring::{BindingSource, ParamType, ParamValue};
use super::curves::Curve;
use super::determinism::{RngKey, StreamId};
use super::Vec3;

/// Absolute tolerance for the `f32` equality and near-zero decisions this
/// module makes (the `dt` reciprocal guard and value comparisons in tests).
///
/// Bare `==` / `!=` on `f32` is avoided throughout; scalar equality is decided
/// through [`approx_eq`] and near-zero denominators are rejected against this
/// tolerance so a reciprocal never divides by (near) zero.
pub const EPS: f32 = 1e-6;

/// The value [`BinderContext::resolve`] returns for an unresolvable binding
/// (an out-of-bounds channel) when the caller has not supplied an expected
/// type: a neutral zero scalar.
///
/// The *checked* entry points ([`BinderContext::resolve_checked`],
/// [`BinderContext::try_resolve`]) instead surface a [`BindError`] so a caller
/// that needs to distinguish "resolved to zero" from "could not resolve" can.
pub const FALLBACK: ParamValue = ParamValue::Scalar(0.0);

/// Returns `true` when two scalars are equal within [`EPS`].
///
/// Used instead of a bare `f32` equality so resolution decisions never rely on
/// exact bit equality of interpolated or re-derived values.
#[must_use]
pub fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= EPS
}

/// The declared [`ParamType`] a concrete [`ParamValue`] carries.
///
/// The mapping is total and lossless: it is the runtime dual of the authored
/// [`ParamType`] tag and is what type validation compares an expected type
/// against.
#[must_use]
pub fn value_type(value: ParamValue) -> ParamType {
    match value {
        ParamValue::Scalar(_) => ParamType::Scalar,
        ParamValue::Vec3(_) => ParamType::Vec3,
        ParamValue::Color(_) => ParamType::Color,
        ParamValue::Int(_) => ParamType::Int,
        ParamValue::Bool(_) => ParamType::Bool,
    }
}

/// A diagnosable failure raised while resolving a binding (design §30).
///
/// The enum is intentionally *fieldless* so it derives [`Eq`] and [`Hash`] and
/// can be collected into a set or used as a map key when a validator tallies
/// binding problems across an effect. Each variant names *why* resolution
/// failed without embedding the offending value, keeping the type cheap to
/// compare and hash. Resolution never panics; every fallible path returns one
/// of these.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BindError {
    /// An external variant referenced a channel index past the end of the
    /// matching live array (a stale asset against a shorter runtime table).
    ChannelOutOfBounds,
    /// The resolved value's [`ParamType`] did not match the type the caller
    /// declared it expected (an authoring/runtime type drift).
    TypeMismatch,
    /// A [`BindingSource::Curve`] referenced a curve slot that does not exist.
    CurveOutOfBounds,
}

/// A `Global` signal that is not stored in any channel array but derived from
/// the frame's clock or a deterministic random stream (design §8.3 *signal*).
///
/// The enum is fieldless-of-`f32` (its only payloads are a [`StreamId`] and a
/// draw index), so it derives [`Eq`] and [`Hash`] and can key a cache of
/// per-frame signal values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GlobalSignal {
    /// The effect-local elapsed time in seconds, as a [`ParamValue::Scalar`].
    Time,
    /// The current frame's timestep in seconds, as a [`ParamValue::Scalar`].
    DeltaTime,
    /// The simulation frame index, as a [`ParamValue::Int`].
    FrameIndex,
    /// A uniform random draw in `[0, 1)` from the named `stream` at draw
    /// `index`, as a [`ParamValue::Scalar`]. The draw is reproduced by the
    /// context's seed and frame index, so the same request always yields the
    /// same value within a frame.
    Random(StreamId, u32),
}

/// The live environment a [`BindingSource`] resolves against (design §8.3, §30).
///
/// It holds the three externally driven channel arrays (`gameplay`, material,
/// timeline), the authored curve table, and the global signals. Because it
/// stores `f32` clock fields and [`ParamValue`] payloads it derives only
/// [`PartialEq`] (never [`Eq`]); compare scalar payloads through [`approx_eq`].
///
/// Channels are addressed by the same opaque `u32` index the authoring layer
/// baked into each [`BindingSource`] variant. Look-ups use `get`, so an index
/// past the end of an array is a diagnosable [`BindError`] (or the neutral
/// [`FALLBACK`]) rather than a panic.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BinderContext {
    /// Live values pushed by `gameplay` code, addressed by
    /// [`BindingSource::Gameplay`] channel index.
    gameplay: Vec<ParamValue>,
    /// Live values pushed by a material instance, addressed by
    /// [`BindingSource::Material`] channel index.
    material: Vec<ParamValue>,
    /// Live values pushed by a sequencer / timeline track, addressed by
    /// [`BindingSource::Timeline`] channel index.
    timeline: Vec<ParamValue>,
    /// Authored scalar curves, addressed by [`BindingSource::Curve`] channel
    /// index and evaluated at [`BinderContext::time`].
    curves: Vec<Curve>,
    /// Effect-local elapsed time in seconds (the abscissa curves sample at).
    time: f32,
    /// The current frame's timestep in seconds.
    dt: f32,
    /// The simulation frame index (folded into random draws for per-frame
    /// decorrelation).
    frame_index: u32,
    /// The effect-wide deterministic random seed.
    seed: u32,
}

impl BinderContext {
    /// Builds a context with the global signals set and every channel array
    /// empty. Populate channels and curves with the `with_*` builders.
    #[must_use]
    pub fn new(time: f32, dt: f32, frame_index: u32, seed: u32) -> Self {
        Self {
            gameplay: Vec::new(),
            material: Vec::new(),
            timeline: Vec::new(),
            curves: Vec::new(),
            time,
            dt,
            frame_index,
            seed,
        }
    }

    /// Replaces the `gameplay` channel array (builder form).
    #[must_use]
    pub fn with_gameplay(mut self, channels: Vec<ParamValue>) -> Self {
        self.gameplay = channels;
        self
    }

    /// Replaces the material channel array (builder form).
    #[must_use]
    pub fn with_material(mut self, channels: Vec<ParamValue>) -> Self {
        self.material = channels;
        self
    }

    /// Replaces the timeline channel array (builder form).
    #[must_use]
    pub fn with_timeline(mut self, channels: Vec<ParamValue>) -> Self {
        self.timeline = channels;
        self
    }

    /// Replaces the authored curve table (builder form).
    #[must_use]
    pub fn with_curves(mut self, curves: Vec<Curve>) -> Self {
        self.curves = curves;
        self
    }

    /// The effect-local elapsed time in seconds.
    #[must_use]
    pub fn time(&self) -> f32 {
        self.time
    }

    /// The current frame's timestep in seconds.
    #[must_use]
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// The simulation frame index.
    #[must_use]
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// The effect-wide deterministic random seed.
    #[must_use]
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Advances the clock to a new frame, updating `time`, `dt`, and the frame
    /// index so a caller can drive the binder across a simulation step.
    pub fn advance(&mut self, new_time: f32, new_dt: f32, new_frame: u32) {
        self.time = new_time;
        self.dt = new_dt;
        self.frame_index = new_frame;
    }

    /// Resolves a [`BindingSource`] to a concrete [`ParamValue`], never failing.
    ///
    /// A [`BindingSource::Constant`] returns its baked value directly. An
    /// external variant looks its channel up in the matching array; a
    /// [`BindingSource::Curve`] evaluates the authored curve at
    /// [`BinderContext::time`]. Any out-of-bounds channel yields the neutral
    /// [`FALLBACK`] scalar rather than panicking — use [`Self::try_resolve`]
    /// when the caller must distinguish a genuine zero from a missing channel.
    #[must_use]
    pub fn resolve(&self, source: BindingSource) -> ParamValue {
        self.try_resolve(source).unwrap_or(FALLBACK)
    }

    /// Resolves a [`BindingSource`], returning a [`BindError`] instead of a
    /// fallback when a channel or curve slot is out of bounds.
    ///
    /// This does *not* type-check the result; layer [`Self::resolve_checked`]
    /// on top when the parameter's declared [`ParamType`] must be enforced.
    pub fn try_resolve(&self, source: BindingSource) -> Result<ParamValue, BindError> {
        match source {
            BindingSource::Constant(value) => Ok(value),
            BindingSource::Gameplay(channel) => Self::lookup(&self.gameplay, channel),
            BindingSource::Material(channel) => Self::lookup(&self.material, channel),
            BindingSource::Timeline(channel) => Self::lookup(&self.timeline, channel),
            BindingSource::Curve(channel) => self.sample_curve(channel).map(ParamValue::Scalar),
        }
    }

    /// Resolves a [`BindingSource`] and validates that the result matches the
    /// declared `expected` [`ParamType`] (design §30 type contract).
    ///
    /// Returns [`BindError::ChannelOutOfBounds`] / [`BindError::CurveOutOfBounds`]
    /// when the source cannot be resolved, or [`BindError::TypeMismatch`] when
    /// the resolved value's type disagrees with `expected`.
    pub fn resolve_checked(
        &self,
        source: BindingSource,
        expected: ParamType,
    ) -> Result<ParamValue, BindError> {
        let value = self.try_resolve(source)?;
        if value_type(value) == expected {
            Ok(value)
        } else {
            Err(BindError::TypeMismatch)
        }
    }

    /// Resolves a binding that must be a [`ParamValue::Vec3`] and returns it as
    /// a shared [`super::Vec3`].
    ///
    /// A convenience over [`Self::resolve_checked`] for the common case of
    /// binding a vector parameter (a spawn offset, a force direction).
    pub fn resolve_vec3(&self, source: BindingSource) -> Result<Vec3, BindError> {
        match self.resolve_checked(source, ParamType::Vec3)? {
            ParamValue::Vec3(a) => Ok(Vec3::new(a[0], a[1], a[2])),
            _ => Err(BindError::TypeMismatch),
        }
    }

    /// Resolves a [`GlobalSignal`] to a concrete [`ParamValue`] (design §8.3
    /// *signal*). This never fails: every global is always available.
    #[must_use]
    pub fn resolve_global(&self, signal: GlobalSignal) -> ParamValue {
        match signal {
            GlobalSignal::Time => ParamValue::Scalar(self.time),
            GlobalSignal::DeltaTime => ParamValue::Scalar(self.dt),
            GlobalSignal::FrameIndex => ParamValue::Int(self.frame_index as i32),
            GlobalSignal::Random(stream, index) => {
                ParamValue::Scalar(self.random_unit(stream, index))
            }
        }
    }

    /// Resolves a [`GlobalSignal`] and validates its declared `expected`
    /// [`ParamType`], mirroring [`Self::resolve_checked`] for globals.
    pub fn resolve_global_checked(
        &self,
        signal: GlobalSignal,
        expected: ParamType,
    ) -> Result<ParamValue, BindError> {
        let value = self.resolve_global(signal);
        if value_type(value) == expected {
            Ok(value)
        } else {
            Err(BindError::TypeMismatch)
        }
    }

    /// A uniform random draw in `[0, 1)` from `stream` at draw `index`.
    ///
    /// Reuses the shared stateless hash `RNG` from [`super::determinism`]: the
    /// draw is keyed by a fixed global particle id (`0`, since a global signal
    /// is not per-particle), the context seed, the stream namespace, and the
    /// frame index, so the same request reproduces the same value.
    #[must_use]
    pub fn random_unit(&self, stream: StreamId, index: u32) -> f32 {
        RngKey::for_stream(0, self.seed, stream, self.frame_index).unit_f32(index)
    }

    /// Samples the authored curve at `channel` at the current [`Self::time`].
    ///
    /// Returns [`BindError::CurveOutOfBounds`] when no curve occupies the slot.
    /// An empty curve samples to `0.0` (the curve layer's own convention).
    pub fn sample_curve(&self, channel: u32) -> Result<f32, BindError> {
        self.curves
            .get(channel as usize)
            .map(|curve| curve.sample(self.time))
            .ok_or(BindError::CurveOutOfBounds)
    }

    /// Converts a per-frame positional `delta` into a per-second velocity by
    /// dividing by `dt`.
    ///
    /// There is no vector division in the shared math API, so the reciprocal is
    /// computed as a scalar `1.0 / dt` behind an [`EPS`] guard and applied with
    /// [`super::Vec3::scale`]. A near-zero `dt` (a paused frame) yields the zero
    /// vector rather than a non-finite result.
    #[must_use]
    pub fn velocity_from_delta(&self, delta: Vec3) -> Vec3 {
        if self.dt.abs() > EPS {
            delta.scale(1.0 / self.dt)
        } else {
            Vec3::ZERO
        }
    }

    /// Looks a channel index up in an array, mapping an out-of-bounds index to
    /// [`BindError::ChannelOutOfBounds`]. Uses `get`, never indexing.
    fn lookup(channels: &[ParamValue], channel: u32) -> Result<ParamValue, BindError> {
        channels
            .get(channel as usize)
            .copied()
            .ok_or(BindError::ChannelOutOfBounds)
    }
}

#[cfg(test)]
mod tests {
    use super::super::curves::{InterpolationMode, Keyframe};
    use super::*;

    fn ctx() -> BinderContext {
        BinderContext::new(0.5, 1.0 / 60.0, 7, 0xABCD_1234)
            .with_gameplay(alloc::vec![
                ParamValue::Scalar(3.5),
                ParamValue::Vec3([1.0, 2.0, 3.0]),
            ])
            .with_material(alloc::vec![ParamValue::Color([0.1, 0.2, 0.3, 1.0])])
            .with_timeline(alloc::vec![ParamValue::Int(42), ParamValue::Bool(true)])
            .with_curves(alloc::vec![Curve::from_keys(
                InterpolationMode::Linear,
                alloc::vec![Keyframe::new(0.0, 0.0), Keyframe::new(1.0, 10.0)],
            )])
    }

    #[test]
    fn constant_passes_through_unchanged() {
        let c = ctx();
        let src = BindingSource::Constant(ParamValue::Scalar(9.0));
        match c.resolve(src) {
            ParamValue::Scalar(v) => assert!(approx_eq(v, 9.0)),
            other => panic!("expected scalar, got {other:?}"),
        }
        // Checked resolution of a matching type succeeds.
        assert!(c.resolve_checked(src, ParamType::Scalar).is_ok());
    }

    #[test]
    fn gameplay_channels_resolve_by_index() {
        let c = ctx();
        match c.resolve(BindingSource::Gameplay(0)) {
            ParamValue::Scalar(v) => assert!(approx_eq(v, 3.5)),
            other => panic!("expected scalar, got {other:?}"),
        }
        match c.resolve(BindingSource::Gameplay(1)) {
            ParamValue::Vec3(a) => {
                assert!(approx_eq(a[0], 1.0));
                assert!(approx_eq(a[1], 2.0));
                assert!(approx_eq(a[2], 3.0));
            }
            other => panic!("expected vec3, got {other:?}"),
        }
    }

    #[test]
    fn material_channel_resolves() {
        let c = ctx();
        match c.resolve_checked(BindingSource::Material(0), ParamType::Color) {
            Ok(ParamValue::Color(a)) => {
                assert!(approx_eq(a[0], 0.1));
                assert!(approx_eq(a[3], 1.0));
            }
            other => panic!("expected color, got {other:?}"),
        }
    }

    #[test]
    fn timeline_channels_resolve() {
        let c = ctx();
        assert_eq!(c.resolve(BindingSource::Timeline(0)), ParamValue::Int(42));
        assert_eq!(
            c.resolve(BindingSource::Timeline(1)),
            ParamValue::Bool(true)
        );
    }

    #[test]
    fn curve_evaluates_at_current_time() {
        // Linear 0..10 over t in 0..1; context time is 0.5 -> midpoint 5.0.
        let c = ctx();
        match c.resolve(BindingSource::Curve(0)) {
            ParamValue::Scalar(v) => assert!(approx_eq(v, 5.0)),
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn curve_tracks_advancing_time() {
        let mut c = ctx();
        c.advance(0.25, 1.0 / 60.0, 8);
        let v = c.sample_curve(0).expect("curve slot exists");
        assert!(approx_eq(v, 2.5));
    }

    #[test]
    fn type_mismatch_is_diagnosable() {
        let c = ctx();
        // Gameplay 0 is a scalar; asking for a vec3 must not panic.
        assert_eq!(
            c.resolve_checked(BindingSource::Gameplay(0), ParamType::Vec3),
            Err(BindError::TypeMismatch)
        );
    }

    #[test]
    fn out_of_bounds_channel_falls_back_without_panic() {
        let c = ctx();
        // Infallible resolve yields the neutral fallback...
        assert_eq!(c.resolve(BindingSource::Gameplay(99)), FALLBACK);
        // ...while the checked path surfaces the reason.
        assert_eq!(
            c.try_resolve(BindingSource::Material(5)),
            Err(BindError::ChannelOutOfBounds)
        );
        assert_eq!(
            c.try_resolve(BindingSource::Curve(3)),
            Err(BindError::CurveOutOfBounds)
        );
    }

    #[test]
    fn global_time_dt_frame_resolve() {
        let c = ctx();
        match c.resolve_global(GlobalSignal::Time) {
            ParamValue::Scalar(v) => assert!(approx_eq(v, 0.5)),
            other => panic!("expected scalar, got {other:?}"),
        }
        match c.resolve_global(GlobalSignal::DeltaTime) {
            ParamValue::Scalar(v) => assert!(approx_eq(v, 1.0 / 60.0)),
            other => panic!("expected scalar, got {other:?}"),
        }
        assert_eq!(
            c.resolve_global(GlobalSignal::FrameIndex),
            ParamValue::Int(7)
        );
    }

    #[test]
    fn global_random_is_in_unit_range_and_deterministic() {
        let c = ctx();
        let signal = GlobalSignal::Random(StreamId::Color, 0);
        let a = match c.resolve_global(signal) {
            ParamValue::Scalar(v) => v,
            other => panic!("expected scalar, got {other:?}"),
        };
        assert!(a >= 0.0 && a < 1.0);
        // Same context, same request -> identical draw.
        let b = match c.resolve_global(signal) {
            ParamValue::Scalar(v) => v,
            other => panic!("expected scalar, got {other:?}"),
        };
        assert!(approx_eq(a, b));
    }

    #[test]
    fn global_random_decorrelates_streams_and_indices() {
        let c = ctx();
        let x = c.random_unit(StreamId::InitialPosition, 0);
        let y = c.random_unit(StreamId::InitialVelocity, 0);
        let z = c.random_unit(StreamId::InitialPosition, 1);
        // Distinct streams / draw indices should not collide.
        assert!(!approx_eq(x, y));
        assert!(!approx_eq(x, z));
    }

    #[test]
    fn global_random_type_check() {
        let c = ctx();
        assert_eq!(
            c.resolve_global_checked(GlobalSignal::Time, ParamType::Int),
            Err(BindError::TypeMismatch)
        );
        assert!(c
            .resolve_global_checked(GlobalSignal::FrameIndex, ParamType::Int)
            .is_ok());
    }

    #[test]
    fn resolve_vec3_converts_and_rejects() {
        let c = ctx();
        let v = c
            .resolve_vec3(BindingSource::Gameplay(1))
            .expect("vec3 channel");
        assert!(approx_eq(v.x, 1.0));
        assert!(approx_eq(v.z, 3.0));
        // A scalar channel cannot be read as a vec3.
        assert_eq!(
            c.resolve_vec3(BindingSource::Gameplay(0)),
            Err(BindError::TypeMismatch)
        );
    }

    #[test]
    fn velocity_from_delta_guards_zero_dt() {
        let c = ctx();
        let per_second = c.velocity_from_delta(Vec3::new(0.0, 1.0 / 60.0, 0.0));
        // delta / (1/60) == 60 * delta -> y component is 1.0.
        assert!(approx_eq(per_second.y, 1.0));

        let paused = BinderContext::new(0.0, 0.0, 0, 1);
        assert_eq!(paused.velocity_from_delta(Vec3::splat(5.0)), Vec3::ZERO);
    }

    #[test]
    fn bind_error_is_hashable_and_eq() {
        // Fieldless BindError derives Eq + Hash: usable as a dedup key.
        let mut seen: Vec<BindError> = Vec::new();
        for e in [
            BindError::ChannelOutOfBounds,
            BindError::ChannelOutOfBounds,
            BindError::TypeMismatch,
        ] {
            if !seen.contains(&e) {
                seen.push(e);
            }
        }
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn value_type_round_trips_every_variant() {
        assert_eq!(value_type(ParamValue::Scalar(0.0)), ParamType::Scalar);
        assert_eq!(value_type(ParamValue::Vec3([0.0; 3])), ParamType::Vec3);
        assert_eq!(value_type(ParamValue::Color([0.0; 4])), ParamType::Color);
        assert_eq!(value_type(ParamValue::Int(0)), ParamType::Int);
        assert_eq!(value_type(ParamValue::Bool(false)), ParamType::Bool);
    }
}
