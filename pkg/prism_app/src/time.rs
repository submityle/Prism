//! The app-driven time context: clocks and the per-frame advancement step
//! (design §8, §25.1).
//!
//! `prism_time` is the **single source of truth** for the clocks and the
//! fixed-step accumulator (design §25.1: *"the fixed-step accumulator is
//! implemented in `prism_time` … App only drives and calls it"*). This module
//! holds the two App-level resources that let the frame loop *drive* those
//! clocks:
//!
//! - [`EngineClocks`]: the [`Clocks`] bundle
//!   ([`Time<Real>`](prism_time::Time), [`Time<Virtual>`](prism_time::Time),
//!   [`Time<Fixed>`](prism_time::Time), plus the context-less default clock)
//!   stored as a world resource.
//! - [`TimeUpdateStrategy`]: how the real clock advances each frame — from the
//!   platform monotonic clock (default, for live runs) or by a caller-supplied
//!   delta (for deterministic headless / server / test runs, design §15).
//!
//! # Why a newtype wrapper
//!
//! The [`Resource`] marker trait lives in `prism_ecs` and
//! [`Clocks`] lives in `prism_time`; both are foreign to
//! `prism_app`, so the orphan rule forbids `impl Resource for Clocks` here.
//! [`EngineClocks`] is a thin newtype that owns the bundle and derefs to it, so
//! every `Clocks` accessor is available on the resource directly. This also
//! keeps the clean layering intact: `prism_time` (a lower runtime-service
//! layer, design §4) never has to depend on `prism_ecs`.

use core::ops::{Deref, DerefMut};

use prism_ecs::resource::Resource;
use prism_time::{Clocks, Duration};

/// The engine's [`Clocks`] bundle, stored as a world
/// resource.
///
/// Dereferences to [`Clocks`], so a system reads the active
/// delta with `clocks.delta_secs()` or a specific context with
/// `clocks.fixed()`, `clocks.virtual_time()`, `clocks.real()`, exactly as on
/// the bare bundle. The App points the default context at
/// [`Virtual`](prism_time::Virtual) during the variable-step phases and at
/// [`Fixed`](prism_time::Fixed) inside the fixed loop (see
/// [`crate::fixed`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineClocks(pub Clocks);

impl Resource for EngineClocks {}

impl EngineClocks {
    /// A fresh bundle with the default clock pointing at
    /// [`Virtual`](prism_time::Virtual).
    #[inline]
    pub fn new() -> Self {
        Self(Clocks::new())
    }

    /// Wrap an existing [`Clocks`] bundle (e.g. one
    /// pre-configured with a non-default fixed timestep).
    #[inline]
    pub fn from_clocks(clocks: Clocks) -> Self {
        Self(clocks)
    }
}

impl Deref for EngineClocks {
    type Target = Clocks;
    #[inline]
    fn deref(&self) -> &Clocks {
        &self.0
    }
}

impl DerefMut for EngineClocks {
    #[inline]
    fn deref_mut(&mut self) -> &mut Clocks {
        &mut self.0
    }
}

/// How [`advance_time`] moves the real clock forward each frame.
///
/// The real clock is the root input: virtual time is derived from the real
/// delta (clamped + scaled) and the fixed accumulator is fed from the virtual
/// delta. Controlling the real delta therefore controls the whole chain, which
/// is what makes deterministic replay (design §15) possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TimeUpdateStrategy {
    /// Advance from the platform monotonic clock (`Instant::now`). The default
    /// for live, wall-clock-paced runs. Requires the `std` feature.
    #[default]
    Automatic,
    /// Advance the real clock by a fixed, caller-supplied delta every frame.
    /// Makes the frame loop independent of wall-clock timing, so headless /
    /// server / test runs step deterministically.
    ManualDelta(Duration),
}

impl Resource for TimeUpdateStrategy {}

/// Advance the clocks one frame: real → virtual → fixed accumulator.
///
/// Called once at the very start of every [`SubApp::update`](crate::sub_app::SubApp::update),
/// before the [`First`](crate::schedule::First) phase. The sequence (design §8):
///
/// 1. Advance [`Time<Real>`](prism_time::Time) per the [`TimeUpdateStrategy`].
/// 2. Advance [`Time<Virtual>`](prism_time::Time) from the real delta — this is
///    where the max-delta clamp (the spiral-of-death guard) and time dilation
///    apply.
/// 3. Feed the virtual delta into the [`Time<Fixed>`](prism_time::Time)
///    accumulator (bounded by `max_substeps`, so a hitch cannot queue unbounded
///    fixed steps).
///
/// The default context is then pointed at [`Virtual`](prism_time::Virtual) so
/// the variable-step phases read virtual time; [`crate::fixed::run_fixed_main_loop`]
/// swaps it to [`Fixed`](prism_time::Fixed) around each fixed step.
///
/// A world without an [`EngineClocks`] resource (e.g. a secondary sub-app that
/// does not own a clock) is a no-op, so this is safe to call unconditionally.
pub fn advance_time(world: &mut prism_ecs::world::World) {
    use prism_time::DefaultSource;

    if world.get_resource::<EngineClocks>().is_none() {
        return;
    }

    let strategy = world
        .get_resource::<TimeUpdateStrategy>()
        .copied()
        .unwrap_or_default();

    let clocks = world.resource_mut::<EngineClocks>();

    match strategy {
        TimeUpdateStrategy::Automatic => advance_real_automatic(clocks),
        TimeUpdateStrategy::ManualDelta(delta) => clocks.real_mut().update_with_delta(delta),
    }

    let real_delta = clocks.real().delta();
    clocks.virtual_time_mut().advance_by(real_delta);
    let virtual_delta = clocks.virtual_time().delta();
    clocks.fixed_mut().accumulate(virtual_delta);

    clocks.set_source(DefaultSource::Virtual);
}

/// Advance the real clock from the platform monotonic clock when `std` is
/// available; without `std` there is no `Instant::now`, so the automatic
/// strategy degenerates to a zero step and callers must drive time with
/// [`TimeUpdateStrategy::ManualDelta`] instead (documented, not faked).
#[inline]
fn advance_real_automatic(clocks: &mut EngineClocks) {
    #[cfg(feature = "std")]
    {
        clocks.real_mut().update();
    }
    #[cfg(not(feature = "std"))]
    {
        clocks.real_mut().update_with_delta(Duration::ZERO);
    }
}
