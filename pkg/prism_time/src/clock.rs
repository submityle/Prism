//! The default-context switch.
//!
//! [`Clocks`] bundles the three clocks and a context-less default [`Time<()>`].
//! The app points the default at [`Virtual`] during the variable update and at
//! [`Fixed`] around the fixed-update schedule (mirroring Bevy's swap of the
//! generic `Time` resource), so a system can read `delta_secs()` and get the
//! value for whichever phase is running, without naming a context.

use crate::{Fixed, Real, Time, TimeKind, Virtual};

impl Time<()> {
    /// Overwrite the shared accessors from another clock's current readings.
    #[inline]
    fn copy_readings_from<T: TimeKind>(&mut self, src: &Time<T>) {
        self.delta = src.delta;
        self.elapsed = src.elapsed;
        self.delta_secs = src.delta_secs;
        self.delta_secs_f64 = src.delta_secs_f64;
        self.elapsed_secs = src.elapsed_secs;
        self.elapsed_secs_f64 = src.elapsed_secs_f64;
    }
}

/// Which context the default [`Time<()>`] currently mirrors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DefaultSource {
    /// The default clock mirrors [`Time<Virtual>`] (variable update).
    #[default]
    Virtual,
    /// The default clock mirrors [`Time<Fixed>`] (fixed update).
    Fixed,
}

/// Bundle of the three clocks plus the context-less default clock.
#[derive(Clone, Copy, Debug)]
pub struct Clocks {
    real: Time<Real>,
    virtual_time: Time<Virtual>,
    fixed: Time<Fixed>,
    default: Time<()>,
    source: DefaultSource,
}

impl Default for Clocks {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Clocks {
    /// Create a fresh bundle with the default pointing at [`Virtual`].
    #[inline]
    pub fn new() -> Self {
        Self {
            real: Time::<Real>::new(),
            virtual_time: Time::<Virtual>::new(),
            fixed: Time::<Fixed>::new(),
            default: Time::default(),
            source: DefaultSource::Virtual,
        }
    }

    /// The real clock.
    #[inline]
    pub fn real(&self) -> &Time<Real> {
        &self.real
    }
    /// Mutable access to the real clock.
    #[inline]
    pub fn real_mut(&mut self) -> &mut Time<Real> {
        &mut self.real
    }

    /// The virtual clock.
    #[inline]
    pub fn virtual_time(&self) -> &Time<Virtual> {
        &self.virtual_time
    }
    /// Mutable access to the virtual clock.
    #[inline]
    pub fn virtual_time_mut(&mut self) -> &mut Time<Virtual> {
        &mut self.virtual_time
    }

    /// The fixed clock.
    #[inline]
    pub fn fixed(&self) -> &Time<Fixed> {
        &self.fixed
    }
    /// Mutable access to the fixed clock.
    #[inline]
    pub fn fixed_mut(&mut self) -> &mut Time<Fixed> {
        &mut self.fixed
    }

    /// The context-less default clock, mirroring the active source.
    #[inline]
    pub fn default_time(&self) -> &Time<()> {
        &self.default
    }

    /// Which context the default clock currently mirrors.
    #[inline]
    pub fn source(&self) -> DefaultSource {
        self.source
    }

    /// Point the default clock at `source` and refresh its readings now.
    #[inline]
    pub fn set_source(&mut self, source: DefaultSource) {
        self.source = source;
        self.sync_default();
    }

    /// Refresh the default clock's readings from the active source. Call after
    /// advancing the source clock so the default reflects the latest step.
    #[inline]
    pub fn sync_default(&mut self) {
        match self.source {
            DefaultSource::Virtual => self.default.copy_readings_from(&self.virtual_time),
            DefaultSource::Fixed => self.default.copy_readings_from(&self.fixed),
        }
    }

    /// The default clock's last delta in seconds (`f32`): the ergonomic
    /// one-liner for systems that do not care which context is active.
    #[inline]
    pub fn delta_secs(&self) -> f32 {
        self.default.delta_secs()
    }
}
