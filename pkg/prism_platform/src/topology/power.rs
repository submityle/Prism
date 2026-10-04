//! Energy / thermal state and the scheduler back-off hints derived from it
//! (design §24.3).
//!
//! On mobile and console the OS continuously signals power and thermal
//! pressure; a well-behaved engine reads those signals and *backs off* —
//! shedding background load, parking work onto efficiency cores, and capping
//! the frame rate — before the OS forcibly down-clocks it. This module models
//! that state as portable data and derives a [`PowerPolicy`] of concrete,
//! testable hints. It reads no sensors itself; a platform backend fills in a
//! [`PowerState`] and the rest is pure, deterministic policy.

/// Where the machine's power is coming from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PowerSource {
    /// Wall power / docked.
    Ac,
    /// Running on battery.
    Battery,
    /// Unknown (desktop with no battery, or not probed).
    #[default]
    Unknown,
}

/// Coarse thermal pressure, mirroring the common OS ladders (Apple
/// `thermalState`, Android `thermalStatus`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ThermalState {
    /// No thermal pressure.
    #[default]
    Nominal,
    /// Mild pressure; the OS has begun light mitigation.
    Fair,
    /// Serious pressure; shed non-essential work now.
    Serious,
    /// Critical pressure; the OS is about to throttle hard or shut down.
    Critical,
}

impl ThermalState {
    /// Whether the engine should actively shed load at this level.
    #[must_use]
    pub const fn should_shed_load(self) -> bool {
        matches!(self, ThermalState::Serious | ThermalState::Critical)
    }
}

/// A snapshot of the machine's power and thermal condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PowerState {
    /// Where power is coming from.
    pub source: PowerSource,
    /// Thermal pressure.
    pub thermal: ThermalState,
    /// Battery charge in percent `0..=100`, when known.
    pub battery_percent: Option<u8>,
    /// Whether the OS reports a user-enabled low-power / battery-saver mode.
    pub low_power_mode: bool,
}

impl PowerState {
    /// A nominal AC-powered state with no pressure.
    #[must_use]
    pub const fn plugged_in() -> Self {
        Self {
            source: PowerSource::Ac,
            thermal: ThermalState::Nominal,
            battery_percent: None,
            low_power_mode: false,
        }
    }

    /// Whether the battery charge is at or below `pct` percent (always `false`
    /// when the charge is unknown).
    #[must_use]
    pub fn battery_at_or_below(&self, pct: u8) -> bool {
        self.battery_percent.is_some_and(|b| b <= pct)
    }

    /// Derive the concrete scheduler back-off policy for this state.
    #[must_use]
    pub fn policy(&self) -> PowerPolicy {
        // Constrained when the user asked to save power, when on a low battery,
        // or under real thermal pressure.
        let low_battery = self.battery_at_or_below(20);
        let constrained = self.low_power_mode || low_battery || self.thermal.should_shed_load();

        let background_load_scale = match self.thermal {
            ThermalState::Nominal if !constrained => 100,
            ThermalState::Nominal | ThermalState::Fair => 60,
            ThermalState::Serious => 30,
            ThermalState::Critical => 0,
        };

        let frame_rate_cap = match self.thermal {
            ThermalState::Critical | ThermalState::Serious => Some(30),
            _ if self.low_power_mode || low_battery => Some(30),
            _ => None,
        };

        PowerPolicy {
            background_load_scale,
            prefer_efficiency_cores: constrained,
            frame_rate_cap,
        }
    }
}

/// Concrete, testable scheduler/pacing hints derived from a [`PowerState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PowerPolicy {
    /// Fraction (in percent, `0..=100`) of the normal background-work budget
    /// the scheduler should admit. `0` means pause background work entirely.
    pub background_load_scale: u8,
    /// Whether background and non-critical work should be steered onto
    /// efficiency cores.
    pub prefer_efficiency_cores: bool,
    /// An advisory frame-rate cap in `FPS` to apply (feeds `prism_time`
    /// pacing), or `None` to leave pacing unconstrained.
    pub frame_rate_cap: Option<u16>,
}
