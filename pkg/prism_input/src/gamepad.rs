//! Gamepad buttons, axes, connection state, and the deadzone/response settings
//! that turn noisy analog hardware into clean, deterministic input.

use alloc::collections::BTreeMap;

/// Identifies a connected gamepad by its stable slot index.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct GamepadId(pub u32);

/// A gamepad button, named by its standard layout role rather than its glyph so
/// the same code works across controller brands.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum GamepadButton {
    /// Bottom face button (Xbox A / `PlayStation` Cross).
    South,
    /// Right face button (Xbox B / `PlayStation` Circle).
    East,
    /// Top face button (Xbox Y / `PlayStation` Triangle).
    North,
    /// Left face button (Xbox X / `PlayStation` Square).
    West,
    /// Left shoulder bumper.
    LeftBumper,
    /// Right shoulder bumper.
    RightBumper,
    /// Left trigger pressed past its digital threshold.
    LeftTrigger,
    /// Right trigger pressed past its digital threshold.
    RightTrigger,
    /// Select / View / Back.
    Select,
    /// Start / Menu / Options.
    Start,
    /// The central brand/guide button.
    Mode,
    /// Left stick pressed in.
    LeftThumb,
    /// Right stick pressed in.
    RightThumb,
    /// D-pad up.
    DPadUp,
    /// D-pad down.
    DPadDown,
    /// D-pad left.
    DPadLeft,
    /// D-pad right.
    DPadRight,
}

/// A gamepad analog axis.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum GamepadAxis {
    /// Left stick horizontal (right positive).
    LeftStickX,
    /// Left stick vertical (up positive).
    LeftStickY,
    /// Right stick horizontal (right positive).
    RightStickX,
    /// Right stick vertical (up positive).
    RightStickY,
    /// Left trigger pressure in `[0, 1]`.
    LeftZ,
    /// Right trigger pressure in `[0, 1]`.
    RightZ,
}

/// A gamepad connection transition.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GamepadConnection {
    /// The gamepad became available.
    Connected,
    /// The gamepad was removed.
    Disconnected,
}

/// Deadzone and rescaling settings for a single analog axis.
///
/// Raw sticks never rest exactly at zero and never quite reach `±1`. These
/// settings remove the center jitter (deadzone) and expand the usable range
/// (livezone) back to the full `[-1, 1]`, so gameplay sees a clean signal.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AxisSettings {
    deadzone: f32,
    livezone: f32,
    threshold: f32,
}

impl Default for AxisSettings {
    fn default() -> Self {
        // Conservative defaults typical of console tuning.
        Self {
            deadzone: 0.1,
            livezone: 0.95,
            threshold: 0.01,
        }
    }
}

impl AxisSettings {
    /// Builds settings, clamping each bound into range and ensuring
    /// `deadzone <= livezone` so [`filter`](Self::filter) stays well-defined.
    #[must_use]
    pub fn new(deadzone: f32, livezone: f32, threshold: f32) -> Self {
        let deadzone = deadzone.clamp(0.0, 1.0);
        let livezone = livezone.clamp(deadzone, 1.0);
        Self {
            deadzone,
            livezone,
            threshold: threshold.max(0.0),
        }
    }

    /// The center deadzone magnitude.
    #[must_use]
    pub const fn deadzone(&self) -> f32 {
        self.deadzone
    }

    /// The outer livezone magnitude that maps to full deflection.
    #[must_use]
    pub const fn livezone(&self) -> f32 {
        self.livezone
    }

    /// Applies the deadzone and livezone rescaling to a raw reading, returning
    /// a value in `[-1, 1]`. Readings inside the deadzone collapse to `0`;
    /// readings past the livezone saturate to `±1`; the band between is
    /// linearly remapped so motion starts exactly at the deadzone edge.
    #[must_use]
    pub fn filter(&self, raw: f32) -> f32 {
        let r = raw.clamp(-1.0, 1.0);
        let magnitude = r.abs();
        if magnitude <= self.deadzone {
            return 0.0;
        }
        if magnitude >= self.livezone {
            return r.signum();
        }
        let span = self.livezone - self.deadzone;
        // `span > 0` because `magnitude` lies strictly between the two bounds.
        let scaled = (magnitude - self.deadzone) / span;
        r.signum() * scaled
    }

    /// Whether a change from `old` to `new` is large enough to report, used to
    /// suppress sub-`threshold` jitter.
    #[must_use]
    pub fn should_report(&self, old: f32, new: f32) -> bool {
        (new - old).abs() >= self.threshold
    }
}

/// Hysteresis thresholds that turn an analog trigger into a digital button.
///
/// Separate press and release thresholds avoid chatter when a trigger hovers
/// near a single cutoff.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ButtonSettings {
    press: f32,
    release: f32,
}

impl Default for ButtonSettings {
    fn default() -> Self {
        Self {
            press: 0.75,
            release: 0.65,
        }
    }
}

impl ButtonSettings {
    /// Builds settings, clamping both thresholds into `[0, 1]` and ensuring
    /// `release <= press` so the hysteresis band is non-inverted.
    #[must_use]
    pub fn new(press: f32, release: f32) -> Self {
        let press = press.clamp(0.0, 1.0);
        let release = release.clamp(0.0, press);
        Self { press, release }
    }

    /// Whether `value` is high enough to count as a press.
    #[must_use]
    pub fn is_pressed(&self, value: f32) -> bool {
        value >= self.press
    }

    /// Whether `value` is low enough to count as a release.
    #[must_use]
    pub fn is_released(&self, value: f32) -> bool {
        value <= self.release
    }
}

/// Per-gamepad input tuning: default curves plus optional per-axis/per-button
/// overrides.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct GamepadSettings {
    /// The axis settings used when an axis has no override.
    pub default_axis: AxisSettings,
    /// The button settings used when a button has no override.
    pub default_button: ButtonSettings,
    axis_overrides: BTreeMap<GamepadAxis, AxisSettings>,
    button_overrides: BTreeMap<GamepadButton, ButtonSettings>,
}

impl GamepadSettings {
    /// Creates settings with the default curves and no overrides.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the settings for one axis.
    pub fn set_axis(&mut self, axis: GamepadAxis, settings: AxisSettings) {
        self.axis_overrides.insert(axis, settings);
    }

    /// Overrides the settings for one button.
    pub fn set_button(&mut self, button: GamepadButton, settings: ButtonSettings) {
        self.button_overrides.insert(button, settings);
    }

    /// The effective settings for `axis` (override if present, else default).
    #[must_use]
    pub fn axis(&self, axis: GamepadAxis) -> AxisSettings {
        self.axis_overrides
            .get(&axis)
            .copied()
            .unwrap_or(self.default_axis)
    }

    /// The effective settings for `button` (override if present, else default).
    #[must_use]
    pub fn button(&self, button: GamepadButton) -> ButtonSettings {
        self.button_overrides
            .get(&button)
            .copied()
            .unwrap_or(self.default_button)
    }
}

/// Applies a radial deadzone to a stick vector `(x, y)`, treating the stick as
/// a 2D control so diagonal inputs are not penalized by per-axis deadzones.
///
/// If the vector's length is within `deadzone` it collapses to `(0, 0)`;
/// otherwise the magnitude is remapped from `[deadzone, 1]` onto `[0, 1]` and
/// the original direction is preserved. `deadzone` is clamped to `[0, 1)`.
#[must_use]
pub fn radial_deadzone(x: f32, y: f32, deadzone: f32) -> (f32, f32) {
    let deadzone = deadzone.clamp(0.0, 0.999_999);
    let length = libm_hypot(x, y);
    if length <= deadzone || length == 0.0 {
        return (0.0, 0.0);
    }
    let clamped = length.min(1.0);
    let scaled = (clamped - deadzone) / (1.0 - deadzone);
    let factor = scaled / length;
    (x * factor, y * factor)
}

/// `hypot` without pulling in a math dependency; adequate for deadzone scaling.
///
/// Uses the scaled form to avoid overflow for the small magnitudes stick axes
/// produce while staying `no_std`.
fn libm_hypot(x: f32, y: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    let (max, min) = if ax >= ay { (ax, ay) } else { (ay, ax) };
    if max == 0.0 {
        return 0.0;
    }
    let ratio = min / max;
    max * sqrt_f32(1.0 + ratio * ratio)
}

/// A `no_std` square root via one Newton refinement over a bit-trick seed.
///
/// Deadzone math only needs a few digits of accuracy, so this avoids a libm
/// dependency while staying deterministic.
fn sqrt_f32(value: f32) -> f32 {
    if value <= 0.0 {
        return 0.0;
    }
    // Initial guess from halving the exponent via the classic bit hack,
    // expressed without `unsafe` through `to_bits`/`from_bits`.
    let mut guess = f32::from_bits((value.to_bits() >> 1).wrapping_add(0x1fbd_1df5));
    // Newton–Raphson iterations converge quadratically from this seed.
    guess = 0.5 * (guess + value / guess);
    guess = 0.5 * (guess + value / guess);
    guess = 0.5 * (guess + value / guess);
    guess
}
