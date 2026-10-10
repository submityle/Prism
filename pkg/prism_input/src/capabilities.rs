//! Backend capability reporting.
//!
//! On startup (and on hot-plug refresh) a backend reports an
//! [`InputBackendCapabilities`] describing what the current platform/device set
//! can actually do. The kernel uses it to drive graceful degradation —
//! *degrade the result, not the model*: a backend with no rumble simply reports
//! `rumble: false` and the output command becomes a no-op; the kernel's
//! vocabulary and semantics never collapse.
//!
//! The name is deliberately `InputBackendCapabilities` (not `WindowCapabilities`
//! or a shared `Capabilities`) so the input and window kernels stay independent
//! and never collide when their ABI layers evolve in parallel.

/// What an input backend can do on the current platform.
///
/// Fields are facts reported by the backend, not requests. Absent capabilities
/// must never cause a panic: the backend reports `false`/`0` and the
/// corresponding feature degrades to a no-op (see the design doc §4.1/§4.2).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InputBackendCapabilities {
    /// A physical keyboard is available.
    pub keyboard: bool,
    /// A mouse (buttons, relative motion, wheel) is available.
    pub mouse: bool,
    /// A touch surface (touchscreen/touchpad) is available.
    pub touch: bool,
    /// At least one gamepad is (or can be) connected.
    pub gamepad: bool,
    /// Raw relative pointer motion is available (high-poll-rate mice, FPS aim),
    /// bypassing OS pointer acceleration/clamping.
    pub raw_motion: bool,
    /// The maximum input polling rate in Hz the backend can deliver
    /// (e.g. `1000`–`8000` for high-poll-rate mice). `0` means "unknown".
    pub max_polling_hz: u32,
    /// The maximum number of simultaneously trackable gamepads. `0` means the
    /// backend reports no gamepad support.
    pub max_gamepads: u32,
    /// Rumble / force feedback output is supported.
    pub rumble: bool,
    /// Adaptive trigger output (DualSense-style) is supported.
    pub adaptive_trigger: bool,
    /// Gyroscope / motion sensing input is available.
    pub gyro: bool,
    /// A gamepad touchpad surface is available.
    pub touchpad: bool,
}

impl InputBackendCapabilities {
    /// A capability set with everything disabled — the correct report for a
    /// headless/no-op backend, and a safe default to degrade from.
    pub const NONE: Self = Self {
        keyboard: false,
        mouse: false,
        touch: false,
        gamepad: false,
        raw_motion: false,
        max_polling_hz: 0,
        max_gamepads: 0,
        rumble: false,
        adaptive_trigger: false,
        gyro: false,
        touchpad: false,
    };

    /// Returns a capability set with everything disabled.
    #[must_use]
    pub const fn none() -> Self {
        Self::NONE
    }

    /// Whether force feedback (rumble) output is available. Alias spelling for
    /// callers that think in terms of "force feedback" rather than "rumble".
    #[must_use]
    pub const fn supports_force_feedback(self) -> bool {
        self.rumble
    }

    /// Whether any gamepad output capability (rumble, adaptive trigger) exists,
    /// so callers can skip the whole output pipeline when nothing is driveable.
    #[must_use]
    pub const fn has_gamepad_output(self) -> bool {
        self.rumble || self.adaptive_trigger
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_all_disabled() {
        let caps = InputBackendCapabilities::none();
        assert_eq!(caps, InputBackendCapabilities::NONE);
        assert_eq!(caps, InputBackendCapabilities::default());
        assert!(!caps.keyboard);
        assert!(!caps.rumble);
        assert_eq!(caps.max_polling_hz, 0);
        assert_eq!(caps.max_gamepads, 0);
        assert!(!caps.has_gamepad_output());
        assert!(!caps.supports_force_feedback());
    }

    #[test]
    fn construct_full_desktop_capabilities() {
        let caps = InputBackendCapabilities {
            keyboard: true,
            mouse: true,
            touch: false,
            gamepad: true,
            raw_motion: true,
            max_polling_hz: 8000,
            max_gamepads: 4,
            rumble: true,
            adaptive_trigger: true,
            gyro: true,
            touchpad: true,
        };
        assert!(caps.keyboard && caps.mouse && caps.gamepad);
        assert!(caps.raw_motion);
        assert_eq!(caps.max_polling_hz, 8000);
        assert_eq!(caps.max_gamepads, 4);
        assert!(caps.supports_force_feedback());
        assert!(caps.has_gamepad_output());
    }

    #[test]
    fn force_feedback_alias_tracks_rumble() {
        let mut caps = InputBackendCapabilities::none();
        assert!(!caps.supports_force_feedback());
        caps.rumble = true;
        assert!(caps.supports_force_feedback());
        assert!(caps.has_gamepad_output());
    }

    #[test]
    fn capabilities_are_copy() {
        let caps = InputBackendCapabilities {
            mouse: true,
            ..InputBackendCapabilities::none()
        };
        let copied = caps;
        assert_eq!(copied, caps);
        assert!(caps.mouse);
    }
}
