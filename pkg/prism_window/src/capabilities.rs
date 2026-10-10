//! Backend capability reporting and HDR configuration.
//!
//! The backend reports a [`WindowCapabilities`] snapshot at startup and on
//! hot-plug. The kernel never panics or assumes a feature exists: when a
//! requested capability is missing it degrades the *backend path* (e.g. no HDR
//! → sRGB) while the kernel's model and semantics stay intact
//! (`prism_window_refactor_zh.md` §4.3 law 3; winit design §18 capability
//! matrix). HDR configuration lives here next to [`HdrSupport`] so all HDR
//! vocabulary stays in one place.

use crate::mode::PresentMode;

/// The HDR pipeline a surface can drive.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum HdrSupport {
    /// Standard dynamic range only (sRGB / BT.709).
    #[default]
    None,
    /// HDR10 (BT.2020 primaries, PQ transfer, 10-bit).
    Hdr10,
    /// Extended-range scRGB (linear FP16, used by Windows/compositor paths).
    ScRgb,
}

impl HdrSupport {
    /// Whether any HDR output is possible.
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Which HDR encoding to drive on the surface.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum HdrMode {
    /// SDR output (HDR disabled).
    #[default]
    Disabled,
    /// HDR10 / BT.2020 PQ.
    Hdr10,
    /// scRGB extended-range linear.
    ScRgb,
}

/// Requested HDR output configuration (a cold-path command payload).
///
/// Luminance values use integer nits (and milli-nits for the small minimum) to
/// stay deterministic and `no_std`-friendly. `0` means "unspecified / let the
/// backend pick a sensible default".
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct HdrConfig {
    /// Target HDR encoding (or [`HdrMode::Disabled`] to turn HDR off).
    pub mode: HdrMode,
    /// Mastering max content luminance in nits (`0` = unspecified).
    pub max_content_luminance_nits: u32,
    /// Max frame-average light level in nits (`0` = unspecified).
    pub max_frame_average_luminance_nits: u32,
    /// Minimum luminance in milli-nits (`0` = unspecified).
    pub min_luminance_millinits: u32,
}

impl HdrConfig {
    /// A disabled (SDR) configuration.
    pub const DISABLED: Self = Self {
        mode: HdrMode::Disabled,
        max_content_luminance_nits: 0,
        max_frame_average_luminance_nits: 0,
        min_luminance_millinits: 0,
    };

    /// Whether this configuration asks for HDR output.
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        !matches!(self.mode, HdrMode::Disabled)
    }
}

/// A snapshot of what the active backend/platform can do.
///
/// Reported at startup and refreshed on hot-plug. `present_modes` is a static
/// slice because a backend's supported set is fixed for a session.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WindowCapabilities {
    /// Can create and manage more than one top-level window.
    pub multi_window: bool,
    /// HDR output support level.
    pub hdr: HdrSupport,
    /// Variable refresh rate (`FreeSync` / G-Sync / VRR) available.
    pub vrr: bool,
    /// Exclusive (mode-switching) fullscreen available.
    pub exclusive_fullscreen: bool,
    /// Per-pixel transparent surfaces supported.
    pub transparency: bool,
    /// Always-on-top / window-level control honored.
    pub always_on_top: bool,
    /// Cursor can be confined to the window bounds.
    pub cursor_confine: bool,
    /// Cursor can be locked in place (relative mouse-look).
    pub cursor_lock: bool,
    /// IME enable/position control is available.
    pub ime_control: bool,
    /// Window can be programmatically dragged/resized by the app.
    pub drag_window: bool,
    /// Per-monitor DPI (mixed-DPI) is reported and honored.
    pub per_monitor_dpi: bool,
    /// Surface contents can be marked protected (excluded from capture).
    pub content_protection: bool,
    /// Present modes the surface supports, in preference order.
    pub present_modes: &'static [PresentMode],
}

/// The present modes always available on every backend (classic v-sync).
pub const FALLBACK_PRESENT_MODES: &[PresentMode] = &[PresentMode::Fifo];

impl Default for WindowCapabilities {
    /// A conservative "lowest common denominator" backend: single realized
    /// window path, SDR, fixed refresh, Fifo-only. Real backends overwrite
    /// this with a probed snapshot.
    fn default() -> Self {
        Self {
            multi_window: false,
            hdr: HdrSupport::None,
            vrr: false,
            exclusive_fullscreen: false,
            transparency: false,
            always_on_top: false,
            cursor_confine: false,
            cursor_lock: false,
            ime_control: false,
            drag_window: false,
            per_monitor_dpi: false,
            content_protection: false,
            present_modes: FALLBACK_PRESENT_MODES,
        }
    }
}

impl WindowCapabilities {
    /// Whether `mode` is in the supported present-mode set.
    #[must_use]
    pub fn supports_present_mode(&self, mode: PresentMode) -> bool {
        self.present_modes.contains(&mode)
    }

    /// Whether the backend can honor the given HDR request. A disabled (SDR)
    /// request is always honorable.
    #[must_use]
    pub const fn supports_hdr(&self, config: HdrConfig) -> bool {
        match config.mode {
            HdrMode::Disabled => true,
            HdrMode::Hdr10 => matches!(self.hdr, HdrSupport::Hdr10),
            HdrMode::ScRgb => matches!(self.hdr, HdrSupport::ScRgb),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_capabilities_are_conservative() {
        let caps = WindowCapabilities::default();
        assert!(!caps.multi_window);
        assert_eq!(caps.hdr, HdrSupport::None);
        assert!(caps.supports_present_mode(PresentMode::Fifo));
        assert!(!caps.supports_present_mode(PresentMode::Immediate));
    }

    #[test]
    fn hdr_support_detection() {
        assert!(!HdrSupport::None.is_hdr());
        assert!(HdrSupport::Hdr10.is_hdr());
        assert!(HdrConfig::DISABLED.mode == HdrMode::Disabled);
        assert!(!HdrConfig::DISABLED.is_hdr());
    }

    #[test]
    fn supports_hdr_matches_mode_to_support() {
        let mut caps = WindowCapabilities::default();
        // SDR always honorable even on an SDR backend.
        assert!(caps.supports_hdr(HdrConfig::DISABLED));
        let hdr10 = HdrConfig {
            mode: HdrMode::Hdr10,
            ..HdrConfig::DISABLED
        };
        assert!(!caps.supports_hdr(hdr10));
        caps.hdr = HdrSupport::Hdr10;
        assert!(caps.supports_hdr(hdr10));
        // scRGB request still unsupported on an HDR10-only backend.
        let scrgb = HdrConfig {
            mode: HdrMode::ScRgb,
            ..HdrConfig::DISABLED
        };
        assert!(!caps.supports_hdr(scrgb));
    }
}
