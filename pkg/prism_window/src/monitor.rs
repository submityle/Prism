//! Monitor and video-mode descriptors reported by the windowing backend.

use alloc::string::String;
use alloc::vec::Vec;

use crate::geometry::{PhysicalPosition, PhysicalSize};

/// Stable identifier for a connected monitor, assigned by the backend.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct MonitorId(pub u64);

/// A single video mode a monitor can be set to (for exclusive fullscreen).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VideoMode {
    /// Resolution in physical pixels.
    pub size: PhysicalSize,
    /// Bits per pixel of the color buffer.
    pub bit_depth: u16,
    /// Refresh rate in millihertz (e.g. `59_950` for 59.95 Hz); `0` if unknown.
    pub refresh_rate_millihertz: u32,
}

impl VideoMode {
    /// Refresh rate in hertz, or `0.0` when unknown.
    #[must_use]
    pub fn refresh_rate_hz(&self) -> f32 {
        self.refresh_rate_millihertz as f32 / 1000.0
    }
}

/// A connected monitor: its placement in the virtual desktop, size, scale, and
/// available video modes.
#[derive(Clone, PartialEq, Debug)]
pub struct Monitor {
    /// Backend-assigned id.
    pub id: MonitorId,
    /// Human-readable name, if the backend provides one.
    pub name: Option<String>,
    /// Top-left position in the virtual desktop, in physical pixels.
    pub position: PhysicalPosition,
    /// Size in physical pixels.
    pub size: PhysicalSize,
    /// The monitor's scale factor (folded to a sane positive value by the
    /// backend before construction).
    pub scale_factor: f32,
    /// Current refresh rate in millihertz, if known.
    pub refresh_rate_millihertz: Option<u32>,
    /// Video modes the monitor supports, for exclusive fullscreen selection.
    pub video_modes: Vec<VideoMode>,
}

impl Monitor {
    /// Picks the video mode whose size and refresh rate best match `target`,
    /// preferring an exact size, then the largest mode that fits, then the
    /// highest refresh rate as a tie-break. Returns `None` only when the
    /// monitor reports no video modes.
    #[must_use]
    pub fn best_video_mode(&self, target: PhysicalSize) -> Option<VideoMode> {
        self.video_modes
            .iter()
            .copied()
            .max_by_key(|mode| {
                let exact = u8::from(mode.size == target);
                // Prefer modes that fit within the target, larger area first.
                let fits = u8::from(mode.size.width <= target.width && mode.size.height <= target.height);
                let area = u64::from(mode.size.width) * u64::from(mode.size.height);
                (exact, fits, area, mode.refresh_rate_millihertz)
            })
    }
}
