//! Deterministic floating-window placement across monitor work areas.

use crate::Rect;
use alloc::vec::Vec;

/// Stable monitor/work-area snapshot supplied by a window host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MonitorWorkArea {
    /// Backend-stable monitor identity.
    pub id: u64,
    /// Work-area origin in physical desktop pixels.
    pub x: i32,
    /// Work-area origin in physical desktop pixels.
    pub y: i32,
    /// Work-area width in physical pixels.
    pub width: u32,
    /// Work-area height in physical pixels.
    pub height: u32,
    /// Current monitor scale factor represented as fixed thousandths.
    pub scale_milli: u32,
}

impl MonitorWorkArea {
    /// Creates a monitor work-area descriptor.
    pub const fn new(id: u64, x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            id,
            x,
            y,
            width,
            height,
            scale_milli: 1000,
        }
    }

    /// Returns the work-area rectangle.
    pub const fn rect(self) -> Rect {
        Rect {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
        }
    }
}

/// Persisted placement metadata for one floating dock window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatingPlacement {
    /// Preferred monitor identity.
    pub monitor_id: u64,
    /// Logical dock rectangle captured on that monitor.
    pub rect: Rect,
    /// Scale factor in fixed thousandths when the rectangle was captured.
    pub scale_milli: u32,
}

/// Result of monitor-aware placement restoration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestoredPlacement {
    /// Monitor actually selected for the restored window.
    pub monitor_id: u64,
    /// Clamped physical rectangle guaranteed to remain visible.
    pub rect: Rect,
}

/// Restores one floating window onto the preferred monitor or the first
/// available monitor, preserving its visible fraction across DPI changes.
pub fn restore_placement(
    placement: FloatingPlacement,
    monitors: &[MonitorWorkArea],
    minimum_visible: u32,
) -> Option<RestoredPlacement> {
    let monitor = monitors
        .iter()
        .find(|monitor| monitor.id == placement.monitor_id)
        .or_else(|| monitors.first())?;
    let work = monitor.rect();
    let source_scale = placement.scale_milli.max(1);
    let target_scale = monitor.scale_milli.max(1);
    let width = scale_extent(placement.rect.width, source_scale, target_scale)
        .min(work.width)
        .max(1);
    let height = scale_extent(placement.rect.height, source_scale, target_scale)
        .min(work.height)
        .max(1);
    let x = clamp_origin(placement.rect.x, width, work.x, work.width, minimum_visible);
    let y = clamp_origin(
        placement.rect.y,
        height,
        work.y,
        work.height,
        minimum_visible,
    );
    Some(RestoredPlacement {
        monitor_id: monitor.id,
        rect: Rect {
            x,
            y,
            width,
            height,
        },
    })
}

fn scale_extent(value: u32, from: u32, to: u32) -> u32 {
    let scaled = u64::from(value)
        .saturating_mul(u64::from(to))
        .saturating_add(u64::from(from / 2))
        / u64::from(from);
    scaled.min(u64::from(u32::MAX)) as u32
}

fn clamp_origin(value: i32, extent: u32, origin: i32, available: u32, visible: u32) -> i32 {
    let available = i64::from(available);
    let extent = i64::from(extent);
    let work_origin = i64::from(origin);
    let candidate = i64::from(value);
    let max_x = work_origin + (available - extent).max(0);
    let min_visible = i64::from(visible).min(extent);
    let min_x = work_origin - extent + min_visible;
    candidate
        .clamp(min_x.min(max_x), max_x)
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// Restores a list of placements in deterministic input order.
pub fn restore_all(
    placements: &[FloatingPlacement],
    monitors: &[MonitorWorkArea],
    minimum_visible: u32,
) -> Vec<RestoredPlacement> {
    placements
        .iter()
        .filter_map(|placement| restore_placement(*placement, monitors, minimum_visible))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_monitor_falls_back_and_clamps_visible_area() {
        let monitors = [MonitorWorkArea::new(2, 100, 50, 800, 600)];
        let restored = restore_placement(
            FloatingPlacement {
                monitor_id: 1,
                rect: Rect {
                    x: -1000,
                    y: -1000,
                    width: 400,
                    height: 300,
                },
                scale_milli: 1000,
            },
            &monitors,
            32,
        )
        .unwrap();
        assert_eq!(restored.monitor_id, 2);
        assert_eq!(restored.rect.x, 100 - 400 + 32);
        assert_eq!(restored.rect.y, 50 - 300 + 32);
    }

    #[test]
    fn dpi_resize_is_deterministic() {
        let mut monitor = MonitorWorkArea::new(1, 0, 0, 1600, 900);
        monitor.scale_milli = 2000;
        let restored = restore_placement(
            FloatingPlacement {
                monitor_id: 1,
                rect: Rect {
                    x: 10,
                    y: 20,
                    width: 320,
                    height: 200,
                },
                scale_milli: 1000,
            },
            &[monitor],
            16,
        )
        .unwrap();
        assert_eq!(restored.rect.width, 640);
        assert_eq!(restored.rect.height, 400);
    }
}
