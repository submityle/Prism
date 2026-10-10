//! Lossless conversions between winit's `dpi` types and the kernel geometry.
//!
//! These are the single source of truth for crossing the type boundary, so
//! rounding and sign conventions live in exactly one place and are unit
//! tested. Everything here is pure and allocation-free.

use prism_window::geometry::{PhysicalPosition, PhysicalSize};

/// Converts a winit physical size (`u32`) into the kernel physical size.
#[must_use]
pub fn size_from_winit(size: winit::dpi::PhysicalSize<u32>) -> PhysicalSize {
    PhysicalSize::new(size.width, size.height)
}

/// Converts a kernel physical size into a winit physical size (`u32`).
#[must_use]
pub fn size_to_winit(size: PhysicalSize) -> winit::dpi::PhysicalSize<u32> {
    winit::dpi::PhysicalSize::new(size.width, size.height)
}

/// Converts a winit physical position (`i32`, used by `Moved`) into the kernel
/// physical position.
#[must_use]
pub fn position_from_winit(pos: winit::dpi::PhysicalPosition<i32>) -> PhysicalPosition {
    PhysicalPosition::new(pos.x, pos.y)
}

/// Converts a kernel physical position into a winit physical position (`i32`).
#[must_use]
pub fn position_to_winit(pos: PhysicalPosition) -> winit::dpi::PhysicalPosition<i32> {
    winit::dpi::PhysicalPosition::new(pos.x, pos.y)
}

/// Converts a winit cursor position (`f64`, window-relative) into the kernel's
/// integer physical position. The cursor stream is sampled in whole pixels;
/// winit may report subpixel values on some platforms, so we round half-up
/// toward positive infinity for determinism (matching the kernel's own
/// `to_physical`).
#[must_use]
pub fn cursor_from_winit(pos: winit::dpi::PhysicalPosition<f64>) -> PhysicalPosition {
    PhysicalPosition::new(round_half_up(pos.x), round_half_up(pos.y))
}

/// Encodes a floating scale factor (physical pixels per logical pixel) as the
/// kernel's fixed-point milli-units (`2.0 -> 2000`). Clamps to a sane positive
/// range so a degenerate `0.0`/`NaN` from the OS can never produce a zero or
/// negative scale downstream.
#[must_use]
pub fn scale_factor_to_milli(scale_factor: f64) -> u32 {
    if !scale_factor.is_finite() || scale_factor <= 0.0 {
        return 1000;
    }
    // Round half-up; cap at a generous ceiling to avoid overflow on bogus input.
    let milli = (scale_factor * 1000.0 + 0.5).floor();
    let clamped = milli.clamp(1.0, f64::from(u32::MAX));
    clamped as u32
}

/// Decodes milli-units back into a floating scale factor.
#[must_use]
pub fn scale_factor_from_milli(milli: u32) -> f64 {
    f64::from(milli) / 1000.0
}

fn round_half_up(v: f64) -> i32 {
    if !v.is_finite() {
        return 0;
    }
    (v + 0.5).floor() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_roundtrips() {
        let w = winit::dpi::PhysicalSize::new(1920, 1080);
        let k = size_from_winit(w);
        assert_eq!(k, PhysicalSize::new(1920, 1080));
        assert_eq!(size_to_winit(k), w);
    }

    #[test]
    fn position_roundtrips_including_negative() {
        let w = winit::dpi::PhysicalPosition::new(-100, 42);
        let k = position_from_winit(w);
        assert_eq!(k, PhysicalPosition::new(-100, 42));
        assert_eq!(position_to_winit(k), w);
    }

    #[test]
    fn cursor_rounds_half_up() {
        assert_eq!(
            cursor_from_winit(winit::dpi::PhysicalPosition::new(10.5, 20.4)),
            PhysicalPosition::new(11, 20)
        );
        assert_eq!(
            cursor_from_winit(winit::dpi::PhysicalPosition::new(-0.5, -1.5)),
            PhysicalPosition::new(0, -1)
        );
    }

    #[test]
    fn cursor_never_panics_on_nonfinite() {
        let p = cursor_from_winit(winit::dpi::PhysicalPosition::new(f64::NAN, f64::INFINITY));
        assert_eq!(p, PhysicalPosition::new(0, 0));
    }

    #[test]
    fn scale_factor_encodes_and_clamps() {
        assert_eq!(scale_factor_to_milli(1.0), 1000);
        assert_eq!(scale_factor_to_milli(2.0), 2000);
        assert_eq!(scale_factor_to_milli(1.25), 1250);
        // Degenerate inputs fold to 1.0 rather than producing a zero scale.
        assert_eq!(scale_factor_to_milli(0.0), 1000);
        assert_eq!(scale_factor_to_milli(-3.0), 1000);
        assert_eq!(scale_factor_to_milli(f64::NAN), 1000);
    }

    #[test]
    fn scale_factor_milli_roundtrips() {
        assert!((scale_factor_from_milli(scale_factor_to_milli(1.5)) - 1.5).abs() < 1e-9);
    }
}
