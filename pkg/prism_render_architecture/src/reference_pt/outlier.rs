//! Neighbourhood-median firefly suppression as an image-space post-process.
//!
//! The per-sample [`FireflyClamp`](super::firefly::FireflyClamp) caps energy
//! before it is accumulated, which removes spikes but applies the same fixed
//! ceiling everywhere and so dims genuinely bright regions (a small emitter, a
//! sharp specular highlight) along with the stray outliers. A complementary,
//! purely classical tool is a *spatial* outlier reject that runs once on the
//! finished image: for each pixel it compares the pixel against the median of
//! its neighbourhood and only pulls down a pixel that is far brighter than the
//! local consensus. Isolated single-pixel "fireflies" stand out from their
//! neighbours and are suppressed; a bright feature that fills a region has
//! bright neighbours, keeps a high local median, and is left untouched.
//!
//! Using the *median* (not the mean) is the key: a single outlier does not move
//! the median of a small window, so the reference the pixel is tested against
//! is itself immune to the firefly it is meant to detect. This is the standard
//! median / "max-of-neighbours" firefly filter shipped in production path
//! tracers, with no machine learning or trained model involved.
//!
//! The filter preserves chromaticity exactly: an offending pixel is scaled
//! uniformly so its luminance equals the cap, keeping its hue and saturation
//! and only shedding the excess brightness, matching the per-sample clamp.
//!
//! The reject is relative, so it is complementary to — not a replacement for —
//! the per-sample clamp. A pixel whose neighbourhood median is zero (an isolated
//! bright point on a pure-black field) has no local consensus to be measured
//! against, so it is left untouched; suppressing absolute spikes on black is the
//! job of [`FireflyClamp`](super::firefly::FireflyClamp), while this pass removes
//! pixels that merely stick out from a non-trivial local neighbourhood.
//!
//! The default policy is [`OutlierFilter::Off`], which returns the image
//! unchanged; the spatial reject is strictly opt-in.

use alloc::vec::Vec;

use super::film::Film;
use super::Vec3;

/// Luma weight for the red channel (standard luma coefficients, summing to one
/// so a unit-white pixel has unit luminance).
const LUMA_R: f32 = 0.2126;
/// Luma weight for the green channel (see [`LUMA_R`]).
const LUMA_G: f32 = 0.7152;
/// Luma weight for the blue channel (see [`LUMA_R`]).
const LUMA_B: f32 = 0.0722;

/// Relative luminance of a linear `RGB` radiance value.
fn luminance(value: Vec3) -> f32 {
    LUMA_R * value.x + LUMA_G * value.y + LUMA_B * value.z
}

/// Image-space firefly suppression policy applied to a finished [`Film`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum OutlierFilter {
    /// No spatial filtering: the image is returned unchanged, keeping the
    /// estimate exactly as the sampler produced it.
    #[default]
    Off,
    /// Reject a pixel whose luminance exceeds `threshold` times the median
    /// luminance of its `(2 * radius + 1)`-square neighbourhood by scaling it
    /// down to that cap, preserving chromaticity.
    MedianGuided {
        /// Multiplier over the neighbourhood median luminance above which a
        /// pixel is treated as an outlier. Must be greater than one to leave
        /// locally consistent brightness untouched; a value at or below one
        /// disables the reject (treated as [`OutlierFilter::Off`]).
        threshold: f32,
        /// Neighbourhood half-extent in pixels. A radius of one tests the eight
        /// surrounding pixels (a `3x3` window); larger radii widen the window
        /// and the median support. A zero radius disables the reject.
        radius: u32,
    },
}

impl OutlierFilter {
    /// Applies the spatial firefly reject to `film`, returning a new image.
    ///
    /// Returns a clone of the input when the filter is off, when the threshold
    /// is at or below one, or when the radius is zero. Border pixels test
    /// against the median of the in-bounds portion of their window, so the
    /// reject never reads outside the image.
    #[must_use]
    pub fn apply(self, film: &Film) -> Film {
        let (threshold, radius) = match self {
            Self::Off => return film.clone(),
            Self::MedianGuided { threshold, radius } => (threshold, radius),
        };
        if threshold <= 1.0 || radius == 0 {
            return film.clone();
        }

        let width = film.width();
        let height = film.height();
        let mut out = Vec::with_capacity((width as usize) * (height as usize));
        // A reusable scratch buffer for the neighbourhood luminances avoids a
        // per-pixel allocation; its length changes with the clamped window.
        let mut window: Vec<f32> =
            Vec::with_capacity(((2 * radius + 1) * (2 * radius + 1)) as usize);

        for y in 0..height {
            for x in 0..width {
                let center = film.pixel(x, y);
                window.clear();

                let x0 = x.saturating_sub(radius);
                let x1 = (x + radius).min(width - 1);
                let y0 = y.saturating_sub(radius);
                let y1 = (y + radius).min(height - 1);
                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        window.push(luminance(film.pixel(nx, ny)));
                    }
                }

                let median = neighbourhood_median(&mut window);
                let cap = median * threshold;
                let center_luma = luminance(center);
                // `cap` is non-negative (median and threshold are), so a pixel
                // is rejected only when it is strictly brighter than the local
                // consensus and that consensus is itself positive.
                if cap > 0.0 && center_luma > cap {
                    out.push(center.scale(cap / center_luma));
                } else {
                    out.push(center);
                }
            }
        }

        Film::from_pixels(width, height, out)
    }
}

/// Median of the luminance window, sorting the scratch buffer in place.
///
/// For an even count the two central samples are averaged, matching the usual
/// statistical median; the buffer is assumed non-empty because every pixel sees
/// at least itself.
fn neighbourhood_median(window: &mut [f32]) -> f32 {
    window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let n = window.len();
    let mid = n / 2;
    if n % 2 == 1 {
        window[mid]
    } else {
        0.5 * (window[mid - 1] + window[mid])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Luminance helper mirrored in the test module for readable assertions.
    fn luma(value: Vec3) -> f32 {
        luminance(value)
    }

    #[test]
    fn off_is_identity() {
        let film = Film::from_pixels(2, 2, vec![Vec3::splat(5.0); 4]);
        let out = OutlierFilter::Off.apply(&film);
        assert_eq!(out.pixels(), film.pixels());
    }

    #[test]
    fn threshold_at_or_below_one_is_identity() {
        let film = Film::from_pixels(2, 2, vec![Vec3::splat(5.0); 4]);
        let filter = OutlierFilter::MedianGuided {
            threshold: 1.0,
            radius: 1,
        };
        assert_eq!(filter.apply(&film).pixels(), film.pixels());
    }

    #[test]
    fn zero_radius_is_identity() {
        let film = Film::from_pixels(2, 2, vec![Vec3::splat(5.0); 4]);
        let filter = OutlierFilter::MedianGuided {
            threshold: 2.0,
            radius: 0,
        };
        assert_eq!(filter.apply(&film).pixels(), film.pixels());
    }

    #[test]
    fn isolated_firefly_is_pulled_down_to_the_cap() {
        // A uniform grey field with one bright spike in the middle. The spike's
        // neighbours are all grey, so the local median stays grey and the spike
        // is rejected down to `threshold * median`.
        let mut pixels = vec![Vec3::splat(1.0); 9];
        pixels[4] = Vec3::splat(100.0);
        let film = Film::from_pixels(3, 3, pixels);
        let filter = OutlierFilter::MedianGuided {
            threshold: 4.0,
            radius: 1,
        };
        let out = filter.apply(&film);
        let center = out.pixel(1, 1);
        assert!(
            (luma(center) - 4.0).abs() < 1.0e-4,
            "the firefly must be capped at threshold*median = 4, got {}",
            luma(center)
        );
    }

    #[test]
    fn firefly_rejection_preserves_chromaticity() {
        let mut pixels = vec![Vec3::splat(1.0); 9];
        pixels[4] = Vec3::new(80.0, 20.0, 5.0);
        let film = Film::from_pixels(3, 3, pixels);
        let filter = OutlierFilter::MedianGuided {
            threshold: 3.0,
            radius: 1,
        };
        let out = filter.apply(&film);
        let center = out.pixel(1, 1);
        // Equal uniform scaling keeps every channel ratio identical.
        let scale = center.x / 80.0;
        assert!((center.y / 20.0 - scale).abs() < 1.0e-5);
        assert!((center.z / 5.0 - scale).abs() < 1.0e-5);
        assert!(luma(center) < luma(Vec3::new(80.0, 20.0, 5.0)));
    }

    #[test]
    fn bright_region_with_bright_neighbours_is_untouched() {
        // When the bright pixel is surrounded by comparably bright pixels the
        // local median is high, the cap is above the pixel, and a genuine
        // feature is preserved rather than dimmed.
        let film = Film::from_pixels(3, 3, vec![Vec3::splat(50.0); 9]);
        let filter = OutlierFilter::MedianGuided {
            threshold: 2.0,
            radius: 1,
        };
        let out = filter.apply(&film);
        assert_eq!(out.pixels(), film.pixels());
    }

    #[test]
    fn spike_on_a_black_field_is_left_to_the_per_sample_clamp() {
        // A spike on a pure-black field has a zero neighbourhood median, so the
        // relative reject has no consensus to measure against and leaves the
        // pixel unchanged (and never divides by zero). Removing absolute spikes
        // on black is the per-sample clamp's job, not this spatial pass's.
        let mut pixels = vec![Vec3::ZERO; 9];
        pixels[4] = Vec3::splat(10.0);
        let film = Film::from_pixels(3, 3, pixels);
        let filter = OutlierFilter::MedianGuided {
            threshold: 4.0,
            radius: 1,
        };
        let out = filter.apply(&film);
        assert_eq!(out.pixel(1, 1), Vec3::splat(10.0));
    }

    #[test]
    fn even_window_median_averages_the_two_central_samples() {
        let mut window = [1.0f32, 3.0, 5.0, 9.0];
        let median = neighbourhood_median(&mut window);
        assert!((median - 4.0).abs() < 1.0e-6, "median of 3 and 5 is 4");
    }

    #[test]
    fn filter_is_deterministic() {
        let mut pixels = vec![Vec3::splat(1.0); 9];
        pixels[4] = Vec3::splat(100.0);
        let film = Film::from_pixels(3, 3, pixels);
        let filter = OutlierFilter::MedianGuided {
            threshold: 4.0,
            radius: 1,
        };
        assert_eq!(filter.apply(&film).pixels(), filter.apply(&film).pixels());
    }
}
