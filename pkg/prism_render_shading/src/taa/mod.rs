//! Backend-neutral CPU reference for **temporal anti-aliasing** (TAA).
//!
//! Prism is a visibility-buffer deferred renderer, so MSAA (which needs
//! per-sample shading of a forward G-buffer) is impractical and TAA is the AAA
//! anti-aliasing path. TAA has two halves, each mirrored bit-for-bit by a GPU
//! twin:
//!
//! * [`jitter`] — the per-frame sub-pixel camera jitter (Halton(2, 3)) that
//!   turns the temporal history into a supersampler.
//! * [`resolve`] — the neighbourhood-clipped, luminance-weighted blend of the
//!   jittered current frame with the motion-reprojected history
//!   (`taa_resolve.wesl`).
//!
//! The history reprojection reuses the same motion-vector G-buffer and
//! neighbourhood-clip machinery as the SSR temporal pass (see
//! [`super::screen_space::temporal`]); TAA differs only in that it runs
//! full-screen on the composited scene colour rather than on the SSR reflection
//! buffer.

mod jitter;
mod resolve;

pub use jitter::{halton, taa_jitter, DEFAULT_TAA_JITTER_LEN};
pub use resolve::{resolve_taa, rgb_to_ycocg, tonemap_weight, ycocg_to_rgb, TaaParams};

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec2;

    #[test]
    fn halton_matches_known_radical_inverses() {
        // Base 2: 1/2, 1/4, 3/4, 1/8 ...
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
        assert!((halton(4, 2) - 0.125).abs() < 1e-6);
        // Base 3: 1/3, 2/3, 1/9 ...
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
        assert!((halton(3, 3) - 1.0 / 9.0).abs() < 1e-6);
        // Index 0 is the sequence origin.
        assert_eq!(halton(0, 2), 0.0);
    }

    #[test]
    fn jitter_is_centred_and_bounded() {
        let len = DEFAULT_TAA_JITTER_LEN;
        let mut mean = Vec2::ZERO;
        for frame in 0..len as u64 {
            let j = taa_jitter(frame, len);
            assert!(
                j.x >= -0.5 && j.x < 0.5 && j.y >= -0.5 && j.y < 0.5,
                "jitter must stay in [-0.5, 0.5): {j:?}"
            );
            mean += j;
        }
        mean /= len as f32;
        // Halton is low-discrepancy, so the mean offset over a full cycle sits
        // near the pixel centre (no net image shift).
        assert!(
            mean.length() < 0.15,
            "the jitter cycle must average near the pixel centre: {mean:?}"
        );
    }

    #[test]
    fn jitter_cycle_wraps_on_sequence_length() {
        let len = DEFAULT_TAA_JITTER_LEN;
        assert_eq!(taa_jitter(0, len), taa_jitter(len as u64, len));
        assert_eq!(taa_jitter(3, len), taa_jitter(len as u64 + 3, len));
    }
}
