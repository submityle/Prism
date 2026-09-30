//! Real-time binaural renderer: block convolution of a mono source with a
//! left/right HRIR pair.
//!
//! Once [`crate::interpolation`] has produced a left/right HRIR for the current
//! source direction, [`BinauralRenderer`] convolves the dry mono signal with
//! both responses to produce a two-channel (headphone) output. The convolution
//! is exact and continuous across block boundaries: a persistent input-history
//! delay line carries the trailing samples needed to convolve the start of the
//! next block, which is the standard overlap-save formulation of partitioned
//! block convolution.
//!
//! # Click-free HRIR updates
//!
//! When the source moves, the HRIR changes. Swapping it abruptly clicks, so
//! [`BinauralRenderer::set_hrir`] installs the new response as a *target* and
//! linearly crossfades the convolution output from the previously committed
//! response to the target over a configurable number of samples. Both
//! convolutions share the same input history, so no extra delay-line state is
//! needed.
//!
//! # Real-time contract
//!
//! [`BinauralRenderer::process_block`], [`BinauralRenderer::set_hrir`], and
//! [`BinauralRenderer::reset`] are **allocation free, lock free, and panic
//! free**: they operate entirely on buffers sized once in
//! [`BinauralRenderer::new`] and use saturating length clamps rather than
//! indexing that could panic. All allocation happens in the constructor, off
//! the audio thread.
//!
//! # Determinism
//!
//! The convolution and crossfade are pure multiply-add arithmetic on `f32`
//! (no transcendental calls), so the output is bit-reproducible across targets
//! and can be golden-compared sample-for-sample.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, or Steam Audio source or derived code**. Overlap-save
//! block convolution and linear crossfading are implemented from standard,
//! publicly documented DSP knowledge.

use alloc::vec;
use alloc::vec::Vec;
use prism_audio_core::math::Sample;

/// Default crossfade length (samples) applied when the HRIR is retargeted.
const DEFAULT_CROSSFADE: usize = 64;

/// A stereo (binaural) block convolver for a mono input and an HRIR pair.
///
/// See the [module documentation](self) for the convolution and crossfade
/// model and the real-time contract.
#[derive(Debug, Clone)]
pub struct BinauralRenderer {
    hrir_len: usize,
    max_block: usize,
    // Committed (active) and pending (target) HRIRs, each `hrir_len` samples.
    active_l: Vec<Sample>,
    active_r: Vec<Sample>,
    target_l: Vec<Sample>,
    target_r: Vec<Sample>,
    // Input-history delay line plus the current block, `hrir_len - 1 + max_block`.
    work: Vec<Sample>,
    // Crossfade state: `crossfade_pos == crossfade_len` means idle (no fade).
    crossfade_len: usize,
    crossfade_pos: usize,
}

impl BinauralRenderer {
    /// Creates a renderer for HRIRs of `hrir_len` taps rendering blocks of up
    /// to `max_block` frames.
    ///
    /// Both arguments are clamped to at least `1`. The renderer starts silent
    /// (both HRIRs are all-zero) until an HRIR is installed via
    /// [`BinauralRenderer::set_hrir`] or
    /// [`BinauralRenderer::set_hrir_immediate`]. Allocates; call off the audio
    /// thread.
    #[must_use]
    pub fn new(hrir_len: usize, max_block: usize) -> Self {
        let hrir_len = hrir_len.max(1);
        let max_block = max_block.max(1);
        let crossfade_len = DEFAULT_CROSSFADE.min(max_block).max(1);
        Self {
            hrir_len,
            max_block,
            active_l: vec![0.0; hrir_len],
            active_r: vec![0.0; hrir_len],
            target_l: vec![0.0; hrir_len],
            target_r: vec![0.0; hrir_len],
            work: vec![0.0; (hrir_len - 1) + max_block],
            crossfade_len,
            crossfade_pos: crossfade_len,
        }
    }

    /// The HRIR tap count this renderer was built for.
    #[must_use]
    #[inline]
    pub fn hrir_len(&self) -> usize {
        self.hrir_len
    }

    /// The maximum block size (frames) this renderer was built for.
    #[must_use]
    #[inline]
    pub fn max_block(&self) -> usize {
        self.max_block
    }

    /// Returns `true` if a crossfade to a target HRIR is currently in progress.
    #[must_use]
    #[inline]
    pub fn is_crossfading(&self) -> bool {
        self.crossfade_pos < self.crossfade_len
    }

    /// Sets the crossfade length (samples) used by subsequent
    /// [`BinauralRenderer::set_hrir`] calls. Clamped to `1..=max_block`.
    #[inline]
    pub fn set_crossfade_len(&mut self, samples: usize) {
        self.crossfade_len = samples.clamp(1, self.max_block);
        // Keep the "idle" invariant if we shrank past an in-flight fade.
        if self.crossfade_pos > self.crossfade_len {
            self.crossfade_pos = self.crossfade_len;
        }
    }

    /// Installs an HRIR immediately, with no crossfade.
    ///
    /// Intended for initialisation (or a hard cut). Each slice is copied into
    /// the fixed `hrir_len` store: extra samples are dropped and a short slice
    /// is zero-padded. Real-time safe.
    pub fn set_hrir_immediate(&mut self, left: &[Sample], right: &[Sample]) {
        copy_padded(&mut self.active_l, left);
        copy_padded(&mut self.active_r, right);
        self.target_l.copy_from_slice(&self.active_l);
        self.target_r.copy_from_slice(&self.active_r);
        self.crossfade_pos = self.crossfade_len;
    }

    /// Retargets the HRIR, crossfading from the committed response to the new
    /// one over [`set_crossfade_len`](BinauralRenderer::set_crossfade_len)
    /// samples.
    ///
    /// Each slice is copied into the fixed `hrir_len` store (truncated or
    /// zero-padded as needed). Real-time safe. If a crossfade was already in
    /// progress the fade restarts toward the new target from the previously
    /// committed response.
    pub fn set_hrir(&mut self, left: &[Sample], right: &[Sample]) {
        copy_padded(&mut self.target_l, left);
        copy_padded(&mut self.target_r, right);
        self.crossfade_pos = 0;
    }

    /// Clears the input-history delay line (silences tails) and cancels any
    /// in-flight crossfade by committing the target. Leaves the installed
    /// HRIR otherwise intact. Real-time safe.
    pub fn reset(&mut self) {
        for s in &mut self.work {
            *s = 0.0;
        }
        self.active_l.copy_from_slice(&self.target_l);
        self.active_r.copy_from_slice(&self.target_r);
        self.crossfade_pos = self.crossfade_len;
    }

    /// Convolves `input` (mono) with the current HRIR pair, writing the left
    /// and right channels into `out_l`/`out_r`.
    ///
    /// Processes `frames = min(input.len(), out_l.len(), out_r.len(),
    /// max_block)` samples and returns that count. Real-time safe: no
    /// allocation, no panic.
    pub fn process_block(
        &mut self,
        input: &[Sample],
        out_l: &mut [Sample],
        out_r: &mut [Sample],
    ) -> usize {
        let frames = input
            .len()
            .min(out_l.len())
            .min(out_r.len())
            .min(self.max_block);
        let m = self.hrir_len;
        let base = m - 1;

        // Append the current block after the persistent history tail.
        self.work[base..base + frames].copy_from_slice(&input[..frames]);

        let inv_fade = 1.0 / self.crossfade_len as Sample;
        for i in 0..frames {
            let end = base + i;
            let (al, ar) = convolve_pair(&self.work, end, &self.active_l, &self.active_r, m);
            if self.crossfade_pos < self.crossfade_len {
                let (tl, tr) =
                    convolve_pair(&self.work, end, &self.target_l, &self.target_r, m);
                let t = (((self.crossfade_pos + 1) as Sample) * inv_fade).min(1.0);
                out_l[i] = al + (tl - al) * t;
                out_r[i] = ar + (tr - ar) * t;
                self.crossfade_pos += 1;
                if self.crossfade_pos >= self.crossfade_len {
                    // Commit immediately so the remainder of this block (and
                    // all future blocks) render from the new response.
                    self.active_l.copy_from_slice(&self.target_l);
                    self.active_r.copy_from_slice(&self.target_r);
                }
            } else {
                out_l[i] = al;
                out_r[i] = ar;
            }
        }

        // Slide the history: keep the last `base` input samples for next block.
        self.work.copy_within(frames..frames + base, 0);

        frames
    }
}

/// Convolves both ears at output position `end` in one history pass:
/// `sum_{k=0}^{m-1} h[k] * work[end - k]`.
#[inline]
fn convolve_pair(
    work: &[Sample],
    end: usize,
    hl: &[Sample],
    hr: &[Sample],
    m: usize,
) -> (Sample, Sample) {
    let mut l = 0.0;
    let mut r = 0.0;
    for k in 0..m {
        let x = work[end - k];
        l += hl[k] * x;
        r += hr[k] * x;
    }
    (l, r)
}

/// Copies `src` into `dst`, truncating a longer source and zero-padding a
/// shorter one.
#[inline]
fn copy_padded(dst: &mut [Sample], src: &[Sample]) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
    for s in &mut dst[n..] {
        *s = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Reference direct linear convolution, `y[n] = sum_k h[k] x[n-k]`.
    fn ref_conv(x: &[f32], h: &[f32]) -> Vec<f32> {
        let mut y = vec![0.0f32; x.len()];
        for n in 0..x.len() {
            let mut acc = 0.0;
            for (k, &hk) in h.iter().enumerate() {
                if n >= k {
                    acc += hk * x[n - k];
                }
            }
            y[n] = acc;
        }
        y
    }

    #[test]
    fn identity_hrir_passes_input_through() {
        let mut r = BinauralRenderer::new(1, 8);
        r.set_hrir_immediate(&[1.0], &[0.5]);
        let input = [1.0, -0.5, 0.25, 0.0];
        let mut l = [0.0; 4];
        let mut rr = [0.0; 4];
        let n = r.process_block(&input, &mut l, &mut rr);
        assert_eq!(n, 4);
        assert_eq!(l, input);
        for i in 0..4 {
            assert!(approx(rr[i], input[i] * 0.5, 1e-6));
        }
    }

    #[test]
    fn matches_reference_convolution_single_block() {
        let hl = [0.2, -0.3, 0.5, 0.1];
        let hr = [0.9, 0.0, -0.2, 0.4];
        let x = [1.0, 0.5, -0.25, 0.75, -1.0, 0.3];
        let mut r = BinauralRenderer::new(hl.len(), x.len());
        r.set_hrir_immediate(&hl, &hr);
        let mut l = vec![0.0; x.len()];
        let mut rr = vec![0.0; x.len()];
        r.process_block(&x, &mut l, &mut rr);
        let el = ref_conv(&x, &hl);
        let er = ref_conv(&x, &hr);
        for i in 0..x.len() {
            assert!(approx(l[i], el[i], 1e-5), "L[{i}] {} vs {}", l[i], el[i]);
            assert!(approx(rr[i], er[i], 1e-5), "R[{i}] {} vs {}", rr[i], er[i]);
        }
    }

    #[test]
    fn convolution_is_continuous_across_blocks() {
        let hl = [0.5, 0.25, -0.5, 0.75, 0.1];
        let hr = [0.1, 0.2, 0.3, 0.4, 0.5];
        let x: Vec<f32> = (0..20).map(|i| ((i as f32) * 0.37).sin()).collect();
        let mut r = BinauralRenderer::new(hl.len(), 8);
        r.set_hrir_immediate(&hl, &hr);
        // Render in irregular chunks and stitch.
        let mut l = Vec::new();
        let mut rr = Vec::new();
        for chunk in [5usize, 8, 7] {
            let start = l.len();
            let mut lb = vec![0.0; chunk];
            let mut rb = vec![0.0; chunk];
            r.process_block(&x[start..start + chunk], &mut lb, &mut rb);
            l.extend_from_slice(&lb);
            rr.extend_from_slice(&rb);
        }
        let el = ref_conv(&x, &hl);
        let er = ref_conv(&x, &hr);
        for i in 0..x.len() {
            assert!(approx(l[i], el[i], 1e-4), "L[{i}] {} vs {}", l[i], el[i]);
            assert!(approx(rr[i], er[i], 1e-4), "R[{i}] {} vs {}", rr[i], er[i]);
        }
    }

    #[test]
    fn silence_in_silence_out() {
        let mut r = BinauralRenderer::new(4, 8);
        r.set_hrir_immediate(&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0]);
        let x = [0.0; 6];
        let mut l = [0.0; 6];
        let mut rr = [0.0; 6];
        r.process_block(&x, &mut l, &mut rr);
        assert!(l.iter().all(|&s| s == 0.0));
        assert!(rr.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn crossfade_transitions_active_to_target() {
        let mut r = BinauralRenderer::new(1, 16);
        r.set_crossfade_len(8);
        r.set_hrir_immediate(&[1.0], &[1.0]); // active gain 1.0
        // DC input so output equals the effective gain each sample.
        let x = [1.0; 16];
        let mut l = [0.0; 16];
        let mut rr = [0.0; 16];
        r.set_hrir(&[3.0], &[3.0]); // target gain 3.0
        assert!(r.is_crossfading());
        r.process_block(&x, &mut l, &mut rr);
        // First sample near 1.0, ramps up toward 3.0, then holds at 3.0.
        assert!(l[0] > 1.0 && l[0] < 3.0);
        assert!(l[7] <= 3.0 + 1e-6 && l[7] >= 2.5);
        assert!(approx(l[8], 3.0, 1e-6));
        assert!(approx(l[15], 3.0, 1e-6));
        assert!(!r.is_crossfading());
    }

    #[test]
    fn reset_clears_history() {
        let hl = [0.5, 0.5, 0.5, 0.5];
        let mut r = BinauralRenderer::new(hl.len(), 4);
        r.set_hrir_immediate(&hl, &hl);
        let x = [1.0, 1.0, 1.0, 1.0];
        let mut l = [0.0; 4];
        let mut rr = [0.0; 4];
        r.process_block(&x, &mut l, &mut rr);
        r.reset();
        // After reset, an impulse should produce the HRIR with no leftover tail.
        let imp = [1.0, 0.0, 0.0, 0.0];
        let mut l2 = [0.0; 4];
        let mut r2 = [0.0; 4];
        r.process_block(&imp, &mut l2, &mut r2);
        assert_eq!(l2, hl);
    }

    #[test]
    fn frames_clamped_to_shortest_buffer_and_max_block() {
        let mut r = BinauralRenderer::new(2, 4);
        r.set_hrir_immediate(&[1.0, 0.0], &[1.0, 0.0]);
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut l = [0.0; 3];
        let mut rr = [0.0; 3];
        // out len 3, but max_block 4 and input 6 -> min is 3.
        let n = r.process_block(&x, &mut l, &mut rr);
        assert_eq!(n, 3);
    }

    #[test]
    fn zero_length_input_yields_zero_frames() {
        let mut r = BinauralRenderer::new(4, 8);
        r.set_hrir_immediate(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]);
        let x: [f32; 0] = [];
        let mut l: [f32; 0] = [];
        let mut rr: [f32; 0] = [];
        assert_eq!(r.process_block(&x, &mut l, &mut rr), 0);
    }

    #[test]
    fn set_hrir_zero_pads_short_slice() {
        let mut r = BinauralRenderer::new(4, 4);
        r.set_hrir_immediate(&[1.0], &[1.0]); // padded to [1,0,0,0]
        let imp = [1.0, 0.0, 0.0, 0.0];
        let mut l = [0.0; 4];
        let mut rr = [0.0; 4];
        r.process_block(&imp, &mut l, &mut rr);
        assert_eq!(l, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn output_is_deterministic() {
        let hl = [0.3, -0.6, 0.2, 0.9];
        let x: Vec<f32> = (0..32).map(|i| ((i * 7 % 13) as f32 / 13.0) - 0.5).collect();
        let run = || {
            let mut r = BinauralRenderer::new(hl.len(), 16);
            r.set_hrir_immediate(&hl, &hl);
            let mut out = Vec::new();
            for c in x.chunks(16) {
                let mut lb = vec![0.0; c.len()];
                let mut rb = vec![0.0; c.len()];
                r.process_block(c, &mut lb, &mut rb);
                out.extend_from_slice(&lb);
            }
            out
        };
        assert_eq!(run(), run());
    }
}
