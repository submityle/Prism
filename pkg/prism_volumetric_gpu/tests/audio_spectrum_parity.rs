//! Real-device parity for the audio-spectrum twin:
//! [`GpuAudioSpectrum`](prism_volumetric_gpu::audio_spectrum::GpuAudioSpectrum)
//! must reproduce the `CPU` golden
//! [`audio_spectrum`](prism_render_architecture::particle::audio_spectrum)
//! across the bin geometry
//! ([`Spectrum::bin_width_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_width_hz),
//! [`Spectrum::nyquist_hz`](prism_render_architecture::particle::audio_spectrum::Spectrum::nyquist_hz)
//! and
//! [`Spectrum::hz_to_bin`](prism_render_architecture::particle::audio_spectrum::Spectrum::hz_to_bin)),
//! the energy aggregation
//! ([`Spectrum::bin_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::bin_energy),
//! [`Spectrum::band_energy`](prism_render_architecture::particle::audio_spectrum::Spectrum::band_energy)
//! and
//! [`Spectrum::low_mid_high`](prism_render_architecture::particle::audio_spectrum::Spectrum::low_mid_high)),
//! the linear attack/release envelope follower
//! ([`EnvelopeState::advance`](prism_render_architecture::particle::audio_spectrum::EnvelopeState::advance)),
//! the spectral-flux onset step
//! ([`OnsetDetector::update`](prism_render_architecture::particle::audio_spectrum::OnsetDetector::update)),
//! and the linear parameter remaps
//! ([`map_range`](prism_render_architecture::particle::audio_spectrum::map_range)
//! and
//! [`normalize`](prism_render_architecture::particle::audio_spectrum::normalize)).
//!
//! The fixtures use non-zero sample rates, window sizes, time constants and
//! reference energies, bins written as simple decimals, and frequencies that
//! sit comfortably inside a bin rather than on a `floor` boundary, so every
//! continuous quantity stays far from any degeneracy. The `band_energy` fixture
//! receives host-located endpoint bins (the honest-host boundary), and the
//! envelope/onset fixtures keep the frame delta below the time constant so the
//! blend coefficient never snaps. All fixtures stay pure and need no external
//! math library and no transcendental math.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The continuous answers (bin widths, summed energies, smoothed envelopes,
//! flux, triggers and remapped scalars) thread through multiplies and adds, so
//! they are compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`). The discrete answers (the located bin index and the
//! onset verdict) are compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::audio_spectrum`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::audio_spectrum::{
    map_range, normalize, EnvelopeParams, EnvelopeState, OnsetDetector, OnsetParams, Spectrum,
};
use prism_volumetric_gpu::audio_spectrum::{
    AudioSpectrumQuery, AudioSpectrumResult, GpuAudioSpectrum,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_audio_spectrum_bin_width_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping audio_spectrum parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    // 8-bin spectrum at 16 kHz over a 16-sample window -> 1000 Hz per bin.
    let bins = [0.0f32; 8];
    let s = Spectrum::new(&bins, 16_000.0, 16);
    let q = AudioSpectrumQuery::BinWidthHz {
        sample_rate_hz: 16_000.0,
        fft_size: 16,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let cpu = s.bin_width_hz();
    let AudioSpectrumResult::BinWidthHz { hz_per_bin } = got[0] else {
        panic!("expected a BinWidthHz result, got {:?}", got[0]);
    };
    assert!(
        approx(hz_per_bin, cpu),
        "bin width mismatch: gpu {hz_per_bin} vs cpu {cpu}"
    );
    assert!(cpu > 1.0, "fixture should have a non-trivial bin width");
}

#[test]
fn gpu_audio_spectrum_nyquist_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    let bins = [0.0f32; 8];
    let s = Spectrum::new(&bins, 44_100.0, 1024);
    let q = AudioSpectrumQuery::NyquistHz {
        sample_rate_hz: 44_100.0,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = s.nyquist_hz();
    let AudioSpectrumResult::NyquistHz { hz } = got[0] else {
        panic!("expected a NyquistHz result, got {:?}", got[0]);
    };
    assert!(approx(hz, cpu), "nyquist mismatch: gpu {hz} vs cpu {cpu}");
}

#[test]
fn gpu_audio_spectrum_hz_to_bin_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    let bins = [0.0f32; 8];
    let s = Spectrum::new(&bins, 16_000.0, 16); // 1000 Hz/bin
                                                // 3700 Hz sits inside bin 3 (floor(3.7) = 3), away from any edge.
    let hz = 3700.0;
    let q = AudioSpectrumQuery::HzToBin {
        sample_rate_hz: 16_000.0,
        fft_size: 16,
        bin_count: 8,
        hz,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = s.hz_to_bin(hz);
    let AudioSpectrumResult::HzToBin { bin } = got[0] else {
        panic!("expected a HzToBin result, got {:?}", got[0]);
    };
    assert_eq!(bin as usize, cpu, "located bin mismatch");
    assert_eq!(bin, 3, "fixture should locate bin 3");
}

#[test]
fn gpu_audio_spectrum_bin_energy_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    let bins = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let s = Spectrum::new(&bins, 16_000.0, 16);
    let index = 3u32;
    let q = AudioSpectrumQuery::BinEnergy {
        bins: bins.to_vec(),
        index,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = s.bin_energy(index as usize);
    let AudioSpectrumResult::BinEnergy { energy } = got[0] else {
        panic!("expected a BinEnergy result, got {:?}", got[0]);
    };
    assert!(
        approx(energy, cpu),
        "bin energy mismatch: gpu {energy} vs cpu {cpu}"
    );
    assert!(cpu > 0.0, "fixture bin should be non-zero");
}

#[test]
fn gpu_audio_spectrum_band_energy_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    let bins = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let s = Spectrum::new(&bins, 16_000.0, 16);
    // Host supplies the located endpoint bins (the honest-host boundary); the
    // twin sums bins[1..=4] inclusively.
    let lo_bin = 1u32;
    let hi_bin = 4u32;
    let q = AudioSpectrumQuery::BandEnergy {
        bins: bins.to_vec(),
        lo_bin,
        hi_bin,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    // Reference inclusive sum over the same host-located endpoints.
    let cpu: f32 = (lo_bin..=hi_bin).map(|i| s.bin_energy(i as usize)).sum();
    let AudioSpectrumResult::BandEnergy { energy } = got[0] else {
        panic!("expected a BandEnergy result, got {:?}", got[0]);
    };
    assert!(
        approx(energy, cpu),
        "band energy mismatch: gpu {energy} vs cpu {cpu}"
    );
    assert!(
        cpu > 1.0,
        "fixture band should accumulate a non-trivial sum"
    );
}

#[test]
fn gpu_audio_spectrum_low_mid_high_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    let bins = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let s = Spectrum::new(&bins, 16_000.0, 16); // 1000 Hz/bin, nyquist 8000
    let q = AudioSpectrumQuery::LowMidHigh {
        bins: bins.to_vec(),
        sample_rate_hz: 16_000.0,
        fft_size: 16,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = s.low_mid_high();
    let AudioSpectrumResult::LowMidHigh { low, mid, high } = got[0] else {
        panic!("expected a LowMidHigh result, got {:?}", got[0]);
    };
    assert!(
        approx(low, cpu.x),
        "low band mismatch: gpu {low} vs cpu {}",
        cpu.x
    );
    assert!(
        approx(mid, cpu.y),
        "mid band mismatch: gpu {mid} vs cpu {}",
        cpu.y
    );
    assert!(
        approx(high, cpu.z),
        "high band mismatch: gpu {high} vs cpu {}",
        cpu.z
    );
}

#[test]
fn gpu_audio_spectrum_envelope_advance_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    // Rising toward 0.8 with dt (0.02) well below attack (0.1): coeff = 0.2, so
    // the value lands at 0.16 without snapping.
    let value = 0.0f32;
    let target = 0.8f32;
    let params = EnvelopeParams::new(0.1, 0.1);
    let dt = 0.02f32;
    let q = AudioSpectrumQuery::EnvelopeAdvance {
        value,
        target,
        attack_seconds: 0.1,
        release_seconds: 0.1,
        dt,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let mut env = EnvelopeState::new(value);
    let cpu = env.advance(target, params, dt);
    let AudioSpectrumResult::EnvelopeAdvance { value: gpu_value } = got[0] else {
        panic!("expected an EnvelopeAdvance result, got {:?}", got[0]);
    };
    assert!(
        approx(gpu_value, cpu),
        "envelope mismatch: gpu {gpu_value} vs cpu {cpu}"
    );
    // The fixture must not snap: it should land strictly between start and target.
    assert!(cpu > value + ABS_EPS && cpu < target - ABS_EPS);
}

#[test]
fn gpu_audio_spectrum_onset_update_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    // A sudden rise from baseline 0.3 to 2.0 fires an onset; dt (0.016) stays
    // below smoothing (0.2) so the baseline advances without snapping.
    let energy = 2.0f32;
    let running_avg = 0.3f32;
    let dt = 0.016f32;
    let params = OnsetParams::new(0.2, 0.5, 1.0);
    let q = AudioSpectrumQuery::OnsetUpdate {
        energy,
        running_avg,
        threshold: 0.5,
        sensitivity: 1.0,
        smoothing_seconds: 0.2,
        dt,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let mut det = OnsetDetector::new(params, running_avg);
    let cpu = det.update(energy, dt);
    let AudioSpectrumResult::OnsetUpdate {
        is_onset,
        flux,
        trigger,
        running_avg: gpu_avg,
    } = got[0]
    else {
        panic!("expected an OnsetUpdate result, got {:?}", got[0]);
    };
    assert_eq!(is_onset, cpu.is_onset, "onset verdict mismatch");
    assert!(is_onset, "fixture should fire an onset");
    assert!(
        approx(flux, cpu.flux),
        "flux mismatch: gpu {flux} vs cpu {}",
        cpu.flux
    );
    assert!(
        approx(trigger, cpu.trigger),
        "trigger mismatch: gpu {trigger} vs cpu {}",
        cpu.trigger
    );
    assert!(
        approx(gpu_avg, det.running_avg),
        "running avg mismatch: gpu {gpu_avg} vs cpu {}",
        det.running_avg
    );
}

#[test]
fn gpu_audio_spectrum_map_range_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    // 5 maps to the midpoint of [0, 10] -> [0, 100] = 50, away from any clamp.
    let x = 5.0f32;
    let q = AudioSpectrumQuery::MapRange {
        x,
        in_lo: 0.0,
        in_hi: 10.0,
        out_lo: 0.0,
        out_hi: 100.0,
    };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = map_range(x, 0.0, 10.0, 0.0, 100.0);
    let AudioSpectrumResult::MapRange { value } = got[0] else {
        panic!("expected a MapRange result, got {:?}", got[0]);
    };
    assert!(
        approx(value, cpu),
        "map_range mismatch: gpu {value} vs cpu {cpu}"
    );
}

#[test]
fn gpu_audio_spectrum_normalize_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    // 2.5 against reference 5 -> 0.5, strictly inside the unit interval.
    let energy = 2.5f32;
    let reference = 5.0f32;
    let q = AudioSpectrumQuery::Normalize { energy, reference };
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let cpu = normalize(energy, reference);
    let AudioSpectrumResult::Normalize { value } = got[0] else {
        panic!("expected a Normalize result, got {:?}", got[0]);
    };
    assert!(
        approx(value, cpu),
        "normalize mismatch: gpu {value} vs cpu {cpu}"
    );
    assert!(value > ABS_EPS && value < 1.0 - ABS_EPS);
}

#[test]
fn gpu_audio_spectrum_batch_matches_elementwise() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    // A mixed batch exercises the one-thread-per-query flattening; each result
    // must be independent of its neighbours.
    let bins = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let s = Spectrum::new(&bins, 16_000.0, 16);
    let batch = [
        AudioSpectrumQuery::BinWidthHz {
            sample_rate_hz: 16_000.0,
            fft_size: 16,
        },
        AudioSpectrumQuery::BinEnergy {
            bins: bins.to_vec(),
            index: 5,
        },
        AudioSpectrumQuery::LowMidHigh {
            bins: bins.to_vec(),
            sample_rate_hz: 16_000.0,
            fft_size: 16,
        },
        AudioSpectrumQuery::MapRange {
            x: 2.5,
            in_lo: 0.0,
            in_hi: 10.0,
            out_lo: 0.0,
            out_hi: 100.0,
        },
    ];
    let got = gpu.evaluate(&ctx, &batch);
    assert_eq!(got.len(), batch.len(), "one result per query");

    let cpu_width = s.bin_width_hz();
    let AudioSpectrumResult::BinWidthHz { hz_per_bin } = got[0] else {
        panic!("expected a BinWidthHz result, got {:?}", got[0]);
    };
    assert!(
        approx(hz_per_bin, cpu_width),
        "batch bin width mismatch: gpu {hz_per_bin} vs cpu {cpu_width}"
    );

    let cpu_energy = s.bin_energy(5);
    let AudioSpectrumResult::BinEnergy { energy } = got[1] else {
        panic!("expected a BinEnergy result, got {:?}", got[1]);
    };
    assert!(
        approx(energy, cpu_energy),
        "batch bin energy mismatch: gpu {energy} vs cpu {cpu_energy}"
    );

    let cpu_lmh = s.low_mid_high();
    let AudioSpectrumResult::LowMidHigh { low, mid, high } = got[2] else {
        panic!("expected a LowMidHigh result, got {:?}", got[2]);
    };
    assert!(approx(low, cpu_lmh.x) && approx(mid, cpu_lmh.y) && approx(high, cpu_lmh.z));

    let cpu_map = map_range(2.5, 0.0, 10.0, 0.0, 100.0);
    let AudioSpectrumResult::MapRange { value } = got[3] else {
        panic!("expected a MapRange result, got {:?}", got[3]);
    };
    assert!(
        approx(value, cpu_map),
        "batch map_range mismatch: gpu {value} vs cpu {cpu_map}"
    );
}

#[test]
fn gpu_audio_spectrum_empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAudioSpectrum::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.evaluate(&ctx, &[]).is_empty());
}
