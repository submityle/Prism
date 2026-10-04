//! End-to-end integration coverage for the `prism_audio_output` delivery chain.
//!
//! Exercises the three composable delivery stages against their real public
//! API: the ITU-R BS.775 [`DownmixMatrix`] fold-down, the Linkwitz-Riley
//! [`BassManager`] LFE crossover, and the [`OutputProfile`] delivery presets.
//! Every test drives real `AudioBuffer` blocks through the stage and asserts a
//! concrete signal, routing, or parameter invariant rather than a smoke check.
//!
//! # Provenance
//!
//! Original work. Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Dolby, MPEG, Google Resonance Audio, Web Audio, or Project Acoustics
//! source or derived code, and no AI/ML. The downmix coefficients and bass
//! management follow publicly published standards (ITU-R BS.775 fold-down,
//! Linkwitz-Riley crossovers) only at the level of their published ideas, not
//! any third-party implementation.
//!
//! # Relationship
//!
//! Covers `docs/prism_audio_engine_design_zh.md` section 48 (输出渲染链与母带
//! 交付档). Depends on the crate under test plus `prism_audio_core`'s
//! `AudioBuffer`/`ChannelLayout`, which this crate re-uses rather than
//! reimplementing.

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_output::bass_management::{BassManager, BassManagerParams};
use prism_audio_output::downmix::{
    DownmixMatrix, DownmixNode, DownmixOptions, MINUS_3DB, MINUS_6DB, SURROUND_TO_MONO,
};
use prism_audio_output::profiles::OutputProfile;

/// Branch-free absolute value, avoiding the std float-math lint in tests.
fn fabs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// Approximate float equality within a fixed tolerance.
fn close(a: f32, b: f32, eps: f32) -> bool {
    fabs(a - b) < eps
}

/// Sum of squares (energy) of a channel's active samples.
fn energy(buf: &AudioBuffer, channel: usize) -> f32 {
    let mut acc = 0.0f32;
    for &s in buf.channel(channel) {
        acc += s * s;
    }
    acc
}

/// Fills every channel of `buf` with the constant `value`.
fn fill_dc(buf: &mut AudioBuffer, value: f32) {
    let channels = buf.channels();
    for ch in 0..channels {
        for s in buf.channel_mut(ch).iter_mut() {
            *s = value;
        }
    }
}

const EPS: f32 = 1.0e-6;

#[test]
fn downmix_5_1_to_stereo_matches_bs775_coefficients() {
    let matrix = DownmixMatrix::new(
        ChannelLayout::Surround5_1,
        ChannelLayout::Stereo,
        DownmixOptions::default(),
    )
    .expect("5.1 -> stereo is a supported conversion");

    assert_eq!(matrix.input_layout(), ChannelLayout::Surround5_1);
    assert_eq!(matrix.output_layout(), ChannelLayout::Stereo);

    // Left row: FL at unity, C and SL at -3 dB, nothing from FR/SR.
    assert!(close(matrix.coeff(0, 0), 1.0, EPS));
    assert!(close(matrix.coeff(0, 2), MINUS_3DB, EPS));
    assert!(close(matrix.coeff(0, 4), MINUS_3DB, EPS));
    assert!(close(matrix.coeff(0, 1), 0.0, EPS));
    assert!(close(matrix.coeff(0, 5), 0.0, EPS));

    // Right row: FR at unity, C and SR at -3 dB.
    assert!(close(matrix.coeff(1, 1), 1.0, EPS));
    assert!(close(matrix.coeff(1, 2), MINUS_3DB, EPS));
    assert!(close(matrix.coeff(1, 5), MINUS_3DB, EPS));
}

#[test]
fn downmix_drops_lfe_by_default_but_folds_it_when_requested() {
    let dropped = DownmixMatrix::new(
        ChannelLayout::Surround5_1,
        ChannelLayout::Stereo,
        DownmixOptions::default(),
    )
    .expect("supported conversion");
    // LFE lives at input index 3; default options drop it from both outputs.
    assert!(close(dropped.coeff(0, 3), 0.0, EPS));
    assert!(close(dropped.coeff(1, 3), 0.0, EPS));

    let folded = DownmixMatrix::new(
        ChannelLayout::Surround5_1,
        ChannelLayout::Stereo,
        DownmixOptions {
            include_lfe: true,
            lfe_gain_db: 0.0,
        },
    )
    .expect("supported conversion");
    // Unity fold-in puts the LFE into both front channels at 0 dB (linear 1.0).
    assert!(close(folded.coeff(0, 3), 1.0, EPS));
    assert!(close(folded.coeff(1, 3), 1.0, EPS));
}

#[test]
fn downmix_apply_routes_center_into_both_fronts() {
    let matrix = DownmixMatrix::new(
        ChannelLayout::Surround5_1,
        ChannelLayout::Stereo,
        DownmixOptions::default(),
    )
    .expect("supported conversion");

    let mut input = AudioBuffer::new(ChannelLayout::Surround5_1, 64);
    // Put a pure centre-channel (index 2) signal in; everything else silent.
    for s in input.channel_mut(2).iter_mut() {
        *s = 1.0;
    }
    let mut output = AudioBuffer::new(ChannelLayout::Stereo, 64);
    matrix.apply(&input, &mut output);

    // Centre folds equally into both fronts at -3 dB.
    for frame in 0..output.active_frames() {
        assert!(close(output.channel(0)[frame], MINUS_3DB, EPS));
        assert!(close(output.channel(1)[frame], MINUS_3DB, EPS));
    }
    assert!(close(energy(&output, 0), energy(&output, 1), EPS));
}

#[test]
fn downmix_identity_preserves_every_channel() {
    let matrix = DownmixMatrix::new(
        ChannelLayout::Stereo,
        ChannelLayout::Stereo,
        DownmixOptions::default(),
    )
    .expect("identity is always supported");

    let mut input = AudioBuffer::new(ChannelLayout::Stereo, 32);
    for (frame, s) in input.channel_mut(0).iter_mut().enumerate() {
        *s = frame as f32 * 0.01;
    }
    for s in input.channel_mut(1).iter_mut() {
        *s = -0.25;
    }
    let mut output = AudioBuffer::new(ChannelLayout::Stereo, 32);
    matrix.apply(&input, &mut output);

    for ch in 0..2 {
        for (frame, &s) in output.channel(ch).iter().enumerate() {
            assert!(close(s, input.channel(ch)[frame], EPS));
        }
    }
}

#[test]
fn downmix_stereo_to_mono_is_level_preserving_average() {
    let matrix = DownmixMatrix::new(
        ChannelLayout::Stereo,
        ChannelLayout::Mono,
        DownmixOptions::default(),
    )
    .expect("supported conversion");
    assert!(close(matrix.coeff(0, 0), MINUS_6DB, EPS));
    assert!(close(matrix.coeff(0, 1), MINUS_6DB, EPS));

    let mut input = AudioBuffer::new(ChannelLayout::Stereo, 16);
    fill_dc(&mut input, 0.5);
    let mut output = AudioBuffer::new(ChannelLayout::Mono, 16);
    matrix.apply(&input, &mut output);
    // 0.5 * 0.5 + 0.5 * 0.5 = 0.5 preserved on the mono fold.
    for &s in output.channel(0) {
        assert!(close(s, 0.5, EPS));
    }
}

#[test]
fn downmix_surround_to_mono_uses_published_weights() {
    let matrix = DownmixMatrix::new(
        ChannelLayout::Surround5_1,
        ChannelLayout::Mono,
        DownmixOptions::default(),
    )
    .expect("supported conversion");
    assert!(close(matrix.coeff(0, 0), MINUS_6DB, EPS));
    assert!(close(matrix.coeff(0, 1), MINUS_6DB, EPS));
    assert!(close(matrix.coeff(0, 2), MINUS_3DB, EPS));
    assert!(close(matrix.coeff(0, 4), SURROUND_TO_MONO, EPS));
    assert!(close(matrix.coeff(0, 5), SURROUND_TO_MONO, EPS));
}

#[test]
fn downmix_rejects_unsupported_upmix() {
    // Mono -> stereo is an upmix, not a published fold-down: unsupported.
    assert!(
        DownmixMatrix::new(
            ChannelLayout::Mono,
            ChannelLayout::Stereo,
            DownmixOptions::default(),
        )
        .is_none()
    );
}

#[test]
fn downmix_node_from_layouts_agrees_with_matrix() {
    let opts = DownmixOptions::default();
    let node = DownmixNode::from_layouts(ChannelLayout::Quad, ChannelLayout::Stereo, opts)
        .expect("quad -> stereo is supported");
    let matrix = DownmixMatrix::new(ChannelLayout::Quad, ChannelLayout::Stereo, opts)
        .expect("quad -> stereo is supported");
    assert_eq!(node.matrix().input_layout(), matrix.input_layout());
    assert_eq!(node.matrix().output_layout(), matrix.output_layout());
    for o in 0..2 {
        for i in 0..4 {
            assert!(close(node.matrix().coeff(o, i), matrix.coeff(o, i), EPS));
        }
    }
    // A node for an unsupported pair is also rejected.
    assert!(DownmixNode::from_layouts(ChannelLayout::Mono, ChannelLayout::Quad, opts).is_none());
}

#[test]
fn bass_manager_redirects_low_band_into_the_lfe() {
    let params = BassManagerParams::default();
    let mut mgr = BassManager::new(48_000, ChannelLayout::Surround5_1, 512, &params);
    assert!(mgr.has_lfe());
    assert_eq!(mgr.layout(), ChannelLayout::Surround5_1);

    let mut buf = AudioBuffer::new(ChannelLayout::Surround5_1, 512);
    // DC (0 Hz) lives entirely below the crossover: it must leave the mains and
    // accumulate, with the calibration gain, into the silent LFE (index 3).
    for ch in [0usize, 1, 2, 4, 5] {
        for s in buf.channel_mut(ch).iter_mut() {
            *s = 0.5;
        }
    }
    let mains_before = energy(&buf, 0);
    let lfe_before = energy(&buf, 3);
    assert!(close(lfe_before, 0.0, EPS));

    mgr.process(&mut buf);

    // The front-left main is high-passed, so its steady-state DC energy drops.
    assert!(energy(&buf, 0) < mains_before);
    // The LFE now carries the summed, gained low band.
    assert!(energy(&buf, 3) > lfe_before);
    assert!(energy(&buf, 3) > energy(&buf, 0));
}

#[test]
fn bass_manager_bypass_is_bit_transparent() {
    let params = BassManagerParams {
        bypassed: true,
        ..BassManagerParams::default()
    };
    let mut mgr = BassManager::new(48_000, ChannelLayout::Surround5_1, 256, &params);
    assert!(mgr.is_bypassed());

    let mut buf = AudioBuffer::new(ChannelLayout::Surround5_1, 256);
    for ch in 0..buf.channels() {
        for (frame, s) in buf.channel_mut(ch).iter_mut().enumerate() {
            *s = (frame as f32).mul_add(0.001, ch as f32 * 0.1);
        }
    }
    let before = buf.clone();
    mgr.process(&mut buf);
    for ch in 0..buf.channels() {
        for (frame, &s) in buf.channel(ch).iter().enumerate() {
            assert_eq!(s.to_bits(), before.channel(ch)[frame].to_bits());
        }
    }
}

#[test]
fn bass_manager_without_lfe_channel_is_a_noop() {
    let params = BassManagerParams::default();
    let mut mgr = BassManager::new(48_000, ChannelLayout::Stereo, 128, &params);
    assert!(!mgr.has_lfe());

    let mut buf = AudioBuffer::new(ChannelLayout::Stereo, 128);
    fill_dc(&mut buf, 0.3);
    let before = buf.clone();
    mgr.process(&mut buf);
    for ch in 0..buf.channels() {
        for (frame, &s) in buf.channel(ch).iter().enumerate() {
            assert_eq!(s.to_bits(), before.channel(ch)[frame].to_bits());
        }
    }
}

#[test]
fn bass_manager_reset_restores_deterministic_output() {
    let params = BassManagerParams::default();
    let mut mgr = BassManager::new(48_000, ChannelLayout::Surround5_1, 256, &params);

    let make_block = || {
        let mut buf = AudioBuffer::new(ChannelLayout::Surround5_1, 256);
        for ch in [0usize, 1, 2, 4, 5] {
            for (frame, s) in buf.channel_mut(ch).iter_mut().enumerate() {
                *s = if frame % 2 == 0 { 0.4 } else { -0.4 };
            }
        }
        buf
    };

    let mut first = make_block();
    mgr.process(&mut first);

    mgr.reset();
    let mut second = make_block();
    mgr.process(&mut second);

    for ch in 0..first.channels() {
        for frame in 0..first.active_frames() {
            assert_eq!(
                first.channel(ch)[frame].to_bits(),
                second.channel(ch)[frame].to_bits()
            );
        }
    }
}

#[test]
fn profiles_increase_in_aggression_from_home_theater_to_night() {
    let ht = OutputProfile::HomeTheater.params();
    let tv = OutputProfile::Tv.params();
    let night = OutputProfile::Night.params();

    assert!(!ht.compresses());
    assert!(tv.compresses());
    assert!(night.compresses());
    assert!(ht.compression_ratio < tv.compression_ratio);
    assert!(tv.compression_ratio < night.compression_ratio);
    // The night preset lifts quiet passages hardest, so its threshold is lowest.
    assert!(night.compression_threshold_db < tv.compression_threshold_db);
}

#[test]
fn headphone_profile_requests_binaural_and_maps_loudness_params() {
    let hp = OutputProfile::Headphone.params();
    assert!(hp.binaural);

    // The resolved normalizer carries the preset's delivery target/ceiling and
    // the externally measured loudness/true-peak.
    let measured_lufs = -23.0;
    let measured_tp = -0.5;
    let lp = hp.loudness_params(measured_lufs, measured_tp);
    assert!(close(lp.target_lufs, hp.target_lufs, EPS));
    assert!(close(lp.max_true_peak_dbtp, hp.true_peak_ceiling_dbtp, EPS));
    assert!(close(lp.measured_lufs, measured_lufs, EPS));
    assert!(close(lp.measured_true_peak_dbtp, measured_tp, EPS));
}
