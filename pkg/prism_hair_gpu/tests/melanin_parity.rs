//! Real-device parity for the isolated melanin-absorption twin:
//! [`GpuHairMelanin`] must reproduce the `CPU` golden
//! [`reference_absorption`](prism_hair_gpu::melanin::reference_absorption)
//! (built on
//! [`melanin_absorption`](prism_render_architecture::hair::melanin::melanin_absorption))
//! for a batch of pigment profiles, folding each fibre's eumelanin/pheomelanin
//! concentrations into an RGB `sigma_a` independently. The suite drives zero
//! pigment, the two unit-concentration spectra, a mixed profile, the four
//! representative natural hair colours, the negative/non-finite clamp, the empty
//! no-op, and a large multi-workgroup batch that crosses the 64-wide dispatch
//! boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each channel is a two-term multiply-add the scalar reference may leave
//! separate while a `GPU` fuses it, so every component is asserted within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3`. No `sin`/`cos` appears anywhere; all
//! inputs are explicit literals or the exported natural-colour profiles.
//!
//! Provenance: `Chiang` 2016 / `pbrt` melanin pigment parameterisation plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::melanin::{reference_absorption, GpuHairMelanin};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::melanin::{MelaninProfile, NaturalHairColor};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Asserts two scalars agree within the documented fma tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs = (got - want).abs();
    let rel = abs / want.abs().max(1.0);
    assert!(
        abs < 1e-4 || rel < 1e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Asserts a whole batch of RGB absorptions matches the `CPU` golden fibre by
/// fibre and channel by channel.
fn assert_batch(got: &[[f32; 3]], profiles: &[MelaninProfile]) {
    assert_eq!(got.len(), profiles.len(), "one absorption per fibre");
    for (i, (&out, &profile)) in got.iter().zip(profiles.iter()).enumerate() {
        let want = reference_absorption(profile);
        assert_close(out[0], want[0], &format!("fibre {i} r"));
        assert_close(out[1], want[1], &format!("fibre {i} g"));
        assert_close(out[2], want[2], &format!("fibre {i} b"));
    }
}

/// Dispatches one batch through the device twin.
fn run(ctx: &GpuContext, profiles: &[MelaninProfile]) -> Vec<[f32; 3]> {
    GpuHairMelanin::new(ctx).eval(ctx, profiles)
}

#[test]
fn zero_pigment_is_zero_absorption() {
    let Some(ctx) = context_or_skip("zero_pigment_is_zero_absorption") else {
        return;
    };
    let profiles = [MelaninProfile::new(0.0, 0.0)];
    assert_batch(&run(&ctx, &profiles), &profiles);
}

#[test]
fn unit_eumelanin_matches_golden() {
    let Some(ctx) = context_or_skip("unit_eumelanin_matches_golden") else {
        return;
    };
    let profiles = [MelaninProfile::new(1.0, 0.0)];
    assert_batch(&run(&ctx, &profiles), &profiles);
}

#[test]
fn unit_pheomelanin_matches_golden() {
    let Some(ctx) = context_or_skip("unit_pheomelanin_matches_golden") else {
        return;
    };
    let profiles = [MelaninProfile::new(0.0, 1.0)];
    assert_batch(&run(&ctx, &profiles), &profiles);
}

#[test]
fn mixed_pigment_matches_golden() {
    let Some(ctx) = context_or_skip("mixed_pigment_matches_golden") else {
        return;
    };
    let profiles = [MelaninProfile::new(2.0, 3.0)];
    assert_batch(&run(&ctx, &profiles), &profiles);
}

#[test]
fn natural_colours_match_golden() {
    let Some(ctx) = context_or_skip("natural_colours_match_golden") else {
        return;
    };
    let profiles = [
        NaturalHairColor::Black.profile(),
        NaturalHairColor::Brown.profile(),
        NaturalHairColor::Blond.profile(),
        NaturalHairColor::Red.profile(),
    ];
    assert_batch(&run(&ctx, &profiles), &profiles);
}

#[test]
fn negative_and_non_finite_clamp_to_zero() {
    let Some(ctx) = context_or_skip("negative_and_non_finite_clamp_to_zero") else {
        return;
    };
    // Negative, NaN and +inf concentrations must each collapse to 0, so every
    // fibre reads as zero absorption exactly like the golden.
    let profiles = [
        MelaninProfile::new(-5.0, -1.0),
        MelaninProfile::new(f32::NAN, f32::INFINITY),
        MelaninProfile::new(f32::NEG_INFINITY, f32::NAN),
    ];
    let got = run(&ctx, &profiles);
    assert_batch(&got, &profiles);
    for (i, out) in got.iter().enumerate() {
        assert!(
            out.iter().all(|c| c.is_finite()),
            "fibre {i} absorption must stay finite, got {out:?}"
        );
    }
}

#[test]
fn empty_batch_is_noop() {
    let Some(ctx) = context_or_skip("empty_batch_is_noop") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no absorptions");
}

#[test]
fn large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 130 fibres span three 64-wide workgroups; sweep a deterministic range of
    // eumelanin/pheomelanin mixes, with every 9th fibre set unphysical (negative
    // or non-finite) so the clamp is exercised across the dispatch boundary.
    let mut profiles = Vec::new();
    for k in 0u32..130 {
        let eu = ((k % 11) as f32) * 0.7;
        let pheo = ((k % 7) as f32) * 0.3;
        let profile = if (k % 9) == 0 {
            MelaninProfile::new(-eu, f32::NAN)
        } else {
            MelaninProfile::new(eu, pheo)
        };
        profiles.push(profile);
    }
    assert_batch(&run(&ctx, &profiles), &profiles);
}
