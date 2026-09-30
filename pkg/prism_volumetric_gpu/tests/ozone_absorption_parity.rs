//! Real-device parity for the ozone-absorption twin: [`GpuOzoneAbsorption`]
//! must reproduce the `CPU` golden
//! [`ozone_absorption`](prism_render_architecture::volumetric::spectral::ozone_absorption)
//! across the full wavelength range, including the two band centres and
//! out-of-range wavelengths that fall to zero.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The exponential uses the same hand-rolled `exp_approx` the reference uses,
//! and the band centres, widths and peaks are the same constants — so `CPU` and
//! `GPU` evaluate the same closed-form algebra. Values are asserted to within
//! `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to fail a wrong port (a
//! dropped lobe, a wrong centre). The scenes also assert the value stays
//! non-negative and peaks at each band centre, so a degenerate kernel could not
//! pass.
//!
//! Provenance: standard two-band Gaussian ozone model; no Unreal Engine source
//! or derived code.

use prism_render_architecture::volumetric::spectral::ozone_absorption;
use prism_volumetric_gpu::{GpuContext, GpuOzoneAbsorption, OzoneAbsorptionQuery};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays non-negative.
fn assert_parity(queries: &[OzoneAbsorptionQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one coefficient per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = ozone_absorption(q.wavelength_nm);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "ozone absorption mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            got >= 0.0,
            "gpu ozone absorption must stay non-negative: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_ozone_absorption_matches_cpu_golden_across_wavelengths() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping ozone absorption parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuOzoneAbsorption::new(&ctx);

    // A deterministic spread: the two band centres, a point between them, and
    // out-of-range wavelengths far below and above every band.
    let mut queries: Vec<OzoneAbsorptionQuery> = vec![
        OzoneAbsorptionQuery {
            wavelength_nm: 320.0,
        },
        OzoneAbsorptionQuery {
            wavelength_nm: 602.0,
        },
        OzoneAbsorptionQuery {
            wavelength_nm: 460.0,
        },
        OzoneAbsorptionQuery {
            wavelength_nm: 100.0,
        },
        OzoneAbsorptionQuery {
            wavelength_nm: 1200.0,
        },
    ];
    // A deterministic sweep across the full visible-plus-UV range.
    for k in 0..=160 {
        queries.push(OzoneAbsorptionQuery {
            wavelength_nm: 200.0 + (k as f32) * 5.0,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Each band centre is a local peak relative to a point one half-width away.
    let huggins_center = gpu_kernel.eval(
        &ctx,
        &[OzoneAbsorptionQuery {
            wavelength_nm: 320.0,
        }],
    )[0];
    let huggins_off = gpu_kernel.eval(
        &ctx,
        &[OzoneAbsorptionQuery {
            wavelength_nm: 360.0,
        }],
    )[0];
    assert!(
        huggins_center > huggins_off,
        "the Huggins centre must exceed a point one half-width away: {huggins_center} vs {huggins_off}"
    );
    let chappuis_center = gpu_kernel.eval(
        &ctx,
        &[OzoneAbsorptionQuery {
            wavelength_nm: 602.0,
        }],
    )[0];
    let chappuis_off = gpu_kernel.eval(
        &ctx,
        &[OzoneAbsorptionQuery {
            wavelength_nm: 692.0,
        }],
    )[0];
    assert!(
        chappuis_center > chappuis_off,
        "the Chappuis centre must exceed a point one half-width away: {chappuis_center} vs {chappuis_off}"
    );
    // Far outside every band the coefficient collapses toward zero.
    assert!(
        gpu[3] < 1e-4 && gpu[4] < 1e-4,
        "far off-band the coefficient collapses toward zero: {} and {}",
        gpu[3],
        gpu[4]
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuOzoneAbsorption::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
