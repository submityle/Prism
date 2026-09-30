//! Real-device parity for the divergence-free curl-noise twin: [`GpuCurl`] must
//! reproduce the `CPU` golden
//! [`curl_noise_3d`](prism_render_architecture::volumetric::noise::curl_noise_3d)
//! across a spread of sample points and seeds.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each output component is the analytic `curl` of a `Perlin` vector potential
//! taken with matched central differences. The lattice `hash` selecting every
//! gradient is pure unsigned-integer work and `WGSL` unsigned integers wrap on
//! overflow exactly like Rust's `wrapping_mul` / `^` / `>>`, so the `GPU`
//! selects bit-identical gradients to the reference. Only the float
//! central-difference algebra can differ, and only by a legal multiply-add
//! contraction of a few `ULP`. Because the curl divides that difference by
//! `2 * CURL_EPS = 0.04`, the potential's `ULP`-level spread is amplified `~25x`
//! into a roughly constant `~2.5e-4` absolute spread on the output; the three
//! components are asserted to within `abs_diff < 5e-4` or `rel_diff < 2e-4`
//! — loose enough to admit that amplified fma contraction, tight enough to
//! fail a wrong port (a swapped offset, a dropped difference term, a mis-hashed
//! coordinate all shift a component by `O(1)`). The
//! scenes also assert the field is non-constant and re-derive the numerical
//! divergence from the `GPU` output stencil, so a degenerate or divergence-
//! breaking kernel could not pass.
//!
//! Provenance: standard curl-of-`Perlin`-potential noise (`Bridson` et al.
//! 2007); no Unreal Engine source or derived code.

use prism_render_architecture::volumetric::noise::curl_noise_3d;
use prism_render_architecture::volumetric::Vec3;
use prism_volumetric_gpu::{CurlQuery, GpuContext, GpuCurl};

/// The fixed central-difference step the reference `curl_noise_3d` uses; the
/// divergence check re-derives div(curl) with the same step so the mixed
/// second differences cancel analytically and only rounding remains.
const CURL_EPS: f32 = 0.02;

/// Absolute parity tolerance for a curl component. The kernel forms each
/// component as a central difference of the `Perlin` vector potential divided
/// by `2 * CURL_EPS = 0.04`, an amplification of `~25x`. The `CPU` and `GPU` sum
/// the potential's dozen-odd multiply-adds in a different fused contraction
/// order, so the potential differs by a few `f32` `ULP` (`~1e-5` absolute at
/// unit magnitude); the `/0.04` amplification lifts that to a roughly constant
/// `~2.5e-4` absolute spread on the curl, largely independent of the output
/// magnitude (measured worst case `2.6e-4` over the full grid on a real
/// device). The tolerance keeps `~2x` margin and stays far below any real port
/// error (a swapped offset or a dropped difference term shifts a component by
/// `O(1)`).
const CURL_ABS_TOL: f32 = 5e-4;

/// Relative fallback for a rare large-magnitude component. The absolute floor
/// already covers the near-zero components a pure relative test would reject, so
/// this only tightens the check where a component is several units large.
const CURL_REL_TOL: f32 = 2e-4;

/// Asserts every `gpu` vector matches the `CPU` golden component-wise to within
/// the documented tolerance.
fn assert_parity(queries: &[CurlQuery], gpu: &[Vec3]) {
    assert_eq!(gpu.len(), queries.len(), "one vector per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = curl_noise_3d(q.point, q.seed);
        let got = gpu[i];
        for (axis, (g, e)) in [(got.x, exp.x), (got.y, exp.y), (got.z, exp.z)]
            .into_iter()
            .enumerate()
        {
            let abs_diff = (g - e).abs();
            let rel_diff = abs_diff / e.abs().max(1e-6);
            assert!(
                abs_diff < CURL_ABS_TOL || rel_diff < CURL_REL_TOL,
                "curl mismatch for query {q:?} axis {axis}: gpu {g}, cpu {e} \
                 (abs {abs_diff}, rel {rel_diff})"
            );
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_curl_matches_cpu_golden_across_points_and_seeds() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping curl parity: no wgpu adapter on this host");
        return;
    };
    let gpu_curl = GpuCurl::new(&ctx);

    // A deterministic lattice sweep: fractional offsets exercise the fade/lerp
    // blend inside each potential, integer-crossing coordinates exercise cell
    // boundaries, negatives exercise the floor/`i32` cast, and several seeds
    // exercise the hash mix (including the two potential-decorrelating xors).
    let mut queries: Vec<CurlQuery> = Vec::new();
    let seeds = [0u32, 1, 7, 1_337, 0x9e37_79b9];
    for &seed in &seeds {
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in -2..=2 {
                    let x = xi as f32 * 0.37 + 0.11;
                    let y = yi as f32 * 0.53 - 0.29;
                    let z = zi as f32 * 0.61 + 0.07;
                    queries.push(CurlQuery {
                        point: Vec3::new(x, y, z),
                        seed,
                    });
                }
            }
        }
    }

    let gpu = gpu_curl.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // The field is not constant: at least two sampled vectors must differ well
    // beyond the parity tolerance, proving a real curl blend ran.
    let first = gpu[0];
    assert!(
        gpu.iter()
            .any(|v| (v.x - first.x).abs() > 1e-3 || (v.y - first.y).abs() > 1e-3),
        "curl field must vary across space, not return a constant"
    );

    // Every sample must be finite and the field must flow somewhere.
    let mut any_flow = false;
    for v in &gpu {
        assert!(
            v.x.is_finite() && v.y.is_finite() && v.z.is_finite(),
            "curl output must be finite"
        );
        if v.length_squared() > 1e-6 {
            any_flow = true;
        }
    }
    assert!(any_flow, "curl field was degenerate everywhere");
}

#[test]
fn gpu_curl_field_is_approximately_divergence_free() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_curl = GpuCurl::new(&ctx);

    // Re-derive div(curl) from the GPU output using the same central-difference
    // step the reference uses. The curl of any field is analytically
    // divergence-free; with matched differences only rounding survives, so the
    // GPU stencil divergence must stay near zero — mirroring the CPU golden's
    // own divergence-free property test.
    let seed = 33u32;
    let e = CURL_EPS;
    let inv = 1.0 / (2.0 * e);
    let bases = [
        Vec3::new(0.37, 1.21, -2.13),
        Vec3::new(-4.51, 3.09, 0.77),
        Vec3::new(12.34, -5.68, 9.01),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-100.25, 42.75, 7.33),
        Vec3::new(3.123, -2.654, 1.789),
    ];

    // Build the six-point stencil for every base point in one dispatch.
    let mut queries: Vec<CurlQuery> = Vec::new();
    for &p in &bases {
        queries.push(CurlQuery {
            point: p.add(Vec3::new(e, 0.0, 0.0)),
            seed,
        });
        queries.push(CurlQuery {
            point: p.sub(Vec3::new(e, 0.0, 0.0)),
            seed,
        });
        queries.push(CurlQuery {
            point: p.add(Vec3::new(0.0, e, 0.0)),
            seed,
        });
        queries.push(CurlQuery {
            point: p.sub(Vec3::new(0.0, e, 0.0)),
            seed,
        });
        queries.push(CurlQuery {
            point: p.add(Vec3::new(0.0, 0.0, e)),
            seed,
        });
        queries.push(CurlQuery {
            point: p.sub(Vec3::new(0.0, 0.0, e)),
            seed,
        });
    }

    let gpu = gpu_curl.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for (b, stencil) in gpu.chunks_exact(6).enumerate() {
        let cx1 = stencil[0];
        let cx0 = stencil[1];
        let cy1 = stencil[2];
        let cy0 = stencil[3];
        let cz1 = stencil[4];
        let cz0 = stencil[5];
        let divergence = ((cx1.x - cx0.x) + (cy1.y - cy0.y) + (cz1.z - cz0.z)) * inv;
        assert!(
            divergence.abs() < 1e-2,
            "gpu curl divergence too large at base {b}: {divergence}"
        );
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_curl = GpuCurl::new(&ctx);
    let out = gpu_curl.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no vectors");
}
