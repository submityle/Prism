//! Real-device parity for the `GI`-probe twin:
//! [`GpuGiProbe`](prism_volumetric_gpu::gi_probe::GpuGiProbe) must reproduce the
//! `CPU` golden
//! [`gi_probe`](prism_render_architecture::particle::gi_probe) across the real
//! spherical-harmonic basis (bands 0-1 and 0-2), the Lambert cosine-lobe
//! convolution weights, the octahedral direction map and its texture-space
//! remaps, the six-face ambient-cube irradiance blend, and the colored
//! [`ShColorL1`](prism_render_architecture::particle::gi_probe::ShColorL1) /
//! [`ShColorL2`](prism_render_architecture::particle::gi_probe::ShColorL2)
//! radiance and irradiance reconstruction, plus a randomized batch compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every routine threads through multiplies, adds and one guarded divide /
//! `sqrt`, so `CPU` and `GPU` are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on every continuous channel.
//! The cosine-lobe `band` is an integer selector, so its weight is a fixed
//! constant per band and still compared within the same tolerance.
//!
//! # Conditioning
//!
//! Direction fixtures are drawn by rejection sampling: a sample is accepted only
//! when its normalized form stays clear of the octahedral seams — away from the
//! `+-z` poles and the quadrant boundaries where the sign fold switches — so a
//! fused multiply-add cannot tip a near-seam direction to the other branch on
//! one device. Octahedral `decode` fixtures are produced by encoding such safe
//! directions, so every `(u, v)` lies off the fold. All randomness comes from a
//! host-side integer generator, so no transcendental appears in a fixture.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gi_probe`；no
//! third-party engine source or derived code.

use prism_render_architecture::particle::gi_probe::{
    cosine_lobe_weight, octa_decode, octa_encode, octa_from_unorm, octa_to_unorm, sh_basis_l1,
    sh_basis_l2, AmbientCube, ShColorL1, ShColorL2,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::gi_probe::{GiProbeQuery, GiProbeResult, GpuGiProbe};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// Builds a [`Vec3`] from a component triple.
fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

/// A normalized direction drawn clear of the octahedral seams by rejection:
/// the raw cube sample must have a usable length, and the unit result must keep
/// every component away from the quadrant edges and the `+-z` poles.
fn rand_dir(state: &mut u64) -> [f32; 3] {
    loop {
        let x = signed(state, 1.0);
        let y = signed(state, 1.0);
        let z = signed(state, 1.0);
        let len_sq = x * x + y * y + z * z;
        if !(0.1..=1.0).contains(&len_sq) {
            continue;
        }
        let n = Vec3::new(x, y, z).normalize_or_zero();
        if n.x.abs() < 0.15 || n.y.abs() < 0.15 || n.z.abs() < 0.15 || n.z.abs() > 0.9 {
            continue;
        }
        return [n.x, n.y, n.z];
    }
}

/// A pseudo-random `RGB` triple with each channel in `[-span, span)`.
fn rand_rgb(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// Six pseudo-random `RGB` ambient-cube faces.
fn rand_faces(state: &mut u64) -> [[f32; 3]; 6] {
    let mut faces = [[0.0_f32; 3]; 6];
    for face in &mut faces {
        *face = rand_rgb(state, 1.5);
    }
    faces
}

/// Four pseudo-random `RGB` `L1` coefficients.
fn rand_coeffs4(state: &mut u64) -> [[f32; 3]; 4] {
    let mut c = [[0.0_f32; 3]; 4];
    for slot in &mut c {
        *slot = rand_rgb(state, 2.0);
    }
    c
}

/// Nine pseudo-random `RGB` `L2` coefficients.
fn rand_coeffs9(state: &mut u64) -> [[f32; 3]; 9] {
    let mut c = [[0.0_f32; 3]; 9];
    for slot in &mut c {
        *slot = rand_rgb(state, 2.0);
    }
    c
}

/// A pseudo-random boolean (used for the irradiance toggle).
fn rand_bool(state: &mut u64) -> bool {
    lcg(state) < 0.5
}

/// Builds an [`AmbientCube`] from the six packed faces in the twin's order.
fn ambient_cube(faces: &[[f32; 3]; 6]) -> AmbientCube {
    AmbientCube {
        pos_x: v3(faces[0]),
        neg_x: v3(faces[1]),
        pos_y: v3(faces[2]),
        neg_y: v3(faces[3]),
        pos_z: v3(faces[4]),
        neg_z: v3(faces[5]),
    }
}

/// Builds an [`ShColorL1`] from four packed `RGB` coefficients.
fn sh_color_l1(coeffs: &[[f32; 3]; 4]) -> ShColorL1 {
    ShColorL1 {
        coeffs: [v3(coeffs[0]), v3(coeffs[1]), v3(coeffs[2]), v3(coeffs[3])],
    }
}

/// Builds an [`ShColorL2`] from nine packed `RGB` coefficients.
fn sh_color_l2(coeffs: &[[f32; 3]; 9]) -> ShColorL2 {
    let mut out = [Vec3::ZERO; 9];
    for (o, c) in out.iter_mut().zip(coeffs.iter()) {
        *o = v3(*c);
    }
    ShColorL2 { coeffs: out }
}

/// Pins a slice of lanes against the reference within tolerance.
fn close_slice(idx: usize, got: &[f32], want: &[f32]) {
    for (lane, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(close(*g, *w), "query {idx} lane {lane}: gpu {g} vs cpu {w}");
    }
}

/// Pins a scalar pair against the reference within tolerance.
fn close_pair(idx: usize, got: (f32, f32), want: (f32, f32)) {
    assert!(
        close(got.0, want.0),
        "query {idx} u: gpu {} vs cpu {}",
        got.0,
        want.0
    );
    assert!(
        close(got.1, want.1),
        "query {idx} v: gpu {} vs cpu {}",
        got.1,
        want.1
    );
}

/// Pins a vector output against the reference within tolerance.
fn close_vec(idx: usize, got: &[f32; 3], want: Vec3) {
    close_slice(idx, got, &[want.x, want.y, want.z]);
}

/// Pins one `GPU` result against the `CPU` golden for `query`, matching the
/// result variant to the query variant and comparing channel-for-channel.
fn pin(idx: usize, query: &GiProbeQuery, got: &GiProbeResult) {
    match (query, got) {
        (GiProbeQuery::ShBasisL1 { dir }, GiProbeResult::ShBasisL1(got_b)) => {
            close_slice(idx, got_b, &sh_basis_l1(v3(*dir)));
        }
        (GiProbeQuery::ShBasisL2 { dir }, GiProbeResult::ShBasisL2(got_b)) => {
            close_slice(idx, got_b, &sh_basis_l2(v3(*dir)));
        }
        (GiProbeQuery::CosineLobe { band }, GiProbeResult::CosineLobe(got_w)) => {
            let want = cosine_lobe_weight(*band as u8);
            assert!(
                close(*got_w, want),
                "query {idx} cosine lobe: gpu {got_w} vs cpu {want}"
            );
        }
        (GiProbeQuery::OctaEncode { dir }, GiProbeResult::OctaEncode { u, v }) => {
            close_pair(idx, (*u, *v), octa_encode(v3(*dir)));
        }
        (GiProbeQuery::OctaDecode { u, v }, GiProbeResult::OctaDecode(got_d)) => {
            close_vec(idx, got_d, octa_decode(*u, *v));
        }
        (GiProbeQuery::OctaToUnorm { u, v }, GiProbeResult::OctaToUnorm { u: ou, v: ov }) => {
            close_pair(idx, (*ou, *ov), octa_to_unorm(*u, *v));
        }
        (GiProbeQuery::OctaFromUnorm { u, v }, GiProbeResult::OctaFromUnorm { u: ou, v: ov }) => {
            close_pair(idx, (*ou, *ov), octa_from_unorm(*u, *v));
        }
        (GiProbeQuery::AmbientCube { faces, normal }, GiProbeResult::AmbientCube(got_s)) => {
            let want = ambient_cube(faces).sample(v3(*normal));
            close_vec(idx, got_s, want);
        }
        (
            GiProbeQuery::ShReconstructL1 {
                coeffs,
                dir,
                irradiance,
            },
            GiProbeResult::ShReconstructL1(got_s),
        ) => {
            let sh = sh_color_l1(coeffs);
            let want = if *irradiance {
                sh.evaluate_irradiance(v3(*dir))
            } else {
                sh.evaluate_radiance(v3(*dir))
            };
            close_vec(idx, got_s, want);
        }
        (
            GiProbeQuery::ShReconstructL2 {
                coeffs,
                dir,
                irradiance,
            },
            GiProbeResult::ShReconstructL2(got_s),
        ) => {
            let sh = sh_color_l2(coeffs);
            let want = if *irradiance {
                sh.evaluate_irradiance(v3(*dir))
            } else {
                sh.evaluate_radiance(v3(*dir))
            };
            close_vec(idx, got_s, want);
        }
        _ => panic!("query {idx}: result variant does not match the query variant"),
    }
}

/// Draws a random query of a random routine with seam-safe fixtures.
fn rand_query(state: &mut u64) -> GiProbeQuery {
    match (lcg(state) * 10.0) as u32 {
        0 => GiProbeQuery::ShBasisL1 {
            dir: rand_dir(state),
        },
        1 => GiProbeQuery::ShBasisL2 {
            dir: rand_dir(state),
        },
        2 => GiProbeQuery::CosineLobe {
            band: (lcg(state) * 5.0) as u32,
        },
        3 => GiProbeQuery::OctaEncode {
            dir: rand_dir(state),
        },
        4 => {
            let (u, v) = octa_encode(v3(rand_dir(state)));
            GiProbeQuery::OctaDecode { u, v }
        }
        5 => {
            let (u, v) = octa_encode(v3(rand_dir(state)));
            GiProbeQuery::OctaToUnorm { u, v }
        }
        6 => GiProbeQuery::OctaFromUnorm {
            u: lcg(state),
            v: lcg(state),
        },
        7 => GiProbeQuery::AmbientCube {
            faces: rand_faces(state),
            normal: rand_dir(state),
        },
        8 => GiProbeQuery::ShReconstructL1 {
            coeffs: rand_coeffs4(state),
            dir: rand_dir(state),
            irradiance: rand_bool(state),
        },
        _ => GiProbeQuery::ShReconstructL2 {
            coeffs: rand_coeffs9(state),
            dir: rand_dir(state),
            irradiance: rand_bool(state),
        },
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuGiProbe, queries: &[GiProbeQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGiProbe::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn sh_basis_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGiProbe::new(&ctx);
    // Many seam-safe directions pin both the four-value L1 basis and the
    // nine-value L2 basis lane-for-lane.
    let mut state = 0x5157_1a2b_3c4d_5e6f_u64;
    let mut queries = Vec::new();
    for _ in 0..48 {
        let dir = rand_dir(&mut state);
        queries.push(GiProbeQuery::ShBasisL1 { dir });
        queries.push(GiProbeQuery::ShBasisL2 { dir });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn cosine_lobe_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGiProbe::new(&ctx);
    // Bands 0-2 return the three Lambert weights; bands 3 and 4 convolve to
    // zero, so every selector is exercised.
    let queries: Vec<GiProbeQuery> = (0u32..5)
        .map(|band| GiProbeQuery::CosineLobe { band })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn octahedral_map_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGiProbe::new(&ctx);
    // Encode seam-safe directions, decode the coordinates those produce, and
    // remap both ways; every octahedral routine is pinned.
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut queries = Vec::new();
    for _ in 0..48 {
        let dir = rand_dir(&mut state);
        let (u, v) = octa_encode(v3(dir));
        queries.push(GiProbeQuery::OctaEncode { dir });
        queries.push(GiProbeQuery::OctaDecode { u, v });
        queries.push(GiProbeQuery::OctaToUnorm { u, v });
        queries.push(GiProbeQuery::OctaFromUnorm {
            u: lcg(&mut state),
            v: lcg(&mut state),
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn ambient_cube_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGiProbe::new(&ctx);
    // Random six-face cubes sampled along seam-safe normals; the squared-weight
    // three-face blend is pinned channel-for-channel.
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    let queries: Vec<GiProbeQuery> = (0..64)
        .map(|_| GiProbeQuery::AmbientCube {
            faces: rand_faces(&mut state),
            normal: rand_dir(&mut state),
        })
        .collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn sh_reconstruct_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGiProbe::new(&ctx);
    // L1 and L2 reconstruction, both radiance (unweighted) and irradiance (with
    // the cosine-lobe weights), over random coefficients and seam-safe
    // directions.
    let mut state = 0x00c0_ffee_dead_beef_u64;
    let mut queries = Vec::new();
    for _ in 0..32 {
        let dir = rand_dir(&mut state);
        for &irradiance in &[false, true] {
            queries.push(GiProbeQuery::ShReconstructL1 {
                coeffs: rand_coeffs4(&mut state),
                dir,
                irradiance,
            });
            queries.push(GiProbeQuery::ShReconstructL2 {
                coeffs: rand_coeffs9(&mut state),
                dir,
                irradiance,
            });
        }
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGiProbe::new(&ctx);
    let mut state = 0x3141_5926_5358_9793_u64;
    // One batch mixing deterministic fixtures with many random queries of every
    // routine, dispatched together so the per-thread indexing and the contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let safe = rand_dir(&mut state);
    let (eu, ev) = octa_encode(v3(safe));
    let mut queries = vec![
        GiProbeQuery::ShBasisL1 { dir: safe },
        GiProbeQuery::ShBasisL2 { dir: safe },
        GiProbeQuery::CosineLobe { band: 1 },
        GiProbeQuery::OctaEncode { dir: safe },
        GiProbeQuery::OctaDecode { u: eu, v: ev },
        GiProbeQuery::OctaToUnorm { u: eu, v: ev },
        GiProbeQuery::OctaFromUnorm { u: 0.25, v: 0.75 },
    ];
    for _ in 0..57 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGiProbe::new(&ctx);
    let mut state = 0x2718_2818_2845_9045_u64;
    // A larger sweep (several workgroups' worth) pins every routine across many
    // random fixtures.
    let queries: Vec<GiProbeQuery> = (0..256).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
