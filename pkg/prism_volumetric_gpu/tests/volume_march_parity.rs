//! Real-device parity for the §20 volumetric ray-march twin:
//! [`GpuVolumeMarch`](prism_volumetric_gpu::volume_march::GpuVolumeMarch) must
//! reproduce the `CPU` golden
//! [`volume_march`](prism_render_architecture::particle::volume_march) lane for
//! lane across its three query classes — the slab-method ray/`AABB` clip
//! [`ray_aabb_slab`](prism_render_architecture::particle::volume_march::ray_aabb_slab),
//! the per-ray start-offset dither
//! [`jittered_start_offset`](prism_render_architecture::particle::volume_march::jittered_start_offset),
//! and the front-to-back transmittance integral of the
//! [`march`](prism_render_architecture::particle::volume_march::march) loop body
//! over a host-presampled density array.
//!
//! The named fixtures cover an empty batch; a frontal slab penetration guarded
//! on two parallel axes; a clear parallel-slab miss; an origin inside the box; a
//! box wholly behind the origin; a negative-direction clip; dither determinism,
//! range and the zero-step degenerate; an empty density integral; a uniform
//! monotone falloff; a cutoff early-stop clear of the threshold; an
//! optical-depth accumulation; a scale/extinction equivalence; a mixed-variant
//! batch that pins input ordering; and a large pseudo-random batch compared lane
//! for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of guarded divisions, an
//! integer avalanche hash or an algebraic opacity composite, so `CPU` and `GPU`
//! evaluate the same closed form in the same associativity. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on
//! continuous values (chord endpoints, offsets, transmittance, optical depth)
//! and asserts an *exact* match on the discrete hit flags and step counts. For
//! the random batch a slab-verdict disagreement is tolerated only when the chord
//! sits inside a narrow tie band (a grazing chord `|t_exit - t_enter| <= 1e-2`
//! or a forward-visibility boundary `|t_exit| <= 1e-2`), the only place a legal
//! `ULP` perturbation can flip a `<=`/`>=` verdict; every named fixture is placed
//! clear of such boundaries so its flags and step counts assert exactly. The
//! random integration lanes disable the cutoff (threshold `0`) so no early-out
//! tie can perturb the step count.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::volume_march`；
//! standard slab-method ray/`AABB` intersection, `PCG`-style integer lattice
//! hash and algebraic (`exp`-free) front-to-back transmittance compositing; no
//! third-party engine source or derived code.

use prism_render_architecture::particle::sort_cull::Aabb;
use prism_render_architecture::particle::volume_march::Ray;
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::volume_march::{
    cpu_reference, GpuVolumeMarch, VolumeMarchQuery, VolumeMarchResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on continuous values. A `GPU` may fuse a multiply-add
/// the scalar reference leaves separate, perturbing the low mantissa bits by a
/// few units in the last place; `1e-4` admits that legal slack while still
/// failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Half-width of the tie band inside which a `<=`/`>=` slab verdict can legally
/// flip under a `ULP`-scale perturbation, so a boolean disagreement there is
/// tolerated for the random batch (never for the clear-of-boundary fixtures).
const TIE: f32 = 1.0e-2;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// The axis-aligned unit box `[-1, 1]` on every axis, the fixture most of the
/// named slab cases probe.
fn unit_box() -> Aabb {
    Aabb {
        min: Vec3::splat(-1.0),
        max: Vec3::splat(1.0),
    }
}

/// Builds one slab query from a ray origin, direction and box.
fn slab(origin: Vec3, dir: Vec3, aabb: Aabb) -> VolumeMarchQuery {
    VolumeMarchQuery::Slab {
        ray: Ray::new(origin, dir),
        aabb,
    }
}

/// Builds one integration query.
fn integrate(
    densities: Vec<f32>,
    step_size: f32,
    density_scale: f32,
    extinction: f32,
    cutoff: f32,
) -> VolumeMarchQuery {
    VolumeMarchQuery::Integrate {
        densities,
        step_size,
        density_scale,
        extinction,
        cutoff,
    }
}

/// Asserts strict lane-for-lane parity of one `GPU` result against the `CPU`
/// golden for a fixture placed clear of every tie band: slab hit flags and
/// integration step counts match exactly, and every continuous value matches
/// within tolerance.
fn assert_lane(lane: usize, got: &VolumeMarchResult, want: &VolumeMarchResult) {
    match (got, want) {
        (
            VolumeMarchResult::Slab {
                hit: hg,
                t_enter: teg,
                t_exit: txg,
            },
            VolumeMarchResult::Slab {
                hit: hc,
                t_enter: tec,
                t_exit: txc,
            },
        ) => {
            assert_eq!(hg, hc, "lane {lane}: slab hit gpu {hg} vs cpu {hc}");
            if *hc {
                assert!(
                    close(*teg, *tec),
                    "lane {lane}: t_enter gpu {teg} vs cpu {tec}"
                );
                assert!(
                    close(*txg, *txc),
                    "lane {lane}: t_exit gpu {txg} vs cpu {txc}"
                );
            }
        }
        (VolumeMarchResult::Jitter { offset: og }, VolumeMarchResult::Jitter { offset: oc }) => {
            assert!(close(*og, *oc), "lane {lane}: offset gpu {og} vs cpu {oc}");
        }
        (
            VolumeMarchResult::Integrate {
                transmittance: tg,
                optical_depth: dg,
                steps_taken: sg,
            },
            VolumeMarchResult::Integrate {
                transmittance: tc,
                optical_depth: dc,
                steps_taken: sc,
            },
        ) => {
            assert_eq!(sg, sc, "lane {lane}: steps gpu {sg} vs cpu {sc}");
            assert!(
                close(*tg, *tc),
                "lane {lane}: transmittance gpu {tg} vs cpu {tc}"
            );
            assert!(
                close(*dg, *dc),
                "lane {lane}: optical_depth gpu {dg} vs cpu {dc}"
            );
        }
        _ => panic!("lane {lane}: result variant mismatch gpu {got:?} vs cpu {want:?}"),
    }
}

/// Runs the `GPU` dispatch and asserts strict parity against the `CPU` golden
/// for every lane; returns the `GPU` verdicts for extra per-test assertions.
/// Use only for fixtures placed clear of every boundary.
fn check(
    ctx: &GpuContext,
    gpu: &GpuVolumeMarch,
    queries: &[VolumeMarchQuery],
) -> Vec<VolumeMarchResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        assert_lane(lane, g, &cpu_reference(q));
    }
    got
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

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn slab_frontal_penetration_with_parallel_guards() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // Straight down +x: the y and z direction components are exactly zero, so
    // both of those axes take the parallel-slab guard with the origin inside the
    // slab. Chord [4, 6] into the unit box.
    let q = slab(
        Vec3::new(-5.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Slab {
        hit,
        t_enter,
        t_exit,
    } = got[0]
    {
        assert!(hit, "frontal ray should hit");
        assert!(close(t_enter, 4.0), "t_enter {t_enter}");
        assert!(close(t_exit, 6.0), "t_exit {t_exit}");
    } else {
        panic!("expected a slab verdict");
    }
}

#[test]
fn slab_parallel_guard_clear_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // Same +x ray, offset to y = 9 — far outside the y slab. The y-axis parallel
    // guard rejects the whole query, so the verdict must be an exact miss.
    let q = slab(
        Vec3::new(-5.0, 9.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Slab { hit, .. } = got[0] {
        assert!(!hit, "a ray parallel to and outside the y slab must miss");
    } else {
        panic!("expected a slab verdict");
    }
}

#[test]
fn slab_origin_inside_box() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // Origin at the center marching +x: the near crossing is behind the origin
    // (t_enter = -1) and the far crossing is forward (t_exit = 1); the chord is
    // still visible because t_exit >= 0.
    let q = slab(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Slab {
        hit,
        t_enter,
        t_exit,
    } = got[0]
    {
        assert!(hit, "an interior origin should still clip a chord");
        assert!(close(t_enter, -1.0), "t_enter {t_enter}");
        assert!(close(t_exit, 1.0), "t_exit {t_exit}");
    } else {
        panic!("expected a slab verdict");
    }
}

#[test]
fn slab_box_behind_origin_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // Origin at x = 5 marching +x leaves the whole box behind (t_exit = -4 < 0),
    // so the forward march reports a miss.
    let q = slab(
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Slab { hit, .. } = got[0] {
        assert!(!hit, "a box wholly behind the origin must miss");
    } else {
        panic!("expected a slab verdict");
    }
}

#[test]
fn slab_negative_direction_diagonal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // A diagonal ray approaching the box from the +x/+y/+z corner, every
    // direction component non-parallel and well clear of EPS. The parity helper
    // pins the chord against the golden.
    let q = slab(
        Vec3::new(6.0, 5.0, 4.0),
        Vec3::new(-1.0, -1.0, -1.0),
        unit_box(),
    );
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Slab { hit, .. } = got[0] {
        assert!(hit, "the diagonal ray should enter the box");
    } else {
        panic!("expected a slab verdict");
    }
}

#[test]
fn jitter_determinism_and_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    let step = 0.5_f32;
    let mut queries = Vec::new();
    for ray_index in 0..32u32 {
        queries.push(VolumeMarchQuery::Jitter {
            ray_index,
            seed: 0x1234_5678,
            step_size: step,
        });
    }
    // Append a duplicate of lane 0 to confirm the hash is a pure function of its
    // inputs (same inputs, same offset).
    queries.push(VolumeMarchQuery::Jitter {
        ray_index: 0,
        seed: 0x1234_5678,
        step_size: step,
    });
    let got = check(&ctx, &gpu, &queries);
    for (lane, g) in got.iter().enumerate() {
        if let VolumeMarchResult::Jitter { offset } = g {
            assert!(*offset >= -EPS, "lane {lane}: offset {offset} below zero");
            assert!(
                *offset <= step + EPS,
                "lane {lane}: offset {offset} outside the step span"
            );
        } else {
            panic!("expected a jitter verdict");
        }
    }
    // Equal inputs must jitter equally: compare the first lane against its
    // appended duplicate through the tolerance helper rather than a bare f32
    // equality.
    let (VolumeMarchResult::Jitter { offset: first }, VolumeMarchResult::Jitter { offset: dup }) =
        (got[0], got[got.len() - 1])
    else {
        panic!("expected jitter verdicts for the duplicated lane");
    };
    assert!(
        close(first, dup),
        "equal inputs jittered differently: {first} vs {dup}"
    );
}

#[test]
fn jitter_zero_step_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    let q = VolumeMarchQuery::Jitter {
        ray_index: 7,
        seed: 1,
        step_size: 0.0,
    };
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Jitter { offset } = got[0] {
        assert!(
            close(offset, 0.0),
            "a zero step must yield a zero offset: {offset}"
        );
    } else {
        panic!("expected a jitter verdict");
    }
}

#[test]
fn integrate_empty_density_is_transparent() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    let q = integrate(Vec::new(), 0.1, 1.0, 1.0, 0.0);
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Integrate {
        transmittance,
        optical_depth,
        steps_taken,
    } = got[0]
    {
        assert!(
            close(transmittance, 1.0),
            "empty medium stays transparent: {transmittance}"
        );
        assert!(
            close(optical_depth, 0.0),
            "empty medium has no depth: {optical_depth}"
        );
        assert_eq!(steps_taken, 0, "empty medium takes no steps");
    } else {
        panic!("expected an integrate verdict");
    }
}

#[test]
fn integrate_uniform_monotone_falloff() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // Uniform density with a per-step opacity of 0.05 and no cutoff: the product
    // telescopes to 0.95^8, a clean monotone decay the kernel must mirror.
    let q = integrate(vec![0.5; 8], 0.1, 1.0, 1.0, 0.0);
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Integrate {
        transmittance,
        steps_taken,
        ..
    } = got[0]
    {
        assert_eq!(steps_taken, 8, "every sample should composite");
        assert!(
            transmittance < 1.0,
            "a dense medium must dim: {transmittance}"
        );
        assert!(
            transmittance > 0.0,
            "a thin medium should not fully occlude: {transmittance}"
        );
    } else {
        panic!("expected an integrate verdict");
    }
}

#[test]
fn integrate_cutoff_early_stop() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // Per-step opacity 0.5 (density 0.5 · step 1.0) drives transmittance
    // 1 -> 0.5 -> 0.25; the cutoff 0.3 sits clear between those values, so the
    // march must stop after exactly two steps even though eight samples exist.
    let q = integrate(vec![0.5; 8], 1.0, 1.0, 1.0, 0.3);
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Integrate {
        transmittance,
        steps_taken,
        ..
    } = got[0]
    {
        assert_eq!(
            steps_taken, 2,
            "the cutoff should fire after the second step"
        );
        assert!(
            close(transmittance, 0.25),
            "surviving transmittance {transmittance}"
        );
    } else {
        panic!("expected an integrate verdict");
    }
}

#[test]
fn integrate_optical_depth_accumulates() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // Distinct densities with a unit step and no cutoff: optical depth is the
    // plain sum 0.1 + 0.2 + 0.3 and transmittance the triple product.
    let q = integrate(vec![0.1, 0.2, 0.3], 1.0, 1.0, 1.0, 0.0);
    let got = check(&ctx, &gpu, &[q]);
    if let VolumeMarchResult::Integrate {
        transmittance,
        optical_depth,
        steps_taken,
    } = got[0]
    {
        assert_eq!(steps_taken, 3, "all three samples composite");
        assert!(close(optical_depth, 0.6), "optical depth {optical_depth}");
        assert!(close(transmittance, 0.504), "transmittance {transmittance}");
    } else {
        panic!("expected an integrate verdict");
    }
}

#[test]
fn integrate_scale_extinction_equivalence() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // sigma = density · scale · extinction, so (0.25, scale 2, ext 1) and
    // (0.5, scale 1, ext 1) feed the same opacity; the two lanes must agree with
    // their golden and each other.
    let a = integrate(vec![0.25; 4], 1.0, 2.0, 1.0, 0.0);
    let b = integrate(vec![0.5; 4], 1.0, 1.0, 1.0, 0.0);
    let got = check(&ctx, &gpu, &[a, b]);
    assert_lane(0, &got[0], &got[1]);
}

#[test]
fn mixed_variant_batch_preserves_order() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);
    // Interleave the three variants so the scatter-back from each sub-kernel into
    // input order is exercised, not just a single homogeneous batch.
    let queries = vec![
        slab(
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            unit_box(),
        ),
        VolumeMarchQuery::Jitter {
            ray_index: 11,
            seed: 99,
            step_size: 0.25,
        },
        integrate(vec![0.4, 0.6], 0.5, 1.0, 1.0, 0.0),
        slab(
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            unit_box(),
        ),
        VolumeMarchQuery::Jitter {
            ray_index: 12,
            seed: 99,
            step_size: 0.25,
        },
        integrate(vec![0.2; 5], 0.3, 1.5, 1.0, 0.0),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVolumeMarch::new(&ctx);

    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut queries: Vec<VolumeMarchQuery> = Vec::new();

    // A spread of slab queries aimed near a box, mixing clear hits and misses,
    // occasionally forcing one direction axis parallel to exercise the guard.
    let box_a = Aabb {
        min: Vec3::new(-1.5, -1.0, -2.0),
        max: Vec3::new(1.5, 1.0, 2.0),
    };
    for _ in 0..96 {
        let origin = Vec3::new(
            lcg(&mut state) * 16.0 - 8.0,
            lcg(&mut state) * 16.0 - 8.0,
            lcg(&mut state) * 16.0 - 8.0,
        );
        let target = Vec3::new(
            lcg(&mut state) * 6.0 - 3.0,
            lcg(&mut state) * 6.0 - 3.0,
            lcg(&mut state) * 6.0 - 3.0,
        );
        let mut dir = Vec3::new(
            target.x - origin.x,
            target.y - origin.y,
            target.z - origin.z,
        );
        match (lcg(&mut state) * 4.0) as u32 {
            0 => dir.x = 0.0,
            1 => dir.y = 0.0,
            2 => dir.z = 0.0,
            _ => {}
        }
        let dd = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
        if dd < 0.25 {
            dir.x += 1.0;
        }
        queries.push(slab(origin, dir, box_a));
    }

    // A spread of dither queries over varied indices, seeds and steps.
    for i in 0..64u32 {
        let seed = (lcg(&mut state) * 4.0e9) as u32;
        let step = lcg(&mut state) * 2.0 + 0.1;
        queries.push(VolumeMarchQuery::Jitter {
            ray_index: i,
            seed,
            step_size: step,
        });
    }

    // A spread of integration queries. The cutoff is disabled (threshold 0) so no
    // early-out tie can perturb the step count; densities and scales stay
    // moderate so the composite is well-conditioned.
    for _ in 0..64 {
        let count = (lcg(&mut state) * 16.0) as usize;
        let mut densities = Vec::with_capacity(count);
        for _ in 0..count {
            densities.push(lcg(&mut state) * 1.5);
        }
        let step = lcg(&mut state) * 0.45 + 0.05;
        let scale = lcg(&mut state) * 1.5 + 0.5;
        let extinction = lcg(&mut state) * 1.5 + 0.5;
        queries.push(integrate(densities, step, scale, extinction, 0.0));
    }

    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");

    let mut saw_hit = false;
    let mut saw_miss = false;
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let want = cpu_reference(q);
        match (g, &want) {
            (
                VolumeMarchResult::Slab {
                    hit: hg,
                    t_enter: teg,
                    t_exit: txg,
                },
                VolumeMarchResult::Slab {
                    hit: hc,
                    t_enter: tec,
                    t_exit: txc,
                },
            ) => {
                saw_hit |= *hc;
                saw_miss |= !*hc;
                if hg != hc {
                    // A verdict flip is legal only inside the tie band, where a
                    // grazing chord or a forward-visibility boundary lets a ULP
                    // perturbation cross a `<=`/`>=` comparison.
                    let gap = if *hg {
                        (txg - teg).abs().min(txg.abs())
                    } else {
                        (txc - tec).abs().min(txc.abs())
                    };
                    assert!(
                        gap <= TIE,
                        "lane {lane}: slab verdict flip off the tie band: gpu {hg} cpu {hc} gap {gap}"
                    );
                } else if *hc {
                    assert!(
                        close(*teg, *tec),
                        "lane {lane}: t_enter gpu {teg} vs cpu {tec}"
                    );
                    assert!(
                        close(*txg, *txc),
                        "lane {lane}: t_exit gpu {txg} vs cpu {txc}"
                    );
                }
            }
            _ => assert_lane(lane, g, &want),
        }
    }

    assert!(
        saw_hit && saw_miss,
        "the random slab spread should produce both hits and misses"
    );
}
