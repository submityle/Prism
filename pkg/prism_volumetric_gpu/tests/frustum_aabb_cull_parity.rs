//! Real-device parity for the six-plane frustum-cull twin:
//! [`GpuFrustumAabbCull`](prism_volumetric_gpu::frustum_aabb_cull::GpuFrustumAabbCull)
//! must reproduce the `CPU` golden
//! [`cull_aabb`](prism_render_architecture::particle::frustum_aabb_cull::cull_aabb)
//! and
//! [`cull_sphere`](prism_render_architecture::particle::frustum_aabb_cull::cull_sphere)
//! verdict-for-verdict across an empty batch, hand-placed boxes and spheres
//! that are clearly inside / outside / intersecting a symmetric perspective
//! frustum and a tilted oblique frustum, and a large batch of random primitives
//! kept clear of every plane's decision band so no tie can flip.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The verdict is a discrete three-way enum, so parity is an *exact* match:
//! each `GPU` verdict must equal the reference verdict with `==`, and the
//! verdict's [`Visibility::is_visible`] must equal the reference boolean
//! shortcuts
//! [`is_visible_aabb`](prism_render_architecture::particle::frustum_aabb_cull::is_visible_aabb)
//! and
//! [`is_visible_sphere`](prism_render_architecture::particle::frustum_aabb_cull::is_visible_sphere).
//! The arithmetic is byte-for-byte the same associativity as the reference and
//! every sign test uses the same tolerant `1e-6` band, so a legal fused
//! multiply-add perturbing the low mantissa bits cannot flip a verdict unless a
//! primitive sits within one `ULP` of a plane. The random fixtures therefore
//! keep every primitive at least `MARGIN` away from each plane's `s + r = 0`
//! and `s - r = 0` boundary, so no near-tie can turn the exact match flaky.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::frustum_aabb_cull`;
//! classic p-vertex / n-vertex symmetric projected-radius frustum test; no
//! third-party engine source or derived code.

use prism_render_architecture::particle::frustum_aabb_cull::{
    cull_aabb, cull_sphere, is_visible_aabb, is_visible_sphere, Aabb, Plane, Sphere, Visibility,
};
use prism_volumetric_gpu::frustum_aabb_cull::{
    FrustumAabbCullPrimitive, FrustumAabbCullQuery, GpuFrustumAabbCull,
};
use prism_volumetric_gpu::GpuContext;

/// `1 / sqrt(2)` as a literal so the diagonal frustum planes stay unit length
/// without a runtime `sqrt`, mirroring the reference test fixtures.
const INV_SQRT2: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// A standard symmetric perspective frustum: apex at the origin looking down
/// `+z` with a 45° half-angle, near plane at `z = 1`, far plane at `z = 100`.
/// All normals point inward and are unit length. Matches the reference fixture.
const SYMMETRIC: [Plane; 6] = [
    Plane::new([INV_SQRT2, 0.0, INV_SQRT2], 0.0),
    Plane::new([-INV_SQRT2, 0.0, INV_SQRT2], 0.0),
    Plane::new([0.0, INV_SQRT2, INV_SQRT2], 0.0),
    Plane::new([0.0, -INV_SQRT2, INV_SQRT2], 0.0),
    Plane::new([0.0, 0.0, 1.0], -1.0),
    Plane::new([0.0, 0.0, -1.0], 100.0),
];

/// An oblique frustum whose side planes are tilted off the world axes, so the
/// projected-radius math is exercised away from any axis-aligned shortcut.
/// Matches the reference fixture.
const OBLIQUE: [Plane; 6] = [
    Plane::new([INV_SQRT2, 0.0, INV_SQRT2], 2.0),
    Plane::new([-INV_SQRT2, 0.0, INV_SQRT2], 2.0),
    Plane::new([0.0, INV_SQRT2, INV_SQRT2], 2.0),
    Plane::new([0.0, -INV_SQRT2, INV_SQRT2], 2.0),
    Plane::new([0.0, 0.0, 1.0], -1.0),
    Plane::new([0.0, 0.0, -1.0], 100.0),
];

/// Clearance each random primitive must keep from every plane's `s + r = 0` and
/// `s - r = 0` boundary. Chosen far above the kernel's `1e-6` sign band and any
/// realistic fused-multiply-add perturbation so no near-tie can flip the exact
/// verdict match.
const MARGIN: f32 = 1.0e-2;

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Draws a value in `[lo, hi)` from `state`.
fn range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * lcg(state)
}

/// Whether the box, against every plane, stays at least `MARGIN` clear of both
/// the p-vertex boundary (`s + r = 0`) and the n-vertex boundary (`s - r = 0`),
/// so neither a reference nor a kernel sign decision can land in a tie.
fn aabb_clear_of_bands(planes: &[Plane; 6], aabb: &Aabb) -> bool {
    for plane in planes {
        let s = plane.signed_distance(aabb.center);
        let r = plane.normal[0].abs() * aabb.half[0]
            + plane.normal[1].abs() * aabb.half[1]
            + plane.normal[2].abs() * aabb.half[2];
        if (s + r).abs() < MARGIN || (s - r).abs() < MARGIN {
            return false;
        }
    }
    true
}

/// Whether the sphere, against every plane, stays at least `MARGIN` clear of
/// both the `s + radius = 0` and `s - radius = 0` boundaries.
fn sphere_clear_of_bands(planes: &[Plane; 6], sphere: &Sphere) -> bool {
    for plane in planes {
        let s = plane.signed_distance(sphere.center);
        if (s + sphere.radius).abs() < MARGIN || (s - sphere.radius).abs() < MARGIN {
            return false;
        }
    }
    true
}

/// Runs the `GPU` cull over `queries` and asserts, verdict-for-verdict, that it
/// matches the `CPU` golden `cull_aabb` / `cull_sphere` exactly and that each
/// verdict's visibility agrees with the reference boolean shortcut.
fn check(ctx: &GpuContext, gpu: &GpuFrustumAabbCull, queries: &[FrustumAabbCullQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one verdict per query");

    for (idx, (query, &verdict)) in queries.iter().zip(got.iter()).enumerate() {
        let (want, want_visible) = match query.primitive {
            FrustumAabbCullPrimitive::Aabb(aabb) => {
                (cull_aabb(&query.planes, &aabb), is_visible_aabb(&query.planes, &aabb))
            }
            FrustumAabbCullPrimitive::Sphere(sphere) => (
                cull_sphere(&query.planes, &sphere),
                is_visible_sphere(&query.planes, &sphere),
            ),
        };
        assert_eq!(verdict, want, "query {idx}: gpu verdict must match the reference");
        assert_eq!(
            verdict.is_visible(),
            want_visible,
            "query {idx}: gpu visibility must match the reference boolean shortcut",
        );
    }
}

#[test]
fn empty_batch_returns_no_verdicts() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumAabbCull::new(&ctx);
    // An empty slice must early-out with no dispatch (storage buffers cannot be
    // zero-sized) and no panic, exactly as the reference maps over nothing.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields no verdicts");
}

#[test]
fn hand_placed_cases_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumAabbCull::new(&ctx);
    // Boxes and spheres deliberately well clear of every plane band so each
    // falls into exactly one verdict; the reference computes the expected
    // verdict and the parity harness pins the match.
    let queries = [
        // Small box at mid-depth: fully inside the symmetric frustum.
        FrustumAabbCullQuery {
            planes: SYMMETRIC,
            primitive: FrustumAabbCullPrimitive::Aabb(Aabb::new([0.0, 0.0, 50.0], [1.0, 1.0, 1.0])),
        },
        // Box pushed far behind the far plane: outside.
        FrustumAabbCullQuery {
            planes: SYMMETRIC,
            primitive: FrustumAabbCullPrimitive::Aabb(Aabb::new(
                [0.0, 0.0, 500.0],
                [1.0, 1.0, 1.0],
            )),
        },
        // Wide box straddling the left plane: intersecting.
        FrustumAabbCullQuery {
            planes: SYMMETRIC,
            primitive: FrustumAabbCullPrimitive::Aabb(Aabb::new(
                [-50.0, 0.0, 50.0],
                [5.0, 1.0, 1.0],
            )),
        },
        // Sphere on the axis at mid-depth: inside.
        FrustumAabbCullQuery {
            planes: SYMMETRIC,
            primitive: FrustumAabbCullPrimitive::Sphere(Sphere::new([0.0, 0.0, 50.0], 2.0)),
        },
        // Sphere far behind the far plane: outside.
        FrustumAabbCullQuery {
            planes: SYMMETRIC,
            primitive: FrustumAabbCullPrimitive::Sphere(Sphere::new([0.0, 0.0, 500.0], 5.0)),
        },
        // Oblique-frustum centered box: inside the tilted volume.
        FrustumAabbCullQuery {
            planes: OBLIQUE,
            primitive: FrustumAabbCullPrimitive::Aabb(Aabb::new([0.0, 0.0, 50.0], [0.5, 0.5, 0.5])),
        },
        // Oblique-frustum box pushed far along +x: outside.
        FrustumAabbCullQuery {
            planes: OBLIQUE,
            primitive: FrustumAabbCullPrimitive::Aabb(Aabb::new(
                [400.0, 0.0, 50.0],
                [1.0, 1.0, 1.0],
            )),
        },
        // Oblique-frustum wide box reaching through a tilted side plane:
        // intersecting.
        FrustumAabbCullQuery {
            planes: OBLIQUE,
            primitive: FrustumAabbCullPrimitive::Aabb(Aabb::new(
                [0.0, 0.0, 50.0],
                [60.0, 1.0, 1.0],
            )),
        },
    ];
    check(&ctx, &gpu, &queries);

    // Confirm the batch is not vacuously one-sided: it must span all three
    // verdicts so the kernel's full decision fold is exercised.
    let verdicts = gpu.eval(&ctx, &queries);
    assert!(verdicts.contains(&Visibility::Inside), "batch covers Inside");
    assert!(
        verdicts.contains(&Visibility::Outside),
        "batch covers Outside"
    );
    assert!(
        verdicts.contains(&Visibility::Intersecting),
        "batch covers Intersecting"
    );
}

#[test]
fn random_boxes_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumAabbCull::new(&ctx);
    let mut state = 0x2b7e_1516_28ae_d2a6_u64;

    let mut queries = Vec::new();
    // Draw random boxes spanning the inside, outside and straddling regions,
    // keeping only those clear of every plane band so the exact verdict match
    // cannot flip on a near-tie.
    while queries.len() < 256 {
        let planes = if queries.len() % 2 == 0 {
            SYMMETRIC
        } else {
            OBLIQUE
        };
        let center = [
            range(&mut state, -120.0, 120.0),
            range(&mut state, -120.0, 120.0),
            range(&mut state, -40.0, 220.0),
        ];
        let half = [
            range(&mut state, 0.2, 25.0),
            range(&mut state, 0.2, 25.0),
            range(&mut state, 0.2, 25.0),
        ];
        let aabb = Aabb::new(center, half);
        if !aabb_clear_of_bands(&planes, &aabb) {
            continue;
        }
        queries.push(FrustumAabbCullQuery {
            planes,
            primitive: FrustumAabbCullPrimitive::Aabb(aabb),
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_spheres_match_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumAabbCull::new(&ctx);
    let mut state = 0x3243_f6a8_885a_308d_u64;

    let mut queries = Vec::new();
    while queries.len() < 256 {
        let planes = if queries.len() % 2 == 0 {
            SYMMETRIC
        } else {
            OBLIQUE
        };
        let center = [
            range(&mut state, -120.0, 120.0),
            range(&mut state, -120.0, 120.0),
            range(&mut state, -40.0, 220.0),
        ];
        let radius = range(&mut state, 0.2, 25.0);
        let sphere = Sphere::new(center, radius);
        if !sphere_clear_of_bands(&planes, &sphere) {
            continue;
        }
        queries.push(FrustumAabbCullQuery {
            planes,
            primitive: FrustumAabbCullPrimitive::Sphere(sphere),
        });
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuFrustumAabbCull::new(&ctx);
    let mut state = 0xa409_3822_299f_31d0_u64;

    // A single dispatch mixing boxes and spheres drawn from both frustums, to
    // confirm the per-query planes and the kind flag are wired independently.
    let mut queries = Vec::new();
    while queries.len() < 192 {
        let planes = if queries.len() % 2 == 0 {
            SYMMETRIC
        } else {
            OBLIQUE
        };
        let center = [
            range(&mut state, -120.0, 120.0),
            range(&mut state, -120.0, 120.0),
            range(&mut state, -40.0, 220.0),
        ];
        if lcg(&mut state) < 0.5 {
            let half = [
                range(&mut state, 0.2, 25.0),
                range(&mut state, 0.2, 25.0),
                range(&mut state, 0.2, 25.0),
            ];
            let aabb = Aabb::new(center, half);
            if !aabb_clear_of_bands(&planes, &aabb) {
                continue;
            }
            queries.push(FrustumAabbCullQuery {
                planes,
                primitive: FrustumAabbCullPrimitive::Aabb(aabb),
            });
        } else {
            let radius = range(&mut state, 0.2, 25.0);
            let sphere = Sphere::new(center, radius);
            if !sphere_clear_of_bands(&planes, &sphere) {
                continue;
            }
            queries.push(FrustumAabbCullQuery {
                planes,
                primitive: FrustumAabbCullPrimitive::Sphere(sphere),
            });
        }
    }
    check(&ctx, &gpu, &queries);
}
