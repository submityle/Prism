//! Real-device parity for the particle camera-query twin:
//! [`GpuCamera`](prism_volumetric_gpu::camera::GpuCamera) must reproduce the
//! `CPU` golden
//! [`CameraProjection`](prism_render_architecture::particle::camera::CameraProjection)
//! query set — the `world`->`view`->`clip`->`NDC`->`screen` transform chain
//! ([`world_to_view`](prism_render_architecture::particle::camera::CameraProjection::world_to_view),
//! [`view_to_clip`](prism_render_architecture::particle::camera::CameraProjection::view_to_clip),
//! [`clip_to_ndc`](prism_render_architecture::particle::camera::CameraProjection::clip_to_ndc),
//! [`world_to_ndc`](prism_render_architecture::particle::camera::CameraProjection::world_to_ndc),
//! [`ndc_to_screen`](prism_render_architecture::particle::camera::CameraProjection::ndc_to_screen),
//! [`world_to_screen`](prism_render_architecture::particle::camera::CameraProjection::world_to_screen)),
//! the coarse frustum bucket
//! ([`classify_visibility`](prism_render_architecture::particle::camera::CameraProjection::classify_visibility)),
//! the depth and camera-relative scalars
//! ([`linearize_depth`](prism_render_architecture::particle::camera::CameraProjection::linearize_depth),
//! [`distance_to_camera`](prism_render_architecture::particle::camera::CameraProjection::distance_to_camera),
//! [`distance_squared_to_camera`](prism_render_architecture::particle::camera::CameraProjection::distance_squared_to_camera),
//! [`direction_to_camera`](prism_render_architecture::particle::camera::CameraProjection::direction_to_camera))
//! and the screen-coverage inputs
//! ([`screen_coverage_ndc`](prism_render_architecture::particle::camera::CameraProjection::screen_coverage_ndc),
//! [`screen_coverage_pixels`](prism_render_architecture::particle::camera::CameraProjection::screen_coverage_pixels))
//! across an empty batch, an on-axis centred projection, an off-axis interior
//! point, a point behind the camera, a point exactly on the camera plane, a
//! laterally frustum-culled point, a depth-invariant orthographic coverage
//! pair, an orthographic point behind the near plane, a near/far linearization
//! pair, a distance/direction probe, a rotated-basis interior point, and a
//! large pseudo-random batch compared lane for lane across three cameras.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each lane is a fixed, non-reorderable sequence of dot products, guarded
//! reciprocals and a perspective divide, so `CPU` and `GPU` evaluate the same
//! closed form in the same associativity. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The continuous outputs
//! therefore allow `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR` `1e-6`)
//! while the discrete classifications — the three [`Option`] flags and the
//! visibility code — must match *exactly*. Every fixture and every random lane
//! is placed clear of the branch boundaries (a `w` at the camera plane, the
//! near/far planes and the `NDC` `±1` edges) by a wide margin, so a legal `ULP`
//! perturbation can never flip a `<=`/`>=` verdict and the exact assertions hold
//! unconditionally; the sole on-plane fixture pins a value both devices compute
//! as an exact `0.0`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::camera`；standard
//! projective camera algebra; no third-party engine source or derived code.

use prism_render_architecture::particle::camera::{
    CameraBasis, CameraProjection, ClipVisibility, ProjectionKind, Viewport,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::camera::{cpu_reference, CameraQuery, CameraResult, GpuCamera};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous outputs. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// Returns whether two vectors agree component-wise within [`close`].
fn vclose(a: Vec3, b: Vec3) -> bool {
    close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
}

/// The axis-aligned reference perspective camera: at the origin looking down
/// `+Z`, a symmetric `90°` `FOV` (`tan(45°) = 1`), near `1`, far `100`, into an
/// `800x600` target.
fn perspective_camera() -> CameraProjection {
    CameraProjection::perspective(
        Vec3::ZERO,
        CameraBasis::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
        1.0,
        100.0,
        1.0,
        1.0,
        Viewport::new(800.0, 600.0),
    )
}

/// The axis-aligned reference orthographic camera: at the origin looking down
/// `+Z`, `10` half-extents, near `1`, far `100`, into an `800x600` target.
fn ortho_camera() -> CameraProjection {
    CameraProjection::orthographic(
        Vec3::ZERO,
        CameraBasis::new(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ),
        1.0,
        100.0,
        10.0,
        10.0,
        Viewport::new(800.0, 600.0),
    )
}

/// A perspective camera with an off-origin position and a rotated orthonormal
/// basis, built with the host-side (sqrt-only, trig-free)
/// [`from_forward_up`](prism_render_architecture::particle::camera::CameraBasis::from_forward_up),
/// so the twin's basis dot products are exercised away from the axis-aligned
/// identity. Asymmetric `FOV` tangents and a `1280x720` target.
fn rotated_camera() -> CameraProjection {
    CameraProjection::perspective(
        Vec3::new(2.0, 1.0, -3.0),
        CameraBasis::from_forward_up(Vec3::new(1.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0)),
        1.0,
        50.0,
        0.8,
        0.6,
        Viewport::new(1280.0, 720.0),
    )
}

/// Builds one query from a world point and the standalone samples.
fn cq(world_pos: Vec3, ndc_point: Vec3, ndc_z: f32, radius: f32) -> CameraQuery {
    CameraQuery {
        world_pos,
        ndc_point,
        ndc_z,
        radius,
    }
}

/// Rebuilds a world point from view-space coordinates through the camera basis,
/// so a fixture can place a point at a chosen view depth/offset: `world =
/// position + right * vx + up * vy + forward * vz`.
fn view_to_world(cam: &CameraProjection, v: Vec3) -> Vec3 {
    cam.world_position
        .add(cam.basis.right.scale(v.x))
        .add(cam.basis.up.scale(v.y))
        .add(cam.basis.forward.scale(v.z))
}

/// Runs the `GPU` dispatch and asserts strict lane-for-lane parity against the
/// `CPU` golden: the visibility code and the three [`Option`] flags match
/// exactly, and every continuous output matches within tolerance (the
/// [`Option`] payloads only where both sides carry a value). Returns the `GPU`
/// verdicts for extra per-test assertions. Use only for lanes placed clear of
/// every branch boundary.
fn check(
    ctx: &GpuContext,
    gpu: &GpuCamera,
    camera: &CameraProjection,
    queries: &[CameraQuery],
) -> Vec<CameraResult> {
    let got = gpu.eval(ctx, camera, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        let c = cpu_reference(camera, q);

        // Discrete classifications: exact.
        assert_eq!(g.visibility, c.visibility, "lane {lane}: visibility");
        assert_eq!(
            g.ndc_from_clip.is_some(),
            c.ndc_from_clip.is_some(),
            "lane {lane}: ndc_from_clip flag"
        );
        assert_eq!(
            g.ndc_from_world.is_some(),
            c.ndc_from_world.is_some(),
            "lane {lane}: ndc_from_world flag"
        );
        assert_eq!(
            g.screen_from_world.is_some(),
            c.screen_from_world.is_some(),
            "lane {lane}: screen_from_world flag"
        );

        // Continuous outputs: tolerant.
        assert!(
            vclose(g.view, c.view),
            "lane {lane}: view gpu {:?} vs cpu {:?}",
            g.view,
            c.view
        );
        for k in 0..4 {
            assert!(
                close(g.clip[k], c.clip[k]),
                "lane {lane}: clip[{k}] gpu {} vs cpu {}",
                g.clip[k],
                c.clip[k]
            );
        }
        if let (Some(a), Some(b)) = (g.ndc_from_clip, c.ndc_from_clip) {
            assert!(
                vclose(a, b),
                "lane {lane}: ndc_from_clip gpu {a:?} vs cpu {b:?}"
            );
        }
        if let (Some(a), Some(b)) = (g.ndc_from_world, c.ndc_from_world) {
            assert!(
                vclose(a, b),
                "lane {lane}: ndc_from_world gpu {a:?} vs cpu {b:?}"
            );
        }
        for k in 0..2 {
            assert!(
                close(g.screen_from_ndc[k], c.screen_from_ndc[k]),
                "lane {lane}: screen_from_ndc[{k}] gpu {} vs cpu {}",
                g.screen_from_ndc[k],
                c.screen_from_ndc[k]
            );
        }
        if let (Some(a), Some(b)) = (g.screen_from_world, c.screen_from_world) {
            for k in 0..2 {
                assert!(
                    close(a[k], b[k]),
                    "lane {lane}: screen_from_world[{k}] gpu {} vs cpu {}",
                    a[k],
                    b[k]
                );
            }
        }
        assert!(
            close(g.linearize_depth, c.linearize_depth),
            "lane {lane}: linearize_depth gpu {} vs cpu {}",
            g.linearize_depth,
            c.linearize_depth
        );
        assert!(
            close(g.distance, c.distance),
            "lane {lane}: distance gpu {} vs cpu {}",
            g.distance,
            c.distance
        );
        assert!(
            close(g.distance_squared, c.distance_squared),
            "lane {lane}: distance_squared gpu {} vs cpu {}",
            g.distance_squared,
            c.distance_squared
        );
        assert!(
            close(g.coverage_ndc, c.coverage_ndc),
            "lane {lane}: coverage_ndc gpu {} vs cpu {}",
            g.coverage_ndc,
            c.coverage_ndc
        );
        assert!(
            close(g.coverage_pixels, c.coverage_pixels),
            "lane {lane}: coverage_pixels gpu {} vs cpu {}",
            g.coverage_pixels,
            c.coverage_pixels
        );
        assert!(
            vclose(g.direction, c.direction),
            "lane {lane}: direction gpu {:?} vs cpu {:?}",
            g.direction,
            c.direction
        );
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
    let gpu = GpuCamera::new(&ctx);
    let cam = perspective_camera();
    let got = gpu.eval(&ctx, &cam, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn on_axis_centre_projection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = perspective_camera();
    // A point straight down the optical axis at depth 10: NDC centre, so the
    // screen position is the viewport centre and the point is visible.
    let q = cq(
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::new(0.0, 0.0, 0.5),
        0.5,
        1.0,
    );
    let got = check(&ctx, &gpu, &cam, &[q]);
    assert_eq!(
        got[0].visibility,
        ClipVisibility::Visible,
        "on-axis visible"
    );
    let screen = got[0]
        .screen_from_world
        .expect("on-axis point is on screen");
    assert!(close(screen[0], 400.0), "screen x {}", screen[0]);
    assert!(close(screen[1], 300.0), "screen y {}", screen[1]);
    let ndc = got[0].ndc_from_world.expect("on-axis point projects");
    assert!(close(ndc.x, 0.0) && close(ndc.y, 0.0), "centre NDC {ndc:?}");
}

#[test]
fn off_axis_interior_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = perspective_camera();
    // View (3, 2, 10) -> NDC (0.3, 0.2, ...): comfortably inside the frustum.
    let q = cq(
        Vec3::new(3.0, 2.0, 10.0),
        Vec3::new(-0.4, 0.6, 0.3),
        0.5,
        1.5,
    );
    let got = check(&ctx, &gpu, &cam, &[q]);
    assert_eq!(
        got[0].visibility,
        ClipVisibility::Visible,
        "interior visible"
    );
    let ndc = got[0].ndc_from_world.expect("interior point projects");
    assert!(close(ndc.x, 0.3) && close(ndc.y, 0.2), "offset NDC {ndc:?}");
}

#[test]
fn behind_camera_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = perspective_camera();
    // View z = -5: behind the camera plane. world_to_ndc and world_to_screen
    // are None, but clip_to_ndc still divides by the (non-degenerate) w = -5, so
    // ndc_from_clip is Some — the twin must reproduce that split.
    let q = cq(
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(0.0, 0.0, 0.5),
        0.5,
        1.0,
    );
    let got = check(&ctx, &gpu, &cam, &[q]);
    assert_eq!(
        got[0].visibility,
        ClipVisibility::BehindCamera,
        "behind-camera class"
    );
    assert!(
        got[0].ndc_from_world.is_none() && got[0].screen_from_world.is_none(),
        "a behind-camera point does not project to NDC or screen"
    );
    assert!(
        got[0].ndc_from_clip.is_some(),
        "clip_to_ndc still divides by a non-degenerate w"
    );
}

#[test]
fn point_exactly_on_camera_plane() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = perspective_camera();
    // rel . forward = 0 exactly on both devices, so clip w is an exact 0.0: the
    // guarded divide collapses and both NDC mappings are None, classified as
    // behind the camera plane.
    let q = cq(Vec3::new(4.0, 3.0, 0.0), Vec3::new(0.0, 0.0, 0.5), 0.5, 1.0);
    let got = check(&ctx, &gpu, &cam, &[q]);
    assert_eq!(
        got[0].visibility,
        ClipVisibility::BehindCamera,
        "on-plane class"
    );
    assert!(
        got[0].ndc_from_clip.is_none() && got[0].ndc_from_world.is_none(),
        "a w at the camera plane yields no NDC"
    );
    assert!(
        got[0].screen_from_world.is_none(),
        "and therefore no screen position"
    );
}

#[test]
fn laterally_frustum_culled_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = perspective_camera();
    // View (20, 0, 10) -> NDC x = 2.0: in front of the camera but well outside
    // the clip cube, so it projects to NDC yet world_to_screen is culled.
    let q = cq(
        Vec3::new(20.0, 0.0, 10.0),
        Vec3::new(0.0, 0.0, 0.5),
        0.5,
        1.0,
    );
    let got = check(&ctx, &gpu, &cam, &[q]);
    assert_eq!(
        got[0].visibility,
        ClipVisibility::OutsideFrustum,
        "outside-frustum class"
    );
    assert!(
        got[0].ndc_from_world.is_some() && got[0].screen_from_world.is_none(),
        "a culled point still has an NDC but no screen position"
    );
}

#[test]
fn ortho_coverage_is_depth_invariant() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = ortho_camera();
    // Orthographic coverage is depth-independent: radius / half_height = 0.2 at
    // both depths.
    let near = cq(
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::new(0.0, 0.0, 0.5),
        0.5,
        2.0,
    );
    let far = cq(
        Vec3::new(0.0, 0.0, 50.0),
        Vec3::new(0.0, 0.0, 0.5),
        0.5,
        2.0,
    );
    let got = check(&ctx, &gpu, &cam, &[near, far]);
    assert!(
        close(got[0].coverage_ndc, got[1].coverage_ndc),
        "coverage should not vary with depth under orthographic"
    );
    assert!(
        close(got[0].coverage_ndc, 0.2),
        "coverage {}",
        got[0].coverage_ndc
    );
}

#[test]
fn ortho_point_behind_near_plane_is_outside() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = ortho_camera();
    // Orthographic has no behind-camera short-circuit (w is always 1), so a
    // point before the near plane lands at NDC z < 0 and is OutsideFrustum, not
    // BehindCamera.
    let q = cq(
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(0.0, 0.0, 0.5),
        0.5,
        1.0,
    );
    let got = check(&ctx, &gpu, &cam, &[q]);
    assert_eq!(
        got[0].visibility,
        ClipVisibility::OutsideFrustum,
        "ortho before-near class"
    );
    assert!(
        got[0].ndc_from_clip.is_some() && got[0].ndc_from_world.is_some(),
        "orthographic w = 1 keeps both NDC mappings valid"
    );
    assert!(
        got[0].screen_from_world.is_none(),
        "but the out-of-range depth culls the screen position"
    );
}

#[test]
fn linearize_at_near_and_far() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = perspective_camera();
    // ndc_z 0 inverts to the near plane, ndc_z 1 to the far plane.
    let at_near = cq(
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::new(0.0, 0.0, 0.5),
        0.0,
        1.0,
    );
    let at_far = cq(
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::new(0.0, 0.0, 0.5),
        1.0,
        1.0,
    );
    let got = check(&ctx, &gpu, &cam, &[at_near, at_far]);
    assert!(
        close(got[0].linearize_depth, 1.0),
        "near {}",
        got[0].linearize_depth
    );
    assert!(
        close(got[1].linearize_depth, 100.0),
        "far {}",
        got[1].linearize_depth
    );
}

#[test]
fn distance_and_direction_probe() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = perspective_camera();
    // From (0, 0, 10) the camera at the origin is straight back along -Z at a
    // distance of 10.
    let q = cq(
        Vec3::new(0.0, 0.0, 10.0),
        Vec3::new(0.0, 0.0, 0.5),
        0.5,
        1.0,
    );
    let got = check(&ctx, &gpu, &cam, &[q]);
    assert!(close(got[0].distance, 10.0), "distance {}", got[0].distance);
    assert!(
        close(got[0].distance_squared, 100.0),
        "dist_sq {}",
        got[0].distance_squared
    );
    assert!(
        vclose(got[0].direction, Vec3::new(0.0, 0.0, -1.0)),
        "direction {:?}",
        got[0].direction
    );
}

#[test]
fn rotated_basis_interior_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cam = rotated_camera();
    // Place a point squarely in front of the rotated camera via its own basis,
    // exercising the non-axis-aligned view dot products.
    let world = view_to_world(&cam, Vec3::new(1.0, 0.5, 12.0));
    let q = cq(world, Vec3::new(0.1, -0.2, 0.4), 0.5, 1.0);
    let got = check(&ctx, &gpu, &cam, &[q]);
    assert_eq!(
        got[0].visibility,
        ClipVisibility::Visible,
        "rotated interior visible"
    );
    assert!(
        vclose(got[0].view, Vec3::new(1.0, 0.5, 12.0)),
        "view round-trips through the basis: {:?}",
        got[0].view
    );
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCamera::new(&ctx);
    let cams = [perspective_camera(), rotated_camera(), ortho_camera()];
    let mut state = 0x_c0ff_ee12_3456_789a_u64;

    let mut saw_visible = false;
    let mut saw_behind = false;
    let mut saw_outside = false;

    for cam in &cams {
        let (p0, p1, is_persp) = match cam.projection {
            ProjectionKind::Perspective {
                tan_half_fov_x,
                tan_half_fov_y,
            } => (tan_half_fov_x, tan_half_fov_y, true),
            ProjectionKind::Orthographic {
                half_width,
                half_height,
            } => (half_width, half_height, false),
        };
        let near = cam.near;
        let far = cam.far;
        let range = far - near;

        let mut queries = Vec::with_capacity(96);
        for _ in 0..96 {
            // Three buckets, each placed far from any branch boundary so the
            // discrete classification is unambiguous on both devices.
            let bucket = (lcg(&mut state) * 3.0) as u32;
            let ndcx = lcg(&mut state) * 1.4 - 0.7;
            let ndcy = lcg(&mut state) * 1.4 - 0.7;
            let ndcz = 0.2 + lcg(&mut state) * 0.6;

            let view = if is_persp {
                match bucket {
                    // Clearly behind the camera plane.
                    1 => {
                        let vz = -(2.0 + lcg(&mut state) * 20.0);
                        Vec3::new(ndcx * 0.3 * p0, ndcy * 0.3 * p1, vz)
                    }
                    // In front but laterally outside the clip cube.
                    2 => {
                        let vz = far * near / (far - ndcz * range);
                        let outx = 1.6 + lcg(&mut state) * 1.4;
                        Vec3::new(outx * vz * p0, ndcy * vz * p1, vz)
                    }
                    // Clearly inside the frustum.
                    _ => {
                        let vz = far * near / (far - ndcz * range);
                        Vec3::new(ndcx * vz * p0, ndcy * vz * p1, vz)
                    }
                }
            } else {
                match bucket {
                    // Before the near plane -> NDC z < 0 -> outside (no behind
                    // class under orthographic).
                    1 | 2 => {
                        let vz = near - (1.0 + lcg(&mut state) * 5.0);
                        Vec3::new(ndcx * p0, ndcy * p1, vz)
                    }
                    // Clearly inside the clip box.
                    _ => {
                        let vz = near + ndcz * range;
                        Vec3::new(ndcx * p0, ndcy * p1, vz)
                    }
                }
            };

            let world = view_to_world(cam, view);
            let ndc_point = Vec3::new(
                lcg(&mut state) * 1.6 - 0.8,
                lcg(&mut state) * 1.6 - 0.8,
                0.1 + lcg(&mut state) * 0.8,
            );
            let ndc_z = 0.1 + lcg(&mut state) * 0.8;
            let radius = 0.5 + lcg(&mut state) * 2.0;
            queries.push(cq(world, ndc_point, ndc_z, radius));
        }

        let got = check(&ctx, &gpu, cam, &queries);
        for g in &got {
            match g.visibility {
                ClipVisibility::Visible => saw_visible = true,
                ClipVisibility::BehindCamera => saw_behind = true,
                ClipVisibility::OutsideFrustum => saw_outside = true,
                ClipVisibility::Degenerate => {}
            }
        }
    }

    // A large spread across both projection kinds must exercise the interior,
    // behind-camera and frustum-culled classes, so the test is not trivially
    // passing on a single-class batch.
    assert!(
        saw_visible && saw_behind && saw_outside,
        "random batch should produce visible, behind-camera and outside-frustum lanes"
    );
}
