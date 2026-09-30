//! Real-device parity tests for the device-resident `GPU` `BVH` closest-hit ray
//! query.
//!
//! Each test builds the tree resident on device with
//! [`GpuLbvh::build_resident`], casts a batch of rays with
//! [`GpuBvhRaycast::query_closest`] (no host round-trip between build and
//! traversal), and compares each hit to the
//! [`cpu_bvh_raycast_closest`] brute-force twin over the host-built [`Lbvh`].
//! The `GPU` traversal and the twin share the slab test bit-for-bit apart from
//! the reciprocal, whose `WGSL` division carries up to 2.5 `ULP`, so the primitive
//! index matches exactly and the entry distance within a small tolerance. The
//! scenes are built so the nearest box is unique and hits are square rather than
//! grazing, so the tiny reciprocal tolerance never flips a winner or a
//! hit/miss. Every test skips cleanly when no adapter is available (for example
//! inside a sandbox) so the suite never fails for lack of a `GPU`.
//!
//! Provenance: exercises Prism's own ray-traversal kernel (Williams et al. 2005
//! slab over Hapala et al. 2011 traversal) on the linear `BVH` of Karras (2012)
//! against its `CPU` twin. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_build_lbvh, cpu_bvh_raycast_any, cpu_bvh_raycast_closest, Aabb, GpuBvhRaycast, GpuContext,
    GpuLbvh, Ray, RayHit,
};

/// A small xorshift generator so the tests are deterministic without pulling in
/// an `RNG` dependency.
struct Rng {
    /// Mutable generator state; never zero.
    state: u64,
}

impl Rng {
    /// Seeds the generator, forcing a non-zero state.
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    /// Advances the state and returns the next 64-bit value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Returns an `f32` in `[lo, lo + span)`.
    fn coord(&mut self, lo: f32, span: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + unit * span
    }
}

/// A unit box centred on `(x, y, z)`.
fn box_at(x: f32, y: f32, z: f32) -> Aabb {
    let c = Vec3::new(x, y, z);
    let half = Vec3::splat(0.5);
    Aabb::new(c - half, c + half)
}

/// The tolerance on the entry distance; the only inexact step is the `WGSL`
/// reciprocal (up to 2.5 `ULP`), which over these scene scales stays well under
/// this bound.
const T_TOL: f32 = 1e-3;

/// Compares one `GPU` closest hit to the `CPU` twin's, allowing only the
/// reciprocal tolerance.
///
/// A miss must match a miss and a hit a hit; when both hit, the primitive must
/// be identical and the distance within [`T_TOL`]. The scenes are unique-nearest
/// and square, so neither the winning primitive nor the hit/miss decision can
/// flip within the tolerance.
#[expect(clippy::print_stderr, reason = "surface the ray on parity failure")]
fn assert_hit_matches(index: usize, ray: Ray, want: Option<RayHit>, got: Option<RayHit>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            if w.prim != g.prim || (w.t - g.t).abs() > T_TOL {
                eprintln!(
                    "ray {index} origin {:?} dir {:?}: cpu {w:?} vs gpu {g:?}",
                    ray.origin, ray.dir
                );
                panic!("closest-hit mismatch at ray {index}");
            }
        }
        (want, got) => {
            eprintln!(
                "ray {index} origin {:?} dir {:?}: cpu {want:?} vs gpu {got:?}",
                ray.origin, ray.dir
            );
            panic!("hit/miss disagreement at ray {index}");
        }
    }
}

/// Builds the tree resident on device, casts `rays`, and asserts every hit equals
/// the `CPU` twin's over the host-built tree.
fn assert_closest_matches(
    builder: &GpuLbvh,
    raycast: &GpuBvhRaycast,
    ctx: &GpuContext,
    boxes: &[Aabb],
    rays: &[Ray],
) {
    let tree = cpu_build_lbvh(boxes);
    let resident = builder.build_resident(ctx, boxes);
    let got = raycast.query_closest(ctx, &resident, rays);
    ctx.wait();
    assert_eq!(got.len(), rays.len(), "one hit slot per ray");
    for (i, ray) in rays.iter().enumerate() {
        let want = cpu_bvh_raycast_closest(&tree, *ray);
        assert_hit_matches(i, *ray, want, got[i]);
    }
}

#[test]
fn wall_of_boxes_closest_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    // Ten unit boxes strung along +x, five units apart, so entry distances are
    // distinct by four units and no reciprocal tolerance can reorder them.
    let boxes: Vec<Aabb> = (1..=10).map(|k| box_at(k as f32 * 5.0, 0.0, 0.0)).collect();
    let mut rng = Rng::new(0x1234_5678_9abc_def0);
    let mut rays = Vec::new();
    // Forward rays from the origin, jittered inside the first box's face so they
    // still strike box 0 (at x = 5) squarely.
    for _ in 0..40 {
        let y = rng.coord(-0.2, 0.4);
        let z = rng.coord(-0.2, 0.4);
        rays.push(Ray::new(
            Vec3::ZERO,
            Vec3::new(1.0, y, z).normalize(),
            100.0,
        ));
    }
    // Reversed rays from beyond the far box strike box 9 (at x = 50) squarely.
    for _ in 0..20 {
        let y = rng.coord(-0.2, 0.4);
        let z = rng.coord(-0.2, 0.4);
        rays.push(Ray::new(
            Vec3::new(60.0, 0.0, 0.0),
            Vec3::new(-1.0, y, z).normalize(),
            100.0,
        ));
    }
    // Rays shooting straight up miss the whole x-axis wall.
    for _ in 0..20 {
        let x = rng.coord(5.0, 45.0);
        rays.push(Ray::new(Vec3::new(x, -5.0, 0.0), Vec3::Y, 3.0));
    }
    assert_closest_matches(&builder, &raycast, &ctx, &boxes, &rays);
}

#[test]
fn sparse_targets_closest_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    // A coarse grid of unit boxes eight units apart: sparse enough that a ray
    // aimed at one centre strikes it squarely and clips no neighbour.
    let mut centres = Vec::new();
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                centres.push(Vec3::new(x as f32 * 8.0, y as f32 * 8.0, z as f32 * 8.0));
            }
        }
    }
    let boxes: Vec<Aabb> = centres.iter().map(|c| box_at(c.x, c.y, c.z)).collect();
    let mut rng = Rng::new(0x0f0e_0d0c_0b0a_0908);
    let mut rays = Vec::new();
    for _ in 0..60 {
        let target = centres[(rng.next_u64() as usize) % centres.len()];
        let origin = Vec3::new(
            rng.coord(-40.0, 20.0),
            rng.coord(-40.0, 20.0),
            rng.coord(-40.0, 20.0),
        );
        let dir = (target - origin).normalize();
        rays.push(Ray::new(origin, dir, 500.0));
    }
    assert_closest_matches(&builder, &raycast, &ctx, &boxes, &rays);
}

#[test]
fn t_max_clamps_the_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    // Two well-separated boxes; the nearest face is at x = 4.5.
    let boxes = [box_at(5.0, 0.0, 0.0), box_at(20.0, 0.0, 0.0)];
    let rays = [
        // t_max below the near face: clean miss, not a grazing boundary.
        Ray::new(Vec3::ZERO, Vec3::X, 4.0),
        // t_max past it: clean hit on box 0.
        Ray::new(Vec3::ZERO, Vec3::X, 6.0),
        // t_max between the two boxes: hit box 0, box 1 out of range.
        Ray::new(Vec3::ZERO, Vec3::X, 10.0),
    ];
    assert_closest_matches(&builder, &raycast, &ctx, &boxes, &rays);
}

#[test]
fn trivial_trees_report_only_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    // A resident tree with fewer than two leaves owns no buffers, so the
    // resident kernel reports a miss for every ray regardless of geometry. The
    // full n <= 1 behaviour is covered by the CPU twin's own unit tests.
    let rays = [
        Ray::new(Vec3::ZERO, Vec3::X, 100.0),
        Ray::new(Vec3::new(1.0, 2.0, 3.0), Vec3::NEG_Z, 100.0),
    ];
    for boxes in [Vec::new(), vec![box_at(0.0, 0.0, 0.0)]] {
        let resident = builder.build_resident(&ctx, &boxes);
        let got = raycast.query_closest(&ctx, &resident, &rays);
        ctx.wait();
        assert_eq!(got.len(), rays.len());
        assert!(got.iter().all(Option::is_none), "trivial tree hits nothing");
    }
}

#[test]
fn empty_ray_batch_returns_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    let boxes = [box_at(0.0, 0.0, 0.0), box_at(2.0, 0.0, 0.0)];
    let resident = builder.build_resident(&ctx, &boxes);
    let got = raycast.query_closest(&ctx, &resident, &[]);
    ctx.wait();
    assert!(got.is_empty(), "no rays yields no hits");
}

/// Builds the tree resident on device, casts `rays` with the any-hit kernel, and
/// asserts each flag exactly equals the [`cpu_bvh_raycast_any`] brute-force twin
/// over the host-built tree.
///
/// Any-hit answers a boolean, so the tolerance that closest-hit distances need
/// never enters: every scene here is built so each ray either clears every box
/// or strikes one squarely, and the flag must match the twin bit-for-bit.
#[expect(clippy::print_stderr, reason = "surface the ray on parity failure")]
fn assert_any_matches(
    builder: &GpuLbvh,
    raycast: &GpuBvhRaycast,
    ctx: &GpuContext,
    boxes: &[Aabb],
    rays: &[Ray],
) {
    let tree = cpu_build_lbvh(boxes);
    let resident = builder.build_resident(ctx, boxes);
    let got = raycast.query_any(ctx, &resident, rays);
    ctx.wait();
    assert_eq!(got.len(), rays.len(), "one flag slot per ray");
    for (i, ray) in rays.iter().enumerate() {
        let want = cpu_bvh_raycast_any(&tree, *ray);
        if want != got[i] {
            eprintln!(
                "ray {i} origin {:?} dir {:?}: cpu {want} vs gpu {}",
                ray.origin, ray.dir, got[i]
            );
            panic!("any-hit mismatch at ray {i}");
        }
    }
}

#[test]
fn wall_of_boxes_any_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    // The same +x wall the closest-hit test uses: ten unit boxes five units
    // apart, so a forward ray blocks and a straight-up ray clears the wall.
    let boxes: Vec<Aabb> = (1..=10).map(|k| box_at(k as f32 * 5.0, 0.0, 0.0)).collect();
    let mut rng = Rng::new(0x5151_5151_a2a2_a2a2);
    let mut rays = Vec::new();
    // Forward rays that strike box 0 squarely: each is blocked.
    for _ in 0..40 {
        let y = rng.coord(-0.2, 0.4);
        let z = rng.coord(-0.2, 0.4);
        rays.push(Ray::new(
            Vec3::ZERO,
            Vec3::new(1.0, y, z).normalize(),
            100.0,
        ));
    }
    // Rays shooting straight up between the boxes clear the whole wall.
    for _ in 0..20 {
        let x = rng.coord(5.0, 45.0);
        rays.push(Ray::new(Vec3::new(x, -5.0, 0.0), Vec3::Y, 3.0));
    }
    assert_any_matches(&builder, &raycast, &ctx, &boxes, &rays);
}

#[test]
fn sparse_targets_any_matches_the_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    // A coarse grid eight units apart. Half the rays aim at a box centre (a
    // clean block); half aim into the wide empty gaps between planes (a clean
    // clear), so no ray grazes a boundary.
    let mut centres = Vec::new();
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                centres.push(Vec3::new(x as f32 * 8.0, y as f32 * 8.0, z as f32 * 8.0));
            }
        }
    }
    let boxes: Vec<Aabb> = centres.iter().map(|c| box_at(c.x, c.y, c.z)).collect();
    let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
    let mut rays = Vec::new();
    // Aimed at a centre: blocked.
    for _ in 0..40 {
        let target = centres[(rng.next_u64() as usize) % centres.len()];
        let origin = Vec3::new(
            rng.coord(-40.0, 20.0),
            rng.coord(-40.0, 20.0),
            rng.coord(-40.0, 20.0),
        );
        let dir = (target - origin).normalize();
        rays.push(Ray::new(origin, dir, 500.0));
    }
    // Fired straight along +x on a y/z gap plane: the box centres sit on
    // multiples of eight, so a track near y = z = 4 stays at least three units
    // clear of every box and the ray traverses the whole grid hitting nothing.
    for _ in 0..20 {
        let y = rng.coord(3.5, 1.0);
        let z = rng.coord(3.5, 1.0);
        let origin = Vec3::new(-40.0, y, z);
        rays.push(Ray::new(origin, Vec3::X, 120.0));
    }
    assert_any_matches(&builder, &raycast, &ctx, &boxes, &rays);
}

#[test]
fn t_max_gates_the_any_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    // Two well-separated boxes; the nearest face is at x = 4.5, the far one at
    // x = 19.5.
    let boxes = [box_at(5.0, 0.0, 0.0), box_at(20.0, 0.0, 0.0)];
    let rays = [
        // t_max short of the near face: nothing in range.
        Ray::new(Vec3::ZERO, Vec3::X, 4.0),
        // t_max past the near face: blocked by box 0.
        Ray::new(Vec3::ZERO, Vec3::X, 6.0),
        // Straight up from between the boxes: clears both.
        Ray::new(Vec3::new(12.0, -5.0, 0.0), Vec3::Y, 3.0),
    ];
    assert_any_matches(&builder, &raycast, &ctx, &boxes, &rays);
}

#[test]
fn trivial_trees_report_no_any_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    // A resident tree with fewer than two leaves owns no buffers, so the
    // resident kernel reports no hit for every ray. The full n <= 1 behaviour is
    // covered by the CPU twin's own unit tests.
    let rays = [
        Ray::new(Vec3::ZERO, Vec3::X, 100.0),
        Ray::new(Vec3::new(1.0, 2.0, 3.0), Vec3::NEG_Z, 100.0),
    ];
    for boxes in [Vec::new(), vec![box_at(0.0, 0.0, 0.0)]] {
        let resident = builder.build_resident(&ctx, &boxes);
        let got = raycast.query_any(&ctx, &resident, &rays);
        ctx.wait();
        assert_eq!(got.len(), rays.len());
        assert!(got.iter().all(|hit| !hit), "trivial tree hits nothing");
    }
}

#[test]
fn empty_ray_batch_returns_empty_any() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let builder = GpuLbvh::new(&ctx);
    let raycast = GpuBvhRaycast::new(&ctx);
    let boxes = [box_at(0.0, 0.0, 0.0), box_at(2.0, 0.0, 0.0)];
    let resident = builder.build_resident(&ctx, &boxes);
    let got = raycast.query_any(&ctx, &resident, &[]);
    ctx.wait();
    assert!(got.is_empty(), "no rays yields no flags");
}
