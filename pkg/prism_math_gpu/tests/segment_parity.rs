//! Real-device parity for the §24.1 GPU segment closest-point shader mirror.
//!
//! Each test builds a batch of segment/point queries, solves them on the CPU
//! with [`prism_math::geom::segment::Segment3`]'s `closest_t` / `closest_point`
//! / `distance_squared`, runs the same query on a real `GPU` from the
//! single-sourced [`WGSL_SEGMENT`](prism_math::shader_mirror::WGSL_SEGMENT)
//! fragment, and asserts the parameter `t`, the closest point, and the squared
//! distance agree within a small tolerance.
//!
//! The projection divide and the dot product are float arithmetic, and Metal
//! compiles WGSL under fast-math, so the three outputs agree only within a
//! small absolute+relative epsilon. The suite uses non-degenerate segments with
//! a comfortable margin above the `1.0e-20` squared-length guard so the clamped
//! parameter and the interior/endpoint classification are unambiguous. It skips
//! gracefully when no adapter is available so it still passes on a device-less
//! CI image while running the full dispatch on a real `GPU`.

use prism_math::geom::segment::Segment3;
use prism_math::Vec3;
use prism_math_gpu::{GpuContext, GpuSegment, GpuSegmentHit, GpuSegmentQuery};

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Absolute+relative closeness, matching the fast-math tolerance contract.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    diff <= 1.0e-4 + 1.0e-4 * a.abs().max(b.abs())
}

/// Compares a decoded GPU hit against the CPU `Segment3` reference trio.
fn assert_query(gpu: GpuSegmentHit, seg: Segment3, point: Vec3, tag: &str) {
    let cpu_t = seg.closest_t(point);
    let cpu_cp = seg.closest_point(point);
    let cpu_ds = seg.distance_squared(point);
    assert!(
        close(gpu.t, cpu_t),
        "{tag}: t mismatch gpu={} cpu={}",
        gpu.t,
        cpu_t
    );
    for (axis, (g, c)) in gpu
        .closest_point
        .iter()
        .zip([cpu_cp.x, cpu_cp.y, cpu_cp.z])
        .enumerate()
    {
        assert!(
            close(*g, c),
            "{tag}: closest_point[{axis}] mismatch gpu={g} cpu={c}"
        );
    }
    assert!(
        close(gpu.distance_squared, cpu_ds),
        "{tag}: distance_squared mismatch gpu={} cpu={}",
        gpu.distance_squared,
        cpu_ds
    );
}

#[test]
fn segment_closest_point_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSegment::new(&ctx);

    // A mix of interior projections (t in (0,1)), both endpoint clamps, an
    // axis-aligned segment, and a diagonal segment — never degenerate.
    let cases = [
        (
            Segment3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
            Vec3::new(1.0, 3.0, 0.0),
        ),
        (
            Segment3::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(5.0, 1.0, 0.0),
        ),
        (
            Segment3::new(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(-5.0, 2.0, 0.0),
        ),
        (
            Segment3::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 4.0, 0.0)),
            Vec3::new(3.0, 2.0, 1.0),
        ),
        (
            Segment3::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 6.0, 3.0)),
            Vec3::new(2.0, 3.0, 5.0),
        ),
        (
            Segment3::new(Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0)),
            Vec3::new(1.0, -1.0, 0.5),
        ),
    ];

    let gpu_queries: Vec<GpuSegmentQuery> = cases
        .iter()
        .map(|(s, p)| {
            GpuSegmentQuery::new(
                [s.start.x, s.start.y, s.start.z],
                [s.end.x, s.end.y, s.end.z],
                [p.x, p.y, p.z],
            )
        })
        .collect();

    let got = kernel.query(&ctx, &gpu_queries);
    assert_eq!(got.len(), cases.len());

    for (i, (seg, point)) in cases.iter().enumerate() {
        assert_query(got[i], *seg, *point, &alloc_tag(i));
    }
}

#[test]
fn empty_batch_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuSegment::new(&ctx);
    assert!(kernel.query(&ctx, &[]).is_empty());
}

/// Small heap tag so panic messages identify the failing element.
fn alloc_tag(i: usize) -> String {
    format!("segment[{i}]")
}
