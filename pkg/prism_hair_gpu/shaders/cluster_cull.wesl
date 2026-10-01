// Nanite-style per-cluster strand culling twin: evaluates on-device the same
// GPU-driven cull verdict the CPU golden `cluster_cull_verdict` computes, one
// thread per strand cluster.
//
// A groom is grouped into spatially-local strand clusters (the Nanite-style
// coarse culling unit). Before the groom is rasterized, each cluster is tested
// against the view in a fixed order -- frustum, then back-facing, then
// occlusion -- and the first failing test decides its verdict. Every cluster is
// independent, so the batch is embarrassingly parallel: one thread owns one
// cluster, loads its bounds / mean tangent / occlusion probe, runs the exact
// closed-form tests the scalar reference runs, and writes a `visible` flag plus
// the deciding `reason`. The clustering (grid assignment) itself is a reduction
// and is NOT ported here; this kernel only mirrors the per-cluster verdict,
// which is the GPU-driven culling hot path.
//
// The three tests mirror the reference exactly:
//
//   1. frustum   -- `Frustum::intersects_aabb`: an invalid (un-grown) box is
//                   treated as outside; otherwise the box passes only when its
//                   centre is within the projected half-extent of every one of
//                   the six inward-facing planes
//                   (`signed_distance(center) >= -projected_extent(half)`).
//   2. backface  -- `is_cluster_backfacing`: the cluster is back-facing when
//                   `dot(normalize(mean_tangent), normalize(eye - center))`
//                   is at or below `-clamp(bias, 0, 1)`. A degenerate tangent
//                   or to-eye direction normalizes to zero, giving a dot of 0,
//                   which never trips a positive bias, so degenerate input is
//                   never culled.
//   3. occlusion -- `OcclusionProbe::is_occluded`: when a probe is present, the
//                   cluster is occluded iff `closest_depth > occluder_depth`.
//                   A cluster with no probe skips this phase.
//
// The reason encoding matches the host `CullReason` repr: 0 = Visible,
// 1 = Frustum, 2 = Backface, 3 = Occlusion. The deciding verdict is an integer
// (visible flag + reason), so the parity test compares it bit-exact; the test
// cases keep every decision well clear of its threshold so legal
// fused-multiply-add contraction in the plane / facing dot products cannot flip
// a verdict.
//
// It uses only the portable core-WGSL subset (`dot`, `sqrt`, `abs`, `clamp`,
// `min`, `max`, mul/add) -- no `exp`, no `pow`, no optional feature -- so the
// twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard GPU-driven cluster culling (frustum projected-radius +
// tangent back-face + Hi-Z occlusion); no Unreal Engine source or derived code.

// Matches `cluster::EPS`: the normalization degeneracy threshold. A vector
// whose squared length is at or below `EPS * EPS` normalizes to the zero vector
// instead of dividing by ~0.
const EPS: f32 = 1.0e-6;

// Cull reason encoding, matching the host `CullReason` repr.
const REASON_VISIBLE: u32 = 0u;
const REASON_FRUSTUM: u32 = 1u;
const REASON_BACKFACE: u32 = 2u;
const REASON_OCCLUSION: u32 = 3u;

// Shared view parameters. `128`-byte scalar-packed layout matching the
// `bytemuck` upload struct `Params`: six inward-facing frustum planes (each
// `normal.xyz` + signed `distance`), the eye position packed with the
// back-facing cosine bias, the cluster count and padding to a 16-byte multiple.
struct Params {
    // Six inward-facing frustum planes: `.xyz` is the unit normal, `.w` the
    // signed origin distance so `signed_distance(p) = dot(normal, p) + w`.
    planes: array<vec4<f32>, 6>,
    // `.xyz` is the eye (camera) position; `.w` is the raw back-facing bias,
    // clamped to `0..=1` on-device exactly as `is_cluster_backfacing` does.
    eye_bias: vec4<f32>,
    // Number of cluster descriptors in `clusters`.
    cluster_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One cluster descriptor. `64`-byte stride matching the `bytemuck` upload struct
// `Cluster`: the bounds min / max corners (raw, so an empty box's +inf / -inf
// reproduces the reference `is_valid` test on-device), the unit mean growth
// tangent, whether an occlusion probe is present, and the probe depths.
struct Cluster {
    bounds_min: vec3<f32>,
    pad_a: f32,
    bounds_max: vec3<f32>,
    pad_b: f32,
    mean_tangent: vec3<f32>,
    has_occlusion: u32,
    closest_depth: f32,
    occluder_depth: f32,
    pad_c: f32,
    pad_d: f32,
}

// One cull verdict. `8`-byte stride matching the host rebuild: a visible flag
// (`1` = draw, `0` = cull) and the deciding `CullReason` code.
struct Verdict {
    visible: u32,
    reason: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read_write> verdicts: array<Verdict>;

// Mirror `cluster::Aabb::is_valid`: the box has enclosed at least one point when
// `min <= max` on every axis (an un-grown +inf / -inf box reports false).
fn aabb_is_valid(lo: vec3<f32>, hi: vec3<f32>) -> bool {
    return lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z;
}

// Mirror `cluster::normalize3`: unit vector along `v`, or the zero vector when
// `v` is degenerate (squared length at or below `EPS * EPS`), so normalization
// never divides by zero or yields a NaN.
fn normalize3(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq <= EPS * EPS) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inv = 1.0 / sqrt(len_sq);
    return v * inv;
}

// Mirror `cluster::Plane::projected_extent`: the box's support radius along the
// plane axis (projection of the half-extents onto `|normal|`).
fn projected_extent(normal: vec3<f32>, half: vec3<f32>) -> f32 {
    return abs(normal.x) * half.x + abs(normal.y) * half.y + abs(normal.z) * half.z;
}

// Mirror `cluster::Frustum::intersects_aabb`: an invalid box is outside;
// otherwise the box is at least partially inside iff its centre is within the
// projected half-extent of every plane.
fn intersects_aabb(lo: vec3<f32>, hi: vec3<f32>) -> bool {
    if (!aabb_is_valid(lo, hi)) {
        return false;
    }
    let center = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = params.planes[i];
        let signed = dot(plane.xyz, center) + plane.w;
        if (signed < -projected_extent(plane.xyz, half)) {
            return false;
        }
    }
    return true;
}

// Mirror `cluster::Aabb::center`: box centre, or the origin for an invalid box.
fn aabb_center(lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    if (!aabb_is_valid(lo, hi)) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return (lo + hi) * 0.5;
}

// Mirror `cluster::is_cluster_backfacing`: the cluster is back-facing when its
// mean tangent points away from the eye by more than the `bias` cosine margin.
fn is_cluster_backfacing(mean_tangent: vec3<f32>, to_eye: vec3<f32>, bias: f32) -> bool {
    let b = clamp(bias, 0.0, 1.0);
    let facing = dot(normalize3(mean_tangent), normalize3(to_eye));
    return facing <= -b;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.cluster_count) {
        return;
    }
    let cluster = clusters[c];
    let lo = cluster.bounds_min;
    let hi = cluster.bounds_max;

    // Phase 1: frustum (rejection takes precedence over every later test).
    if (!intersects_aabb(lo, hi)) {
        verdicts[c] = Verdict(0u, REASON_FRUSTUM);
        return;
    }

    // Phase 2: back-facing, using the to-eye direction from the box centre.
    let to_eye = params.eye_bias.xyz - aabb_center(lo, hi);
    if (is_cluster_backfacing(cluster.mean_tangent, to_eye, params.eye_bias.w)) {
        verdicts[c] = Verdict(0u, REASON_BACKFACE);
        return;
    }

    // Phase 3: occlusion, only when a probe is present for this cluster.
    if (cluster.has_occlusion != 0u && cluster.closest_depth > cluster.occluder_depth) {
        verdicts[c] = Verdict(0u, REASON_OCCLUSION);
        return;
    }

    verdicts[c] = Verdict(1u, REASON_VISIBLE);
}
