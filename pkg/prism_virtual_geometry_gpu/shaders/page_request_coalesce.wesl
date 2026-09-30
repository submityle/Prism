// wgpu compute twin of the virtual-geometry per-frame page-request coalescer.
//
// The CPU golden `plan_frame` walks a frame's drawn clusters and, for each, adds
// one reference to that cluster's page at its screen-coverage priority via
// `PageRequestBatch::record`, which keeps the highest priority any referencing
// cluster reported (`coverage_priority` = extent^2 / distance^2). Many clusters
// share one page, so the essential per-frame compute is: (1) evaluate
// `coverage_priority` per cluster, (2) reduce it to the max over all clusters
// referencing the same page. This kernel runs one thread per cluster and folds
// each cluster's priority into its page's dense slot with a single `atomicMax`.
//
// The mutable residency bookkeeping (assigning each unique page a dense slot and
// flushing the coalesced priorities into the residency table) stays host-side,
// exactly as the CPU golden leaves `table.request` outside the pure decision;
// the host passes the slot index each cluster maps to and reads the per-slot
// maxima back.
//
// `coverage_priority` is non-negative (extent^2 >= 0, distance_sq >= EPSILON > 0),
// and for non-negative f32 the IEEE-754 bit pattern is monotonic in value, so
// `atomicMax` on `bitcast<u32>(priority)` selects the true maximum priority. The
// slot buffer is pre-cleared to 0 (== +0.0) by the host, the identity for max
// over non-negative priorities.
//
// The kernel uses only the portable core-WGSL subset (a storage buffer of
// `atomic<u32>` plus `atomicMax`), so it needs no optional device feature and
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard screen-coverage page-priority coalescing plus wgpu
// compute dispatch; no Unreal Engine source or derived code.

// Matches the reference distance clamp `distance_sq.max(f32::EPSILON)`.
const F32_EPSILON: f32 = 1.1920929e-7;

// Uniform parameters. 32-byte layout: view origin (3 f32) + focal length, then
// the cluster and slot counts and two pad words.
struct Params {
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    focal_length_pixels: f32,
    cluster_count: u32,
    slot_count: u32,
    pad0: u32,
    pad1: u32,
}

// One drawn cluster's page-request input. 32-byte stride: bounds center (3 f32)
// + radius, then the dense page slot this cluster references and three pad words.
struct Cluster {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
    slot: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read_write> priorities: array<atomic<u32>>;

// Mirrors `hierarchy::coverage_priority`:
//   distance_sq = max(|center - origin|^2, EPSILON)
//   extent      = max(radius, 0) * focal_length_pixels
//   priority    = extent^2 / distance_sq
fn coverage_priority(c: Cluster) -> f32 {
    let dx = c.center_x - params.origin_x;
    let dy = c.center_y - params.origin_y;
    let dz = c.center_z - params.origin_z;
    let distance_sq = max(dx * dx + dy * dy + dz * dz, F32_EPSILON);
    let extent = max(c.radius, 0.0) * params.focal_length_pixels;
    return extent * extent / distance_sq;
}

@compute @workgroup_size(64)
fn coalesce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cluster_count) {
        return;
    }
    let c = clusters[i];
    // Guard against a malformed slot index so a bad input cannot scribble out
    // of range; well-formed input always indexes a valid slot.
    if (c.slot >= params.slot_count) {
        return;
    }
    let priority = coverage_priority(c);
    // Non-negative f32 bit patterns order like the values, so atomicMax on the
    // bits keeps the highest priority any cluster referencing this page reported.
    atomicMax(&priorities[c.slot], bitcast<u32>(priority));
}
