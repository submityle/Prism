// Raster-bin partition twin: fans a selected cluster cut out into the four
// GPU raster-path buckets on-device, mirroring the CPU golden `bin_cut`.
//
// Once the hierarchy has resolved which cluster nodes to draw this frame, each
// drawn cluster is routed to one of four rasterization paths (mesh shader,
// compute software, hardware indirect, plain fallback) so the backend can emit
// one indirect batch per path. The CPU golden `bin_cut` (in
// `prism_render_architecture::virtual_geometry`) owns that partition; this
// kernel is the on-device twin that produces a bit-identical, order-preserving
// partition as a permutation of the input cut indices plus a per-bucket count.
//
// The partition is a *stable* multi-bucket split: within each bucket the cut's
// original order is preserved, and a cut element whose node index falls outside
// the statistics slice is skipped rather than crashing. To stay faithful and
// deterministic it runs as three passes over shared storage:
//
//   1. `classify` (one thread per cut element): resolves each element's bucket
//      in {0,1,2,3} via the same integer branching + single `<=` compare as
//      the raster-path selector, or `SKIP` when the node index is out of range.
//   2. `scan` (one serial invocation): walks the cut in order computing each
//      element's within-bucket rank and the four bucket counts, exactly as the
//      reference's sequential push does. A single-threaded scan matches the
//      reference's inherently sequential ordering, so the result is
//      deterministic and bit-exact.
//   3. `scatter` (one thread per cut element): places each non-skipped element
//      at `base[bucket] + rank` in the output permutation, where `base` is the
//      fixed-order prefix sum of the bucket counts.
//
// Bucket indices match `GeometryRasterPath as u32` in declaration order:
//   0 = MeshShader, 1 = ComputeSoftware, 2 = IndirectHardware, 3 = FallbackMesh.
//
// The only floating-point work is a `max(threshold, 0.0)` clamp and a single
// `<=` compare on identical operands, so every bucket assignment is bit-exact
// against the reference regardless of fused-multiply-add contraction, and the
// emitted permutation is a discrete index array with no tolerance. It uses only
// the portable core-WGSL subset - no optional feature - so this twin runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard stable multi-bucket partition and cluster raster-path
// selection heuristic; no Unreal Engine source or derived code.

struct Params {
    // Number of cut elements in `node_index` and outputs.
    count: u32,
    // Number of valid entries in `stats`; a node index at or above this is
    // out of range and its cut element is skipped.
    stats_count: u32,
    // Backend capability bits: bit 0 = mesh_shader, bit 1 = hardware_indirect.
    capability_bits: u32,
    // Software-raster pixel threshold; a cluster whose largest triangle is at
    // or below `max(threshold, 0.0)` takes the compute software bucket.
    threshold: f32,
}

// One cluster's raster statistics. `8`-byte stride matching the `bytemuck`
// upload struct.
struct ClusterStats {
    // Largest projected triangle area in the cluster, in pixels squared.
    max_triangle_pixels: f32,
    // Number of triangles in the cluster.
    triangle_count: u32,
}

// Bucket index constants, matching `GeometryRasterPath as u32`.
const BUCKET_MESH_SHADER: u32 = 0u;
const BUCKET_COMPUTE_SOFTWARE: u32 = 1u;
const BUCKET_INDIRECT_HARDWARE: u32 = 2u;
const BUCKET_FALLBACK_MESH: u32 = 3u;

// Sentinel bucket for a cut element whose node index is out of range.
const SKIP: u32 = 0xffffffffu;

// Capability bit masks.
const CAP_MESH_SHADER: u32 = 1u;
const CAP_HARDWARE_INDIRECT: u32 = 2u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stats: array<ClusterStats>;
@group(0) @binding(2) var<storage, read> node_index: array<u32>;
@group(0) @binding(3) var<storage, read_write> bucket: array<u32>;
@group(0) @binding(4) var<storage, read_write> rank: array<u32>;
@group(0) @binding(5) var<storage, read_write> counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_order: array<u32>;

// Resolves the raster-path bucket for a cluster, mirroring the CPU golden
// `select_raster_path`.
fn select_bucket(s: ClusterStats) -> u32 {
    // Empty clusters resolve to the inert fallback default.
    if (s.triangle_count == 0u) {
        return BUCKET_FALLBACK_MESH;
    }
    // Tiny clusters take the compute software bucket. The threshold is clamped
    // to `>= 0` exactly as the reference's `software_pixel_threshold.max(0.0)`.
    let clamped_threshold = max(params.threshold, 0.0);
    if (s.max_triangle_pixels <= clamped_threshold) {
        return BUCKET_COMPUTE_SOFTWARE;
    }
    // Larger clusters take the best available hardware bucket.
    let mesh_shader = (params.capability_bits & CAP_MESH_SHADER) != 0u;
    let hardware_indirect = (params.capability_bits & CAP_HARDWARE_INDIRECT) != 0u;
    if (mesh_shader) {
        return BUCKET_MESH_SHADER;
    } else if (hardware_indirect) {
        return BUCKET_INDIRECT_HARDWARE;
    }
    return BUCKET_FALLBACK_MESH;
}

// Pass 1: one thread per cut element resolves its bucket (or SKIP).
@compute @workgroup_size(64, 1, 1)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let n = node_index[i];
    if (n >= params.stats_count) {
        bucket[i] = SKIP;
        return;
    }
    bucket[i] = select_bucket(stats[n]);
}

// Pass 2: a single serial invocation walks the cut in order, computing each
// element's within-bucket rank and the four bucket counts. This mirrors the
// reference's sequential `push`, so per-bucket order is preserved exactly.
@compute @workgroup_size(1, 1, 1)
fn scan() {
    var c0: u32 = 0u;
    var c1: u32 = 0u;
    var c2: u32 = 0u;
    var c3: u32 = 0u;
    for (var i: u32 = 0u; i < params.count; i = i + 1u) {
        let b = bucket[i];
        if (b == BUCKET_MESH_SHADER) {
            rank[i] = c0;
            c0 = c0 + 1u;
        } else if (b == BUCKET_COMPUTE_SOFTWARE) {
            rank[i] = c1;
            c1 = c1 + 1u;
        } else if (b == BUCKET_INDIRECT_HARDWARE) {
            rank[i] = c2;
            c2 = c2 + 1u;
        } else if (b == BUCKET_FALLBACK_MESH) {
            rank[i] = c3;
            c3 = c3 + 1u;
        } else {
            // SKIP: rank is unused; leave it defined for determinism.
            rank[i] = 0u;
        }
    }
    counts[0] = c0;
    counts[1] = c1;
    counts[2] = c2;
    counts[3] = c3;
}

// Pass 3: one thread per cut element scatters each non-skipped element to
// `base[bucket] + rank` in the output permutation, where `base` is the
// fixed-order prefix sum of the bucket counts.
@compute @workgroup_size(64, 1, 1)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let b = bucket[i];
    if (b == SKIP) {
        return;
    }
    // Prefix sum over the fixed bucket order 0,1,2,3.
    var base: u32 = 0u;
    if (b > BUCKET_MESH_SHADER) {
        base = base + counts[0];
    }
    if (b > BUCKET_COMPUTE_SOFTWARE) {
        base = base + counts[1];
    }
    if (b > BUCKET_INDIRECT_HARDWARE) {
        base = base + counts[2];
    }
    out_order[base + rank[i]] = i;
}
