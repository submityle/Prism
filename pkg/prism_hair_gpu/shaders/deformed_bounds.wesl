// Hair sim -> render-graph scene-bounds reduction twin.
//
// Device mirror of the `CPU` golden
// `prism_render_architecture::hair::gpu_scene_handoff::deformed_bounds`. Once a
// frame's solve + resolve has produced a fresh set of world-space deformed
// render points, the groom's instance in the shared `gpu_scene` must be told
// its world-space bounds moved, so the incremental uploader re-publishes the
// bounds row. This kernel folds the groom's deformed point cloud into that one
// world-space axis-aligned box: the per-axis `min` / `max` over every finite
// point, then the `center`, `half_extents` and bounding `radius` (half-diagonal)
// the `SceneBounds` row carries.
//
// # Reduction shape
//
// Like `analysis_reduce`, this is a many-inputs-to-one-output fold rather than
// an embarrassingly-parallel one-thread-per-output map: a single `256`-wide
// workgroup cooperatively reduces the whole point cloud. Each invocation
// grid-strides across the points accumulating a private per-axis `min` / `max`
// (plus a `seen` flag so an invocation that folded no finite point contributes
// nothing), the partials are staged into workgroup memory, and a logarithmic
// tree fold collapses them to lane `0`, which derives and writes the box. There
// is exactly one output and one workgroup, so there is no cross-workgroup race.
//
// # Correctness model
//
// A point with any non-finite component is skipped, bit-faithfully to the
// golden, so a stray `NaN` / `+/-inf` from a diverged solve cannot poison the
// box. The `min` / `max` fold selects existing finite coordinates with no
// arithmetic, so it is bit-exact regardless of fold order; `center`,
// `half_extents` and `radius` are the same `(min +/- max) * 0.5` and
// `sqrt(dot(h, h))` the golden computes. If no finite point survives (empty
// groom or all-degenerate input) lane `0` emits a zero-extent box at the origin,
// exactly as the reference does.
//
// # Portability
//
// The kernel uses only comparisons, add / multiply, `sqrt` and workgroup shared
// memory in the portable core-`WGSL` subset -- no `exp`, `pow`, atomics or
// optional device feature -- so the twin runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard single-workgroup shared-memory tree reduction plus
// `wgpu` compute dispatch; no Unreal Engine source or derived code.

// `point_count` bounds the grid-stride load; scalar-packed and padded to `16`
// bytes so the uniform block is a multiple of 16.
struct Params {
    point_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flat deformed render points: `3 * point_count` f32 (`x, y, z` per point).
@group(0) @binding(1) var<storage, read> points: array<f32>;
// The folded `SceneBounds` row: `[center.xyz, radius, half_extents.xyz, pad]`.
@group(0) @binding(2) var<storage, read_write> out_bounds: array<f32>;

const WG: u32 = 256u;

// Largest finite f32; `v <= MAX_FINITE_F32 && v >= -MAX_FINITE_F32` rejects
// `+/-inf` and `v == v` rejects `NaN`, so the guard reproduces the golden's
// `f32::is_finite()` exactly.
const MAX_FINITE_F32: f32 = 3.40282347e+38;

// Per-axis workgroup partials for the tree reduction, plus a `seen` flag marking
// invocations / sub-ranges that folded at least one finite point.
var<workgroup> wmin_x: array<f32, 256>;
var<workgroup> wmin_y: array<f32, 256>;
var<workgroup> wmin_z: array<f32, 256>;
var<workgroup> wmax_x: array<f32, 256>;
var<workgroup> wmax_y: array<f32, 256>;
var<workgroup> wmax_z: array<f32, 256>;
var<workgroup> wseen: array<u32, 256>;

// Mirrors the golden's `f32::is_finite()`: rejects `NaN` (`v == v` is false) and
// `+/-inf` (outside the finite magnitude range).
fn is_finite(v: f32) -> bool {
    return v == v && v <= MAX_FINITE_F32 && v >= -MAX_FINITE_F32;
}

@compute @workgroup_size(256)
fn evaluate(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let count = params.point_count;

    // Grid-stride load: each invocation folds every WG-th point into a private
    // per-axis box, skipping any point with a non-finite component.
    var seen = false;
    var mnx = 0.0;
    var mny = 0.0;
    var mnz = 0.0;
    var mxx = 0.0;
    var mxy = 0.0;
    var mxz = 0.0;
    var i = tid;
    loop {
        if i >= count {
            break;
        }
        let base = i * 3u;
        let x = points[base];
        let y = points[base + 1u];
        let z = points[base + 2u];
        if is_finite(x) && is_finite(y) && is_finite(z) {
            if seen {
                mnx = min(mnx, x);
                mny = min(mny, y);
                mnz = min(mnz, z);
                mxx = max(mxx, x);
                mxy = max(mxy, y);
                mxz = max(mxz, z);
            } else {
                mnx = x;
                mny = y;
                mnz = z;
                mxx = x;
                mxy = y;
                mxz = z;
                seen = true;
            }
        }
        i = i + WG;
    }
    wmin_x[tid] = mnx;
    wmin_y[tid] = mny;
    wmin_z[tid] = mnz;
    wmax_x[tid] = mxx;
    wmax_y[tid] = mxy;
    wmax_z[tid] = mxz;
    wseen[tid] = select(0u, 1u, seen);
    workgroupBarrier();

    // Logarithmic tree fold: halve the active range each step until lane 0 holds
    // the whole-workgroup box. A `seen` sub-range absorbs the other only when
    // both saw a finite point; otherwise the seen side (if any) survives.
    var stride = WG / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if tid < stride {
            let o = tid + stride;
            let a_seen = wseen[tid] == 1u;
            let b_seen = wseen[o] == 1u;
            if a_seen && b_seen {
                wmin_x[tid] = min(wmin_x[tid], wmin_x[o]);
                wmin_y[tid] = min(wmin_y[tid], wmin_y[o]);
                wmin_z[tid] = min(wmin_z[tid], wmin_z[o]);
                wmax_x[tid] = max(wmax_x[tid], wmax_x[o]);
                wmax_y[tid] = max(wmax_y[tid], wmax_y[o]);
                wmax_z[tid] = max(wmax_z[tid], wmax_z[o]);
            } else if b_seen {
                wmin_x[tid] = wmin_x[o];
                wmin_y[tid] = wmin_y[o];
                wmin_z[tid] = wmin_z[o];
                wmax_x[tid] = wmax_x[o];
                wmax_y[tid] = wmax_y[o];
                wmax_z[tid] = wmax_z[o];
                wseen[tid] = 1u;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        if wseen[0] == 1u {
            let cx = (wmin_x[0] + wmax_x[0]) * 0.5;
            let cy = (wmin_y[0] + wmax_y[0]) * 0.5;
            let cz = (wmin_z[0] + wmax_z[0]) * 0.5;
            let hx = (wmax_x[0] - wmin_x[0]) * 0.5;
            let hy = (wmax_y[0] - wmin_y[0]) * 0.5;
            let hz = (wmax_z[0] - wmin_z[0]) * 0.5;
            let r = sqrt(hx * hx + hy * hy + hz * hz);
            out_bounds[0] = cx;
            out_bounds[1] = cy;
            out_bounds[2] = cz;
            out_bounds[3] = r;
            out_bounds[4] = hx;
            out_bounds[5] = hy;
            out_bounds[6] = hz;
            out_bounds[7] = 0.0;
        } else {
            out_bounds[0] = 0.0;
            out_bounds[1] = 0.0;
            out_bounds[2] = 0.0;
            out_bounds[3] = 0.0;
            out_bounds[4] = 0.0;
            out_bounds[5] = 0.0;
            out_bounds[6] = 0.0;
            out_bounds[7] = 0.0;
        }
    }
}
