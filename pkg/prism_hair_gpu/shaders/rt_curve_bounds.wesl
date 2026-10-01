// Linear-Swept-Spheres (LSS) batch bounding-box union twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::rt_curve::segments_aabb`. Where
// `rt_curve_aabb.wesl` emits one conservative box per segment (an
// embarrassingly-parallel map), this kernel folds a whole batch of LSS
// segments into the single conservative axis-aligned bounding box (AABB) that
// encloses every segment - the groom-global box a downstream ray-tracing BLAS
// build registers for a strand (or a whole strand set) at once.
//
// # Reduction shape
//
// Like `analysis_reduce.wesl`, this is a shared-memory tree reduction, not a
// one-thread-per-output map: a single 256-wide workgroup cooperatively folds
// the whole segment array. Each invocation grid-strides across the segments
// accumulating a private partial box from the union identity (the empty box
// `min = +inf`, `max = -inf`), the partials are staged into workgroup memory,
// and a logarithmic tree fold unions them down to lane 0, which writes the one
// output box. There is exactly one output and one workgroup, so there is no
// cross-workgroup race.
//
// # Bit-faithfulness
//
// The union of a set of boxes is a per-axis `min` of the mins and `max` of the
// maxes. `min`/`max` select an existing operand with no arithmetic and are both
// commutative *and* associative, so the tree's pairwise fold order yields the
// exact same bits as the golden's left-to-right `Aabb::union` walk - the twin
// is *bit-exact* with the scalar reference (the parity test compares raw bit
// patterns, not a tolerance). Each endpoint box is centre +/- radius (single
// add/sub, no multiply to fuse), and each segment is sanitised in-shader
// bit-faithfully to the golden's `LssSegment::sanitized` (non-finite position
// component -> 0, negative or non-finite radius -> 0), so a stray NaN can never
// poison the box.
//
// # Portability
//
// The kernel uses only comparisons, `min`/`max`, `select`, single add/sub and
// workgroup shared memory in the portable core-WGSL subset - no `sqrt`, `exp`,
// `pow`, atomics or optional device feature - so the twin runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: hardware ray-traced curve / LSS strand primitive BLAS contract
// (RTX DXR / OptiX LSS) plus a standard single-workgroup shared-memory tree
// reduction and wgpu compute dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of LSS segments in `segments`.
    segment_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per segment, stride 8 f32: a.x, a.y, a.z, radius_a, b.x, b.y, b.z, radius_b.
@group(0) @binding(1) var<storage, read> segments: array<f32>;
// Single groom-global box, 6 f32: min.x, min.y, min.z, max.x, max.y, max.z.
@group(0) @binding(2) var<storage, read_write> out_bounds: array<f32>;

const WG: u32 = 256u;

// Per-workgroup partial boxes for the tree reduction.
var<workgroup> partial_min: array<vec3<f32>, 256>;
var<workgroup> partial_max: array<vec3<f32>, 256>;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Finite means not NaN (v == v) and not an infinity (|v| below +inf).
fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < inf();
}

// Force a non-finite coordinate to 0 (the golden's `sanitize_coord`).
fn sanitize_coord(v: f32) -> f32 {
    return select(0.0, v, finite_scalar(v));
}

fn sanitize_point(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(sanitize_coord(p.x), sanitize_coord(p.y), sanitize_coord(p.z));
}

// Clamp a radius to a finite, non-negative value (the golden's
// `sanitize_radius`): negative or non-finite radii collapse to 0.
fn sanitize_radius(r: f32) -> f32 {
    return select(0.0, r, finite_scalar(r) && r >= 0.0);
}

@compute @workgroup_size(256)
fn evaluate(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let count = params.segment_count;

    // Grid-stride load: each invocation unions every WG-th segment's box into a
    // private partial, starting from the empty box (union identity).
    var lo = vec3<f32>(inf(), inf(), inf());
    var hi = vec3<f32>(-inf(), -inf(), -inf());
    var i = tid;
    loop {
        if i >= count {
            break;
        }
        let si = 8u * i;
        // Sanitise the raw segment exactly as the golden `LssSegment::sanitized`.
        let a = sanitize_point(vec3<f32>(segments[si], segments[si + 1u], segments[si + 2u]));
        let ra = sanitize_radius(segments[si + 3u]);
        let b = sanitize_point(vec3<f32>(segments[si + 4u], segments[si + 5u], segments[si + 6u]));
        let rb = sanitize_radius(segments[si + 7u]);

        // Endpoint boxes: centre +/- radius on each axis (no sqrt, no multiply).
        let box_a_min = a - vec3<f32>(ra, ra, ra);
        let box_a_max = a + vec3<f32>(ra, ra, ra);
        let box_b_min = b - vec3<f32>(rb, rb, rb);
        let box_b_max = b + vec3<f32>(rb, rb, rb);

        // Segment box, then fold into the running union.
        lo = min(lo, min(box_a_min, box_b_min));
        hi = max(hi, max(box_a_max, box_b_max));
        i = i + WG;
    }
    partial_min[tid] = lo;
    partial_max[tid] = hi;
    workgroupBarrier();

    // Logarithmic tree fold: halve the active range each step until lane 0
    // holds the whole-batch union.
    var stride = WG / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if tid < stride {
            partial_min[tid] = min(partial_min[tid], partial_min[tid + stride]);
            partial_max[tid] = max(partial_max[tid], partial_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        let m = partial_min[0];
        let x = partial_max[0];
        out_bounds[0] = m.x;
        out_bounds[1] = m.y;
        out_bounds[2] = m.z;
        out_bounds[3] = x.x;
        out_bounds[4] = x.y;
        out_bounds[5] = x.z;
    }
}
