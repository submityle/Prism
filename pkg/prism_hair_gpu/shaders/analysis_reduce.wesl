//! `wgpu` compute twin of Prism's groom-global analysis reduction
//! (`prism_render_architecture::hair::analysis_readback::reduce_lane`).
//!
//! The hair analysis passes (`hair_guide_metrics`, `hair_binding_metrics`,
//! `hair_motion_energy`) each emit a per-element output buffer — a `vec4<f32>`
//! of arc length / curvature / radius (or a scalar packed into `.x`). A `CPU`
//! decision then needs one *groom-global* scalar folded from a chosen lane of
//! that buffer: the `Max` over root radius / length / curvature that normalizes
//! the density/decimation `LOD`, and the `Sum` over per-particle motion energy
//! that gates the hysteretic sleep. This kernel is the on-device twin of that
//! fold — it reduces one `lane` of every element with one operator (`Max` or
//! `Sum`) into a single scalar, so a passing real-device parity test is direct
//! evidence the ported reduction reads the same groom-global value as the
//! reference, not merely that its shader compiles.
//!
//! # Reduction shape
//!
//! Unlike every other kernel in this crate — each of which is an
//! embarrassingly-parallel one-thread-per-output map or gather — a reduction
//! folds *many* inputs into *one* output, so it is the crate's first
//! shared-memory tree reduction. A single `256`-wide workgroup cooperatively
//! reduces the whole buffer: each invocation grid-strides across the element
//! array accumulating a private partial from the operator identity (`0`), the
//! partials are staged into workgroup memory, and a logarithmic tree fold
//! collapses them to lane `0`, which writes the single output. There is exactly
//! one output and one workgroup, so there is no cross-workgroup race.
//!
//! # Bit-faithfulness
//!
//! The fold mirrors the golden operator: `Max` keeps the larger operand from
//! the identity `0` (so an all-negative lane reduces to `0`, exactly as the
//! golden's `if x > acc` update does), and `Sum` accumulates. `Max` selects an
//! existing element value with no arithmetic, so it is bit-exact regardless of
//! fold order. `Sum` is floating-point addition, which is commutative but not
//! associative, so the tree's pairwise order differs from the golden's
//! left-to-right walk by a few low-mantissa `ULP`; the parity test asserts a
//! per-value tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) for `Sum` while
//! `Max` is asserted exactly.
//!
//! # Portability
//!
//! The kernel uses only comparisons, addition and workgroup shared memory in
//! the portable core-`WGSL` subset — no `exp`, `pow`, atomics or optional
//! device feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! Provenance: standard single-workgroup shared-memory tree reduction plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

/// Reduction parameters. `element_count` bounds the grid-stride load; `lane`
/// selects the `vec4` component (`0`=`.x`, `1`=`.y`, `2`=`.z`, `3`=`.w`); `op`
/// selects the fold (`0`=`Max`, `1`=`Sum`). Scalar-packed and padded to `16`
/// bytes so the uniform block is a multiple of 16.
struct HairReduceParams {
    element_count: u32,
    lane: u32,
    op: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: HairReduceParams;

// Flat per-element analysis outputs; length `element_count`.
@group(0) @binding(1) var<storage, read> elements: array<vec4<f32>>;

// Single groom-global reduced scalar; length 1.
@group(0) @binding(2) var<storage, read_write> out_value: array<f32>;

const WG: u32 = 256u;

// Per-workgroup partials for the tree reduction.
var<workgroup> partials: array<f32, 256>;

// Selects the reduced lane component of one element.
fn lane_value(e: vec4<f32>, lane: u32) -> f32 {
    if lane == 0u {
        return e.x;
    }
    if lane == 1u {
        return e.y;
    }
    if lane == 2u {
        return e.z;
    }
    return e.w;
}

// Folds `b` into `a` with the selected operator, matching the golden `apply`:
// `Sum` adds; `Max` keeps the larger (identity `0`, ties keep `a`).
fn fold(a: f32, b: f32, op: u32) -> f32 {
    if op == 1u {
        return a + b;
    }
    if b > a {
        return b;
    }
    return a;
}

@compute @workgroup_size(256)
fn evaluate(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let op = params.op;
    let lane = params.lane;
    let count = params.element_count;

    // Grid-stride load: each invocation folds every WG-th element into a
    // private partial, starting from the operator identity (`0`).
    var local = 0.0;
    var i = tid;
    loop {
        if i >= count {
            break;
        }
        local = fold(local, lane_value(elements[i], lane), op);
        i = i + WG;
    }
    partials[tid] = local;
    workgroupBarrier();

    // Logarithmic tree fold: halve the active range each step until lane 0
    // holds the whole-workgroup reduction.
    var stride = WG / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if tid < stride {
            partials[tid] = fold(partials[tid], partials[tid + stride], op);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        out_value[0] = partials[0];
    }
}
