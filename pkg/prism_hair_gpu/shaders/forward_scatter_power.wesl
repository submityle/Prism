// Compute twin of Prism's accumulated global forward-scatter transmittance
// (prism_render_architecture::hair::dual_scatter_sh::forward_scatter_power),
// the Zinke dual-scattering global multiplier a_f^n over n crossed strands.
// One thread per (base, exponent) query sanitises the base (non-finite -> 0),
// then raises it to the n-th power with an explicit integer multiply loop
// (not pow/exp): accum starts at 1 and is multiplied by base exactly n times.
//
// The exponent is a per-thread u32 read from a storage array, so the loop trip
// count is data-dependent and varies between invocations in the same workgroup
// (a divergent loop); n = 0 yields exactly 1.0 without entering the loop.
//
// The kernel is a dependent chain of IEEE-754 single-precision multiplies with
// no add to fuse and no compile-time-known trip count to fold into pow, so the
// GPU reproduces the scalar reference's rounding bit-for-bit. The parity test
// therefore compares raw bit patterns rather than a tolerance. Only integer
// arithmetic and f32 multiply are used, so it runs unmodified on Metal, Vulkan
// and DX12 with no optional device feature.

struct Params {
    element_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query: the forward-scatter factor base and the crossed-strand count n.
struct PowerQuery {
    base: f32,
    exponent: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-thread (base, exponent) queries; length element_count.
@group(0) @binding(1) var<storage, read> queries: array<PowerQuery>;
// One accumulated a_f^n per query, in input order.
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const MAX_FINITE_F32: f32 = 3.40282347e+38;

// Mirror the golden `is_finite()` test without a float equality on a literal.
fn is_finite_f32(x: f32) -> bool {
    return x == x && x >= -MAX_FINITE_F32 && x <= MAX_FINITE_F32;
}

// Golden `sanitize`: replace a non-finite value with 0, leaving finite intact.
fn sanitize(x: f32) -> f32 {
    if is_finite_f32(x) {
        return x;
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.element_count {
        return;
    }
    let base = sanitize(queries[idx].base);
    let n = queries[idx].exponent;

    // Explicit integer multiply loop (mirrors the golden's `while i < n`): a
    // dependent product chain with no fma opportunity, so it stays bit-exact.
    var accum: f32 = 1.0;
    var i: u32 = 0u;
    loop {
        if i >= n {
            break;
        }
        accum = accum * base;
        i = i + 1u;
    }

    out[idx] = accum;
}
