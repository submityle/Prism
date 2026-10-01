// Stratified per-lobe sample-weight twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::scatter_lod::stratified_weights`. One thread
// per `(pdf, total)` tuple first reproduces the largest-remainder (Hamilton)
// allocation of `total` samples across the three Marschner lobes (see
// `stratified_allocation.wesl`), then emits each lobe's Monte-Carlo sample
// weight `sanitize_nonneg(pdf_i) * total / count_i`, i.e. the reciprocal of the
// per-sample probability so the stratified estimator stays unbiased. A lobe that
// received no samples (`count_i == 0`) or a zero budget yields weight `0`.
//
// Each pdf component is sanitised in-shader bit-faithfully to the golden
// (`sanitize_nonneg`: negative or non-finite collapses to 0). The allocation is
// integer-exact; only the final `prob * total / count` divide can round a few
// `ULP` differently on the device, so the parity test compares the weights with
// a tolerance.
//
// The kernel uses only compares, `max`, multiply, `floor`, a single divide and
// small bounded loops (at most three leftover/overflow fix-ups since there are
// three lobes) in the portable core-WGSL subset - no `exp`, `pow` or optional
// device feature - so the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: largest-remainder (Hamilton) apportionment of a Marschner
// three-lobe pdf and its unbiased stratified weights plus wgpu compute dispatch;
// no Unreal Engine source or derived code.

const LOBE_COUNT: u32 = 3u;

struct Params {
    // Number of (pdf, total) tuples in `inputs`; three weights per tuple out.
    elem_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per element, stride 4 f32: pdf.r, pdf.tt, pdf.trt, total (an exact integer).
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
// Per element, stride 3 f32: the per-lobe sample weight (R, TT, TRT).
@group(0) @binding(2) var<storage, read_write> out_weights: array<f32>;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Finite means not NaN (v == v) and not an infinity (|v| below +inf).
fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < inf();
}

// Clamp to a finite, non-negative value (the golden's `sanitize_nonneg`):
// negative or non-finite values collapse to 0.
fn sanitize_nonneg(v: f32) -> f32 {
    return select(0.0, max(v, 0.0), finite_scalar(v));
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.elem_count) {
        return;
    }
    let si = 4u * i;
    let oi = LOBE_COUNT * i;

    let total: u32 = u32(inputs[si + 3u]);
    var probs = array<f32, 3>(
        sanitize_nonneg(inputs[si]),
        sanitize_nonneg(inputs[si + 1u]),
        sanitize_nonneg(inputs[si + 2u]),
    );

    if (total == 0u) {
        out_weights[oi] = 0.0;
        out_weights[oi + 1u] = 0.0;
        out_weights[oi + 2u] = 0.0;
        return;
    }

    let total_f = f32(total);
    var counts = array<u32, 3>(0u, 0u, 0u);
    var quotas = array<f32, 3>(0.0, 0.0, 0.0);
    var assigned: u32 = 0u;
    for (var k: u32 = 0u; k < LOBE_COUNT; k = k + 1u) {
        let q = probs[k] * total_f;
        let base = floor(q);
        quotas[k] = q - base;
        let base_count = u32(max(base, 0.0));
        counts[k] = base_count;
        assigned = assigned + base_count;
    }

    if (assigned > total) {
        // Trim from the smallest-remainder lobes first (reverse of top-up).
        var over = assigned - total;
        for (var step: u32 = 0u; step < LOBE_COUNT; step = step + 1u) {
            if (over == 0u) {
                break;
            }
            var victim: u32 = 0xffffffffu;
            var worst: f32 = inf();
            for (var j: u32 = 0u; j < LOBE_COUNT; j = j + 1u) {
                if (counts[j] > 0u && quotas[j] < worst) {
                    worst = quotas[j];
                    victim = j;
                }
            }
            if (victim == 0xffffffffu) {
                break;
            }
            counts[victim] = counts[victim] - 1u;
            over = over - 1u;
        }
    } else {
        var leftover = total - assigned;
        for (var step: u32 = 0u; step < LOBE_COUNT; step = step + 1u) {
            if (leftover == 0u) {
                break;
            }
            var best: u32 = 0xffffffffu;
            var best_rem: f32 = -1.0;
            for (var j: u32 = 0u; j < LOBE_COUNT; j = j + 1u) {
                if (quotas[j] > best_rem) {
                    best_rem = quotas[j];
                    best = j;
                }
            }
            if (best == 0xffffffffu) {
                break;
            }
            counts[best] = counts[best] + 1u;
            quotas[best] = quotas[best] - 1.0;
            leftover = leftover - 1u;
        }
    }

    // Per-lobe unbiased weight: pdf_i * total / count_i, or 0 for a dead lobe.
    for (var k: u32 = 0u; k < LOBE_COUNT; k = k + 1u) {
        var w: f32 = 0.0;
        if (counts[k] > 0u) {
            w = probs[k] * total_f / f32(counts[k]);
        }
        out_weights[oi + k] = w;
    }
}
