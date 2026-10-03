// Per-pixel history-rejection (disocclusion) classifier.
//
// One invocation classifies one reprojected history sample against the current
// surface, mirroring the CPU golden
// `prism_render_architecture::motion::disocclusion::classify` and its helpers
// `depth_consistency` / `normal_consistency` exactly:
//   * surface-identity mismatch (when required) hard-rejects and zeroes the
//     confidence,
//   * depth continuity grades 1 -> 0 linearly across the relative tolerance,
//   * normal continuity grades 1 -> 0 linearly from the cosine threshold to 1,
//   * the combined confidence is the minimum of the three sub-confidences, and
//     the sample is accepted when no hard reason fired and the confidence
//     reaches the accept threshold.
//
// The host sanitises the thresholds (DisocclusionParams::new) so `tolerance`
// and `cos_threshold` are never NaN here; the max/clamp flooring is replicated
// so the arithmetic matches the golden bit-for-bit on finite inputs. clamp01
// resolves NaN via an IEEE-754 bit test because Metal compiles WGSL with
// fast-math (where `x != x` folds to false).
//
// The only operation that can diverge from the scalar golden is the normal dot
// product, which a fast-math backend may contract to an FMA; the confidence is
// therefore compared within a tight tolerance while the discrete verdict
// (accepted + rejection reasons) is compared exactly.
//
// Provenance: standard TAA/temporal-upsampler disocclusion heuristic (depth +
// normal + surface-id continuity). Classical, data-oblivious arithmetic; no
// neural, learned, or data-driven components. No Unreal Engine source or
// derived code.

const EPS: f32 = 1e-6;

const SURFACE_MISMATCH: u32 = 1u;
const DEPTH_DISCONTINUITY: u32 = 2u;
const NORMAL_DISCONTINUITY: u32 = 4u;

struct Params {
    count: u32,
    // 0 = ignore surface id, nonzero = mismatched ids hard-reject.
    require_surface_match: u32,
    depth_relative_tolerance: f32,
    normal_cos_threshold: f32,
    accept_threshold: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// A surface sample: unit normal (nx, ny, nz), stored depth, and a 64-bit
// surface id split into (lo, hi) 32-bit halves.
struct Surface {
    nx: f32,
    ny: f32,
    nz: f32,
    depth: f32,
    sid_lo: u32,
    sid_hi: u32,
    pad0: u32,
    pad1: u32,
};

struct Verdict {
    accepted: u32,
    confidence: f32,
    reasons: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> current: array<Surface>;
@group(0) @binding(2) var<storage, read> history: array<Surface>;
@group(0) @binding(3) var<storage, read_write> out_verdict: array<Verdict>;

// IEEE-754 NaN test by bit pattern (fast-math safe; `x != x` folds to false
// under Metal's fast-math).
fn is_nan_bits(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

// Clamp to [0, 1], resolving NaN (and negatives) to 0, matching golden clamp01.
fn clamp01(x: f32) -> f32 {
    if (is_nan_bits(x) || x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
}

// Depth-continuity confidence: 1 at a perfect match, falling linearly to 0 as
// the relative gap reaches `tolerance`, 0 beyond. `tolerance` is pre-sanitised
// non-NaN by the host; the `max(EPS)` floor mirrors the golden exactly.
fn depth_consistency(current_depth: f32, history_depth: f32, tolerance: f32) -> f32 {
    let tol = max(tolerance, EPS);
    let denom = max(max(abs(current_depth), abs(history_depth)), EPS);
    let relative = abs(current_depth - history_depth) / denom;
    return clamp01(1.0 - relative / tol);
}

// Normal-continuity confidence: 1 when aligned, 0 at/below the cosine
// threshold. `cos_threshold` is pre-sanitised non-NaN by the host; the clamp to
// [-1, 1 - EPS] mirrors the golden. The dot product is written in the golden's
// left-to-right association.
fn normal_consistency(
    cnx: f32, cny: f32, cnz: f32,
    hnx: f32, hny: f32, hnz: f32,
    cos_threshold: f32,
) -> f32 {
    let threshold = clamp(cos_threshold, -1.0, 1.0 - EPS);
    let cos_v = cnx * hnx + cny * hny + cnz * hnz;
    return clamp01((cos_v - threshold) / (1.0 - threshold));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }

    let cur = current[i];
    let hist = history[i];

    var reasons = 0u;
    var confidence = 1.0;

    if (params.require_surface_match != 0u
        && (cur.sid_lo != hist.sid_lo || cur.sid_hi != hist.sid_hi)) {
        reasons = reasons | SURFACE_MISMATCH;
        confidence = 0.0;
    }

    let depth_conf = depth_consistency(cur.depth, hist.depth, params.depth_relative_tolerance);
    if (depth_conf <= 0.0) {
        reasons = reasons | DEPTH_DISCONTINUITY;
    }
    confidence = min(confidence, depth_conf);

    let normal_conf = normal_consistency(
        cur.nx, cur.ny, cur.nz,
        hist.nx, hist.ny, hist.nz,
        params.normal_cos_threshold,
    );
    if (normal_conf <= 0.0) {
        reasons = reasons | NORMAL_DISCONTINUITY;
    }
    confidence = min(confidence, normal_conf);

    var accepted = 0u;
    if (reasons == 0u && confidence >= params.accept_threshold) {
        accepted = 1u;
    }

    out_verdict[i] = Verdict(accepted, confidence, reasons, 0u);
}
