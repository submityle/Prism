// Follicle-binding root-transfer twin: evaluates on-device the same per-root
// world position the CPU golden
// `prism_render_architecture::hair::follicle_bind::transfer_root_map` computes,
// one thread per follicle binding.
//
// A groom is authored once in a rest pose, but the scalp it grows from is an
// animated skinned mesh. Each follicle (hair root) is bound to one scalp
// triangle by its barycentric coordinates plus a small signed offset along the
// interpolated surface normal (roots sit a hair's breadth above the skin). Once
// the owning renderer skins the scalp, every bound root is reconstructed from
// the *deformed* triangle so the hair rides the head without sliding or poking
// through — the root-skinning transfer `UE5` Groom / `TressFX` apply per frame.
// This kernel is the on-device twin of exactly that transfer, broadcasting one
// deformed triangle to a whole batch of bindings: one thread per binding
// sanitises its barycentric weights, interpolates the surface point and the
// per-vertex surface normal, then floats the root off along that normal by the
// stored signed offset.
//
// It is a distinct sibling of the `root_bind` / `root_resolve` pair: those
// resolve an indexed `MeshBinding` against a flattened vertex pool + triangle
// index list and rebuild a *geometric* face normal plus a full orthonormal
// `RootFrame`. This one takes an explicit per-vertex `TriangleFrame` shared by
// all bindings, interpolates the authored per-vertex normals (not a cross
// product), sanitises the barycentric weights (clamp negatives, renormalise,
// centroid fallback), and emits a position only.
//
// Buffer layout: `params` carries the one shared deformed triangle (three
// vertex positions and three per-vertex normals as `vec4`, xyz used) plus the
// binding count; `bindings[t]` is follicle `t`'s attachment (three barycentric
// weights and the signed normal offset); `out[t]` receives the transferred
// world root position (xyz in a `vec4`, w unused) at index `t`.
//
// Degenerate inputs (an all-non-positive weight set, a zero-length interpolated
// normal, a non-finite transferred position) fall back exactly as the golden
// does — the triangle centroid, the canonical +Z normal, and the bare surface
// point respectively — so a bad binding never panics and never poisons its
// neighbours. An empty batch dispatches nothing.
//
// Provenance: standard barycentric mesh-attachment transfer plus wgpu compute
// dispatch; no Unreal Engine source or derived code.

// Length degeneracy guard, mirroring `follicle_bind::DEGENERATE_EPS`. Compared
// against the *length* (not squared length), exactly like the golden's
// `normalize_or`.
const DEGENERATE_EPS: f32 = 1.0e-12;
// Largest finite `f32`; used with the self-equality test to reject NaN/inf the
// way the golden's `is_finite` checks do.
const F32_MAX: f32 = 3.4028235e38;

struct Params {
    // Deformed triangle vertex positions (xyz used, w padding).
    pos0: vec4<f32>,
    pos1: vec4<f32>,
    pos2: vec4<f32>,
    // Deformed per-vertex surface normals (xyz used, w padding).
    nrm0: vec4<f32>,
    nrm1: vec4<f32>,
    nrm2: vec4<f32>,
    // Number of follicle bindings, bounding the dispatch.
    binding_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One follicle attachment. Matches the host `GpuBinding`: three barycentric
// weights and the signed offset above the surface along the normal.
struct Binding {
    bary0: f32,
    bary1: f32,
    bary2: f32,
    normal_offset: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-follicle attachment, transferred against the shared deformed triangle.
@group(0) @binding(1) var<storage, read> bindings: array<Binding>;
// Per-follicle transferred world root position (xyz in a vec4, w unused).
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;

// `f32::is_finite` port: rejects NaN (self-inequality) and ±inf (magnitude
// above the largest finite value).
fn is_finite_scalar(x: f32) -> bool {
    return x == x && abs(x) <= F32_MAX;
}

fn is_finite3(v: vec3<f32>) -> bool {
    return is_finite_scalar(v.x) && is_finite_scalar(v.y) && is_finite_scalar(v.z);
}

// `Barycentric::sanitized`'s `clamp0`: keep a finite, strictly-positive weight,
// otherwise zero it.
fn sanitize_component(x: f32) -> f32 {
    if is_finite_scalar(x) && x > 0.0 {
        return x;
    }
    return 0.0;
}

// Bit-faithful port of `follicle_bind::normalize_or`: unit vector, or `fallback`
// when the length is at or below `DEGENERATE_EPS` (or non-finite).
fn normalize_or(a: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = sqrt(dot(a, a));
    if len > DEGENERATE_EPS && is_finite_scalar(len) {
        return a * (1.0 / len);
    }
    return fallback;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.binding_count {
        return;
    }

    let b = bindings[t];

    // `Barycentric::sanitized`: clamp negatives/non-finite to zero, renormalise,
    // and fall back to the triangle centroid when the sum underflows.
    let cu = sanitize_component(b.bary0);
    let cv = sanitize_component(b.bary1);
    let cw = sanitize_component(b.bary2);
    let sum = cu + cv + cw;
    var bu = 1.0 / 3.0;
    var bv = 1.0 / 3.0;
    var bw = 1.0 / 3.0;
    if sum > DEGENERATE_EPS {
        bu = cu / sum;
        bv = cv / sum;
        bw = cw / sum;
    }

    // Barycentric surface point, in the golden's `bary_mix` association order
    // `((p0*u) + (p1*v)) + (p2*w)`.
    let surface = (params.pos0.xyz * bu + params.pos1.xyz * bv) + params.pos2.xyz * bw;
    // Interpolated per-vertex normal, re-normalised with the +Z fallback.
    let raw_normal = (params.nrm0.xyz * bu + params.nrm1.xyz * bv) + params.nrm2.xyz * bw;
    let normal = normalize_or(raw_normal, vec3<f32>(0.0, 0.0, 1.0));

    // Float the surface point off along the normal by the stored offset; a
    // non-finite result falls back to the bare surface point.
    let floated = surface + normal * b.normal_offset;
    var result = surface;
    if is_finite3(floated) {
        result = floated;
    }

    out[t] = vec4<f32>(result, 0.0);
}
