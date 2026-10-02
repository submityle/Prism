// GPU multi-layer garment coupling — the parallel-safe Jacobi twin of
// `prism_physics_core`'s `resolve_layer_coupling_jacobi`.
//
// A dressed character stacks garments (shirt under jacket, lining under skirt).
// Each garment simulates its own cloth with its own *layer number*, and the
// inter-layer coupling tier keeps a higher-numbered (outer) layer on the
// outward side of the lower-numbered (inner) one, at least `thickness` apart.
// The Gauss-Seidel core walks the cross-layer pairs sequentially and scatters
// each contact in place, so a later pair already sees an earlier pair's moved
// positions — a reduction order no compute kernel can reproduce, because every
// GPU invocation reads the *same* frozen snapshot. This kernel therefore
// mirrors the engine's Jacobi golden: a frozen-snapshot own-slot accumulate.
//
// Unlike the point self-collision tier (unordered pairs resolved in a per-pair
// pass), inter-layer coupling is a directed, per-particle own-slot sum, so a
// single pass suffices: one invocation per particle gathers that particle's
// cross-layer neighbors through a host-built CSR adjacency (built in the
// golden's exact reduction order, same-layer neighbors already filtered), sums
// only its own half of each separating push from the frozen snapshot, and
// writes `out_positions`.
//
// The adjacency and cell assignment are built on the host (integer-exact), so
// the only floating-point work on the GPU is the separating-push arithmetic,
// which the parity suite checks against the CPU golden within a tight
// tolerance.
//
// Provenance: the layer-number stacking constraint and inverse-mass-weighted
// separation are standard position-based dynamics; the Jacobi own-slot
// accumulate/apply split is standard parallel position-based dynamics; uniform
// spatial hashing is the classical Teschner et al. 2003 scheme. No Unreal
// Engine source or derived code.

// Coincident/zero-length floor below which an offset or normal has no defined
// direction, mirroring `prism_physics_core`'s `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Largest finite `f32`, used to reject the `+inf` reciprocal of a zero-length
// normal so `normalize_or_zero` matches `glam` bit-for-bit.
const F32_MAX: f32 = 3.40282347e38;

struct Params {
    // Number of addressable particles (kernel thread count).
    particle_count: u32,
    // Sanitized minimum inter-layer separation.
    thickness: f32,
    // thickness * thickness, precomputed on the host like the golden.
    thickness_sq: f32,
    // Padding to a 16-byte uniform.
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Frozen positions, xyz used (w carried through).
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
// Index-aligned outward surface normals, xyz used.
@group(0) @binding(2) var<storage, read> normals: array<vec4<f32>>;
// Index-aligned inverse masses (0 marks a pinned particle).
@group(0) @binding(3) var<storage, read> inverse_masses: array<f32>;
// Index-aligned layer numbers (lower = inner).
@group(0) @binding(4) var<storage, read> layer_of: array<u32>;
// CSR offsets into `nbr_entries`, length `particle_count + 1`.
@group(0) @binding(5) var<storage, read> nbr_offsets: array<u32>;
// Each particle's cross-layer neighbors in the golden's reduction order.
@group(0) @binding(6) var<storage, read> nbr_entries: array<u32>;
// Output: the frozen positions advanced by their accumulated corrections.
@group(0) @binding(7) var<storage, read_write> out_positions: array<vec4<f32>>;

// Normalizes `v`, returning the zero vector when `v` has (near) zero length —
// an exact mirror of `glam::Vec3::normalize_or_zero`, which keeps the result
// only when the reciprocal length is finite and positive.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    let rcp = 1.0 / len;
    if (rcp == rcp && rcp > 0.0 && rcp <= F32_MAX) {
        return v * rcp;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Particle `ai`'s own half of the cross-layer separating push against neighbor
// `bi`, mirroring `prism_physics_core`'s `half_layer_correction`.
//
// The lower layer number is the inner surface whose outward normal orients the
// contact, so the outer particle is driven to at least `thickness` along that
// normal. Falls back to a symmetric radial minimum-distance push when the inner
// normal is (near) zero. Returns the zero vector when the pair is already
// separated or jointly immovable.
fn half_layer_correction(ai: u32, bi: u32) -> vec3<f32> {
    // Lower layer number is the inner surface whose normal orients the contact.
    var inner: u32;
    var outer: u32;
    if (layer_of[ai] < layer_of[bi]) {
        inner = ai;
        outer = bi;
    } else {
        inner = bi;
        outer = ai;
    }

    let w_inner = max(inverse_masses[inner], 0.0);
    let w_outer = max(inverse_masses[outer], 0.0);
    let w_sum = w_inner + w_outer;
    if (w_sum <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let p_inner = positions[inner].xyz;
    let p_outer = positions[outer].xyz;
    let unit = normalize_or_zero(normals[inner].xyz);

    if (dot(unit, unit) > EPS_LEN_SQ) {
        // Oriented plane contact: force the outer particle at least `thickness`
        // along the inner's outward normal.
        let signed = dot(p_outer - p_inner, unit);
        if (signed >= params.thickness) {
            return vec3<f32>(0.0, 0.0, 0.0);
        }
        let penetration = params.thickness - signed;
        if (ai == inner) {
            return unit * (-penetration * (w_inner / w_sum));
        }
        return unit * (penetration * (w_outer / w_sum));
    }

    // No usable normal: symmetric radial separation.
    let delta = p_outer - p_inner;
    let dist_sq = dot(delta, delta);
    if (dist_sq >= params.thickness_sq) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    var dir: vec3<f32>;
    var penetration: f32;
    if (dist_sq <= EPS_LEN_SQ) {
        dir = vec3<f32>(1.0, 0.0, 0.0);
        penetration = params.thickness;
    } else {
        let dist = sqrt(dist_sq);
        dir = delta * (1.0 / dist);
        penetration = params.thickness - dist;
    }
    if (ai == inner) {
        return dir * (-penetration * (w_inner / w_sum));
    }
    return dir * (penetration * (w_outer / w_sum));
}

@compute @workgroup_size(64)
fn apply_layer_coupling(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }
    let frozen = positions[i].xyz;
    let start = nbr_offsets[i];
    let end = nbr_offsets[i + 1u];
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var e = start; e < end; e = e + 1u) {
        let b = nbr_entries[e];
        acc += half_layer_correction(i, b);
    }
    out_positions[i] = vec4<f32>(frozen + acc, positions[i].w);
}
