//! GPU compute twin of the inter-layer garment coupling Jacobi pass.
//!
//! CPU golden reference:
//! prism_render_architecture::cloth::layers_jacobi::accumulate_layer_jacobi_corrections
//! -- the own-slot Jacobi form of the Gauss-Seidel core in cloth::layers, which
//! keeps stacked garments (shirt under jacket, lining under skirt) from sinking
//! through each other while preserving their inner-to-outer stacking order. Each
//! particle owns its own correction slot and sums only its own half of the
//! separating push against every penetrating cross-layer neighbour, reading
//! positions from a frozen read-only snapshot, so the parallel kernel reproduces
//! the Jacobi golden value-for-value regardless of dispatch order. This is the
//! same layered-cloth constraint used by production engines (UE5 Chaos Cloth,
//! Houdini Vellum), expressed here from first principles without reusing code.
//!
//! ## No spatial hash on the GPU (host-prebuilt CSR neighbourhood)
//!
//! The golden buckets every particle into a uniform spatial hash and walks each
//! particle's 27-cell neighbourhood in ascending BTreeMap-cell / ascending
//! bucket order. Rebuilding that hash on the GPU would be non-deterministic, so
//! instead the host mirrors the aerodynamics gather kernel: it prebuilds, in the
//! exact golden traversal order, the list of cross-layer candidate neighbours
//! for every particle and uploads it as a CSR adjacency (layer_offsets of length
//! particle_count + 1, layer_entries holding neighbour particle indices). The
//! host filters same-layer pairs and the self index out, so this kernel only
//! gathers its own CSR row, sums half corrections in the uploaded order, and
//! writes layer_corrections[a]. One invocation per particle, no atomics, no
//! reduction hazard.
//!
//! ## Math (mirrors the golden half_layer_correction line for line)
//!
//! For a cross-layer pair the lower layer number is the inner surface. Weights
//! are inverse masses clamped to non-negative (positions.w); a jointly immovable
//! pair (w_sum <= 0) contributes nothing. When the inner particle carries a
//! usable outward normal the contact is an oriented plane: the outer layer is
//! driven to at least thickness along that normal (signed = dot(p_outer -
//! p_inner, unit); skip when signed >= thickness). Otherwise it falls back to a
//! symmetric radial minimum-distance push (skip when dist_sq >= thickness_sq;
//! coincident particles separate along +X). Each endpoint receives only its own
//! mass-weighted half, so applying both halves reproduces the two-sided move.
//!
//! normalize_or_zero mirrors the physics single source (glam::Vec3::
//! normalize_or_zero): unit = normal * (1.0 / length(normal)) whenever that
//! reciprocal length is finite and positive, else the zero vector. Only an
//! exactly-zero or non-finite normal falls back to the radial push, matching
//! prism_physics_core and its WGSL twin value-for-value.
//!
//! ## Determinism and validation
//!
//! Every invocation writes only layer_corrections[a] and reads only shared
//! read-only buffers, so there is no write hazard. The sandbox has no GPU, so
//! this shader is validated by naga compilation only (via cloth::shader_tests);
//! real-machine value parity against the Jacobi golden is proven by
//! cloth::layers_gpu_tests on oriented, radial, pinned-inner and same-layer
//! fixtures.

/// Squared-length floor below which a vector is treated as zero. Used by the
/// `dot(unit, unit)` oriented-branch test and the radial coincidence test,
/// mirroring the CPU EPS_LEN_SQ.
const CLOTH_LAYERS_EPS_LEN_SQ: f32 = 1.0e-12;

/// Largest finite f32, used to reject non-finite reciprocal lengths in
/// [`cloth_layers_normalize_or_zero`] exactly as glam does.
const CLOTH_LAYERS_F32_MAX: f32 = 3.4028235e38;

/// Normalizes `v`, returning the zero vector when `v` has zero or non-finite
/// length — an exact mirror of `glam::Vec3::normalize_or_zero` (and the physics
/// engine's WGSL twin), which keeps the result only when the reciprocal length
/// is finite and positive. A tiny but nonzero normal is therefore still treated
/// as a usable direction, matching prism_physics_core.
fn cloth_layers_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    let rcp = 1.0 / len;
    if rcp == rcp && rcp > 0.0 && rcp <= CLOTH_LAYERS_F32_MAX {
        return v * rcp;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

/// Uniform parameters for the inter-layer coupling pass. Flat 16-byte block: the
/// sanitized separation thickness, the particle count bounding the dispatch, and
/// padding to a 16-byte uniform stride. The spatial-hash cell size lives only on
/// the host (it shapes the prebuilt CSR) and is not needed by the kernel.
struct GpuClothLayerParams {
    /// Minimum separation enforced between particles of different layers.
    thickness: f32,
    /// Number of particles, bounding the per-particle dispatch.
    particle_count: u32,
    /// Padding to a 16-byte uniform stride.
    pad0: u32,
    /// Padding to a 16-byte uniform stride.
    pad1: u32,
}

// Frozen particle snapshot: xyz = world position, w = inverse mass. Read only.
@group(0) @binding(0) var<storage, read> layer_positions: array<vec4<f32>>;
// Per-particle layer number (lower = inner). Read only.
@group(0) @binding(1) var<storage, read> layer_of: array<u32>;
// Per-particle outward surface normal in xyz. Read only.
@group(0) @binding(2) var<storage, read> layer_normals: array<vec4<f32>>;
// CSR row offsets, length particle_count + 1.
@group(0) @binding(3) var<storage, read> layer_offsets: array<u32>;
// CSR entries: cross-layer neighbour particle indices, host-ordered.
@group(0) @binding(4) var<storage, read> layer_entries: array<u32>;
// Per-particle total position correction; xyz written, w cleared to 0.
@group(0) @binding(5) var<storage, read_write> layer_corrections: array<vec4<f32>>;
// Flat 16-byte uniform block.
@group(0) @binding(6) var<uniform> layer_params: GpuClothLayerParams;

/// Particle a's own half of the cross-layer separating push against neighbour b.
///
/// The lower layer number is the inner surface whose outward normal orients the
/// contact; the outer particle is driven to at least thickness along that normal.
/// Falls back to a symmetric radial push when the inner normal is (near) zero.
/// Returns the zero vector when the pair is already separated or jointly
/// immovable. Positions come from the frozen snapshot, so the value matches the
/// golden half_layer_correction for particle a exactly.
fn half_layer_correction(a: u32, b: u32, thickness: f32, thickness_sq: f32) -> vec3<f32> {
    var inner = b;
    var outer = a;
    if layer_of[a] < layer_of[b] {
        inner = a;
        outer = b;
    }

    let w_inner = max(layer_positions[inner].w, 0.0);
    let w_outer = max(layer_positions[outer].w, 0.0);
    let w_sum = w_inner + w_outer;
    if w_sum <= 0.0 {
        return vec3<f32>(0.0);
    }

    let p_inner = layer_positions[inner].xyz;
    let p_outer = layer_positions[outer].xyz;
    let normal = layer_normals[inner].xyz;

    // normalize_or_zero mirrors glam: a tiny but nonzero normal stays oriented.
    let unit = cloth_layers_normalize_or_zero(normal);

    if dot(unit, unit) > CLOTH_LAYERS_EPS_LEN_SQ {
        // Oriented plane contact along the inner particle's outward normal.
        let signed = dot(p_outer - p_inner, unit);
        if signed >= thickness {
            return vec3<f32>(0.0);
        }
        let penetration = thickness - signed;
        if a == inner {
            return unit * (-penetration * (w_inner / w_sum));
        }
        return unit * (penetration * (w_outer / w_sum));
    }

    // No usable normal: symmetric radial minimum-distance separation.
    let delta = p_outer - p_inner;
    let dist_sq = dot(delta, delta);
    if dist_sq >= thickness_sq {
        return vec3<f32>(0.0);
    }
    var dir = vec3<f32>(1.0, 0.0, 0.0);
    var penetration = thickness;
    if dist_sq > CLOTH_LAYERS_EPS_LEN_SQ {
        let dist = sqrt(dist_sq);
        dir = delta * (1.0 / dist);
        penetration = thickness - dist;
    }
    if a == inner {
        return dir * (-penetration * (w_inner / w_sum));
    }
    return dir * (penetration * (w_outer / w_sum));
}

/// Inter-layer coupling main kernel: one invocation per particle sums its own
/// half of every penetrating cross-layer contact from its prebuilt CSR row.
///
/// Out-of-range invocations return. A non-positive thickness leaves the
/// correction at zero (mirroring the golden's disabled-pass guard). The CSR row
/// already excludes same-layer pairs and the self index, so no extra filtering
/// is needed here. The result is written to layer_corrections[a] with w = 0.
@compute @workgroup_size(64)
fn cloth_layer_coupling(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a = gid.x;
    if a >= layer_params.particle_count {
        return;
    }
    var accum = vec3<f32>(0.0);
    let thickness = layer_params.thickness;
    if thickness > 0.0 {
        let thickness_sq = thickness * thickness;
        let start = layer_offsets[a];
        let end = layer_offsets[a + 1u];
        for (var e = start; e < end; e = e + 1u) {
            let b = layer_entries[e];
            accum = accum + half_layer_correction(a, b, thickness, thickness_sq);
        }
    }
    layer_corrections[a] = vec4<f32>(accum, 0.0);
}
