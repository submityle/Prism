//! GPU groom-root skinning resolve for guide strands.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::binding::resolve_root_frames`. A groom is
//! authored in a character's rest pose, but the scalp is a skinned mesh that
//! deforms every frame; for hair to ride the head instead of floating, each
//! strand root is pinned to the surface it grew from and follows it as it skins.
//! This is the runtime half of a `UE5` Groom binding asset / the `TressFX` root
//! skinning step: at import time each root is projected onto the closest scalp
//! triangle and stored as a barycentric attachment plus a signed height along
//! the face normal (CPU `bind_roots`); every frame the deformed triangle is
//! re-read and the root world transform is reconstructed from it. This shader is
//! that per-frame reconstruction, feeding the pinned particles of the XPBD sim
//! (`hair_sim.wesl`, `inverse_mass == 0`) and the basis that orients each
//! strand's rest-pose shape (design §8 GPU-driven persistent pipeline: roots ride
//! the skinned scalp *before* the sim step advances the free particles).
//!
//! One invocation owns one root binding. The triangle topology is stable
//! between bind time and every resolve — only the vertex positions change — so
//! the shader only reads the current skinned `vertices`. Everything is
//! bit-faithful to the CPU golden: the same bindings and vertices produce
//! identical frames. An unbound binding (`triangle == HAIR_ROOT_UNBOUND`), an
//! out-of-range triangle or vertex index, and a zero-area deformed face all
//! resolve to the identity frame so a malformed binding never panics and never
//! poisons a neighbour. A `GPU` is now available, so this shader
//! is compile- and type-checked through the same render-world `ShaderCache` /
//! `wesl` pipeline the runtime uses (see `shader_tests`), with on-device
//! dispatch/perf calibration still awaiting real hardware.

/// Sentinel triangle index meaning "this root could not be bound". Mirrors the
/// CPU `binding::UNBOUND` (`u32::MAX`).
const HAIR_ROOT_UNBOUND: u32 = 0xffffffffu;

/// Squared-length threshold below which a face is treated as zero-area. Mirrors
/// the CPU `binding::EPS_LEN_SQ`.
const HAIR_ROOT_EPS_LEN_SQ: f32 = 1.0e-12;

/// `f32::EPSILON`; the guard `normalize_or` uses to fall back to a fixed axis.
/// Mirrors the CPU `interpolation::Vec3::normalize_or` (which compares the
/// squared length against `f32::EPSILON`, not `EPS_LEN_SQ`).
const HAIR_ROOT_F32_EPSILON: f32 = 1.1920929e-7;

/// One strand root's attachment to the scalp mesh. Packs the CPU `MeshBinding`
/// into a `vec4` plus the triangle index on a clean 32-byte stride:
/// `bary_height.xyz` = barycentric weights, `bary_height.w` = signed height
/// along the face normal, `triangle` = triangle index (or `HAIR_ROOT_UNBOUND`).
struct HairMeshBinding {
    bary_height: vec4<f32>,
    triangle: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A resolved root world transform: an origin plus a right-handed orthonormal
/// basis. Mirrors the CPU `RootFrame`; each field's `xyz` carries the vector.
struct HairRootFrame {
    position: vec4<f32>,
    tangent: vec4<f32>,
    normal: vec4<f32>,
    bitangent: vec4<f32>,
}

/// Dispatch parameters. `root_count` bounds the dispatch; `triangle_count` and
/// `vertex_count` bound the index reads so an out-of-range binding degrades to
/// the identity frame instead of reading out of bounds.
struct HairRootParams {
    root_count: u32,
    triangle_count: u32,
    vertex_count: u32,
}

// Per-root attachments recorded at bind time.
@group(0) @binding(0) var<storage, read> bindings: array<HairMeshBinding>;
// Current (deformed / skinned) scalp vertex positions (`xyz` used).
@group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
// Flat triangle index list, three `u32` per triangle (same topology as bind time).
@group(0) @binding(2) var<storage, read> indices: array<u32>;
// Resolved world-space root frames, one per binding.
@group(0) @binding(3) var<storage, read_write> frames: array<HairRootFrame>;

var<immediate> params: HairRootParams;

// Mirror of CPU `interpolation::Vec3::normalize_or`: unit vector, or `fallback`
// when `self` is (near) zero length. Note the guard is `f32::EPSILON`, matching
// the CPU golden exactly (not the coarser `EPS_LEN_SQ`).
fn hair_root_normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq <= HAIR_ROOT_F32_EPSILON {
        return fallback;
    }
    return v * (1.0 / sqrt(len_sq));
}

// The identity frame at the origin, used for unbound or degenerate roots.
// Mirrors CPU `RootFrame::IDENTITY`.
fn hair_root_identity() -> HairRootFrame {
    return HairRootFrame(
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
    );
}

@compute @workgroup_size(64)
fn resolve_root_frames(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index >= params.root_count {
        return;
    }

    let binding = bindings[index];

    // Unbound attachment → identity (mirrors `is_bound`).
    if binding.triangle == HAIR_ROOT_UNBOUND {
        frames[index] = hair_root_identity();
        return;
    }
    // Out-of-range triangle → identity (mirrors `triangle_corners` returning None).
    if binding.triangle >= params.triangle_count {
        frames[index] = hair_root_identity();
        return;
    }
    let base = binding.triangle * 3u;
    let i0 = indices[base];
    let i1 = indices[base + 1u];
    let i2 = indices[base + 2u];
    // Out-of-range vertex index → identity.
    if i0 >= params.vertex_count || i1 >= params.vertex_count || i2 >= params.vertex_count {
        frames[index] = hair_root_identity();
        return;
    }
    let a = vertices[i0].xyz;
    let b = vertices[i1].xyz;
    let c = vertices[i2].xyz;

    // Face normal; a zero-area deformed face → identity (mirrors `face_normal`).
    let n_raw = cross(b - a, c - a);
    if dot(n_raw, n_raw) <= HAIR_ROOT_EPS_LEN_SQ {
        frames[index] = hair_root_identity();
        return;
    }
    let normal = hair_root_normalize_or(n_raw, vec3<f32>(0.0, 1.0, 0.0));

    let bary = binding.bary_height.xyz;
    let height = binding.bary_height.w;
    // Barycentric surface point, then float off along the face normal.
    let surface = a * bary.x + b * bary.y + c * bary.z;
    let position = surface + normal * height;

    // Tangent follows the first edge; re-orthonormalize against the normal so the
    // basis stays right-handed even when the edge is not perpendicular to it.
    let tangent0 = hair_root_normalize_or(b - a, vec3<f32>(1.0, 0.0, 0.0));
    let bitangent = hair_root_normalize_or(cross(normal, tangent0), vec3<f32>(0.0, 0.0, 1.0));
    let tangent = hair_root_normalize_or(cross(bitangent, normal), tangent0);

    frames[index] = HairRootFrame(
        vec4<f32>(position, 0.0),
        vec4<f32>(tangent, 0.0),
        vec4<f32>(normal, 0.0),
        vec4<f32>(bitangent, 0.0),
    );
}
