// Root-binding resolve twin: evaluates on-device the same per-strand world root
// frame the CPU golden
// `prism_render_architecture::hair::binding::resolve_root_frames` computes, one
// thread per binding.
//
// A groom is authored in a character's rest pose, but the scalp is a skinned
// mesh that deforms every frame. At import time each strand root is projected
// onto its closest scalp triangle and stored as a barycentric attachment plus a
// signed height along the face normal (the `bind_roots` bake). Every frame the
// deformed triangle is re-read and the root world transform is reconstructed
// from it so the hair rides the head instead of floating — the root-skinning
// step `UE5` Groom / `TressFX` run on the GPU. This kernel is the on-device twin
// of exactly that per-frame replay: one thread per binding reads its three
// deformed triangle corners, rebuilds the outward face normal, interpolates the
// barycentric surface point, floats it off along the normal by the stored
// height, and completes a right-handed orthonormal basis. It does not bake the
// binding (nearest-triangle search stays the once-per-import `bind_roots` pass);
// it owns the naturally parallel per-strand resolve.
//
// Buffer layout: `params` carries the binding / triangle / vertex counts;
// `bindings[t]` is render strand `t`'s attachment (triangle index, three
// barycentric weights, signed height); `tri_indices` is the flattened triangle
// list (three vertex indices per face); `vertices` is the current deformed
// vertex pool (xyz in a vec4, w padding); `frames` receives four vec4 per
// binding — position, tangent, normal, bitangent — at `4*t`.
//
// Degenerate inputs (an unbound or out-of-range triangle, an out-of-range vertex
// index, a zero-area deformed face) resolve to the identity frame exactly as the
// reference does, so a bad binding never panics and never poisons its neighbours.
//
// Provenance: standard barycentric mesh-attachment resolve plus a Gram-Schmidt
// orthonormal basis plus wgpu compute dispatch; no Unreal Engine source or
// derived code.

// `MeshBinding::UNBOUND` sentinel: this root failed to bind to any triangle.
const UNBOUND: u32 = 0xffffffffu;
// Squared-length degeneracy guard (mirrors `binding::EPS_LEN_SQ`).
const EPS_LEN_SQ: f32 = 1.0e-12;
// `f32::EPSILON`, the threshold `Vec3::normalize_or` compares squared length to.
const F32_EPSILON: f32 = 1.1920929e-7;

struct Params {
    // Number of render-strand bindings, bounding the dispatch.
    binding_count: u32,
    // Number of scalp triangles; a binding triangle `>= triangle_count` is unbound.
    triangle_count: u32,
    // Number of deformed vertices; a corner index `>= vertex_count` is unbound.
    vertex_count: u32,
    pad0: u32,
}

// One strand root's attachment. Matches the host `GpuMeshBinding`: the triangle
// index, the three barycentric weights, the signed height, and three pad words
// so the stride is a 16-byte multiple with no `vec3` alignment hazard.
struct MeshBinding {
    triangle: u32,
    bary0: f32,
    bary1: f32,
    bary2: f32,
    height: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-binding scalp attachment, replayed against the deformed vertices.
@group(0) @binding(1) var<storage, read> bindings: array<MeshBinding>;
// Flattened triangle list: three vertex indices per face at `3*triangle`.
@group(0) @binding(2) var<storage, read> tri_indices: array<u32>;
// Current deformed vertex positions (xyz in a vec4, w unused).
@group(0) @binding(3) var<storage, read> vertices: array<vec4<f32>>;
// Per-binding resolved frame: position/tangent/normal/bitangent at `4*t`.
@group(0) @binding(4) var<storage, read_write> frames: array<vec4<f32>>;

// Bit-faithful port of `Vec3::normalize_or`: returns the unit vector, or
// `fallback` when the squared length is at or below `f32::EPSILON`.
fn normalize_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let ls = dot(v, v);
    if ls <= F32_EPSILON {
        return fallback;
    }
    return v * (1.0 / sqrt(ls));
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.binding_count {
        return;
    }
    let base = t * 4u;

    // Identity frame (the reference `RootFrame::IDENTITY`), overwritten below on
    // a valid, non-degenerate binding.
    var position = vec3<f32>(0.0, 0.0, 0.0);
    var tangent = vec3<f32>(1.0, 0.0, 0.0);
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var bitangent = vec3<f32>(0.0, 0.0, 1.0);

    let b = bindings[t];
    // `is_bound()` plus the `triangle_corners` range check, in the golden's order.
    if b.triangle != UNBOUND && b.triangle < params.triangle_count {
        let ti = b.triangle * 3u;
        let ia = tri_indices[ti + 0u];
        let ib = tri_indices[ti + 1u];
        let ic = tri_indices[ti + 2u];
        if ia < params.vertex_count && ib < params.vertex_count && ic < params.vertex_count {
            let a = vertices[ia].xyz;
            let bb = vertices[ib].xyz;
            let cc = vertices[ic].xyz;
            // `face_normal`: zero-area faces stay on the identity frame.
            let n_raw = cross(bb - a, cc - a);
            if dot(n_raw, n_raw) > EPS_LEN_SQ {
                let nrm = normalize_or(n_raw, vec3<f32>(0.0, 1.0, 0.0));
                // Barycentric surface point, floated off along the normal.
                let surface = a * b.bary0 + bb * b.bary1 + cc * b.bary2;
                position = surface + nrm * b.height;
                // First edge as the seed tangent, re-orthonormalized against the
                // normal so the basis stays right-handed (Gram-Schmidt).
                let tan0 = normalize_or(bb - a, vec3<f32>(1.0, 0.0, 0.0));
                let bit = normalize_or(cross(nrm, tan0), vec3<f32>(0.0, 0.0, 1.0));
                tangent = normalize_or(cross(bit, nrm), tan0);
                normal = nrm;
                bitangent = bit;
            }
        }
    }

    frames[base + 0u] = vec4<f32>(position, 0.0);
    frames[base + 1u] = vec4<f32>(tangent, 0.0);
    frames[base + 2u] = vec4<f32>(normal, 0.0);
    frames[base + 3u] = vec4<f32>(bitangent, 0.0);
}
