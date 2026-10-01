//! GPU compute twin of the render-mesh barycentric embedding pass.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::cloth::embed::embed_render_vertex` (and its batch
//! wrapper `embed_render_mesh`). This is the `ClothKernel::SkinEmbed` dispatch
//! whose contract — 3 storage buffers + 1 uniform buffer, a 64-lane linear
//! workgroup, and the `DispatchDomain::RenderVertex` launch shape — is declared
//! by `prism_render_architecture::cloth::gpu::kernels`; the stable WESL entry
//! point name emitted by that codegen is `cloth_skin_embed`, matched exactly by
//! the entry point below.
//!
//! One invocation owns one render-mesh vertex: it reads the vertex's host
//! triangle indices, barycentric weights, and signed normal offset, fetches the
//! three current sim positions, reconstructs the in-plane point
//! `p0*w0 + p1*w1 + p2*w2`, and pushes it back out along the re-evaluated face
//! normal by the stored offset — restoring the garment thickness while tracking
//! both the deformation and the orientation of the host triangle. Every
//! out-of-range index degrades to a zero output (mirroring the CPU
//! `host_positions` `None` -> `Vec3::ZERO` fallback) and a degenerate (zero
//! area) host triangle contributes a zero normal (mirroring
//! `Vec3::normalize_or_zero`), so no `NaN` can ever enter the render stream.
//!
//! The sandbox has no GPU, so this shader is validated by naga compilation only
//! (via the resolve `shader_tests`); no dispatch is executed here.
//!
//! ## Binding stride note (`EMBED_STRIDE`)
//!
//! The full CPU golden `BarycentricBinding` carries `tri: [u32; 3]` (three host
//! indices), `bary: (f32, f32, f32)` (three weights), and `normal_offset: f32`
//! — seven 4-byte words = 28 bytes, which rounds up to a **32-byte std430
//! stride** (seven useful `u32`/`f32` words plus one padding word). The host
//! constant `prism_render_architecture::cloth::gpu::buffers::EMBED_STRIDE` is
//! `32` to match, so the host allocation and the `ClothEmbedBinding` struct
//! below agree byte-for-byte.

/// Squared-length floor below which a vector is treated as zero, so
/// normalization never divides by ~0. Mirrors the CPU `EPS_LEN_SQ` shared by
/// `Vec3::normalize_or_zero` and the whole cloth subsystem.
const CLOTH_EMBED_EPS_LEN_SQ: f32 = 1.0e-12;

/// One render-mesh vertex's binding to a single sim-mesh triangle. Byte layout
/// mirrors the CPU golden `BarycentricBinding`: three host-triangle particle
/// indices, three barycentric weights, and a signed face-normal offset that
/// restores the render surface's distance from the sim cage (garment
/// thickness). `pad` rounds the record up to the 32-byte std430 stride the
/// storage buffer uses (see the module-level `EMBED_STRIDE` note).
struct ClothEmbedBinding {
    /// Index of the first host-triangle sim particle.
    tri0: u32,
    /// Index of the second host-triangle sim particle.
    tri1: u32,
    /// Index of the third host-triangle sim particle.
    tri2: u32,
    /// Barycentric weight of the first host vertex.
    w0: f32,
    /// Barycentric weight of the second host vertex.
    w1: f32,
    /// Barycentric weight of the third host vertex.
    w2: f32,
    /// Signed distance along the host triangle's face normal (thickness).
    normal_offset: f32,
    /// Padding to the 32-byte std430 stride; never read.
    pad: u32,
}

/// Global parameters bounding the render-vertex dispatch. Mirrors the single
/// resident count the host derives from the buffer sizing contract.
struct ClothEmbedParams {
    /// Number of render-mesh vertices, bounding the linear dispatch.
    render_vertex_count: u32,
}

// Sim-mesh particle positions (`xyz` = world position); source of the host
// triangle vertices. Read-only.
@group(0) @binding(0) var<storage, read> sim_positions: array<vec4<f32>>;
// Render-mesh output positions (`xyz` = embedded position, `w` preserved).
@group(0) @binding(1) var<storage, read_write> render_positions: array<vec4<f32>>;
// Per-render-vertex embedding bindings, index-aligned with `render_positions`.
@group(0) @binding(2) var<storage, read> embed_bindings: array<ClothEmbedBinding>;
// Dispatch bounds.
@group(0) @binding(3) var<uniform> params: ClothEmbedParams;

/// Unit vector along `v`, or the zero vector when `v` is (numerically) zero, so
/// normalization never yields `NaN`. Mirrors the CPU `Vec3::normalize_or_zero`
/// (compares the squared length against `CLOTH_EMBED_EPS_LEN_SQ`).
fn cloth_embed_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq > CLOTH_EMBED_EPS_LEN_SQ {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

/// Unit face normal of the triangle `(p0, p1, p2)`, or the zero vector when the
/// triangle is degenerate. Mirrors the CPU `face_normal`:
/// `normalize_or_zero(cross(p1 - p0, p2 - p0))`.
fn cloth_embed_face_normal(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec3<f32> {
    return cloth_embed_normalize_or_zero(cross(p1 - p0, p2 - p0));
}

/// Embeds one render-mesh vertex from its binding and the current sim
/// positions. One invocation per render vertex. Mirrors `embed_render_vertex`:
/// out-of-range host indices write a zero position (host `w` preserved) and
/// return; otherwise the in-plane barycentric point is pushed along the
/// re-evaluated face normal by the stored offset.
@compute @workgroup_size(64)
fn cloth_skin_embed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.render_vertex_count {
        return;
    }

    let binding = embed_bindings[i];
    let count = arrayLength(&sim_positions);
    // Mirror the CPU `host_positions`: any out-of-range index degrades to a
    // defined zero output rather than a panic / out-of-bounds read.
    if binding.tri0 >= count || binding.tri1 >= count || binding.tri2 >= count {
        render_positions[i] = vec4<f32>(0.0, 0.0, 0.0, render_positions[i].w);
        return;
    }

    let p0 = sim_positions[binding.tri0].xyz;
    let p1 = sim_positions[binding.tri1].xyz;
    let p2 = sim_positions[binding.tri2].xyz;

    let base = p0 * binding.w0 + p1 * binding.w1 + p2 * binding.w2;
    let normal = cloth_embed_face_normal(p0, p1, p2);
    let embedded = base + normal * binding.normal_offset;

    render_positions[i] = vec4<f32>(embedded, render_positions[i].w);
}
