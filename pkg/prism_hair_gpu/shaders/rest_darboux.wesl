// Per-edge rest `Darboux` vector twin.
//
// Device mirror of the `CPU` golden
// `prism_render_architecture::hair::cosserat::rest_darboux_from_frames`. Given
// a strand's material-frame chain (the unit quaternions produced by
// `parallel_transport_frames` / `build_strand_frames`), the rest `Darboux`
// vector between adjacent frames `a` and `b` is the imaginary part of the
// relative rotation `conjugate(a) * b`:
//
//   rel  = conjugate(a) * b         (Hamilton product)
//   drb  = (rel.x, rel.y, rel.z)    (imaginary / bend-twist part)
//
// This is the rest companion the `Cosserat` bend-twist constraint
// (`simulate_strand_cosserat`) reads back, so the on-device twin must match the
// scalar reference term for term.
//
// # Why one thread per edge
//
// Unlike the serial `parallel_transport` recurrence that builds the frames,
// each rest `Darboux` vector depends only on one adjacent frame pair, so this
// is embarrassingly parallel: one thread owns one edge, reads `frames[a]` and
// `frames[a + 1]` and writes one vector. The host lays out a flat per-edge base
// index so edges never straddle a strand boundary, exactly like the per-edge
// `strand_tangents` dispatch.
//
// Quaternions travel through the storage buffer as `vec4<f32>` in `(w, x, y, z)`
// lane order. The product is a single sum-of-four-products per component (no
// chained recurrence, no normalize), so the only `CPU` vs `GPU` divergence is
// legal fused-multiply-add contraction.
//
// It uses only multiply/add in the portable core-`WGSL` subset — no `sqrt`,
// `exp`, `pow`, hardware `sin` or optional device feature — so this twin runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard `Cosserat`/`Kirchhoff` rod rest-`Darboux` construction
// plus a `wgpu` compute dispatch; no Unreal Engine source or derived code.

// Shared parameters for one dispatch: the total edge count padded to one
// `16`-byte uniform slot.
struct Params {
    edge_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Material frames in `(w, x, y, z)` lane order, concatenated across strands.
@group(0) @binding(1) var<storage, read> frames: array<vec4<f32>>;
// Per edge: the absolute index of frame `a` in `frames` (frame `b` is `a + 1`).
@group(0) @binding(2) var<storage, read> edge_base: array<u32>;
// Per edge: the rest `Darboux` vector `(x, y, z, 0)`.
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if (e >= params.edge_count) {
        return;
    }
    let ai = edge_base[e];
    let a = frames[ai];
    let b = frames[ai + 1u];

    // (w, x, y, z) lane order.
    let aw = a.x;
    let ax = a.y;
    let ay = a.z;
    let az = a.w;
    let bw = b.x;
    let bx = b.y;
    let by = b.z;
    let bz = b.w;

    // conjugate(a) = (w, -x, -y, -z)
    let cw = aw;
    let cx = -ax;
    let cy = -ay;
    let cz = -az;

    // rel = conjugate(a) * b (Hamilton product); only the imaginary part is
    // the rest Darboux vector, matching `Quat::mul` term for term.
    let rx = cw * bx + cx * bw + cy * bz - cz * by;
    let ry = cw * by - cx * bz + cy * bw + cz * bx;
    let rz = cw * bz + cx * by - cy * bx + cz * bw;

    out[e] = vec4<f32>(rx, ry, rz, 0.0);
}
