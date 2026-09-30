// Cluster-granularity 64-bit vis-buffer twin: rasterizes an *indexed* cluster
// (a shared vertex array plus triangle index triples) into the shipping
// `(depth << 32) | payload` visibility word, mirroring the CPU golden
// `rasterize_cluster`.
//
// Where `vis_buffer_payload_raster.wesl` receives already-flattened triangles
// with an explicit per-triangle payload, this kernel reproduces the shipping
// per-cluster dispatch: each triangle fetches its three vertices through the
// index list, synthesizes its own payload as `pack_cluster_triangle` -
// `(cluster_id << 7) | (triangle_id & 127)` - and composites through a single
// 64-bit `atomicMax`. A triangle whose index list points past the vertex array
// writes nothing (matching the CPU `vertices.get(..)` skip), and the per-cluster
// triangle cap is enforced by the host dispatch (`tri_count` is pre-clamped to
// `MAX_CLUSTER_TRIANGLES` so ids never alias the 7-bit triangle field).
//
// This kernel requires the `SHADER_INT64` and `SHADER_INT64_ATOMIC_MIN_MAX`
// device features; `GpuClusterRaster::new` only compiles it after
// `GpuContext::supports_u64_atomics` confirms both are enabled.
//
// Coverage arithmetic (signed edge function, winding swap, top-left fill rule),
// the reversed-Z depth encode and the payload packing are identical to the CPU
// golden `rasterize_cluster` / `rasterize_triangle` / `pack_cluster_triangle`,
// so the covered texel set and winning payload are bit-exact and the depth
// field matches within one ULP (the same reassociation tolerance the other
// vis-buffer twins document).
//
// Provenance: standard signed-edge / top-left-rule software rasterization; no
// Unreal Engine source or derived code.

struct Params {
    // Framebuffer width in pixels.
    width: u32,
    // Framebuffer height in pixels.
    height: u32,
    // Number of triangles to rasterize, already clamped to the per-cluster cap.
    tri_count: u32,
    // Number of valid vertices in the `verts` array; indices at or past this
    // are treated as out of range and drop their triangle.
    vert_count: u32,
    // Cluster id folded into the high bits of every payload.
    cluster_id: u32,
    // Non-zero requests back-face culling of the original winding.
    cull_back: u32,
    // Padding to a 16-byte multiple for uniform layout.
    pad0: u32,
    pad1: u32,
}

// One screen-space vertex: pixel-space position (y-down) and reversed-Z depth.
// Twelve-byte stride matching the `bytemuck` upload struct exactly.
struct Vert {
    x: f32,
    y: f32,
    d: f32,
}

// One triangle's vertex indices into `verts`. Twelve-byte stride.
struct Idx {
    i0: u32,
    i1: u32,
    i2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> verts: array<Vert>;
@group(0) @binding(2) var<storage, read> indices: array<Idx>;
@group(0) @binding(3) var<storage, read_write> vis: array<atomic<u64>>;

// Signed edge function of point `p` against the directed edge `a -> b`.
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// True when the directed edge `a -> b` is a top or left edge (y-down, positive
// area convention). Mirrors the CPU `is_top_left`.
fn is_top_left(ax: f32, ay: f32, bx: f32, by: f32) -> bool {
    let dx = bx - ax;
    let dy = by - ay;
    return dy < 0.0 || (dy == 0.0 && dx > 0.0);
}

// Top-left fill test for one edge value. Mirrors the CPU `covers`.
fn covers(e: f32, tl: bool) -> bool {
    return e > 0.0 || (e == 0.0 && tl);
}

// One thread per (pixel, triangle); `global_id.z` selects the triangle so
// overlapping triangles composite through genuine 64-bit `atomicMax`
// contention, faithfully exercising the shipping scatter model.
@compute @workgroup_size(8, 8, 1)
fn raster(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let t = gid.z;
    if (x >= params.width || y >= params.height || t >= params.tri_count) {
        return;
    }

    let tri = indices[t];
    // Out-of-range index: this triangle contributes nothing, mirroring the CPU
    // `vertices.get(..)` skip. Never index the vertex array out of bounds.
    if (tri.i0 >= params.vert_count || tri.i1 >= params.vert_count || tri.i2 >= params.vert_count) {
        return;
    }

    let a = verts[tri.i0];
    let b = verts[tri.i1];
    let c = verts[tri.i2];

    var v0x = a.x;
    var v0y = a.y;
    var d0 = a.d;
    var v1x = b.x;
    var v1y = b.y;
    var d1 = b.d;
    var v2x = c.x;
    var v2y = c.y;
    var d2 = c.d;

    let raw_area = edge(v0x, v0y, v1x, v1y, v2x, v2y);
    if (raw_area == 0.0) {
        return;
    }
    if (params.cull_back != 0u && raw_area < 0.0) {
        return;
    }
    if (raw_area < 0.0) {
        // Swap v1 and v2 (position and depth together) so the interior test
        // runs against a positive area, matching the CPU winding swap.
        let sx = v1x;
        let sy = v1y;
        let sd = d1;
        v1x = v2x;
        v1y = v2y;
        d1 = d2;
        v2x = sx;
        v2y = sy;
        d2 = sd;
    }

    let area = edge(v0x, v0y, v1x, v1y, v2x, v2y);
    let inv_area = 1.0 / area;

    let px = f32(x) + 0.5;
    let py = f32(y) + 0.5;

    let e0 = edge(v1x, v1y, v2x, v2y, px, py);
    let e1 = edge(v2x, v2y, v0x, v0y, px, py);
    let e2 = edge(v0x, v0y, v1x, v1y, px, py);

    let tl0 = is_top_left(v1x, v1y, v2x, v2y);
    let tl1 = is_top_left(v2x, v2y, v0x, v0y);
    let tl2 = is_top_left(v0x, v0y, v1x, v1y);

    if (!(covers(e0, tl0) && covers(e1, tl1) && covers(e2, tl2))) {
        return;
    }

    let b0 = e0 * inv_area;
    let b1 = e1 * inv_area;
    let b2 = e2 * inv_area;
    let d = b0 * d0 + b1 * d1 + b2 * d2;
    let depth_key = bitcast<u32>(clamp(d, 0.0, 1.0));

    // Payload = pack_cluster_triangle(cluster_id, t): cluster id in the high
    // bits, 7-bit triangle id in the low bits. `t < tri_count <= 128`, so the
    // mask is a no-op that documents the field width.
    let payload = (params.cluster_id << 7u) | (t & 127u);

    let key: u64 = (u64(depth_key) << 32u) | u64(payload);
    let idx = y * params.width + x;
    atomicMax(&vis[idx], key);
}
