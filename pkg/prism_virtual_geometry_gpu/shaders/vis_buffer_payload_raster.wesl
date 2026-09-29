// Full 64-bit vis-buffer twin: composites the shipping `(depth << 32) | payload`
// key through a single 64-bit `atomicMax`.
//
// Where `vis_buffer_software_raster.wesl` composites only the 32-bit depth key
// (the portable core-WGSL subset), this variant reproduces the shipping meshlet
// rasterizer's actual visibility word bit-for-bit: one `atomic<u64>` per texel
// holding `(depth_bits << 32) | payload`, resolved by `atomicMax`. Because the
// reversed-Z depth occupies the high 32 bits, the maximum key is the nearest
// surface and its low 32 bits carry that surface's winning payload for free -
// no separate depth pre-pass, exactly the Nanite-style single-atomic model.
//
// This kernel requires the `SHADER_INT64` and `SHADER_INT64_ATOMIC_MIN_MAX`
// device features; `GpuPayloadRaster::new` only compiles it after
// `GpuContext::supports_u64_atomics` confirms both are enabled.
//
// Coverage arithmetic (signed edge function, winding swap, top-left fill rule)
// and the reversed-Z depth encode are identical to the depth-only twin and to
// the CPU golden `rasterize_triangle`, so the covered texel set is bit-exact
// and the packed word matches the CPU `pack_vis(encode_depth(d), payload)`.
//
// Provenance: standard signed-edge / top-left-rule software rasterization; no
// Unreal Engine source or derived code.

struct Params {
    // Framebuffer width in pixels.
    width: u32,
    // Framebuffer height in pixels.
    height: u32,
    // Number of valid triangles in the `tris` storage array.
    tri_count: u32,
    // Non-zero requests back-face culling of the original winding.
    cull_back: u32,
}

// One triangle in y-down pixel space: nine tightly packed `f32`s for the three
// vertices' positions and reversed-Z depths, then the 32-bit payload written
// into the low bits of every covered texel. Matches the `bytemuck` upload
// struct (`40`-byte stride) exactly.
struct Tri {
    x0: f32, y0: f32, d0: f32,
    x1: f32, y1: f32, d1: f32,
    x2: f32, y2: f32, d2: f32,
    payload: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tris: array<Tri>;
@group(0) @binding(2) var<storage, read_write> vis: array<atomic<u64>>;

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

    let tri = tris[t];
    var v0x = tri.x0;
    var v0y = tri.y0;
    var d0 = tri.d0;
    var v1x = tri.x1;
    var v1y = tri.y1;
    var d1 = tri.d1;
    var v2x = tri.x2;
    var v2y = tri.y2;
    var d2 = tri.d2;

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

    // Compose the shipping visibility word: reversed-Z depth in the high 32
    // bits, payload in the low 32. A single `atomicMax` resolves nearest depth
    // and records the winner's payload together.
    let key: u64 = (u64(depth_key) << 32u) | u64(tri.payload);

    let idx = y * params.width + x;
    atomicMax(&vis[idx], key);
}
