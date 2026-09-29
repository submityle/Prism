// Portable compute twin of the vis-buffer software rasterizer.
//
// This shader reproduces, texel-for-texel, the coverage and reversed-Z depth
// arithmetic of the CPU golden standard in
// `prism_render_architecture::virtual_geometry::software_raster`. It is written
// in the core WGSL subset (`atomic<u32>` storage buffer + `atomicMax`) so it
// runs unmodified on Metal, Vulkan, and DX12 with no optional device feature.
//
// Fidelity model (depth-only twin):
//   * The shipping meshlet raster packs a 64-bit `(depth << 32) | payload` key
//     into an `r64uint` storage-texture atomic (`textureAtomicMax`), which is a
//     Metal-unsupported extension. This twin instead composites the 32-bit
//     reversed-Z depth key alone through `atomicMax` on a plain storage buffer,
//     the portable subset every backend implements. Carrying the 64-bit payload
//     key is the documented follow-up extension path.
//   * Coverage uses the identical signed edge function, winding swap, and
//     top-left fill rule as the CPU reference, so the covered texel set matches
//     bit-for-bit. Depth is the same screen-space-linear barycentric blend,
//     encoded with `bitcast<u32>(clamp(depth, 0, 1))` == the CPU `encode_depth`.
//   * A cleared pixel is `0` (the farthest reversed-Z key), matching
//     `VisBuffer::new`; the first covering triangle always wins over the clear.
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

// One triangle in y-down pixel space with reversed-Z depth per vertex. Laid out
// as nine tightly packed `f32`s to match the `bytemuck` upload struct exactly.
struct Tri {
    x0: f32, y0: f32, d0: f32,
    x1: f32, y1: f32, d1: f32,
    x2: f32, y2: f32, d2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tris: array<Tri>;
@group(0) @binding(2) var<storage, read_write> depth: array<atomic<u32>>;

// Signed edge function of point `p` against the directed edge `a -> b`.
// Byte-identical to the CPU `edge`: positive on the interior side of a
// positive-area (counter-clockwise, y-down) triangle edge.
fn edge(ax: f32, ay: f32, bx: f32, by: f32, px: f32, py: f32) -> f32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// True when the directed edge `a -> b` is a top or left edge under the y-down,
// positive-area convention. Mirrors the CPU `is_top_left`.
fn is_top_left(ax: f32, ay: f32, bx: f32, by: f32) -> bool {
    let dx = bx - ax;
    let dy = by - ay;
    return dy < 0.0 || (dy == 0.0 && dx > 0.0);
}

// Top-left fill test for one edge value. Mirrors the CPU `covers`.
fn covers(e: f32, tl: bool) -> bool {
    return e > 0.0 || (e == 0.0 && tl);
}

// One thread per (pixel, triangle). `global_id.z` selects the triangle, so
// overlapping triangles composite through genuine `atomicMax` contention on the
// shared depth buffer, faithfully exercising the scatter model the shipping
// meshlet raster uses. Testing coverage over the whole framebuffer is
// equivalent to the CPU's per-triangle bounding-box scan: every covered pixel
// center lies inside the triangle's own bounds, and the coverage predicate is
// identical, so the resulting texel set matches.
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
        // Degenerate triangle: no coverage.
        return;
    }
    if (params.cull_back != 0u && raw_area < 0.0) {
        // Back-facing under the original winding.
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
    let key = bitcast<u32>(clamp(d, 0.0, 1.0));

    let idx = y * params.width + x;
    atomicMax(&depth[idx], key);
}
