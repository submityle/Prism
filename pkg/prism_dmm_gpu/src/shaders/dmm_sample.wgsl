// Displaced-micro-map per-micro-vertex height sampling kernel.
//
// One invocation samples the displacement height at one micro-vertex. The
// canonical lattice index is decoded into integer barycentric weights
// (w0, w1, w2) summing to n == 2^level in the exact row-major order of
// `prism_dmm::subdivision::micro_vertex_at`, mapped to a base-triangle UV with
// the same floating-point accumulation order as the CPU `interp2`, and sampled
// from a dense f32 height texture with the same half-texel-center bilinear
// filter and wrap handling as `prism_dmm::heightmap::TextureDisplacementMap`.
//
// This mirrors the first pass of `prism_dmm::bake_triangle` element for element
// so the host CPU golden and this GPU twin agree. Provenance: classical
// barycentric interpolation and bilinear filtering; no Unreal Engine source and
// no AI/ML.

struct Params {
    ux0: f32,
    uy0: f32,
    ux1: f32,
    uy1: f32,
    ux2: f32,
    uy2: f32,
    n: u32,
    width: u32,
    height: u32,
    wrap: u32,
    mode: u32,
    vertex_count: u32,
    fixed_min: f32,
    fixed_max: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> heightmap: array<f32>;
@group(0) @binding(2) var<storage, read_write> heights: array<f32>;

// Resolves a (possibly out-of-range) integer texel coordinate along one axis of
// `size` texels: wrap 0 = Repeat (Euclidean remainder), 1 = Clamp.
fn resolve(coord: i32, size: u32) -> u32 {
    let size_i = i32(size);
    var wrapped: i32;
    if params.wrap == 0u {
        wrapped = coord % size_i;
        if wrapped < 0 {
            wrapped += size_i;
        }
    } else {
        wrapped = clamp(coord, 0, size_i - 1);
    }
    return u32(wrapped);
}

// Fetches the raw height at an integer texel coordinate, applying the wrap mode
// to both axes.
fn texel(x: i32, y: i32) -> f32 {
    let xi = resolve(x, params.width);
    let yi = resolve(y, params.height);
    return heightmap[yi * params.width + xi];
}

// Bilinearly samples the height texture at normalized coordinate (u, v) using
// the half-texel-center convention, matching the CPU `sample_height`.
fn sample_height(u: f32, v: f32) -> f32 {
    let fx = u * f32(params.width) - 0.5;
    let fy = v * f32(params.height) - 0.5;
    let x0f = floor(fx);
    let y0f = floor(fy);
    let tx = fx - x0f;
    let ty = fy - y0f;
    let x0 = i32(x0f);
    let y0 = i32(y0f);

    let h00 = texel(x0, y0);
    let h10 = texel(x0 + 1, y0);
    let h01 = texel(x0, y0 + 1);
    let h11 = texel(x0 + 1, y0 + 1);

    let top = h00 + (h10 - h00) * tx;
    let bottom = h01 + (h11 - h01) * tx;
    return top + (bottom - top) * ty;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.vertex_count {
        return;
    }

    let n = params.n;

    // Decode the canonical micro-vertex index into lattice coordinates. The
    // outer loop walks row `b`; row `b` holds `n - b + 1` micro-vertices.
    var remaining = idx;
    var b = 0u;
    loop {
        let row_len = n - b + 1u;
        if remaining < row_len {
            break;
        }
        remaining -= row_len;
        b += 1u;
    }
    let a = remaining;
    let w0 = n - a - b;

    let inv_n = 1.0 / f32(n);
    let bary0 = f32(w0) * inv_n;
    let bary1 = f32(a) * inv_n;
    let bary2 = f32(b) * inv_n;

    let u = params.ux0 * bary0 + params.ux1 * bary1 + params.ux2 * bary2;
    let v = params.uy0 * bary0 + params.uy1 * bary1 + params.uy2 * bary2;

    heights[idx] = sample_height(u, v);
}
