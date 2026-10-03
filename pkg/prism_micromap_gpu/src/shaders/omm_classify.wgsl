// Opacity-micromap tri-state classification kernel.
//
// One invocation classifies one micro-triangle. The micro-triangle index is
// decoded into Prism's canonical row-major triangle-strip (a, b, upright)
// coordinates, mapped to three base-triangle UVs, and classified by sampling
// six barycentric anchors plus a uniform centroid grid against a nearest-texel
// alpha mask. The result is written as a 2-bit opacity code (0 transparent,
// 1 opaque, 2 unknown-transparent, 3 unknown-opaque), normalised to the output
// format.
//
// This mirrors `prism_micromap::omm::classify` element for element so the host
// CPU golden and this GPU twin agree. Provenance: classical conservative
// coverage sampling; no Unreal Engine source and no AI/ML.

struct Params {
    ux0: f32,
    uy0: f32,
    ux1: f32,
    uy1: f32,
    ux2: f32,
    uy2: f32,
    n: u32,
    format: u32,
    samples: u32,
    width: u32,
    height: u32,
    wrap: u32,
    threshold: f32,
    micro_count: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> states: array<u32>;

// Applies the wrap mode to one coordinate: 0 = Repeat, 1 = Clamp.
fn wrap_apply(coord: f32) -> f32 {
    if params.wrap == 1u {
        return clamp(coord, 0.0, 1.0);
    }
    let fr = coord - floor(coord);
    if fr < 0.0 {
        return fr + 1.0;
    }
    return fr;
}

// Converts a wrapped coordinate in [0, 1] to a texel index on `size` texels.
fn coord_to_texel(coord: f32, size: u32) -> u32 {
    let scaled = floor(coord * f32(size));
    if scaled <= 0.0 {
        return 0u;
    }
    return min(u32(scaled), size - 1u);
}

// Nearest-texel opacity test at base-triangle UV `(u, v)`.
fn is_opaque(u: f32, v: f32) -> bool {
    let wu = wrap_apply(u);
    let wv = wrap_apply(v);
    let x = coord_to_texel(wu, params.width);
    let y = coord_to_texel(wv, params.height);
    let idx = y * params.width + x;
    return mask[idx] >= params.threshold;
}

// Interpolates a UV over the three micro-triangle corners.
fn interp_uv(m0: vec2<f32>, m1: vec2<f32>, m2: vec2<f32>, b0: f32, b1: f32, b2: f32) -> vec2<f32> {
    return vec2<f32>(
        m0.x * b0 + m1.x * b1 + m2.x * b2,
        m0.y * b0 + m1.y * b1 + m2.y * b2,
    );
}

// Maps integer lattice weights `(w0, w1, w2)` (summing to n) to a base UV.
fn lattice_uv(w0: u32, w1: u32, w2: u32, inv_n: f32) -> vec2<f32> {
    let b0 = f32(w0) * inv_n;
    let b1 = f32(w1) * inv_n;
    let b2 = f32(w2) * inv_n;
    return vec2<f32>(
        params.ux0 * b0 + params.ux1 * b1 + params.ux2 * b2,
        params.uy0 * b0 + params.uy1 * b1 + params.uy2 * b2,
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.micro_count {
        return;
    }

    let n = params.n;

    // Decode the canonical micro-triangle index into row-strip coordinates.
    var remaining = idx;
    var b = 0u;
    loop {
        let len = 2u * (n - b) - 1u;
        if remaining < len {
            break;
        }
        remaining -= len;
        b += 1u;
    }
    let a = remaining / 2u;
    let upright = (remaining % 2u) == 0u;

    // Three integer lattice vertices (w0, w1, w2) with w0 = n - w1 - w2.
    var v0x: u32;
    var v0y: u32;
    var v1x: u32;
    var v1y: u32;
    var v2x: u32;
    var v2y: u32;
    if upright {
        v0x = a;        v0y = b;
        v1x = a + 1u;   v1y = b;
        v2x = a;        v2y = b + 1u;
    } else {
        v0x = a + 1u;   v0y = b;
        v1x = a;        v1y = b + 1u;
        v2x = a + 1u;   v2y = b + 1u;
    }

    let inv_n = 1.0 / f32(n);
    let m0 = lattice_uv(n - v0x - v0y, v0x, v0y, inv_n);
    let m1 = lattice_uv(n - v1x - v1y, v1x, v1y, inv_n);
    let m2 = lattice_uv(n - v2x - v2y, v2x, v2y, inv_n);

    var opaque = 0u;
    var transparent = 0u;

    // Six fixed anchors: three vertices and three edge midpoints.
    var anchors = array<vec3<f32>, 6>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(0.5, 0.5, 0.0),
        vec3<f32>(0.0, 0.5, 0.5),
        vec3<f32>(0.5, 0.0, 0.5),
    );
    for (var i = 0u; i < 6u; i += 1u) {
        let bc = anchors[i];
        let p = interp_uv(m0, m1, m2, bc.x, bc.y, bc.z);
        if is_opaque(p.x, p.y) {
            opaque += 1u;
        } else {
            transparent += 1u;
        }
    }

    // Uniform centroid grid of an s-segment sub-subdivision.
    var s = params.samples;
    if s < 1u {
        s = 1u;
    }
    let inv_s = 1.0 / f32(s);
    let third = 1.0 / 3.0;
    for (var bb = 0u; bb < s; bb += 1u) {
        for (var aa = 0u; aa < s - bb; aa += 1u) {
            // Upright sub-cell centroid of (aa,bb),(aa+1,bb),(aa,bb+1).
            {
                let s1 = f32(aa + (aa + 1u) + aa) * inv_s * third;
                let s2 = f32(bb + bb + (bb + 1u)) * inv_s * third;
                let s0 = 1.0 - s1 - s2;
                let p = interp_uv(m0, m1, m2, s0, s1, s2);
                if is_opaque(p.x, p.y) {
                    opaque += 1u;
                } else {
                    transparent += 1u;
                }
            }
            if aa < s - 1u - bb {
                // Inverted sub-cell centroid of (aa+1,bb),(aa,bb+1),(aa+1,bb+1).
                let s1 = f32((aa + 1u) + aa + (aa + 1u)) * inv_s * third;
                let s2 = f32(bb + (bb + 1u) + (bb + 1u)) * inv_s * third;
                let s0 = 1.0 - s1 - s2;
                let p = interp_uv(m0, m1, m2, s0, s1, s2);
                if is_opaque(p.x, p.y) {
                    opaque += 1u;
                } else {
                    transparent += 1u;
                }
            }
        }
    }

    // Resolve the tri-state coverage into a 2-bit opacity code.
    var code: u32;
    if opaque == 0u && transparent == 0u {
        code = 0u;
    } else if transparent == 0u {
        code = 1u;
    } else if opaque == 0u {
        code = 0u;
    } else if opaque >= transparent {
        code = 3u;
    } else {
        code = 2u;
    }

    // Normalise to the output format: 0 = TwoState, 1 = FourState.
    if params.format == 0u {
        if code == 0u || code == 2u {
            code = 0u;
        } else {
            code = 1u;
        }
    }

    states[idx] = code;
}
