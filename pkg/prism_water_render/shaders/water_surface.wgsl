// Water-surface rasterisation shader for Prism.
//
// The vertex stage synthesises a grid procedurally from the vertex index (no
// vertex buffer), displaces it with a small sum of Gerstner waves, and derives
// the analytic surface normal. The fragment stage shades the surface with a
// Schlick-Fresnel blend between a deep-water colour and a sky colour, plus an
// analytic sun specular highlight. Only portable core WGSL is used.

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    deep_color: vec4<f32>,
    sky_color: vec4<f32>,
    // x = time, y = amplitude, z = grid extent, w = grid resolution (cells/side)
    params: vec4<f32>,
    // x = base wavelength, y = phase-speed scale, z = choppiness, w = unused
    wave: vec4<f32>,
    // xyz = per-wave angular frequency from the shared dispersion relation
    // (evaluated on the host), w = water-architecture version
    wave_omega: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

const PI: f32 = 3.14159265;
const WAVE_COUNT: u32 = 3u;

fn wave_dir(i: u32) -> vec2<f32> {
    // Three fixed, well-spread directions keep the surface anisotropic.
    if (i == 0u) {
        return normalize(vec2<f32>(1.0, 0.15));
    }
    if (i == 1u) {
        return normalize(vec2<f32>(-0.4, 1.0));
    }
    return normalize(vec2<f32>(0.7, -0.6));
}

fn wave_scale(i: u32) -> f32 {
    // Successive waves are shorter and lower amplitude (a tiny spectrum).
    if (i == 0u) {
        return 1.0;
    }
    if (i == 1u) {
        return 0.55;
    }
    return 0.3;
}

fn wave_omega(i: u32) -> f32 {
    // Angular frequency supplied by the host via the shared dispersion
    // relation, so the surface animates with the solver's deep-water physics.
    if (i == 0u) {
        return u.wave_omega.x;
    }
    if (i == 1u) {
        return u.wave_omega.y;
    }
    return u.wave_omega.z;
}

struct Surface {
    position: vec3<f32>,
    normal: vec3<f32>,
};

fn gerstner(base_xz: vec2<f32>) -> Surface {
    let t = u.params.x;
    let amp = u.params.y;
    let chop = u.wave.z;

    var pos = vec3<f32>(base_xz.x, 0.0, base_xz.y);
    var nrm = vec3<f32>(0.0, 1.0, 0.0);

    for (var i: u32 = 0u; i < WAVE_COUNT; i = i + 1u) {
        let s = wave_scale(i);
        let dir = wave_dir(i);
        let wavelength = max(u.wave.x * s, 1.0e-3);
        let k = 2.0 * PI / wavelength;
        let a = amp * s;
        let w = wave_omega(i);
        let phase = k * dot(dir, base_xz) - w * t;
        let c = cos(phase);
        let sn = sin(phase);
        let wa = k * a;

        pos.x = pos.x + chop * a * dir.x * c;
        pos.z = pos.z + chop * a * dir.y * c;
        pos.y = pos.y + a * sn;

        nrm.x = nrm.x - dir.x * wa * c;
        nrm.z = nrm.z - dir.y * wa * c;
        nrm.y = nrm.y - chop * wa * sn;
    }

    return Surface(pos, normalize(nrm));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let res = max(u32(u.params.w), 1u);
    let quad = vi / 6u;
    let corner = vi % 6u;
    let qx = quad % res;
    let qy = quad / res;

    // Two triangles per quad: (0,0)(1,0)(1,1) and (0,0)(1,1)(0,1).
    var ox = 0u;
    var oy = 0u;
    if (corner == 1u) { ox = 1u; oy = 0u; }
    else if (corner == 2u) { ox = 1u; oy = 1u; }
    else if (corner == 3u) { ox = 0u; oy = 0u; }
    else if (corner == 4u) { ox = 1u; oy = 1u; }
    else if (corner == 5u) { ox = 0u; oy = 1u; }

    let gx = f32(qx + ox) / f32(res);
    let gy = f32(qy + oy) / f32(res);
    let extent = u.params.z;
    let base_xz = vec2<f32>((gx - 0.5) * extent, (gy - 0.5) * extent);

    let surf = gerstner(base_xz);

    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(surf.position, 1.0);
    out.world_pos = surf.position;
    out.normal = surf.normal;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let v = normalize(u.camera_pos.xyz - in.world_pos);
    let n_dot_v = max(dot(n, v), 0.0);

    let f0 = 0.02;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - n_dot_v, 5.0);

    let base = mix(u.deep_color.rgb, u.sky_color.rgb, clamp(fresnel, 0.0, 1.0));

    let sun = normalize(u.sun_dir.xyz);
    let h = normalize(v + sun);
    let spec = pow(max(dot(n, h), 0.0), 220.0);
    let sun_color = vec3<f32>(1.0, 0.97, 0.88);

    // Subtle height tint so wave crests read brighter than troughs.
    let crest = clamp(in.world_pos.y * 0.08 + 0.5, 0.0, 1.0);
    let tinted = base * (0.82 + 0.18 * crest);

    let color = tinted + sun_color * (spec * 0.9);
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
