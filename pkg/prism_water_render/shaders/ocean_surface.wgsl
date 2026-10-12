// Tessendorf FFT ocean-surface shader for Prism.
//
// Unlike the procedural Gerstner demo, this stage consumes a real vertex buffer
// whose positions and normals were synthesised on the host from an inverse-FFT
// ocean surface (prism_render_architecture spectral chain). Each vertex also
// carries a Jacobian-fold foam weight so sharp, pinched crests read as
// whitecaps the way Crest / WaveWorks shade them. The fragment stage blends a
// deep-water colour and a sky colour through a Schlick-Fresnel term, adds an
// analytic sun specular highlight, and layers the foam on top. Only portable
// core WGSL is used.

struct Uniforms {
    view_proj: mat4x4<f32>,
    // xyz = camera world position, w = water-architecture version
    camera_pos: vec4<f32>,
    // xyz = direction toward the sun, w = foam strength
    sun_dir: vec4<f32>,
    deep_color: vec4<f32>,
    sky_color: vec4<f32>,
    foam_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) foam: f32,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) foam: f32,
) -> VsOut {
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(position, 1.0);
    out.world_pos = position;
    out.normal = normal;
    out.foam = foam;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let v = normalize(u.camera_pos.xyz - in.world_pos);
    let n_dot_v = max(dot(n, v), 0.0);

    // Schlick-Fresnel: calm water reflects the sky at grazing angles.
    let f0 = 0.02;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - n_dot_v, 5.0);
    let base = mix(u.deep_color.rgb, u.sky_color.rgb, clamp(fresnel, 0.0, 1.0));

    // Analytic sun specular (Blinn-Phong half vector).
    let sun = normalize(u.sun_dir.xyz);
    let h = normalize(v + sun);
    let spec = pow(max(dot(n, h), 0.0), 220.0);
    let sun_color = vec3<f32>(1.0, 0.97, 0.88);

    // Height tint so crests read brighter than troughs.
    let crest = clamp(in.world_pos.y * 0.10 + 0.5, 0.0, 1.0);
    let tinted = base * (0.82 + 0.18 * crest);

    // Jacobian-fold whitecaps on top of the water colour.
    let foam = clamp(in.foam * u.sun_dir.w, 0.0, 1.0);
    let watered = mix(tinted, u.foam_color.rgb, foam);

    let color = watered + sun_color * (spec * 0.9 * (1.0 - foam));
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
