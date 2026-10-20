struct Axes { origin: vec4<f32>, ends: array<vec4<f32>, 3> }
@group(0) @binding(0) var<uniform> axes: Axes;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let colors = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.18, 0.12), vec3<f32>(0.2, 1.0, 0.25), vec3<f32>(0.2, 0.45, 1.0));
    for (var i = 0u; i < 3u; i++) {
        let a = axes.origin.xy;
        let b = axes.ends[i].xy;
        let d = b - a;
        let t = clamp(dot(p.xy - a, d) / dot(d, d), 0.0, 1.0);
        if distance(p.xy, a + t * d) <= 2.0 * axes.origin.w || distance(p.xy, b) <= 4.0 * axes.origin.w {
            if axes.origin.z == f32(i) { return vec4<f32>(1.0, 0.9, 0.15, 1.0); }
            return vec4<f32>(colors[i], 1.0);
        }
    }
    discard;
}
