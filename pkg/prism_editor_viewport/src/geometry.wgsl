@group(0) @binding(0) var<uniform> view_projection: mat4x4<f32>;
@group(1) @binding(0) var<uniform> model: mat4x4<f32>;
@group(2) @binding(0) var<uniform> identity: vec4<u32>;

@vertex
fn vs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return view_projection * model * vec4<f32>(position, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<u32> {
    return identity;
}
