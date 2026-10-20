struct ParticleInput {
    center_radius: vec4<f32>,
    depth: f32,
    particle_index: u32,
    flags: u32,
    reserved: u32,
}

struct Material {
    scale_softness: vec4<f32>,
    tint: vec4<f32>,
    ribbon_width: vec4<f32>,
    _pad: vec4<u32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> particles: array<ParticleInput>;
@group(0) @binding(1) var<storage, read> visible_indices: array<u32>;
@group(0) @binding(2) var<storage, read> mesh_vertices: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> material: Material;

fn output_at(position: vec2<f32>, uv: vec2<f32>, particle_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let particle = particles[particle_index];
    let center = particle.center_radius.xy;
    let radius = max(particle.center_radius.w, 0.025);
    out.clip_position = vec4<f32>(center + position * radius * material.scale_softness.xy, 0.0, 1.0);
    out.uv = uv;
    out.color = material.tint;
    return out;
}

@vertex
fn vs_sprite(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let point = corners[vertex % 6u];
    return output_at(point, point * 0.5 + vec2<f32>(0.5), visible_indices[instance - material._pad.x]);
}

@vertex
fn vs_mesh(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let mesh_vertex = mesh_vertices[vertex];
    let local = mesh_vertex.xy;
    return output_at(local * 2.0, mesh_vertex.zw, visible_indices[instance - material._pad.x]);
}

fn strip_vertex(vertex: u32, instance: u32, beam: bool) -> VertexOutput {
    let current = visible_indices[instance - material._pad.x];
    let segment = vertex / 4u;
    let side = select(-1.0, 1.0, (vertex & 1u) != 0u);
    let endpoint = (vertex & 2u) != 0u;
    let next = min(current + segment + 1u, arrayLength(&particles) - 1u);
    let a = particles[current + segment].center_radius.xyz;
    let b = particles[next].center_radius.xyz;
    let axis = normalize(b.xy - a.xy + vec2<f32>(0.00001, 0.0));
    let normal = vec2<f32>(-axis.y, axis.x);
    let taper = select(1.0, 0.65, beam && endpoint);
    let half_width = max(particles[current + segment].center_radius.w, material.ribbon_width.x) * taper;
    let base = select(a.xy, b.xy, endpoint);
    let uv = vec2<f32>(f32(segment) + select(0.0, 1.0, endpoint), select(0.0, 1.0, side > 0.0));
    var out: VertexOutput;
    out.clip_position = vec4<f32>(base + normal * side * half_width, 0.0, 1.0);
    out.uv = uv;
    out.color = material.tint;
    return out;
}

@vertex
fn vs_ribbon(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    return strip_vertex(vertex, instance, false);
}

@vertex
fn vs_beam(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    return strip_vertex(vertex, instance, true);
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    let radial = 1.0 - material.scale_softness.z * max(abs(input.uv.y * 2.0 - 1.0), 0.0);
    return vec4<f32>(input.color.rgb * radial, input.color.a);
}
