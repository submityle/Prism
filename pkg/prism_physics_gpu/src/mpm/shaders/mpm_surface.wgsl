// Fixed-domain density sampling and deterministic Freudenthal tetrahedra.
// Output order comes from the GPU exclusive scan, never atomic append order.
struct Params {
    origin_dx: vec4<f32>,
    dims_particles: vec4<u32>,
    sizes: vec4<u32>,
    kernel: vec4<f32>,
}
struct Vertex { position: vec4<f32>, normal: vec4<f32> }
struct Draw { count: u32, instances: u32, first: u32, first_instance: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;
@group(0) @binding(3) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(5) var<storage, read_write> draw: Draw;
@group(0) @binding(6) var<storage, read_write> status: atomic<u32>;

const TETS = array<vec4<u32>, 6>(
    vec4(0u,1u,3u,7u), vec4(0u,1u,5u,7u), vec4(0u,2u,3u,7u),
    vec4(0u,2u,6u,7u), vec4(0u,4u,5u,7u), vec4(0u,4u,6u,7u));
// Determinants of the lattice tetrahedra. TRIS is outward for a positive
// canonical tetrahedron; reflected tetrahedra require the opposite winding.
const TET_ORIENTATION = array<i32, 6>(1,-1,-1,1,1,-1);
const EDGES = array<vec2<u32>, 6>(
    vec2(0u,1u), vec2(0u,2u), vec2(0u,3u), vec2(1u,2u), vec2(1u,3u), vec2(2u,3u));
const TRIS = array<array<i32,6>,16>(
    array(-1,-1,-1,-1,-1,-1), array(0,1,2,-1,-1,-1),
    array(0,4,3,-1,-1,-1), array(1,2,4,1,4,3),
    array(1,3,5,-1,-1,-1), array(0,3,5,0,5,2),
    array(0,5,1,0,4,5), array(2,4,5,-1,-1,-1),
    array(2,5,4,-1,-1,-1), array(0,5,4,0,1,5),
    array(0,5,3,0,2,5), array(1,5,3,-1,-1,-1),
    array(1,3,4,1,4,2), array(0,3,4,-1,-1,-1),
    array(0,2,1,-1,-1,-1), array(-1,-1,-1,-1,-1,-1));

fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn node_coord(index: u32) -> vec3<u32> {
    let dims = params.dims_particles.xyz;
    return vec3(index % dims.x, (index / dims.x) % dims.y, index / (dims.x * dims.y));
}
fn node_index(c: vec3<u32>) -> u32 {
    return c.x + params.dims_particles.x * (c.y + params.dims_particles.y * c.z);
}
fn world(c: vec3<u32>) -> vec3<f32> {
    return params.origin_dx.xyz + params.origin_dx.w * vec3<f32>(c);
}
fn corner_node(cell: u32, corner: u32) -> u32 {
    let dims = params.dims_particles.xyz - vec3(1u);
    let c = vec3(cell % dims.x, (cell / dims.x) % dims.y, cell / (dims.x * dims.y));
    return node_index(c + vec3(corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u));
}
fn tet_nodes(cell: u32, tet: u32) -> vec4<u32> {
    let t = TETS[tet];
    return vec4(corner_node(cell,t.x), corner_node(cell,t.y), corner_node(cell,t.z), corner_node(cell,t.w));
}
fn mask_for(nodes: vec4<u32>) -> u32 {
    var mask = 0u;
    for (var v = 0u; v < 4u; v++) {
        if field[nodes[v]] > params.kernel.y { mask |= 1u << v; }
    }
    return mask;
}

@compute @workgroup_size(64)
fn sample_field(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.sizes.x { return; }
    let p = world(node_coord(id.x));
    var density = 0.0;
    for (var i = 0u; i < params.dims_particles.w; i++) {
        let particle = particles[i].xyz;
        if !finite(particle.x) || !finite(particle.y) || !finite(particle.z) {
            atomicOr(&status, 1u);
            continue;
        }
        let delta = p - particle;
        // Reject far points before scaling/squaring, avoiding finite-input
        // overflow. The support is spherical and contained in this box.
        if any(abs(delta) >= vec3(params.kernel.x)) { continue; }
        let q = delta / params.kernel.x;
        let weight = max(0.0, 1.0 - dot(q,q));
        density += weight * weight * weight;
    }
    if !finite(density) { atomicOr(&status, 1u); density = 0.0; }
    field[id.x] = density;
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.sizes.y { return; }
    var count = 0u;
    for (var t = 0u; t < 6u; t++) {
        let table = TRIS[mask_for(tet_nodes(id.x,t))];
        for (var e = 0u; e < 6u; e++) { if table[e] >= 0 { count++; } }
    }
    offsets[id.x] = count;
}

@compute @workgroup_size(1)
fn finalize() {
    let total = offsets[params.sizes.y];
    if total > params.sizes.z { atomicOr(&status, 2u); }
    draw.count = select(0u, total, atomicLoad(&status) == 0u);
    draw.instances = 1u;
    draw.first = 0u;
    draw.first_instance = 0u;
}

fn sample_at(p: vec3<f32>) -> f32 {
    let hi = params.dims_particles.xyz - vec3(1u);
    let q = clamp((p - params.origin_dx.xyz) / params.origin_dx.w, vec3(0.0), vec3<f32>(hi));
    let low = vec3<u32>(floor(q));
    let high = min(low + vec3(1u), hi);
    let a = q - vec3<f32>(low);
    let x00 = mix(field[node_index(low)], field[node_index(vec3(high.x,low.y,low.z))], a.x);
    let x10 = mix(field[node_index(vec3(low.x,high.y,low.z))], field[node_index(vec3(high.x,high.y,low.z))], a.x);
    let x01 = mix(field[node_index(vec3(low.x,low.y,high.z))], field[node_index(vec3(high.x,low.y,high.z))], a.x);
    let x11 = mix(field[node_index(vec3(low.x,high.y,high.z))], field[node_index(high)], a.x);
    return mix(mix(x00,x10,a.y), mix(x01,x11,a.y), a.z);
}
fn outward(p: vec3<f32>) -> vec3<f32> {
    let h = params.origin_dx.w * 0.5;
    // Omit common 1/(2h): only the normalized direction is needed.
    let g = vec3(sample_at(p-vec3(h,0.0,0.0))-sample_at(p+vec3(h,0.0,0.0)),
                 sample_at(p-vec3(0.0,h,0.0))-sample_at(p+vec3(0.0,h,0.0)),
                 sample_at(p-vec3(0.0,0.0,h))-sample_at(p+vec3(0.0,0.0,h)));
    let scale = max(abs(g.x), max(abs(g.y), abs(g.z)));
    if scale == 0.0 { return vec3(0.0,1.0,0.0); }
    return normalize(g / scale);
}
fn crossing(nodes: vec4<u32>, edge: u32) -> Vertex {
    let ends = EDGES[edge];
    // Canonical endpoint ordering gives bit-identical shared-edge positions.
    let a = min(nodes[ends.x], nodes[ends.y]);
    let b = max(nodes[ends.x], nodes[ends.y]);
    let ratio = clamp((params.kernel.y - field[a]) / (field[b] - field[a]), 0.0, 1.0);
    let p = mix(world(node_coord(a)), world(node_coord(b)), ratio);
    return Vertex(vec4(p,0.0), vec4(outward(p),0.0));
}

@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.sizes.y || atomicLoad(&status) != 0u { return; }
    var output = offsets[id.x];
    for (var t = 0u; t < 6u; t++) {
        let nodes = tet_nodes(id.x,t);
        let table = TRIS[mask_for(nodes)];
        for (var i = 0u; i < 6u; i += 3u) {
            if table[i] < 0 { break; }
            let a = crossing(nodes,u32(table[i]));
            var b = crossing(nodes,u32(table[i+1u]));
            var c = crossing(nodes,u32(table[i+2u]));
            // Geometry follows the tetrahedral linear field, not the smooth
            // trilinear shading gradient (which can point the other way).
            // Topological winding also avoids tiny world-space cross products.
            if TET_ORIENTATION[t] < 0 {
                let swap = b; b = c; c = swap;
            }
            vertices[output] = a;
            vertices[output+1u] = b;
            vertices[output+2u] = c;
            output += 3u;
        }
    }
}
