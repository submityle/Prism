// ABI mirrors prism_mesh::deformation. Static inputs are uploaded once.
struct Vertex { position: vec4<f32>, normal: vec4<f32>, spans: vec4<u32> }
struct Morph { delta: array<f32, 3>, target_index: u32 }
struct Influence { joint: u32, weight: f32 }
struct Joint { rows: array<vec4<f32>, 6> }
struct Output { position: vec4<f32>, normal: vec4<f32> }
@group(0) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, read> morphs: array<Morph>;
@group(0) @binding(2) var<storage, read> influences: array<Influence>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read> joints: array<Joint>;
@group(0) @binding(5) var<storage, read> corrections: array<Output>;
@group(0) @binding(6) var<storage, read_write> output: array<Output>;
@group(0) @binding(7) var<uniform> params: vec4<u32>;

fn unit_normal(n: vec3<f32>) -> vec3<f32> {
    let scale = max(max(abs(n.x), abs(n.y)), abs(n.z));
    if scale == 0.0 { return vec3<f32>(0.0); }
    let scaled = n / scale;
    return scaled / sqrt(dot(scaled, scaled));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.x { return; }
    let vertex = vertices[i];
    var position = vertex.position;
    var normal = vertex.normal;
    for (var m = 0u; m < vertex.spans.y; m++) {
        let delta = morphs[vertex.spans.x + m];
        position = vec4<f32>(position.xyz + vec3<f32>(delta.delta[0], delta.delta[1], delta.delta[2]) * weights[delta.target_index], 1.0);
    }
    if vertex.spans.w != 0u {
        var skinned = vec3<f32>(0.0);
        var skinned_normal = vec3<f32>(0.0);
        for (var s = 0u; s < vertex.spans.w; s++) {
            let influence = influences[vertex.spans.z + s];
            let joint = joints[influence.joint];
            skinned += vec3<f32>(dot(joint.rows[0], position), dot(joint.rows[1], position), dot(joint.rows[2], position)) * influence.weight;
            skinned_normal += vec3<f32>(dot(joint.rows[3], normal), dot(joint.rows[4], normal), dot(joint.rows[5], normal)) * influence.weight;
        }
        position = vec4<f32>(skinned, 1.0);
        normal = vec4<f32>(skinned_normal, 0.0);
    }
    output[i].position = vec4<f32>(position.xyz + corrections[i].position.xyz, 1.0);
    output[i].normal = vec4<f32>(unit_normal(normal.xyz + corrections[i].normal.xyz), 0.0);
}
