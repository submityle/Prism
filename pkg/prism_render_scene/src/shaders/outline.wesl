// NPR silhouette/crease outline — GPU twin of `prism_render_shading::outline`.
//
// Mirrors the CPU golden arm-for-arm: for a shaded pixel it reads a fixed
// four-neighbour cross (left/right/up/down) of the geometry buffer and raises
// an edge on any of three discontinuities — outline/material id, relative
// linear depth, or surface-normal turn — then returns their union as a [0, 1]
// outline coverage the composite pass `mix`es toward the authored line colour.
//
// Self-contained (no intra-crate `import`s, matching ssgi.wesl / gtao.wesl) so
// the WESL compile test can validate it standalone with no GPU present.

// Matches `OutlineParams` in outline.rs. `id_edges` is a 0/1 flag (WGSL has no
// bool in a host-shared struct convention here; the CPU mirror is a `bool`).
struct OutlineParams {
    depth_threshold: f32,
    depth_softness: f32,
    normal_threshold: f32,
    normal_softness: f32,
    id_edges: u32,
}

const OUTLINE_DEPTH_EPSILON: f32 = 1.0e-4;

// Self-contained smoothstep matching the CPU twin bit-for-bit: a collapsed or
// inverted window degrades to a hard step at edge0 rather than dividing by
// zero, so a zero half-width yields a crisp edge on CPU and GPU alike.
fn outline_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 <= edge0) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Mirrors `outline_id_edge`: any neighbour with a different id is a hard
// silhouette when id edges are enabled.
fn outline_id_edge(center: u32, neighbors: vec4<u32>, id_edges: u32) -> f32 {
    if (id_edges == 0u) {
        return 0.0;
    }
    var differs = false;
    differs = differs || (neighbors.x != center);
    differs = differs || (neighbors.y != center);
    differs = differs || (neighbors.z != center);
    differs = differs || (neighbors.w != center);
    if (differs) {
        return 1.0;
    }
    return 0.0;
}

// Mirrors `outline_depth_edge`: largest relative depth jump across the cross,
// re-shaped by smoothstep. Non-positive neighbour depth (sky) forces the edge
// fully open; a non-positive centre depth is a background pixel and never
// self-outlines. Uses a large sentinel (bignum) in place of the CPU INFINITY.
fn outline_depth_edge(center: f32, neighbors: vec4<f32>, params: OutlineParams) -> f32 {
    if (center <= 0.0) {
        return 0.0;
    }
    let denom = max(center, OUTLINE_DEPTH_EPSILON);
    var max_rel = 0.0;
    var saw_sky = false;
    var taps = array<f32, 4>(neighbors.x, neighbors.y, neighbors.z, neighbors.w);
    for (var i = 0; i < 4; i = i + 1) {
        let n = taps[i];
        if (n <= 0.0) {
            saw_sky = true;
        } else {
            let rel = abs(center - n) / denom;
            if (rel > max_rel) {
                max_rel = rel;
            }
        }
    }
    if (saw_sky) {
        return 1.0;
    }
    let half = max(params.depth_softness, 0.0);
    return outline_smoothstep(params.depth_threshold, params.depth_threshold + half, max_rel);
}

fn outline_normalize_or_z(v: vec3<f32>) -> vec3<f32> {
    let len2 = dot(v, v);
    if (len2 > 1.0e-12) {
        return v * inverseSqrt(len2);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// Mirrors `outline_normal_edge`: sharpest normal turn 1 - dot(n, n_n) across
// the cross, re-shaped by smoothstep. Neighbours are the four columns of a
// 3x4 matrix so the twin stays a fixed cross like the CPU golden.
fn outline_normal_edge(center: vec3<f32>, neighbors: mat3x4<f32>, params: OutlineParams) -> f32 {
    let cn = outline_normalize_or_z(center);
    var max_turn = 0.0;
    // mat3x4 stores four vec3-like columns; reconstruct each neighbour normal.
    let n0 = vec3<f32>(neighbors[0].x, neighbors[1].x, neighbors[2].x);
    let n1 = vec3<f32>(neighbors[0].y, neighbors[1].y, neighbors[2].y);
    let n2 = vec3<f32>(neighbors[0].z, neighbors[1].z, neighbors[2].z);
    let n3 = vec3<f32>(neighbors[0].w, neighbors[1].w, neighbors[2].w);
    var cols = array<vec3<f32>, 4>(n0, n1, n2, n3);
    for (var i = 0; i < 4; i = i + 1) {
        let nn = outline_normalize_or_z(cols[i]);
        let turn = 1.0 - clamp(dot(cn, nn), -1.0, 1.0);
        if (turn > max_turn) {
            max_turn = turn;
        }
    }
    let half = max(params.normal_softness, 0.0);
    return outline_smoothstep(params.normal_threshold, params.normal_threshold + half, max_turn);
}

// Mirrors `evaluate_outline`: union of the id, depth and normal edges.
fn evaluate_outline(
    center_id: u32,
    center_depth: f32,
    center_normal: vec3<f32>,
    neighbor_ids: vec4<u32>,
    neighbor_depths: vec4<f32>,
    neighbor_normals: mat3x4<f32>,
    params: OutlineParams,
) -> f32 {
    let id_edge = outline_id_edge(center_id, neighbor_ids, params.id_edges);
    let depth_edge = outline_depth_edge(center_depth, neighbor_depths, params);
    let normal_edge = outline_normal_edge(center_normal, neighbor_normals, params);
    return clamp(max(max(id_edge, depth_edge), normal_edge), 0.0, 1.0);
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the render-world pass wires real bindings later.
@compute @workgroup_size(8, 8, 1)
fn outline_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    var params: OutlineParams;
    params.depth_threshold = 0.05;
    params.depth_softness = 0.05;
    params.normal_threshold = 0.2;
    params.normal_softness = 0.2;
    params.id_edges = 1u;

    let ids = vec4<u32>(gid.x, gid.x, gid.x, gid.x);
    let depths = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let normals = mat3x4<f32>(
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 0.0),
        vec4<f32>(1.0, 1.0, 1.0, 1.0),
    );
    let e = evaluate_outline(
        gid.x,
        1.0,
        vec3<f32>(0.0, 0.0, 1.0),
        ids,
        depths,
        normals,
        params,
    );
    _ = e;
}
