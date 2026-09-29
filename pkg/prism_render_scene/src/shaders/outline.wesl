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

// ---------------------------------------------------------------------------
// Render-world entry point: composite the golden edge coverage over the scene.
// ---------------------------------------------------------------------------

// Immediate (push-constant) block. Mirrors `GpuOutlineParams` in abi.rs
// byte-for-byte: the clip -> view inverse projection leads the block (it
// reconstructs a linear view-space depth from the reverse-Z device depth,
// exactly like dof.wesl / ssgi.wesl), then the packed line colour + strength,
// the four golden edge thresholds, the framebuffer extent, the id-edge flag
// and a trailing pad to the 16-byte immediate alignment the mat4x4 forces.
struct GpuOutlineParams {
    view_from_clip: mat4x4<f32>,
    line_color: vec4<f32>,
    depth_threshold: f32,
    depth_softness: f32,
    normal_threshold: f32,
    normal_softness: f32,
    screen_size: vec2<u32>,
    id_edges: u32,
    _pad0: u32,
}

// group-0 bindings, matching the pipeline's sequential {0, 1, 2, 3} layout.
// 0: the pre-exposed scene colour (read); 1: the outlined output (storage
// write); 2: the SSR reverse-Z device depth (0 = sky); 3: the SSR view-space
// unit normal (sky = 0).
@group(0) @binding(0) var outline_scene_color: texture_2d<f32>;
@group(0) @binding(1) var outline_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var outline_depth: texture_2d<f32>;
@group(0) @binding(3) var outline_normal: texture_2d<f32>;

var<immediate> outline_params: GpuOutlineParams;

// Reconstruct a view-space position from a UV + reverse-Z device depth via the
// inverse projection (mirrors `ssgi.wesl::reconstruct_view_position`).
fn outline_reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = outline_params.view_from_clip * clip;
    return view.xyz / view.w;
}

// Clamp integer texel coords into [0, size - 1] so cross taps never sample out
// of bounds (the edge texel is repeated, matching a clamp-to-edge sampler).
fn outline_clamp_coords(coords: vec2<i32>, size: vec2<i32>) -> vec2<i32> {
    return clamp(coords, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
}

// Linear view-space depth at a texel. Reverse-Z 0 = sky/background, which the
// golden treats as a non-positive depth (a maximal silhouette against the
// background); the inverse projection is degenerate there (w -> 0), so return
// 0.0 without reconstructing, matching the CPU golden's sky sentinel.
fn outline_linear_depth(coord: vec2<i32>, size: vec2<i32>) -> f32 {
    let c = outline_clamp_coords(coord, size);
    let device_depth = textureLoad(outline_depth, c, 0).r;
    if (device_depth <= 0.0) {
        return 0.0;
    }
    let uv = vec2<f32>(
        (f32(c.x) + 0.5) / f32(size.x),
        (f32(c.y) + 0.5) / f32(size.y),
    );
    let view_pos = outline_reconstruct_view_position(uv, device_depth);
    return max(-view_pos.z, 0.0);
}

// View-space unit normal at a texel (sky = 0; the golden normalises
// defensively so a zero tap cannot open a spurious crease).
fn outline_view_normal(coord: vec2<i32>, size: vec2<i32>) -> vec3<f32> {
    let c = outline_clamp_coords(coord, size);
    return textureLoad(outline_normal, c, 0).xyz;
}

@compute @workgroup_size(8, 8, 1)
fn outline_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(outline_params.screen_size.x);
    let height = i32(outline_params.screen_size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }
    let size = vec2<i32>(width, height);

    // Centre tap.
    let center_depth = outline_linear_depth(pc, size);
    let center_normal = outline_view_normal(pc, size);

    // Fixed four-neighbour cross (left/right/up/down), matching the golden.
    let left = pc + vec2<i32>(-1, 0);
    let right = pc + vec2<i32>(1, 0);
    let up = pc + vec2<i32>(0, -1);
    let down = pc + vec2<i32>(0, 1);

    let neighbor_depths = vec4<f32>(
        outline_linear_depth(left, size),
        outline_linear_depth(right, size),
        outline_linear_depth(up, size),
        outline_linear_depth(down, size),
    );

    let nl = outline_view_normal(left, size);
    let nr = outline_view_normal(right, size);
    let nu = outline_view_normal(up, size);
    let nd = outline_view_normal(down, size);
    // Pack the four neighbour normals as the columns of a mat3x4, exactly as
    // `outline_normal_edge` reconstructs them.
    let neighbor_normals = mat3x4<f32>(
        vec4<f32>(nl.x, nr.x, nu.x, nd.x),
        vec4<f32>(nl.y, nr.y, nu.y, nd.y),
        vec4<f32>(nl.z, nr.z, nu.z, nd.z),
    );

    var params: OutlineParams;
    params.depth_threshold = outline_params.depth_threshold;
    params.depth_softness = outline_params.depth_softness;
    params.normal_threshold = outline_params.normal_threshold;
    params.normal_softness = outline_params.normal_softness;
    params.id_edges = outline_params.id_edges;

    // No outline-id G-buffer exists in the prescribed binding layout, so feed
    // equal ids (all 0u): `outline_id_edge` then returns 0 and the id term is
    // inert. This is the hook point for a future material/outline-id buffer —
    // sample it here and pass the centre + neighbour ids through unchanged.
    let neighbor_ids = vec4<u32>(0u, 0u, 0u, 0u);
    let coverage = evaluate_outline(
        0u,
        center_depth,
        center_normal,
        neighbor_ids,
        neighbor_depths,
        neighbor_normals,
        params,
    );

    // Composite the authored line colour over the shaded pixel by the coverage
    // scaled to opacity. Componentwise lerp a + (b - a) * t (not mix) so the
    // blend matches the CPU golden bit-for-bit.
    let scene = textureLoad(outline_scene_color, pc, 0);
    let line_rgb = outline_params.line_color.xyz;
    let t = clamp(coverage * outline_params.line_color.w, 0.0, 1.0);
    let out_r = scene.r + (line_rgb.x - scene.r) * t;
    let out_g = scene.g + (line_rgb.y - scene.g) * t;
    let out_b = scene.b + (line_rgb.z - scene.b) * t;
    textureStore(outline_out, pc, vec4<f32>(out_r, out_g, out_b, scene.a));
}
