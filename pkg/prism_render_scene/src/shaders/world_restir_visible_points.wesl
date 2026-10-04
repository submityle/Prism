// World-space ReSTIR DI — visible-point producer pass (GPU twin).
//
// One compute invocation per screen *tile*. Each invocation samples the tile's
// centre pixel of the SSR prepass (reverse-Z device depth + packed view-space
// `normal_roughness`), reconstructs the view-space shading point from the
// inverse projection, lifts it (and its normal) into world space, and writes a
// single `InjectPoint` record. The downstream inject pass hashes each of these
// points into its SHARC world cell and claims a reservoir slot, so this pass is
// the screen-space seed of the resident world-space reservoir table: it turns
// the frame's visible geometry into the sparse set of world cells the ReSTIR
// seed/fill passes then light.
//
// Tiling (one point per `tile_size x tile_size` block) keeps the injected point
// count bounded (~129600 @1080p for tile_size 4, under the 131072 default table
// capacity) so the open-addressed hash grid never thrashes; a 4K framebuffer
// simply saturates the table, which the inject probe window handles by dropping
// the overflow rather than evicting a live cell.
//
// View-position reconstruction is the exact inverse projection the SSGI trace
// uses (`reconstruct_view_position`, golden `project_view_to_screen` inverted),
// so a point produced here lands on the identical surface the reflection /
// indirect-diffuse passes see. The packed `normal_roughness` is decoded with
// the same `xyz * 2 - 1` convention the SSGI trace reads it with.
//
// Background sentinel: a tile whose centre pixel is the cleared background
// (reverse-Z device depth <= 0) writes a NaN position. The inject kernel's
// `quantize_position` collapses any NaN position to the origin cell (0, 0, 0),
// where the first-wins open-addressing claim keeps a single slot and every
// other background tile is dropped — so the background never floods the table.

// 32-byte std430 twin of `abi::GpuWorldRestirInjectPoint`, byte-identical to
// the inject kernel's `InjectPoint` (two vec4 lanes: a vec3 payload + one pad
// word each).
struct InjectPoint {
    // world_position.xyz + one pad word.
    position_pad: vec4<f32>,
    // world_normal.xyz + one pad word.
    normal_pad: vec4<f32>,
}

// 160-byte std430 twin of `abi::GpuVisiblePointsParams` (two mat4x4 + two
// vec4<u32> lanes, no implicit padding).
struct VisiblePointsParams {
    // Clip -> view (inverse reverse-Z projection); reconstructs the view-space
    // shading point from a tile-centre device depth.
    view_from_clip: mat4x4<f32>,
    // View -> world; lifts the reconstructed point and its normal into world
    // space for the SHARC hash.
    world_from_view: mat4x4<f32>,
    // (screen_width, screen_height, tiles_x, tiles_y) in texels / tiles.
    screen_tiles: vec4<u32>,
    // (tile_size, point_count, 0, 0): the tile edge in pixels and the dispatch
    // extent / per-invocation bounds check.
    tile_point: vec4<u32>,
}

// SSR prepass reverse-Z device depth (R32Float); 0 is the cleared background.
@group(0) @binding(0) var depth_tex: texture_2d<f32>;
// SSR packed view-space normal + roughness (Rgba16Float); `.xyz * 2 - 1` is the
// unit view-space normal (matching the SSGI trace's decode).
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
// The per-frame visible-point list, one record per tile.
@group(0) @binding(2) var<storage, read_write> points: array<InjectPoint>;

var<immediate> params: VisiblePointsParams;

// Quiet NaN (golden background sentinel); the inject pass maps a NaN position
// to the origin cell and the first-wins claim dedups it.
const QUIET_NAN: f32 = 0x7fc00000;

// Reconstruct the view-space position of `uv` at reverse-Z `device_depth`
// (exact inverse of the golden `project_view_to_screen`).
fn reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = params.view_from_clip * clip;
    return view.xyz / view.w;
}

@compute @workgroup_size(8, 8, 1)
fn visible_points_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tiles_x = params.screen_tiles.z;
    let tiles_y = params.screen_tiles.w;
    if (gid.x >= tiles_x || gid.y >= tiles_y) {
        return;
    }
    let index = gid.y * tiles_x + gid.x;
    let point_count = params.tile_point.y;
    if (index >= point_count) {
        return;
    }

    let screen = params.screen_tiles.xy;
    let tile_size = params.tile_point.x;
    // Tile-centre pixel, clamped to the last valid texel on the right/bottom
    // edges so a partial tile still samples inside the framebuffer.
    let half = tile_size / 2u;
    let px = min(gid.x * tile_size + half, screen.x - 1u);
    let py = min(gid.y * tile_size + half, screen.y - 1u);
    let pixel = vec2<i32>(i32(px), i32(py));

    let device_depth = textureLoad(depth_tex, pixel, 0).x;
    var out: InjectPoint;
    // Reverse-Z: device depth 0 is the cleared far-plane background — no surface
    // to inject. Emit the NaN sentinel so the inject pass folds it into the
    // origin cell and drops it.
    if (!(device_depth > 0.0)) {
        out.position_pad = vec4<f32>(QUIET_NAN, QUIET_NAN, QUIET_NAN, 0.0);
        out.normal_pad = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        points[index] = out;
        return;
    }

    let uv = (vec2<f32>(f32(px), f32(py)) + vec2<f32>(0.5)) / vec2<f32>(f32(screen.x), f32(screen.y));
    let view_pos = reconstruct_view_position(uv, device_depth);
    let world_pos = (params.world_from_view * vec4<f32>(view_pos, 1.0)).xyz;

    // Decode the packed view-space normal exactly as the SSGI trace does, then
    // rotate it into world space with the upper-left 3x3 of `world_from_view`.
    let packed = textureLoad(normal_tex, pixel, 0).xyz;
    let view_normal = normalize(packed * 2.0 - 1.0);
    let c0 = params.world_from_view[0].xyz;
    let c1 = params.world_from_view[1].xyz;
    let c2 = params.world_from_view[2].xyz;
    let world_normal_raw = c0 * view_normal.x + c1 * view_normal.y + c2 * view_normal.z;
    let len = length(world_normal_raw);
    let world_normal = select(vec3<f32>(0.0, 1.0, 0.0), world_normal_raw / len, len > 1e-8);

    out.position_pad = vec4<f32>(world_pos, 0.0);
    out.normal_pad = vec4<f32>(world_normal, 0.0);
    points[index] = out;
}
