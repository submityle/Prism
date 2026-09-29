// Virtual-shadow-map receiver generation.
//
// GPU twin of the CPU golden prism_render_shading::shadow::virtual_sm
// receiver_gen (reconstruct_world_position + ReceiverProjection::project).
// This is the first stage of the VSM page pipeline: it reads the camera depth
// buffer, reconstructs each visible pixel's world position from the inverse
// view-projection, projects it onto the light's clipmap plane and writes a
// VsmReceiver the following vsm_page_mark pass consumes to request pages.
//
// The arithmetic mirrors the golden line for line so the CPU allocator
// reference and this GPU pass request the same page set:
//   * Bevy screen conventions -- UV origin top-left with y growing downward, so
//     it is flipped to the y-up NDC the matrix expects; the depth sample is
//     passed straight through so any convention (incl. reverse-Z) round-trips
//     through the matrix that produced it;
//   * absolute clipmap-plane projection -- light_space = (dot(world, right),
//     dot(world, up)), a function of world position only, so a static surface
//     keeps the same page as the camera pans (the cache-stability invariant);
//   * view distance -- Euclidean camera-to-world distance, which selects the
//     clip level in the downstream page-mark pass.
//
// Self-contained (no imports); group 0 is
//   0 = depth      (camera device depth, R32Float, non-linear reverse-Z)
//   1 = receivers  (output receiver records, storage read/write)
//   2 = params     (inverse view-projection + light basis + camera, uniform)

// One shadow receiver in the light's clipmap plane; byte-for-byte twin of the
// CPU golden Receiver and identical to vsm_page_mark's VsmReceiver.
struct VsmReceiver {
    // Receiver position in the light-space clipmap plane (world units).
    light_space_x: f32,
    light_space_y: f32,
    // Positive view-space distance; selects the clip level (finer when nearer).
    view_distance: f32,
    // Soft-shadow filter kernel half-width in shadow texels.
    filter_radius_texels: f32,
}

// Per-view constants: the inverse view-projection used to unproject depth, the
// light's orthonormal clipmap basis, the camera position (view-distance origin)
// and the receiver footprint width every pixel inherits.
struct VsmReceiverGenParams {
    inverse_view_proj: mat4x4<f32>,
    // Clipmap plane basis (unit, perpendicular to the light direction).
    light_right: vec3<f32>,
    filter_radius_texels: f32,
    light_up: vec3<f32>,
    viewport_width: u32,
    // Camera world position; view distance is measured from here.
    camera_world: vec3<f32>,
    viewport_height: u32,
}

@group(0) @binding(0) var depth: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> receivers: array<VsmReceiver>;
@group(0) @binding(2) var<uniform> params: VsmReceiverGenParams;

// Reconstructs the world-space position of a screen pixel from its UV and the
// depth-buffer sample. Twin of golden reconstruct_world_position: Bevy UV
// (y-down, [0,1]) -> NDC (y-up, [-1,1]), unproject, perspective divide.
fn vsm_reconstruct_world(uv: vec2<f32>, ndc_depth: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let clip = params.inverse_view_proj * vec4<f32>(ndc_x, ndc_y, ndc_depth, 1.0);
    let inv_w = 1.0 / clip.w;
    return clip.xyz * inv_w;
}

// Projects a world position onto the clipmap plane and measures its camera view
// distance, producing the receiver. Twin of ReceiverProjection::project.
fn vsm_project_receiver(world: vec3<f32>) -> VsmReceiver {
    var receiver: VsmReceiver;
    receiver.light_space_x = dot(world, params.light_right);
    receiver.light_space_y = dot(world, params.light_up);
    receiver.view_distance = length(world - params.camera_world);
    receiver.filter_radius_texels = max(params.filter_radius_texels, 0.0);
    return receiver;
}

@compute @workgroup_size(8, 8, 1)
fn vsm_generate_receivers(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.viewport_width || gid.y >= params.viewport_height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));

    // Pixel-centre UV in Bevy's y-down framebuffer convention.
    let dims = vec2<f32>(f32(params.viewport_width), f32(params.viewport_height));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / dims;

    let ndc_depth = textureLoad(depth, coord, 0).x;
    let world = vsm_reconstruct_world(uv, ndc_depth);
    let receiver = vsm_project_receiver(world);

    let index = gid.y * params.viewport_width + gid.x;
    receivers[index] = receiver;
}
