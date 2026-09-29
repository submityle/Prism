// GPU twin of `prism_render_shading::dof` (physically-based, thin-lens depth of
// field: circle-of-confusion optics, near/far field separation, mm->pixel CoC
// conversion, a soft-edged bokeh gather weight and the sharp/blurred blend).
// Self-contained (no intra-crate imports, matching bloom.wesl / exposure.wesl)
// so the WESL compile test can validate it standalone. Every CoC helper mirrors
// the CPU golden arm-for-arm; lerps are written `a + (b - a) * t` rather than
// `mix`, and squares are written `x * x` rather than `pow`, to stay
// bit-identical with the golden.
//
// The module carries three compute entry points wired as a chain of passes:
//
//   * `dof_coc`       reconstructs each pixel's view-space distance from the
//                     reverse-Z device depth, converts it through the thin-lens
//                     optics to a signed circle of confusion and writes the
//                     near/far gather radii (in pixels) into an `rg16float`
//                     target (r = near radius, g = far radius).
//   * `dof_gather`    disk-bokeh gathers the pre-exposed HDR scene colour: a
//                     far-field gather over the receiver's own far disc (gated
//                     so sharp foreground does not bleed into the background)
//                     and a near-field scatter-as-gather (a foreground tap
//                     spreads onto the receiver when its near disc covers it),
//                     composited near-over-far into an `rgba16float` target.
//   * `dof_composite` blends the sharp scene colour toward the blurred image by
//                     the golden `dof_blend_factor` of the pixel's CoC radius
//                     and writes the result to the storage output the dispatch
//                     then copies back over `scene_color`.
//
// Depth is the SSR geometry prepass's reverse-Z device depth, unprojected to a
// linear view distance with `view_from_clip` exactly like `ssgi.wesl`, so the
// golden optics see the same view-space distances the CPU reference was
// validated against (scaled from scene units to millimetres by
// `scene_units_to_mm`, since the golden works in mm while the view space is in
// scene units).

// --- Immediate (push-constant) blocks. One per pass; each entry point
//     references exactly one, mirroring the `#[repr(C)]` structs in `abi.rs`.

// `dof_coc` block (112 bytes): a `mat4x4<f32>` inverse projection then the
// framebuffer extent, the thin-lens camera parameters, the scene-unit->mm scale
// and the near plane, three trailing `f32` pads to the 16-byte alignment the
// matrix forces on the struct.
struct GpuDofCocParams {
    // Clip -> view (inverse projection); reconstructs a linear view-space
    // position from the reverse-Z device depth. Uploaded column-major.
    view_from_clip: mat4x4<f32>,
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<u32>,
    // Distance the lens is focused at, in millimetres (golden `focus_distance`).
    focus_distance_mm: f32,
    // Lens focal length in millimetres (golden `focal_length`).
    focal_length_mm: f32,
    // Aperture f-number / f-stop (golden `aperture_f_stop`).
    aperture_f_stop: f32,
    // Sensor width in millimetres (golden `sensor_size` / `sensor_width_mm`).
    sensor_width_mm: f32,
    // Rendered image width in pixels for the mm->pixel CoC conversion.
    image_width_px: f32,
    // Scene-unit -> millimetre scale (the golden optics are in mm while the
    // reconstructed view distance is in scene units); full-frame default 1000.
    scene_units_to_mm: f32,
    // Positive near-plane distance in front of the camera along -Z.
    near: f32,
    // Padding to the 16-byte immediate alignment the matrix forces.
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// `dof_gather` block (16 bytes): the framebuffer extent, the max CoC radius the
// gather is bounded by and the disk sample count.
struct GpuDofGatherParams {
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<u32>,
    // Largest gather radius (pixels); bounds the near/far disks.
    max_coc_pixels: f32,
    // Disk taps per field (near and far each spiral this many samples).
    sample_count: u32,
}

// `dof_composite` block (16 bytes): the framebuffer extent, the max CoC radius
// the blend normalises against and the global effect scale.
struct GpuDofCompositeParams {
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<u32>,
    // Largest gather radius (pixels) the blend saturates at (golden
    // `max_coc_pixels`).
    max_coc_pixels: f32,
    // Global effect scale in [0, 1]; 0 disables DoF (golden `enabled_scale`).
    enabled_scale: f32,
}

// SSR geometry-prepass reverse-Z device depth (R32Float, 0 = background). Read
// by `dof_coc` to reconstruct the per-pixel view distance.
@group(0) @binding(0) var dof_depth: texture_2d<f32>;
// Per-pixel near/far CoC gather radii in pixels (rg16float, r = near, g = far).
// Written by `dof_coc`.
@group(0) @binding(1) var dof_coc_out: texture_storage_2d<rg16float, write>;
// Pre-exposed HDR scene colour. Read by `dof_gather` (source radiance) and by
// `dof_composite` (the sharp image).
@group(0) @binding(2) var dof_scene_color: texture_2d<f32>;
// Near/far CoC radii read by `dof_gather` and `dof_composite`.
@group(0) @binding(3) var dof_coc_in: texture_2d<f32>;
// Blurred HDR output written by `dof_gather` (rgba16float).
@group(0) @binding(4) var dof_blurred_out: texture_storage_2d<rgba16float, write>;
// Blurred HDR field read by `dof_composite`.
@group(0) @binding(5) var dof_blurred_in: texture_2d<f32>;
// Final composited HDR output written by `dof_composite` (rgba16float), copied
// back over `scene_color` by the dispatch.
@group(0) @binding(6) var dof_out: texture_storage_2d<rgba16float, write>;

var<immediate> coc_params: GpuDofCocParams;
var<immediate> gather_params: GpuDofGatherParams;
var<immediate> composite_params: GpuDofCompositeParams;

// Golden-angle (radians) for the Vogel/Fibonacci disk sample spiral used by the
// bokeh gather. Device-only (the CPU golden defines only the CoC optics, the
// bokeh weight and the blend, not the sampling pattern), so trig is fine here.
const DOF_GOLDEN_ANGLE: f32 = 2.399963229728653;

// ---------------------------------------------------------------------------
// Golden twins (mirror `prism_render_shading::dof` arm-for-arm).
// ---------------------------------------------------------------------------

// Aperture (entrance-pupil) diameter A = f / N. Non-positive f-stop -> 0.
fn dof_aperture_diameter(focal_length_mm: f32, aperture_f_stop: f32) -> f32 {
    if (aperture_f_stop <= 0.0) {
        return 0.0;
    }
    return focal_length_mm / aperture_f_stop;
}

// Circle-of-confusion diameter (mm on sensor):
//   A   = focal_length / aperture_f_stop
//   CoC = | A * f * (d_o - d_f) / (d_o * (d_f - f)) |
// clamped to [0, sensor_size]. Focus plane -> 0; degenerate geometry -> 0.
fn dof_circle_of_confusion(
    focus_distance: f32,
    object_distance: f32,
    focal_length: f32,
    aperture_f_stop: f32,
    sensor_size: f32,
) -> f32 {
    let denom = object_distance * (focus_distance - focal_length);
    if (object_distance <= 0.0 || denom == 0.0) {
        return 0.0;
    }
    let a = dof_aperture_diameter(focal_length, aperture_f_stop);
    let coc = a * focal_length * (object_distance - focus_distance) / denom;
    return clamp(abs(coc), 0.0, max(sensor_size, 0.0));
}

// Signed CoC diameter (mm): negative near (d_o < d_f), positive far. Magnitude
// equals dof_circle_of_confusion; only the sign of (d_o - d_f) is carried.
fn dof_signed_coc(
    focus_distance: f32,
    object_distance: f32,
    focal_length: f32,
    aperture_f_stop: f32,
    sensor_size: f32,
) -> f32 {
    let magnitude = dof_circle_of_confusion(
        focus_distance,
        object_distance,
        focal_length,
        aperture_f_stop,
        sensor_size,
    );
    if (object_distance < focus_distance) {
        return -magnitude;
    }
    return magnitude;
}

// Near-field CoC diameter (mm): blur for objects closer than focus, else 0.
fn dof_near_field_coc(
    focus_distance: f32,
    object_distance: f32,
    focal_length: f32,
    aperture_f_stop: f32,
    sensor_size: f32,
) -> f32 {
    let s = dof_signed_coc(
        focus_distance,
        object_distance,
        focal_length,
        aperture_f_stop,
        sensor_size,
    );
    return max(-s, 0.0);
}

// Far-field CoC diameter (mm): blur for objects beyond focus, else 0.
fn dof_far_field_coc(
    focus_distance: f32,
    object_distance: f32,
    focal_length: f32,
    aperture_f_stop: f32,
    sensor_size: f32,
) -> f32 {
    let s = dof_signed_coc(
        focus_distance,
        object_distance,
        focal_length,
        aperture_f_stop,
        sensor_size,
    );
    return max(s, 0.0);
}

// CoC diameter (mm) -> gather radius (pixels):
//   radius_px = coc_mm / sensor_width_mm * image_width_px * 0.5
// Non-positive sensor width -> 0.
fn dof_coc_to_pixels(coc_mm: f32, sensor_width_mm: f32, image_width_px: f32) -> f32 {
    if (sensor_width_mm <= 0.0) {
        return 0.0;
    }
    return coc_mm / sensor_width_mm * image_width_px * 0.5;
}

// Bokeh gather weight for a tap `sample_offset` px from the disc centre inside a
// disc of radius `coc_radius` px. Weight 1 at centre, soft quadratic edge,
// exactly 0 at/beyond the boundary; non-positive radius -> point sample.
fn dof_bokeh_weight(sample_offset: f32, coc_radius: f32) -> f32 {
    if (coc_radius <= 0.0) {
        if (sample_offset <= 0.0) {
            return 1.0;
        }
        return 0.0;
    }
    let t = clamp(sample_offset / coc_radius, 0.0, 1.0);
    let e = 1.0 - t;
    return e * e;
}

// Blend factor in [0, 1]: normalise the CoC radius by max_coc_pixels and scale
// by enabled_scale (0 disables DoF). Degenerate max -> 0.
fn dof_blend_factor(coc_radius_px: f32, max_coc_pixels: f32, enabled_scale: f32) -> f32 {
    if (max_coc_pixels <= 0.0) {
        return 0.0;
    }
    let normalised = clamp(coc_radius_px / max_coc_pixels, 0.0, 1.0);
    return normalised * clamp(enabled_scale, 0.0, 1.0);
}

// Per-channel lerp of `sharp` toward `blurred` by `factor`.
fn dof_apply(sharp: vec3<f32>, blurred: vec3<f32>, factor: f32) -> vec3<f32> {
    return sharp + (blurred - sharp) * factor;
}

// ---------------------------------------------------------------------------
// Device-only helpers.
// ---------------------------------------------------------------------------

// Reconstruct a view-space position from a UV + reverse-Z device depth via the
// inverse projection (mirrors `ssgi.wesl::reconstruct_view_position`).
fn dof_reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = coc_params.view_from_clip * clip;
    return view.xyz / view.w;
}

// Clamp integer texel coords into [0, size - 1] so gather taps never sample out
// of bounds (the edge texel is repeated, matching a clamp-to-edge sampler).
fn dof_clamp_coords(coords: vec2<i32>, size: vec2<i32>) -> vec2<i32> {
    return clamp(coords, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
}

// ---------------------------------------------------------------------------
// Pass 1: circle-of-confusion prepass.
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn dof_coc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(coc_params.screen_size.x);
    let height = i32(coc_params.screen_size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }

    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(coc_params.screen_size.x),
        (f32(gid.y) + 0.5) / f32(coc_params.screen_size.y),
    );

    let device_depth = textureLoad(dof_depth, pc, 0).r;

    // Reverse-Z: 0 = background/far plane. The inverse projection is degenerate
    // there (w -> 0), so treat the background as an object at optical infinity,
    // which drives the far field to its sensor-clamped asymptote.
    var d_view: f32;
    if (device_depth <= 0.0) {
        d_view = 1.0e9;
    } else {
        let view_pos = dof_reconstruct_view_position(uv, device_depth);
        d_view = max(-view_pos.z, 0.0);
    }

    let object_distance_mm = d_view * coc_params.scene_units_to_mm;

    let signed = dof_signed_coc(
        coc_params.focus_distance_mm,
        object_distance_mm,
        coc_params.focal_length_mm,
        coc_params.aperture_f_stop,
        coc_params.sensor_width_mm,
    );
    let near_mm = max(-signed, 0.0);
    let far_mm = max(signed, 0.0);

    let near_px = dof_coc_to_pixels(near_mm, coc_params.sensor_width_mm, coc_params.image_width_px);
    let far_px = dof_coc_to_pixels(far_mm, coc_params.sensor_width_mm, coc_params.image_width_px);

    textureStore(dof_coc_out, pc, vec4<f32>(near_px, far_px, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Pass 2: disk-bokeh gather (near/far separated, near composited over far).
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn dof_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<i32>(i32(gather_params.screen_size.x), i32(gather_params.screen_size.y));
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= size.x || pc.y >= size.y) {
        return;
    }

    let center = textureLoad(dof_coc_in, pc, 0).rg;
    let center_far = center.g;
    let max_r = max(gather_params.max_coc_pixels, 0.0);
    let taps = max(gather_params.sample_count, 1u);

    // --- Far field: gather within the receiver's own far disc. Sharp taps
    //     (small far radius) are gated out so foreground/midground edges do not
    //     bleed into the defocused background.
    let far_radius = clamp(center_far, 0.0, max_r);
    let center_color = textureLoad(dof_scene_color, pc, 0).rgb;
    let center_weight = dof_bokeh_weight(0.0, far_radius);
    var far_accum = center_color * center_weight;
    var far_wsum = center_weight;
    for (var i = 0u; i < taps; i = i + 1u) {
        let t = (f32(i) + 0.5) / f32(taps);
        let sample_radius = sqrt(t) * far_radius;
        let angle = f32(i) * DOF_GOLDEN_ANGLE;
        let offset = vec2<f32>(cos(angle), sin(angle)) * sample_radius;
        let dist = length(offset);
        let tap = dof_clamp_coords(
            pc + vec2<i32>(i32(round(offset.x)), i32(round(offset.y))),
            size,
        );
        let tap_far = textureLoad(dof_coc_in, tap, 0).g;
        // Gate: a tap only contributes to the background blur if its own far
        // disc reaches the receiver (soft one-pixel ramp), so sharp geometry in
        // front of the background does not smear into it.
        let gate = clamp(tap_far - dist + 1.0, 0.0, 1.0);
        let weight = dof_bokeh_weight(dist, far_radius) * gate;
        far_accum = far_accum + textureLoad(dof_scene_color, tap, 0).rgb * weight;
        far_wsum = far_wsum + weight;
    }
    let far_color = far_accum / max(far_wsum, 1.0e-4);

    // --- Near field: scatter-as-gather over the max disc. A foreground tap
    //     scatters onto the receiver when its near disc radius covers the
    //     distance to it; the near field is composited *over* the far/sharp
    //     midground by its coverage.
    var near_accum = vec3<f32>(0.0);
    var near_wsum = 0.0;
    for (var i = 0u; i < taps; i = i + 1u) {
        let t = (f32(i) + 0.5) / f32(taps);
        let sample_radius = sqrt(t) * max_r;
        let angle = f32(i) * DOF_GOLDEN_ANGLE;
        let offset = vec2<f32>(cos(angle), sin(angle)) * sample_radius;
        let dist = length(offset);
        let tap = dof_clamp_coords(
            pc + vec2<i32>(i32(round(offset.x)), i32(round(offset.y))),
            size,
        );
        let tap_near = textureLoad(dof_coc_in, tap, 0).r;
        if (tap_near >= dist) {
            let weight = dof_bokeh_weight(dist, tap_near);
            near_accum = near_accum + textureLoad(dof_scene_color, tap, 0).rgb * weight;
            near_wsum = near_wsum + weight;
        }
    }
    let near_color = near_accum / max(near_wsum, 1.0e-4);
    // Coverage: how much of the disc's tap budget scattered onto the receiver,
    // normalised so a densely-covered near disc composites fully over.
    let coverage = clamp(near_wsum / (0.5 * f32(taps) + 1.0e-4), 0.0, 1.0);

    let blurred = far_color + (near_color - far_color) * coverage;
    textureStore(dof_blurred_out, pc, vec4<f32>(blurred, 1.0));
}

// ---------------------------------------------------------------------------
// Pass 3: composite the blurred image over the sharp one by the CoC blend.
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn dof_composite(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(composite_params.screen_size.x);
    let height = i32(composite_params.screen_size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }

    let sharp = textureLoad(dof_scene_color, pc, 0);
    let blurred = textureLoad(dof_blurred_in, pc, 0).rgb;
    let coc = textureLoad(dof_coc_in, pc, 0).rg;
    // The blend radius is whichever field this pixel sits in (near or far);
    // exactly one is non-zero away from the focus plane, both are zero at it.
    let radius = max(coc.r, coc.g);
    let factor = dof_blend_factor(
        radius,
        composite_params.max_coc_pixels,
        composite_params.enabled_scale,
    );
    let out_rgb = dof_apply(sharp.rgb, blurred, factor);
    textureStore(dof_out, pc, vec4<f32>(out_rgb, sharp.a));
}
