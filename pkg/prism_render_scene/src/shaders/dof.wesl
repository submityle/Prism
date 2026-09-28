// GPU twin of `prism_render_shading::dof` (physically-based, thin-lens depth of
// field: circle-of-confusion optics, near/far field separation, mm->pixel CoC
// conversion, a soft-edged bokeh gather weight and the sharp/blurred blend).
// Self-contained (no intra-crate imports, matching bloom.wesl / exposure.wesl)
// so the WESL compile test can validate it standalone. Every function mirrors
// the CPU golden arm-for-arm; lerps are written `a + (b - a) * t` rather than
// `mix`, and squares are written `x * x` rather than `pow`, to stay
// bit-identical with the golden.

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

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real DoF passes wire depth/HDR bindings later.
@compute @workgroup_size(64, 1, 1)
fn dof_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d_o = f32(gid.x) + 100.0;
    let a = dof_aperture_diameter(50.0, 2.8);
    let coc = dof_circle_of_confusion(2000.0, d_o, 50.0, 2.8, 36.0);
    let s = dof_signed_coc(2000.0, d_o, 50.0, 2.8, 36.0);
    let nf = dof_near_field_coc(2000.0, d_o, 50.0, 2.8, 36.0);
    let ff = dof_far_field_coc(2000.0, d_o, 50.0, 2.8, 36.0);
    let px = dof_coc_to_pixels(coc, 36.0, 1920.0);
    let w = dof_bokeh_weight(f32(gid.x), px);
    let bf = dof_blend_factor(px, 32.0, 1.0);
    let blended = dof_apply(vec3<f32>(0.2, 0.4, 0.6), vec3<f32>(1.0, 1.0, 1.0), bf);

    _ = a + coc + s + nf + ff + px + w + bf + blended.x;
}
