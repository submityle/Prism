// GPU twin of `prism_render_shading::color_grade` (industry-standard HDR colour
// grading in the scene-referred linear domain, applied before the display
// tone-map curve: von Kries white balance, ASC CDL lift/gamma/gain, contrast
// about a pivot and a luma-preserving saturation).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// exposure.wesl) so the WESL compile test can validate it standalone. Every
// function mirrors the CPU golden arm-for-arm: same operation order, same
// constants, matrix products written out per component (not `mat3x3`, to avoid
// column-major ambiguity) and lerps written `a + (b - a) * t` rather than `mix`
// so the GPU twin stays bit-identical with the golden. `pow(vec3, vec3)` and
// `max(vec3, vec3)` are componentwise in WGSL, matching the per-channel Rust.

// Rec. 709 luma weights (linear sRGB primaries), used by the saturation mix.
const COLOR_GRADE_LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// D65 reference white in the same LMS basis as `color_grade_lin_to_lms`, the
// target of the von Kries adaptation (Unity PPv2 constant).
const COLOR_GRADE_D65_LMS: vec3<f32> = vec3<f32>(0.949237, 1.03542, 1.08728);

// Rec. 709 relative luma of a linear RGB sample.
fn color_grade_luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, COLOR_GRADE_LUMA_WEIGHTS);
}

// Planckian-locus `y` chromaticity approximation `2.87x - 3x^2 - 0.27509507`
// (Unity PPv2 `StandardIlluminantY`).
fn color_grade_standard_illuminant_y(x: f32) -> f32 {
    return 2.87 * x - 3.0 * x * x - 0.27509507;
}

// Linear sRGB -> LMS (Unity PPv2 `LIN_2_LMS_MAT`, row-major, per component).
fn color_grade_lin_to_lms(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        3.90405e-1 * c.x + 5.49941e-1 * c.y + 8.92632e-3 * c.z,
        7.08416e-2 * c.x + 9.63172e-1 * c.y + 1.35775e-3 * c.z,
        2.31082e-2 * c.x + 1.28021e-1 * c.y + 9.36245e-1 * c.z,
    );
}

// LMS -> linear sRGB (Unity PPv2 `LMS_2_LIN_MAT`, row-major, per component).
fn color_grade_lms_to_lin(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        2.85847e0 * c.x + -1.62879e0 * c.y + -2.48910e-2 * c.z,
        -2.10182e-1 * c.x + 1.15820e0 * c.y + 3.24281e-4 * c.z,
        -4.18120e-2 * c.x + -1.18169e-1 * c.y + 1.06867e0 * c.z,
    );
}

// White balance by a von Kries adaptation in LMS space. `temperature` and
// `tint` are normalised knobs; the neutral (0, 0) is the identity to
// floating-point tolerance. Mirrors the CPU golden arm-for-arm.
fn color_grade_white_balance(rgb: vec3<f32>, temperature: f32, tint: f32) -> vec3<f32> {
    let t1 = temperature * 10.0 / 6.0;
    let t2 = tint * 10.0 / 6.0;

    // Reference-white chromaticity offset from D65 along the Planckian locus.
    let scale = select(0.05, 0.1, t1 < 0.0);
    let x = 0.31271 - t1 * scale;
    let y = color_grade_standard_illuminant_y(x) + t2 * 0.05;

    // CIExy -> LMS at unit luminance.
    let cie_y = 1.0;
    let cie_x = cie_y * x / y;
    let cie_z = cie_y * (1.0 - x - y) / y;
    let lms_l = 0.7328 * cie_x + 0.4296 * cie_y - 0.1624 * cie_z;
    let lms_m = -0.7036 * cie_x + 1.6975 * cie_y + 0.0061 * cie_z;
    let lms_s = 0.0030 * cie_x + 0.0136 * cie_y + 0.9834 * cie_z;

    // Diagonal von Kries scale toward the D65 white point.
    let balance = COLOR_GRADE_D65_LMS / vec3<f32>(lms_l, lms_m, lms_s);
    let lms = color_grade_lin_to_lms(rgb);
    let adapted = lms * balance;
    return color_grade_lms_to_lin(adapted);
}

// ASC CDL lift / gamma / gain per channel: `out = (in * gain + lift)^gamma`.
// Negative bases are clamped to 0 before the power (ASC CDL convention); the
// neutral lift=0, gamma=1, gain=1 returns the input.
fn color_grade_lift_gamma_gain(rgb: vec3<f32>, lift: vec3<f32>, gamma: vec3<f32>, gain: vec3<f32>) -> vec3<f32> {
    return pow(max(rgb * gain + lift, vec3<f32>(0.0)), gamma);
}

// Saturation as a luma-preserving mix about the Rec. 709 luma:
// `luma + (in - luma) * sat`. sat=1 is the identity; the Rec. 709 luminance is
// preserved for every sat.
fn color_grade_saturation(rgb: vec3<f32>, sat: f32) -> vec3<f32> {
    let l = color_grade_luma(rgb);
    let grey = vec3<f32>(l, l, l);
    return grey + (rgb - grey) * sat;
}

// Linear contrast expansion about `pivot`: `pivot + (in - pivot) * contrast`.
// contrast=1 is the identity; `pivot` is a fixed point for every contrast.
fn color_grade_contrast(rgb: vec3<f32>, contrast: f32, pivot: f32) -> vec3<f32> {
    let p = vec3<f32>(pivot, pivot, pivot);
    return p + (rgb - p) * contrast;
}

// Per-channel additive offset `in + offset`. Provided for completeness; not
// part of the fixed pipeline order. Neutral offset 0 is the identity.
fn color_grade_offset(rgb: vec3<f32>, offset: vec3<f32>) -> vec3<f32> {
    return rgb + offset;
}

// Artist controls for the full grade (mirrors the CPU `ColorGradeParams`; the
// neutral field values give an identity grade).
struct ColorGradeParams {
    lift: vec3<f32>,
    gamma: vec3<f32>,
    gain: vec3<f32>,
    temperature: f32,
    tint: f32,
    contrast: f32,
    pivot: f32,
    saturation: f32,
}

// Apply the full grade in the fixed pipeline order
// white balance -> lift/gamma/gain -> contrast -> saturation.
fn color_grade_apply(rgb: vec3<f32>, params: ColorGradeParams) -> vec3<f32> {
    var c = color_grade_white_balance(rgb, params.temperature, params.tint);
    c = color_grade_lift_gamma_gain(c, params.lift, params.gamma, params.gain);
    c = color_grade_contrast(c, params.contrast, params.pivot);
    return color_grade_saturation(c, params.saturation);
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real grade pass wires the HDR target and the
// params uniform later. References every function so none drifts unused.
@compute @workgroup_size(64, 1, 1)
fn color_grade_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.01;
    let rgb = vec3<f32>(0.15 + seed, 0.4, 0.85);

    let lum = color_grade_luma(rgb);
    let siy = color_grade_standard_illuminant_y(0.31271);
    let lms = color_grade_lin_to_lms(rgb);
    let lin = color_grade_lms_to_lin(lms);
    let wb = color_grade_white_balance(rgb, 0.2, -0.1);
    let lgg = color_grade_lift_gamma_gain(rgb, vec3<f32>(0.02, 0.0, 0.0), vec3<f32>(1.1, 1.0, 0.9), vec3<f32>(1.05, 1.0, 0.95));
    let sat = color_grade_saturation(rgb, 0.8);
    let con = color_grade_contrast(rgb, 1.2, 0.18);
    let off = color_grade_offset(rgb, vec3<f32>(0.1, -0.1, 0.05));

    var params: ColorGradeParams;
    params.lift = vec3<f32>(0.0, 0.0, 0.0);
    params.gamma = vec3<f32>(1.0, 1.0, 1.0);
    params.gain = vec3<f32>(1.0, 1.0, 1.0);
    params.temperature = 0.0;
    params.tint = 0.0;
    params.contrast = 1.0;
    params.pivot = 0.18;
    params.saturation = 1.0;
    let graded = color_grade_apply(rgb, params);

    _ = lum + siy + lin.x + lms.x + wb.x + lgg.x + sat.x + con.x + off.x + graded.x;
}
