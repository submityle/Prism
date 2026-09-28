// GPU twin of `prism_render_shading::exposure` (physically-based exposure,
// histogram auto-exposure and temporal eye adaptation). Self-contained (no
// intra-crate imports, matching ssgi.wesl / volumetrics.wesl) so the WESL
// compile test can validate it standalone. Every function mirrors the CPU
// golden arm-for-arm; the histogram total is accumulated as f32 here because
// WGSL core has no u64 (the CPU golden sums u64 then converts to f32, which is
// exact for realistic pixel counts).

const METER_CALIBRATION_K: f32 = 12.5;
const MAX_LUMINANCE_FACTOR: f32 = 1.2;
const LUMINANCE_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Rec. 709 relative luminance of a linear RGB radiance sample.
fn exposure_luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, LUMINANCE_WEIGHTS);
}

// EV100 from a physical camera: log2(N^2 / t) - log2(S / 100). Degenerate
// settings fall back to 0.0, mirroring the CPU guard.
fn ev100_from_physical(aperture: f32, shutter_time: f32, iso: f32) -> f32 {
    if (aperture <= 0.0 || shutter_time <= 0.0 || iso <= 0.0) {
        return 0.0;
    }
    return log2(aperture * aperture / shutter_time) - log2(iso / 100.0);
}

// EV100 implied by a measured average luminance: log2(L * 100 / K).
fn ev100_from_average_luminance(average_luminance: f32) -> f32 {
    let l = max(average_luminance, 1.0e-6);
    return log2(l * 100.0 / METER_CALIBRATION_K);
}

// Maximum sensor luminance Lmax = 1.2 * 2^EV100.
fn max_luminance_for_ev100(ev100: f32) -> f32 {
    return MAX_LUMINANCE_FACTOR * exp2(ev100);
}

// Linear exposure multiplier for a given EV100.
fn exposure_from_ev100(ev100: f32) -> f32 {
    return 1.0 / max_luminance_for_ev100(ev100);
}

// --- Histogram (HistogramRange) ---

fn histogram_span(min_log2: f32, max_log2: f32) -> f32 {
    return max(max_log2 - min_log2, 1.0e-6);
}

// Normalised [0,1] histogram position of a linear luminance.
fn histogram_luminance_to_unit(luminance: f32, min_log2: f32, max_log2: f32) -> f32 {
    if (luminance <= 0.0) {
        return 0.0;
    }
    let log2_lum = log2(luminance);
    return clamp((log2_lum - min_log2) / histogram_span(min_log2, max_log2), 0.0, 1.0);
}

// Bin index in [0, bin_count - 1] a luminance falls into.
fn histogram_luminance_to_bin(luminance: f32, min_log2: f32, max_log2: f32, bin_count: u32) -> u32 {
    if (bin_count == 0u) {
        return 0u;
    }
    let unit = histogram_luminance_to_unit(luminance, min_log2, max_log2);
    let scaled = unit * f32(bin_count);
    let idx = u32(scaled);
    return min(idx, bin_count - 1u);
}

// Representative linear luminance of a bin's centre.
fn histogram_bin_center_luminance(bin: u32, min_log2: f32, max_log2: f32, bin_count: u32) -> f32 {
    if (bin_count == 0u) {
        return 0.0;
    }
    let unit = (f32(bin) + 0.5) / f32(bin_count);
    let log2_lum = min_log2 + unit * histogram_span(min_log2, max_log2);
    return exp2(log2_lum);
}

// Count-weighted mean of the surviving bins' centre luminances, discarding the
// darkest `low` and brightest `high` fraction of samples. Mirrors the CPU
// golden including the collapsed-window fall-back to the untrimmed mean.
fn histogram_average_luminance(
    bins: ptr<function, array<u32, 64>>,
    bin_count: u32,
    min_log2: f32,
    max_log2: f32,
    low: f32,
    high: f32,
) -> f32 {
    if (bin_count == 0u) {
        return 0.0;
    }
    var total: f32 = 0.0;
    for (var i = 0u; i < bin_count; i = i + 1u) {
        total = total + f32((*bins)[i]);
    }
    if (total == 0.0) {
        return 0.0;
    }

    let lo_clamped = clamp(low, 0.0, 1.0);
    let hi_clamped = clamp(high, 0.0, 1.0);
    let drop_low = lo_clamped * total;
    let keep_high = total - hi_clamped * total;

    var cumulative: f32 = 0.0;
    var weighted_sum: f32 = 0.0;
    var weight: f32 = 0.0;
    for (var bin = 0u; bin < bin_count; bin = bin + 1u) {
        let count_f = f32((*bins)[bin]);
        let bin_start = cumulative;
        let bin_end = cumulative + count_f;
        cumulative = bin_end;
        let lo = max(bin_start, drop_low);
        let hi = min(bin_end, keep_high);
        let surviving = max(hi - lo, 0.0);
        if (surviving <= 0.0) {
            continue;
        }
        let center = histogram_bin_center_luminance(bin, min_log2, max_log2, bin_count);
        weighted_sum = weighted_sum + center * surviving;
        weight = weight + surviving;
    }

    if (weight <= 0.0) {
        var fallback_sum: f32 = 0.0;
        for (var bin = 0u; bin < bin_count; bin = bin + 1u) {
            fallback_sum = fallback_sum
                + histogram_bin_center_luminance(bin, min_log2, max_log2, bin_count)
                    * f32((*bins)[bin]);
        }
        return fallback_sum / total;
    }

    return weighted_sum / weight;
}

// --- Auto-exposure (AutoExposureSettings) ---

// Clamped, compensated EV100 for a metered average luminance.
fn auto_exposure_resolve_ev100(
    average_luminance: f32,
    min_ev100: f32,
    max_ev100: f32,
    compensation_stops: f32,
) -> f32 {
    var lo = min_ev100;
    var hi = max_ev100;
    if (min_ev100 > max_ev100) {
        lo = max_ev100;
        hi = min_ev100;
    }
    let metered = ev100_from_average_luminance(average_luminance);
    return clamp(metered + compensation_stops, lo, hi);
}

// Target linear exposure multiplier for a metered average luminance.
fn auto_exposure_resolve_exposure(
    average_luminance: f32,
    min_ev100: f32,
    max_ev100: f32,
    compensation_stops: f32,
) -> f32 {
    return exposure_from_ev100(
        auto_exposure_resolve_ev100(average_luminance, min_ev100, max_ev100, compensation_stops),
    );
}

// --- Eye adaptation (EyeAdaptation) ---

// Eases `current` toward `target` with an exponential response using the
// brightening speed when the target is brighter, the darkening speed otherwise.
fn eye_adaptation_adapt(
    current: f32,
    target_luminance: f32,
    delta_seconds: f32,
    speed_up: f32,
    speed_down: f32,
) -> f32 {
    if (delta_seconds <= 0.0) {
        return current;
    }
    var speed = speed_down;
    if (target_luminance > current) {
        speed = speed_up;
    }
    speed = max(speed, 0.0);
    let factor = 1.0 - exp(-delta_seconds * speed);
    return current + (target_luminance - current) * factor;
}

// --- GPU passes ---
//
// Two compute entry points drive the auto-exposure loop; both reuse the pure
// maths above so the CPU golden and the on-device kernel stay bit-aligned.
// Following the SSGI-composite precedent the module declares the two passes'
// distinct group-0 resources at overlapping binding slots — each entry only
// touches a consistent subset, so per-entry-point validation accepts the
// module even though the slots differ between passes.

const EXPOSURE_BIN_COUNT: u32 = 64u;

// Immediate block for `build_histogram`. Mirrors `GpuExposureHistogramConfig`
// (the histogram log2 window then the framebuffer extent).
struct HistogramParams {
    min_log2: f32,
    max_log2: f32,
    width: u32,
    height: u32,
}

var<immediate> histogram_params: HistogramParams;

// group 0 for the histogram build: the resolved HDR scene colour (read) and
// the 64-bin log-luminance histogram (atomically accumulated).
@group(0) @binding(0) var histogram_scene_color: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram_bins: array<atomic<u32>, EXPOSURE_BIN_COUNT>;

// One invocation per pixel: bin the pixel's Rec.709 luminance into the shared
// log-luminance histogram. The per-pixel coverage guard drops the ragged tile
// edge so out-of-bounds texels never bias the metering.
@compute @workgroup_size(8, 8, 1)
fn build_histogram(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= histogram_params.width || id.y >= histogram_params.height) {
        return;
    }
    let coord = vec2<i32>(i32(id.x), i32(id.y));
    let rgb = textureLoad(histogram_scene_color, coord, 0).rgb;
    let lum = exposure_luminance(rgb);
    let bin = histogram_luminance_to_bin(
        lum,
        histogram_params.min_log2,
        histogram_params.max_log2,
        EXPOSURE_BIN_COUNT,
    );
    atomicAdd(&histogram_bins[bin], 1u);
}

// Immediate block for `resolve_exposure`. Mirrors `GpuExposureResolveConfig`
// (the histogram window, the percentile trim, the auto-exposure clamp/comp and
// the eye-adaptation speeds + frame delta).
struct ResolveParams {
    min_log2: f32,
    max_log2: f32,
    low_percent: f32,
    high_percent: f32,
    min_ev100: f32,
    max_ev100: f32,
    compensation_stops: f32,
    speed_up: f32,
    speed_down: f32,
    delta_seconds: f32,
    _pad0: f32,
    _pad1: f32,
}

var<immediate> resolve_params: ResolveParams;

// Persistent per-view exposure state, carried across frames so eye adaptation
// integrates. `adapted_exposure` is the multiplier the composite applies;
// `adapted_luminance` is the smoothed metered luminance the next frame eases
// from; `average_luminance` is the raw metered value (diagnostic).
struct ExposureState {
    adapted_exposure: f32,
    adapted_luminance: f32,
    average_luminance: f32,
    _pad0: f32,
}

// group 0 for the resolve: the histogram (read to reduce, then zeroed for the
// next frame) and the persistent exposure state (read the prior adaptation,
// write the new multiplier).
@group(0) @binding(0) var<storage, read_write> resolve_bins: array<u32, EXPOSURE_BIN_COUNT>;
@group(0) @binding(1) var<storage, read_write> resolve_state: ExposureState;

// Single invocation: reduce the histogram to a percentile-trimmed average
// luminance, ease the persistent eye-adaptation state toward it, resolve the
// clamped/compensated exposure multiplier, and zero the histogram so the next
// frame accumulates from empty.
@compute @workgroup_size(1, 1, 1)
fn resolve_exposure() {
    var bins: array<u32, EXPOSURE_BIN_COUNT>;
    for (var i = 0u; i < EXPOSURE_BIN_COUNT; i = i + 1u) {
        bins[i] = resolve_bins[i];
    }

    let average = histogram_average_luminance(
        &bins,
        EXPOSURE_BIN_COUNT,
        resolve_params.min_log2,
        resolve_params.max_log2,
        resolve_params.low_percent,
        resolve_params.high_percent,
    );

    let adapted_luminance = eye_adaptation_adapt(
        resolve_state.adapted_luminance,
        average,
        resolve_params.delta_seconds,
        resolve_params.speed_up,
        resolve_params.speed_down,
    );
    let exposure = auto_exposure_resolve_exposure(
        adapted_luminance,
        resolve_params.min_ev100,
        resolve_params.max_ev100,
        resolve_params.compensation_stops,
    );

    resolve_state.adapted_exposure = exposure;
    resolve_state.adapted_luminance = adapted_luminance;
    resolve_state.average_luminance = average;

    for (var j = 0u; j < EXPOSURE_BIN_COUNT; j = j + 1u) {
        resolve_bins[j] = 0u;
    }
}
