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

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real auto-exposure passes wire histogram/adaptation
// storage bindings later.
@compute @workgroup_size(64, 1, 1)
fn exposure_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let min_log2 = -10.0;
    let max_log2 = 12.0;

    var bins: array<u32, 64>;
    for (var i = 0u; i < 64u; i = i + 1u) {
        bins[i] = gid.x + i;
    }

    let lum = exposure_luminance(vec3<f32>(0.2, 0.5, 0.3));
    let bin = histogram_luminance_to_bin(lum, min_log2, max_log2, 64u);
    let avg = histogram_average_luminance(&bins, 64u, min_log2, max_log2, 0.5, 0.1);
    let ev_cam = ev100_from_physical(16.0, 1.0 / 125.0, 100.0);
    let exposure = auto_exposure_resolve_exposure(avg, -8.0, 16.0, 0.0);
    let adapted = eye_adaptation_adapt(1.0, avg, 0.016, 3.0, 1.0);

    _ = bin + u32(lum + avg + ev_cam + exposure + adapted);
}
