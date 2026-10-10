//! Shared expensive-lobe sample-budget layout and validators; no bindings or
//! entry points.
//!
//! GPU twin of `prism_render_shading`'s `quality::LobeSampleBudget`. The host
//! turns a single `ShadingQuality` tier into these five sample counts and
//! uploads them as an ordinary pass uniform (never a pipeline specialization
//! key), so quality is a smooth per-frame knob that scales how many samples
//! each already-compiled expensive lobe takes — separable SSS taps, Penner
//! pre-integration steps, Burley translucency radial-energy samples, and the
//! sky multiple-scattering direction / ray-march counts — without any shader
//! recompilation, lobe collapse, or per-pixel branching.
//!
//! The struct field order and the model-preserving floors are kept byte-for-
//! byte identical to the CPU golden (`quality.rs`); `lb_clamped` is the exact
//! GPU mirror of `LobeSampleBudget::clamped`, forcing the SSS tap count odd and
//! `>= 3` and lifting every other count to its floor. Keep arithmetic order,
//! constants, and branch structure aligned with the CPU golden.

/// Lowest tap count the symmetric SSS kernel accepts (odd, `>= 3`).
const LB_MIN_SSS_TAPS: u32 = 3u;
/// Lowest pre-integration quadrature step count that keeps the curvature
/// integral stable.
const LB_MIN_PREINTEGRATION_STEPS: u32 = 16u;
/// Lowest translucency radial-energy sample count.
const LB_MIN_TRANSLUCENCY_SAMPLES: u32 = 8u;
/// Lowest multiple-scattering direction-sample count.
const LB_MIN_MULTISCATTER_DIR: u32 = 4u;
/// Lowest multiple-scattering march-step count.
const LB_MIN_MULTISCATTER_MARCH: u32 = 8u;

/// Per-frame sample counts for the expensive shade-time lobes.
///
/// Field order mirrors the CPU `LobeSampleBudget` exactly and must not be
/// reordered without updating the Rust golden.
struct LobeSampleBudget {
    /// Separable SSS screen-space blur taps (odd, `>= 3`).
    sss_separable_taps: u32,
    /// Penner pre-integrated skin curvature-quadrature steps.
    sss_preintegration_steps: u32,
    /// Burley translucency radial-energy integration samples.
    translucency_profile_samples: u32,
    /// Sky multiple-scattering Fibonacci-sphere direction samples.
    multiscatter_dir_samples: u32,
    /// Sky multiple-scattering per-direction ray-march steps.
    multiscatter_march_samples: u32,
}

/// `max(value, floor)`, the GPU mirror of the CPU `clamp_min`.
fn lb_clamp_min(value: u32, floor: u32) -> u32 {
    if (value < floor) {
        return floor;
    }
    return value;
}

/// Force `n` odd and `>= 3`, matching the SSS kernel contract so a budget can
/// never request an even or sub-minimal tap count. Exact mirror of the CPU
/// `LobeSampleBudget::force_odd_at_least_three`.
fn lb_force_odd_at_least_three(n: u32) -> u32 {
    var m: u32 = n;
    if (m < LB_MIN_SSS_TAPS) {
        m = LB_MIN_SSS_TAPS;
    }
    if ((m % 2u) == 0u) {
        m = m + 1u;
    }
    return m;
}

/// Clamp every field to its model-preserving floor, forcing the SSS taps odd.
/// Exact mirror of the CPU `LobeSampleBudget::clamped`; idempotent and a no-op
/// on any tier budget.
fn lb_clamped(b: LobeSampleBudget) -> LobeSampleBudget {
    var out: LobeSampleBudget;
    out.sss_separable_taps = lb_force_odd_at_least_three(b.sss_separable_taps);
    out.sss_preintegration_steps =
        lb_clamp_min(b.sss_preintegration_steps, LB_MIN_PREINTEGRATION_STEPS);
    out.translucency_profile_samples =
        lb_clamp_min(b.translucency_profile_samples, LB_MIN_TRANSLUCENCY_SAMPLES);
    out.multiscatter_dir_samples =
        lb_clamp_min(b.multiscatter_dir_samples, LB_MIN_MULTISCATTER_DIR);
    out.multiscatter_march_samples =
        lb_clamp_min(b.multiscatter_march_samples, LB_MIN_MULTISCATTER_MARCH);
    return out;
}

/// Total sample count across all expensive lobes — a cheap scalar cost proxy,
/// mirroring the CPU `LobeSampleBudget::total_samples`. Realistic per-frame
/// budgets stay far below `u32` overflow, so a plain sum matches the CPU
/// saturating sum bit-for-bit.
fn lb_total_samples(b: LobeSampleBudget) -> u32 {
    return b.sss_separable_taps
        + b.sss_preintegration_steps
        + b.translucency_profile_samples
        + b.multiscatter_dir_samples
        + b.multiscatter_march_samples;
}
