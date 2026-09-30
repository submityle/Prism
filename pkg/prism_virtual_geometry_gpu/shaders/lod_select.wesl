// LOD-selection twin: chooses per cluster the display and prefetch LOD level
// on-device, mirroring the CPU golden `select_lod`.
//
// A GPU-driven virtual-geometry pipeline picks, per drawn cluster, the coarsest
// LOD whose screen-space-projected geometric error still fits the pixel budget,
// with hysteresis to suppress popping and a prefetch query at a look-ahead
// distance so a fast-approaching camera streams finer pages in time. The CPU
// golden `select_lod` (in `prism_render_architecture::virtual_geometry`) owns
// that decision; this kernel is the on-device twin that runs one thread per
// cluster and writes the same `(level, prefetch_level)` the reference returns.
//
// All clusters in a dispatch share one LOD chain (the common case: every
// cluster of a mesh shares its level chain and differs only by view distance,
// closing speed and previously displayed level). The chain need not be sorted;
// a level is identified by its stored `level` index, not its array slot.
//
// The projected-error expression matches the reference term order
// (`geometric_error.max(0) * focal / max(distance, EPS)`), and every branch is
// an integer level compare or a float budget compare on identical operands, so
// away from a razor-thin budget boundary the emitted level is identical to the
// reference regardless of fused-multiply-add contraction. It uses only the
// portable core-WGSL subset - no optional feature - so this twin runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: standard screen-space-error LOD selection with hysteresis and
// velocity-scaled prefetch; no Unreal Engine source or derived code.

// `f32::EPSILON`, matching the reference's `f32::EPSILON` distance clamp.
const F32_EPSILON: f32 = 1.1920929e-7;
// Sentinel for "no chain level matched", larger than any real level index.
const NO_LEVEL: u32 = 0xffffffffu;

struct Params {
    // Half viewport height / tan(0.5 * vertical fov); a world size times this
    // over view distance is the size in pixels.
    focal_length_pixels: f32,
    // Projected geometric-error budget in pixels.
    target_error_pixels: f32,
    // Dead-band in pixels a level must cross before the selector switches.
    hysteresis_pixels: f32,
    // World-space look-ahead applied per unit of closing speed.
    prefetch_velocity_scale: f32,
    // Number of valid entries in the shared `chain`.
    level_count: u32,
    // Number of clusters in `clusters` / outputs in `selections`.
    cluster_count: u32,
    pad0: u32,
    pad1: u32,
}

// One LOD chain entry. `8`-byte stride matching the `bytemuck` upload struct.
struct ChainLevel {
    // Level index; `0` is finest, larger is coarser.
    level: u32,
    // Object-space geometric deviation bound in world units.
    geometric_error: f32,
}

// One cluster's LOD query inputs. `16`-byte stride matching the upload struct.
struct Cluster {
    // View-space distance to the cluster.
    view_distance: f32,
    // Component of relative velocity reducing that distance (positive =
    // approaching), world units per selection step.
    closing_speed: f32,
    // `1` when `previous_level` holds a valid prior display level, else `0`.
    has_previous: u32,
    // Display level chosen on the prior step (used for hysteresis).
    previous_level: u32,
}

// One cluster's selection result. `8`-byte stride.
struct Selection {
    level: u32,
    prefetch_level: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> chain: array<ChainLevel>;
@group(0) @binding(2) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(3) var<storage, read_write> selections: array<Selection>;

// Projects an object-space `geometric_error` at `view_distance` to pixels,
// mirroring `LodProjection::projected_error_pixels`.
fn projected_error(geometric_error: f32, view_distance: f32) -> f32 {
    let distance = max(view_distance, F32_EPSILON);
    return max(geometric_error, 0.0) * params.focal_length_pixels / distance;
}

// Coarsest chain level whose projected error is within `budget` at `distance`;
// the finest level when none fit. Mirrors `coarsest_within`.
fn coarsest_within(view_distance: f32, budget: f32) -> u32 {
    var finest: u32 = 0u;
    var have_finest = false;
    var best: u32 = NO_LEVEL;
    var have_best = false;
    for (var i = 0u; i < params.level_count; i = i + 1u) {
        let c = chain[i];
        if (!have_finest || c.level < finest) {
            finest = c.level;
            have_finest = true;
        }
        if (projected_error(c.geometric_error, view_distance) <= budget) {
            if (!have_best || c.level > best) {
                best = c.level;
                have_best = true;
            }
        }
    }
    if (have_best) {
        return best;
    }
    return finest;
}

// Projected error of the chain entry whose stored level equals `level`, or
// `NO_LEVEL`-tagged absence. Returns the error in `out_error` and presence in
// the boolean result. Mirrors `error_of`.
fn error_of(view_distance: f32, level: u32, out_error: ptr<function, f32>) -> bool {
    for (var i = 0u; i < params.level_count; i = i + 1u) {
        let c = chain[i];
        if (c.level == level) {
            *out_error = projected_error(c.geometric_error, view_distance);
            return true;
        }
    }
    return false;
}

// Chooses one level at `view_distance`, applying hysteresis against an optional
// `previous`. Mirrors `pick_level`.
fn pick_level(view_distance: f32, has_previous: bool, previous: u32) -> u32 {
    let target_px = max(params.target_error_pixels, 0.0);
    let hysteresis = max(params.hysteresis_pixels, 0.0);
    let desired = coarsest_within(view_distance, target_px);

    if (!has_previous) {
        return desired;
    }
    var previous_error: f32 = 0.0;
    if (!error_of(view_distance, previous, &previous_error)) {
        return desired;
    }
    if (desired == previous) {
        return previous;
    }
    if (desired < previous) {
        // Fresh pick is finer: refine only once `previous` clearly overshoots
        // the relaxed budget.
        if (previous_error > target_px + hysteresis) {
            return desired;
        }
        return previous;
    }
    // Fresh pick is coarser: coarsen only once it sits under the tightened
    // budget.
    var desired_error: f32 = 0.0;
    if (!error_of(view_distance, desired, &desired_error)) {
        // `desired` is always a chain level, so this is unreachable; the golden
        // treats a missing level as +inf, which never clears the tightened
        // budget, i.e. hold `previous`.
        return previous;
    }
    if (desired_error <= max(target_px - hysteresis, 0.0)) {
        return desired;
    }
    return previous;
}

// One thread per cluster.
@compute @workgroup_size(64, 1, 1)
fn select(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cluster_count) {
        return;
    }
    let cl = clusters[i];
    let has_prev = cl.has_previous == 1u;

    let display = pick_level(cl.view_distance, has_prev, cl.previous_level);

    let look_ahead = max(cl.closing_speed, 0.0) * params.prefetch_velocity_scale;
    let prefetch_distance = max(cl.view_distance - look_ahead, 0.0);
    let prefetch = pick_level(prefetch_distance, true, display);

    // A closer prefetch distance can only ask for an equal or finer level.
    selections[i] = Selection(display, min(prefetch, display));
}
