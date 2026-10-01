// Dynamic Follow-The-Leader (DFTL) inextensible strand integration twin:
// advances on-device the same single-pass length solve the CPU golden
// `simulate_guides_dftl` computes, one thread per guide strand.
//
// DFTL (Müller, Kim and Chentanez 2012, the method TressFX popularized) makes a
// guide strand rigidly inextensible in one root-to-tip sweep, then corrects the
// velocity so that one-sweep projection does not pump fake energy into the
// hair. Each guide is an independent poly-line, so the batch is embarrassingly
// parallel: one thread owns one strand's contiguous particle range and runs the
// full substep schedule over it, exactly mirroring `simulate_strand_dftl`
// applied to each strand slice by `simulate_guides_dftl`. Because every thread
// mutates a disjoint particle range (and a disjoint scratch range), the in-place
// state updates are race free without any barrier, and the read-after-write
// ordering inside a strand matches the scalar reference sweep for sweep.
//
// Each substep runs three phases, in this order (mirroring the reference):
//
//   1. predict           -- semi-implicit gravity + velocity damping
//   2. propagate_ftl      -- root->tip length snap onto the rest lengths
//   3. correct_velocities -- Müller's follower-displacement velocity fix
//
// The solve parameters are sanitized on-device exactly as `DftlParams::sanitized`
// does on the host (non-finite / non-positive `dt` -> 1/60, `substeps` -> at
// least 1, `damping` / `correction` clamped to 0..=1 with non-finite falling
// back to 0 / 0.9, `gravity` components made finite), so a poisoned parameter
// set can never make the device diverge from or outrun the reference. The
// derived scalars (`sub_dt`, `inv_sub_dt`, `velocity_retain`, `gravity_step`)
// are then recomputed here with the reference's exact single-op arithmetic.
//
// The per-strand rest-length companion slice is all-or-nothing in the reference
// (`rest_lengths.get(offset..end)` is `Some` for the whole strand or `None`),
// reproduced here as a single `has_rest` flag per strand: when it is `0` the
// length pass is skipped entirely, matching every `rest_lengths.get(i-1)`
// returning `None`.
//
// Every guard is reproduced bit-for-bit: a strand of fewer than two particles
// is a no-op, pinned particles (`inverse_mass <= 0`) are never integrated and
// never moved by the length pass, a (numerically) zero-length segment falls
// back to the fixed `+X` safe axis instead of dividing by ~0, and the tip
// particle receives no follower correction term.
//
// It uses only the portable core-WGSL subset (sqrt via `sqrt`, `dot`, `min`,
// `max`, `clamp`, mul/add) -- no `exp`, no `pow`, no optional feature -- so the
// twin runs unmodified on Metal, Vulkan and DX12. It is not bit-exact with the
// scalar CPU reference: a GPU may fuse a multiply-add the reference leaves
// separate, perturbing the low mantissa bits over the iterated substeps. The
// parity test therefore asserts a per-component tolerance rather than exact
// equality.
//
// Provenance: standard DFTL inextensible-strand integrator (single-pass FTL
// position propagation + velocity correction); no Unreal Engine source or
// derived code.

// Matches `dftl::EPS_LEN_SQ`: vectors whose squared length is at or below this
// are treated as zero-length so normalization falls back to a safe axis.
const EPS_LEN_SQ: f32 = 1.0e-24;
// Matches `dftl::DEFAULT_DT`: the 60 Hz fallback for a rejected time step. Kept
// as a const division so the compiler folds it to the same f32 the host does.
const DEFAULT_DT: f32 = 1.0 / 60.0;
// Matches `dftl::DEFAULT_CORRECTION`: the canonical DFTL damping substituted for
// a non-finite correction fraction.
const DEFAULT_CORRECTION: f32 = 0.9;
// Largest finite f32, used to mirror the golden `is_finite` test without a
// float equality against a literal.
const MAX_FINITE_F32: f32 = 3.40282347e+38;

// Mirror the golden `f32::is_finite`: true only for a non-NaN value inside the
// finite range.
fn is_finite_f32(x: f32) -> bool {
    return x == x && x >= -MAX_FINITE_F32 && x <= MAX_FINITE_F32;
}

// Golden `finite_or`: `x` when finite, else `d`.
fn finite_or(x: f32, d: f32) -> f32 {
    if is_finite_f32(x) {
        return x;
    }
    return d;
}

// Golden `sanitize_unit`: clamp a 0..=1 tuning fraction, substituting `d` for a
// non-finite authoring value.
fn sanitize_unit(x: f32, d: f32) -> f32 {
    if is_finite_f32(x) {
        return clamp(x, 0.0, 1.0);
    }
    return d;
}

// Golden `Vec3::direction_or`: unit vector along `v` using the reference's term
// order (`v * (1 / sqrt(len_sq))`), or `fallback` when `v` is (numerically)
// zero so the result is never NaN.
fn direction_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return fallback;
}

// Solve-wide parameters, uploaded raw and sanitized on-device so the non-finite
// sanitation path is exercised on the GPU too. `32`-byte scalar-packed layout
// matching the `bytemuck` upload struct `Params`.
struct Params {
    // Raw frame time step (seconds); sanitized to DEFAULT_DT if non-finite / <= 0.
    dt: f32,
    // Raw gravity acceleration components (world units/s^2); made finite on-device.
    gx: f32,
    gy: f32,
    gz: f32,
    // Raw velocity damping; sanitized to a 0..=1 fraction.
    damping: f32,
    // Raw DFTL velocity-correction fraction; sanitized to 0..=1.
    correction: f32,
    // Raw substep count; sanitized to at least 1.
    substeps: u32,
    // Number of strand descriptors in `strands`.
    strand_count: u32,
}

// One guide-strand descriptor. `16`-byte stride matching the `bytemuck` upload
// struct `Strand`: the base particle index of this strand in the flat `state`
// pool, its particle count, whether its rest-length companion slice is present,
// and a pad lane for 16-byte alignment.
struct Strand {
    point_offset: u32,
    point_count: u32,
    has_rest: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> strands: array<Strand>;
// Per-particle state, stride `8` f32: position xyz, velocity xyz, inverse_mass,
// rest_length (the length of the segment leaving this particle).
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
// Per-particle substep scratch, stride `6` f32: start_position xyz (the
// position at the start of the substep) and predicted_position xyz (the
// pre-FTL predicted position). Both are needed by the velocity correction.
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

fn load_pos(gi: u32) -> vec3<f32> {
    let b = 8u * gi;
    return vec3<f32>(state[b], state[b + 1u], state[b + 2u]);
}

fn store_pos(gi: u32, v: vec3<f32>) {
    let b = 8u * gi;
    state[b] = v.x;
    state[b + 1u] = v.y;
    state[b + 2u] = v.z;
}

fn load_vel(gi: u32) -> vec3<f32> {
    let b = 8u * gi;
    return vec3<f32>(state[b + 3u], state[b + 4u], state[b + 5u]);
}

fn store_vel(gi: u32, v: vec3<f32>) {
    let b = 8u * gi;
    state[b + 3u] = v.x;
    state[b + 4u] = v.y;
    state[b + 5u] = v.z;
}

fn load_inv_mass(gi: u32) -> f32 {
    return state[8u * gi + 6u];
}

fn load_rest(gi: u32) -> f32 {
    return state[8u * gi + 7u];
}

fn load_start(gi: u32) -> vec3<f32> {
    let b = 6u * gi;
    return vec3<f32>(scratch[b], scratch[b + 1u], scratch[b + 2u]);
}

fn store_start(gi: u32, v: vec3<f32>) {
    let b = 6u * gi;
    scratch[b] = v.x;
    scratch[b + 1u] = v.y;
    scratch[b + 2u] = v.z;
}

fn load_pred(gi: u32) -> vec3<f32> {
    let b = 6u * gi;
    return vec3<f32>(scratch[b + 3u], scratch[b + 4u], scratch[b + 5u]);
}

fn store_pred(gi: u32, v: vec3<f32>) {
    let b = 6u * gi;
    scratch[b + 3u] = v.x;
    scratch[b + 4u] = v.y;
    scratch[b + 5u] = v.z;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= params.strand_count) {
        return;
    }
    let strand = strands[s];
    let offset = strand.point_offset;
    let n = strand.point_count;
    // A strand with fewer than two particles has no segment to constrain.
    if (n < 2u) {
        return;
    }

    // Sanitize the raw parameters, mirroring `DftlParams::sanitized`.
    var dt = params.dt;
    if (!(is_finite_f32(dt) && dt > 0.0)) {
        dt = DEFAULT_DT;
    }
    let substeps = max(params.substeps, 1u);
    let damping = sanitize_unit(params.damping, 0.0);
    let correction = sanitize_unit(params.correction, DEFAULT_CORRECTION);
    let gravity = vec3<f32>(
        finite_or(params.gx, 0.0),
        finite_or(params.gy, 0.0),
        finite_or(params.gz, 0.0),
    );

    // Derived scalars, recomputed with the reference's single-op arithmetic.
    let sub_dt = dt / f32(substeps);
    let inv_sub_dt = 1.0 / sub_dt;
    let velocity_retain = 1.0 - damping;
    let gravity_step = gravity * sub_dt;
    let safe_axis = vec3<f32>(1.0, 0.0, 0.0);

    for (var sub: u32 = 0u; sub < substeps; sub = sub + 1u) {
        // Phase 1: semi-implicit prediction. Record each particle's start
        // position, then advance free particles by damping velocity, adding the
        // gravity impulse and stepping position forward; the predicted position
        // is recorded for the later velocity correction. Pinned particles keep
        // their pose and their predicted position equals their start position.
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let gi = offset + i;
            let pos = load_pos(gi);
            store_start(gi, pos);
            if (load_inv_mass(gi) <= 0.0) {
                store_pred(gi, pos);
            } else {
                let new_velocity = load_vel(gi) * velocity_retain + gravity_step;
                store_vel(gi, new_velocity);
                let predicted = pos + new_velocity * sub_dt;
                store_pos(gi, predicted);
                store_pred(gi, predicted);
            }
        }

        // Phase 2: FTL position propagation from root to tip. Each free child is
        // snapped onto its rest length along the current segment direction; a
        // collapsed segment uses the safe axis. Pinned particles are never moved;
        // a missing rest companion slice (`has_rest == 0`) skips the pass whole.
        if (strand.has_rest == 1u) {
            for (var i: u32 = 1u; i < n; i = i + 1u) {
                let gi = offset + i;
                if (load_inv_mass(gi) <= 0.0) {
                    continue;
                }
                let rest = load_rest(offset + i - 1u);
                let parent = load_pos(offset + i - 1u);
                let child = load_pos(gi);
                let direction = direction_or(child - parent, safe_axis);
                store_pos(gi, parent + direction * max(rest, 0.0));
            }
        }

        // Phase 3: DFTL velocity correction. Each free particle's velocity is set
        // from its net position change over the substep, then corrected by its
        // follower's FTL displacement scaled by `correction / sub_dt`. The tip
        // has no follower and so receives no correction term.
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let gi = offset + i;
            if (load_inv_mass(gi) <= 0.0) {
                continue;
            }
            let base = (load_pos(gi) - load_start(gi)) * inv_sub_dt;
            var follower = vec3<f32>(0.0, 0.0, 0.0);
            if (i + 1u < n) {
                let fi = offset + i + 1u;
                follower = (load_pred(fi) - load_pos(fi)) * (correction * inv_sub_dt);
            }
            store_vel(gi, base + follower);
        }
    }
}
