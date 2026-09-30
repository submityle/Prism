// Wind-velocity twin: evaluates the total wind velocity at a point on-device,
// mirroring the CPU golden `WindField::velocity_at` (in
// `prism_render_architecture::volumetric::weather`).
//
// The velocity combines the base flow `direction * speed` with a
// divergence-free curl perturbation derived from the stream function
// `psi = sin(f x) sin(f y)`, giving the shape `(sin(f x) cos(f y),
// -cos(f x) sin(f y))` with `f = CURL_FREQUENCY`. That shape is scaled by the
// gust `speed * curl_strength * (1 + saturate(local_disturbance))` and added to
// the base flow (design section 9 / §4 curl advection):
//   base = direction * speed
//   gust = speed * curl_strength * (1 + saturate(local_disturbance))
//   curl = (sin(f*px) * cos(f*py), -cos(f*px) * sin(f*py))
//   velocity = base + curl * gust
//
// The CPU golden owns that logic; this kernel is the on-device twin that runs
// one thread per query.
//
// The CPU golden reduces angles with a hand-rolled `sin_approx`/`cos_approx`
// (Taylor polynomial after range reduction to [-PI/2, PI/2], no float
// intrinsic, for cross-target determinism); this kernel mirrors that same
// range reduction and polynomial so the on-device result matches bit-close,
// rather than calling the WGSL `sin`/`cos` builtins. `cos_approx(x)` is
// `sin_approx(x + PI/2)`, exactly as on the CPU. It uses only the portable
// core-WGSL subset, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard divergence-free curl-noise wind advection; no Unreal
// Engine source or derived code.

// Angle constants matching `PI`, `TWO_PI` and `FRAC_PI_2` in the CPU `math`
// module (high-precision literals round to the same f32 bits).
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FRAC_PI_2: f32 = 1.57079632679489661923;
// Matches `CURL_FREQUENCY` in the CPU `weather` module.
const CURL_FREQUENCY: f32 = 0.15;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One velocity query: the wind field parameters plus the sample position and
// per-cell wind disturbance. Matches `GpuQuery` in `src/velocity_at.rs`.
struct Query {
    direction_x: f32,
    direction_y: f32,
    speed: f32,
    curl_strength: f32,
    pos_x: f32,
    pos_y: f32,
    local_disturbance: f32,
    pad0: u32,
}

// One result: the velocity vector. Matches `Res` in `src/velocity_at.rs`.
struct Res {
    x: f32,
    y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Round halves away from zero, matching Rust's `f32::round`. WGSL's native
// `round` rounds ties to even, so it cannot be used here.
fn round_ties_away(v: f32) -> f32 {
    let r = floor(abs(v) + 0.5);
    return select(r, -r, v < 0.0);
}

// Reduce an angle to [-PI, PI], mirroring `math::wrap_pi`.
fn wrap_pi(x: f32) -> f32 {
    let k = round_ties_away(x / TWO_PI);
    return x - k * TWO_PI;
}

// Hand-rolled sine, mirroring `math::sin_approx` bit for bit.
fn sin_approx(x: f32) -> f32 {
    var r = wrap_pi(x);
    if (r > FRAC_PI_2) {
        r = PI - r;
    } else if (r < -FRAC_PI_2) {
        r = -PI - r;
    }
    let x2 = r * r;
    return r * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0))));
}

// Hand-rolled cosine via `cos(x) = sin(x + PI/2)`, mirroring `math::cos_approx`.
fn cos_approx(x: f32) -> f32 {
    return sin_approx(x + FRAC_PI_2);
}

// Total wind velocity at `pos`, mirroring `WindField::velocity_at`.
fn velocity_at(q: Query) -> Res {
    let gust = q.speed * q.curl_strength * (1.0 + saturate_s(q.local_disturbance));
    let ax = CURL_FREQUENCY * q.pos_x;
    let ay = CURL_FREQUENCY * q.pos_y;
    let curl_x = sin_approx(ax) * cos_approx(ay);
    let curl_y = -cos_approx(ax) * sin_approx(ay);

    var out: Res;
    out.x = q.direction_x * q.speed + curl_x * gust;
    out.y = q.direction_y * q.speed + curl_y * gust;
    return out;
}

@compute @workgroup_size(64)
fn velocity_at_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    results[idx] = velocity_at(queries[idx]);
}
