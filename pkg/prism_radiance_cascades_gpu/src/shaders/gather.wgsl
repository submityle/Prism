// Radiance Cascades — device gather kernel.
//
// One invocation per (probe, direction) ray record of a single cascade. The
// probe origin is reconstructed from the probe-major linear index exactly as
// the `prism_render_architecture` CPU golden's `Cascade::gather` /
// `probe_position` do, and the direction is read from a host-precomputed
// buffer (host-side `bin_angle` + `Vec2::from_angle`) so the shader performs no
// trigonometry of its own. The scene sampler is a pure `+ - * /` rational of
// the origin, direction, and interval bounds, matching the Rust twin op-for-op.
//
// Provenance: Sannikov, *Radiance Cascades* (2023). No Unreal Engine source.

struct Interval {
    radiance: vec3<f32>,
    transmittance: f32,
};

struct GatherParams {
    n: u32,
    cols: u32,
    rows: u32,
    angular: u32,
    origin_x: f32,
    origin_y: f32,
    spacing: f32,
    t0: f32,
    t1: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> params: GatherParams;
@group(0) @binding(1) var<storage, read> directions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out_intervals: array<Interval>;

fn sample_interval(o: vec2<f32>, d: vec2<f32>, t0: f32, t1: f32) -> Interval {
    let m = o.x * 0.5 + o.y * 0.25 + d.x * 2.0 - d.y * 1.5 + t0 * 0.1;
    let base = m * m;
    let denom = 1.0 + base;
    let tr_raw = 1.0 / denom;
    let tr = clamp(tr_raw, 0.0, 1.0);
    let k = (t1 - t0) * 0.05;
    var out: Interval;
    out.radiance = vec3<f32>(m + k, m * 0.5 + d.x, base * 0.25 + k);
    out.transmittance = tr;
    return out;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let angular = params.angular;
    let dir = i % angular;
    let probe = i / angular;
    let col = probe % params.cols;
    let row = probe / params.cols;
    let s = params.spacing;
    let ox = params.origin_x + (f32(col) + 0.5) * s;
    let oy = params.origin_y + (f32(row) + 0.5) * s;
    let d = directions[dir];
    out_intervals[i] = sample_interval(vec2<f32>(ox, oy), d, params.t0, params.t1);
}
