// Radiance Cascades — device upward-merge kernel.
//
// One invocation per child (probe, direction) ray record. Each invocation
// bilinearly interpolates the 2x-coarser parent cascade at the child probe's
// world position, angular-averages the four parent sub-bins that subdivide the
// child direction's sector, and composites the child's near interval `over`
// that averaged far interval — mirroring `merge::merge_into` /
// `parent_far` / `bilinear` from the `prism_render_architecture` CPU golden
// operation-for-operation, including the floor/clamp fractional-weight rules.
//
// Provenance: Sannikov, *Radiance Cascades* (2023); Porter-Duff "over" (1984).
// No Unreal Engine source.

struct Interval {
    radiance: vec3<f32>,
    transmittance: f32,
};

struct MergeParams {
    n: u32,
    cols: u32,
    rows: u32,
    angular: u32,
    pcols: u32,
    prows: u32,
    pangular: u32,
    pad0: u32,
    origin_x: f32,
    origin_y: f32,
    child_spacing: f32,
    parent_spacing: f32,
};

struct Clamped {
    lo: u32,
    hi: u32,
    frac: f32,
};

@group(0) @binding(0) var<uniform> params: MergeParams;
@group(0) @binding(1) var<storage, read> parent: array<Interval>;
@group(0) @binding(2) var<storage, read_write> child: array<Interval>;

const ANGULAR_SCALE: u32 = 4u;

fn lerp_interval(a: Interval, b: Interval, t_in: f32) -> Interval {
    let t = clamp(t_in, 0.0, 1.0);
    var o: Interval;
    o.radiance = a.radiance + (b.radiance - a.radiance) * t;
    o.transmittance = a.transmittance + (b.transmittance - a.transmittance) * t;
    return o;
}

fn over_interval(near: Interval, far: Interval) -> Interval {
    var o: Interval;
    o.radiance = near.radiance + near.transmittance * far.radiance;
    o.transmittance = near.transmittance * far.transmittance;
    return o;
}

fn parent_get(x: u32, y: u32, dir: u32) -> Interval {
    let idx = ((y * params.pcols) + x) * params.pangular + dir;
    return parent[idx];
}

fn clamp_axis(v: f32, hi: u32) -> Clamped {
    let lo_f = floor(v);
    var frac = v - lo_f;
    var lo = u32(max(lo_f, 0.0));
    lo = min(lo, hi - 1u);
    let hi_i = min(lo + 1u, hi - 1u);
    if ((v < 0.0) || (lo == hi_i)) {
        frac = 0.0;
    }
    var c: Clamped;
    c.lo = lo;
    c.hi = hi_i;
    c.frac = frac;
    return c;
}

fn bilinear(col: u32, row: u32, dir: u32) -> Interval {
    let world_x = params.origin_x + (f32(col) + 0.5) * params.child_spacing;
    let world_y = params.origin_y + (f32(row) + 0.5) * params.child_spacing;
    let gx = (world_x - params.origin_x) / params.parent_spacing - 0.5;
    let gy = (world_y - params.origin_y) / params.parent_spacing - 0.5;
    let cx = clamp_axis(gx, params.pcols);
    let cy = clamp_axis(gy, params.prows);
    let c00 = parent_get(cx.lo, cy.lo, dir);
    let c10 = parent_get(cx.hi, cy.lo, dir);
    let c01 = parent_get(cx.lo, cy.hi, dir);
    let c11 = parent_get(cx.hi, cy.hi, dir);
    let top = lerp_interval(c00, c10, cx.frac);
    let bot = lerp_interval(c01, c11, cx.frac);
    return lerp_interval(top, bot, cy.frac);
}

fn parent_far(col: u32, row: u32, child_dir: u32) -> Interval {
    var radiance = vec3<f32>(0.0, 0.0, 0.0);
    var transmittance = 0.0;
    for (var k: u32 = 0u; k < ANGULAR_SCALE; k = k + 1u) {
        let pdir = child_dir * ANGULAR_SCALE + k;
        let s = bilinear(col, row, pdir);
        radiance = radiance + s.radiance;
        transmittance = transmittance + s.transmittance;
    }
    let inv = 1.0 / f32(ANGULAR_SCALE);
    var o: Interval;
    o.radiance = radiance * inv;
    o.transmittance = clamp(transmittance * inv, 0.0, 1.0);
    return o;
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
    let near = child[i];
    let far = parent_far(col, row, dir);
    child[i] = over_interval(near, far);
}
