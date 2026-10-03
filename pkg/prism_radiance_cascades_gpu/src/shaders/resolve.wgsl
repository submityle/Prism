// Radiance Cascades — device resolve kernel.
//
// One invocation per cascade-0 probe. Averages the merged directional radiance
// over the probe's angular bins, reproducing `resolve::mean_radiance` from the
// `prism_render_architecture` CPU golden (probe-major row/col/dir layout, mean
// = sum * (1 / angular)). The caller multiplies by TAU on the host for fluence.
//
// Provenance: Sannikov, *Radiance Cascades* (2023). No Unreal Engine source.

struct Interval {
    radiance: vec3<f32>,
    transmittance: f32,
};

struct Mean {
    value: vec3<f32>,
    pad: f32,
};

struct ResolveParams {
    n: u32,
    angular: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: ResolveParams;
@group(0) @binding(1) var<storage, read> cascade0: array<Interval>;
@group(0) @binding(2) var<storage, read_write> out_means: array<Mean>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.n) {
        return;
    }
    let angular = params.angular;
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var dir: u32 = 0u; dir < angular; dir = dir + 1u) {
        sum = sum + cascade0[p * angular + dir].radiance;
    }
    let inv = 1.0 / f32(angular);
    var m: Mean;
    m.value = sum * inv;
    m.pad = 0.0;
    out_means[p] = m;
}
