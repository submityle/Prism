// Compute twin of Prism's forward-scatter crossing accumulation
// (prism_render_architecture::hair::dual_scattering::accumulate_forward_scatter):
// the coverage-weighted count `n` of strands crossed at or in front of a
// receiver depth, which the dual-scattering shading side raises into `a_f^n`
// and `n * beta_f^2`.
//
// The golden, per light ray/texel, sums each sample's opacity (clamped to
// `0..=1`) over every sample whose depth is at or in front of the receiver
// depth. The reduction is ordering-independent and never panics; an empty ray
// yields `0`. Passing an infinitely deep receiver reduces to `total_crossings`
// (the saturation count for a fully buried receiver), which this twin also
// exercises.
//
// The per-ray slicing is a host concern: the host concatenates every ray's
// (depth, opacity) samples into a shared pool and uploads each ray's
// (sample_start, sample_count) run plus its receiver depth. This kernel owns
// exactly the per-ray depth-gated sum: one thread per ray walks its sample
// slice, adds `clamp(opacity, 0, 1)` for each sample at or in front of the
// receiver depth, and writes one crossing count per ray.
//
// # Correctness model
//
// The reduction is a closed-form clamped sum with no transcendental call, so
// CPU and GPU evaluate the same arithmetic. They are not bit-exact: a GPU may
// reassociate the running sum or fuse a multiply-add, perturbing the low
// mantissa bits by a few ULP. The parity test asserts a per-value tolerance
// (`abs_diff < 1e-4` or `rel_diff < 1e-3`) on the crossing count.
//
// # Portability
//
// The kernel uses only compare, `clamp` and add in the portable core-WGSL
// subset -- no `exp`, `pow`, `sin` or optional device feature -- so the twin
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard coverage-weighted depth-gated crossing sum plus a wgpu
// compute dispatch; no Unreal Engine source or derived code.

// Just the ray count; the per-ray and per-sample layout lives in the storage
// buffers. A single `u32` fills (and is padded to) the `16`-byte uniform slot.
struct Params {
    ray_count: u32,
}

// One ray's sample slice descriptor plus its receiver depth: `(sample_start,
// sample_count)` index the shared `samples` pool and `receiver_depth` gates the
// sum. `16`-byte layout matching the host `GpuRay`.
struct Ray {
    sample_start: u32,
    sample_count: u32,
    receiver_depth: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flattened `(depth, opacity)` pairs, concatenated across every ray; stride 2.
@group(0) @binding(1) var<storage, read> samples: array<f32>;
// Per ray: `(sample_start, sample_count, receiver_depth, pad)`.
@group(0) @binding(2) var<storage, read> rays: array<Ray>;
// One coverage-weighted crossing count per ray, ray-index aligned.
@group(0) @binding(3) var<storage, read_write> out_n: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= params.ray_count) {
        return;
    }
    let ray = rays[r];

    // Coverage-weighted crossings accumulate only for samples at or in front of
    // the receiver depth; the sum is ordering-independent and never negative.
    var crossings: f32 = 0.0;
    for (var j = 0u; j < ray.sample_count; j = j + 1u) {
        let s = ray.sample_start + j;
        let depth = samples[s * 2u];
        let opacity = samples[s * 2u + 1u];
        if (depth <= ray.receiver_depth) {
            crossings = crossings + clamp(opacity, 0.0, 1.0);
        }
    }
    out_n[r] = crossings;
}
