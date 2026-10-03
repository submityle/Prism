// Water ocean-spectrum inverse-FFT normalization compute kernel
// (entry point `water_fft_normalize`).
//
// The separable 2D inverse `Cooley-Tukey` transform runs its unnormalized
// butterfly along every row and column; the forward and inverse butterflies
// differ only in twiddle sign, so the round trip is scaled up by the element
// count. The inverse must therefore divide the whole grid by `N*N` once, after
// the final butterfly axis, to recover the spatial ocean field. This pass is
// that single scalar scaling: each invocation multiplies its own `(x, y)`
// complex texel by `1 / (n*n)`.
//
// This mirrors the normalization tail of the CPU golden `fft::transform2`
// (the `inverse` branch's `1 / (N*N)` fold) byte-for-byte: `n*n` is a power of
// two, so both the product and its reciprocal are exact in `f32`, and the
// per-lane multiply introduces no rounding. There is no transcendental here,
// nothing for the determinism policy to constrain.
//
// ABI (matches `WaterKernel::FftNormalize.descriptor()`: 2 storage buffers,
// 1 uniform, 0 textures, 8x8 tile, Grid2d domain):
//   @binding(0) src    : array<vec2<f32>>, the n*n complex grid (re, im)
//   @binding(1) dst     : array<vec2<f32>>, the normalized complex grid (written)
//   @binding(2) params  : FftNormalizeParams uniform

struct FftNormalizeParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: FftNormalizeParams;

@compute @workgroup_size(8, 8, 1)
fn water_fft_normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n || gid.y >= params.n) {
        return;
    }
    let total = f32(params.n * params.n);
    let inv = 1.0 / total;
    let idx = gid.y * params.n + gid.x;
    dst[idx] = src[idx] * inv;
}
