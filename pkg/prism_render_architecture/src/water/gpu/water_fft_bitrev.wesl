// Water ocean-spectrum FFT bit-reversal reorder compute kernel
// (entry point `water_fft_bitrev`).
//
// A decimation-in-time radix-2 `Cooley-Tukey` butterfly must first permute its
// input into bit-reversed order so the in-place butterfly stages that follow
// combine the right sub-transforms. On the `GPU` the separable 2D inverse
// transform runs this reorder once per axis as an out-of-place ping-pong pass:
// each invocation writes its own `(x, y)` texel by reading the source element
// whose index along the active axis is bit-reversed. Bit reversal is an
// involution, so reading the reversed source is the same permutation as writing
// the reversed destination, and every output slot is written exactly once.
//
// This mirrors the decimation-in-time reorder step of the CPU golden
// `fft::transform` (the private `reverse_bits` permutation) byte-for-byte: the
// permutation is pure integer index math, so there is no float arithmetic and
// nothing for the determinism policy to constrain.
//
// ABI (matches `WaterKernel::FftBitReverse.descriptor()`: 2 storage buffers,
// 1 uniform, 0 textures, 8x8 tile, Grid2d domain):
//   @binding(0) src    : array<vec2<f32>>, the n*n complex grid (re, im)
//   @binding(1) dst     : array<vec2<f32>>, the reordered complex grid (written)
//   @binding(2) params  : FftBitReverseParams uniform
//
// `axis` selects the transform direction: 0 reorders along rows (the x index),
// 1 reorders along columns (the y index). `log2n` is the bit width to reverse,
// i.e. `n == 1u << log2n`.

struct FftBitReverseParams {
    n: u32,
    log2n: u32,
    axis: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: FftBitReverseParams;

// Reverses the low `bits` bits of `x`, the bit-reversal permutation index.
fn reverse_bits_low(x: u32, bits: u32) -> u32 {
    var reversed = 0u;
    var v = x;
    var b = 0u;
    loop {
        if (b >= bits) {
            break;
        }
        reversed = (reversed << 1u) | (v & 1u);
        v = v >> 1u;
        b = b + 1u;
    }
    return reversed;
}

@compute @workgroup_size(8, 8, 1)
fn water_fft_bitrev(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n || gid.y >= params.n) {
        return;
    }
    var sx = gid.x;
    var sy = gid.y;
    if (params.axis == 0u) {
        sx = reverse_bits_low(gid.x, params.log2n);
    } else {
        sy = reverse_bits_low(gid.y, params.log2n);
    }
    let dst_idx = gid.y * params.n + gid.x;
    let src_idx = sy * params.n + sx;
    dst[dst_idx] = src[src_idx];
}
