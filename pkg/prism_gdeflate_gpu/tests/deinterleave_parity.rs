//! Real-device parity tests for the `GDeflate` word-transpose de-interleave.
//!
//! Each test acquires a real headless `GPU` via [`GpuContext::try_headless`]
//! and skips cleanly when no adapter is present. The de-interleave is a pure
//! integer word permutation, so parity is asserted with exact equality:
//!
//! * against the independent host oracle [`reference_deinterleave`], over a
//!   multi-workgroup buffer and a single-round edge case; and
//! * end to end, by feeding a real [`gdeflate_compress`] tile payload through
//!   the device de-interleave and then the golden [`inflate`], recovering the
//!   original tile bytes.
//!
//! No Unreal Engine or NVIDIA `GDeflate` source or derived code.

use prism_gdeflate_gpu::{reference_deinterleave, GpuContext, GpuDeinterleave, GROUP};
use prism_render_architecture::compression::{gdeflate_compress, inflate};

/// Serialized `GDeflateHeader` length (little-endian, documented layout).
const HEADER_LEN: usize = 20;
/// Serialized `GDeflateTileDescriptor` length (little-endian, documented
/// layout).
const DESCRIPTOR_LEN: usize = 12;

/// Runs `body` with a real device, or prints a skip note and returns when none
/// is available so the suite passes on adapterless hosts.
#[expect(
    clippy::print_stderr,
    reason = "a skipped GPU test must say so on hosts without an adapter"
)]
fn with_gpu(name: &str, body: impl FnOnce(&GpuContext)) {
    match GpuContext::try_headless() {
        Some(ctx) => body(&ctx),
        None => eprintln!("[skip] {name}: no usable GPU adapter on this host"),
    }
}

/// Deterministic LCG byte stream (classical, no external entropy).
fn lcg_bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut data = vec![0u8; len];
    for byte in &mut data {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    data
}

/// Reads a little-endian `u32` at `offset` within `bytes`.
fn u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// Rounds `value` up to the next multiple of `multiple`.
fn round_up(value: usize, multiple: usize) -> usize {
    value.div_ceil(multiple) * multiple
}

#[test]
fn device_matches_reference_over_many_workgroups() {
    with_gpu("device_matches_reference_over_many_workgroups", |ctx| {
        let kernel = GpuDeinterleave::new(ctx);

        // 640 words spans three `@workgroup_size(256)` groups and 20 interleave
        // rounds, so the lane/round index math is exercised past one workgroup.
        let interleaved = lcg_bytes(GROUP * 20, 0xa5a5_1234);
        assert_eq!(interleaved.len() % GROUP, 0);

        let gpu = kernel.deinterleave(ctx, &interleaved);
        let cpu = reference_deinterleave(&interleaved);

        assert_eq!(gpu.len(), interleaved.len());
        assert_eq!(gpu, cpu, "device de-interleave must equal the host oracle");
        // Anti-vacuous: a non-trivial permutation actually reorders the bytes.
        assert_ne!(
            gpu, interleaved,
            "20-round transpose must permute the words"
        );
    });
}

#[test]
fn device_matches_reference_single_round() {
    with_gpu("device_matches_reference_single_round", |ctx| {
        let kernel = GpuDeinterleave::new(ctx);

        // Exactly one lane group: 32 words, `rounds == 1`. With one round the
        // permutation is the identity (`stored == logical`), which still must
        // match the oracle exactly.
        let interleaved = lcg_bytes(GROUP, 0x0f0f_abcd);
        let gpu = kernel.deinterleave(ctx, &interleaved);
        let cpu = reference_deinterleave(&interleaved);

        assert_eq!(gpu.len(), GROUP);
        assert_eq!(gpu, cpu);
        assert_eq!(gpu, interleaved, "single round is the identity permutation");
    });
}

#[test]
fn device_deinterleave_feeds_golden_inflate() {
    with_gpu("device_deinterleave_feeds_golden_inflate", |ctx| {
        let kernel = GpuDeinterleave::new(ctx);

        // A structured, sub-tile payload: compresses to many lane groups so the
        // transpose has multiple rounds, and inflates back to the exact tile.
        let mut tile = Vec::new();
        for i in 0..1_200u32 {
            tile.extend_from_slice(format!("row {i}: v={}\n", (i * 31) % 211).as_bytes());
        }

        let compressed = gdeflate_compress(&tile);
        assert!(compressed.len() >= HEADER_LEN + DESCRIPTOR_LEN);

        // Single-tile container (payload < 64 KiB), so there is one descriptor.
        let tile_count = u32_le(&compressed, 8) as usize;
        assert_eq!(tile_count, 1, "sub-tile input must produce one tile");

        let uncompressed_size = u32_le(&compressed, HEADER_LEN) as usize;
        let compressed_size = u32_le(&compressed, HEADER_LEN + 4) as usize;
        assert_eq!(uncompressed_size, tile.len());

        let payload = &compressed[HEADER_LEN + DESCRIPTOR_LEN..];
        let padded = round_up(compressed_size, GROUP);
        assert!(padded >= GROUP, "payload must span at least one lane group");
        let chunk = &payload[..padded];

        // Device reverses the warp interleave; host truncates + inflates exactly
        // as `gdeflate_decompress` does.
        let mut deflated = kernel.deinterleave(ctx, chunk);
        assert_eq!(deflated.len(), padded);
        deflated.truncate(compressed_size);

        let recovered = inflate(&deflated).expect("golden inflate must succeed");
        assert_eq!(
            recovered, tile,
            "device-deinterleaved tile must inflate exactly"
        );

        // Cross-check: the device de-interleave equals the host oracle here too.
        assert_eq!(
            kernel.deinterleave(ctx, chunk),
            reference_deinterleave(chunk)
        );
    });
}
