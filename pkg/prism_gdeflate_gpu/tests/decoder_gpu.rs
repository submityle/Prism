//! End-to-end full GPU DEFLATE decoder validation.

use prism_gdeflate_gpu::{DecodePath, GDeflateDecoder, GpuContext};
use prism_render_architecture::compression::gdeflate_compress;

#[expect(
    clippy::print_stderr,
    reason = "a skipped GPU test must explain why it did not dispatch"
)]
#[test]
fn unified_decoder_reports_full_gpu_deflate_path() {
    let input: Vec<u8> = (0..90_000u32)
        .map(|value| value.wrapping_mul(37) as u8)
        .collect();
    let encoded = gdeflate_compress(&input);
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("[skip] unified_decoder_reports_full_gpu_deflate_path: no GPU");
        return;
    };

    let decoder = GDeflateDecoder::new(&ctx);
    let decoded = decoder
        .decode(&ctx, &encoded)
        .expect("full GPU decode must preserve the payload");
    assert_eq!(decoded.bytes, input);
    assert_eq!(decoded.report.path, DecodePath::GpuDeflate);
    assert!(decoded.report.gpu_attempted);
    assert_eq!(decoded.report.uncompressed_bytes, input.len());
}
