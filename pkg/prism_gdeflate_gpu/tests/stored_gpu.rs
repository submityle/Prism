//! Real-device validation of the stored-block `GDeflate` GPU fast path.
//!
//! This test intentionally does not call the CPU inflate implementation for
//! the successful path: the output is produced by the de-interleave pass and
//! the GPU stored-block copy pass, then compared directly with the source
//! bytes. A host without a compute adapter skips cleanly.

use prism_gdeflate_gpu::{GpuContext, GpuStoredGDeflate, StoredPlanError, GROUP, WORD};

fn stored_stream(payloads: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for (index, payload) in payloads.iter().enumerate() {
        let length = payload.len() as u16;
        out.push(if index + 1 == payloads.len() { 1 } else { 0 });
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&(!length).to_le_bytes());
        out.extend_from_slice(payload);
    }
    out
}

fn interleave(linear: &[u8]) -> Vec<u8> {
    let padded_len = linear.len().div_ceil(GROUP) * GROUP;
    let mut padded = vec![0u8; padded_len];
    padded[..linear.len()].copy_from_slice(linear);
    let rounds = padded_len / (GROUP);
    let mut output = vec![0u8; padded_len];
    for lane in 0..32usize {
        for round in 0..rounds {
            let logical = round * 32 + lane;
            let stored = lane * rounds + round;
            output[stored * WORD..stored * WORD + WORD]
                .copy_from_slice(&padded[logical * WORD..logical * WORD + WORD]);
        }
    }
    output
}

#[expect(
    clippy::print_stderr,
    reason = "a skipped GPU test must explain why it did not dispatch"
)]
#[test]
fn gpu_stored_blocks_recover_payload_without_cpu_inflate() {
    let first: Vec<u8> = (0..251u16).map(|value| value as u8).collect();
    let second: Vec<u8> = (0..137u16)
        .map(|value| value.wrapping_mul(17) as u8)
        .collect();
    let expected = [first.as_slice(), second.as_slice()].concat();
    let linear = stored_stream(&[first.as_slice(), second.as_slice()]);
    let plan = prism_gdeflate_gpu::plan_stored_deflate(&linear, expected.len()).unwrap();
    let interleaved = interleave(&linear);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("[skip] gpu_stored_blocks_recover_payload_without_cpu_inflate: no GPU");
        return;
    };
    let decoder = GpuStoredGDeflate::new(&ctx);
    let recovered = decoder
        .decompress(&ctx, &interleaved, &plan)
        .expect("validated stored blocks must decode on the GPU");
    assert_eq!(recovered, expected);

    let mut extra_padding_group = interleaved;
    extra_padding_group.resize(extra_padding_group.len() + GROUP, 0);
    assert_eq!(
        decoder.decompress(&ctx, &extra_padding_group, &plan),
        Err(StoredPlanError::InvalidSize),
        "extra padding changes the transpose round count and must be rejected"
    );
}
