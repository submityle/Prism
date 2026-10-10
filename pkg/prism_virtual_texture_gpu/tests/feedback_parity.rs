//! Real-device parity tests for the sampler-feedback min-mip decode twin.
//!
//! Each test acquires a best-effort headless `GPU` via
//! [`GpuContext::try_headless`]. On a machine with a usable adapter (e.g. Apple
//! `M`-series Metal) the decode kernel runs for real and is compared against the
//! device-free CPU golden
//! [`decode_feedback`](prism_render_architecture::texture_streaming::decode_feedback);
//! where no adapter exists the test prints a skip note and returns, so CI
//! without a `GPU` stays green.
//!
//! The per-cell map is integer-only, so the device result must equal the golden
//! result *exactly* — these tests assert bit-for-bit equality of the decoded
//! [`PageDemand`] list over a multi-workgroup grid with a non-trivial residency
//! closure, exercising clamp, coarse-page collapse, dedup, and the `REQ_NONE`
//! skip path together.

use prism_render_architecture::texture_streaming::{
    decode_feedback, FeedbackTextureDesc, TexturePageKey, TextureSemantic, NOT_REQUESTED,
};
use prism_virtual_texture_gpu::{GpuContext, GpuFeedbackDecode};

/// Acquires the device or prints a skip note and returns `None`.
#[expect(
    clippy::print_stderr,
    reason = "test diagnostics: explain a GPU-less skip so CI logs are legible"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!(
                "[prism_virtual_texture_gpu] no headless GPU adapter available; \
                 skipping feedback decode parity test"
            );
            None
        }
    }
}

fn desc(pages_x: u16, pages_y: u16) -> FeedbackTextureDesc {
    FeedbackTextureDesc {
        texture: 11,
        layer: 1,
        semantic: TextureSemantic::RoughnessMetalAo,
        base_mip: 1,
        mip_count: 5, // streamable mips 1..=5
        pages_x,
        pages_y,
        page_byte_cost: 32_768,
        screen_importance: 1200,
    }
}

/// A deterministic, varied min-mip grid: a mix of not-requested cells, finest
/// requests, coarser requests that collapse neighbours, and absurd requests
/// that clamp to the coarsest streamable mip.
fn varied_grid(pages_x: u16, pages_y: u16) -> Vec<u8> {
    let mut grid = vec![NOT_REQUESTED; pages_x as usize * pages_y as usize];
    for y in 0..pages_y as usize {
        for x in 0..pages_x as usize {
            let i = y * pages_x as usize + x;
            // Deterministic pattern touching every branch:
            //  - every 5th cell stays NOT_REQUESTED
            //  - some ask for the base (finest streamable) mip
            //  - some ask for progressively coarser mips (collapse)
            //  - some ask absurdly coarse (clamp to max)
            match (x + y) % 5 {
                0 => {}             // leave NOT_REQUESTED
                1 => grid[i] = 1,   // base mip
                2 => grid[i] = 2,   // one coarser -> 2x2 collapse
                3 => grid[i] = 3,   // two coarser -> 4x4 collapse
                _ => grid[i] = 240, // absurd -> clamp to max mip 5
            }
        }
    }
    grid
}

/// Residency closure with a deterministic, key-dependent spread of resident
/// mips and holes, so `PageDemand::resident_mip` is non-trivial.
fn resident(k: TexturePageKey) -> Option<u8> {
    if (u32::from(k.x) + u32::from(k.y) + u32::from(k.mip)) % 3 == 0 {
        None
    } else {
        Some(k.mip.saturating_add(1))
    }
}

#[test]
fn decode_large_grid_matches_golden_exactly() {
    let Some(ctx) = with_gpu() else { return };
    // 48x40 = 1920 cells -> spans several @workgroup_size(256) groups.
    let (px, py) = (48u16, 40u16);
    let d = desc(px, py);
    let grid = varied_grid(px, py);

    let kernel = GpuFeedbackDecode::new(&ctx);
    let device = kernel.decode(&ctx, &d, &grid, resident, 777);
    let golden = decode_feedback(&d, &grid, resident, 777);

    assert!(!device.is_empty(), "varied grid must produce demand");
    assert_eq!(
        device, golden,
        "device decode must equal the CPU golden bit-for-bit"
    );

    // Anti-vacuous guards: the fixture must actually exercise the branches.
    assert!(
        golden
            .iter()
            .any(|d| d.key.mip == d.desired_mip && d.desired_mip == 5),
        "some demand clamps to the coarsest streamable mip"
    );
    assert!(
        golden.iter().any(|d| d.resident_mip.is_some()),
        "some demand is partially resident"
    );
    assert!(
        golden.iter().any(|d| d.resident_mip.is_none()),
        "some demand is fully missing"
    );
}

#[test]
fn decode_all_not_requested_is_empty_on_device() {
    let Some(ctx) = with_gpu() else { return };
    let (px, py) = (16u16, 16u16);
    let d = desc(px, py);
    let grid = vec![NOT_REQUESTED; px as usize * py as usize];

    let kernel = GpuFeedbackDecode::new(&ctx);
    let device = kernel.decode(&ctx, &d, &grid, resident, 1);
    assert!(device.is_empty(), "an all-skip grid yields no demand");
    assert_eq!(device, decode_feedback(&d, &grid, resident, 1));
}
