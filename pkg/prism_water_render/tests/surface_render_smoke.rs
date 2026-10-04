//! Real-device smoke test for the water-surface renderer.
//!
//! The test acquires a headless adapter and, when one exists, renders the
//! default ocean preset and asserts the frame is genuinely rasterised and
//! shaded rather than a flat clear. On a host with no usable adapter it skips
//! so the suite still passes in a `GPU`-less sandbox.

use std::path::PathBuf;

use prism_water_render::{render, GpuContext, WaterSurfaceScene};

/// The canonical eight-byte `PNG` signature.
const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "a skipped GPU test should say so on hosts without an adapter"
)]
fn renders_shaded_water_surface() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping: no usable GPU adapter on this host");
        return;
    };

    let width = 512;
    let height = 320;
    let scene = WaterSurfaceScene::preset(width, height);
    let frame = render(&ctx, &scene);

    assert_eq!(frame.width, width);
    assert_eq!(frame.height, height);
    assert_eq!(
        frame.rgba.len(),
        width as usize * height as usize * 4,
        "frame must be tightly packed RGBA8"
    );

    // A flat clear yields a single colour; a rasterised, shaded surface yields
    // many. The threshold is deliberately generous so the test stays robust
    // across adapters while still proving real drawing happened.
    let distinct = frame.distinct_colors();
    assert!(
        distinct > 200,
        "expected a shaded surface with many colours, got {distinct}"
    );

    // The surface should read as water: many pixels where blue dominates.
    let bluish = frame
        .rgba
        .chunks_exact(4)
        .filter(|px| px[2] > px[0] && px[2] > 40)
        .count();
    assert!(
        bluish > (width as usize * height as usize) / 10,
        "expected a water-dominated frame, only {bluish} bluish pixels"
    );

    // Persist a viewable image and confirm it is a valid PNG on disk.
    let mut path: PathBuf = std::env::temp_dir();
    path.push("prism_water_surface_smoke.png");
    frame.save_png(&path).expect("save_png should succeed");

    let bytes = std::fs::read(&path).expect("rendered PNG should be readable");
    assert!(bytes.len() > PNG_SIGNATURE.len());
    assert_eq!(
        &bytes[..PNG_SIGNATURE.len()],
        &PNG_SIGNATURE,
        "output must start with the PNG signature"
    );
}
