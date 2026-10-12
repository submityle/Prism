//! Real-device smoke test for the `Tessendorf` `FFT` ocean renderer.
//!
//! The test acquires a headless adapter and, when one exists, renders the
//! default ocean preset and asserts the frame is a genuinely rasterised,
//! shaded spectral surface rather than a flat clear. On a host with no usable
//! adapter it skips so the suite still passes in a `GPU`-less sandbox.

use std::path::PathBuf;

use prism_water_render::{render_ocean, GpuContext, OceanScene};

/// The canonical eight-byte `PNG` signature.
const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "a skipped GPU test should say so on hosts without an adapter"
)]
fn renders_shaded_fft_ocean() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping: no usable GPU adapter on this host");
        return;
    };

    let width = 512;
    let height = 320;
    let scene = OceanScene::preset(width, height);
    let frame = render_ocean(&ctx, &scene);

    assert_eq!(frame.width, width);
    assert_eq!(frame.height, height);
    assert_eq!(
        frame.rgba.len(),
        width as usize * height as usize * 4,
        "frame must be tightly packed RGBA8"
    );

    // A flat clear yields a single colour; a rasterised, shaded spectral
    // surface yields many. The threshold is generous so the test stays robust
    // across adapters while still proving real drawing happened.
    let distinct = frame.distinct_colors();
    assert!(
        distinct > 200,
        "expected a shaded ocean with many colours, got {distinct}"
    );

    // The ocean should read as water: many pixels where blue dominates.
    let bluish = frame.bluish_pixels();
    assert!(
        bluish > (width as usize * height as usize) / 10,
        "expected a water-dominated frame, only {bluish} bluish pixels"
    );

    // A non-power-of-two grid is an honest no-op: the architecture returns an
    // empty surface, so the frame must clear to a single background colour.
    let mut empty = OceanScene::preset(64, 64);
    empty.resolution = 100;
    let cleared = render_ocean(&ctx, &empty);
    assert_eq!(
        cleared.distinct_colors(),
        1,
        "a degenerate ocean must clear to one colour, not fabricate a surface"
    );

    // Persist a viewable image and confirm it is a valid PNG on disk.
    let mut path: PathBuf = std::env::temp_dir();
    path.push("prism_fft_ocean_smoke.png");
    frame.save_png(&path).expect("save_png should succeed");

    let bytes = std::fs::read(&path).expect("rendered PNG should be readable");
    assert!(bytes.len() > PNG_SIGNATURE.len());
    assert_eq!(
        &bytes[..PNG_SIGNATURE.len()],
        &PNG_SIGNATURE,
        "output must start with the PNG signature"
    );
}
