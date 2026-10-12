//! Renders the default `Tessendorf` `FFT` ocean preset to a `PNG` on a real
//! device.
//!
//! Usage: `cargo run -p prism_water_render --example fft_ocean [OUT.png]`.
//! With no argument the image is written to the system temp directory. On a
//! host with no usable `GPU` adapter the example prints a notice and exits
//! without error.

use std::path::PathBuf;

use prism_water_render::{render_ocean, GpuContext, OceanScene};

#[expect(
    clippy::print_stdout,
    reason = "a CLI example should report where it wrote its output"
)]
fn main() {
    let out: PathBuf = std::env::args().nth(1).map_or_else(
        || {
            let mut p = std::env::temp_dir();
            p.push("prism_fft_ocean.png");
            p
        },
        PathBuf::from,
    );

    let Some(ctx) = GpuContext::try_headless() else {
        println!("no usable GPU adapter on this host; nothing rendered");
        return;
    };

    let scene = OceanScene::preset(1280, 720);
    let frame = render_ocean(&ctx, &scene);
    frame.save_png(&out).expect("failed to write output PNG");

    println!(
        "rendered {}x{} FFT ocean ({} distinct colours) to {}",
        frame.width,
        frame.height,
        frame.distinct_colors(),
        out.display()
    );
}
