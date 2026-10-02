//! GPU-vs-CPU parity: the optional `wgpu` backend must reproduce the headless
//! reference rasteriser pixel-for-pixel, up to GPU floating-point rounding.
//!
//! The whole file is gated on the `gpu` feature. On a host without a usable
//! adapter the rasteriser constructor returns `None` and every test skips
//! gracefully (as `prism_physics_gpu`'s GPU tests do) rather than failing.
#![cfg(feature = "gpu")]

use prism_ui_layout::{Point, Rect, Size};
use prism_ui_render_backend::draw::{DrawCommand, LayerCmd};
use prism_ui_render_backend::{
    rasterize, DrawList, Framebuffer, GpuRasterizer, RectCmd, ShadowCmd,
};
use prism_ui_style::Color;

/// Per-channel tolerance. The GPU runs the identical arithmetic to the CPU
/// reference, so the only divergence is driver fused-multiply-add / rounding,
/// which stays within a few ULP. A real algorithm bug produces diffs orders of
/// magnitude larger than this.
const TOL: f32 = 2.0e-3;

fn rect(x: f32, y: f32, w: f32, h: f32) -> Rect {
    Rect::new(Point::new(x, y), Size::new(w, h))
}

fn assert_parity(list: &DrawList, w: u32, h: u32) {
    let Some(gpu) = GpuRasterizer::try_new() else {
        return;
    };
    let cpu: Framebuffer = rasterize(list, w, h);
    let gpu_fb: Framebuffer = gpu.rasterize(list, w, h);
    let diff = cpu.max_diff(&gpu_fb);
    assert!(
        diff <= TOL,
        "GPU and CPU rasterisers diverged by {diff} (tolerance {TOL})"
    );
}

#[test]
fn solid_rect_parity() {
    let mut dl = DrawList::new();
    dl.push_rect(RectCmd {
        rect: rect(4.0, 4.0, 24.0, 16.0),
        fill: Some(Color::rgba(0.9, 0.2, 0.1, 1.0)),
        radius: 0.0,
        border_width: 0.0,
        border_color: None,
        opacity: 1.0,
    });
    assert_parity(&dl, 32, 24);
}

#[test]
fn rounded_border_parity() {
    let mut dl = DrawList::new();
    dl.push_rect(RectCmd {
        rect: rect(3.0, 3.0, 26.0, 26.0),
        fill: Some(Color::rgba(0.1, 0.6, 0.9, 0.8)),
        radius: 8.0,
        border_width: 3.0,
        border_color: Some(Color::rgba(0.0, 0.0, 0.0, 1.0)),
        opacity: 1.0,
    });
    assert_parity(&dl, 32, 32);
}

#[test]
fn shadow_parity() {
    let mut dl = DrawList::new();
    dl.push_shadow(ShadowCmd {
        rect: rect(16.0, 16.0, 20.0, 20.0),
        radius: 6.0,
        blur: 10.0,
        offset: Point::new(3.0, 4.0),
        color: Color::rgba(0.0, 0.0, 0.0, 0.7),
        opacity: 1.0,
    });
    assert_parity(&dl, 56, 56);
}

#[test]
fn overlapping_translucent_parity() {
    let mut dl = DrawList::new();
    dl.push_rect(RectCmd {
        rect: rect(2.0, 2.0, 24.0, 24.0),
        fill: Some(Color::rgba(0.9, 0.1, 0.1, 0.6)),
        radius: 4.0,
        border_width: 0.0,
        border_color: None,
        opacity: 1.0,
    });
    dl.push_rect(RectCmd {
        rect: rect(12.0, 12.0, 24.0, 24.0),
        fill: Some(Color::rgba(0.1, 0.1, 0.9, 0.6)),
        radius: 4.0,
        border_width: 0.0,
        border_color: None,
        opacity: 1.0,
    });
    assert_parity(&dl, 40, 40);
}

#[test]
fn layered_opacity_parity() {
    let mut dl = DrawList::new();
    dl.push(DrawCommand::PushLayer(LayerCmd {
        bounds: rect(0.0, 0.0, 40.0, 40.0),
        opacity: 0.5,
    }));
    dl.push_rect(RectCmd {
        rect: rect(6.0, 6.0, 28.0, 28.0),
        fill: Some(Color::rgba(0.2, 0.8, 0.3, 1.0)),
        radius: 10.0,
        border_width: 2.0,
        border_color: Some(Color::rgba(0.9, 0.9, 0.9, 1.0)),
        opacity: 1.0,
    });
    dl.push(DrawCommand::PopLayer);
    assert_parity(&dl, 40, 40);
}

#[test]
fn empty_list_parity() {
    let dl = DrawList::new();
    assert_parity(&dl, 16, 16);
}
