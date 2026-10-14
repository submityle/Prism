//! Real Dock Element assembly through Loom layout and the optional wgpu twin.
//!
//! This test is intentionally in the render-backend crate: `prism_ui_dock`
//! remains a pure data crate while this integration proves that its `render`
//! feature emits a drawable tree whose CPU and GPU draw streams agree.
#![cfg(feature = "gpu")]

use prism_ui::Element;
use prism_ui_component_kit::stylesheet;
use prism_ui_dock::{
    Axis, DockHost, DockLayout, DockNode, DocumentViewRegistry, LayoutMetrics, PanelDescriptor,
    PanelRegistry, PanelViewRegistry, Ratio, SplitChild, Viewport,
};
use prism_ui_layout::Size;
use prism_ui_render_backend::raster::rasterize;
use prism_ui_render_backend::{ElementRenderer, Framebuffer, GpuRasterizer};
use prism_ui_style::{Breakpoint, Theme, TokenStore};
use prism_ui_theme::{compile_theme, ThemeDefinition, ThemeMode};

const WIDTH: u32 = 400;
const HEIGHT: u32 = 220;
const GPU_TOLERANCE: f32 = 2.0e-3;

fn glass_theme() -> Theme {
    let compiled = compile_theme(&ThemeDefinition::glass(), &ThemeMode::Light)
        .expect("glass theme should compile");
    let mut tokens = TokenStore::new();
    for (name, value) in compiled.iter() {
        tokens.insert(name, value.clone());
    }
    Theme {
        tokens,
        breakpoints: Breakpoint::ALL,
    }
}

fn dock_element() -> Element {
    let mut panels = PanelRegistry::new();
    panels
        .register(PanelDescriptor::new("scene", "Scene"))
        .expect("scene panel registration");
    panels
        .register(
            PanelDescriptor::new("inspector", "Inspector")
                .with_default_slot(prism_ui_dock::DefaultSlot::RightBar),
        )
        .expect("inspector panel registration");

    let mut views = PanelViewRegistry::new();
    views
        .register("scene", || Element::box_().class("scene-content"))
        .expect("scene view registration");
    views
        .register("inspector", || Element::box_().class("inspector-content"))
        .expect("inspector view registration");

    let layout = DockLayout {
        root: DockNode::Split {
            axis: Axis::Horizontal,
            children: vec![
                SplitChild {
                    ratio: Ratio::from_millionths(650_000),
                    node: DockNode::panel("scene"),
                },
                SplitChild {
                    ratio: Ratio::from_millionths(350_000),
                    node: DockNode::panel("inspector"),
                },
            ],
        },
        ..DockLayout::default()
    };
    let documents = DocumentViewRegistry::new();
    let mut host = DockHost::new(layout, 8).expect("dock host should validate");
    host.render_element(
        Viewport::new(WIDTH, HEIGHT),
        LayoutMetrics::default(),
        &panels,
        &views,
        &documents,
    )
    .expect("dock plan should assemble")
}

fn draw_list() -> prism_ui_render_backend::DrawList {
    let element = dock_element();
    let mut renderer = ElementRenderer::new(glass_theme(), stylesheet());
    renderer.update(&element, Size::new(WIDTH as f32, HEIGHT as f32))
}

#[test]
fn dock_element_gpu_matches_cpu_reference() {
    let list = draw_list();
    assert!(!list.is_empty(), "Dock Element produced no draw commands");
    let cpu: Framebuffer = rasterize(&list, WIDTH, HEIGHT);
    let gpu = GpuRasterizer::try_new()
        .expect("dock GPU parity requires a real wgpu adapter; refusing to skip validation");
    let gpu_frame = gpu.rasterize(&list, WIDTH, HEIGHT);
    let diff = cpu.max_diff(&gpu_frame);
    let (worst_x, worst_y, cpu_pixel, gpu_pixel) = (0..WIDTH)
        .flat_map(|x| (0..HEIGHT).map(move |y| (x, y)))
        .max_by(|(left_x, left_y), (right_x, right_y)| {
            let left = cpu.pixel(*left_x, *left_y);
            let left_gpu = gpu_frame.pixel(*left_x, *left_y);
            let right = cpu.pixel(*right_x, *right_y);
            let right_gpu = gpu_frame.pixel(*right_x, *right_y);
            let left_diff = left
                .into_iter()
                .zip(left_gpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let right_diff = right
                .into_iter()
                .zip(right_gpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            left_diff
                .partial_cmp(&right_diff)
                .expect("finite framebuffer values")
        })
        .map(|(x, y)| (x, y, cpu.pixel(x, y), gpu_frame.pixel(x, y)))
        .expect("non-empty framebuffer");
    assert!(
        diff <= GPU_TOLERANCE,
        "Dock CPU/GPU parity diverged by {diff} (tolerance {GPU_TOLERANCE}) at ({worst_x}, {worst_y}): CPU={cpu_pixel:?}, GPU={gpu_pixel:?}"
    );
    assert!(
        (0..WIDTH).any(|x| (0..HEIGHT).any(|y| cpu.pixel(x, y)[3] > 0.01)),
        "Dock rasterization was blank"
    );
}
