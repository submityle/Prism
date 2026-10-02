#![cfg_attr(not(feature = "std"), no_std)]
//! Loom GPU-driven retained draw-stream backend.
//!
//! Deepened design lives in `docs/prism_ui_loom_design_zh.md` §9.8. This crate
//! turns the `prism_ui` runtime's minimal [`BackendOp`](prism_ui::BackendOp)
//! stream into a GPU-friendly, paint-ordered draw stream and provides a
//! headless CPU reference backend so an optional `wgpu` backend can be
//! validated against it pixel-for-pixel.
//!
//! The pipeline is deliberately layered, one concern per module:
//!
//! - [`sdf`] — closed-form signed-distance primitives shared by every backend.
//! - [`draw`] — the flat, absolute [`DrawList`](draw::DrawList) hand-off format.
//! - [`scene`] — a [`RetainedScene`](scene::RetainedScene) that absorbs
//!   [`BackendOp`](prism_ui::BackendOp)s and lowers them to a `DrawList`.
//! - [`batch`] — coalesces the draw list into instanced GPU batches.
//! - [`layer`] — parses compositing layers for independent recompositing.
//! - [`raster`] — the headless golden-twin rasteriser.
//!
//! `unsafe` is confined to the optional `gpu` feature (wgpu buffer mapping); the
//! default build is safe.
//!
//! ```
//! use prism_ui::{Backend, BackendId, BackendOp, ElementKind, PaintStyle};
//! use prism_ui_layout::{Point, Size};
//! use prism_ui_render_backend::scene::RetainedScene;
//! use prism_ui_render_backend::{batch, raster};
//!
//! let mut scene = RetainedScene::default();
//! scene.apply(BackendOp::Create {
//!     id: BackendId(1),
//!     kind: ElementKind::Box,
//!     parent: None,
//!     index: 0,
//! });
//! scene.apply(BackendOp::SetLayout {
//!     id: BackendId(1),
//!     location: Point::new(2.0, 2.0),
//!     size: Size::new(6.0, 6.0),
//! });
//! let paint = PaintStyle {
//!     background_color: Some(prism_ui_style::Color::rgba(1.0, 0.0, 0.0, 1.0)),
//!     ..PaintStyle::default()
//! };
//! scene.apply(BackendOp::SetPaint { id: BackendId(1), paint });
//!
//! let list = scene.to_draw_list();
//! assert_eq!(list.len(), 1);
//!
//! // The same list drives both the GPU batcher and the reference rasteriser.
//! let batches = batch::batch(&list);
//! assert_eq!(batch::instance_count(&batches), 1);
//!
//! let fb = raster::rasterize(&list, 10, 10);
//! // Dead centre of the box is fully covered red.
//! assert!(fb.pixel(5, 5)[0] > 0.9);
//! ```
#![cfg_attr(not(feature = "gpu"), forbid(unsafe_code))]

extern crate alloc;

pub mod batch;
pub mod draw;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod layer;
pub mod raster;
pub mod scene;
pub mod sdf;

pub use batch::{batch, instance_count, Batch, GlyphInstance, RectInstance, ShadowInstance};
pub use draw::{DrawCommand, DrawList, GlyphCmd, LayerCmd, RectCmd, ShadowCmd};
#[cfg(feature = "gpu")]
pub use gpu::GpuRasterizer;
pub use layer::{Layer, LayerTree};
pub use raster::{rasterize, Framebuffer};
pub use scene::RetainedScene;
