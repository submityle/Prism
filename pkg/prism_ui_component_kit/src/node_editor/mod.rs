//! `node_editor/` controls. See the kit design doc, section 5.
//!
//! A small node-graph editor family: a render-free data model
//! ([`graph`]) plus the presentational controls that draw it — [`Port`],
//! [`Node`], [`EdgeView`], [`NodeCanvas`] and [`Minimap`].
//!
//! Each control emits a data-only [`Element`](prism_ui::Element) carrying only
//! kit class names; this module's [`register_styles`] owns the token-backed
//! [`Class`](prism_ui_style::Class) rules for those names. Controls stay plain
//! [`Component`](prism_ui_component::Component)s so they compose and unit-test
//! without a running runtime. The style layer exposes no `position`/`transform`
//! property, so canvas placement is approximated with inline left/top margins.

use crate::preset::StyleSheet;

pub mod edge;
pub mod graph;
pub mod minimap;
pub mod node;
pub mod node_canvas;
pub mod port;

pub use edge::{EdgeView, EdgeViewProps};
pub use graph::{Edge, GraphModel, NodeData, NodeId, PortId};
pub use minimap::{Minimap, MinimapProps};
pub use node::{Node, NodeProps};
pub use node_canvas::{NodeCanvas, NodeCanvasProps};
pub use port::{Port, PortKind, PortProps};

/// Registers every `node_editor/` control's token-backed classes into `sheet`.
///
/// The data-only [`graph`] module has no classes to register.
pub fn register_styles(sheet: &mut StyleSheet) {
    port::register_styles(sheet);
    node::register_styles(sheet);
    edge::register_styles(sheet);
    node_canvas::register_styles(sheet);
    minimap::register_styles(sheet);
}
