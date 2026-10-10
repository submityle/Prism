//! Gallery instances for the `node_editor` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui_component::mount_component;

use super::Showcase;
use crate::node_editor::{
    EdgeView, EdgeViewProps, Minimap, MinimapProps, Node, NodeCanvas, NodeCanvasProps, NodeProps,
    Port, PortKind, PortProps,
};

/// Real, named instances of every `node_editor` control.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        Showcase::new(
            "NodeCanvas / Graph",
            mount_component(
                &NodeCanvas,
                NodeCanvasProps::new()
                    .node(NodeProps::new("Source").pos(20.0, 20.0).output("out"))
                    .node(NodeProps::new("Sink").pos(220.0, 60.0).input("in"))
                    .edge(EdgeViewProps::new((120.0, 40.0), (220.0, 80.0))),
            ),
        ),
        Showcase::new(
            "Node / Selected",
            mount_component(
                &Node,
                NodeProps::new("Add")
                    .pos(12.0, 34.0)
                    .inputs(["a", "b"])
                    .output("sum")
                    .selected(true),
            ),
        ),
        Showcase::new(
            "Port / Output",
            mount_component(
                &Port,
                PortProps::new("result").kind(PortKind::Output).connected(true),
            ),
        ),
        Showcase::new(
            "EdgeView / Link",
            mount_component(&EdgeView, EdgeViewProps::new((30.0, 40.0), (210.0, 120.0))),
        ),
        Showcase::new(
            "Minimap / Nodes",
            mount_component(
                &Minimap,
                MinimapProps::new().nodes([(20.0, 20.0), (220.0, 60.0)]),
            ),
        ),
    ]
}
