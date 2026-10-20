# prism_editor_graph

Engine-neutral typed graph kernel plus a bounded authoring layer. Generic
Graph<N> owns nodes, ports, groups, edge indexes and deterministic evaluation.
GraphEditor owns opaque byte payloads and integer positions, with explicit IDs,
atomic GraphCommand batches, bounded local inverse history and a canonical v1
command codec. No Bevy, renderer or project node types are required.

Use GraphCommand::Edit for CreateNode/DeleteNode/MoveNode/SetPayload/Connect/
Disconnect, and GraphCommand::Undo/Redo for history. Read graph() to evaluate
using host semantics. Successful undo restores identities without reusing the
allocation high-water marks; failed batches restore all state. The editor
validates final DAG topology and does not clone the entire graph per undo.

Resource limits bound live records, payloads, commands and retained history.
History bytes are conservative admission charges, not allocator measurements.
Command replay requires the same limits and ordered successful command stream.
The codec is not a graph snapshot or durable transaction format.

The Loom graph panel is provided by prism_editor_ui. Native material-domain
integration, WESL preview/roundtrip, group/subgraph editing and persistence are
not implemented in this slice. See docs/prism_editor_graph_authoring_zh.md in
the repository for protocol, UI wiring, verified tests and remaining boundaries.
