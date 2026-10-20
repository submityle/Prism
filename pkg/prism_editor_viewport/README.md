# prism_editor_viewport

`prism_editor_viewport` owns the deterministic editor-side contracts for
multiple viewports, asynchronous picking, and command-oriented gizmo drags.
It does not acquire a GPU, perform ray casting, or draw gizmos. Those concerns
are renderer and input adapters that consume these contracts.

The model rejects stale pick results, keeps viewport identities monotonic, and
coalesces a drag into one final transform delta so an editor command stack can
undo the interaction as a unit.
