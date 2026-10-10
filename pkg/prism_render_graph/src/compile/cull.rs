//! Dead-pass culling by reverse reachability from the frame's roots.
//!
//! A pass survives only if something outside the graph observes its work. The
//! culler seeds a worklist with the mandatory roots — present passes, passes
//! flagged [`PassFlags::SIDE_EFFECT`]/[`PassFlags::NEVER_CULL`], and any pass
//! that writes an imported or persistent resource (whose contents outlive the
//! frame) — then walks backwards along producer edges, marking every pass that
//! contributes an input to an already-live pass. Passes never reached are dead
//! and are dropped before scheduling.
//!
//! [`PassFlags::SIDE_EFFECT`]: crate::PassFlags::SIDE_EFFECT
//! [`PassFlags::NEVER_CULL`]: crate::PassFlags::NEVER_CULL

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::pass::PassNode;
use crate::resource::{BufferResource, Lifetime, TextureResource};

/// Returns per-pass liveness: `alive[i]` is `true` iff pass `i` must execute.
///
/// Deterministic: liveness depends only on the declared accesses and resource
/// lifetimes, never on iteration order of a hash container.
pub(crate) fn cull(
    passes: &[PassNode],
    textures: &[TextureResource],
    buffers: &[BufferResource],
) -> Vec<bool> {
    // (resource, produced-version) -> producing pass index.
    let mut tex_producer: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    let mut buf_producer: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for (idx, pass) in passes.iter().enumerate() {
        for acc in &pass.texture_accesses {
            if acc.produces {
                tex_producer.insert((acc.resource.get(), acc.output_version), idx);
            }
        }
        for acc in &pass.buffer_accesses {
            if acc.produces {
                buf_producer.insert((acc.resource.get(), acc.output_version), idx);
            }
        }
    }

    let mut alive = vec![false; passes.len()];
    let mut stack: Vec<usize> = Vec::new();
    for (idx, pass) in passes.iter().enumerate() {
        if (pass.is_root() || writes_external(pass, textures, buffers)) && !alive[idx] {
            alive[idx] = true;
            stack.push(idx);
        }
    }

    while let Some(p) = stack.pop() {
        for acc in &passes[p].texture_accesses {
            if let Some(&prod) = tex_producer.get(&(acc.resource.get(), acc.input_version))
                && !alive[prod]
            {
                alive[prod] = true;
                stack.push(prod);
            }
        }
        for acc in &passes[p].buffer_accesses {
            if let Some(&prod) = buf_producer.get(&(acc.resource.get(), acc.input_version))
                && !alive[prod]
            {
                alive[prod] = true;
                stack.push(prod);
            }
        }
    }

    alive
}

/// Whether a pass writes an imported or persistent resource, whose contents are
/// observable after the frame and therefore anchor the pass as a culling root.
fn writes_external(
    pass: &PassNode,
    textures: &[TextureResource],
    buffers: &[BufferResource],
) -> bool {
    let tex = pass
        .texture_accesses
        .iter()
        .any(|a| a.produces && is_external(textures[a.resource.get() as usize].lifetime));
    let buf = pass
        .buffer_accesses
        .iter()
        .any(|a| a.produces && is_external(buffers[a.resource.get() as usize].lifetime));
    tex || buf
}

/// Whether a lifetime class makes a resource's final contents observable beyond
/// the frame (imported targets, cross-frame history buffers).
const fn is_external(lifetime: Lifetime) -> bool {
    matches!(lifetime, Lifetime::Imported | Lifetime::Persistent)
}
