//! Per-pass recording, shared by the sequential and parallel executors.
//!
//! A pass's execute closure is run against an [`ExecuteContext`] over the
//! realized resource tables, and the commands it recorded are captured, tagged
//! by the pass's kind. Tagging by kind is what keeps parallel assembly
//! identical to sequential assembly: the sequential executor also selects
//! exactly one command vector per pass (a closure that records, say, render
//! commands inside a compute pass has them dropped), so the tag reproduces that
//! selection instead of concatenating everything a closure happened to record.

use alloc::vec::Vec;

use prism_render_driver::{ComputeCommand, CopyCommand, RenderCommand};

use crate::builder::ExecuteContext;
use crate::pass::{BoxedExecute, PassKind};
use crate::resource::{BufferResource, TextureResource};

/// The commands one pass recorded during execution, tagged by pass kind.
///
/// Mirrors the sequential executor's per-kind selection so both executors emit
/// identical command buffers.
#[derive(PartialEq, Debug)]
pub(crate) enum PassCommands {
    /// Commands recorded by a raster pass.
    Render(Vec<RenderCommand>),
    /// Commands recorded by a compute pass.
    Compute(Vec<ComputeCommand>),
    /// Copy commands recorded by a transfer pass.
    Transfer(Vec<CopyCommand>),
    /// A present pass: state tracking only, no recorded GPU pass.
    None,
}

/// Runs one pass's execute closure and captures the commands it recorded.
///
/// The context borrows the realized resource tables immutably, so recording
/// never mutates shared graph state; a pass's only output is the returned
/// command set. Passes still run their closure when they record no GPU pass
/// (present), matching the sequential executor exactly.
pub(crate) fn record_pass(
    execute: Option<BoxedExecute>,
    kind: PassKind,
    textures: &[TextureResource],
    buffers: &[BufferResource],
) -> PassCommands {
    let mut ctx = ExecuteContext::new(textures, buffers, kind);
    if let Some(execute) = execute {
        execute(&mut ctx);
    }
    match kind {
        PassKind::Raster => PassCommands::Render(ctx.take_render_commands()),
        PassKind::Compute => PassCommands::Compute(ctx.take_compute_commands()),
        PassKind::Transfer => PassCommands::Transfer(ctx.take_copy_commands()),
        PassKind::Present => PassCommands::None,
    }
}
