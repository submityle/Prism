//! Multi-threaded pass recording, enabled by the `parallel` feature.
//!
//! The executor partitions the frame's ordered work into contiguous chunks,
//! records each chunk on its own worker thread, then concatenates the per-chunk
//! results in chunk order. Because chunks are contiguous and ascending, the
//! concatenation reproduces the plan's order exactly; because each worker writes
//! only its own local output, recording needs no shared mutable state — only the
//! immutable, already-realized resource tables.
//!
//! ## Determinism
//!
//! Recording a pass is a pure function of its closure and the realized
//! resources, and assembly is order-preserving. The number of workers, the
//! chunk split, and thread completion order therefore never affect the result:
//! the parallel executor produces a byte-for-byte identical [`CommandBuffer`]
//! to the sequential one for the same graph.
//!
//! [`CommandBuffer`]: prism_render_driver::CommandBuffer

use alloc::vec::Vec;
use std::thread;

use crate::pass::{BoxedExecute, PassKind};
use crate::record::{self, PassCommands};
use crate::resource::{BufferResource, TextureResource};

/// One ordered unit of recording work: a surviving pass and the closure it will
/// run. Closures are moved here up front so each worker owns exactly the ones it
/// records and never touches another pass's output.
pub(crate) struct RecordUnit {
    /// Position in the compiled `ExecutionPlan::order`.
    pub(crate) pos: usize,
    /// Index into the graph's pass list.
    pub(crate) pidx: usize,
    /// The kind of pass, selecting which recorded commands are kept.
    pub(crate) kind: PassKind,
    /// The deferred recording closure, taken by the worker that records it.
    pub(crate) execute: Option<BoxedExecute>,
}

/// Records every unit concurrently, returning `(pos, pidx, commands)` pairs in
/// ascending `pos` order.
///
/// `units` is borrowed mutably only to move each closure out into the worker
/// that records it; the graph's resource tables are shared immutably.
pub(crate) fn record(
    units: &mut [RecordUnit],
    textures: &[TextureResource],
    buffers: &[BufferResource],
) -> Vec<(usize, usize, PassCommands)> {
    let total = units.len();
    if total == 0 {
        return Vec::new();
    }

    let workers = worker_count(total);
    let chunk_size = total.div_ceil(workers);

    let mut recorded: Vec<(usize, usize, PassCommands)> = Vec::with_capacity(total);
    thread::scope(|scope| {
        let mut rest: &mut [RecordUnit] = units;
        let mut handles = Vec::with_capacity(workers);
        while !rest.is_empty() {
            let take = chunk_size.min(rest.len());
            let (chunk, tail) = rest.split_at_mut(take);
            rest = tail;
            handles.push(scope.spawn(move || record_chunk(chunk, textures, buffers)));
        }
        for handle in handles {
            recorded.extend(
                handle
                    .join()
                    .expect("a render-graph recording worker panicked"),
            );
        }
    });
    recorded
}

/// Records one contiguous chunk of units on the calling thread.
fn record_chunk(
    chunk: &mut [RecordUnit],
    textures: &[TextureResource],
    buffers: &[BufferResource],
) -> Vec<(usize, usize, PassCommands)> {
    chunk
        .iter_mut()
        .map(|unit| {
            let execute = unit.execute.take();
            let commands = record::record_pass(execute, unit.kind, textures, buffers);
            (unit.pos, unit.pidx, commands)
        })
        .collect()
}

/// The number of worker threads: the machine's available parallelism, capped by
/// the amount of work so a small frame never spawns idle threads.
fn worker_count(total: usize) -> usize {
    let available = thread::available_parallelism().map_or(1, core::num::NonZeroUsize::get);
    available.min(total).max(1)
}
