//! Pass nodes: the unit of work the graph schedules.
//!
//! A pass is pure declaration plus a deferred recording closure. During the
//! *setup* phase a pass declares which virtual resources it reads and writes
//! (captured here as [`TextureAccessRecord`] / [`BufferAccessRecord`]); during
//! the *execute* phase its closure records real driver commands. Keeping the
//! two phases apart is what lets the compiler reason about the whole frame
//! before a single command is emitted.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use prism_render_driver::SubresourceRange;

use crate::access::{BufferUse, TextureUse};
use crate::builder::ExecuteContext;
use crate::handle::ResourceIndex;

/// The boxed, move-only closure a pass runs during execution to record its
/// draw/dispatch commands against the resolved [`ExecuteContext`].
///
/// With the `parallel` feature enabled the closure must be [`Send`], because
/// the executor may record the pass on a worker thread; the default build
/// carries no such bound.
#[cfg(feature = "parallel")]
pub type BoxedExecute = Box<dyn FnOnce(&mut ExecuteContext<'_>) + Send>;

/// The boxed, move-only closure a pass runs during execution to record its
/// draw/dispatch commands against the resolved [`ExecuteContext`].
#[cfg(not(feature = "parallel"))]
pub type BoxedExecute = Box<dyn FnOnce(&mut ExecuteContext<'_>)>;

/// The threading bound an execute closure must satisfy in this build.
///
/// With the `parallel` feature the graph records surviving passes on worker
/// threads, so [`RenderGraph::add_raster_pass`] and friends require their
/// closure to be [`Send`]. Without the feature every type satisfies this trait,
/// so the default single-threaded build accepts exactly the closures it did
/// before.
///
/// [`RenderGraph::add_raster_pass`]: crate::RenderGraph::add_raster_pass
#[cfg(feature = "parallel")]
pub trait ThreadBound: Send {}
#[cfg(feature = "parallel")]
impl<T: Send + ?Sized> ThreadBound for T {}

/// The threading bound an execute closure must satisfy in this build.
///
/// Without the `parallel` feature this is satisfied by every type, so the
/// default build places no thread-safety requirement on execute closures.
#[cfg(not(feature = "parallel"))]
pub trait ThreadBound {}
#[cfg(not(feature = "parallel"))]
impl<T: ?Sized> ThreadBound for T {}

/// What kind of work a pass performs.
///
/// The kind selects the default pipeline stage used to lower shader accesses
/// and decides how the executor turns the pass into a driver pass.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PassKind {
    /// A raster pass: binds color/depth attachments and records draw commands.
    Raster,
    /// A compute pass: records dispatch commands.
    Compute,
    /// A transfer pass: records copy/blit commands into a driver transfer pass.
    Transfer,
    /// A present pass: hands an imported swapchain image to the surface.
    Present,
}

impl PassKind {
    /// Whether passes of this kind are candidates for a recorded encoder pass
    /// (raster/compute/transfer). Present contributes only state tracking: the
    /// surface, not the command buffer, performs the actual presentation.
    #[must_use]
    pub const fn records_commands(self) -> bool {
        matches!(self, Self::Raster | Self::Compute | Self::Transfer)
    }
}

/// Bit flags that steer how the compiler treats a pass.
///
/// A tiny hand-rolled bit set (the driver's `bitflags!` macro is private to
/// that crate). Combine with [`PassFlags::union`] / test with
/// [`PassFlags::contains`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassFlags(u32);

impl PassFlags {
    /// No flags set.
    pub const EMPTY: Self = Self(0);
    /// The pass runs on the async-compute queue when the backend offers one.
    pub const ASYNC_COMPUTE: Self = Self(1 << 0);
    /// Never cull this pass, even if its outputs look unused (debug overlays,
    /// readbacks whose consumer lives outside the graph).
    pub const NEVER_CULL: Self = Self(1 << 1);
    /// The pass has observable side effects outside the graph and is always a
    /// culling root.
    pub const SIDE_EFFECT: Self = Self(1 << 2);
    /// The pass may be merged with an adjacent raster pass into one render pass
    /// (subpass/tile merging) when inputs allow.
    pub const MERGE_CANDIDATE: Self = Self(1 << 3);

    /// The raw bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// The union of two flag sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every bit in `other` is set in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Whether no flags are set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl Default for PassFlags {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl core::ops::BitOr for PassFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl fmt::Debug for PassFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        let mut emit = |name: &str, f: &mut fmt::Formatter<'_>| -> fmt::Result {
            if !first {
                f.write_str(" | ")?;
            }
            first = false;
            f.write_str(name)
        };
        if self.contains(Self::ASYNC_COMPUTE) {
            emit("ASYNC_COMPUTE", f)?;
        }
        if self.contains(Self::NEVER_CULL) {
            emit("NEVER_CULL", f)?;
        }
        if self.contains(Self::SIDE_EFFECT) {
            emit("SIDE_EFFECT", f)?;
        }
        if self.contains(Self::MERGE_CANDIDATE) {
            emit("MERGE_CANDIDATE", f)?;
        }
        if first {
            f.write_str("EMPTY")?;
        }
        Ok(())
    }
}

/// One recorded texture access by a pass.
///
/// `input_version` is the resource version the pass observes on entry; if the
/// access writes (`produces`), it mints the resource's next output version,
/// which may be newer than `input_version + 1` for a stale input handle.
/// The `range` scopes the access to specific mips/layers for subresource-exact
/// barriers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TextureAccessRecord {
    /// The virtual texture touched.
    pub resource: ResourceIndex,
    /// The version observed on entry.
    pub input_version: u32,
    /// Whether this access writes a new version.
    pub produces: bool,
    /// The version produced, valid only when `produces`.
    pub output_version: u32,
    /// How the texture is used.
    pub usage: TextureUse,
    /// The subresource range touched.
    pub range: SubresourceRange,
}

/// One recorded buffer access by a pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BufferAccessRecord {
    /// The virtual buffer touched.
    pub resource: ResourceIndex,
    /// The version observed on entry.
    pub input_version: u32,
    /// Whether this access writes a new version.
    pub produces: bool,
    /// The version produced, valid only when `produces`.
    pub output_version: u32,
    /// How the buffer is used.
    pub usage: BufferUse,
}

/// A node in the frame graph: a pass's identity, declared accesses, and its
/// deferred recording closure.
///
/// Not `Clone`/`PartialEq`: the execute closure is move-only. The compiler
/// reads the access vectors and never needs to copy a node.
pub struct PassNode {
    /// A debug name surfaced in diagnostics and GPU tooling.
    pub name: String,
    /// The kind of work performed.
    pub kind: PassKind,
    /// Compiler-steering flags.
    pub flags: PassFlags,
    /// Explicit prerequisites supplied by a declaration frontend.
    ///
    /// Executable passes normally leave this empty and rely on SSA resource
    /// dependencies. Startup topology declarations may use it for ordering
    /// constraints that do not have a resource edge of their own.
    pub(crate) dependencies: Vec<usize>,
    /// Texture accesses in declaration order.
    pub texture_accesses: Vec<TextureAccessRecord>,
    /// Buffer accesses in declaration order.
    pub buffer_accesses: Vec<BufferAccessRecord>,
    /// The deferred recording closure, taken (consumed) at execution time.
    pub(crate) execute: Option<BoxedExecute>,
}

impl PassNode {
    /// Whether this pass is a mandatory culling root.
    ///
    /// Present passes and any pass flagged [`PassFlags::SIDE_EFFECT`] or
    /// [`PassFlags::NEVER_CULL`] survive culling unconditionally; writes to
    /// imported/persistent resources are handled separately by the culler.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.kind == PassKind::Present
            || self.flags.contains(PassFlags::SIDE_EFFECT)
            || self.flags.contains(PassFlags::NEVER_CULL)
    }
}

impl fmt::Debug for PassNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PassNode")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("flags", &self.flags)
            .field("texture_accesses", &self.texture_accesses)
            .field("buffer_accesses", &self.buffer_accesses)
            .field("has_execute", &self.execute.is_some())
            .finish()
    }
}
