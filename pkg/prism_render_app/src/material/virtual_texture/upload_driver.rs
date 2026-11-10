//! Backend-selected staging without specializing the generic frame scheduler.

use prism_ecs::prelude::Resource;
use prism_render_driver::{Instance, RenderDevice};
use prism_render_runtime::RenderRuntime;

use super::{NativeMaterialVt, NativeVtError, NativeVtResidency, NativeVtSources};
use crate::material::TextureSlotHeap;

type Stage<I> = dyn FnMut(
        &RenderRuntime<I>,
        &mut NativeMaterialVt<<I as Instance>::Device>,
        &NativeVtSources,
        &TextureSlotHeap,
        &NativeVtResidency,
    ) -> Result<(), NativeVtError>
    + Send
    + Sync;

/// Optional backend staging policy used by the production VT prepare system.
/// The callback must stage the complete requested snapshot on `owner` and must
/// not publish it. Completion qualification, obsolete-work cancellation, source
/// validation and atomic material-row publication remain in the scheduler.
///
/// The callback receives the committed runtime every time; backend caches must
/// authenticate that device and rebuild after recovery, never use a factory's
/// uncommitted candidate. Absence selects ordinary CPU-source staging.
pub struct NativeVtUploadDriver<I: Instance + 'static>
where
    I::Device: RenderDevice + Send + Sync + 'static,
{
    stage: Box<Stage<I>>,
}

impl<I: Instance + 'static> Resource for NativeVtUploadDriver<I> where
    I::Device: RenderDevice + Send + Sync + 'static
{
}

impl<I: Instance + 'static> NativeVtUploadDriver<I>
where
    I::Device: RenderDevice + Send + Sync + 'static,
{
    pub fn new(
        stage: impl FnMut(
                &RenderRuntime<I>,
                &mut NativeMaterialVt<I::Device>,
                &NativeVtSources,
                &TextureSlotHeap,
                &NativeVtResidency,
            ) -> Result<(), NativeVtError>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            stage: Box::new(stage),
        }
    }

    pub(crate) fn stage(
        &mut self,
        runtime: &RenderRuntime<I>,
        owner: &mut NativeMaterialVt<I::Device>,
        sources: &NativeVtSources,
        heap: &TextureSlotHeap,
        residency: &NativeVtResidency,
    ) -> Result<(), NativeVtError> {
        (self.stage)(runtime, owner, sources, heap, residency)
    }
}
