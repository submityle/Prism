//! Native material VT upload/publication, independent of the surface queue.
//!
//! Stage a *complete* desired resident set into the inactive slice. Publication
//! and dependent-row refresh must run before material-buffer sync. Bind groups
//! must retain a `NativeVtBinding` lease and submit consumers through its
//! `execute_prepared` method; raw RHI IDs do not own resources.
//! After device recovery, use `recreate_for_device` and refresh rows before
//! buffer sync; constructing a separate owner loses stale-row ownership.
//!
//! Integration order: observe/capture sources before image preparation consumes
//! extraction, stage complete residency, then try publication and refresh rows
//! before material-buffer sync. Obtain a new active lease when its resource IDs
//! change. `Waiting` is not permission to expose the pending slice.
//!
//! Residency refresh uses an atomic, generation/revision-checked texture-only
//! registry transaction. Authored revisions and texture identities, execution
//! bindings and dynamic channels remain unchanged, including at revision MAX.

mod gpu;
mod publication;
mod residency;
#[cfg(feature = "resident-page-streaming")]
mod resident_pack;
#[cfg(feature = "resident-page-streaming")]
mod resident_plugin;
#[cfg(feature = "resident-page-streaming")]
mod resident_upload;
mod scheduler;
mod source;
mod upload_driver;

pub use gpu::{
    MaterialVtLeaseSnapshot, NativeVtBinding, NativeVtConfig, NativeVtGpuConfig,
    NativeVtSubmitError,
};
pub use publication::{NativeMaterialVt, NativeVtPublish, NativeVtResidency};
pub use residency::{
    plan_residency, NativeVtPlan, NativeVtPlanIssue, NativeVtPlanStats, NativeVtRejectedSource,
    NativeVtUnmetDemand,
};
#[cfg(feature = "resident-page-streaming")]
pub use resident_plugin::NativeVtResidentPlugin;
#[cfg(feature = "resident-page-streaming")]
pub use resident_upload::{
    NativeVtResidentError, NativeVtResidentLimits, NativeVtResidentUploader,
};
pub use scheduler::{
    NativeVtFeedbackOrigin, NativeVtScheduleReport, NativeVtScheduler, MAX_NATIVE_VT_FEEDBACK,
};
pub use source::{NativeVtSource, NativeVtSourceRequest, NativeVtSources};
pub use upload_driver::NativeVtUploadDriver;

use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_render_material::{MaterialRegistryError, MaterialVtAtlasUploadError, MaterialVtError};

/// Failures leave the currently published slice and all material rows intact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeVtError {
    InvalidConfig,
    DeviceMismatch,
    DeviceLost,
    UnsupportedFormat,
    ReservedTextureId,
    Source(MaterialVtError),
    ImageMipLayout,
    IdentityExhausted,
    AllocationFailed,
    MissingSource(u32),
    StaleSource(u32),
    StaleSlot(u32),
    AtlasMismatch(u32),
    DuplicatePage(TexturePageKey),
    DuplicateSlot(u32),
    SlotOutOfBounds(u32),
    MissingCoarsestPage(u32),
    InvalidPortableSource(u32),
    TableCapacity,
    Upload(MaterialVtAtlasUploadError),
    PendingUpload,
    SliceBusy,
    RequiresResidentUpload(u32),
    /// Backend upload diagnostic. Source/slot errors retain their typed variants.
    ResidentUpload(String),
    MaterialRegistry(MaterialRegistryError),
}

impl core::fmt::Display for NativeVtError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "native material VT: {self:?}")
    }
}

impl std::error::Error for NativeVtError {}

#[cfg(test)]
mod tests;
