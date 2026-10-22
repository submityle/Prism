//! Deferred texture-only bind groups, realized before pass recording begins.

use alloc::string::String;
use alloc::vec::Vec;

use prism_render_driver::{BindGroupId, BindGroupLayoutId};

use crate::handle::{TextureBindGroupHandle, TextureHandle};

/// A CPU declaration plus its immutable, record-time resolution.
pub(crate) struct TextureBindGroup {
    pub(crate) handle: TextureBindGroupHandle,
    pub(crate) layout: BindGroupLayoutId,
    pub(crate) entries: Vec<(u32, TextureHandle)>,
    pub(crate) label: Option<String>,
    pub(crate) realized: Option<BindGroupId>,
}
