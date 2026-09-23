use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
    renderer::{RenderDevice, RenderQueue},
};

use super::rows::{RenderVisibilityRange, RenderVisibilityView, RenderVisibilityWorkItem};

#[derive(Resource)]
pub(crate) struct UnifiedVisibilityBuffers {
    views: RawBufferVec<RenderVisibilityView>,
    work: RawBufferVec<RenderVisibilityWorkItem>,
    ranges: RawBufferVec<RenderVisibilityRange>,
    version: u32,
}

impl FromWorld for UnifiedVisibilityBuffers {
    fn from_world(_: &mut World) -> Self {
        let mut views = RawBufferVec::new(BufferUsages::STORAGE);
        views.set_label(Some("prism visibility views"));
        let mut work = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::INDIRECT);
        work.set_label(Some("prism visibility work"));
        let mut ranges = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::INDIRECT);
        ranges.set_label(Some("prism visibility ranges"));
        Self {
            views,
            work,
            ranges,
            version: 1,
        }
    }
}

impl UnifiedVisibilityBuffers {
    pub(crate) fn stage(
        &mut self,
        views: impl IntoIterator<Item = RenderVisibilityView>,
        work: impl IntoIterator<Item = RenderVisibilityWorkItem>,
        ranges: impl IntoIterator<Item = RenderVisibilityRange>,
    ) {
        self.views.clear();
        self.work.clear();
        self.ranges.clear();
        self.views.extend(views);
        self.work.extend(work);
        self.ranges.extend(ranges);
    }

    pub(crate) fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.views.write_buffer(device, queue);
        self.work.write_buffer(device, queue);
        self.ranges.write_buffer(device, queue);
    }

    pub(crate) fn reset_after_device_loss(&mut self) {
        self.version = self.version.wrapping_add(1).max(1);
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn buffers(&self) -> Option<(&Buffer, &Buffer, &Buffer)> {
        Some((
            self.views.buffer()?,
            self.work.buffer()?,
            self.ranges.buffer()?,
        ))
    }
}
