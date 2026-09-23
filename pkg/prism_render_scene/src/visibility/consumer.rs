use bevy_ecs::system::SystemParam;
use bevy_render::render_resource::Buffer;
use prism_render_visibility::{BufferRange, ViewHandle, VisibilityFrame};

use super::{buffers::UnifiedVisibilityBuffers, runtime::UnifiedVisibilityState};

#[derive(SystemParam)]
pub struct UnifiedVisibilityReader<'w> {
    state: bevy_ecs::prelude::Res<'w, UnifiedVisibilityState>,
    buffers: bevy_ecs::prelude::Res<'w, UnifiedVisibilityBuffers>,
}

impl UnifiedVisibilityReader<'_> {
    pub fn frame(&self) -> &VisibilityFrame {
        &self.state.frame
    }

    pub fn work_range(&self, view: ViewHandle) -> Option<BufferRange> {
        self.state
            .frame
            .views
            .get(&view)
            .map(|output| output.visible_instances)
    }

    pub fn buffers(&self) -> Option<UnifiedVisibilityBufferBindings<'_>> {
        let (views, work, ranges) = self.buffers.buffers()?;
        Some(UnifiedVisibilityBufferBindings {
            views,
            work,
            ranges,
            version: self.buffers.version(),
        })
    }
}

pub struct UnifiedVisibilityBufferBindings<'a> {
    pub views: &'a Buffer,
    pub work: &'a Buffer,
    pub ranges: &'a Buffer,
    pub version: u32,
}
