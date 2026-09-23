use bevy_ecs::system::SystemParam;
use bevy_render::render_resource::Buffer;
use bevy_render::view::RetainedViewEntity;
use prism_render_architecture::gpu_scene::GeometryHandle;
use prism_render_visibility::{BufferRange, ViewDrawBins, ViewHandle, VisibilityFrame};

use super::{buffers::UnifiedVisibilityBuffers, runtime::UnifiedVisibilityState};

#[derive(SystemParam)]
pub struct UnifiedVisibilityReader<'w> {
    state: bevy_ecs::prelude::Res<'w, UnifiedVisibilityState>,
    buffers: bevy_ecs::prelude::Res<'w, UnifiedVisibilityBuffers>,
    geometry: bevy_ecs::prelude::Res<'w, crate::RenderGeometryRegistry>,
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

    pub fn view_handle(&self, retained: RetainedViewEntity) -> Option<ViewHandle> {
        self.state
            .frame
            .views
            .keys()
            .copied()
            .find(|handle| self.state.retained_view(*handle) == Some(retained))
    }

    pub fn buffers(&self) -> Option<UnifiedVisibilityBufferBindings<'_>> {
        let (views, work, ranges) = self.buffers.buffers()?;
        let (indexed_indirect, non_indexed_indirect) = self.buffers.indirect()?;
        Some(UnifiedVisibilityBufferBindings {
            views,
            work,
            ranges,
            indexed_indirect,
            non_indexed_indirect,
            version: self.buffers.version(),
        })
    }

    pub fn geometry_buffer_classes(&self, geometry: GeometryHandle) -> Option<(u32, u32)> {
        self.geometry.buffer_classes(geometry)
    }

    pub fn draw_bins(&self, view: ViewHandle) -> Option<&ViewDrawBins> {
        self.state.draw_bins.iter().find(|bins| bins.view == view)
    }
}

pub struct UnifiedVisibilityBufferBindings<'a> {
    pub views: &'a Buffer,
    pub work: &'a Buffer,
    pub ranges: &'a Buffer,
    pub indexed_indirect: &'a Buffer,
    pub non_indexed_indirect: &'a Buffer,
    pub version: u32,
}
