use crate::{GpuRenderWorkItem, ViewHandle, VisibilityDiagnostics};
use alloc::{collections::BTreeMap, vec::Vec};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BufferRange {
    pub start: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ViewVisibilityOutput {
    pub visible_instances: BufferRange,
    pub visible_clusters: BufferRange,
    pub material_bins: BufferRange,
    pub raster_commands: BufferRange,
    pub shadow_commands: BufferRange,
    pub ray_scene_updates: BufferRange,
    pub streaming_feedback: BufferRange,
    pub statistics: BufferRange,
}

#[derive(Clone, Debug, Default)]
pub struct VisibilityFrame {
    pub work_items: Vec<GpuRenderWorkItem>,
    pub views: BTreeMap<ViewHandle, ViewVisibilityOutput>,
    pub diagnostics: BTreeMap<ViewHandle, VisibilityDiagnostics>,
}

impl VisibilityFrame {
    pub fn push_view(
        &mut self,
        view: ViewHandle,
        mut work: Vec<GpuRenderWorkItem>,
        diagnostics: VisibilityDiagnostics,
    ) {
        work.sort_by_key(|item| item.sort_key);
        let start = self.work_items.len() as u32;
        let count = work.len() as u32;
        self.work_items.extend(work);
        self.views.insert(
            view,
            ViewVisibilityOutput {
                visible_instances: BufferRange { start, count },
                raster_commands: BufferRange { start, count },
                shadow_commands: BufferRange {
                    start,
                    count: diagnostics.shadow_casters,
                },
                ray_scene_updates: BufferRange {
                    start,
                    count: diagnostics.ray_scene_instances,
                },
                material_bins: BufferRange {
                    start,
                    count: diagnostics.material_bins,
                },
                ..Default::default()
            },
        );
        self.diagnostics.insert(view, diagnostics);
    }
}
