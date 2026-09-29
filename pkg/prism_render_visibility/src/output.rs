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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RenderPassMask, VisibilityStageMask, WorkSortKey};
    use prism_render_architecture::abi::GenerationalHandle;

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle {
            index,
            generation: 1,
        }
    }

    fn work_item(sort_key: u64) -> GpuRenderWorkItem {
        GpuRenderWorkItem {
            scene: handle(sort_key as u32),
            geometry: handle(1),
            material: handle(1),
            lod_or_cluster: 0,
            pass_mask: RenderPassMask::OPAQUE,
            visibility_stages: VisibilityStageMask::EARLY,
            sort_key: WorkSortKey(sort_key),
        }
    }

    #[test]
    fn empty_view_publishes_zero_length_ranges() {
        let mut frame = VisibilityFrame::default();
        frame.push_view(handle(4), Vec::new(), VisibilityDiagnostics::default());

        assert!(frame.work_items.is_empty());
        let output = &frame.views[&handle(4)];
        assert_eq!(output.visible_instances, BufferRange { start: 0, count: 0 });
        assert_eq!(output.raster_commands.count, 0);
        assert_eq!(output.shadow_commands.count, 0);
        // A view with no work still records diagnostics.
        assert!(frame.diagnostics.contains_key(&handle(4)));
    }

    #[test]
    fn push_view_sorts_work_and_derives_command_counts() {
        let mut frame = VisibilityFrame::default();
        frame.push_view(
            handle(1),
            vec![work_item(30), work_item(10), work_item(20)],
            VisibilityDiagnostics {
                shadow_casters: 2,
                ray_scene_instances: 1,
                material_bins: 3,
                ..Default::default()
            },
        );

        // Work is sorted ascending by sort key before publishing.
        assert_eq!(frame.work_items[0].sort_key, WorkSortKey(10));
        assert_eq!(frame.work_items[1].sort_key, WorkSortKey(20));
        assert_eq!(frame.work_items[2].sort_key, WorkSortKey(30));

        let output = &frame.views[&handle(1)];
        assert_eq!(output.visible_instances, BufferRange { start: 0, count: 3 });
        assert_eq!(output.raster_commands, BufferRange { start: 0, count: 3 });
        // Pass-specific command counts come straight from diagnostics.
        assert_eq!(output.shadow_commands, BufferRange { start: 0, count: 2 });
        assert_eq!(output.ray_scene_updates, BufferRange { start: 0, count: 1 });
        assert_eq!(output.material_bins, BufferRange { start: 0, count: 3 });
    }

    #[test]
    fn successive_views_receive_contiguous_disjoint_ranges() {
        let mut frame = VisibilityFrame::default();
        frame.push_view(
            handle(1),
            vec![work_item(1), work_item(2)],
            VisibilityDiagnostics::default(),
        );
        frame.push_view(
            handle(2),
            vec![work_item(3), work_item(4), work_item(5)],
            VisibilityDiagnostics::default(),
        );

        let first = frame.views[&handle(1)].visible_instances;
        let second = frame.views[&handle(2)].visible_instances;
        assert_eq!(first, BufferRange { start: 0, count: 2 });
        // The second view starts exactly where the first ended: no overlap,
        // no gap.
        assert_eq!(second, BufferRange { start: 2, count: 3 });
        assert_eq!(second.start, first.start + first.count);
        assert_eq!(frame.work_items.len(), 5);
    }
}
