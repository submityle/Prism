use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{
        Buffer, BufferUsages, DrawIndexedIndirectArgs, DrawIndirectArgs, RawBufferVec,
    },
    renderer::{RenderDevice, RenderQueue},
};

use super::rows::{
    RenderVisibilityCounter, RenderVisibilityRange, RenderVisibilityView, RenderVisibilityWorkItem,
};

#[derive(Resource)]
pub(crate) struct UnifiedVisibilityBuffers {
    views: RawBufferVec<RenderVisibilityView>,
    work: RawBufferVec<RenderVisibilityWorkItem>,
    ranges: RawBufferVec<RenderVisibilityRange>,
    gpu_work: RawBufferVec<RenderVisibilityWorkItem>,
    gpu_ranges: RawBufferVec<RenderVisibilityRange>,
    indexed_indirect: RawBufferVec<DrawIndexedIndirectArgs>,
    non_indexed_indirect: RawBufferVec<DrawIndirectArgs>,
    late_indexed_indirect: RawBufferVec<DrawIndexedIndirectArgs>,
    late_non_indexed_indirect: RawBufferVec<DrawIndirectArgs>,
    bin_headers: RawBufferVec<super::rows::RenderDrawBinHeader>,
    late_bin_headers: RawBufferVec<super::rows::RenderDrawBinHeader>,
    late_counters: RawBufferVec<RenderVisibilityCounter>,
    candidate_bins: RawBufferVec<u32>,
    previous_lods: RawBufferVec<u32>,
    counters: RawBufferVec<RenderVisibilityCounter>,
    overflow: RawBufferVec<u32>,
    gpu_work_capacity: usize,
    history_slots_per_view: u32,
    history_view_handles: Vec<(u32, u32)>,
    version: u32,
}

impl FromWorld for UnifiedVisibilityBuffers {
    fn from_world(_: &mut World) -> Self {
        let mut views = RawBufferVec::new(BufferUsages::STORAGE);
        views.set_label(Some("prism visibility views"));
        let mut work = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        work.set_label(Some("prism visibility work"));
        let mut ranges = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        ranges.set_label(Some("prism visibility ranges"));
        let mut counters = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        counters.set_label(Some("prism visibility counters"));
        let mut overflow = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        overflow.set_label(Some("prism visibility overflow"));
        let mut gpu_work = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        gpu_work.set_label(Some("prism visibility gpu parity work"));
        let mut gpu_ranges = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        gpu_ranges.set_label(Some("prism visibility gpu parity ranges"));
        let mut indexed_indirect = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::INDIRECT);
        indexed_indirect.set_label(Some("prism visibility indexed indirect"));
        let mut non_indexed_indirect =
            RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::INDIRECT);
        non_indexed_indirect.set_label(Some("prism visibility non-indexed indirect"));
        let mut late_indexed_indirect =
            RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::INDIRECT);
        late_indexed_indirect.set_label(Some("prism late visibility indexed indirect"));
        let mut late_non_indexed_indirect =
            RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::INDIRECT);
        late_non_indexed_indirect.set_label(Some("prism late visibility non-indexed indirect"));
        let mut previous_lods = RawBufferVec::new(BufferUsages::STORAGE);
        previous_lods.set_label(Some("prism visibility previous lods"));
        let mut bin_headers = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        bin_headers.set_label(Some("prism visibility draw bin headers"));
        let mut late_bin_headers =
            RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        late_bin_headers.set_label(Some("prism late visibility draw bin headers"));
        let mut late_counters = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        late_counters.set_label(Some("prism late visibility counters"));
        let mut candidate_bins = RawBufferVec::new(BufferUsages::STORAGE);
        candidate_bins.set_label(Some("prism visibility candidate bins"));
        Self {
            views,
            work,
            ranges,
            gpu_work,
            gpu_ranges,
            indexed_indirect,
            non_indexed_indirect,
            late_indexed_indirect,
            late_non_indexed_indirect,
            bin_headers,
            late_bin_headers,
            late_counters,
            candidate_bins,
            previous_lods,
            counters,
            overflow,
            gpu_work_capacity: 0,
            history_slots_per_view: 0,
            history_view_handles: Vec::new(),
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
        gpu_slots_per_view: u32,
    ) {
        let staged_views: Vec<_> = views.into_iter().collect();
        let history_views: Vec<_> = staged_views
            .iter()
            .map(|view| (view.handle_index, view.handle_generation))
            .collect();
        let history_compatible = self.history_slots_per_view == gpu_slots_per_view
            && self.history_view_handles == history_views;
        self.views.clear();
        self.work.clear();
        self.ranges.clear();
        self.gpu_ranges.clear();
        self.indexed_indirect.clear();
        self.non_indexed_indirect.clear();
        self.late_indexed_indirect.clear();
        self.late_non_indexed_indirect.clear();
        self.bin_headers.clear();
        self.late_bin_headers.clear();
        self.late_counters.clear();
        self.candidate_bins.clear();
        self.counters.clear();
        self.overflow.clear();
        self.views.extend(staged_views);
        self.work.extend(work);
        self.ranges.extend(ranges);
        self.gpu_ranges
            .extend(
                self.ranges
                    .values()
                    .iter()
                    .enumerate()
                    .map(|(view_index, range)| RenderVisibilityRange {
                        start: (view_index as u32).saturating_mul(gpu_slots_per_view),
                        count: 0,
                        ..*range
                    }),
            );
        self.gpu_work_capacity = self.views.len().saturating_mul(gpu_slots_per_view as usize);
        self.indexed_indirect
            .extend((0..self.gpu_work_capacity).map(|_| DrawIndexedIndirectArgs::default()));
        self.non_indexed_indirect
            .extend((0..self.gpu_work_capacity).map(|_| DrawIndirectArgs::default()));
        self.late_indexed_indirect
            .extend((0..self.gpu_work_capacity).map(|_| DrawIndexedIndirectArgs::default()));
        self.late_non_indexed_indirect
            .extend((0..self.gpu_work_capacity).map(|_| DrawIndirectArgs::default()));
        if !history_compatible {
            self.previous_lods.clear();
        }
        if self.previous_lods.len() < self.gpu_work_capacity {
            self.previous_lods.extend(
                (self.previous_lods.len()..self.gpu_work_capacity).map(|_| u32::MAX),
            );
        }
        self.previous_lods.truncate(self.gpu_work_capacity);
        self.history_slots_per_view = gpu_slots_per_view;
        self.history_view_handles = history_views;
        self.counters
            .extend((0..self.views.len()).map(|_| RenderVisibilityCounter::default()));
        self.late_counters
            .extend((0..self.views.len()).map(|_| RenderVisibilityCounter::default()));
        self.overflow.extend((0..self.views.len()).map(|_| 0));
    }

    pub(crate) fn stage_draw_bins(&mut self, views: &[prism_render_visibility::ViewDrawBins]) {
        self.bin_headers.clear();
        self.late_bin_headers.clear();
        self.candidate_bins.clear();
        for view in views {
            self.bin_headers.extend(view.bins.iter().copied().map(|range| {
                super::rows::RenderDrawBinHeader::from(
                    prism_render_visibility::GpuDrawBinHeader::from_range(view.view, range),
                )
            }));
            self.late_bin_headers.extend(view.bins.iter().copied().map(|range| {
                super::rows::RenderDrawBinHeader::from(
                    prism_render_visibility::GpuDrawBinHeader::from_range(view.view, range),
                )
            }));
            self.candidate_bins.extend(view.candidate_bins.iter().copied());
        }
    }

    #[cfg(test)]
    pub(crate) fn draw_bin_headers(&self) -> &[super::rows::RenderDrawBinHeader] {
        self.bin_headers.values()
    }

    #[cfg(test)]
    pub(crate) fn late_draw_bin_headers(&self) -> &[super::rows::RenderDrawBinHeader] {
        self.late_bin_headers.values()
    }

    #[cfg(test)]
    pub(crate) fn late_counters(&self) -> &[RenderVisibilityCounter] {
        self.late_counters.values()
    }

    pub(crate) fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.views.write_buffer(device, queue);
        self.work.write_buffer(device, queue);
        self.ranges.write_buffer(device, queue);
        self.gpu_work.reserve(self.gpu_work_capacity.max(1), device);
        self.gpu_ranges.write_buffer(device, queue);
        self.indexed_indirect.write_buffer(device, queue);
        self.non_indexed_indirect.write_buffer(device, queue);
        self.late_indexed_indirect.write_buffer(device, queue);
        self.late_non_indexed_indirect.write_buffer(device, queue);
        self.previous_lods.write_buffer(device, queue);
        self.bin_headers.write_buffer(device, queue);
        self.late_bin_headers.write_buffer(device, queue);
        self.late_counters.write_buffer(device, queue);
        self.candidate_bins.write_buffer(device, queue);
        self.counters.write_buffer(device, queue);
        self.overflow.write_buffer(device, queue);
    }

    pub(crate) fn reset_after_device_loss(&mut self) {
        self.version = self.version.wrapping_add(1).max(1);
        for lod in self.previous_lods.values_mut() {
            *lod = u32::MAX;
        }
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

    pub(crate) fn compute_buffers(
        &self,
    ) -> Option<(&Buffer, &Buffer, &Buffer, &Buffer, &Buffer, &Buffer, &Buffer, &Buffer, &Buffer, &Buffer)> {
        Some((
            self.views.buffer()?,
            self.counters.buffer()?,
            self.gpu_work.buffer()?,
            self.gpu_ranges.buffer()?,
            self.indexed_indirect.buffer()?,
            self.non_indexed_indirect.buffer()?,
            self.overflow.buffer()?,
            self.previous_lods.buffer()?,
            self.bin_headers.buffer()?,
            self.candidate_bins.buffer()?,
        ))
    }

    pub(crate) fn candidate_bin_buffer(&self) -> Option<&Buffer> {
        self.candidate_bins.buffer()
    }

    pub(crate) fn gpu_slots_per_view(&self) -> u32 {
        if self.views.is_empty() {
            0
        } else {
            (self.gpu_work_capacity / self.views.len()) as u32
        }
    }

    pub(crate) fn parity_readback_buffers(
        &self,
    ) -> Option<(&Buffer, &Buffer, &Buffer, &Buffer, &Buffer)> {
        Some((
            self.counters.buffer()?,
            self.gpu_work.buffer()?,
            self.bin_headers.buffer()?,
            self.late_counters.buffer()?,
            self.late_bin_headers.buffer()?,
        ))
    }

    pub(crate) fn indirect(&self) -> Option<(&Buffer, &Buffer)> {
        self.indirect_for(false)
    }

    pub(crate) fn draw_bin_buffer(&self) -> Option<&Buffer> {
        self.bin_headers.buffer()
    }

    pub(crate) fn indirect_for(&self, late: bool) -> Option<(&Buffer, &Buffer)> {
        if late {
            Some((
                self.late_indexed_indirect.buffer()?,
                self.late_non_indexed_indirect.buffer()?,
            ))
        } else {
            Some((self.indexed_indirect.buffer()?, self.non_indexed_indirect.buffer()?))
        }
    }

    pub(crate) fn late_compute_buffers(
        &self,
    ) -> Option<(&Buffer, &Buffer, &Buffer, &Buffer)> {
        Some((
            self.late_counters.buffer()?,
            self.late_indexed_indirect.buffer()?,
            self.late_non_indexed_indirect.buffer()?,
            self.late_bin_headers.buffer()?,
        ))
    }

}

#[cfg(test)]
mod tests {
    use prism_render_architecture::abi::GenerationalHandle;

    use super::*;

    #[test]
    fn gpu_parity_ranges_are_disjoint_from_cpu_truth() {
        let mut world = World::new();
        let mut buffers = UnifiedVisibilityBuffers::from_world(&mut world);
        let view = RenderVisibilityView::default();
        let cpu_work = RenderVisibilityWorkItem {
            scene_index: 7,
            ..Default::default()
        };
        let range = RenderVisibilityRange::new(
            GenerationalHandle {
                index: 1,
                generation: 1,
            },
            0,
            1,
        );
        buffers.stage([view, view], [cpu_work], [range, range], 8);

        assert_eq!(buffers.work.values(), &[cpu_work]);
        assert_eq!(buffers.gpu_ranges.values()[0].start, 0);
        assert_eq!(buffers.gpu_ranges.values()[1].start, 8);
        assert_eq!(buffers.gpu_work_capacity, 16);
        assert_eq!(buffers.gpu_slots_per_view(), 8);
        assert_eq!(buffers.indexed_indirect.len(), 16);
        assert_eq!(buffers.non_indexed_indirect.len(), 16);
        assert_eq!(buffers.previous_lods.len(), 16);
    }

    #[test]
    fn lod_history_survives_stable_views_and_resets_on_identity_change() {
        let mut world = World::new();
        let mut buffers = UnifiedVisibilityBuffers::from_world(&mut world);
        let view = RenderVisibilityView {
            handle_index: 1,
            handle_generation: 1,
            ..Default::default()
        };
        buffers.stage([view], [], [], 4);
        buffers.previous_lods.values_mut()[2] = 3;
        buffers.stage([view], [], [], 4);
        assert_eq!(buffers.previous_lods.values()[2], 3);

        buffers.stage(
            [RenderVisibilityView {
                handle_generation: 2,
                ..view
            }],
            [],
            [],
            4,
        );
        assert!(buffers.previous_lods.values().iter().all(|lod| *lod == u32::MAX));
    }

    #[test]
    fn draw_bin_headers_and_candidate_tables_use_global_offsets() {
        let mut world = World::new();
        let mut buffers = UnifiedVisibilityBuffers::from_world(&mut world);
        let key = prism_render_visibility::DrawBinKey {
            geometry: GenerationalHandle { index: 2, generation: 1 },
            lod_or_cluster: 0,
            pipeline_class: 4,
            vertex_buffer_class: 5,
            index_buffer_class: 6,
            indexed: true,
            primitive_kind: prism_render_architecture::geometry::GeometryPrimitiveKind::Indexed,
            pass_mask: prism_render_visibility::RenderPassMask::OPAQUE.0,
        };
        let mut first = prism_render_visibility::build_view_draw_bins(
            GenerationalHandle { index: 10, generation: 1 },
            4,
            [prism_render_visibility::DrawBinCandidate {
                scene: GenerationalHandle { index: 1, generation: 1 },
                key,
                visibility_stages: prism_render_visibility::VisibilityStageMask::EARLY,
            }],
        );
        let mut second = prism_render_visibility::build_view_draw_bins(
            GenerationalHandle { index: 11, generation: 1 },
            4,
            [prism_render_visibility::DrawBinCandidate {
                scene: GenerationalHandle { index: 2, generation: 1 },
                key,
                visibility_stages: prism_render_visibility::VisibilityStageMask::EARLY,
            }],
        );
        first.global_bin_start = 0;
        first.global_candidate_start = 0;
        first.command_buffer_start = 0;
        second.global_bin_start = 1;
        second.global_candidate_start = 4;
        second.command_buffer_start = 1;
        buffers.stage_draw_bins(&[first, second]);
        assert_eq!(buffers.draw_bin_headers().len(), 2);
        assert_eq!(buffers.candidate_bins.values().len(), 8);
        assert_eq!(buffers.candidate_bins.values()[1], 0);
        assert_eq!(buffers.candidate_bins.values()[6], 0);
        assert_eq!(buffers.late_draw_bin_headers().len(), 2);
        assert!(buffers
            .late_draw_bin_headers()
            .iter()
            .all(|header| header.command_count == 0));
    }

    #[test]
    fn staging_resets_one_late_counter_per_view() {
        let mut world = World::new();
        let mut buffers = UnifiedVisibilityBuffers::from_world(&mut world);
        buffers.stage(
            [RenderVisibilityView::default(), RenderVisibilityView::default()],
            [],
            [],
            4,
        );
        assert_eq!(buffers.late_counters().len(), 2);
        assert!(buffers
            .late_counters()
            .iter()
            .all(|counter| *counter == RenderVisibilityCounter::default()));
    }
}
