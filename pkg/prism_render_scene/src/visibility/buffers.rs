use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_render::{
    render_resource::{Buffer, BufferUsages, RawBufferVec},
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
    counters: RawBufferVec<RenderVisibilityCounter>,
    overflow: RawBufferVec<u32>,
    gpu_work_capacity: usize,
    version: u32,
}

impl FromWorld for UnifiedVisibilityBuffers {
    fn from_world(_: &mut World) -> Self {
        let mut views = RawBufferVec::new(BufferUsages::STORAGE);
        views.set_label(Some("prism visibility views"));
        let mut work = RawBufferVec::new(
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
        );
        work.set_label(Some("prism visibility work"));
        let mut ranges = RawBufferVec::new(
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
        );
        ranges.set_label(Some("prism visibility ranges"));
        let mut counters = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        counters.set_label(Some("prism visibility counters"));
        let mut overflow = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        overflow.set_label(Some("prism visibility overflow"));
        let mut gpu_work = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        gpu_work.set_label(Some("prism visibility gpu parity work"));
        let mut gpu_ranges = RawBufferVec::new(BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        gpu_ranges.set_label(Some("prism visibility gpu parity ranges"));
        Self {
            views,
            work,
            ranges,
            gpu_work,
            gpu_ranges,
            counters,
            overflow,
            gpu_work_capacity: 0,
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
        self.views.clear();
        self.work.clear();
        self.ranges.clear();
        self.gpu_ranges.clear();
        self.counters.clear();
        self.overflow.clear();
        self.views.extend(views);
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
        self.counters
            .extend((0..self.views.len()).map(|_| RenderVisibilityCounter::default()));
        self.overflow.extend((0..self.views.len()).map(|_| 0));
    }

    pub(crate) fn upload(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.views.write_buffer(device, queue);
        self.work.write_buffer(device, queue);
        self.ranges.write_buffer(device, queue);
        self.gpu_work.reserve(self.gpu_work_capacity.max(1), device);
        self.gpu_ranges.write_buffer(device, queue);
        self.counters.write_buffer(device, queue);
        self.overflow.write_buffer(device, queue);
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

    pub(crate) fn compute_buffers(&self) -> Option<(&Buffer, &Buffer, &Buffer, &Buffer, &Buffer)> {
        Some((
            self.views.buffer()?,
            self.counters.buffer()?,
            self.gpu_work.buffer()?,
            self.gpu_ranges.buffer()?,
            self.overflow.buffer()?,
        ))
    }

    pub(crate) fn gpu_slots_per_view(&self) -> u32 {
        if self.views.is_empty() {
            0
        } else {
            (self.gpu_work_capacity / self.views.len()) as u32
        }
    }

    pub(crate) fn parity_readback_buffers(&self) -> Option<(&Buffer, &Buffer)> {
        Some((self.counters.buffer()?, self.gpu_work.buffer()?))
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
    }
}
