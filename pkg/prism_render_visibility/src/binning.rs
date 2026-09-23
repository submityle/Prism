use alloc::{collections::BTreeMap, vec::Vec};
use crate::ViewHandle;
use prism_render_architecture::{
    geometry::GeometryPrimitiveKind,
    gpu_scene::{GeometryHandle, SceneHandle},
};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct DrawBinKey {
    pub geometry: GeometryHandle,
    pub lod_or_cluster: u32,
    pub pipeline_class: u32,
    pub vertex_buffer_class: u32,
    pub index_buffer_class: u32,
    pub indexed: bool,
    pub primitive_kind: GeometryPrimitiveKind,
    pub pass_mask: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawBinCandidate {
    pub scene: SceneHandle,
    pub key: DrawBinKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawBinRange {
    pub key: DrawBinKey,
    pub command_start: u32,
    pub command_capacity: u32,
    pub representative_scene: SceneHandle,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ViewDrawBins {
    pub view: ViewHandle,
    pub bins: Vec<DrawBinRange>,
    pub candidate_bins: Vec<u32>,
    pub command_count: u32,
    /// First command slot owned by this view in the indirect command buffers.
    /// This is independent from `global_candidate_start`: candidate lookup is
    /// scene-capacity-strided, while commands are GPU-work-capacity-strided.
    pub command_buffer_start: u32,
    pub global_bin_start: u32,
    pub global_candidate_start: u32,
}

pub const DRAW_BIN_HEADER_WORDS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDrawBinHeader {
    pub geometry_index: u32,
    pub geometry_generation: u32,
    pub pipeline_class: u32,
    pub vertex_buffer_class: u32,
    pub index_buffer_class: u32,
    pub indexed: u32,
    pub command_start: u32,
    pub command_capacity: u32,
    pub command_count: u32,
    pub view_index: u32,
    pub view_generation: u32,
    pub lod_or_cluster: u32,
    pub primitive_kind: u32,
    pub _padding_tail: [u32; 3],
}

impl GpuDrawBinHeader {
    pub fn from_range(view: ViewHandle, range: DrawBinRange) -> Self {
        Self {
            geometry_index: range.key.geometry.index,
            geometry_generation: range.key.geometry.generation,
            pipeline_class: range.key.pipeline_class,
            vertex_buffer_class: range.key.vertex_buffer_class,
            index_buffer_class: range.key.index_buffer_class,
            indexed: u32::from(range.key.indexed),
            command_start: range.command_start,
            command_capacity: range.command_capacity,
            command_count: 0,
            view_index: view.index,
            view_generation: view.generation,
            lod_or_cluster: range.key.lod_or_cluster,
            primitive_kind: match range.key.primitive_kind {
                GeometryPrimitiveKind::Indexed => 0,
                GeometryPrimitiveKind::NonIndexed => 1,
            },
            _padding_tail: [0; 3],
        }
    }
}

pub fn build_view_draw_bins(
    view: ViewHandle,
    scene_capacity: u32,
    candidates: impl IntoIterator<Item = DrawBinCandidate>,
) -> ViewDrawBins {
    let mut grouped = BTreeMap::<DrawBinKey, Vec<SceneHandle>>::new();
    for candidate in candidates {
        grouped.entry(candidate.key).or_default().push(candidate.scene);
    }
    let mut bins = Vec::with_capacity(grouped.len());
    let mut candidate_bins = vec![u32::MAX; scene_capacity as usize];
    let mut command_start = 0_u32;
    for (key, scenes) in grouped {
        let bin_index = bins.len() as u32;
        for scene in &scenes {
            if let Some(slot) = candidate_bins.get_mut(scene.index as usize) {
                *slot = bin_index;
            }
        }
        let command_capacity = scenes.len() as u32;
        bins.push(DrawBinRange {
            key,
            command_start,
            command_capacity,
            representative_scene: scenes[0],
        });
        command_start = command_start.saturating_add(command_capacity);
    }
    ViewDrawBins {
        view,
        bins,
        candidate_bins,
        command_count: command_start,
        command_buffer_start: 0,
        global_bin_start: 0,
        global_candidate_start: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::abi::GenerationalHandle;

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle { index, generation: 1 }
    }

    #[test]
    fn bins_are_deterministic_disjoint_and_cover_candidates() {
        let key = |geometry| DrawBinKey {
            geometry: handle(geometry),
            lod_or_cluster: 0,
            pipeline_class: geometry,
            vertex_buffer_class: 1,
            index_buffer_class: 2,
            indexed: true,
            primitive_kind: GeometryPrimitiveKind::Indexed,
            pass_mask: crate::RenderPassMask::OPAQUE.0,
        };
        let bins = build_view_draw_bins(
            handle(9),
            8,
            [
                DrawBinCandidate {
                    scene: handle(5),
                    key: key(2),
                },
                DrawBinCandidate {
                    scene: handle(1),
                    key: key(1),
                },
                DrawBinCandidate {
                    scene: handle(3),
                    key: key(2),
                },
            ],
        );
        assert_eq!(bins.bins.len(), 2);
        assert_eq!(bins.bins[0].command_start, 0);
        assert_eq!(bins.bins[0].command_capacity, 1);
        assert_eq!(bins.bins[1].command_start, 1);
        assert_eq!(bins.bins[1].command_capacity, 2);
        assert_eq!(bins.command_count, 3);
        assert_eq!(bins.candidate_bins[1], 0);
        assert_eq!(bins.candidate_bins[3], 1);
        assert_eq!(bins.candidate_bins[5], 1);
        assert_eq!(bins.bins[1].representative_scene, handle(5));
        let header = GpuDrawBinHeader::from_range(bins.view, bins.bins[1]);
        assert_eq!(header.command_start, 1);
        assert_eq!(header.command_capacity, 2);
        assert_eq!(size_of::<GpuDrawBinHeader>(), DRAW_BIN_HEADER_WORDS * 4);
    }
}
