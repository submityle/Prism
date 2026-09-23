use alloc::{collections::BTreeMap, vec::Vec};
use crate::ViewHandle;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneHandle};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct DrawBinKey {
    pub geometry: GeometryHandle,
    pub pipeline_class: u32,
    pub vertex_buffer_class: u32,
    pub index_buffer_class: u32,
    pub indexed: bool,
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
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ViewDrawBins {
    pub view: ViewHandle,
    pub bins: Vec<DrawBinRange>,
    pub candidate_bins: Vec<u32>,
    pub command_count: u32,
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
        });
        command_start = command_start.saturating_add(command_capacity);
    }
    ViewDrawBins {
        view,
        bins,
        candidate_bins,
        command_count: command_start,
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
            pipeline_class: geometry,
            vertex_buffer_class: 1,
            index_buffer_class: 2,
            indexed: true,
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
    }
}
