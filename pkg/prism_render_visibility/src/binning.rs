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
    pub visibility_stages: crate::VisibilityStageMask,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawBinRange {
    pub key: DrawBinKey,
    pub command_start: u32,
    pub command_capacity: u32,
    pub representative_scene: SceneHandle,
    pub visibility_stages: crate::VisibilityStageMask,
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
    let mut grouped = BTreeMap::<DrawBinKey, Vec<(SceneHandle, crate::VisibilityStageMask)>>::new();
    for candidate in candidates {
        grouped
            .entry(candidate.key)
            .or_default()
            .push((candidate.scene, candidate.visibility_stages));
    }
    let mut bins = Vec::with_capacity(grouped.len());
    let mut candidate_bins = vec![u32::MAX; scene_capacity as usize];
    let mut command_start = 0_u32;
    for (key, scenes) in grouped {
        let bin_index = bins.len() as u32;
        for (scene, _) in &scenes {
            if let Some(slot) = candidate_bins.get_mut(scene.index as usize) {
                *slot = bin_index;
            }
        }
        let command_capacity = scenes.len() as u32;
        bins.push(DrawBinRange {
            key,
            command_start,
            command_capacity,
            representative_scene: scenes[0].0,
            visibility_stages: crate::VisibilityStageMask(
                scenes.iter().fold(0, |mask, (_, stages)| mask | stages.0),
            ),
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

/// Early- and late-pass draw bins for a Nanite-style two-phase raster.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TwoPhaseViewDrawBins {
    /// Bins drawn before the current-frame HZB is rebuilt.
    pub early: ViewDrawBins,
    /// Bins drawn after the HZB rebuild, from deferred-then-revealed instances.
    pub late: ViewDrawBins,
}

/// Builds independent early- and late-pass draw bins from a candidate stream.
///
/// Each candidate is routed by [`raster_phase_of`](crate::raster_phase_of) into
/// at most one phase: early-visible instances never appear in the late pass,
/// and candidates that are neither early nor late-visible (for example a bare
/// [`LATE_RETEST`](crate::VisibilityStageMask::LATE_RETEST) that failed the
/// current HZB) are dropped. The two resulting [`ViewDrawBins`] are separate
/// indirect-draw command streams, each with its own `command_start` offsets
/// beginning at zero, matching the two draw dispatches the rasterizer issues
/// around the HZB rebuild.
pub fn build_two_phase_view_draw_bins(
    view: ViewHandle,
    scene_capacity: u32,
    candidates: impl IntoIterator<Item = DrawBinCandidate>,
) -> TwoPhaseViewDrawBins {
    let mut early = Vec::new();
    let mut late = Vec::new();
    for candidate in candidates {
        match crate::raster_phase_of(candidate.visibility_stages) {
            Some(crate::RasterPhase::Early) => early.push(candidate),
            Some(crate::RasterPhase::Late) => late.push(candidate),
            None => {}
        }
    }
    TwoPhaseViewDrawBins {
        early: build_view_draw_bins(view, scene_capacity, early),
        late: build_view_draw_bins(view, scene_capacity, late),
    }
}

impl TwoPhaseViewDrawBins {
    /// Packs the early and late streams into one shared set of GPU buffers,
    /// placing the early pass first. After this call the late stream's
    /// `global_bin_start`, `command_buffer_start`, and `global_candidate_start`
    /// sit immediately past the early stream, so both can be uploaded into a
    /// single bin/command/candidate buffer without overlap.
    pub fn pack(&mut self) {
        pack_draw_bin_streams([&mut self.early, &mut self.late]);
    }
}

/// Packs a sequence of draw-bin streams into shared global buffers by assigning
/// each stream disjoint offsets via a running prefix sum in iteration order.
///
/// [`build_view_draw_bins`] emits every stream with its global offsets at zero,
/// which is only valid for a single stream drawn in isolation. When several
/// streams share one GPU bin/command/candidate buffer — multiple views, or the
/// early and late passes of one view — each must begin past the previous one.
/// This assigns:
/// - `global_bin_start` += the sum of preceding `bins.len()`,
/// - `command_buffer_start` += the sum of preceding `command_count`,
/// - `global_candidate_start` += the sum of preceding `candidate_bins.len()`.
///
/// Candidate strides use the full `candidate_bins` length (the scene capacity
/// the stream was built with), matching the scene-capacity-strided candidate
/// lookup documented on [`ViewDrawBins::command_buffer_start`]. All cursors use
/// saturating addition so a pathological total cannot wrap. The per-bin
/// `command_start` stored in each [`DrawBinRange`] stays stream-local; the GPU
/// adds `command_buffer_start` to reach the absolute slot.
pub fn pack_draw_bin_streams<'a, I>(streams: I)
where
    I: IntoIterator<Item = &'a mut ViewDrawBins>,
{
    let mut bin_cursor = 0_u32;
    let mut command_cursor = 0_u32;
    let mut candidate_cursor = 0_u32;
    for stream in streams {
        stream.global_bin_start = bin_cursor;
        stream.command_buffer_start = command_cursor;
        stream.global_candidate_start = candidate_cursor;
        bin_cursor = bin_cursor.saturating_add(stream.bins.len() as u32);
        command_cursor = command_cursor.saturating_add(stream.command_count);
        candidate_cursor =
            candidate_cursor.saturating_add(stream.candidate_bins.len() as u32);
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
                    visibility_stages: crate::VisibilityStageMask::EARLY,
                },
                DrawBinCandidate {
                    scene: handle(1),
                    key: key(1),
                    visibility_stages: crate::VisibilityStageMask::EARLY,
                },
                DrawBinCandidate {
                    scene: handle(3),
                    key: key(2),
                    visibility_stages: crate::VisibilityStageMask::LATE_RETEST,
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

    #[test]
    fn two_phase_bins_split_candidates_into_independent_command_streams() {
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
        let late = crate::VisibilityStageMask::LATE_RETEST | crate::VisibilityStageMask::LATE_VISIBLE;
        let bins = build_two_phase_view_draw_bins(
            handle(9),
            8,
            [
                DrawBinCandidate {
                    scene: handle(1),
                    key: key(1),
                    visibility_stages: crate::VisibilityStageMask::EARLY,
                },
                DrawBinCandidate {
                    scene: handle(2),
                    key: key(2),
                    visibility_stages: late,
                },
                DrawBinCandidate {
                    scene: handle(3),
                    key: key(1),
                    visibility_stages: crate::VisibilityStageMask::EARLY,
                },
                // Deferred but still occluded late: drawn by neither phase.
                DrawBinCandidate {
                    scene: handle(4),
                    key: key(2),
                    visibility_stages: crate::VisibilityStageMask::LATE_RETEST,
                },
            ],
        );
        // Early pass: two instances in one bin (same key), own stream from 0.
        assert_eq!(bins.early.bins.len(), 1);
        assert_eq!(bins.early.bins[0].command_start, 0);
        assert_eq!(bins.early.bins[0].command_capacity, 2);
        assert_eq!(bins.early.command_count, 2);
        // Late pass: one revealed instance, own stream also from 0.
        assert_eq!(bins.late.bins.len(), 1);
        assert_eq!(bins.late.bins[0].command_start, 0);
        assert_eq!(bins.late.bins[0].command_capacity, 1);
        assert_eq!(bins.late.command_count, 1);
        assert_eq!(bins.late.bins[0].representative_scene, handle(2));
        // The bare LATE_RETEST candidate is routed to neither phase.
        assert_eq!(
            bins.early.command_count + bins.late.command_count,
            3
        );
    }

    #[test]
    fn two_phase_early_visible_candidate_is_not_binned_late() {
        let both = crate::VisibilityStageMask::EARLY | crate::VisibilityStageMask::LATE_VISIBLE;
        let bins = build_two_phase_view_draw_bins(
            handle(9),
            8,
            [DrawBinCandidate {
                scene: handle(1),
                key: DrawBinKey {
                    geometry: handle(1),
                    lod_or_cluster: 0,
                    pipeline_class: 1,
                    vertex_buffer_class: 1,
                    index_buffer_class: 2,
                    indexed: true,
                    primitive_kind: GeometryPrimitiveKind::Indexed,
                    pass_mask: crate::RenderPassMask::OPAQUE.0,
                },
                visibility_stages: both,
            }],
        );
        assert_eq!(bins.early.command_count, 1);
        assert_eq!(bins.late.command_count, 0);
    }

    #[test]
    fn pack_assigns_disjoint_offsets_to_early_then_late() {
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
        let late =
            crate::VisibilityStageMask::LATE_RETEST | crate::VisibilityStageMask::LATE_VISIBLE;
        let mut bins = build_two_phase_view_draw_bins(
            handle(9),
            8,
            [
                DrawBinCandidate {
                    scene: handle(1),
                    key: key(1),
                    visibility_stages: crate::VisibilityStageMask::EARLY,
                },
                DrawBinCandidate {
                    scene: handle(2),
                    key: key(2),
                    visibility_stages: crate::VisibilityStageMask::EARLY,
                },
                DrawBinCandidate {
                    scene: handle(3),
                    key: key(3),
                    visibility_stages: late,
                },
            ],
        );
        // Before packing every stream starts at zero.
        assert_eq!(bins.early.global_bin_start, 0);
        assert_eq!(bins.late.global_bin_start, 0);
        bins.pack();
        // Early pass stays at the origin of each shared buffer.
        assert_eq!(bins.early.global_bin_start, 0);
        assert_eq!(bins.early.command_buffer_start, 0);
        assert_eq!(bins.early.global_candidate_start, 0);
        // Late pass starts immediately past the early stream.
        assert_eq!(bins.late.global_bin_start, bins.early.bins.len() as u32);
        assert_eq!(bins.late.command_buffer_start, bins.early.command_count);
        assert_eq!(
            bins.late.global_candidate_start,
            bins.early.candidate_bins.len() as u32
        );
        // Candidate stride is the per-view scene capacity (8), not the bin count.
        assert_eq!(bins.late.global_candidate_start, 8);
    }

    #[test]
    fn pack_streams_prefix_sums_three_views() {
        let make = |view, caps: &[(u32, u32)], scene_capacity| {
            let candidates = caps.iter().map(|&(scene, geo)| DrawBinCandidate {
                scene: handle(scene),
                key: DrawBinKey {
                    geometry: handle(geo),
                    lod_or_cluster: 0,
                    pipeline_class: geo,
                    vertex_buffer_class: 1,
                    index_buffer_class: 2,
                    indexed: true,
                    primitive_kind: GeometryPrimitiveKind::Indexed,
                    pass_mask: crate::RenderPassMask::OPAQUE.0,
                },
                visibility_stages: crate::VisibilityStageMask::EARLY,
            });
            build_view_draw_bins(handle(view), scene_capacity, candidates)
        };
        let mut a = make(1, &[(1, 1), (2, 2)], 4); // 2 bins, 2 commands, cap 4
        let mut b = make(2, &[(1, 1)], 8); // 1 bin, 1 command, cap 8
        let mut c = make(3, &[(1, 1), (2, 1), (3, 2)], 16); // 2 bins, 3 commands, cap 16
        pack_draw_bin_streams([&mut a, &mut b, &mut c]);
        assert_eq!((a.global_bin_start, a.command_buffer_start, a.global_candidate_start), (0, 0, 0));
        assert_eq!((b.global_bin_start, b.command_buffer_start, b.global_candidate_start), (2, 2, 4));
        assert_eq!((c.global_bin_start, c.command_buffer_start, c.global_candidate_start), (3, 3, 12));
    }
}
