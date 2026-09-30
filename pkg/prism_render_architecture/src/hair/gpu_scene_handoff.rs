//! Hair sim → render-graph scene handoff (design §1 "形变顶点（汇入 `gpu_scene`）",
//! §8/§10 "双缓冲交形变顶点给 `gpu_scene`").
//!
//! [`async_pipeline`](super::async_pipeline) publishes the *temporal* half of
//! the `GPU`-driven persistence boundary — how a groom's deformed guide/render
//! points are replicated across in-flight frames so a completed frame can be
//! handed downstream while the next frame solves. This module publishes the
//! *scene* half: once a frame's solve + resolve has produced a fresh set of
//! world-space deformed render points in its ring copy, the groom's instance in
//! the shared [`gpu_scene`](crate::gpu_scene) must be told that
//!
//! * its world-space bounds moved (the groom swung), and
//! * its geometry content (the double-buffered deformed vertex copy) changed,
//!
//! so the incremental uploader re-publishes the instance row and the bounds row
//! for that frame.
//!
//! Hair does **not** own the scene mirror or the uploader — matching the design
//! rule that hair *consumes* shared services rather than owning them. It only
//! *produces* the authoritative [`SceneTransaction`](crate::gpu_scene::SceneTransaction)
//! that a [`CpuRenderScene`](crate::gpu_scene::CpuRenderScene) applies, yielding
//! the `BOUNDS` / `INSTANCE` dirty slots the
//! [`UploadPlanner`](crate::gpu_scene::UploadPlanner) turns into copy regions.
//! Everything here is deterministic, array-in / transaction-out, and never
//! panics on empty or degenerate input.

use crate::gpu_scene::{
    GeometryHandle, SceneBounds, SceneHandle, SceneOperation, SceneTransaction,
    SceneTransactionBuilder,
};
use crate::hair::async_pipeline::HairFramePipeline;

/// Computes the world-space [`SceneBounds`] enclosing a groom's deformed render
/// points for one completed frame.
///
/// Each point is `[x, y, z]` in world space (the resolve output the sim ring
/// hands downstream). A point with any non-finite component is skipped so a
/// stray `NaN` / `inf` from a diverged solve cannot poison the whole bound; if
/// no finite point survives (empty groom or all-degenerate input) the result is
/// a zero-extent bound at the origin. The returned `radius` is the half-diagonal
/// of the box, so it always encloses `half_extents`.
#[must_use]
pub fn deformed_bounds(points: &[[f32; 3]]) -> SceneBounds {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any = false;
    for point in points {
        let [x, y, z] = *point;
        if !(x.is_finite() && y.is_finite() && z.is_finite()) {
            continue;
        }
        any = true;
        min[0] = min[0].min(x);
        min[1] = min[1].min(y);
        min[2] = min[2].min(z);
        max[0] = max[0].max(x);
        max[1] = max[1].max(y);
        max[2] = max[2].max(z);
    }
    if !any {
        return SceneBounds {
            center: [0.0; 3],
            radius: 0.0,
            half_extents: [0.0; 3],
            _padding: 0.0,
        };
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let half_extents = [
        (max[0] - min[0]) * 0.5,
        (max[1] - min[1]) * 0.5,
        (max[2] - min[2]) * 0.5,
    ];
    let radius = (half_extents[0] * half_extents[0]
        + half_extents[1] * half_extents[1]
        + half_extents[2] * half_extents[2])
        .sqrt();
    SceneBounds {
        center,
        radius,
        half_extents,
        _padding: 0.0,
    }
}

/// Identifies a deforming groom's instance in the shared
/// [`gpu_scene`](crate::gpu_scene) plus the geometry handle naming the
/// double-buffered deformed vertex copy the render graph resolved for the
/// completed frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HairGroomHandoff {
    /// The groom's scene instance.
    pub instance: SceneHandle,
    /// The geometry table handle for the completed frame's deformed vertices.
    /// Its generation advances each published frame (see [`Self::advanced`]) so
    /// downstream can distinguish one frame's deformed copy from the next.
    pub geometry: GeometryHandle,
}

impl HairGroomHandoff {
    /// Builds the two authoritative scene operations that publish one completed
    /// sim frame: the moved world-space `bounds` and the (re-bound) geometry
    /// handle for this frame's deformed vertex copy.
    ///
    /// Order is deterministic — bounds first, then geometry — so replaying the
    /// same frame yields byte-identical transactions.
    #[must_use]
    pub fn deformation_operations(&self, bounds: SceneBounds) -> [SceneOperation; 2] {
        [
            SceneOperation::SetBounds {
                handle: self.instance,
                bounds,
            },
            SceneOperation::SetGeometry {
                handle: self.instance,
                geometry: self.geometry,
            },
        ]
    }

    /// Packages [`Self::deformation_operations`] into a producer
    /// [`SceneTransaction`] ready for
    /// [`CpuRenderScene::apply`](crate::gpu_scene::CpuRenderScene::apply).
    ///
    /// `frame_epoch` is the scene frame the deformation belongs to, `sequence`
    /// orders this producer's transactions within that epoch, and `producer`
    /// tags the hair subsystem so concurrent producers merge deterministically.
    #[must_use]
    pub fn publish(
        &self,
        frame_epoch: u64,
        sequence: u64,
        producer: u32,
        bounds: SceneBounds,
    ) -> SceneTransaction {
        let mut builder = SceneTransactionBuilder::for_producer(frame_epoch, sequence, producer);
        for operation in self.deformation_operations(bounds) {
            builder.push(operation);
        }
        builder.finish()
    }

    /// Returns the handoff for the next published frame: the same instance with
    /// its geometry generation advanced by one, so the freshly written ring copy
    /// resolves to a distinct handle from the copy still being read.
    #[must_use]
    pub fn advanced(&self) -> Self {
        Self {
            instance: self.instance,
            geometry: self.geometry.bumped(),
        }
    }
}

/// The frame-replicated ring copy holding `completed_frame`'s deformed state —
/// the copy the scene handoff makes readable downstream while the next frame
/// writes its own copy. A thin bridge to the temporal residency plan in
/// [`async_pipeline`](super::async_pipeline): the scene publish for
/// `completed_frame` names the geometry the render graph read out of this ring
/// slot.
#[must_use]
pub fn published_copy_index(pipeline: HairFramePipeline, completed_frame: u64) -> u32 {
    pipeline.write_index(completed_frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_scene::{
        CpuRenderScene, InstanceRecord, SceneFieldMask, SceneOperation, SceneTransactionBuilder,
    };

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6
    }

    #[test]
    fn bounds_of_axis_aligned_box_are_exact() {
        let points = [
            [-1.0, -2.0, -3.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.5, -1.0, 2.0],
        ];
        let bounds = deformed_bounds(&points);
        assert!(close(bounds.center[0], 0.0));
        assert!(close(bounds.center[1], 0.0));
        assert!(close(bounds.center[2], 0.0));
        assert!(close(bounds.half_extents[0], 1.0));
        assert!(close(bounds.half_extents[1], 2.0));
        assert!(close(bounds.half_extents[2], 3.0));
        // Half-diagonal encloses the extents: sqrt(1 + 4 + 9).
        assert!(close(bounds.radius, 14.0_f32.sqrt()));
        assert!(bounds.radius >= bounds.half_extents[2]);
    }

    #[test]
    fn empty_input_is_zero_bound_no_panic() {
        let bounds = deformed_bounds(&[]);
        assert!(close(bounds.center[0], 0.0));
        assert!(close(bounds.half_extents[0], 0.0));
        assert!(close(bounds.half_extents[1], 0.0));
        assert!(close(bounds.half_extents[2], 0.0));
        assert!(close(bounds.radius, 0.0));
    }

    #[test]
    fn non_finite_points_are_skipped() {
        let points = [
            [f32::NAN, 0.0, 0.0],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.0, f32::INFINITY, 0.0],
        ];
        let bounds = deformed_bounds(&points);
        // Only the two finite corners survive → unit cube centred at origin.
        assert!(close(bounds.center[0], 0.0));
        assert!(close(bounds.half_extents[0], 1.0));
        assert!(close(bounds.half_extents[1], 1.0));
        assert!(close(bounds.half_extents[2], 1.0));
    }

    #[test]
    fn all_non_finite_falls_back_to_zero_bound() {
        let points = [[f32::NAN, 0.0, 0.0], [0.0, f32::INFINITY, 0.0]];
        let bounds = deformed_bounds(&points);
        assert!(close(bounds.radius, 0.0));
        assert!(close(bounds.center[1], 0.0));
    }

    #[test]
    fn single_point_is_zero_extent_at_that_point() {
        let bounds = deformed_bounds(&[[4.0, -5.0, 6.0]]);
        assert!(close(bounds.center[0], 4.0));
        assert!(close(bounds.center[1], -5.0));
        assert!(close(bounds.center[2], 6.0));
        assert!(close(bounds.half_extents[0], 0.0));
        assert!(close(bounds.radius, 0.0));
    }

    #[test]
    fn advanced_bumps_only_geometry_generation() {
        let handoff = HairGroomHandoff {
            instance: SceneHandle::new(3, 1),
            geometry: GeometryHandle::new(7, 4),
        };
        let next = handoff.advanced();
        assert_eq!(next.instance, handoff.instance);
        assert_eq!(next.geometry.index, 7);
        assert_eq!(next.geometry.generation, 5);
    }

    #[test]
    fn published_copy_index_tracks_double_buffer() {
        let pipeline = HairFramePipeline::double_buffered();
        assert_eq!(published_copy_index(pipeline, 0), 0);
        assert_eq!(published_copy_index(pipeline, 1), 1);
        assert_eq!(published_copy_index(pipeline, 2), 0);
        let single = HairFramePipeline::single_buffered();
        assert_eq!(published_copy_index(single, 7), 0);
    }

    #[test]
    fn publish_is_consumed_end_to_end_by_the_scene() {
        // A groom instance must already live in the scene before it deforms.
        let instance = SceneHandle::new(1, 1);
        let geometry = GeometryHandle::new(9, 2);
        let mut scene = CpuRenderScene::default();
        let mut create = SceneTransactionBuilder::new(0, 0);
        create.push(SceneOperation::Create {
            handle: instance,
            record: InstanceRecord {
                geometry,
                ..InstanceRecord::default()
            },
        });
        let create_report = scene.apply(&create.finish());
        assert!(create_report.errors.is_empty());
        assert_eq!(create_report.created, 1);

        // A completed sim frame publishes moved bounds + advanced geometry.
        let handoff = HairGroomHandoff { instance, geometry }.advanced();
        let bounds = deformed_bounds(&[[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]);
        let transaction = handoff.publish(1, 0, 42, bounds);
        assert_eq!(transaction.producer, 42);
        assert_eq!(transaction.frame_epoch, 1);

        let report = scene.apply(&transaction);
        assert!(report.errors.is_empty());
        assert_eq!(report.updated, 2);

        // The uploader is driven off the dirty slots: bounds row + instance row
        // (which carries geometry_generation) must both be flagged.
        let slot = report
            .dirty_slots
            .iter()
            .find(|slot| slot.handle == instance)
            .expect("groom instance is dirty after deformation publish");
        assert!(slot.fields.contains(SceneFieldMask::BOUNDS));
        assert!(slot.fields.contains(SceneFieldMask::INSTANCE));

        // The scene now mirrors the deformed bounds and the advanced geometry.
        let record = scene.get(instance).expect("instance still live");
        assert_eq!(record.bounds, bounds);
        assert_eq!(record.geometry.generation, 3);
    }
}
