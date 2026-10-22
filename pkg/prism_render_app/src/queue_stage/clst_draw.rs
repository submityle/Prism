//! Generation-bound CLST submissions admitted to the main RenderApp queue.
//!
//! The producer owns CLST publication, bind-group construction and pipeline
//! specialization. This bridge accepts only an immutable typed draw record;
//! queue admission checks the current device generation, publication proof,
//! vertex range, storage declarations, and an owner lease before making
//! it visible to the frame graph.

use alloc::{collections::BTreeMap, sync::Arc, vec::Vec};

use prism_ecs::prelude::Resource;
use prism_render_architecture::gpu_scene::{GeometryHandle, SceneHandle, SceneMaterialHandle};
use prism_render_driver::{BufferId, IndexFormat, RenderPipelineId};

use crate::{
    draw::{
        BindGroupBindingId, DrawFunctionId, DrawFunctions, MeshRecorder, RenderBindGroupBinding,
        RenderBindGroupBindingContract, RenderBindGroupBufferAccess,
    },
    queue::{PhaseItem, RenderQueues},
    sets::RenderPhase,
};

pub(super) const CLST_BINDING_ID: BindGroupBindingId = BindGroupBindingId(0x434c_5354);

/// A producer-owned CLST draw ready for main-view rasterization.
///
/// `pipeline` must use a vertex-pulling vertex entry point compatible with the
/// supplied CLST group(s). The submission describes one Cluster: the queue
/// draws `triangle_count * 3` non-indexed vertices, and the shader uses
/// `builtin(vertex_index)` to read that Cluster's triangle indices from its
/// generation-matched CLST payload. This is queue transport only; it does not
/// create the shader, pipeline, or bindings. The owner lease must retain every
/// referenced GPU object until all graphs containing this submission complete.
#[derive(Clone)]
pub(crate) struct ClstDrawSubmission {
    pub camera: prism_ecs::prelude::Entity,
    pub device_generation: u64,
    pub resource_generation: u64,
    pub publication_generation: u64,
    pub bound_generation: u64,
    pub pipeline: RenderPipelineId,
    pub geometry: GeometryHandle,
    pub scene: SceneHandle,
    pub material: SceneMaterialHandle,
    /// Number of triangles in the selected Cluster.
    pub triangle_count: u32,
    pub order: u64,
    pub bind_groups: Vec<RenderBindGroupBinding>,
    pub owner: Arc<dyn core::any::Any + Send + Sync>,
}

/// RenderApp ingress for immutable, typed CLST draws.
///
/// Producers replace one camera's submission set atomically after page-table
/// publication and pipeline/bind-group preparation. Queue admission consumes
/// that camera's snapshot once; stale or malformed records are discarded.
#[derive(Default, Resource)]
pub(crate) struct ClstDrawSubmissions {
    by_camera: BTreeMap<prism_ecs::prelude::Entity, Vec<ClstDrawSubmission>>,
    draw_function: Option<DrawFunctionId>,
}

impl ClstDrawSubmissions {
    /// Atomically replaces one view's pending CLST draw set.
    pub(crate) fn publish(
        &mut self,
        camera: prism_ecs::prelude::Entity,
        submissions: Vec<ClstDrawSubmission>,
    ) {
        self.by_camera.insert(camera, submissions);
    }

    /// Revokes pending CLST work for a view.
    pub(crate) fn revoke(&mut self, camera: prism_ecs::prelude::Entity) {
        self.by_camera.remove(&camera);
    }
}

pub(super) fn admit_for_view(
    submissions: &mut ClstDrawSubmissions,
    camera: prism_ecs::prelude::Entity,
    device_generation: u64,
    draw_function: DrawFunctionId,
    queues: &mut RenderQueues,
) -> u64 {
    let Some(items) = submissions.by_camera.remove(&camera) else {
        return 0;
    };
    let mut admitted = 0_u64;
    for submission in items {
        if submission.camera != camera || !valid(&submission, device_generation) {
            continue;
        }
        let Some(vertex_count) = vertex_count(&submission) else {
            continue;
        };
        let item = PhaseItem {
            pipeline: submission.pipeline,
            // CLST payload indices are pulled by the shader; no separate RHI
            // index buffer is part of this transport contract.
            draw: draw_function,
            geometry: submission.geometry,
            vertices: Vec::new(),
            vertex_sizes: Vec::new(),
            indices: None,
            index_size: None,
            first: 0,
            count: vertex_count,
            base_vertex: 0,
            indirect: None,
            lod_level: 0,
            material: submission.material,
            scene: submission.scene,
            order: submission.order,
        };
        let mut bind_groups = submission.bind_groups;
        // Retain the submission owner even if a producer's group wrapper did
        // not separately attach it. The frame graph clones these bindings into
        // its recording closure before the render-world snapshot can retire.
        if let Some(group) = bind_groups.first_mut() {
            group.external_owners.push(submission.owner);
        } else {
            continue;
        }
        queues.push_with_bind_groups(RenderPhase::Opaque, item, bind_groups);
        admitted += 1;
    }
    admitted
}

pub(super) fn ensure_draw_function(
    functions: &mut DrawFunctions,
    submissions: &mut ClstDrawSubmissions,
) -> DrawFunctionId {
    if let Some(id) = submissions.draw_function {
        return id;
    }
    let id = functions.register_with_bind_groups_and_dynamic_contract(
        clst_vertex_pull_draw,
        Vec::new(),
        vec![RenderBindGroupBindingContract {
            identity: CLST_BINDING_ID,
            index: 0,
        }],
    );
    submissions.draw_function = Some(id);
    id
}

fn clst_vertex_pull_draw(item: &PhaseItem, ctx: &mut dyn MeshRecorder) {
    let Some(vertex_end) = item.first.checked_add(item.count) else {
        return;
    };
    if !item.vertices.is_empty()
        || !item.vertex_sizes.is_empty()
        || item.count == 0
        || item.indices.is_some()
        || item.index_size.is_some()
    {
        return;
    }
    ctx.set_pipeline(item.pipeline);
    ctx.draw(item.first..vertex_end, 0..1);
}

fn valid(submission: &ClstDrawSubmission, device_generation: u64) -> bool {
    if submission.device_generation != device_generation
        || submission.resource_generation != device_generation
        || submission.publication_generation == 0
        || submission.publication_generation != submission.bound_generation
        || !submission.geometry.is_valid()
        || !submission.scene.is_valid()
        || !submission.material.is_valid()
        || vertex_count(submission).is_none()
    {
        return false;
    }
    if submission.bind_groups.is_empty() {
        return false;
    }
    let mut has_storage = false;
    for (index, group) in submission.bind_groups.iter().enumerate() {
        if group.bind_group.generation() == 0
            || submission.bind_groups[index + 1..]
                .iter()
                .any(|other| other.index == group.index || other.identity == group.identity)
        {
            return false;
        }
        has_storage |= group.buffers.iter().any(|buffer| {
            buffer.access == RenderBindGroupBufferAccess::Storage && buffer.buffer.generation() != 0
        });
        if group
            .buffers
            .iter()
            .any(|buffer| buffer.buffer.generation() == 0)
        {
            return false;
        }
    }
    has_storage
        && submission.pipeline.generation() != 0
        && submission
            .bind_groups
            .iter()
            .any(|group| group.identity == CLST_BINDING_ID && group.index == 0)
}

/// Computes the non-indexed draw count for one Cluster without overflow.
fn vertex_count(submission: &ClstDrawSubmission) -> Option<u32> {
    submission
        .triangle_count
        .checked_mul(3)
        .filter(|count| *count != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::{DrawFunctions, MeshRecorder};
    use core::ops::Range;
    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_driver::{BindGroupId, BufferId, RenderPipelineId};

    fn submission(camera: prism_ecs::prelude::Entity) -> ClstDrawSubmission {
        let owner: Arc<dyn core::any::Any + Send + Sync> = Arc::new(());
        ClstDrawSubmission {
            camera,
            device_generation: 9,
            resource_generation: 9,
            publication_generation: 4,
            bound_generation: 4,
            pipeline: RenderPipelineId::from_parts(1, 2),
            geometry: GenerationalHandle::new(3, 1),
            scene: GenerationalHandle::new(4, 1),
            material: GenerationalHandle::new(5, 1),
            triangle_count: 3,
            order: 12,
            bind_groups: alloc::vec![RenderBindGroupBinding {
                identity: CLST_BINDING_ID,
                index: 0,
                bind_group: BindGroupId::from_parts(7, 2),
                dynamic_offsets: Vec::new(),
                buffers: alloc::vec![crate::draw::RenderBindGroupBuffer {
                    buffer: BufferId::from_parts(8, 2),
                    access: RenderBindGroupBufferAccess::Storage,
                }],
                sampled_textures: Vec::new(),
                external_owners: alloc::vec![Arc::clone(&owner)],
            }],
            owner,
        }
    }

    fn camera() -> prism_ecs::prelude::Entity {
        prism_ecs::prelude::Entity::from_bits((1_u64 << 32) | 1).unwrap()
    }

    #[derive(Default)]
    struct Recorder(Vec<(&'static str, Option<Range<u32>>)>);

    impl MeshRecorder for Recorder {
        fn set_pipeline(&mut self, _: RenderPipelineId) {
            self.0.push(("pipeline", None));
        }
        fn set_render_bind_group(&mut self, _: u32, _: BindGroupId, _: Vec<u32>) {
            self.0.push(("group", None));
        }
        fn set_vertex_buffer(
            &mut self,
            _: u32,
            _: BufferId,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
        }
        fn set_index_buffer(
            &mut self,
            _: BufferId,
            _: IndexFormat,
            _: u64,
            _: Option<core::num::NonZeroU64>,
        ) {
            self.0.push(("index", None));
        }
        fn draw(&mut self, vertices: Range<u32>, _: Range<u32>) {
            self.0.push(("draw", Some(vertices)));
        }
        fn draw_indexed(&mut self, _: Range<u32>, _: i32, _: Range<u32>) {
            self.0.push(("indexed", None));
        }
        fn draw_indexed_indirect(&mut self, _: prism_render_graph::BufferHandle, _: u64) {}
        fn multi_draw_indirect(
            &mut self,
            _: prism_render_graph::BufferHandle,
            _: u64,
            _: u32,
            _: u32,
        ) {
        }
    }

    #[test]
    fn admitted_clst_submission_records_cluster_vertex_range_without_index_buffer() {
        let camera = camera();
        let mut submissions = ClstDrawSubmissions::default();
        submissions.publish(camera, alloc::vec![submission(camera)]);
        let mut queues = RenderQueues::new();
        let mut functions = DrawFunctions::default();
        let draw = ensure_draw_function(&mut functions, &mut submissions);
        assert_eq!(
            admit_for_view(&mut submissions, camera, 9, draw, &mut queues),
            1
        );
        assert_eq!(queues.phase(RenderPhase::Opaque).len(), 1);
        let mut recorder = Recorder::default();
        queues.record(RenderPhase::Opaque, &mut recorder, &functions);
        assert_eq!(
            recorder.0,
            [("pipeline", None), ("group", None), ("draw", Some(0..9))]
        );
        assert!(queues.phase(RenderPhase::Opaque)[0].vertices.is_empty());
        assert!(queues.phase(RenderPhase::Opaque)[0].indices.is_none());
        assert!(queues.phase(RenderPhase::Opaque)[0].index_size.is_none());
    }

    #[test]
    fn stale_empty_and_out_of_range_publications_fail_closed() {
        let camera = camera();
        let mut submissions = ClstDrawSubmissions::default();
        submissions.publish(camera, alloc::vec![submission(camera)]);
        let mut queues = RenderQueues::new();
        let mut functions = DrawFunctions::default();
        let draw = ensure_draw_function(&mut functions, &mut submissions);
        assert_eq!(
            admit_for_view(&mut submissions, camera, 10, draw, &mut queues),
            0
        );
        assert!(queues.phase(RenderPhase::Opaque).is_empty());

        let mut empty = submission(camera);
        empty.triangle_count = 0;
        submissions.publish(camera, alloc::vec![empty]);
        assert_eq!(
            admit_for_view(&mut submissions, camera, 9, draw, &mut queues),
            0
        );

        let mut out_of_range = submission(camera);
        out_of_range.triangle_count = u32::MAX;
        submissions.publish(camera, alloc::vec![out_of_range]);
        assert_eq!(
            admit_for_view(&mut submissions, camera, 9, draw, &mut queues),
            0
        );
        assert!(queues.phase(RenderPhase::Opaque).is_empty());
    }

    #[test]
    fn publication_generation_must_match_bound_clst_tables() {
        let camera = camera();
        let mut submissions = ClstDrawSubmissions::default();
        let mut stale = submission(camera);
        stale.bound_generation = stale.publication_generation + 1;
        submissions.publish(camera, alloc::vec![stale]);
        let mut queues = RenderQueues::new();
        let mut functions = DrawFunctions::default();
        let draw = ensure_draw_function(&mut functions, &mut submissions);
        assert_eq!(
            admit_for_view(&mut submissions, camera, 9, draw, &mut queues),
            0
        );
        assert!(queues.phase(RenderPhase::Opaque).is_empty());

        let mut zero_generation = submission(camera);
        zero_generation.publication_generation = 0;
        zero_generation.bound_generation = 0;
        submissions.publish(camera, alloc::vec![zero_generation]);
        assert_eq!(
            admit_for_view(&mut submissions, camera, 9, draw, &mut queues),
            0
        );
    }

    #[test]
    fn submissions_are_camera_scoped_and_consumed_once() {
        let camera = camera();
        let another = prism_ecs::prelude::Entity::from_bits((1_u64 << 32) | 2).unwrap();
        let mut item = submission(camera);
        item.camera = camera;
        let mut submissions = ClstDrawSubmissions::default();
        submissions.publish(camera, alloc::vec![item]);
        let mut queues = RenderQueues::new();
        let mut functions = DrawFunctions::default();
        let draw = ensure_draw_function(&mut functions, &mut submissions);
        assert_eq!(
            admit_for_view(&mut submissions, another, 9, draw, &mut queues),
            0
        );
        assert_eq!(
            admit_for_view(&mut submissions, camera, 9, draw, &mut queues),
            1
        );
        assert_eq!(
            admit_for_view(&mut submissions, camera, 9, draw, &mut queues),
            0
        );
    }
}
