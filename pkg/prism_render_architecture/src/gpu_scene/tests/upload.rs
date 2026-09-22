use super::super::{
    DirtySceneSlot, SceneFieldMask, SceneHandle, UploadBudget, UploadPlanner, UploadStrategy,
};

fn dirty(index: u32, fields: SceneFieldMask) -> DirtySceneSlot {
    DirtySceneSlot {
        handle: SceneHandle {
            index,
            generation: 1,
        },
        fields,
    }
}

#[test]
fn planner_selects_sparse_contiguous_and_full_rewrite() {
    let planner = UploadPlanner::new(UploadBudget::default());
    let sparse = planner.plan(
        1000,
        &[
            dirty(1, SceneFieldMask::INSTANCE),
            dirty(500, SceneFieldMask::INSTANCE),
        ],
    );
    assert_eq!(sparse.instances.strategy, UploadStrategy::SparseScatter);
    let contiguous: Vec<_> = (10..30)
        .map(|index| dirty(index, SceneFieldMask::BOUNDS))
        .collect();
    assert_eq!(
        planner.plan(1000, &contiguous).bounds.strategy,
        UploadStrategy::ContiguousRanges
    );
    assert_eq!(
        planner.plan(32, &contiguous).bounds.strategy,
        UploadStrategy::FullRewrite
    );
}
