use super::*;
use alloc::{collections::BTreeMap, vec, vec::Vec};
use prism_world_residency::{SdfClipmapConfig, SdfToroidalClipmap};

#[derive(Debug, PartialEq)]
struct Rejected;

struct Backend {
    planner: SdfToroidalClipmap,
    values: BTreeMap<BrickCoord, Option<Vec<i16>>>,
    fail: bool,
    rolls: usize,
    uploads: usize,
}

impl UploadBackend for Backend {
    type Error = Rejected;

    fn update_center(
        &mut self,
        grid: &SparseSdfBrickGrid,
        center: BrickCoord,
    ) -> Result<u64, Rejected> {
        if self.fail {
            return Err(Rejected);
        }
        let delta = self.planner.update_center(center).unwrap();
        self.values
            .retain(|&coord, _| self.planner.slot_index(coord).is_some());
        let coords: Vec<_> = delta.layers().iter().flatten().copied().collect();
        self.rolls += 1;
        self.upload_bricks(grid, &coords)
    }

    fn upload_bricks(
        &mut self,
        grid: &SparseSdfBrickGrid,
        coords: &[BrickCoord],
    ) -> Result<u64, Rejected> {
        if self.fail {
            return Err(Rejected);
        }
        let mut bytes = 0;
        for &coord in coords {
            let samples = grid.brick_samples(coord).unwrap();
            bytes += 32 + samples.map_or(0, |s| s.len().div_ceil(2) as u64 * 4);
            self.values.insert(coord, samples.map(Vec::from));
        }
        self.uploads += 1;
        Ok(bytes)
    }
}

fn coord(x: i32, z: i32) -> BrickCoord {
    BrickCoord::new(x, 0, z, 0)
}

fn state(brick_size: u8) -> UploadState<Backend> {
    let clipmap = SdfClipmapConfig::new(1, 1, 4096).unwrap();
    let config = WorldSdfClipmapConfig::new(clipmap, 9, 4096).unwrap();
    UploadState::new(config, brick_size, 4096, || {
        Ok(Backend {
            planner: SdfToroidalClipmap::new(clipmap).unwrap(),
            values: BTreeMap::new(),
            fail: false,
            rolls: 0,
            uploads: 0,
        })
    })
    .unwrap()
}

#[test]
fn roll_uploads_edges_once_and_static_center_is_constant_work() {
    let mut state = state(2);
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    grid.insert_brick(coord(0, 0), vec![7; 8]).unwrap();
    let first = state.update_center(&grid, coord(0, 0)).unwrap();
    assert_eq!(first.uploaded_bytes, 9 * 32 + 16);
    assert_eq!(first.pending_bricks, 0);
    state.require_ready().unwrap();
    let same = state.update_center(&grid, coord(0, 0)).unwrap();
    assert_eq!(same.uploaded_bytes, 0);
    assert_eq!(same.revision, first.revision);
    assert_eq!(state.backend.rolls, 1);
    let moved = state.update_center(&grid, coord(1, 0)).unwrap();
    assert_eq!(moved.uploaded_bytes, 3 * 32);
    assert_eq!(moved.pending_bricks, 0);
    assert_eq!(state.backend.values[&coord(0, 0)], Some(vec![7; 8]));
    assert!(!state.backend.values.contains_key(&coord(-1, 0)));
}

#[test]
fn retained_edits_survive_roll_and_byte_bounded_batches_match_latest_grid() {
    let mut state = state(2);
    let mut grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    state.update_center(&grid, coord(0, 0)).unwrap();
    grid.insert_brick(coord(0, 0), vec![11; 8]).unwrap();
    grid.insert_brick(coord(1, 0), vec![12; 8]).unwrap();
    state
        .scheduler
        .mark_dirty(&[coord(1, 0), coord(0, 0)])
        .unwrap();
    assert_eq!(
        state
            .update_center(&grid, coord(1, 0))
            .unwrap()
            .pending_bricks,
        2
    );
    assert_eq!(
        state.require_ready(),
        Err(WorldSdfClipmapUploadError::PendingDirty { bricks: 2 })
    );
    let batch = state.flush_dirty(&grid, usize::MAX, 95).unwrap();
    assert_eq!(batch.uploaded_bytes, 48);
    assert_eq!(batch.pending_bricks, 1);
    assert_eq!(state.backend.values[&coord(1, 0)], Some(vec![12; 8]));
    grid.insert_brick(coord(0, 0), vec![13; 8]).unwrap();
    grid.mark_dirty_region([0, 0, 0], [0, 0, 0], 0).unwrap();
    state.scheduler.mark_dirty(&[coord(0, 0)]).unwrap();
    assert_eq!(state.flush_dirty(&grid, 1, 48).unwrap().pending_bricks, 0);
    assert_eq!(state.backend.values[&coord(0, 0)], Some(vec![13; 8]));
    state.require_ready().unwrap();
    assert_eq!(
        grid.take_dirty(),
        vec![coord(0, 0)],
        "must not consume other consumers' CPU dirty set"
    );
}

#[test]
fn removed_brick_uploads_unknown_and_odd_samples_charge_word_padding() {
    let mut state = state(1);
    let mut grid = SparseSdfBrickGrid::new(1, 1).unwrap();
    grid.insert_brick(coord(0, 0), vec![4]).unwrap();
    state.update_center(&grid, coord(0, 0)).unwrap();
    state.scheduler.mark_dirty(&[coord(0, 0)]).unwrap();
    assert_eq!(
        state.flush_dirty(&grid, 1, 35),
        Err(WorldSdfClipmapUploadError::UploadBudgetTooSmall {
            required: 36,
            budget: 35
        })
    );
    assert_eq!(state.flush_dirty(&grid, 1, 36).unwrap().uploaded_bytes, 36);
    grid.remove_brick(coord(0, 0)).unwrap();
    state.scheduler.mark_dirty(&[coord(0, 0)]).unwrap();
    assert_eq!(state.flush_dirty(&grid, 1, 32).unwrap().uploaded_bytes, 32);
    assert_eq!(state.backend.values[&coord(0, 0)], None);
}

#[test]
fn failed_roll_preserves_both_centers_revision_and_dirty_then_retries() {
    let mut state = state(2);
    let grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    state.update_center(&grid, coord(0, 0)).unwrap();
    state.scheduler.mark_dirty(&[coord(0, 0)]).unwrap();
    let revision = state.scheduler.revision();
    state.backend.fail = true;
    assert_eq!(
        state.update_center(&grid, coord(10, 0)),
        Err(WorldSdfClipmapUploadError::Backend(Rejected))
    );
    assert_eq!(state.scheduler.center(), Some(coord(0, 0)));
    assert_eq!(state.backend.planner.center(), Some(coord(0, 0)));
    assert_eq!(state.scheduler.revision(), revision);
    assert_eq!(state.scheduler.dirty_bricks(), &[coord(0, 0)]);
    state.backend.fail = false;
    assert_eq!(
        state
            .update_center(&grid, coord(10, 0))
            .unwrap()
            .pending_bricks,
        0
    );
    assert_eq!(state.backend.planner.center(), Some(coord(10, 0)));
}

#[test]
fn failed_upload_preserves_pending_and_revision_then_retries() {
    let mut state = state(2);
    let grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    state.update_center(&grid, coord(0, 0)).unwrap();
    state.scheduler.mark_dirty(&[coord(0, 0)]).unwrap();
    let revision = state.scheduler.revision();
    state.backend.fail = true;
    assert_eq!(
        state.flush_dirty(&grid, 9, 4096),
        Err(WorldSdfClipmapUploadError::Backend(Rejected))
    );
    assert_eq!(state.scheduler.revision(), revision);
    assert_eq!(state.scheduler.dirty_bricks(), &[coord(0, 0)]);
    state.backend.fail = false;
    assert_eq!(state.flush_dirty(&grid, 9, 4096).unwrap().pending_bricks, 0);
}

#[test]
fn preflight_geometry_count_and_bytes_do_not_call_backend() {
    let mut state = state(2);
    let grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    let wrong_grid = SparseSdfBrickGrid::new(1, 1).unwrap();
    assert_eq!(
        state.require_ready(),
        Err(WorldSdfClipmapUploadError::Uninitialized)
    );
    assert_eq!(
        state.flush_dirty(&grid, 1, 1),
        Err(WorldSdfClipmapUploadError::Uninitialized)
    );
    assert_eq!(
        state.update_center(&wrong_grid, coord(0, 0)),
        Err(WorldSdfClipmapUploadError::GridMismatch)
    );
    assert_eq!(state.backend.rolls, 0);
    state.update_center(&grid, coord(0, 0)).unwrap();
    state.scheduler.mark_dirty(&[coord(0, 0)]).unwrap();
    assert_eq!(
        state.update_center(&wrong_grid, coord(0, 0)),
        Err(WorldSdfClipmapUploadError::GridMismatch)
    );
    assert_eq!(
        state.flush_dirty(&wrong_grid, 1, 4096),
        Err(WorldSdfClipmapUploadError::GridMismatch)
    );
    assert_eq!(
        state.flush_dirty(&grid, 0, 4096),
        Err(WorldSdfClipmapUploadError::ZeroBrickBudget)
    );
    state.max_upload_bytes = 31;
    assert_eq!(
        state.flush_dirty(&grid, 1, u64::MAX),
        Err(WorldSdfClipmapUploadError::UploadBudgetTooSmall {
            required: 32,
            budget: 31
        })
    );
    assert_eq!(state.backend.uploads, 1);
    assert_eq!(state.scheduler.dirty_bricks(), &[coord(0, 0)]);
}

#[test]
fn count_limit_and_noop_flush_are_exact() {
    let mut state = state(2);
    let grid = SparseSdfBrickGrid::new(2, 1).unwrap();
    state.update_center(&grid, coord(0, 0)).unwrap();
    state
        .scheduler
        .mark_dirty(&[coord(0, 0), coord(1, 0)])
        .unwrap();
    let first = state.flush_dirty(&grid, 1, 4096).unwrap();
    assert_eq!(first.pending_bricks, 1);
    assert_eq!(first.uploaded_bytes, 32);
    let last = state.flush_dirty(&grid, 1, 4096).unwrap();
    let noop = state.flush_dirty(&grid, 0, 0).unwrap();
    assert_eq!(noop.revision, last.revision);
    assert_eq!(noop.uploaded_bytes, 0);
    assert_eq!(state.backend.uploads, 3);
}

#[test]
fn invalid_configuration_rejected_before_backend_creation() {
    let clipmap = SdfClipmapConfig::new(1, 0, 4096).unwrap();
    let config = WorldSdfClipmapConfig::new(clipmap, 1, 16).unwrap();
    for (brick_size, budget) in [(0, 4096), (33, 4096), (2, 0)] {
        let result = UploadState::<Backend>::new(config, brick_size, budget, || {
            panic!("backend must not be constructed")
        });
        assert!(matches!(
            result,
            Err(WorldSdfClipmapUploadError::InvalidConfiguration)
        ));
    }
}
