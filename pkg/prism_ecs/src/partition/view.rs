//! Floating-origin ↔ LOD bridge (design §13.2 + §13.3).
//!
//! [`EntityLodProcessor`](crate::partition::processor::EntityLodProcessor) measures
//! distance in a flat `[f32; 3]` space, but a large world stores positions as a
//! coarse [`GridCell`] plus a fine [`LocalPos`] so that `f32` precision never
//! decays with distance from the world origin (design §13.3). [`OriginView`]
//! is the glue: given a [`FloatingOrigin`] and the camera's cell/offset, it
//! rebases every entity sample into the camera's local space and hands the
//! processor jitter-free coordinates.
//!
//! The typical per-frame flow for the owner:
//!
//! 1. quantize the camera's absolute [`WorldPos`] into `(cell, local)` and
//!    [`recenter`](FloatingOrigin::recenter) the origin onto the camera cell so
//!    rebased offsets stay tiny;
//! 2. build an [`OriginView`] for that origin and camera position;
//! 3. [`drive`](OriginView::drive) the LOD processor with the world-space
//!    population — the view rebases each sample before measuring distance.
//!
//! Rebasing is a pure coordinate transform, so the view borrows the origin
//! immutably and allocates only the rebased batch it feeds the processor.

use alloc::vec::Vec;

use crate::entity::Entity;
use crate::partition::dormant::DormancySet;
use crate::partition::floating_origin::{FloatingOrigin, GridCell, LocalPos, WorldPos};
use crate::partition::processor::{EntityLodProcessor, LodTickResult};

/// A rebasing view over a [`FloatingOrigin`] that converts big-world
/// `(GridCell, LocalPos)` samples into the active origin's local `[f32; 3]`
/// space for the LOD processor (design §13.2 + §13.3).
///
/// The view caches the rebased **viewpoint** (usually the camera, near the
/// local origin after a recenter) so repeated [`sample`](Self::sample) /
/// [`drive`](Self::drive) calls share it. It holds the origin by shared
/// reference and never mutates it.
#[derive(Clone, Copy, Debug)]
pub struct OriginView<'a> {
    origin: &'a FloatingOrigin,
    viewpoint: [f32; 3],
}

impl<'a> OriginView<'a> {
    /// Builds a view whose viewpoint is the camera at `(cell, local)`, rebased
    /// into `origin`'s local space.
    ///
    /// Pair this with a prior [`FloatingOrigin::recenter`] onto the camera cell
    /// so the viewpoint lands near the local origin and every nearby entity
    /// keeps full `f32` precision.
    pub fn new(origin: &'a FloatingOrigin, cell: GridCell, local: LocalPos) -> Self {
        Self {
            origin,
            viewpoint: origin.rebase_array(cell, local),
        }
    }

    /// Builds a view from an already-rebased `viewpoint` in local space, for
    /// callers that computed the camera's local offset themselves.
    pub fn with_local_viewpoint(origin: &'a FloatingOrigin, viewpoint: [f32; 3]) -> Self {
        Self { origin, viewpoint }
    }

    /// Builds a view whose viewpoint is the camera at an absolute
    /// [`WorldPos`], quantized and rebased against `origin` (design §13.3).
    pub fn from_world(origin: &'a FloatingOrigin, camera: WorldPos) -> Self {
        Self {
            origin,
            viewpoint: origin.rebase_world(camera),
        }
    }

    /// The grid this view rebases against.
    #[inline]
    pub fn origin(&self) -> &FloatingOrigin {
        self.origin
    }

    /// The rebased viewpoint fed to the processor as the distance reference.
    #[inline]
    pub fn viewpoint(&self) -> [f32; 3] {
        self.viewpoint
    }

    /// Rebases one big-world `(cell, local)` sample into the active origin's
    /// local `[f32; 3]` space.
    #[inline]
    pub fn sample(&self, cell: GridCell, local: LocalPos) -> [f32; 3] {
        self.origin.rebase_array(cell, local)
    }

    /// Rebases one absolute [`WorldPos`] sample into the active origin's local
    /// `[f32; 3]` space (design §13.3).
    #[inline]
    pub fn sample_world(&self, pos: WorldPos) -> [f32; 3] {
        self.origin.rebase_world(pos)
    }

    /// Rebases a whole `(entity, cell, local)` batch into the
    /// `(entity, [f32; 3])` population shape the processor consumes. Entity
    /// order is preserved.
    pub fn rebase_population(
        &self,
        population: &[(Entity, GridCell, LocalPos)],
    ) -> Vec<(Entity, [f32; 3])> {
        population
            .iter()
            .map(|&(entity, cell, local)| (entity, self.sample(cell, local)))
            .collect()
    }

    /// Rebases `population` and drives it through `processor` for `frame`,
    /// returning the processor's [`LodTickResult`] (design §13.2).
    ///
    /// This is the end-to-end big-world entry point: callers pass absolute
    /// grid positions and get back the per-frame active set plus the dormancy
    /// transitions, with all distance measured in jitter-free local space.
    pub fn drive(
        &self,
        processor: &EntityLodProcessor,
        population: &[(Entity, GridCell, LocalPos)],
        frame: u64,
        dormancy: &mut DormancySet,
    ) -> LodTickResult {
        let rebased = self.rebase_population(population);
        processor.drive(self.viewpoint, &rebased, frame, dormancy)
    }

    /// Rebases an absolute `(entity, WorldPos)` batch and drives it through
    /// `processor` for `frame` (design §13.2 + §13.3).
    ///
    /// The absolute-coordinate twin of [`drive`](Self::drive): callers that
    /// keep positions as double-precision [`WorldPos`] (rather than a
    /// pre-split `(cell, local)`) hand them in directly.
    pub fn drive_world(
        &self,
        processor: &EntityLodProcessor,
        population: &[(Entity, WorldPos)],
        frame: u64,
        dormancy: &mut DormancySet,
    ) -> LodTickResult {
        let rebased: Vec<(Entity, [f32; 3])> = population
            .iter()
            .map(|&(entity, pos)| (entity, self.sample_world(pos)))
            .collect();
        processor.drive(self.viewpoint, &rebased, frame, dormancy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::floating_origin::WorldPos;
    use crate::partition::lod::{LodLevel, LodSchedule, OutOfRange};

    fn ent(index: u32, generation: u32) -> Entity {
        Entity::from_bits(((generation as u64) << 32) | index as u64)
            .expect("test entity needs a nonzero generation")
    }

    /// Band 0 (< 100) full rate, band 1 (< 2500) every 4th, dormant beyond.
    fn schedule() -> LodSchedule {
        LodSchedule::from_sorted_pairs(&[(100.0, 1), (2_500.0, 4)])
            .with_policy(OutOfRange::Dormant)
    }

    #[test]
    fn viewpoint_is_rebased_against_origin() {
        // Camera sits in cell (10, 0, 0); recenter the origin onto it so the
        // camera's rebased viewpoint is just its local offset.
        let grid = FloatingOrigin::new(1000.0).with_origin(GridCell::new(10, 0, 0));
        let view = OriginView::new(&grid, GridCell::new(10, 0, 0), LocalPos::new(5.0, 0.0, 0.0));
        assert_eq!(view.viewpoint(), [5.0, 0.0, 0.0]);
    }

    #[test]
    fn sample_rebases_distant_cells_into_local_space() {
        let grid = FloatingOrigin::new(1000.0).with_origin(GridCell::new(10, 0, 0));
        let view = OriginView::new(&grid, GridCell::new(10, 0, 0), LocalPos::new(0.0, 0.0, 0.0));
        // One cell east of the origin + 25 m local = 1025 m local-x.
        assert_eq!(
            view.sample(GridCell::new(11, 0, 0), LocalPos::new(25.0, 0.0, 0.0)),
            [1025.0, 0.0, 0.0]
        );
        // Same cell as the origin: just the local offset.
        assert_eq!(
            view.sample(GridCell::new(10, 0, 0), LocalPos::new(7.0, 0.0, 0.0)),
            [7.0, 0.0, 0.0]
        );
    }

    #[test]
    fn drive_bands_entities_by_rebased_distance() {
        let grid = FloatingOrigin::new(1000.0).with_origin(GridCell::new(10, 0, 0));
        let view = OriginView::new(&grid, GridCell::new(10, 0, 0), LocalPos::new(0.0, 0.0, 0.0));
        let processor = EntityLodProcessor::new(schedule());
        let mut dormancy = DormancySet::new();

        let near = ent(1, 1); // same cell, 5 m away → band 0, ticks every frame
        let far = ent(2, 1); // one cell away → ~1000 m → beyond last band → dormant
        let population = [
            (near, GridCell::new(10, 0, 0), LocalPos::new(5.0, 0.0, 0.0)),
            (far, GridCell::new(11, 0, 0), LocalPos::new(0.0, 0.0, 0.0)),
        ];

        let r = view.drive(&processor, &population, 0, &mut dormancy);
        assert_eq!(r.to_tick, alloc::vec![near]);
        assert_eq!(r.slept, alloc::vec![far]);
        assert!(dormancy.is_dormant(far));
        // The far entity's decision still rides along in input order.
        assert_eq!(r.decisions.len(), 2);
        assert_eq!(r.decisions[0].1.level, Some(LodLevel(0)));
        assert_eq!(r.decisions[1].1.level, None);
    }

    #[test]
    fn recentering_the_origin_keeps_relative_distances_stable() {
        // The same two entities, viewed from two different origins, must band
        // identically: rebasing is origin-relative, so moving the origin with
        // the camera does not change who is near.
        let camera_cell = GridCell::new(500, 0, 0);
        let a = ent(1, 1);
        let b = ent(2, 1);
        let pop = [
            (a, GridCell::new(500, 0, 0), LocalPos::new(10.0, 0.0, 0.0)),
            (b, GridCell::new(501, 0, 0), LocalPos::new(0.0, 0.0, 0.0)),
        ];

        let processor = EntityLodProcessor::new(schedule());

        let grid_lo = FloatingOrigin::new(1000.0).with_origin(camera_cell);
        let view_lo = OriginView::new(&grid_lo, camera_cell, LocalPos::new(10.0, 0.0, 0.0));
        let mut dorm_lo = DormancySet::new();
        let r_lo = view_lo.drive(&processor, &pop, 0, &mut dorm_lo);

        // A far-away world origin at (0,0,0): absolute f64 coords are huge, but
        // rebasing against the camera cell yields the same local geometry.
        let grid_hi = FloatingOrigin::new(1000.0).with_origin(camera_cell);
        let view_hi = OriginView::new(&grid_hi, camera_cell, LocalPos::new(10.0, 0.0, 0.0));
        let mut dorm_hi = DormancySet::new();
        let r_hi = view_hi.drive(&processor, &pop, 0, &mut dorm_hi);

        assert_eq!(r_lo.to_tick, r_hi.to_tick);
        assert_eq!(r_lo.slept, r_hi.slept);
    }
    #[test]
    fn from_world_matches_split_construction() {
        let grid = FloatingOrigin::new(1000.0).with_origin(GridCell::new(3, 0, 0));
        // Camera at absolute 3025 m → cell 3, local 25 m.
        let via_world = OriginView::from_world(&grid, WorldPos::new(3025.0, 0.0, 0.0));
        let via_split =
            OriginView::new(&grid, GridCell::new(3, 0, 0), LocalPos::new(25.0, 0.0, 0.0));
        assert_eq!(via_world.viewpoint(), via_split.viewpoint());
        assert_eq!(via_world.viewpoint(), [25.0, 0.0, 0.0]);
    }

    #[test]
    fn drive_world_bands_absolute_positions() {
        let mut grid = FloatingOrigin::new(1000.0);
        // Follow the camera: recenter onto its absolute world position.
        let camera = WorldPos::new(10_000.5, 0.0, 0.0);
        grid.recenter_to(camera);
        let view = OriginView::from_world(&grid, camera);
        let processor = EntityLodProcessor::new(schedule());
        let mut dormancy = DormancySet::new();

        let near = ent(1, 1); // ~5 m from camera → band 0
        let far = ent(2, 1); // ~1 km from camera → dormant
        let population = [
            (near, WorldPos::new(10_005.5, 0.0, 0.0)),
            (far, WorldPos::new(11_000.5, 0.0, 0.0)),
        ];

        let r = view.drive_world(&processor, &population, 0, &mut dormancy);
        assert_eq!(r.to_tick, alloc::vec![near]);
        assert_eq!(r.slept, alloc::vec![far]);
        assert!(dormancy.is_dormant(far));
    }

}
