//! Continuous SDF queries must follow the same atomic SceneHost publication.

use core::convert::Infallible;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use prism_world_physics_bridge::{
    SdfDistanceGradient, SdfPhysicsBridgeBudget, SdfQueryResult, WorldSdfSnapshot,
    WorldSdfTransform,
};
use prism_world_physics_cell_runtime::{
    SdfCellDecoder, SdfCellSnapshot, SdfPhysicsCellHost, SdfResidencyBudget,
};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};
use prism_world_runtime::SceneHost;
use prism_world_stream::{AabbI64, CellManifest, DataLayerMask, GlobalPos, SectorId};

struct Host(Arc<AtomicBool>);

impl SceneHost for Host {
    type Error = &'static str;

    fn load_cell(&mut self, _: &CellManifest, _: &[u8]) -> Result<(), Self::Error> {
        if self.0.load(Ordering::Relaxed) {
            Err("host rejected load")
        } else {
            Ok(())
        }
    }

    fn evict_cell(&mut self, _: SectorId) -> Result<(), Self::Error> {
        if self.0.load(Ordering::Relaxed) {
            Err("host rejected eviction")
        } else {
            Ok(())
        }
    }
}

struct Decoder(SdfCellSnapshot);

impl SdfCellDecoder for Decoder {
    type Error = Infallible;

    fn decode(&mut self, _: &CellManifest, _: &[u8]) -> Result<SdfCellSnapshot, Self::Error> {
        Ok(self.0.clone())
    }
}

fn source(sector: SectorId, revision: u64, offset: i16) -> SdfCellSnapshot {
    let mut grid = SparseSdfBrickGrid::new(2, 1).expect("grid");
    grid.insert_brick(
        BrickCoord::new(0, 0, 0, 0),
        (0..8).map(|index| index + offset).collect(),
    )
    .expect("brick");
    SdfCellSnapshot::new(
        sector,
        WorldSdfSnapshot::new(
            WorldSdfTransform::new([0; 3], 100, 10).expect("transform"),
            revision,
            grid,
        ),
    )
}

/// Load/reload/evict failures preserve distance, gradient, revision and usage.
#[test]
fn distance_and_gradient_follow_committed_cell_lifecycle() {
    let sector = SectorId::new(0, 0, 0, 0);
    let manifest = CellManifest {
        sector,
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(200, 200, 200))
            .expect("bounds"),
        scene_key: 1,
        content_version: 1,
        derived_hash: 1,
        io_bytes: 1,
        data_layers: DataLayerMask::EMPTY,
    };
    let reject = Arc::new(AtomicBool::new(true));
    let mut downstream = Host(Arc::clone(&reject));
    let mut host = SdfPhysicsCellHost::new(
        &mut downstream,
        Decoder(source(sector, 1, 0)),
        SdfPhysicsBridgeBudget::new(1, 8, 80).expect("budget"),
        SdfResidencyBudget::new(1, 1, 8, 80),
    );
    let position = [25, 50, 75];
    let old = SdfQueryResult::Value(SdfDistanceGradient {
        distance_mm: 42.5,
        gradient: [0.1, 0.2, 0.4],
    });
    assert!(host.load_cell(&manifest, &[]).is_err());
    assert_eq!(
        host.query_world_distance_gradient_mm(sector, position, 0),
        Ok(SdfQueryResult::Unknown)
    );
    reject.store(false, Ordering::Relaxed);
    host.load_cell(&manifest, &[]).expect("initial load");
    assert_eq!(
        host.query_world_distance_gradient_mm(sector, position, 0),
        Ok(old)
    );
    let usage = host.resident_usage();
    let before = host.resident(sector).cloned().expect("resident");

    host.decoder_mut().0 = source(sector, 2, 7);
    reject.store(true, Ordering::Relaxed);
    assert!(host.load_cell(&manifest, &[]).is_err());
    assert_eq!(host.resident(sector), Some(&before));
    assert_eq!(host.resident_usage(), usage);
    assert_eq!(
        host.query_world_distance_gradient_mm(sector, position, 0),
        Ok(old)
    );

    reject.store(false, Ordering::Relaxed);
    host.load_cell(&manifest, &[]).expect("new revision");
    let new = SdfQueryResult::Value(SdfDistanceGradient {
        distance_mm: 112.5,
        gradient: [0.1, 0.2, 0.4],
    });
    assert_eq!(
        host.query_world_distance_gradient_mm(sector, position, 0),
        Ok(new)
    );
    assert_eq!(host.resident(sector).expect("resident").revision(), 2);
    assert!(
        host.load_cell(&manifest, &[]).is_err(),
        "equal revision is stale"
    );
    assert_eq!(
        host.query_world_distance_gradient_mm(sector, position, 0),
        Ok(new)
    );

    reject.store(true, Ordering::Relaxed);
    assert!(host.evict_cell(sector).is_err());
    assert_eq!(host.resident_usage(), usage);
    assert_eq!(
        host.query_world_distance_gradient_mm(sector, position, 0),
        Ok(new)
    );
    reject.store(false, Ordering::Relaxed);
    host.evict_cell(sector).expect("evict");
    assert_eq!(host.resident_usage(), (0, 0, 0, 0));
    assert_eq!(
        host.query_world_distance_gradient_mm(sector, position, 0),
        Ok(SdfQueryResult::Unknown)
    );
}
