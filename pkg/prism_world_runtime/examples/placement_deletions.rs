//! Durable projected-placement deletion, undo, redo and fresh regeneration.
//!
//! Run with `cargo run -p prism_world_runtime --features placement
//! --example placement_deletions`. Uses only its exclusively created temp file.
#![forbid(unsafe_code)]

use std::{
    error::Error,
    fs, io,
    mem::size_of,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use prism_world_persist::{
    ActorId, FieldId, FileJournal, LogicalClock, PersistentId, Snapshot, Timeline, TimelineError,
    Transaction, TransactionId, WorldState,
};
use prism_world_residency::{
    DensityBrush, DensityBrushOp, ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig,
};
use prism_world_runtime::{PlacementDeletionBinding, PlacementDeletionCodecBudget};
use prism_world_terrain::{
    scatter_on_heightfield, BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler,
    HeightfieldPlacementBudget, HeightfieldPlacementConfig, HeightfieldPlacementDeletionBudget,
    HeightfieldPlacementDeletionSet, HeightfieldPlacementFilters, HeightfieldPlacementOutput,
    HeightfieldPlacementRegion, HeightfieldSpec, HeightfieldTile, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, TileKey,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn debug_error(error: impl std::fmt::Debug) -> Box<dyn Error> {
    format!("{error:?}").into()
}

struct TemporaryJournal(PathBuf);

impl TemporaryJournal {
    fn create() -> Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        for attempt in 0..128 {
            let path = std::env::temp_dir().join(format!(
                "prism-placement-example-{}-{nonce}-{attempt}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err("could not create an exclusive example directory".into())
    }

    fn path(&self) -> PathBuf {
        self.0.join("placement.journal")
    }
}

impl Drop for TemporaryJournal {
    fn drop(&mut self) {
        // No recursive removal: never delete unknown entries or shared targets.
        for result in [fs::remove_file(self.path()), fs::remove_dir(&self.0)] {
            if let Err(error) = result
                && error.kind() != io::ErrorKind::NotFound
            {
                eprintln!("temporary placement journal cleanup failed: {error}");
            }
        }
    }
}

struct Ramp;

impl HeightSampler for Ramp {
    type Error = core::convert::Infallible;

    fn sample_height(
        &self,
        sample: HeightSample,
    ) -> std::result::Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(
            i64::from(sample.u_q16()) / 256 - i64::from(sample.v_q16()) / 256,
        ))
    }
}

fn budget() -> PlacementDeletionCodecBudget {
    PlacementDeletionCodecBudget {
        max_encoded_bytes: 16 + 400 * 56,
        deletions: HeightfieldPlacementDeletionBudget {
            max_keys: 400,
            max_key_bytes: 400 * size_of::<prism_world_terrain::HeightfieldPlacementKey>(),
        },
    }
}

fn generate() -> Result<HeightfieldPlacementOutput> {
    let surface = PlanarHeightfieldSurface::new(
        Arc::new(
            HeightfieldTile::bake(
                TileKey::root(CubeFace::PositiveY),
                HeightfieldSpec::new(3, 1).map_err(debug_error)?,
                &Ramp,
                BakeLimits {
                    max_samples: 9,
                    max_bytes: 18,
                },
            )
            .map_err(debug_error)?,
        ),
        PlanarHeightfieldEmbedding {
            sample_origin_mm: [-1_000, 100, -1_000],
            sample_spacing_mm: 1_000,
        },
    )?;
    let field = ScatterDensityField::new(
        ScatterFieldConfig {
            seed: 8,
            version: 1,
            base_density: u16::MAX,
            base_species: 3,
            noise: None,
        },
        vec![DensityBrush {
            id: 1,
            center_mm: [0, 107, 0],
            inner_radius_mm: 600,
            outer_radius_mm: 600,
            opacity: u16::MAX,
            density: DensityBrushOp::Add(0),
            species: Some(17),
        }],
        ScatterFieldBuildBudget {
            max_brushes: 1,
            max_brush_bytes: size_of::<DensityBrush>(),
        },
    )?;
    Ok(scatter_on_heightfield(
        HeightfieldPlacementConfig {
            seed: 17,
            version: 1,
            salt: 2,
            grid_spacing_mm: 100,
            min_distance_mm: 0,
            vertical_offset_mm: 7,
        },
        HeightfieldPlacementRegion::new([-1_000; 2], [1_000; 2])?,
        &field,
        &surface,
        HeightfieldPlacementFilters {
            surface_height_mm: [i64::MIN, i64::MAX],
            max_slope_rise: u32::MAX,
            max_slope_run: 1,
        },
        HeightfieldPlacementBudget {
            max_candidate_evaluations: 400,
            max_sample_reads: 1_600,
            max_field_work: 10_000,
            max_output_points: 400,
            max_output_bytes: 1_000_000,
        },
    )?)
}

fn publish(
    timeline: &mut Timeline,
    journal: &mut FileJournal,
    operation: impl FnOnce(&mut Timeline) -> std::result::Result<Transaction, TimelineError>,
) -> Result<()> {
    // Complete clone is demo-only: history/state cost is not incremental.
    let mut staged = timeline.clone();
    let transaction = operation(&mut staged).map_err(debug_error)?;
    journal
        .append(transaction)
        .map_err(|error| -> Box<dyn Error> {
            format!("append failed; stop publication and reopen/recover authority: {error:?}")
                .into()
        })?;
    // Disk sync precedes publication. A crash in between requires replay; these
    // are not a cross-resource atomic instruction. Never reuse a poisoned writer.
    *timeline = staged;
    Ok(())
}

fn recover(
    path: &Path,
    timeline: &Timeline,
    binding: PlacementDeletionBinding,
    expected: &HeightfieldPlacementDeletionSet,
) -> Result<FileJournal> {
    let journal = FileJournal::open(path).map_err(debug_error)?;
    let recovered = Timeline::replay(&Snapshot::new(0, WorldState::new()), journal.log())
        .map_err(debug_error)?;
    assert_eq!(&recovered, timeline.state());
    let snapshot = binding.read(&recovered, budget())?;
    assert_eq!(snapshot.binding(), binding);
    assert_eq!(snapshot.deletions(), expected);
    let original = generate()?;
    let reference: Vec<_> = original
        .points
        .iter()
        .zip(&original.attributes)
        .filter(|(point, _)| !expected.is_deleted(point.key))
        .map(|(point, attribute)| (*point, *attribute))
        .collect();
    let upstream_cache = original.cache_key.clone();
    drop(original);
    // Neither geometry nor attributes survive eviction; reconstruct from inputs.
    let filtered = snapshot.filter(generate()?)?;
    assert!(filtered
        .source
        .points
        .iter()
        .copied()
        .zip(filtered.source.attributes.iter().copied())
        .eq(reference));
    assert_eq!(filtered.source.cache_key, upstream_cache);
    assert_eq!(filtered.removed, expected.keys().len());
    assert_eq!(filtered.snapshot, snapshot);
    Ok(journal)
}

fn main() -> Result<()> {
    let temporary = TemporaryJournal::create()?;
    let original = generate()?;
    assert_eq!(original.points.len(), 400);
    let deletions =
        HeightfieldPlacementDeletionSet::new(vec![original.points[137].key], budget().deletions)?;
    drop(original);
    let binding = PlacementDeletionBinding {
        object: PersistentId(0x1234_abcd),
        field: FieldId(932),
    };
    let mut timeline = Timeline::new();
    let mut journal = FileJournal::open(temporary.path()).map_err(debug_error)?;
    let delta = binding.delta(&deletions, budget().max_encoded_bytes)?;
    publish(&mut timeline, &mut journal, |staged| {
        staged.commit(
            TransactionId(1),
            ActorId(7),
            LogicalClock(1),
            binding.object,
            delta,
        )
    })?;
    drop(journal);
    let mut journal = recover(&temporary.path(), &timeline, binding, &deletions)?;
    // Retain in-process undo history; replay only reconstructs WorldState.
    publish(&mut timeline, &mut journal, |staged| {
        staged.undo(TransactionId(2), ActorId(7), LogicalClock(2))
    })?;
    drop(journal);
    let mut journal = recover(
        &temporary.path(),
        &timeline,
        binding,
        &HeightfieldPlacementDeletionSet::default(),
    )?;
    publish(&mut timeline, &mut journal, |staged| {
        staged.redo(TransactionId(3), ActorId(7), LogicalClock(3))
    })?;
    drop(journal);
    let journal = recover(&temporary.path(), &timeline, binding, &deletions)?;
    assert_eq!(journal.log().len(), 3);
    drop(journal);
    println!("Verified 400 regenerated placements and paired attributes: delete/undo/redo=1/0/1; 3 durable transactions replayed.");
    Ok(())
}
