//! Durable full-key scatter deletions, with staged in-process publication.
//!
//! Run: cargo run --offline -p prism_world_runtime --example scatter_deletions
//! The default `std` feature supplies FileJournal; without it this example only
//! prints the feature requirement so no-default-features all-target checks work.
#![forbid(unsafe_code)]

#[cfg(not(feature = "std"))]
fn main() {
    println!("scatter_deletions requires the prism_world_runtime `std` feature");
}

#[cfg(feature = "std")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    demo::run()
}

#[cfg(feature = "std")]
mod demo {
    use std::{
        error::Error,
        fmt::Debug,
        fs, io,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use prism_world_persist::{
        ActorId, FieldId, FileJournal, LogicalClock, PersistentId, Snapshot, Timeline,
        TimelineError, Transaction, TransactionId, WorldState,
    };
    use prism_world_residency::{
        scatter, ScatterBudget, ScatterConfig, ScatterDeletionBudget, ScatterDeletionSet,
        ScatterOutput, ScatterRegion,
    };
    use prism_world_runtime::{ScatterDeletionBinding, ScatterDeletionCodecBudget};

    type Result<T> = std::result::Result<T, Box<dyn Error>>;

    // TimelineError deliberately need not implement Display/Error. Convert custom
    // errors through Debug rather than assuming every layer supports `?` directly.
    fn debug_error(error: impl Debug) -> Box<dyn Error> {
        format!("{error:?}").into()
    }

    /// Owns only an exclusively created directory and its one journal file.
    struct TemporaryJournal {
        directory: PathBuf,
        journal: PathBuf,
    }

    impl TemporaryJournal {
        fn create() -> Result<Self> {
            let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
            for attempt in 0..128 {
                let directory = std::env::temp_dir().join(format!(
                    "prism-scatter-example-{}-{nonce}-{attempt}",
                    std::process::id()
                ));
                match fs::create_dir(&directory) {
                    Ok(()) => {
                        let journal = directory.join("deletions.journal");
                        return Ok(Self { directory, journal });
                    }
                    // A collision never grants ownership of an existing path.
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => return Err(error.into()),
                }
            }
            Err("could not create an exclusive temporary example directory".into())
        }
    }

    impl Drop for TemporaryJournal {
        fn drop(&mut self) {
            // Never recurse through the system temp/build directory or delete
            // unknown entries. Journal handles are dropped before this guard.
            if let Err(error) = fs::remove_file(&self.journal)
                && error.kind() != io::ErrorKind::NotFound
            {
                eprintln!("could not remove {}: {error}", self.journal.display());
            }
            if let Err(error) = fs::remove_dir(&self.directory)
                && error.kind() != io::ErrorKind::NotFound
            {
                eprintln!("could not remove {}: {error}", self.directory.display());
            }
        }
    }

    fn budget() -> ScatterDeletionCodecBudget {
        ScatterDeletionCodecBudget {
            max_encoded_bytes: 1_000_000,
            deletions: ScatterDeletionBudget {
                max_keys: 10_000,
                max_key_bytes: 1_000_000,
            },
        }
    }

    fn generate_source() -> Result<ScatterOutput> {
        let region = ScatterRegion::new([-2000; 3], [2000; 3]).map_err(debug_error)?;
        scatter(
            ScatterConfig {
                seed: 17,
                version: 1,
                salt: 2,
                grid_spacing_mm: 1000,
                min_distance_mm: 600,
            },
            region,
            ScatterBudget {
                max_candidate_evaluations: 1_000_000,
                max_output_points: 10_000,
                max_output_bytes: 1_000_000,
            },
        )
        .map_err(debug_error)
    }

    fn stage_and_publish(
        timeline: &mut Timeline,
        journal: &mut FileJournal,
        operation: impl FnOnce(&mut Timeline) -> std::result::Result<Transaction, TimelineError>,
    ) -> Result<()> {
        // This deep-clones the complete log, state and undo/redo history. It is
        // demonstration-only, not a production incremental transaction design.
        let mut staged = timeline.clone();
        let transaction = operation(&mut staged).map_err(debug_error)?;
        // FileJournal::append flushes and syncs before returning success. On an
        // I/O error its handle is poisoned; never publish staged state or retry
        // against this handle. Every caller propagates this error and stops.
        // A real caller must reopen/recover disk authority before continuing:
        // failed rollback may leave additional complete records on disk.
        journal
            .append(transaction)
            .map_err(|error| -> Box<dyn Error> {
                format!(
                    "append failed; stop publication and reopen/recover before reuse: {error:?}"
                )
                .into()
            })?;
        *timeline = staged;
        // This is not crash-atomic across disk and memory: a crash between sync
        // and publication requires replay from the journal on restart.
        Ok(())
    }

    fn reopen_and_verify(
        path: &Path,
        timeline: &Timeline,
        binding: ScatterDeletionBinding,
        expected_deletions: &ScatterDeletionSet,
        original_count: usize,
    ) -> Result<FileJournal> {
        let journal = FileJournal::open(path).map_err(debug_error)?;
        let recovered = Timeline::replay(&Snapshot::new(0, WorldState::new()), journal.log())
            .map_err(debug_error)?;
        if &recovered != timeline.state() {
            return Err("reopened journal differs from the published timeline".into());
        }
        let snapshot = binding.read(&recovered, budget()).map_err(debug_error)?;
        if snapshot.binding() != binding || snapshot.deletions() != expected_deletions {
            return Err("replay changed the author binding or complete deletion keys".into());
        }

        // Both buffers are newly generated. No original point/proxy Vec survives
        // the write; only full structural deletion keys carry persistent identity.
        let regenerated = generate_source()?;
        if regenerated.points.len() != original_count {
            return Err("regenerated source count changed".into());
        }
        let upstream_key = regenerated.cache_key;
        let filtered = snapshot.filter_source(regenerated);
        let reference = generate_source()?;
        let expected = reference
            .points
            .iter()
            .filter(|point| !expected_deletions.is_deleted(point.key));
        if !filtered.source.points.iter().eq(expected)
            || filtered.removed != expected_deletions.keys().len()
            || filtered.source.cache_key != upstream_key
            || filtered.snapshot != snapshot
        {
            return Err(
                "replay/filter changed point order, removal count or cache dependencies".into(),
            );
        }
        Ok(journal)
    }

    pub fn run() -> Result<()> {
        // Declared first, so the guard runs after journal handles on all Result
        // error paths. It touches only paths created by this example process.
        let temporary = TemporaryJournal::create()?;
        let original = generate_source()?;
        let original_count = original.points.len();
        let full_key = original
            .points
            .first()
            .ok_or("example source unexpectedly contains no points")?
            .key;
        // This stable scene-author node ID is separate from any candidate key;
        // field IDs belong to the calling application's persistence schema.
        let binding = ScatterDeletionBinding {
            object: PersistentId(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210),
            field: FieldId(931),
        };
        let deletions =
            ScatterDeletionSet::new(vec![full_key], budget().deletions).map_err(debug_error)?;
        drop(original);

        let mut timeline = Timeline::new();
        let mut journal = FileJournal::open(&temporary.journal).map_err(debug_error)?;
        let delta = binding
            .delta(&deletions, budget().max_encoded_bytes)
            .map_err(debug_error)?;
        stage_and_publish(&mut timeline, &mut journal, |staged| {
            // commit derives the inverse from the staged authoritative state;
            // no hand-written inverse or prior value is trusted here.
            staged.commit(
                TransactionId(1),
                ActorId(7),
                LogicalClock(1),
                binding.object,
                delta,
            )
        })?;
        drop(journal);
        let mut journal = reopen_and_verify(
            &temporary.journal,
            &timeline,
            binding,
            &deletions,
            original_count,
        )?;

        // Replay restores WorldState, not an editable Timeline/undo stack. This
        // example retains its published in-process Timeline across journal opens;
        // it does not claim to reconstruct undo/redo history after process restart.
        stage_and_publish(&mut timeline, &mut journal, |staged| {
            staged.undo(TransactionId(2), ActorId(7), LogicalClock(2))
        })?;
        drop(journal);
        let mut journal = reopen_and_verify(
            &temporary.journal,
            &timeline,
            binding,
            &ScatterDeletionSet::default(),
            original_count,
        )?;

        stage_and_publish(&mut timeline, &mut journal, |staged| {
            staged.redo(TransactionId(3), ActorId(7), LogicalClock(3))
        })?;
        drop(journal);
        let journal = reopen_and_verify(
            &temporary.journal,
            &timeline,
            binding,
            &deletions,
            original_count,
        )?;
        if journal.log().len() != 3 {
            return Err("expected commit, undo and redo as three durable transactions".into());
        }
        drop(journal);
        println!(
            "Verified {original_count} regenerated source points: delete 1, undo 0, redo 1; \
             3 staged durable transactions replayed successfully."
        );
        // The guard removes the closed journal and its now-empty directory.
        Ok(())
    }
}
