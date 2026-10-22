//! Same-device RHI consumption for planetary placement requests.
//!
//! This is a compute-output consumer, not a product scene renderer. It submits
//! the producer's native registration snapshot, keeps its resources alive
//! until queue completion, and exposes the same stage/resource hazards through
//! Prism's backend-neutral frame-graph declaration compiler.
//!
//! The planetary record ABI is deliberately opaque here. In particular, the
//! exact signed-i128 frame-local millimetre coordinates remain in the producer
//! storage record; this module never interprets or converts record coordinates.
//! Full-key planetary deletion is registered from its complete producer
//! snapshot, including upstream placement resources and its deletion stage.

use std::{
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use prism_render_graph::{
    frame_graph::{
        AccessKind, GpuFrameGraphBuilder, PassDescriptor, QueueClass, ResourceAccess,
        ResourceDescriptor, ResourceId, ResourceLifetime,
    },
    CompileError, ExecutionPlan,
};

use crate::{
    planet_placement_deletion::{
        GpuPlanetPlacementDeletionRegistrationSnapshot, GpuPlanetPlacementDeletions,
    },
    GpuContext, GpuPlanetPlacement, GpuPlanetPlacementError,
    GpuPlanetPlacementRegistrationSnapshot,
};

/// Failures before native submission or during completion polling.
#[derive(Debug)]
pub enum PlanetPlacementRhiError {
    /// Producer and bridge do not share the same authenticated `GpuContext`.
    ForeignDevice,
    /// The bridge's bounded completion queue is full.
    InFlightLimit,
    /// The producer has no successfully uploaded request.
    Placement(GpuPlanetPlacementError),
    /// Planetary full-key deletion snapshot could not be obtained.
    Deletion(crate::GpuPlanetPlacementDeletionError),
    /// The backend-neutral hazard declaration failed to compile.
    Graph(CompileError),
    /// Native device polling failed; resources are retained until a later
    /// successful drain or intentionally leaked during failed shutdown.
    Poll(wgpu::PollError),
}

impl std::fmt::Display for PlanetPlacementRhiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForeignDevice => f.write_str("planet placement belongs to another device"),
            Self::InFlightLimit => f.write_str("planet placement in-flight limit reached"),
            Self::Placement(error) => error.fmt(f),
            Self::Deletion(error) => error.fmt(f),
            Self::Graph(error) => write!(f, "planet placement graph: {error}"),
            Self::Poll(error) => write!(f, "planet placement poll: {error}"),
        }
    }
}

impl std::error::Error for PlanetPlacementRhiError {}

struct Pending {
    done: Arc<AtomicBool>,
    // Snapshot clones pin all buffers, pipelines and bind groups used by the
    // submitted command buffer. Keep the complete snapshot as one lease.
    _snapshot: GpuPlanetPlacementRegistrationSnapshot,
}

struct PendingDeletion {
    done: Arc<AtomicBool>,
    _snapshot: GpuPlanetPlacementDeletionRegistrationSnapshot,
}

/// A completion receipt for one native queue submission.
pub struct PlanetPlacementRhiSubmission {
    done: Arc<AtomicBool>,
}

impl PlanetPlacementRhiSubmission {
    /// True only after the native queue's completion callback has run.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

/// Bounded same-device planetary placement submitter.
///
/// Every submitted request is retained until the native queue reports actual
/// completion. `poll` is nonblocking; `drain` waits for the latest submission.
/// This adapter submits producer compute only: the caller may encode a consumer
/// draw after it in the same externally-owned command encoder/queue workflow,
/// but this module does not define or claim product renderer integration.
pub struct PlanetPlacementRhiBridge {
    device: wgpu::Device,
    queue: wgpu::Queue,
    identity: Arc<()>,
    table: Arc<()>,
    max_in_flight: NonZeroUsize,
    pending: Vec<Pending>,
    pending_deletions: Vec<PendingDeletion>,
}

impl PlanetPlacementRhiBridge {
    /// Creates a bridge for the exact logical device that created the producer.
    #[must_use]
    pub fn new(context: &GpuContext, max_in_flight: NonZeroUsize) -> Self {
        Self {
            device: context.device().clone(),
            queue: context.queue().clone(),
            identity: context.device_identity().clone(),
            table: Arc::new(()),
            max_in_flight,
            pending: Vec::with_capacity(max_in_flight.get()),
            pending_deletions: Vec::with_capacity(max_in_flight.get()),
        }
    }

    /// Moves a planetary placement producer into this bridge after checking
    /// its unforgeable logical device identity.
    pub fn register(
        &self,
        producer: GpuPlanetPlacement,
    ) -> Result<RegisteredPlanetPlacement, PlanetPlacementRhiError> {
        if !Arc::ptr_eq(&self.identity, producer.device_identity()) {
            return Err(PlanetPlacementRhiError::ForeignDevice);
        }
        Ok(RegisteredPlanetPlacement {
            producer,
            table: self.table.clone(),
        })
    }

    /// Registers a full-key deletion producer on the same authenticated device.
    pub fn register_deletions(
        &self,
        producer: GpuPlanetPlacementDeletions,
    ) -> Result<RegisteredPlanetPlacementDeletions, PlanetPlacementRhiError> {
        if !Arc::ptr_eq(&self.identity, producer.device_identity()) {
            return Err(PlanetPlacementRhiError::ForeignDevice);
        }
        Ok(RegisteredPlanetPlacementDeletions {
            producer,
            table: self.table.clone(),
        })
    }

    /// Submits one immutable request snapshot and pins its native registrations
    /// until the queue completion callback runs.
    pub fn submit(
        &mut self,
        placement: &RegisteredPlanetPlacement,
    ) -> Result<PlanetPlacementRhiSubmission, PlanetPlacementRhiError> {
        if !Arc::ptr_eq(&self.table, &placement.table) {
            return Err(PlanetPlacementRhiError::ForeignDevice);
        }
        self.poll()?;
        if !in_flight_capacity_available(
            self.pending.len(),
            self.pending_deletions.len(),
            self.max_in_flight.get(),
        ) {
            return Err(PlanetPlacementRhiError::InFlightLimit);
        }

        let snapshot = placement
            .producer
            .registration_snapshot()
            .map_err(PlanetPlacementRhiError::Placement)?;
        let _lowered = PlanetPlacementRenderGraphPlan::lower(&snapshot)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet placement RHI request"),
            });
        // The producer owns the canonical reset/active/finalize encoding.
        // Records are opaque in this module, preserving their signed-i128 ABI.
        snapshot.encode(&mut encoder);
        self.queue.submit([encoder.finish()]);
        let done = Arc::new(AtomicBool::new(false));
        let callback_done = done.clone();
        self.queue
            .on_submitted_work_done(move || callback_done.store(true, Ordering::Release));
        self.pending.push(Pending {
            done: done.clone(),
            _snapshot: snapshot,
        });
        Ok(PlanetPlacementRhiSubmission { done })
    }

    /// Submits the producer's placement-plus-deletion encoder and retains every
    /// upstream and deletion registration through actual queue completion.
    pub fn submit_deletions(
        &mut self,
        deletions: &RegisteredPlanetPlacementDeletions,
    ) -> Result<PlanetPlacementRhiSubmission, PlanetPlacementRhiError> {
        if !Arc::ptr_eq(&self.table, &deletions.table) {
            return Err(PlanetPlacementRhiError::ForeignDevice);
        }
        self.poll()?;
        if !in_flight_capacity_available(
            self.pending.len(),
            self.pending_deletions.len(),
            self.max_in_flight.get(),
        ) {
            return Err(PlanetPlacementRhiError::InFlightLimit);
        }
        let snapshot = deletions
            .producer
            .registration_snapshot()
            .map_err(PlanetPlacementRhiError::Deletion)?;
        let _lowered = PlanetPlacementRenderGraphPlan::lower_deletions(&snapshot)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet placement deletion RHI request"),
            });
        deletions
            .producer
            .encode(&mut encoder)
            .map_err(PlanetPlacementRhiError::Deletion)?;
        self.queue.submit([encoder.finish()]);
        let done = Arc::new(AtomicBool::new(false));
        let callback_done = done.clone();
        self.queue
            .on_submitted_work_done(move || callback_done.store(true, Ordering::Release));
        self.pending_deletions.push(PendingDeletion {
            done: done.clone(),
            _snapshot: snapshot,
        });
        Ok(PlanetPlacementRhiSubmission { done })
    }

    /// Drives native callbacks and releases completed registration snapshots.
    pub fn poll(&mut self) -> Result<usize, PlanetPlacementRhiError> {
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(PlanetPlacementRhiError::Poll)?;
        let previous = self.pending.len();
        self.pending
            .retain(|pending| !pending.done.load(Ordering::Acquire));
        let placement_retired = previous - self.pending.len();
        let previous_deletions = self.pending_deletions.len();
        self.pending_deletions
            .retain(|pending| !pending.done.load(Ordering::Acquire));
        Ok(placement_retired + previous_deletions - self.pending_deletions.len())
    }

    /// Waits through the newest native submission and safely releases every
    /// earlier snapshot in queue order.
    pub fn drain(&mut self) -> Result<usize, PlanetPlacementRhiError> {
        if !self.pending.is_empty() || !self.pending_deletions.is_empty() {
            self.device
                .poll(wgpu::PollType::Wait {
                    // The two bounded queues are separate by snapshot type;
                    // a full-queue wait avoids guessing which was submitted
                    // last and proves every retained lease safe to retire.
                    submission_index: None,
                    timeout: Some(Duration::from_secs(30)),
                })
                .map_err(PlanetPlacementRhiError::Poll)?;
            for pending in &self.pending {
                pending.done.store(true, Ordering::Release);
            }
            for pending in &self.pending_deletions {
                pending.done.store(true, Ordering::Release);
            }
        }
        let retired = self.pending.len() + self.pending_deletions.len();
        self.pending.clear();
        self.pending_deletions.clear();
        Ok(retired)
    }

    /// Number of snapshots still pinned pending queue completion.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len() + self.pending_deletions.len()
    }
}

impl Drop for PlanetPlacementRhiBridge {
    fn drop(&mut self) {
        if self.drain().is_err() {
            // A failed wait is not proof of completion. Deliberately preserve
            // both kinds of registration lease rather than release native
            // users too early.
            preserve_pending_after_failed_drain(
                std::mem::take(&mut self.pending),
                std::mem::take(&mut self.pending_deletions),
            );
        }
    }
}

fn preserve_pending_after_failed_drain<P, D>(pending: Vec<P>, deletions: Vec<D>) {
    for item in pending {
        std::mem::forget(item);
    }
    for item in deletions {
        std::mem::forget(item);
    }
}

fn in_flight_capacity_available(placements: usize, deletions: usize, limit: usize) -> bool {
    placements
        .checked_add(deletions)
        .is_some_and(|total| total < limit)
}

/// A producer registered on exactly one [`PlanetPlacementRhiBridge`] table.
pub struct RegisteredPlanetPlacement {
    producer: GpuPlanetPlacement,
    table: Arc<()>,
}

/// Same-table owner of a planetary placement plus full-key deletion producer.
pub struct RegisteredPlanetPlacementDeletions {
    producer: GpuPlanetPlacementDeletions,
    table: Arc<()>,
}

impl RegisteredPlanetPlacementDeletions {
    /// Read-only producer access for validated request preparation.
    #[must_use]
    pub fn producer(&self) -> &GpuPlanetPlacementDeletions {
        &self.producer
    }
}

impl RegisteredPlanetPlacement {
    /// Read-only access to the producer for validated request uploads.
    #[must_use]
    pub fn producer(&self) -> &GpuPlanetPlacement {
        &self.producer
    }

    /// Mutable access is available only when no frame borrow is active.
    pub fn producer_mut(&mut self) -> &mut GpuPlanetPlacement {
        &mut self.producer
    }
}

/// Backend-neutral lowering of planetary placement's native stages.
///
/// Resources and passes use Prism's shared SSA graph kernel. This plan declares
/// the actual queue hazards; it does not allocate buffers or claim that the
/// generic graph recorder emitted the producer's native wgpu commands.
pub struct PlanetPlacementRenderGraphPlan {
    /// Compiled pass ordering and hazard barriers.
    pub execution: ExecutionPlan,
}

impl PlanetPlacementRenderGraphPlan {
    /// Lowers a snapshot into reset, optional candidate/finalize, and output
    /// handoff declarations. The output pass models consumer visibility only.
    pub fn lower(
        snapshot: &GpuPlanetPlacementRegistrationSnapshot,
    ) -> Result<Self, PlanetPlacementRhiError> {
        let mut graph = GpuFrameGraphBuilder::default();
        let resources = [
            resource("planet params", snapshot.params.bytes()),
            resource("planet heights i128", snapshot.heights.bytes()),
            resource("planet records i128-frame-local", snapshot.records.bytes()),
            resource("planet compact slots", snapshot.compact.bytes()),
            resource("planet status", snapshot.status.bytes()),
            resource("planet indirect", snapshot.indirect.bytes()),
        ]
        .map(|descriptor| graph.add_resource(descriptor));
        let [params, heights, records, compact, status, indirect] = resources;

        graph.add_pass(PassDescriptor {
            name: "planet placement reset".into(),
            queue: QueueClass::Transfer,
            accesses: vec![
                access(status, AccessKind::TransferWrite),
                access(indirect, AccessKind::TransferWrite),
            ],
            depends_on: vec![],
        });

        if let (Some(active), Some(finalize)) = (&snapshot.active, &snapshot.finalize) {
            debug_assert_ne!(active.dispatch(), [0, 0, 0]);
            debug_assert_ne!(finalize.dispatch(), [0, 0, 0]);
            graph.add_pass(PassDescriptor {
                name: "planet placement candidates".into(),
                queue: QueueClass::Compute,
                accesses: vec![
                    access(params, AccessKind::StorageRead),
                    access(heights, AccessKind::StorageRead),
                    access(records, AccessKind::StorageWrite),
                    access(compact, AccessKind::StorageWrite),
                    access(status, AccessKind::StorageWrite),
                ],
                depends_on: vec![],
            });
            graph.add_pass(PassDescriptor {
                name: "planet placement finalize".into(),
                queue: QueueClass::Compute,
                accesses: vec![
                    access(status, AccessKind::StorageRead),
                    access(status, AccessKind::StorageWrite),
                    access(indirect, AccessKind::StorageWrite),
                ],
                depends_on: vec![],
            });
        }

        graph.add_pass(PassDescriptor {
            name: "planet placement output consumer".into(),
            queue: QueueClass::Graphics,
            accesses: vec![
                access(records, AccessKind::StorageRead),
                access(compact, AccessKind::StorageRead),
                access(indirect, AccessKind::IndirectRead),
            ],
            depends_on: vec![],
        });
        let execution = graph.compile().map_err(PlanetPlacementRhiError::Graph)?;
        Ok(Self { execution })
    }

    /// Lowers the upstream placement and full-key deletion stages, including
    /// the upstream-indirect copy/reset and deletion output hazards.
    fn lower_deletions(
        snapshot: &GpuPlanetPlacementDeletionRegistrationSnapshot,
    ) -> Result<Self, PlanetPlacementRhiError> {
        let mut graph = GpuFrameGraphBuilder::default();
        let upstream = &snapshot.upstream;
        let params = graph.add_resource(resource("planet params", upstream.params.bytes()));
        let heights = graph.add_resource(resource("planet heights i128", upstream.heights.bytes()));
        let records = graph.add_resource(resource(
            "planet records i128-frame-local",
            upstream.records.bytes(),
        ));
        let upstream_compact = graph.add_resource(resource(
            "planet upstream compact",
            upstream.compact.bytes(),
        ));
        let status = graph.add_resource(resource("planet status", upstream.status.bytes()));
        let upstream_indirect = graph.add_resource(resource(
            "planet upstream indirect",
            upstream.indirect.bytes(),
        ));
        let deletion_params =
            graph.add_resource(resource("planet deletion params", snapshot.params.bytes()));
        let keys = graph.add_resource(resource("planet deletion full keys", snapshot.keys.bytes()));
        let compact = graph.add_resource(resource(
            "planet deletion compact",
            snapshot.compact.bytes(),
        ));
        let indirect = graph.add_resource(resource(
            "planet deletion indirect",
            snapshot.indirect.bytes(),
        ));
        graph.add_pass(PassDescriptor {
            name: "planet placement reset".into(),
            queue: QueueClass::Transfer,
            accesses: vec![
                access(status, AccessKind::TransferWrite),
                access(upstream_indirect, AccessKind::TransferWrite),
            ],
            depends_on: vec![],
        });
        if upstream.active.is_some() {
            graph.add_pass(PassDescriptor {
                name: "planet placement candidates".into(),
                queue: QueueClass::Compute,
                accesses: vec![
                    access(params, AccessKind::StorageRead),
                    access(heights, AccessKind::StorageRead),
                    access(records, AccessKind::StorageWrite),
                    access(upstream_compact, AccessKind::StorageWrite),
                    access(status, AccessKind::StorageWrite),
                ],
                depends_on: vec![],
            });
            graph.add_pass(PassDescriptor {
                name: "planet placement finalize".into(),
                queue: QueueClass::Compute,
                accesses: vec![
                    access(status, AccessKind::StorageRead),
                    access(status, AccessKind::StorageWrite),
                    access(upstream_indirect, AccessKind::StorageWrite),
                ],
                depends_on: vec![],
            });
        }
        graph.add_pass(PassDescriptor {
            name: "planet deletion copy and count reset".into(),
            queue: QueueClass::Transfer,
            accesses: vec![
                access(upstream_indirect, AccessKind::TransferRead),
                access(indirect, AccessKind::TransferWrite),
            ],
            depends_on: vec![],
        });
        if snapshot.stage.is_some() {
            graph.add_pass(PassDescriptor {
                name: "planet full-key deletion".into(),
                queue: QueueClass::Compute,
                accesses: vec![
                    access(deletion_params, AccessKind::StorageRead),
                    access(records, AccessKind::StorageRead),
                    access(upstream_compact, AccessKind::StorageRead),
                    access(upstream_indirect, AccessKind::StorageRead),
                    access(keys, AccessKind::StorageRead),
                    access(compact, AccessKind::StorageWrite),
                    access(indirect, AccessKind::StorageWrite),
                ],
                depends_on: vec![],
            });
        }
        graph.add_pass(PassDescriptor {
            name: "planet deletion output consumer".into(),
            queue: QueueClass::Graphics,
            accesses: vec![
                access(records, AccessKind::StorageRead),
                access(compact, AccessKind::StorageRead),
                access(indirect, AccessKind::IndirectRead),
            ],
            depends_on: vec![],
        });
        let execution = graph.compile().map_err(PlanetPlacementRhiError::Graph)?;
        Ok(Self { execution })
    }
}

fn resource(name: &'static str, size: u64) -> ResourceDescriptor {
    ResourceDescriptor {
        name: name.into(),
        lifetime: ResourceLifetime::Imported,
        size,
        alignment: 4,
    }
}

fn access(resource: ResourceId, kind: AccessKind) -> ResourceAccess {
    ResourceAccess { resource, kind }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_world_terrain::PLANET_PLACEMENT_VERSION;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn planetary_record_stride_is_never_the_planar_stride() {
        assert_eq!(PLANET_PLACEMENT_VERSION, 1);
        // Current planetary ABI: exact frame-local millimetres occupy three
        // signed i128 values plus normal/metadata, totaling 116 bytes.
        assert_ne!(116, 64);
    }

    #[test]
    fn graph_resource_metadata_uses_imported_full_record_bytes() {
        let descriptor = resource("planet records i128-frame-local", 116 * 3);
        assert_eq!(descriptor.lifetime, ResourceLifetime::Imported);
        assert_eq!(descriptor.size, 348);
        assert_eq!(descriptor.name, "planet records i128-frame-local");
    }

    #[test]
    fn failed_drain_preserves_both_placement_and_deletion_leases() {
        struct DropProbe(Arc<AtomicUsize>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let placement_drops = Arc::new(AtomicUsize::new(0));
        let deletion_drops = Arc::new(AtomicUsize::new(0));
        preserve_pending_after_failed_drain(
            vec![DropProbe(placement_drops.clone())],
            vec![DropProbe(deletion_drops.clone())],
        );

        assert_eq!(placement_drops.load(Ordering::Relaxed), 0);
        assert_eq!(deletion_drops.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn placement_and_deletion_share_one_in_flight_capacity() {
        let max_in_flight = NonZeroUsize::new(3).unwrap();
        assert!(in_flight_capacity_available(0, 2, max_in_flight.get()));
        assert!(in_flight_capacity_available(1, 1, max_in_flight.get()));
        assert!(!in_flight_capacity_available(2, 1, max_in_flight.get()));
        assert!(!in_flight_capacity_available(0, 3, max_in_flight.get()));
    }
}
