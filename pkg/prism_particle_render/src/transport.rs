//! Typed transport metadata for the native Particle RenderApp -> SceneFabric bridge.
//!
//! This module does not resolve a native [`BufferId`] into a Bevy buffer and it
//! does not manufacture one. A producer-owned [`ParticleFabricLeaseIssuer`]
//! wraps the exact id already returned by the producer's resource table. The
//! resulting lease is the only physical identity accepted by
//! [`ParticleFabricSceneTransport`], together with the device generation and
//! publication frame that establish its provenance.

use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

use prism_render_driver::BufferId;
use thiserror::Error;

use crate::ParticleFabricScenePublication;

static NEXT_PARTICLE_FABRIC_LEASE_ID: AtomicU64 = AtomicU64::new(1);

/// Identity of the device/resource table that owns a native Particle buffer.
///
/// Resource ids are local to a device generation. A numeric id that happens
/// to match on another device is therefore not a valid transport binding.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParticleFabricDeviceGeneration(u64);

impl ParticleFabricDeviceGeneration {
    /// Creates a device generation supplied by the owning renderer runtime.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the renderer runtime generation.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Opaque producer-issued identity for one retained native buffer lease.
///
/// The zero value is deliberately not representable. This makes an omitted
/// lease distinguishable from a producer-issued lease without using a
/// sentinel `BufferId`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParticleFabricLeaseId(core::num::NonZeroU64);

impl ParticleFabricLeaseId {
    /// Returns the monotonically assigned lease number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Provenance carried with one native buffer lease.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ParticleFabricLeaseProvenance {
    device_generation: ParticleFabricDeviceGeneration,
    lease_id: ParticleFabricLeaseId,
    publication_frame: u64,
}

impl ParticleFabricLeaseProvenance {
    /// Device/resource-table generation that owns the leased buffer.
    #[must_use]
    pub const fn device_generation(self) -> ParticleFabricDeviceGeneration {
        self.device_generation
    }

    /// Producer-issued identity of this retained lease.
    #[must_use]
    pub const fn lease_id(self) -> ParticleFabricLeaseId {
        self.lease_id
    }

    /// Render-frame sequence for which the producer retained this lease.
    #[must_use]
    pub const fn publication_frame(self) -> u64 {
        self.publication_frame
    }
}

/// A real native Particle buffer id coupled to producer-issued provenance.
///
/// The id is private and can only be obtained from [`ParticleFabricLeaseIssuer`]
/// or by borrowing a previously issued lease. There is no constructor that
/// accepts an index/generation pair, so this contract cannot be used to create
/// a replacement id for a Bevy resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ParticleFabricBufferLease {
    buffer: BufferId,
    provenance: ParticleFabricLeaseProvenance,
}

impl ParticleFabricBufferLease {
    /// Returns the exact producer-owned id supplied to the issuer.
    #[must_use]
    pub const fn buffer(self) -> BufferId {
        self.buffer
    }

    /// Returns the device/frame/lease provenance for this id.
    #[must_use]
    pub const fn provenance(self) -> ParticleFabricLeaseProvenance {
        self.provenance
    }
}

/// Errors produced while allocating a lease identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum ParticleFabricLeaseError {
    /// The producer's lease sequence exhausted the representable non-zero ids.
    #[error("Particle Fabric lease id sequence exhausted")]
    Exhausted,
    /// A producer attempted to publish an older frame after a newer lease.
    #[error("Particle Fabric lease frame {actual} regressed after frame {previous}")]
    FrameRegression {
        /// Most recently issued publication frame.
        previous: u64,
        /// Older frame requested by the producer.
        actual: u64,
    },
}

/// Producer-owned lease issuer for one device generation.
///
/// The issuer is intentionally independent of any renderer framework. The
/// native RenderApp owns it next to its resource table and passes the exact
/// table id to [`Self::issue`]. A new device generation must use a new issuer,
/// which prevents ids from different device lifetimes from sharing an issuer
/// provenance domain.
#[derive(Clone, Debug)]
pub struct ParticleFabricLeaseIssuer {
    device_generation: ParticleFabricDeviceGeneration,
    last_publication_frame: Option<u64>,
}

impl ParticleFabricLeaseIssuer {
    /// Starts a lease sequence for one device/resource-table generation.
    #[must_use]
    pub const fn new(device_generation: ParticleFabricDeviceGeneration) -> Self {
        Self {
            device_generation,
            last_publication_frame: None,
        }
    }

    /// Returns the generation owned by this issuer.
    #[must_use]
    pub const fn device_generation(&self) -> ParticleFabricDeviceGeneration {
        self.device_generation
    }

    /// Wraps an exact producer-table id in a new retained lease.
    ///
    /// This function never allocates, derives, or replaces a [`BufferId`]. The
    /// caller remains responsible for keeping the underlying native resource
    /// alive until the host reports completion for the matching lease.
    pub fn issue(
        &mut self,
        buffer: BufferId,
        publication_frame: u64,
    ) -> Result<ParticleFabricBufferLease, ParticleFabricLeaseError> {
        if let Some(previous) = self.last_publication_frame {
            if publication_frame < previous {
                return Err(ParticleFabricLeaseError::FrameRegression {
                    previous,
                    actual: publication_frame,
                });
            }
        }

        let lease_number = NEXT_PARTICLE_FABRIC_LEASE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| ParticleFabricLeaseError::Exhausted)?;
        let lease_id =
            core::num::NonZeroU64::new(lease_number).ok_or(ParticleFabricLeaseError::Exhausted)?;

        self.last_publication_frame = Some(publication_frame);
        Ok(ParticleFabricBufferLease {
            buffer,
            provenance: ParticleFabricLeaseProvenance {
                device_generation: self.device_generation,
                lease_id: ParticleFabricLeaseId(lease_id),
                publication_frame,
            },
        })
    }
}

/// A validated native Particle publication ready to cross the RenderApp /
/// SceneFabric transport boundary.
///
/// This is deliberately separate from [`ParticleFabricScenePublication`]. The
/// latter is the existing semantic/physical publication used by native code;
/// this wrapper adds the missing device-lifetime and lease identity required
/// before a host may resolve the ids to its own buffer objects.
#[derive(Clone, Debug, PartialEq)]
pub struct ParticleFabricSceneTransport {
    publication: ParticleFabricScenePublication,
    device_generation: ParticleFabricDeviceGeneration,
    stream: Option<ParticleFabricBufferLease>,
    alive: Option<ParticleFabricBufferLease>,
}

impl ParticleFabricSceneTransport {
    /// Validates and packages a publication with its exact native leases.
    ///
    /// Empty publications carry no physical bindings. Non-empty publications
    /// require both stream and alive leases, and every draw must refer to the
    /// exact ids and buffer generations carried by those leases.
    pub fn try_new(
        publication: ParticleFabricScenePublication,
        device_generation: ParticleFabricDeviceGeneration,
        stream: Option<ParticleFabricBufferLease>,
        alive: Option<ParticleFabricBufferLease>,
    ) -> Result<Self, ParticleFabricTransportError> {
        if publication.is_empty() {
            if stream.is_some() || alive.is_some() {
                return Err(ParticleFabricTransportError::BindingsOnEmptyPublication);
            }
            return Ok(Self {
                publication,
                device_generation,
                stream: None,
                alive: None,
            });
        }

        let stream = stream.ok_or(ParticleFabricTransportError::MissingStreamLease)?;
        let alive = alive.ok_or(ParticleFabricTransportError::MissingAliveLease)?;
        validate_lease(device_generation, publication.frame(), stream)?;
        validate_lease(device_generation, publication.frame(), alive)?;
        if stream.provenance.lease_id == alive.provenance.lease_id {
            return Err(ParticleFabricTransportError::DuplicateLeaseId {
                lease_id: stream.provenance.lease_id,
            });
        }

        let first = publication
            .draws()
            .first()
            .ok_or(ParticleFabricTransportError::MissingDraws)?;
        if stream.buffer != first.stream.buffer {
            return Err(ParticleFabricTransportError::StreamBufferMismatch {
                expected: first.stream.buffer,
                actual: stream.buffer,
            });
        }
        if alive.buffer != first.alive.buffer {
            return Err(ParticleFabricTransportError::AliveBufferMismatch {
                expected: first.alive.buffer,
                actual: alive.buffer,
            });
        }
        if stream.buffer.generation() != first.stream.version {
            return Err(ParticleFabricTransportError::StreamVersionMismatch {
                expected: first.stream.version,
                actual: stream.buffer.generation(),
            });
        }
        if alive.buffer.generation() != first.alive.version {
            return Err(ParticleFabricTransportError::AliveVersionMismatch {
                expected: first.alive.version,
                actual: alive.buffer.generation(),
            });
        }

        for draw in publication.draws() {
            if draw.stream.buffer != stream.buffer {
                return Err(ParticleFabricTransportError::StreamBufferMismatch {
                    expected: stream.buffer,
                    actual: draw.stream.buffer,
                });
            }
            if draw.alive.buffer != alive.buffer {
                return Err(ParticleFabricTransportError::AliveBufferMismatch {
                    expected: alive.buffer,
                    actual: draw.alive.buffer,
                });
            }
        }

        Ok(Self {
            publication,
            device_generation,
            stream: Some(stream),
            alive: Some(alive),
        })
    }

    /// Returns the validated semantic/physical publication.
    #[must_use]
    pub fn publication(&self) -> &ParticleFabricScenePublication {
        &self.publication
    }

    /// Returns the device generation required for host resolution.
    #[must_use]
    pub const fn device_generation(&self) -> ParticleFabricDeviceGeneration {
        self.device_generation
    }

    /// Returns the validated stream lease for a non-empty publication.
    #[must_use]
    pub const fn stream(&self) -> Option<ParticleFabricBufferLease> {
        self.stream
    }

    /// Returns the validated alive-index lease for a non-empty publication.
    #[must_use]
    pub const fn alive(&self) -> Option<ParticleFabricBufferLease> {
        self.alive
    }

    /// Returns the shared publication frame.
    #[must_use]
    pub const fn frame(&self) -> u64 {
        self.publication.frame()
    }
}

/// Failure before a host is allowed to resolve or import a native buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum ParticleFabricTransportError {
    /// A no-work frame must not carry stale physical bindings.
    #[error("empty Particle publication carries physical bindings")]
    BindingsOnEmptyPublication,
    /// A non-empty frame has no stream lease.
    #[error("non-empty Particle publication has no stream lease")]
    MissingStreamLease,
    /// A non-empty frame has no alive-index lease.
    #[error("non-empty Particle publication has no alive lease")]
    MissingAliveLease,
    /// A non-empty frame has no draw metadata to authenticate against.
    #[error("non-empty Particle publication has no draw metadata")]
    MissingDraws,
    /// A lease was issued by another device generation.
    #[error(
        "Particle lease device generation {actual:?} does not match transport generation {expected:?}"
    )]
    DeviceGenerationMismatch {
        /// Generation required by the transport publication.
        expected: ParticleFabricDeviceGeneration,
        /// Generation carried by the supplied lease.
        actual: ParticleFabricDeviceGeneration,
    },
    /// A lease was retained for another frame.
    #[error("Particle lease frame {actual} does not match publication frame {expected}")]
    PublicationFrameMismatch {
        /// Frame carried by the publication.
        expected: u64,
        /// Frame carried by the lease.
        actual: u64,
    },
    /// The two resources accidentally reused one lease identity.
    #[error("Particle stream and alive resources reuse lease id {lease_id:?}")]
    DuplicateLeaseId {
        /// Shared lease identity detected on two physical resources.
        lease_id: ParticleFabricLeaseId,
    },
    /// The stream lease does not identify the published native buffer.
    #[error(
        "Particle stream lease buffer {actual:?} does not match publication buffer {expected:?}"
    )]
    StreamBufferMismatch {
        /// Buffer id authenticated by the publication.
        expected: BufferId,
        /// Buffer id carried by the stream lease.
        actual: BufferId,
    },
    /// The alive lease does not identify the published native buffer.
    #[error(
        "Particle alive lease buffer {actual:?} does not match publication buffer {expected:?}"
    )]
    AliveBufferMismatch {
        /// Buffer id authenticated by the publication.
        expected: BufferId,
        /// Buffer id carried by the alive lease.
        actual: BufferId,
    },
    /// The stream id generation disagrees with the physical publication.
    #[error(
        "Particle stream buffer generation {actual} does not match publication version {expected}"
    )]
    StreamVersionMismatch {
        /// Buffer generation authenticated by the publication.
        expected: u32,
        /// Buffer generation carried by the stream lease id.
        actual: u32,
    },
    /// The alive id generation disagrees with the physical publication.
    #[error(
        "Particle alive buffer generation {actual} does not match publication version {expected}"
    )]
    AliveVersionMismatch {
        /// Buffer generation authenticated by the publication.
        expected: u32,
        /// Buffer generation carried by the alive lease id.
        actual: u32,
    },
}

fn validate_lease(
    expected_device_generation: ParticleFabricDeviceGeneration,
    expected_frame: u64,
    lease: ParticleFabricBufferLease,
) -> Result<(), ParticleFabricTransportError> {
    if lease.provenance.device_generation != expected_device_generation {
        return Err(ParticleFabricTransportError::DeviceGenerationMismatch {
            expected: expected_device_generation,
            actual: lease.provenance.device_generation,
        });
    }
    if lease.provenance.publication_frame != expected_frame {
        return Err(ParticleFabricTransportError::PublicationFrameMismatch {
            expected: expected_frame,
            actual: lease.provenance.publication_frame,
        });
    }
    Ok(())
}

impl fmt::Display for ParticleFabricLeaseId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        particle_resource_payload, ParticleFabricGpuBuffer, ParticleFabricGpuDraw,
        PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE, PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
    };
    use prism_render_architecture::{
        motion::MotionSource,
        particle::indirect_draw::RendererGeometry,
        render_fabric::{
            DomainObjectHandle, DomainPayload, FrameRenderItem, RenderDomain, RenderObjectId,
            PARTICLE_ALIVE_RESOURCE, PARTICLE_STREAM_RESOURCE,
        },
    };
    use prism_render_driver::{BufferState, BufferUsages, PipelineStages};

    fn publication(
        frame: u64,
        stream: BufferId,
        alive: BufferId,
    ) -> ParticleFabricScenePublication {
        let object = RenderObjectId::Particle(DomainObjectHandle::new(3, 1));
        let stream_semantic = particle_resource_payload(
            PARTICLE_STREAM_RESOURCE,
            PARTICLE_STREAM_RESOURCE_PAYLOAD_ROLE,
        )
        .expect("stream template");
        let alive_semantic = particle_resource_payload(
            PARTICLE_ALIVE_RESOURCE,
            PARTICLE_ALIVE_RESOURCE_PAYLOAD_ROLE,
        )
        .expect("alive template");
        let metadata = |semantic, buffer, usage| ParticleFabricGpuBuffer {
            semantic,
            buffer,
            version: buffer.generation(),
            size: 256,
            usage,
            entry_state: BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        };
        ParticleFabricScenePublication::new(
            frame,
            vec![
                FrameRenderItem {
                    object_id: object,
                    domain: RenderDomain::Particle,
                    history_key: 0,
                    motion_source: MotionSource::Particle,
                    payload: DomainPayload::Resource(stream_semantic),
                },
                FrameRenderItem {
                    object_id: object,
                    domain: RenderDomain::Particle,
                    history_key: 0,
                    motion_source: MotionSource::Particle,
                    payload: DomainPayload::Resource(alive_semantic),
                },
            ],
            vec![ParticleFabricGpuDraw {
                object_id: object,
                renderer_index: 0,
                history_key: 0,
                bounds_min: [-1.0; 3],
                bounds_max: [1.0; 3],
                geometry: RendererGeometry::Sprite {
                    alive_count: 1,
                    first_instance: 0,
                },
                stream: metadata(
                    stream_semantic,
                    stream,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                ),
                alive: metadata(
                    alive_semantic,
                    alive,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                ),
            }],
        )
    }

    #[test]
    fn issuer_preserves_exact_buffer_id_and_carries_generation_and_frame() {
        let generation = ParticleFabricDeviceGeneration::new(17);
        let mut issuer = ParticleFabricLeaseIssuer::new(generation);
        let buffer = BufferId::from_parts(9, 4);

        let lease = issuer.issue(buffer, 23).expect("lease issued");

        assert_eq!(lease.buffer(), buffer);
        assert_eq!(lease.provenance().device_generation(), generation);
        assert_eq!(lease.provenance().publication_frame(), 23);
        assert_ne!(lease.provenance().lease_id().get(), 0);
    }

    #[test]
    fn transport_rejects_a_lease_from_another_device_generation() {
        let stream = BufferId::from_parts(1, 5);
        let alive = BufferId::from_parts(2, 6);
        let publication = publication(31, stream, alive);
        let mut old_issuer = ParticleFabricLeaseIssuer::new(ParticleFabricDeviceGeneration::new(4));
        let stream_lease = old_issuer.issue(stream, 31).unwrap();
        let alive_lease = old_issuer.issue(alive, 31).unwrap();

        let error = ParticleFabricSceneTransport::try_new(
            publication,
            ParticleFabricDeviceGeneration::new(5),
            Some(stream_lease),
            Some(alive_lease),
        )
        .expect_err("old-device resources must not cross the boundary");
        assert!(matches!(
            error,
            ParticleFabricTransportError::DeviceGenerationMismatch { .. }
        ));
    }

    #[test]
    fn valid_transport_preserves_the_publication_and_both_real_leases() {
        let stream = BufferId::from_parts(4, 7);
        let alive = BufferId::from_parts(5, 8);
        let publication = publication(31, stream, alive);
        let generation = ParticleFabricDeviceGeneration::new(12);
        let mut issuer = ParticleFabricLeaseIssuer::new(generation);
        let stream_lease = issuer.issue(stream, 31).unwrap();
        let alive_lease = issuer.issue(alive, 31).unwrap();

        let transport = ParticleFabricSceneTransport::try_new(
            publication.clone(),
            generation,
            Some(stream_lease),
            Some(alive_lease),
        )
        .expect("matching native bindings form a valid transport");

        assert_eq!(transport.publication(), &publication);
        assert_eq!(transport.device_generation(), generation);
        assert_eq!(transport.frame(), 31);
        assert_eq!(transport.stream().unwrap().buffer(), stream);
        assert_eq!(transport.alive().unwrap().buffer(), alive);
        assert_ne!(
            transport.stream().unwrap().provenance().lease_id(),
            transport.alive().unwrap().provenance().lease_id()
        );
    }

    #[test]
    fn transport_rejects_a_buffer_id_or_generation_that_does_not_match_publication() {
        let published_stream = BufferId::from_parts(1, 5);
        let published_alive = BufferId::from_parts(2, 6);
        let publication_32 = publication(32, published_stream, published_alive);
        let mut issuer = ParticleFabricLeaseIssuer::new(ParticleFabricDeviceGeneration::new(8));
        let wrong_stream = issuer
            .issue(BufferId::from_parts(99, 5), 32)
            .expect("lease issued");
        let alive = issuer.issue(published_alive, 32).unwrap();

        let error = ParticleFabricSceneTransport::try_new(
            publication_32,
            ParticleFabricDeviceGeneration::new(8),
            Some(wrong_stream),
            Some(alive),
        )
        .expect_err("a different native id must be rejected");
        assert!(matches!(
            error,
            ParticleFabricTransportError::StreamBufferMismatch { .. }
        ));

        let publication_33 = publication(33, published_stream, published_alive);
        let stale_generation = issuer
            .issue(BufferId::from_parts(1, 4), 33)
            .expect("lease issued");
        let alive = issuer.issue(published_alive, 33).unwrap();
        let error = ParticleFabricSceneTransport::try_new(
            publication_33,
            ParticleFabricDeviceGeneration::new(8),
            Some(stale_generation),
            Some(alive),
        )
        .expect_err("a recycled id generation must be rejected");
        assert!(matches!(
            error,
            ParticleFabricTransportError::StreamBufferMismatch { .. }
                | ParticleFabricTransportError::StreamVersionMismatch { .. }
        ));
    }

    #[test]
    fn empty_transport_cannot_retain_stale_physical_bindings() {
        let empty = ParticleFabricScenePublication::new(44, Vec::new(), Vec::new());
        let mut issuer = ParticleFabricLeaseIssuer::new(ParticleFabricDeviceGeneration::new(1));
        let lease = issuer.issue(BufferId::from_parts(4, 2), 44).unwrap();
        let error = ParticleFabricSceneTransport::try_new(
            empty,
            ParticleFabricDeviceGeneration::new(1),
            Some(lease),
            None,
        )
        .expect_err("empty frames must not retain prior physical resources");
        assert_eq!(
            error,
            ParticleFabricTransportError::BindingsOnEmptyPublication
        );
    }

    #[test]
    fn issuer_rejects_frame_regression() {
        let mut issuer = ParticleFabricLeaseIssuer::new(ParticleFabricDeviceGeneration::new(2));
        issuer.issue(BufferId::from_parts(1, 1), 10).unwrap();
        let error = issuer
            .issue(BufferId::from_parts(1, 1), 9)
            .expect_err("producer publication frames must be monotonic");
        assert_eq!(
            error,
            ParticleFabricLeaseError::FrameRegression {
                previous: 10,
                actual: 9,
            }
        );
    }

    #[test]
    fn separate_issuers_cannot_reuse_a_lease_id() {
        let generation = ParticleFabricDeviceGeneration::new(3);
        let mut first = ParticleFabricLeaseIssuer::new(generation);
        let mut second = ParticleFabricLeaseIssuer::new(generation);
        let first_lease = first.issue(BufferId::from_parts(1, 1), 1).unwrap();
        let second_lease = second.issue(BufferId::from_parts(2, 1), 1).unwrap();

        assert_ne!(
            first_lease.provenance().lease_id(),
            second_lease.provenance().lease_id()
        );
    }
}
