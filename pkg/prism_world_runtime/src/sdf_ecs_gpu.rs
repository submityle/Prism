//! ECS-facing bounded GPU SDF sampling.

use std::{sync::mpsc, time::Duration};

use prism_ecs::prelude::{Component, Entity, World};
use prism_world_gpu::{GpuAcquireError, GpuContext, GpuSdfQuery};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};

use crate::{
    WorldSdfClipmapConfig, WorldSdfClipmapScheduler, WorldSdfFocusConfig, WorldSdfGpuClipmapError,
    WorldSdfGpuFocus, WorldSdfGpuFocusError,
};

/// One ECS-owned integer SDF lookup request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfEcsRequest {
    /// Integer world voxel coordinate.
    pub voxel: [i32; 3],
    /// Clipmap mip level.
    pub mip: u8,
    /// Caller-owned version used to reject stale results.
    pub request_id: u64,
}

impl Component for WorldSdfEcsRequest {}

/// ECS result value with explicit absence semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldSdfEcsSampleValue {
    /// Resident quantized signed-distance sample.
    Value(i16),
    /// Requested brick was not resident.
    Unknown,
}

/// Result written to the requesting entity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfEcsSample {
    /// Request version that produced this result.
    pub request_id: u64,
    /// Explicit GPU result.
    pub value: WorldSdfEcsSampleValue,
}

impl Component for WorldSdfEcsSample {}

/// Fixed construction parameters for the ECS/GPU bridge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfEcsGpuConfig {
    /// World-mm focus mapping.
    pub focus: WorldSdfFocusConfig,
    /// Runtime clipmap and dirty publication budgets.
    pub clipmap: WorldSdfClipmapConfig,
    /// Maximum requests in one GPU batch.
    pub max_queries: usize,
    /// Maximum physical upload bytes per clipmap call.
    pub max_upload_bytes: u64,
}

/// Explicit policy for unavailable GPU hardware.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldSdfEcsGpuPolicy {
    /// Return None when hardware is unavailable.
    SkipUnavailable,
    /// Return the acquisition error.
    Strict,
}

/// Errors that fail the batch before result publication.
#[derive(Debug, PartialEq)]
pub enum WorldSdfEcsGpuError {
    /// Hardware acquisition failed in strict mode.
    Acquire(GpuAcquireError),
    /// Strict mode found no physical adapter.
    NoPhysicalGpu,
    /// Zero query or upload capacity.
    InvalidConfiguration,
    /// Request count exceeds the fixed capacity.
    RequestCapacityExceeded {
        /// Number of requests discovered.
        required: usize,
        /// Maximum requests accepted by the bridge.
        capacity: usize,
    },
    /// Request mip is outside the fixed ring geometry.
    InvalidRequestMip {
        /// Position in the deterministic entity-sorted batch.
        index: usize,
        /// Invalid clipmap mip level.
        mip: u8,
    },
    /// Request entity is no longer live.
    EntityUnavailable(Entity),
    /// Runtime focus/upload/dispatch failed.
    Runtime(WorldSdfGpuFocusError),
    /// Query submission or GPU upload failed.
    Submit(WorldSdfGpuClipmapError),
    /// Device polling failed.
    ReadbackPoll,
    /// Mapping callback did not arrive.
    ReadbackCallback,
    /// Result mapping failed.
    ReadbackMap,
    /// Result ABI was invalid.
    InvalidGpuRecord {
        /// Position of the malformed 16-byte ABI record.
        index: usize,
    },
    /// Readback size overflowed.
    ArithmeticOverflow,
}

impl core::fmt::Display for WorldSdfEcsGpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Acquire(e) => write!(f, "SDF ECS GPU acquisition failed: {e}"),
            Self::NoPhysicalGpu => f.write_str("SDF ECS GPU requires a physical GPU"),
            Self::InvalidConfiguration => f.write_str("invalid SDF ECS GPU configuration"),
            Self::RequestCapacityExceeded { required, capacity } => write!(
                f,
                "SDF ECS request batch {required} exceeds capacity {capacity}"
            ),
            Self::InvalidRequestMip { index, mip } => {
                write!(f, "SDF ECS request {index} has invalid mip {mip}")
            }
            Self::EntityUnavailable(entity) => write!(f, "SDF ECS entity unavailable: {entity:?}"),
            Self::Runtime(e) => write!(f, "SDF ECS runtime GPU path: {e}"),
            Self::Submit(e) => write!(f, "SDF ECS GPU submission: {e}"),
            Self::ReadbackPoll => f.write_str("SDF ECS GPU readback poll failed"),
            Self::ReadbackCallback => f.write_str("SDF ECS GPU readback callback failed"),
            Self::ReadbackMap => f.write_str("SDF ECS GPU result mapping failed"),
            Self::InvalidGpuRecord { index } => {
                write!(f, "SDF ECS GPU result record {index} is invalid")
            }
            Self::ArithmeticOverflow => f.write_str("SDF ECS GPU readback size overflowed"),
        }
    }
}

impl std::error::Error for WorldSdfEcsGpuError {}

/// Summary of one successful ECS/GPU transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfEcsSampleReport {
    /// Number of request entities dispatched and updated.
    pub sampled_entities: usize,
}

/// Owns the runtime focus/ring and performs bounded ECS sampling.
pub struct WorldSdfEcsGpuBridge {
    context: GpuContext,
    focus: WorldSdfGpuFocus,
    max_queries: usize,
}

impl WorldSdfEcsGpuBridge {
    /// Builds a bridge from an acquired physical GPU context.
    pub fn new(
        context: GpuContext,
        config: WorldSdfEcsGpuConfig,
    ) -> Result<Self, WorldSdfEcsGpuError> {
        validate_config(config)?;
        let focus = WorldSdfGpuFocus::new(
            &context,
            config.focus,
            config.clipmap,
            config.max_queries,
            config.max_upload_bytes,
        )
        .map_err(WorldSdfEcsGpuError::Runtime)?;
        Ok(Self {
            context,
            focus,
            max_queries: config.max_queries,
        })
    }

    /// Acquires a physical GPU under an explicit skip or strict policy.
    pub fn from_headless(
        policy: WorldSdfEcsGpuPolicy,
        required_limits: wgpu::Limits,
        config: WorldSdfEcsGpuConfig,
    ) -> Result<Option<Self>, WorldSdfEcsGpuError> {
        validate_config(config)?;
        match GpuContext::try_headless_with_limits(required_limits) {
            Ok(Some(context)) => Self::new(context, config).map(Some),
            Ok(None) => match policy {
                WorldSdfEcsGpuPolicy::SkipUnavailable => Ok(None),
                WorldSdfEcsGpuPolicy::Strict => Err(WorldSdfEcsGpuError::NoPhysicalGpu),
            },
            Err(GpuAcquireError::NonHardware(_)) => match policy {
                WorldSdfEcsGpuPolicy::SkipUnavailable => Ok(None),
                WorldSdfEcsGpuPolicy::Strict => Err(WorldSdfEcsGpuError::NoPhysicalGpu),
            },
            Err(GpuAcquireError::Adapter(_)) => match policy {
                WorldSdfEcsGpuPolicy::SkipUnavailable => Ok(None),
                WorldSdfEcsGpuPolicy::Strict => Err(WorldSdfEcsGpuError::NoPhysicalGpu),
            },
            Err(error) => Err(WorldSdfEcsGpuError::Acquire(error)),
        }
    }

    /// Returns the paired runtime focus owner.
    pub fn focus(&self) -> &WorldSdfGpuFocus {
        &self.focus
    }

    /// Updates world-mm focus and publishes entering SDF bricks.
    pub fn update_focus(
        &mut self,
        grid: &SparseSdfBrickGrid,
        position_mm: [i64; 3],
    ) -> Result<crate::WorldSdfFocusUpdate<crate::WorldSdfClipmapUploadStats>, WorldSdfEcsGpuError>
    {
        self.focus
            .update_focus(grid, position_mm)
            .map_err(WorldSdfEcsGpuError::Runtime)
    }

    /// Forwards resident-brick edits to the paired GPU dirty fence.
    pub fn mark_dirty(&mut self, coords: &[BrickCoord]) -> Result<usize, WorldSdfEcsGpuError> {
        self.focus
            .mark_dirty(coords)
            .map_err(WorldSdfEcsGpuError::Submit)
    }

    /// Uploads a bounded dirty suffix before queries become consumable.
    pub fn flush_dirty(
        &mut self,
        grid: &SparseSdfBrickGrid,
        max_bricks: usize,
        max_bytes: u64,
    ) -> Result<crate::WorldSdfClipmapUploadStats, WorldSdfEcsGpuError> {
        self.focus
            .flush_dirty(grid, max_bricks, max_bytes)
            .map_err(WorldSdfEcsGpuError::Submit)
    }

    /// Dispatches ECS requests, reads GPU results, then publishes all results.
    pub fn sample_world(
        &mut self,
        world: &mut World,
    ) -> Result<WorldSdfEcsSampleReport, WorldSdfEcsGpuError> {
        let requests = collect_requests(
            world,
            self.max_queries,
            self.focus.ring().scheduler().config().clipmap.mip_levels,
        )?;
        if requests.is_empty() {
            return Ok(WorldSdfEcsSampleReport {
                sampled_entities: 0,
            });
        }
        let bytes = requests
            .len()
            .checked_mul(16)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorldSdfEcsGpuError::ArithmeticOverflow)?;
        let staging = self
            .context
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("world SDF ECS result readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        let queries: Vec<_> = requests
            .iter()
            .map(|(_, request)| GpuSdfQuery {
                voxel: request.voxel,
                mip: request.mip,
            })
            .collect();
        let encoder =
            self.context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("world SDF ECS dispatch"),
                });
        self.focus
            .submit_queries(&queries, encoder, |results, encoder| {
                encoder.copy_buffer_to_buffer(results, 0, &staging, 0, bytes)
            })
            .map_err(WorldSdfEcsGpuError::Submit)?;
        let values = readback_results(&self.context, staging, requests.len())?;
        let samples: Vec<_> = requests
            .iter()
            .zip(values)
            .map(|((_, request), value)| WorldSdfEcsSample {
                request_id: request.request_id,
                value,
            })
            .collect();
        for ((entity, _), sample) in requests.into_iter().zip(samples) {
            if !world.insert(entity, sample) {
                return Err(WorldSdfEcsGpuError::EntityUnavailable(entity));
            }
        }
        Ok(WorldSdfEcsSampleReport {
            sampled_entities: queries.len(),
        })
    }
}

fn collect_requests(
    world: &mut World,
    capacity: usize,
    mip_levels: u8,
) -> Result<Vec<(Entity, WorldSdfEcsRequest)>, WorldSdfEcsGpuError> {
    let query = world.query::<(Entity, &WorldSdfEcsRequest)>();
    let mut requests = Vec::new();
    for (entity, request) in query.iter(world) {
        if requests.len() == capacity {
            return Err(WorldSdfEcsGpuError::RequestCapacityExceeded {
                required: capacity + 1,
                capacity,
            });
        }
        requests.push((entity, *request));
    }
    requests.sort_by_key(|(entity, _)| *entity);
    for (index, (_, request)) in requests.iter().enumerate() {
        if request.mip >= mip_levels {
            return Err(WorldSdfEcsGpuError::InvalidRequestMip {
                index,
                mip: request.mip,
            });
        }
    }
    Ok(requests)
}

fn validate_config(config: WorldSdfEcsGpuConfig) -> Result<(), WorldSdfEcsGpuError> {
    if config.max_queries == 0 || config.max_upload_bytes == 0 {
        return Err(WorldSdfEcsGpuError::InvalidConfiguration);
    }
    config
        .focus
        .validate_single_slice_coverage(config.clipmap.clipmap.ring_radius)
        .map_err(|error| WorldSdfEcsGpuError::Runtime(WorldSdfGpuFocusError::Focus(error)))?;
    WorldSdfClipmapScheduler::new(config.clipmap).map_err(|error| {
        WorldSdfEcsGpuError::Runtime(WorldSdfGpuFocusError::Apply(
            WorldSdfGpuClipmapError::Scheduler(error),
        ))
    })?;
    Ok(())
}

fn readback_results(
    context: &GpuContext,
    staging: wgpu::Buffer,
    count: usize,
) -> Result<Vec<WorldSdfEcsSampleValue>, WorldSdfEcsGpuError> {
    let (sender, receiver) = mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    let result = context.device().poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(Duration::from_secs(30)),
    });
    if result.is_err() {
        staging.unmap();
        return Err(WorldSdfEcsGpuError::ReadbackPoll);
    }
    let callback = receiver.recv_timeout(Duration::from_secs(30));
    if !matches!(callback, Ok(Ok(()))) {
        staging.unmap();
        return Err(match callback {
            Err(_) => WorldSdfEcsGpuError::ReadbackCallback,
            Ok(Err(_)) => WorldSdfEcsGpuError::ReadbackMap,
            Ok(Ok(())) => unreachable!(),
        });
    }
    let mapped = match staging.get_mapped_range(..) {
        Ok(mapped) => mapped,
        Err(_) => {
            staging.unmap();
            return Err(WorldSdfEcsGpuError::ReadbackMap);
        }
    };
    let result = decode_gpu_records(&mapped, count);
    drop(mapped);
    staging.unmap();
    result
}

fn decode_gpu_records(
    bytes: &[u8],
    count: usize,
) -> Result<Vec<WorldSdfEcsSampleValue>, WorldSdfEcsGpuError> {
    let expected = count
        .checked_mul(16)
        .ok_or(WorldSdfEcsGpuError::ArithmeticOverflow)?;
    if bytes.len() != expected || !bytes.chunks_exact(16).remainder().is_empty() {
        return Err(WorldSdfEcsGpuError::InvalidGpuRecord { index: count });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| WorldSdfEcsGpuError::ArithmeticOverflow)?;
    for (index, record) in bytes.chunks_exact(16).enumerate() {
        if record[8..16] != [0; 8] {
            return Err(WorldSdfEcsGpuError::InvalidGpuRecord { index });
        }
        let status = u32::from_le_bytes(record[0..4].try_into().unwrap());
        let value = i32::from_le_bytes(record[4..8].try_into().unwrap());
        values.push(match status {
            0 if value == 0 => WorldSdfEcsSampleValue::Unknown,
            1 if (i16::MIN as i32..=i16::MAX as i32).contains(&value) => {
                WorldSdfEcsSampleValue::Value(value as i16)
            }
            _ => return Err(WorldSdfEcsGpuError::InvalidGpuRecord { index }),
        });
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(status: u32, value: i32, reserved: [u8; 8]) -> [u8; 16] {
        let mut bytes = [0; 16];
        bytes[0..4].copy_from_slice(&status.to_le_bytes());
        bytes[4..8].copy_from_slice(&value.to_le_bytes());
        bytes[8..16].copy_from_slice(&reserved);
        bytes
    }

    #[test]
    fn requests_are_collected_in_entity_order() {
        let mut world = World::new();
        let first = world.spawn(WorldSdfEcsRequest {
            voxel: [2, 0, 0],
            mip: 0,
            request_id: 2,
        });
        let second = world.spawn(WorldSdfEcsRequest {
            voxel: [1, 0, 0],
            mip: 0,
            request_id: 1,
        });
        let requests = collect_requests(&mut world, 2, 1).unwrap();
        assert_eq!(requests[0].0, first);
        assert_eq!(requests[1].0, second);
    }

    #[test]
    fn capacity_and_mip_are_rejected_before_dispatch() {
        let mut world = World::new();
        world.spawn(WorldSdfEcsRequest {
            voxel: [0, 0, 0],
            mip: 3,
            request_id: 7,
        });
        assert_eq!(
            collect_requests(&mut world, 1, 2),
            Err(WorldSdfEcsGpuError::InvalidRequestMip { index: 0, mip: 3 })
        );
        let mut world = World::new();
        world.spawn(WorldSdfEcsRequest {
            voxel: [0, 0, 0],
            mip: 0,
            request_id: 1,
        });
        world.spawn(WorldSdfEcsRequest {
            voxel: [1, 0, 0],
            mip: 0,
            request_id: 2,
        });
        assert_eq!(
            collect_requests(&mut world, 1, 1),
            Err(WorldSdfEcsGpuError::RequestCapacityExceeded {
                required: 2,
                capacity: 1
            })
        );
    }

    #[test]
    fn gpu_abi_decodes_extremes_and_unknown_but_rejects_malformed_records() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&record(1, i16::MIN as i32, [0; 8]));
        bytes.extend_from_slice(&record(1, i16::MAX as i32, [0; 8]));
        bytes.extend_from_slice(&record(0, 0, [0; 8]));
        assert_eq!(
            decode_gpu_records(&bytes, 3).unwrap(),
            vec![
                WorldSdfEcsSampleValue::Value(i16::MIN),
                WorldSdfEcsSampleValue::Value(i16::MAX),
                WorldSdfEcsSampleValue::Unknown,
            ]
        );

        let mut bad_status = record(2, 0, [0; 8]);
        assert!(matches!(
            decode_gpu_records(&bad_status, 1),
            Err(WorldSdfEcsGpuError::InvalidGpuRecord { index: 0 })
        ));
        bad_status = record(1, i16::MAX as i32 + 1, [0; 8]);
        assert!(matches!(
            decode_gpu_records(&bad_status, 1),
            Err(WorldSdfEcsGpuError::InvalidGpuRecord { index: 0 })
        ));
        assert!(matches!(
            decode_gpu_records(&record(0, 1, [0; 8]), 1),
            Err(WorldSdfEcsGpuError::InvalidGpuRecord { index: 0 })
        ));
        assert!(matches!(
            decode_gpu_records(&record(1, 1, [1; 8]), 1),
            Err(WorldSdfEcsGpuError::InvalidGpuRecord { index: 0 })
        ));
        assert!(matches!(
            decode_gpu_records(&bytes, 2),
            Err(WorldSdfEcsGpuError::InvalidGpuRecord { index: 2 })
        ));
    }

    #[test]
    fn empty_and_invalid_configurations_fail_before_gpu_creation() {
        let base = WorldSdfEcsGpuConfig {
            focus: WorldSdfFocusConfig::new([0; 3], 1, 1, [0; 3]).unwrap(),
            clipmap: WorldSdfClipmapConfig::new(
                prism_world_residency::SdfClipmapConfig::new(1, 1, 1).unwrap(),
                9,
                1 << 20,
            )
            .unwrap(),
            max_queries: 1,
            max_upload_bytes: 1 << 20,
        };
        assert_eq!(validate_config(base), Ok(()));
        assert_eq!(
            validate_config(WorldSdfEcsGpuConfig {
                max_queries: 0,
                ..base
            }),
            Err(WorldSdfEcsGpuError::InvalidConfiguration)
        );
        assert_eq!(
            validate_config(WorldSdfEcsGpuConfig {
                max_upload_bytes: 0,
                ..base
            }),
            Err(WorldSdfEcsGpuError::InvalidConfiguration)
        );
    }
}
