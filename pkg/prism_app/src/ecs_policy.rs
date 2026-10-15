//! Opt-in execution of explicit, project-independent ECS profiles.
//!
//! [`App::configure_ecs_policy`] binds a policy to the main World and replaces
//! its schedule/QoS pool. It never probes hardware or changes the App's separate
//! capability, quality-tier or run-mode resources. Secondary worlds, render
//! pipelines and externally held task pools have independent budgets.
//!
//! Entity limits apply to explicit admission calls, not arbitrary World or
//! Commands operations. Under `partition`, the host supplies positions and frame
//! numbers to real LOD callbacks and streaming eviction; no transform type,
//! gameplay system, disk loader or automatic frame/viewpoint selection is assumed.

use core::fmt;

use prism_ecs::bundle::Bundle;
use prism_ecs::entity::Entity;
use prism_ecs::policy::{
    AdmissionError, EcsCapabilities, EcsFeature, EcsPolicy, EcsProfile, PolicyError,
};
use prism_ecs::world::WorldId;

use crate::App;

#[cfg(feature = "partition")]
mod partition;
#[cfg(feature = "partition")]
pub use partition::{EcsCellLoad, EcsStreamingResult};

/// A rejected configuration or explicit runtime operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EcsPolicyError {
    /// Invalid profile or insufficient explicit host capabilities.
    Policy(PolicyError),
    /// Entity budget or outstanding reservations prevent admission.
    Admission(AdmissionError),
    /// The App adapter cannot execute this facility, even if ECS feature
    /// unification enabled it in another dependency.
    FeatureUnavailable(EcsFeature),
    /// No policy has been applied to this App.
    NotConfigured,
    /// The main World was replaced after this runtime was configured.
    WorldChanged,
    /// The selected task backend cannot satisfy the worker minimum.
    BackendWorkers {
        /// Required total workers, including the caller.
        required: usize,
        /// Actual total workers, including the caller.
        actual: usize,
    },
    /// The configured profile has no LOD schedule.
    LodDisabled,
    /// The configured profile has no streaming grid.
    StreamingDisabled,
    /// Replacing/disabling a nonempty streaming grid would abandon residency
    /// or pending loads. Drain its interests first.
    StreamingBusy,
    /// Nonfinite coordinates, overflowing LOD distance, or streaming coordinates
    /// outside the adapter's checked integer-grid range.
    InvalidPosition,
    /// A supplied population/settlement handle is not live in the main World.
    EntityNotLive(Entity),
    /// A population/settlement contains a duplicate entity.
    DuplicateEntity(Entity),
    /// An asynchronous cell completion belongs to a different World, an evicted
    /// request, or an already completed request.
    InvalidLoad,
    /// A settlement would silently move an entity owned by another cell.
    EntityAlreadyStreamed(Entity),
}

impl fmt::Display for EcsPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(error) => write!(f, "ECS profile: {error}"),
            Self::Admission(error) => write!(f, "ECS admission: {error}"),
            Self::FeatureUnavailable(feature) => write!(f, "App cannot execute {feature:?}"),
            Self::NotConfigured => f.write_str("no ECS policy has been applied"),
            Self::WorldChanged => f.write_str("ECS policy is bound to a different main World"),
            Self::BackendWorkers { required, actual } => {
                write!(f, "ECS backend provides {actual} workers, needs {required}")
            }
            Self::LodDisabled => f.write_str("ECS policy does not enable LOD"),
            Self::StreamingDisabled => f.write_str("ECS policy does not enable streaming"),
            Self::StreamingBusy => f.write_str("cannot replace a nonempty ECS streaming grid"),
            Self::InvalidPosition => {
                f.write_str("position exceeds the checked ECS coordinate range")
            }
            Self::EntityNotLive(entity) => {
                write!(f, "entity {entity:?} is not live in the main World")
            }
            Self::DuplicateEntity(entity) => write!(f, "duplicate entity {entity:?}"),
            Self::InvalidLoad => f.write_str("stale or foreign ECS cell load completion"),
            Self::EntityAlreadyStreamed(entity) => {
                write!(f, "entity {entity:?} belongs to another cell")
            }
        }
    }
}

impl core::error::Error for EcsPolicyError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Policy(error) => Some(error),
            Self::Admission(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PolicyError> for EcsPolicyError {
    fn from(error: PolicyError) -> Self {
        Self::Policy(error)
    }
}

impl From<AdmissionError> for EcsPolicyError {
    fn from(error: AdmissionError) -> Self {
        Self::Admission(error)
    }
}

pub(crate) struct AppliedEcsPolicy {
    policy: EcsPolicy,
    world: WorldId,
    #[cfg(feature = "partition")]
    partition: partition::PolicyPartition,
}

impl App {
    /// Resolve explicit host/profile inputs and apply them to the main World.
    ///
    /// This does not consult `Capabilities::detect` or a quality-tier heuristic.
    pub fn configure_ecs_policy(
        &mut self,
        profile: EcsProfile,
        capabilities: EcsCapabilities,
    ) -> Result<&mut Self, EcsPolicyError> {
        self.apply_ecs_policy(EcsPolicy::resolve(profile, capabilities)?)
    }

    /// Apply a resolved policy at a host-controlled quiescent point.
    ///
    /// All returned validation errors leave the existing policy, pool, queued
    /// QoS jobs and partition state untouched. With `multi_thread`, this creates
    /// a real pool with at most `worker_limit() - 1` background workers, reserving
    /// one worker for the caller. Affinity is off; custom affinity is available
    /// through `apply_ecs_policy_with_pool_config`. Backend-enforced serial mode
    /// is accepted only if it satisfies the profile's minimum.
    ///
    /// Unchanged streaming configurations retain residency and load tickets.
    /// Replacing the main World explicitly starts fresh state on reapplication.
    /// Existing clones of an old pool are not retired or cancelled: the host
    /// must quiesce independent producers/executors before reconfiguration.
    /// An explicit later `set_task_pool` call overrides this pool configuration.
    ///
    /// # Panics
    /// Allocation or OS thread creation may panic; this is not a fallible
    /// allocator/thread-start API. No claim of rollback from those panics is made.
    pub fn apply_ecs_policy(&mut self, policy: EcsPolicy) -> Result<&mut Self, EcsPolicyError> {
        #[cfg(feature = "multi_thread")]
        {
            // Do not call TaskPoolConfig::default: its worker count probes the host.
            self.apply_ecs_policy_with_pool_config(
                policy,
                prism_tasks::TaskPoolConfig {
                    threads: 0,
                    pin_workers: false,
                    core_class_policy: prism_tasks::CoreClassPolicy::PerformanceFirst,
                },
            )
        }
        #[cfg(not(feature = "multi_thread"))]
        {
            self.validate_ecs_policy(&policy)?;
            self.install_ecs_policy(policy);
            Ok(self)
        }
    }

    /// Apply a policy using explicit task affinity preferences.
    ///
    /// The policy replaces `config.threads` (background workers), preserving
    /// affinity fields. Pending QoS jobs and cloned queue handles remain live
    /// through the App's existing pool-replacement path.
    #[cfg(feature = "multi_thread")]
    pub fn apply_ecs_policy_with_pool_config(
        &mut self,
        policy: EcsPolicy,
        config: prism_tasks::TaskPoolConfig,
    ) -> Result<&mut Self, EcsPolicyError> {
        self.validate_ecs_policy(&policy)?;
        let pool = prism_tasks::TaskPool::with_config(policy.task_pool_config(config));
        let actual = pool.worker_count() + 1;
        let required = policy.profile().requirements.min_workers;
        if actual < required {
            return Err(EcsPolicyError::BackendWorkers { required, actual });
        }
        self.set_task_pool(pool);
        self.install_ecs_policy(policy);
        Ok(self)
    }

    fn validate_ecs_policy(&self, policy: &EcsPolicy) -> Result<(), EcsPolicyError> {
        #[cfg(not(feature = "multi_thread"))]
        if policy.worker_limit() > 1 {
            return Err(EcsPolicyError::FeatureUnavailable(EcsFeature::MultiThread));
        }
        #[cfg(not(feature = "partition"))]
        if policy.profile().lod.is_some() || policy.profile().streaming.is_some() {
            return Err(EcsPolicyError::FeatureUnavailable(EcsFeature::Partition));
        }
        policy.check_world_admission(self.world(), 0)?;
        #[cfg(feature = "partition")]
        partition::validate_reconfiguration(
            self.ecs_policy
                .as_ref()
                .filter(|p| p.world == self.world().id()),
            policy,
        )?;
        Ok(())
    }

    fn install_ecs_policy(&mut self, policy: EcsPolicy) {
        #[cfg(feature = "partition")]
        let partition = partition::PolicyPartition::reconfigure(
            self.ecs_policy
                .take()
                .filter(|p| p.world == self.world().id()),
            &policy,
        );
        // A readable resource for systems. The App-owned policy remains the
        // admission authority; manually overwriting this copy is not reconfiguration.
        self.world_mut().insert_resource(policy.clone());
        self.ecs_policy = Some(AppliedEcsPolicy {
            policy,
            world: self.world().id(),
            #[cfg(feature = "partition")]
            partition,
        });
    }

    /// The policy applied to the current main World, if any.
    ///
    /// The World resource is a read-only-by-convention copy for systems. To
    /// reconfigure execution, use the App methods, not resource replacement.
    pub fn ecs_policy(&self) -> Option<&EcsPolicy> {
        self.ecs_policy
            .as_ref()
            .filter(|p| p.world == self.world().id())
            .map(|p| &p.policy)
    }

    fn applied_ecs_policy(&self) -> Result<&AppliedEcsPolicy, EcsPolicyError> {
        let applied = self
            .ecs_policy
            .as_ref()
            .ok_or(EcsPolicyError::NotConfigured)?;
        if applied.world != self.world().id() {
            return Err(EcsPolicyError::WorldChanged);
        }
        Ok(applied)
    }

    /// Check an explicitly accounted allocation batch at a synchronization point.
    ///
    /// This returns no permit and reserves no slots. Outstanding reservations
    /// are rejected, never flushed implicitly. Raw World/Commands spawns and
    /// allocations made by hooks/observers bypass this admission path; the host
    /// must account for those separately. Limits are counts, not byte quotas.
    pub fn check_entity_admission(&self, additional: u32) -> Result<(), EcsPolicyError> {
        self.applied_ecs_policy()?
            .policy
            .check_world_admission(self.world(), additional)?;
        Ok(())
    }

    /// Check and immediately perform one direct spawn in the main World.
    ///
    /// The exclusive App borrow closes the check/spawn gap for this direct
    /// allocation only. Hook/observer side effects are neither pre-budgeted nor
    /// rolled back; use an explicitly accounted batch for those workflows.
    pub fn try_spawn_with_policy<B: Bundle>(
        &mut self,
        bundle: B,
    ) -> Result<Entity, EcsPolicyError> {
        self.check_entity_admission(1)?;
        Ok(self.world_mut().spawn(bundle))
    }
}

#[cfg(test)]
mod tests;
