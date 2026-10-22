//! Fixed-tick, single-World execution for externally owned authority commands.
//!
//! This bridge deliberately leaves command schema and gameplay policy to the
//! application. Its handler must validate before mutating the World. Returning
//! `Retry` promises that no externally visible mutation was committed; returning
//! `Applied` or `Rejected` is terminal and acknowledges the reliable command.

use core::marker::PhantomData;

use prism_app::{FixedPreUpdate, Plugin};
use prism_ecs::resource::Resource;
use prism_ecs::world::World;

use super::{LeasedAuthorityCommand, MultiPeerAuthority, ServerStrategy};

/// A command handler's transaction decision for the current fixed tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorityCommandDecision {
    /// The handler committed the World mutation; acknowledge and remove the RPC.
    Applied,
    /// The handler permanently rejected the command without applying it.
    Rejected,
    /// No mutation was committed; retain the same lease and retry next tick.
    Retry,
}

/// Invalid configuration for the fixed-tick authority executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthorityWorldExecutorConfigError;

/// Bounded executor diagnostics. Command ownership remains in the server peer
/// queue until a terminal handler decision is acknowledged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorityWorldExecutorStats {
    /// Commands committed to the World.
    pub applied: u64,
    /// Commands explicitly rejected by application policy.
    pub rejected: u64,
    /// Commands currently leased and awaiting a terminal decision.
    pub pending: bool,
    /// Number of lease/acknowledgement errors observed.
    pub authority_errors: u64,
    /// The handler panicked; the command remains leased and execution is
    /// stopped to avoid replaying a possibly partially applied mutation.
    pub poisoned: bool,
}

/// Plugin that executes one leased authority command at a time in the same
/// exclusive `&mut World` fixed-tick phase for all peers.
pub struct AuthorityWorldExecutorPlugin<S, H>
where
    S: ServerStrategy,
    H: FnMut(&mut World, &LeasedAuthorityCommand) -> AuthorityCommandDecision
        + Send
        + Sync
        + 'static,
{
    handler: std::sync::Mutex<Option<H>>,
    max_commands_per_tick: usize,
    marker: PhantomData<fn() -> S>,
}

impl<S, H> AuthorityWorldExecutorPlugin<S, H>
where
    S: ServerStrategy,
    H: FnMut(&mut World, &LeasedAuthorityCommand) -> AuthorityCommandDecision
        + Send
        + Sync
        + 'static,
{
    /// Create a fixed-tick authority executor using an application-owned,
    /// project-specific command handler.
    #[must_use]
    pub const fn new(handler: H) -> Self {
        Self {
            handler: std::sync::Mutex::new(Some(handler)),
            max_commands_per_tick: 1,
            marker: PhantomData,
        }
    }

    /// Set the bounded number of terminal commands this executor may process
    /// during one fixed tick. A retry, ack error, or panic always stops the
    /// current tick before another command is leased.
    pub fn try_with_max_commands_per_tick(
        mut self,
        max_commands_per_tick: usize,
    ) -> Result<Self, AuthorityWorldExecutorConfigError> {
        if max_commands_per_tick == 0 {
            return Err(AuthorityWorldExecutorConfigError);
        }
        self.max_commands_per_tick = max_commands_per_tick;
        Ok(self)
    }
}

struct AuthorityWorldExecutor<S, H>
where
    S: ServerStrategy,
    H: FnMut(&mut World, &LeasedAuthorityCommand) -> AuthorityCommandDecision
        + Send
        + Sync
        + 'static,
{
    handler: H,
    pending: Option<LeasedAuthorityCommand>,
    terminal_decision: Option<AuthorityCommandDecision>,
    poisoned: bool,
    max_commands_per_tick: usize,
    stats: AuthorityWorldExecutorStats,
    marker: PhantomData<fn() -> S>,
}

impl<S, H> Resource for AuthorityWorldExecutor<S, H>
where
    S: ServerStrategy,
    H: FnMut(&mut World, &LeasedAuthorityCommand) -> AuthorityCommandDecision
        + Send
        + Sync
        + 'static,
{
}

impl<S, H> Plugin for AuthorityWorldExecutorPlugin<S, H>
where
    S: ServerStrategy,
    H: FnMut(&mut World, &LeasedAuthorityCommand) -> AuthorityCommandDecision
        + Send
        + Sync
        + 'static,
{
    fn build(&self, app: &mut prism_app::App) {
        let handler = self
            .handler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("AuthorityWorldExecutorPlugin::build ran twice");
        app.insert_resource(AuthorityWorldExecutor::<S, H> {
            handler,
            pending: None,
            terminal_decision: None,
            poisoned: false,
            max_commands_per_tick: self.max_commands_per_tick,
            stats: AuthorityWorldExecutorStats::default(),
            marker: PhantomData,
        });
        app.insert_resource(AuthorityWorldExecutorStatus::<S>::default());
        app.add_systems(FixedPreUpdate, execute_authority_world::<S, H>);
    }
}

fn execute_authority_world<S, H>(world: &mut World)
where
    S: ServerStrategy,
    H: FnMut(&mut World, &LeasedAuthorityCommand) -> AuthorityCommandDecision
        + Send
        + Sync
        + 'static,
{
    let Some(mut authority) = world.remove_resource::<MultiPeerAuthority<S>>() else {
        return;
    };
    let Some(mut executor) = world.remove_resource::<AuthorityWorldExecutor<S, H>>() else {
        world.insert_resource(authority);
        return;
    };
    let Some(mut status) = world.remove_resource::<AuthorityWorldExecutorStatus<S>>() else {
        world.insert_resource(authority);
        world.insert_resource(executor);
        return;
    };

    let mut processed = 0;
    let mut handler_panic = None;
    while !executor.poisoned && processed < executor.max_commands_per_tick {
        if executor.pending.is_none() {
            match authority.lease_inbound_commands(1) {
                Ok(mut commands) => executor.pending = commands.pop(),
                Err(_) => {
                    executor.stats.authority_errors =
                        executor.stats.authority_errors.saturating_add(1);
                    break;
                }
            }
        }
        let Some(command) = executor.pending.as_ref() else {
            break;
        };
        let decision = match executor.terminal_decision {
            Some(decision) => Some(decision),
            None => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (executor.handler)(world, command)
            })) {
                Ok(decision) => Some(decision),
                Err(payload) => {
                    executor.poisoned = true;
                    executor.stats.poisoned = true;
                    handler_panic = Some(payload);
                    None
                }
            },
        };
        let Some(decision) = decision else {
            executor.stats.pending = true;
            status.stats = executor.stats;
            world.insert_resource(authority);
            world.insert_resource(executor);
            world.insert_resource(status);
            if let Some(payload) = handler_panic {
                std::panic::resume_unwind(payload);
            }
            return;
        };
        match decision {
            AuthorityCommandDecision::Applied => {
                executor.terminal_decision = Some(AuthorityCommandDecision::Applied);
                if authority.acknowledge_inbound_command(command).is_ok() {
                    executor.stats.applied = executor.stats.applied.saturating_add(1);
                    executor.pending = None;
                    executor.terminal_decision = None;
                    processed += 1;
                } else {
                    executor.stats.authority_errors =
                        executor.stats.authority_errors.saturating_add(1);
                }
            }
            AuthorityCommandDecision::Rejected => {
                executor.terminal_decision = Some(AuthorityCommandDecision::Rejected);
                if authority.acknowledge_inbound_command(command).is_ok() {
                    executor.stats.rejected = executor.stats.rejected.saturating_add(1);
                    executor.pending = None;
                    executor.terminal_decision = None;
                    processed += 1;
                } else {
                    executor.stats.authority_errors =
                        executor.stats.authority_errors.saturating_add(1);
                }
            }
            AuthorityCommandDecision::Retry => {
                executor.terminal_decision = None;
                break;
            }
        }
    }
    executor.stats.pending = executor.pending.is_some();
    status.stats = executor.stats;
    world.insert_resource(authority);
    world.insert_resource(executor);
    world.insert_resource(status);
}

/// Read-only executor diagnostics resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuthorityWorldExecutorStatus<S: ServerStrategy> {
    /// Current cumulative counters.
    pub stats: AuthorityWorldExecutorStats,
    marker: PhantomData<fn() -> S>,
}

impl<S: ServerStrategy> Default for AuthorityWorldExecutorStatus<S> {
    fn default() -> Self {
        Self {
            stats: AuthorityWorldExecutorStats::default(),
            marker: PhantomData,
        }
    }
}

impl<S: ServerStrategy> Resource for AuthorityWorldExecutorStatus<S> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authority_aoi::AuthorityAoiConfig;
    use crate::delta::{encode_snapshot, FieldKind, FieldValue, Record, Schema};
    use crate::plugin::{
        AuthorityReplicationProvider, AuthorityReplicationProviderError, MultiPeerAuthorityConfig,
        Replication, ReplicationStrategy,
    };
    use crate::transport::memory::InMemoryTransport;
    use crate::transport::Transport;
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Default)]
    struct AppliedCount(usize);
    impl Resource for AppliedCount {}

    #[derive(Default)]
    struct EmptyProvider;
    impl AuthorityReplicationProvider for EmptyProvider {
        fn snapshot_entity(
            &self,
            _peer_id: u64,
            _entity_id: u32,
            _aoi_revision: u64,
        ) -> Result<Vec<u8>, AuthorityReplicationProviderError> {
            Err(AuthorityReplicationProviderError::EntityUnavailable)
        }
    }

    struct AddAuthority;
    impl ServerStrategy for AddAuthority {
        type Command = ();
        fn decode(&self, _bytes: &[u8]) -> Option<()> {
            Some(())
        }
        fn apply(&self, _record: &mut Record, _command: ()) {}
    }

    struct ClientStrategy;
    impl ReplicationStrategy for ClientStrategy {
        type Input = ();
        fn simulate(&self, _record: &mut Record, _input: &()) {}
        fn encode(&self, _input: &()) -> Vec<u8> {
            Vec::new()
        }
    }

    fn schema_and_seed() -> (Schema, Vec<u8>) {
        let schema = Schema::new(vec![FieldKind::Int { bits: 16 }]);
        let mut seed = Vec::new();
        encode_snapshot(
            &schema,
            0,
            &Record::new(vec![FieldValue::Int(0)]),
            &mut seed,
        )
        .unwrap();
        (schema, seed)
    }

    #[test]
    fn fixed_tick_executor_retries_without_ack_then_commits_once_in_shared_world() {
        let config = MultiPeerAuthorityConfig {
            aoi: AuthorityAoiConfig {
                max_peers: 2,
                max_entities: 8,
                max_entities_per_peer: 4,
                max_radius: 100.0,
                cell_size: 10.0,
            },
            max_pending_aoi_events_per_peer: 4,
            max_aoi_payload_bytes: 1024,
        };
        let mut authority = MultiPeerAuthority::new(config, EmptyProvider).unwrap();
        let (client_transport, server_transport) = InMemoryTransport::pair_secure();
        let (schema, seed) = schema_and_seed();
        let mut client = Replication::new(
            schema.clone(),
            8,
            &seed,
            ClientStrategy,
            Box::new(client_transport),
        )
        .unwrap();
        authority
            .admit_peer(
                42,
                (0.0, 0.0),
                10.0,
                schema,
                8,
                &seed,
                AddAuthority,
                Box::new(server_transport) as Box<dyn Transport>,
            )
            .unwrap();
        for _ in 0..16 {
            client.pump();
            authority.lease_inbound_commands(0).unwrap();
        }
        assert!(client.negotiated().is_some());
        client.try_send_rpc(b"apply once").unwrap();
        client.try_send_rpc(b"reject once").unwrap();
        client.try_send_rpc(b"panic once").unwrap();
        for _ in 0..8 {
            client.pump();
            authority.lease_inbound_commands(0).unwrap();
        }

        let invocations = Arc::new(AtomicUsize::new(0));
        let handler_invocations = Arc::clone(&invocations);
        let mut app = prism_app::App::new();
        app.insert_resource(authority);
        app.insert_resource(AppliedCount::default());
        app.add_plugins(
            AuthorityWorldExecutorPlugin::<AddAuthority, _>::new(move |world, command| {
                assert_eq!(command.peer_id, 42);
                let invocation = handler_invocations.fetch_add(1, Ordering::Relaxed);
                if command.payload == b"panic once" {
                    panic!("test handler failure");
                }
                if command.payload == b"reject once" {
                    return AuthorityCommandDecision::Rejected;
                }
                assert_eq!(command.payload, b"apply once");
                if invocation == 0 {
                    return AuthorityCommandDecision::Retry;
                }
                world.resource_mut::<AppliedCount>().0 += 1;
                AuthorityCommandDecision::Applied
            })
            .try_with_max_commands_per_tick(2)
            .unwrap(),
        );
        assert!(
            AuthorityWorldExecutorPlugin::<AddAuthority, _>::new(|_, _| {
                AuthorityCommandDecision::Rejected
            })
            .try_with_max_commands_per_tick(0)
            .is_err()
        );

        app.world_mut().run_schedule(FixedPreUpdate);
        assert_eq!(app.world().resource::<AppliedCount>().0, 0);
        assert_eq!(
            app.world()
                .resource::<AuthorityWorldExecutorStatus<AddAuthority>>()
                .stats,
            AuthorityWorldExecutorStats {
                applied: 0,
                rejected: 0,
                pending: true,
                authority_errors: 0,
                poisoned: false,
            }
        );

        app.world_mut().run_schedule(FixedPreUpdate);
        assert_eq!(app.world().resource::<AppliedCount>().0, 1);
        assert_eq!(
            app.world()
                .resource::<AuthorityWorldExecutorStatus<AddAuthority>>()
                .stats
                .rejected,
            1
        );
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            app.world_mut().run_schedule(FixedPreUpdate);
        }))
        .is_err());
        assert!(
            app.world()
                .resource::<AuthorityWorldExecutorStatus<AddAuthority>>()
                .stats
                .poisoned
        );
        app.world_mut().run_schedule(FixedPreUpdate);
        assert_eq!(app.world().resource::<AppliedCount>().0, 1);
        assert_eq!(invocations.load(Ordering::Relaxed), 4);
        assert!(app
            .world_mut()
            .resource_mut::<MultiPeerAuthority<AddAuthority>>()
            .lease_inbound_commands(1)
            .unwrap()
            .is_empty());
    }
}
