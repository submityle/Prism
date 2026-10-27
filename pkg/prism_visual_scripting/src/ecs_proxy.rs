//! Host-owned, read-only ECS bridge for visual-script nodes.
//!
//! The core VM never borrows an ECS `World`. Hosts publish a stable proxy for one
//! execution, keeping world lifetimes and thread affinity outside the scripting crate.

use crate::{InvokeCtx, VsError, VsResult};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Minimal read-only world surface safe to expose across an execution boundary.
pub trait WorldProxy: Send + Sync {
    fn entity_count(&self) -> u32;
    fn contains_bits(&self, entity_bits: u64) -> bool;

    /// Stable content identity when the host supplies a deterministic snapshot.
    fn snapshot_digest(&self) -> Option<[u8; 32]> {
        None
    }
}

/// Immutable host snapshot suitable for deterministic execution and replay.
#[derive(Clone, Debug, Default)]
pub struct SnapshotWorldProxy {
    entities: Arc<BTreeSet<u64>>,
    count: u32,
}

impl SnapshotWorldProxy {
    pub fn new<I>(entities: I) -> VsResult<Self>
    where
        I: IntoIterator<Item = u64>,
    {
        let entities = entities.into_iter().collect::<BTreeSet<_>>();
        let count = u32::try_from(entities.len())
            .map_err(|_| VsError::Validate("world entity count exceeds u32".into()))?;
        Ok(Self {
            entities: Arc::new(entities),
            count,
        })
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"prism.visual_scripting.ecs.snapshot.v1");
        hasher.update(self.count.to_le_bytes());
        for entity_bits in self.entities.iter() {
            hasher.update(entity_bits.to_le_bytes());
        }
        hasher.finalize().into()
    }
}

impl WorldProxy for SnapshotWorldProxy {
    fn entity_count(&self) -> u32 {
        self.count
    }

    fn contains_bits(&self, entity_bits: u64) -> bool {
        self.entities.contains(&entity_bits)
    }

    fn snapshot_digest(&self) -> Option<[u8; 32]> {
        Some(self.digest())
    }
}

/// Concrete external context injected through [`crate::ExecutionEnv::with_external`].
#[derive(Clone)]
pub struct WorldProxyContext {
    proxy: Arc<dyn WorldProxy>,
}

impl WorldProxyContext {
    pub fn new(proxy: Arc<dyn WorldProxy>) -> Self {
        Self { proxy }
    }

    pub fn proxy(&self) -> &dyn WorldProxy {
        self.proxy.as_ref()
    }
}

/// Retrieve the host-owned world proxy, producing a structured error when a graph
/// requires ECS access but the execution was intentionally run without a world.
pub fn require_world_proxy<'a>(ctx: &'a InvokeCtx<'a>) -> VsResult<&'a dyn WorldProxy> {
    ctx.external
        .and_then(|external| external.downcast_ref::<WorldProxyContext>())
        .map(WorldProxyContext::proxy)
        .ok_or_else(|| VsError::Runtime("ECS world proxy is not configured".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExecutionEnv, InMemoryHost, InvokeCtx, Value};
    use std::collections::BTreeMap;

    struct FakeWorld {
        count: u32,
        entities: Vec<u64>,
    }

    impl WorldProxy for FakeWorld {
        fn entity_count(&self) -> u32 {
            self.count
        }

        fn contains_bits(&self, bits: u64) -> bool {
            self.entities.contains(&bits)
        }
    }

    #[test]
    fn snapshot_proxy_is_deterministic_and_read_only() {
        let snapshot = SnapshotWorldProxy::new([9, 3, 9]).unwrap();
        assert_eq!(snapshot.entity_count(), 2);
        assert!(snapshot.contains_bits(3));
        assert!(!snapshot.contains_bits(4));
        assert_eq!(
            snapshot.digest(),
            SnapshotWorldProxy::new([3, 9]).unwrap().digest()
        );
        assert_eq!(
            WorldProxy::snapshot_digest(&snapshot),
            Some(snapshot.digest())
        );
    }

    #[test]
    fn proxy_is_read_only_and_host_owned() {
        let external = WorldProxyContext::new(Arc::new(FakeWorld {
            count: 2,
            entities: vec![0x100000001],
        }));
        let env = ExecutionEnv::new().with_external(&external);
        let cancel = tokio_util::sync::CancellationToken::new();
        let mut host = InMemoryHost::default();
        let config = BTreeMap::<String, Value>::new();
        let ctx = InvokeCtx::from_env(&config, &mut host, &cancel, &env);
        let proxy = require_world_proxy(&ctx).unwrap();
        assert_eq!(proxy.entity_count(), 2);
        assert!(proxy.contains_bits(0x100000001));
        assert!(!proxy.contains_bits(0x200000001));
    }

    #[test]
    fn missing_proxy_is_an_explicit_runtime_error() {
        let env = ExecutionEnv::new();
        let cancel = tokio_util::sync::CancellationToken::new();
        let mut host = InMemoryHost::default();
        let config = BTreeMap::<String, Value>::new();
        let ctx = InvokeCtx::from_env(&config, &mut host, &cancel, &env);
        assert!(matches!(
            require_world_proxy(&ctx),
            Err(VsError::Runtime(message)) if message.contains("not configured")
        ));
    }
}
