//! Host-owned, read-only ECS bridge for visual-script nodes.
//!
//! The core VM never borrows an ECS `World`. Hosts publish a stable proxy for one
//! execution, keeping world lifetimes and thread affinity outside the scripting crate.

use crate::{InvokeCtx, VsError, VsResult};
use std::sync::Arc;

/// Minimal read-only world surface safe to expose across an execution boundary.
pub trait WorldProxy: Send + Sync {
    fn entity_count(&self) -> u32;
    fn contains_bits(&self, entity_bits: u64) -> bool;
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
