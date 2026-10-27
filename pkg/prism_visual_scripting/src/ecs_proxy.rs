//! Host-owned, read-only ECS bridge for visual-script nodes.
//!
//! The core VM never borrows an ECS `World`. Hosts publish a stable proxy for one
//! execution, keeping world lifetimes and thread affinity outside the scripting crate.

use crate::{InvokeCtx, VsError, VsResult};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const MAX_COMPONENT_NAME_BYTES: usize = 128;
const MAX_COMPONENT_BYTES: usize = 1024 * 1024;
const MAX_TOTAL_COMPONENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_COMPONENT_ENTRIES: usize = 65_536;
const MAX_SNAPSHOT_ENTITIES: usize = 1_000_000;

/// Minimal read-only world surface safe to expose across an execution boundary.
pub trait WorldProxy: Send + Sync {
    fn entity_count(&self) -> u32;
    fn contains_bits(&self, entity_bits: u64) -> bool;

    /// Return a deterministic, owned encoding of one reflected component.
    fn component_bytes(&self, entity_bits: u64, component: &str) -> Option<Vec<u8>> {
        let _ = (entity_bits, component);
        None
    }

    /// Return sorted component type names visible on one entity.
    fn component_names(&self, entity_bits: u64) -> Vec<String> {
        let _ = entity_bits;
        Vec::new()
    }

    /// Stable content identity when the host supplies a deterministic snapshot.
    fn snapshot_digest(&self) -> Option<[u8; 32]> {
        None
    }
}

/// Immutable host snapshot suitable for deterministic execution and replay.
#[derive(Clone, Debug, Default)]
pub struct SnapshotWorldProxy {
    entities: Arc<BTreeSet<u64>>,
    components: Arc<BTreeMap<(u64, String), Vec<u8>>>,
    count: u32,
}

impl SnapshotWorldProxy {
    pub fn new<I>(entities: I) -> VsResult<Self>
    where
        I: IntoIterator<Item = u64>,
    {
        Self::with_components(entities, std::iter::empty::<(u64, String, Vec<u8>)>())
    }

    pub fn with_components<I, C>(entities: I, components: C) -> VsResult<Self>
    where
        I: IntoIterator<Item = u64>,
        C: IntoIterator<Item = (u64, String, Vec<u8>)>,
    {
        let entities = entities.into_iter().collect::<BTreeSet<_>>();
        if entities.len() > MAX_SNAPSHOT_ENTITIES {
            return Err(VsError::Validate(
                "world entity count exceeds snapshot limit".into(),
            ));
        }
        let mut component_map = BTreeMap::new();
        let mut total_component_bytes = 0usize;
        for (entity_bits, component, bytes) in components {
            if component_map.len() >= MAX_COMPONENT_ENTRIES
                || component.is_empty()
                || component.len() > MAX_COMPONENT_NAME_BYTES
                || !component.is_ascii()
                || bytes.len() > MAX_COMPONENT_BYTES
                || !entities.contains(&entity_bits)
                || component_map.contains_key(&(entity_bits, component.clone()))
                || total_component_bytes
                    .checked_add(bytes.len())
                    .is_none_or(|total| total > MAX_TOTAL_COMPONENT_BYTES)
            {
                return Err(VsError::Validate(
                    "invalid ECS component snapshot entry".into(),
                ));
            }
            total_component_bytes += bytes.len();
            component_map.insert((entity_bits, component), bytes);
        }
        let count = u32::try_from(entities.len())
            .map_err(|_| VsError::Validate("world entity count exceeds u32".into()))?;
        Ok(Self {
            entities: Arc::new(entities),
            components: Arc::new(component_map),
            count,
        })
    }

    pub fn component_json(&self, entity_bits: u64, component: &str) -> VsResult<serde_json::Value> {
        let bytes = self
            .component_bytes(entity_bits, component)
            .ok_or_else(|| VsError::Runtime("ECS component is not present in snapshot".into()))?;
        serde_json::from_slice(&bytes)
            .map_err(|error| VsError::Runtime(format!("ECS component JSON decode failed: {error}")))
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"prism.visual_scripting.ecs.snapshot.v1");
        hasher.update(self.count.to_le_bytes());
        for entity_bits in self.entities.iter() {
            hasher.update(entity_bits.to_le_bytes());
        }
        for ((entity_bits, component), bytes) in self.components.iter() {
            hasher.update(entity_bits.to_le_bytes());
            hasher.update((component.len() as u64).to_le_bytes());
            hasher.update(component.as_bytes());
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
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

    fn component_bytes(&self, entity_bits: u64, component: &str) -> Option<Vec<u8>> {
        self.components
            .get(&(entity_bits, component.to_owned()))
            .cloned()
    }

    fn component_names(&self, entity_bits: u64) -> Vec<String> {
        self.components
            .range((entity_bits, String::new())..=(entity_bits, "\u{10ffff}".to_owned()))
            .map(|((_, component), _)| component.clone())
            .collect()
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

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        self.proxy.snapshot_digest()
    }

    pub fn component_bytes(&self, entity_bits: u64, component: &str) -> Option<Vec<u8>> {
        self.proxy.component_bytes(entity_bits, component)
    }

    pub fn component_names(&self, entity_bits: u64) -> Vec<String> {
        self.proxy.component_names(entity_bits)
    }

    pub fn component_json(&self, entity_bits: u64, component: &str) -> VsResult<serde_json::Value> {
        let bytes = self
            .component_bytes(entity_bits, component)
            .ok_or_else(|| VsError::Runtime("ECS component is not present in snapshot".into()))?;
        serde_json::from_slice(&bytes)
            .map_err(|error| VsError::Runtime(format!("ECS component JSON decode failed: {error}")))
    }
}

pub fn require_entity_count<'a>(ctx: &'a InvokeCtx<'a>) -> VsResult<u32> {
    Ok(require_world_proxy(ctx)?.entity_count())
}

pub fn require_component_bytes<'a>(
    ctx: &'a InvokeCtx<'a>,
    entity_bits: u64,
    component: &str,
) -> VsResult<Vec<u8>> {
    if component.is_empty() || component.len() > MAX_COMPONENT_NAME_BYTES || !component.is_ascii() {
        return Err(VsError::Validate("invalid ECS component name".into()));
    }
    let proxy = require_world_proxy(ctx)?;
    if !proxy.contains_bits(entity_bits) {
        return Err(VsError::Runtime(
            "ECS entity is not present in snapshot".into(),
        ));
    }
    proxy
        .component_bytes(entity_bits, component)
        .ok_or_else(|| VsError::Runtime("ECS component is not present in snapshot".into()))
}

pub fn require_component_names<'a>(
    ctx: &'a InvokeCtx<'a>,
    entity_bits: u64,
) -> VsResult<Vec<String>> {
    let proxy = require_world_proxy(ctx)?;
    if !proxy.contains_bits(entity_bits) {
        return Err(VsError::Runtime(
            "ECS entity is not present in snapshot".into(),
        ));
    }
    let mut names = proxy.component_names(entity_bits);
    if names.len() > MAX_COMPONENT_ENTRIES {
        return Err(VsError::Validate(
            "ECS component name count exceeds snapshot limit".into(),
        ));
    }
    if names
        .iter()
        .any(|name| name.is_empty() || name.len() > MAX_COMPONENT_NAME_BYTES || !name.is_ascii())
    {
        return Err(VsError::Validate("invalid ECS component name list".into()));
    }
    names.sort_unstable();
    if names.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(VsError::Validate(
            "ECS component name list contains duplicates".into(),
        ));
    }
    Ok(names)
}

pub fn require_component_json<'a>(
    ctx: &'a InvokeCtx<'a>,
    entity_bits: u64,
    component: &str,
) -> VsResult<serde_json::Value> {
    let bytes = require_component_bytes(ctx, entity_bits, component)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| VsError::Runtime(format!("ECS component JSON decode failed: {error}")))
}

pub fn require_snapshot_digest<'a>(ctx: &'a InvokeCtx<'a>) -> VsResult<[u8; 32]> {
    let proxy = require_world_proxy(ctx)?;
    proxy
        .snapshot_digest()
        .ok_or_else(|| VsError::Runtime("ECS snapshot digest is not configured".into()))
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
    fn snapshot_rejects_duplicate_or_unbounded_component_entries() {
        assert!(SnapshotWorldProxy::with_components(
            [7],
            [
                (7, "Transform".to_owned(), vec![1]),
                (7, "Transform".to_owned(), vec![2]),
            ],
        )
        .is_err());
        assert!(
            SnapshotWorldProxy::with_components([7], [(7, "位置".to_owned(), vec![1])],).is_err()
        );
        assert!(SnapshotWorldProxy::with_components(
            [7],
            [(7, "Transform".to_owned(), vec![0; MAX_COMPONENT_BYTES + 1])],
        )
        .is_err());
    }

    #[test]
    fn snapshot_component_reads_are_owned_and_digest_bound() {
        let snapshot = SnapshotWorldProxy::with_components(
            [7],
            [
                (7, "Transform".to_owned(), vec![1, 2, 3]),
                (7, "Velocity".to_owned(), vec![4]),
            ],
        )
        .unwrap();
        assert_eq!(snapshot.component_names(7), vec!["Transform", "Velocity"]);
        assert!(snapshot.component_names(8).is_empty());
        assert_eq!(
            snapshot.component_bytes(7, "Transform"),
            Some(vec![1, 2, 3])
        );
        assert_eq!(snapshot.component_bytes(7, "Health"), None);
        let changed =
            SnapshotWorldProxy::with_components([7], [(7, "Transform".to_owned(), vec![1, 2, 4])])
                .unwrap();
        assert_ne!(snapshot.digest(), changed.digest());
    }

    #[test]
    fn snapshot_rejects_component_for_unknown_entity() {
        assert!(matches!(
            SnapshotWorldProxy::with_components(
                [7],
                [(8, "Transform".to_owned(), vec![1])],
            ),
            Err(VsError::Validate(message)) if message.contains("component snapshot")
        ));
    }

    #[test]
    fn snapshot_rejects_total_component_payload_limit() {
        let components =
            (0..17).map(|entity| (entity, "Transform".to_owned(), vec![0; MAX_COMPONENT_BYTES]));
        assert!(SnapshotWorldProxy::with_components(0..17, components).is_err());
    }

    #[test]
    fn snapshot_rejects_entity_limit() {
        assert!(SnapshotWorldProxy::with_components(
            0..=MAX_SNAPSHOT_ENTITIES as u64,
            std::iter::empty::<(u64, String, Vec<u8>)>(),
        )
        .is_err());
    }

    #[test]
    fn snapshot_rejects_component_entry_limit() {
        let entities = 0..=MAX_COMPONENT_ENTRIES as u64;
        let components = (0..=MAX_COMPONENT_ENTRIES as u64)
            .map(|entity| (entity, format!("C{entity}"), Vec::new()));
        assert!(SnapshotWorldProxy::with_components(entities, components).is_err());
    }

    #[test]
    fn component_json_decoding_is_structured() {
        let snapshot = SnapshotWorldProxy::with_components(
            [7],
            [
                (7, "Transform".to_owned(), br#"{"x":1}"#.to_vec()),
                (7, "Broken".to_owned(), b"not-json".to_vec()),
            ],
        )
        .unwrap();
        assert_eq!(
            snapshot.component_json(7, "Transform").unwrap()["x"],
            serde_json::json!(1)
        );
        assert!(matches!(
            snapshot.component_json(7, "Broken"),
            Err(VsError::Runtime(message)) if message.contains("JSON decode")
        ));
    }

    #[test]
    fn required_component_reads_are_structured() {
        let snapshot =
            SnapshotWorldProxy::with_components([7], [(7, "Transform".to_owned(), vec![1, 2])])
                .unwrap();
        let external = WorldProxyContext::new(Arc::new(snapshot));
        let env = ExecutionEnv::new().with_external(&external);
        let cancel = tokio_util::sync::CancellationToken::new();
        let mut host = InMemoryHost::default();
        let config = BTreeMap::<String, Value>::new();
        let ctx = InvokeCtx::from_env(&config, &mut host, &cancel, &env);
        assert_eq!(require_entity_count(&ctx).unwrap(), 1);
        assert_eq!(require_component_names(&ctx, 7).unwrap(), vec!["Transform"]);
        assert!(matches!(
            require_component_names(&ctx, 8),
            Err(VsError::Runtime(message)) if message.contains("entity")
        ));
        assert_eq!(
            require_component_bytes(&ctx, 7, "Transform").unwrap(),
            vec![1, 2]
        );
        assert!(matches!(
            require_component_bytes(&ctx, 8, "Transform"),
            Err(VsError::Runtime(message)) if message.contains("entity")
        ));
        assert!(matches!(
            require_component_bytes(&ctx, 7, "Velocity"),
            Err(VsError::Runtime(message)) if message.contains("component")
        ));
        assert!(matches!(
            require_component_bytes(&ctx, 7, "位置"),
            Err(VsError::Validate(message)) if message.contains("component name")
        ));
    }

    #[test]
    fn required_component_names_rejects_noncanonical_host_output() {
        struct MaliciousWorld;
        impl WorldProxy for MaliciousWorld {
            fn entity_count(&self) -> u32 {
                1
            }

            fn contains_bits(&self, entity_bits: u64) -> bool {
                entity_bits == 7
            }

            fn component_names(&self, _entity_bits: u64) -> Vec<String> {
                vec!["Velocity".into(), "Velocity".into(), "位置".into()]
            }
        }
        let external = WorldProxyContext::new(Arc::new(MaliciousWorld));
        let env = ExecutionEnv::new().with_external(&external);
        let cancel = tokio_util::sync::CancellationToken::new();
        let mut host = InMemoryHost::default();
        let config = BTreeMap::<String, Value>::new();
        let ctx = InvokeCtx::from_env(&config, &mut host, &cancel, &env);
        assert!(matches!(
            require_component_names(&ctx, 7),
            Err(VsError::Validate(message)) if message.contains("component name")
        ));
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
        let context = WorldProxyContext::new(Arc::new(snapshot.clone()));
        assert_eq!(context.snapshot_digest(), Some(snapshot.digest()));
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
        assert!(matches!(
            require_snapshot_digest(&ctx),
            Err(VsError::Runtime(message)) if message.contains("snapshot digest")
        ));
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
