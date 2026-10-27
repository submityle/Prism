//! Production script runtime control plane.
//!
//! This module contains the backend-neutral pieces that make a script module
//! shippable: stable bindings, capability policy, content-addressed hot swap,
//! schema migration, and bounded execution admission.  The WASM backend is an
//! adapter behind these contracts; the control plane remains deterministic and
//! easy to test without a GPU or a particular VM.

use crate::error::{VsError, VsResult};
use crate::value::{Value, ValueType};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub category: String,
    pub inputs: Vec<ValueType>,
    pub output: Option<ValueType>,
    pub capabilities: BTreeSet<String>,
    pub deterministic: bool,
}

#[derive(Clone, Debug, Default)]
pub struct BindingRegistry {
    bindings: BTreeMap<String, Binding>,
}

impl BindingRegistry {
    pub fn register(&mut self, binding: Binding) -> VsResult<()> {
        if binding.name.is_empty() || binding.name.contains(' ') {
            return Err(VsError::Validate(
                "binding name must be non-empty and space-free".into(),
            ));
        }
        if self
            .bindings
            .insert(binding.name.clone(), binding)
            .is_some()
        {
            return Err(VsError::Validate("binding already registered".into()));
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Binding> {
        self.bindings.get(name)
    }
    pub fn iter(&self) -> impl Iterator<Item = &Binding> {
        self.bindings.values()
    }
    pub fn contains(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    pub fn authorize(&self, name: &str, policy: &ScriptPolicy) -> VsResult<()> {
        let binding = self
            .bindings
            .get(name)
            .ok_or_else(|| VsError::Runtime(format!("unknown host binding: {name}")))?;
        if !binding.capabilities.is_subset(&policy.capabilities) {
            return Err(VsError::Runtime(format!(
                "capability denied for binding: {name}"
            )));
        }
        if policy.deterministic && !binding.deterministic {
            return Err(VsError::Runtime(format!(
                "non-deterministic binding in deterministic mode: {name}"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxTier {
    Trusted,
    Sandboxed,
    Fortified,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceBudget {
    pub fuel: u64,
    pub memory_pages: u32,
    pub wall_time: Duration,
    pub max_calls: u32,
}

impl ResourceBudget {
    pub fn trusted() -> Self {
        Self {
            fuel: u64::MAX,
            memory_pages: 65_536,
            wall_time: Duration::from_secs(10),
            max_calls: u32::MAX,
        }
    }
    pub fn sandboxed() -> Self {
        Self {
            fuel: 2_000_000,
            memory_pages: 256,
            wall_time: Duration::from_millis(4),
            max_calls: 128,
        }
    }
    pub fn fortified() -> Self {
        Self {
            fuel: 250_000,
            memory_pages: 64,
            wall_time: Duration::from_millis(1),
            max_calls: 32,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptPolicy {
    pub tier: SandboxTier,
    pub capabilities: BTreeSet<String>,
    pub budget: ResourceBudget,
    pub deterministic: bool,
}

impl ScriptPolicy {
    pub fn new(tier: SandboxTier) -> Self {
        let budget = match tier {
            SandboxTier::Trusted => ResourceBudget::trusted(),
            SandboxTier::Sandboxed => ResourceBudget::sandboxed(),
            SandboxTier::Fortified => ResourceBudget::fortified(),
        };
        Self {
            tier,
            capabilities: BTreeSet::new(),
            budget,
            deterministic: false,
        }
    }
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.insert(capability.into());
        self
    }
    pub fn deterministic(mut self, enabled: bool) -> Self {
        self.deterministic = enabled;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptModule {
    pub name: String,
    pub digest: [u8; 32],
    pub schema_version: u32,
    pub bytes: Arc<[u8]>,
}

impl ScriptModule {
    pub fn from_bytes(
        name: impl Into<String>,
        schema_version: u32,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        let bytes: Arc<[u8]> = Arc::from(bytes.into());
        let mut hash = Sha256::new();
        hash.update(&bytes);
        let digest: [u8; 32] = hash.finalize().into();
        Self {
            name: name.into(),
            digest,
            schema_version,
            bytes,
        }
    }
    pub fn digest_hex(&self) -> String {
        self.digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StateBlob {
    pub schema_version: u32,
    pub fields: BTreeMap<String, Value>,
}

pub type MigrationFn = Arc<dyn Fn(StateBlob) -> VsResult<StateBlob> + Send + Sync>;

#[derive(Clone, Default)]
pub struct StateMigrator {
    steps: Arc<RwLock<HashMap<(u32, u32), MigrationFn>>>,
}

impl StateMigrator {
    pub fn register(&self, from: u32, to: u32, migration: MigrationFn) -> VsResult<()> {
        if to <= from {
            return Err(VsError::Validate("migration target must be newer".into()));
        }
        let mut steps = self
            .steps
            .write()
            .map_err(|_| VsError::Runtime("migration registry poisoned".into()))?;
        if steps.insert((from, to), migration).is_some() {
            return Err(VsError::Validate("migration already registered".into()));
        }
        Ok(())
    }
    pub fn migrate(&self, mut state: StateBlob, target: u32) -> VsResult<StateBlob> {
        while state.schema_version < target {
            let from = state.schema_version;
            let step = {
                let steps = self
                    .steps
                    .read()
                    .map_err(|_| VsError::Runtime("migration registry poisoned".into()))?;
                steps
                    .iter()
                    .filter(|((f, t), _)| *f == from && *t <= target)
                    .max_by_key(|((_, t), _)| *t)
                    .map(|(_, f)| Arc::clone(f))
            }
            .ok_or_else(|| VsError::Runtime(format!("missing state migration {from}->{target}")))?;
            state = step(state)?;
            if state.schema_version <= from {
                return Err(VsError::Runtime("migration did not advance schema".into()));
            }
        }
        if state.schema_version != target {
            return Err(VsError::Runtime("migration overshot target schema".into()));
        }
        Ok(state)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwapRecord {
    pub old_digest: Option<[u8; 32]>,
    pub new_digest: [u8; 32],
    pub generation: u64,
}

#[derive(Clone, Default)]
pub struct HotSwap {
    modules: Arc<RwLock<BTreeMap<String, (ScriptModule, u64)>>>,
}

impl HotSwap {
    pub fn publish(&self, module: ScriptModule) -> VsResult<SwapRecord> {
        let mut modules = self
            .modules
            .write()
            .map_err(|_| VsError::Runtime("hot-swap registry poisoned".into()))?;
        let record = match modules.get(&module.name) {
            Some((old, generation)) if old.digest == module.digest => SwapRecord {
                old_digest: Some(old.digest),
                new_digest: module.digest,
                generation: *generation,
            },
            Some((old, generation)) => SwapRecord {
                old_digest: Some(old.digest),
                new_digest: module.digest,
                generation: generation.saturating_add(1),
            },
            None => SwapRecord {
                old_digest: None,
                new_digest: module.digest,
                generation: 1,
            },
        };
        modules.insert(module.name.clone(), (module, record.generation));
        Ok(record)
    }
    pub fn current(&self, name: &str) -> Option<ScriptModule> {
        self.modules.read().ok()?.get(name).map(|(m, _)| m.clone())
    }
    pub fn generation(&self, name: &str) -> Option<u64> {
        self.modules.read().ok()?.get(name).map(|(_, g)| *g)
    }
}

pub trait ScriptBackend: Send + Sync {
    type Instance: Send + Sync;
    fn compile(
        &self,
        module: &ScriptModule,
        bindings: &BindingRegistry,
    ) -> VsResult<Self::Instance>;
    fn call(
        &self,
        instance: &Self::Instance,
        operation: &str,
        args: &[Value],
        budget: &mut CallBudget,
    ) -> VsResult<Vec<Value>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScriptCall {
    pub module: String,
    pub operation: String,
    pub args: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScriptCallResult {
    pub module_digest: [u8; 32],
    pub outputs: Vec<Value>,
    pub generation: u64,
}

#[derive(Clone)]
pub struct ScriptHost {
    pub bindings: Arc<BindingRegistry>,
    pub policy: ScriptPolicy,
    pub hotswap: HotSwap,
    pub migrator: StateMigrator,
    backend: Option<Arc<dyn ScriptBackend<Instance = Vec<u8>>>>,
    instances: Arc<RwLock<BTreeMap<String, (Vec<u8>, [u8; 32])>>>,
}

impl ScriptHost {
    pub fn new(policy: ScriptPolicy, bindings: BindingRegistry) -> Self {
        Self {
            bindings: Arc::new(bindings),
            policy,
            hotswap: HotSwap::default(),
            migrator: StateMigrator::default(),
            backend: None,
            instances: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }
    pub fn with_backend<B>(mut self, backend: B) -> Self
    where
        B: ScriptBackend<Instance = Vec<u8>> + 'static,
    {
        self.backend = Some(Arc::new(backend));
        self
    }
    pub fn publish(&self, module: ScriptModule) -> VsResult<SwapRecord> {
        if let Some(backend) = &self.backend {
            let instance = backend.compile(&module, &self.bindings)?;
            self.instances
                .write()
                .map_err(|_| VsError::Runtime("script instance registry poisoned".into()))?
                .insert(module.name.clone(), (instance, module.digest));
        }
        self.hotswap.publish(module)
    }
    pub fn authorize(&self, binding: &str) -> VsResult<()> {
        self.bindings.authorize(binding, &self.policy)
    }
    pub fn begin_call(&self) -> CallBudget {
        CallBudget::new(self.policy.budget.clone())
    }
    pub fn call(&self, request: ScriptCall) -> VsResult<ScriptCallResult> {
        let module = self.hotswap.current(&request.module).ok_or_else(|| {
            VsError::Runtime(format!("script module not installed: {}", request.module))
        })?;
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| VsError::Runtime("script backend is not configured".into()))?;
        let instance = self
            .instances
            .read()
            .map_err(|_| VsError::Runtime("script instance registry poisoned".into()))?
            .get(&request.module)
            .cloned()
            .ok_or_else(|| VsError::Runtime("script instance is not compiled".into()))?;
        if instance.1 != module.digest {
            return Err(VsError::Runtime("script instance digest is stale".into()));
        }
        let mut budget = self.begin_call();
        let outputs = backend.call(&instance.0, &request.operation, &request.args, &mut budget)?;
        Ok(ScriptCallResult {
            module_digest: module.digest,
            outputs,
            generation: self.hotswap.generation(&request.module).unwrap_or(0),
        })
    }
}

#[derive(Clone, Debug)]
pub struct CallBudget {
    pub remaining_fuel: u64,
    pub calls: u32,
    pub started: Instant,
    pub limits: ResourceBudget,
}

impl CallBudget {
    pub fn new(limits: ResourceBudget) -> Self {
        Self {
            remaining_fuel: limits.fuel,
            calls: 0,
            started: Instant::now(),
            limits,
        }
    }
    pub fn charge(&mut self, fuel: u64) -> VsResult<()> {
        self.calls = self
            .calls
            .checked_add(1)
            .ok_or_else(|| VsError::Runtime("call counter overflow".into()))?;
        if self.calls > self.limits.max_calls {
            return Err(VsError::Runtime("script call budget exceeded".into()));
        }
        self.remaining_fuel = self
            .remaining_fuel
            .checked_sub(fuel)
            .ok_or_else(|| VsError::Runtime("script fuel budget exceeded".into()))?;
        if self.started.elapsed() > self.limits.wall_time {
            return Err(VsError::Runtime("script wall-time budget exceeded".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct EchoBackend;

    impl ScriptBackend for EchoBackend {
        type Instance = Vec<u8>;

        fn compile(
            &self,
            module: &ScriptModule,
            _bindings: &BindingRegistry,
        ) -> VsResult<Self::Instance> {
            Ok(module.bytes.to_vec())
        }

        fn call(
            &self,
            instance: &Self::Instance,
            operation: &str,
            args: &[Value],
            budget: &mut CallBudget,
        ) -> VsResult<Vec<Value>> {
            budget.charge(1)?;
            if instance.is_empty() || operation != "echo" {
                return Err(VsError::Runtime("invalid echo invocation".into()));
            }
            Ok(args.to_vec())
        }
    }

    #[test]
    fn host_publishes_compiles_and_calls_backend() {
        let host = ScriptHost::new(
            ScriptPolicy::new(SandboxTier::Sandboxed),
            BindingRegistry::default(),
        )
        .with_backend(EchoBackend);
        let module = ScriptModule::from_bytes("echo", 1, b"component");
        let digest = module.digest;
        host.publish(module).unwrap();
        let result = host
            .call(ScriptCall {
                module: "echo".into(),
                operation: "echo".into(),
                args: vec![Value::I64(7)],
            })
            .unwrap();
        assert_eq!(result.module_digest, digest);
        assert_eq!(result.generation, 1);
        assert_eq!(result.outputs, vec![Value::I64(7)]);
    }

    #[test]
    fn content_addressed_publish_is_atomic_and_monotonic() {
        let swap = HotSwap::default();
        let first = swap
            .publish(ScriptModule::from_bytes("combat", 1, b"v1".to_vec()))
            .unwrap();
        let second = swap
            .publish(ScriptModule::from_bytes("combat", 2, b"v2".to_vec()))
            .unwrap();
        assert_eq!(first.generation, 1);
        assert_eq!(second.generation, 2);
        assert_eq!(swap.current("combat").unwrap().schema_version, 2);
    }
    #[test]
    fn policy_rejects_capability_and_nondeterminism() {
        let mut registry = BindingRegistry::default();
        registry
            .register(Binding {
                name: "time.now".into(),
                category: "time".into(),
                inputs: vec![],
                output: Some(ValueType::I64),
                capabilities: ["clock".into()].into(),
                deterministic: false,
            })
            .unwrap();
        let policy = ScriptPolicy::new(SandboxTier::Fortified).deterministic(true);
        assert!(registry.authorize("time.now", &policy).is_err());
    }
    #[test]
    fn migrator_walks_multiple_schema_steps() {
        let migrator = StateMigrator::default();
        migrator
            .register(
                1,
                2,
                Arc::new(|mut s| {
                    s.schema_version = 2;
                    Ok(s)
                }),
            )
            .unwrap();
        migrator
            .register(
                2,
                3,
                Arc::new(|mut s| {
                    s.schema_version = 3;
                    Ok(s)
                }),
            )
            .unwrap();
        let state = migrator
            .migrate(
                StateBlob {
                    schema_version: 1,
                    fields: BTreeMap::new(),
                },
                3,
            )
            .unwrap();
        assert_eq!(state.schema_version, 3);
    }
    #[test]
    fn budget_enforces_call_count() {
        let mut budget = CallBudget::new(ResourceBudget {
            fuel: 10,
            memory_pages: 1,
            wall_time: Duration::from_secs(1),
            max_calls: 1,
        });
        budget.charge(5).unwrap();
        assert!(budget.charge(1).is_err());
    }
}
