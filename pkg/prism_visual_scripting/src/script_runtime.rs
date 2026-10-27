//! Production script runtime control plane.
//!
//! This module contains the backend-neutral pieces that make a script module
//! shippable: stable bindings, capability policy, content-addressed hot swap,
//! schema migration, and bounded execution admission.  The WASM backend is an
//! adapter behind these contracts; the control plane remains deterministic and
//! easy to test without a GPU or a particular VM.

use crate::error::{VsError, VsResult};
use crate::registry::library::ExecNext;
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
        if !valid_identifier(&binding.name) {
            return Err(VsError::Validate(
                "binding name must be non-empty and space-free".into(),
            ));
        }
        if self.bindings.contains_key(&binding.name) {
            return Err(VsError::Validate("binding already registered".into()));
        }
        self.bindings.insert(binding.name.clone(), binding);
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

    pub fn validate_args(&self, name: &str, args: &[Value]) -> VsResult<()> {
        let binding = self
            .bindings
            .get(name)
            .ok_or_else(|| VsError::Runtime(format!("unknown host binding: {name}")))?;
        if binding.inputs.len() != args.len() {
            return Err(VsError::Validate(format!(
                "binding `{name}` expects {} arguments, received {}",
                binding.inputs.len(),
                args.len()
            )));
        }
        for (index, (expected, actual)) in binding.inputs.iter().zip(args).enumerate() {
            if !expected.accepts(&actual.ty()) {
                return Err(VsError::Validate(format!(
                    "binding `{name}` argument {index} expects {expected:?}, received {:?}",
                    actual.ty()
                )));
            }
        }
        Ok(())
    }

    pub fn validate_output(&self, name: &str, outputs: &[Value]) -> VsResult<()> {
        let binding = self
            .bindings
            .get(name)
            .ok_or_else(|| VsError::Runtime(format!("unknown host binding: {name}")))?;
        match (&binding.output, outputs) {
            (None, []) => Ok(()),
            (None, _) => Err(VsError::Runtime(format!(
                "binding `{name}` declares no output, received {}",
                outputs.len()
            ))),
            (Some(expected), [actual]) if expected.accepts(&actual.ty()) => Ok(()),
            (Some(expected), [actual]) => Err(VsError::Runtime(format!(
                "binding `{name}` output expects {expected:?}, received {:?}",
                actual.ty()
            ))),
            (Some(_), received) => Err(VsError::Runtime(format!(
                "binding `{name}` declares one output, received {}",
                received.len()
            ))),
        }
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
        let name = name.into();
        let bytes: Arc<[u8]> = Arc::from(bytes.into());
        let digest = Self::compute_digest(&name, schema_version, &bytes);
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

    pub fn validate(&self) -> VsResult<()> {
        if !valid_identifier(&self.name) {
            return Err(VsError::Validate(
                "module name must be non-empty and space-free".into(),
            ));
        }
        if self.schema_version == 0 {
            return Err(VsError::Validate(
                "module schema version must be non-zero".into(),
            ));
        }
        if self.bytes.is_empty() {
            return Err(VsError::Validate("module bytes must be non-empty".into()));
        }
        if self.digest != Self::compute_digest(&self.name, self.schema_version, &self.bytes) {
            return Err(VsError::Validate(
                "module digest does not match content".into(),
            ));
        }
        Ok(())
    }

    fn compute_digest(name: &str, schema_version: u32, bytes: &[u8]) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"prism.script.module.v1\0");
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(schema_version.to_le_bytes());
        hash.update(bytes);
        hash.finalize().into()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StateBlob {
    pub schema_version: u32,
    pub fields: BTreeMap<String, Value>,
}

pub type MigrationFn = Arc<dyn Fn(StateBlob) -> VsResult<StateBlob> + Send + Sync>;

const MAX_IDENTIFIER_BYTES: usize = 128;
const MAX_CONFIG_ENTRIES: usize = 256;
const MAX_ID_BYTES: usize = 128;

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_IDENTIFIER_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'-'))
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptLease {
    pub module: ScriptModule,
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
                generation: generation
                    .checked_add(1)
                    .ok_or_else(|| VsError::Runtime("hot-swap generation exhausted".into()))?,
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
    pub fn acquire(&self, name: &str) -> Option<ScriptLease> {
        self.modules
            .read()
            .ok()?
            .get(name)
            .map(|(module, generation)| ScriptLease {
                module: module.clone(),
                generation: *generation,
            })
    }

    pub fn is_current(&self, lease: &ScriptLease) -> bool {
        self.modules
            .read()
            .ok()
            .and_then(|modules| modules.get(&lease.module.name).cloned())
            .is_some_and(|(module, generation)| {
                generation == lease.generation && module.digest == lease.module.digest
            })
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
        request: &ScriptCall,
        budget: &mut CallBudget,
    ) -> VsResult<BackendCallResult>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScriptCall {
    pub module: String,
    pub operation: String,
    pub request_id: String,
    pub execution_id: String,
    pub attempt: u16,
    pub args: Vec<Value>,
    pub config: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackendCallResult {
    pub outputs: Vec<Value>,
    pub next: ExecNext,
    pub chunks: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScriptCallResult {
    pub module_digest: [u8; 32],
    pub outputs: Vec<Value>,
    pub next: ExecNext,
    pub chunks: Vec<Value>,
    pub generation: u64,
}

#[derive(Clone)]
struct PublishedScript {
    module: ScriptModule,
    instance: Option<Vec<u8>>,
    generation: u64,
}

#[derive(Clone)]
pub struct ScriptHost {
    pub bindings: Arc<BindingRegistry>,
    pub policy: ScriptPolicy,
    pub migrator: StateMigrator,
    backend: Option<Arc<dyn ScriptBackend<Instance = Vec<u8>>>>,
    published: Arc<RwLock<BTreeMap<String, PublishedScript>>>,
}

impl ScriptHost {
    pub fn new(policy: ScriptPolicy, bindings: BindingRegistry) -> Self {
        Self {
            bindings: Arc::new(bindings),
            policy,
            migrator: StateMigrator::default(),
            backend: None,
            published: Arc::new(RwLock::new(BTreeMap::new())),
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
        module.validate()?;
        if let Ok(published) = self.published.read() {
            if let Some(existing) = published.get(&module.name) {
                if existing.module.digest == module.digest {
                    return Ok(SwapRecord {
                        old_digest: Some(existing.module.digest),
                        new_digest: module.digest,
                        generation: existing.generation,
                    });
                }
            }
        }
        let max_module_bytes = (self.policy.budget.memory_pages as usize)
            .checked_mul(64 * 1024)
            .ok_or_else(|| VsError::Validate("script memory budget overflow".into()))?;
        if module.bytes.len() > max_module_bytes {
            return Err(VsError::Validate(format!(
                "script module exceeds memory budget: {} > {} bytes",
                module.bytes.len(),
                max_module_bytes
            )));
        }
        let instance = if let Some(backend) = &self.backend {
            Some(backend.compile(&module, &self.bindings)?)
        } else {
            None
        };
        let mut published = self
            .published
            .write()
            .map_err(|_| VsError::Runtime("published script registry poisoned".into()))?;
        let generation = published.get(&module.name).map_or(1, |current| {
            if current.module.digest == module.digest {
                current.generation
            } else {
                current.generation.saturating_add(1)
            }
        });
        let record = SwapRecord {
            old_digest: published
                .get(&module.name)
                .map(|current| current.module.digest),
            new_digest: module.digest,
            generation,
        };
        published.insert(
            module.name.clone(),
            PublishedScript {
                module,
                instance,
                generation,
            },
        );
        Ok(record)
    }
    pub fn authorize(&self, binding: &str) -> VsResult<()> {
        self.bindings.authorize(binding, &self.policy)
    }
    pub fn begin_call(&self) -> CallBudget {
        CallBudget::new(self.policy.budget.clone())
    }
    pub fn call(&self, request: ScriptCall) -> VsResult<ScriptCallResult> {
        if request.request_id.is_empty()
            || request.execution_id.is_empty()
            || request.request_id.len() > MAX_ID_BYTES
            || request.execution_id.len() > MAX_ID_BYTES
        {
            return Err(VsError::Validate(
                "script request and execution identities must be non-empty and bounded".into(),
            ));
        }
        if request.config.len() > MAX_CONFIG_ENTRIES {
            return Err(VsError::Validate(format!(
                "script config exceeds {} entries",
                MAX_CONFIG_ENTRIES
            )));
        }
        if !valid_identifier(&request.module) {
            return Err(VsError::Validate(
                "script module must be non-empty and space-free".into(),
            ));
        }
        if !valid_identifier(&request.operation) {
            return Err(VsError::Validate(
                "script operation must be non-empty and space-free".into(),
            ));
        }
        if self.bindings.contains(&request.operation) {
            self.bindings.authorize(&request.operation, &self.policy)?;
        } else {
            return Err(VsError::Runtime(format!(
                "unknown host binding: {}",
                request.operation
            )));
        }
        self.bindings
            .validate_args(&request.operation, &request.args)?;
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| VsError::Runtime("script backend is not configured".into()))?;
        let (module, instance, generation) = {
            let published = self
                .published
                .read()
                .map_err(|_| VsError::Runtime("published script registry poisoned".into()))?;
            let script = published.get(&request.module).ok_or_else(|| {
                VsError::Runtime(format!("script module not installed: {}", request.module))
            })?;
            let instance = script
                .instance
                .as_ref()
                .ok_or_else(|| VsError::Runtime("script instance is not compiled".into()))?
                .clone();
            (script.module.clone(), instance, script.generation)
        };
        let mut budget = self.begin_call();
        let result = backend.call(&instance, &request, &mut budget)?;
        self.bindings
            .validate_output(&request.operation, &result.outputs)?;
        budget.check_deadline()?;
        Ok(ScriptCallResult {
            module_digest: module.digest,
            outputs: result.outputs,
            next: result.next,
            chunks: result.chunks,
            generation,
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
    pub fn check_deadline(&self) -> VsResult<()> {
        if self.started.elapsed() > self.limits.wall_time {
            return Err(VsError::Runtime("script wall-time budget exceeded".into()));
        }
        Ok(())
    }
    pub fn deadline(&self) -> Instant {
        self.started + self.limits.wall_time
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
            request: &ScriptCall,
            budget: &mut CallBudget,
        ) -> VsResult<BackendCallResult> {
            budget.charge(1)?;
            if instance.is_empty() || request.operation != "echo" {
                return Err(VsError::Runtime("invalid echo invocation".into()));
            }
            Ok(BackendCallResult {
                outputs: request.args.clone(),
                next: ExecNext::End,
                chunks: Vec::new(),
            })
        }
    }

    struct RejectingBackend;

    impl ScriptBackend for RejectingBackend {
        type Instance = Vec<u8>;

        fn compile(
            &self,
            module: &ScriptModule,
            _bindings: &BindingRegistry,
        ) -> VsResult<Self::Instance> {
            if module.bytes.as_ref() == b"reject" {
                return Err(VsError::Runtime("compile rejected".into()));
            }
            Ok(module.bytes.to_vec())
        }

        fn call(
            &self,
            instance: &Self::Instance,
            _request: &ScriptCall,
            budget: &mut CallBudget,
        ) -> VsResult<BackendCallResult> {
            budget.charge(1)?;
            Ok(BackendCallResult {
                outputs: vec![Value::I64(instance.len() as i64)],
                next: ExecNext::End,
                chunks: Vec::new(),
            })
        }
    }
    #[test]
    fn failed_publish_preserves_previous_snapshot() {
        let mut bindings = BindingRegistry::default();
        bindings
            .register(Binding {
                name: "measure".into(),
                category: "test".into(),
                inputs: vec![],
                output: Some(ValueType::I64),
                capabilities: BTreeSet::new(),
                deterministic: true,
            })
            .unwrap();
        let host = ScriptHost::new(ScriptPolicy::new(SandboxTier::Sandboxed), bindings)
            .with_backend(RejectingBackend);
        host.publish(ScriptModule::from_bytes("stable", 1, b"good"))
            .unwrap();
        assert!(matches!(
            host.publish(ScriptModule::from_bytes("stable", 1, b"reject")),
            Err(VsError::Runtime(message)) if message.contains("compile rejected")
        ));
        let result = host
            .call(ScriptCall {
                module: "stable".into(),
                operation: "measure".into(),
                request_id: "request".into(),
                execution_id: "execution".into(),
                attempt: 0,
                args: Vec::new(),
                config: BTreeMap::new(),
            })
            .unwrap();
        assert_eq!(
            result.module_digest,
            ScriptModule::from_bytes("stable", 1, b"good").digest
        );
        assert_eq!(result.generation, 1);
    }

    #[test]
    fn hot_swap_leases_revoke_old_generation() {
        let swaps = HotSwap::default();
        let first = ScriptModule::from_bytes("lease", 1, b"v1");
        swaps.publish(first).unwrap();
        let lease = swaps.acquire("lease").unwrap();
        assert!(swaps.is_current(&lease));

        swaps
            .publish(ScriptModule::from_bytes("lease", 1, b"v2"))
            .unwrap();
        assert!(!swaps.is_current(&lease));
        let current = swaps.acquire("lease").unwrap();
        assert_eq!(current.generation, 2);
        assert!(swaps.is_current(&current));
    }

    #[test]
    fn publish_rejects_invalid_module_metadata() {
        let host = ScriptHost::new(
            ScriptPolicy::new(SandboxTier::Sandboxed),
            BindingRegistry::default(),
        );
        for module in [
            ScriptModule::from_bytes("x".repeat(129), 1, b"module"),
            ScriptModule::from_bytes("empty-schema", 0, b"module"),
            ScriptModule::from_bytes("empty-bytes", 1, Vec::<u8>::new()),
        ] {
            assert!(matches!(host.publish(module), Err(VsError::Validate(_))));
        }
    }
    #[test]
    fn publish_rejects_tampered_module_digest() {
        let host = ScriptHost::new(
            ScriptPolicy::new(SandboxTier::Sandboxed),
            BindingRegistry::default(),
        );
        let mut module = ScriptModule::from_bytes("tampered", 1, b"module");
        module.digest[0] ^= 0xff;
        assert!(
            matches!(host.publish(module), Err(VsError::Validate(message)) if message.contains("digest"))
        );
    }
    #[test]
    fn publish_rejects_modules_larger_than_memory_budget() {
        let mut policy = ScriptPolicy::new(SandboxTier::Fortified);
        policy.budget.memory_pages = 1;
        let host = ScriptHost::new(policy, BindingRegistry::default());
        let oversized = ScriptModule::from_bytes("large", 1, vec![0_u8; 64 * 1024 + 1]);
        let error = host.publish(oversized).unwrap_err();
        assert!(
            matches!(error, VsError::Validate(message) if message.contains("exceeds memory budget"))
        );
    }

    #[test]
    fn binding_argument_contract_rejects_wrong_shape() {
        let mut registry = BindingRegistry::default();
        registry
            .register(Binding {
                name: "typed".into(),
                category: "test".into(),
                inputs: vec![ValueType::I64],
                output: None,
                capabilities: BTreeSet::new(),
                deterministic: true,
            })
            .unwrap();
        let error = registry
            .validate_args("typed", &[Value::Bool(true)])
            .unwrap_err();
        assert!(matches!(error, VsError::Validate(message) if message.contains("expects I64")));
    }

    #[test]
    fn binding_output_contract_rejects_wrong_type() {
        let mut registry = BindingRegistry::default();
        registry
            .register(Binding {
                name: "typed.output".into(),
                category: "test".into(),
                inputs: vec![],
                output: Some(ValueType::I64),
                capabilities: BTreeSet::new(),
                deterministic: true,
            })
            .unwrap();
        let error = registry
            .validate_output("typed.output", &[Value::Bool(true)])
            .unwrap_err();
        assert!(
            matches!(error, VsError::Runtime(message) if message.contains("output expects I64"))
        );
    }

    #[test]
    fn host_publishes_compiles_and_calls_backend() {
        let mut bindings = BindingRegistry::default();
        bindings
            .register(Binding {
                name: "echo".into(),
                category: "test".into(),
                inputs: vec![ValueType::I64],
                output: Some(ValueType::I64),
                capabilities: BTreeSet::new(),
                deterministic: true,
            })
            .unwrap();
        let host = ScriptHost::new(ScriptPolicy::new(SandboxTier::Sandboxed), bindings)
            .with_backend(EchoBackend);
        let module = ScriptModule::from_bytes("echo", 1, b"component");
        let digest = module.digest;
        host.publish(module).unwrap();
        let result = host
            .call(ScriptCall {
                module: "echo".into(),
                operation: "echo".into(),
                request_id: "req-1".into(),
                execution_id: "exec-1".into(),
                attempt: 0,
                args: vec![Value::I64(7)],
                config: BTreeMap::new(),
            })
            .unwrap();
        assert_eq!(result.module_digest, digest);
        assert_eq!(result.generation, 1);
        assert_eq!(result.outputs, vec![Value::I64(7)]);
    }

    #[test]
    fn call_rejects_oversized_identity_and_config() {
        let host = ScriptHost::new(
            ScriptPolicy::new(SandboxTier::Sandboxed),
            BindingRegistry::default(),
        );
        let oversized_id = "i".repeat(MAX_ID_BYTES + 1);
        let identity_error = host.call(ScriptCall {
            module: "module".into(),
            operation: "op".into(),
            request_id: oversized_id,
            execution_id: "execution".into(),
            attempt: 0,
            args: Vec::new(),
            config: BTreeMap::new(),
        });
        assert!(
            matches!(identity_error, Err(VsError::Validate(message)) if message.contains("identities"))
        );

        let config = (0..=MAX_CONFIG_ENTRIES)
            .map(|index| (format!("key-{index}"), Value::I64(index as i64)))
            .collect();
        let config_error = host.call(ScriptCall {
            module: "module".into(),
            operation: "op".into(),
            request_id: "request".into(),
            execution_id: "execution".into(),
            attempt: 0,
            args: Vec::new(),
            config,
        });
        assert!(
            matches!(config_error, Err(VsError::Validate(message)) if message.contains("config"))
        );
    }
    #[test]
    fn call_rejects_invalid_module_identifier() {
        let host = ScriptHost::new(
            ScriptPolicy::new(SandboxTier::Sandboxed),
            BindingRegistry::default(),
        );
        let error = host.call(ScriptCall {
            module: "module/../escape".into(),
            operation: "op".into(),
            request_id: "request".into(),
            execution_id: "execution".into(),
            attempt: 0,
            args: Vec::new(),
            config: BTreeMap::new(),
        });
        assert!(matches!(error, Err(VsError::Validate(message)) if message.contains("module")));
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
