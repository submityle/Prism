//! Deterministic side-effect boundary for durable workflows.
//!
//! Effects are descriptions, not callbacks embedded in the graph. The VM assigns every
//! invocation a stable identity and delegates execution to an [`EffectExecutor`]. Executors may
//! route the request to a local service, a remote worker, or an `uwu_wasm` sandbox.

use crate::{
    DurableMutation, DurableStore, ExecNext, RecordKey, RevisionGuard, Value, VsError, VsResult,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectClass {
    ReadOnly,
    ExternalWrite,
    HumanInteraction,
    SandboxedCompute,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdempotencyPolicy {
    Required,
    BestEffort,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetryPolicy {
    pub max_attempts: u16,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
}

impl RetryPolicy {
    pub const fn never() -> Self {
        Self {
            max_attempts: 1,
            initial_backoff_ms: 0,
            max_backoff_ms: 0,
        }
    }

    pub fn backoff_ms(&self, failed_attempt: u16) -> u64 {
        if failed_attempt == 0 || self.initial_backoff_ms == 0 {
            return 0;
        }
        let shift = u32::from(failed_attempt.saturating_sub(1)).min(63);
        self.initial_backoff_ms
            .saturating_mul(1_u64 << shift)
            .min(self.max_backoff_ms)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::never()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectSpec {
    pub capability: String,
    pub operation: String,
    pub class: EffectClass,
    pub idempotency: IdempotencyPolicy,
    pub retry: RetryPolicy,
    pub timeout_ms: u64,
    #[serde(default)]
    pub required_permissions: Vec<String>,
}

impl EffectSpec {
    pub fn validate(&self) -> VsResult<()> {
        if self.capability.trim().is_empty() || self.operation.trim().is_empty() {
            return Err(VsError::Validate(
                "effect capability and operation must not be empty".into(),
            ));
        }
        if self.retry.max_attempts == 0 {
            return Err(VsError::Validate(
                "effect retry max_attempts must be greater than zero".into(),
            ));
        }
        if self.retry.max_backoff_ms < self.retry.initial_backoff_ms {
            return Err(VsError::Validate(
                "effect max_backoff_ms must be >= initial_backoff_ms".into(),
            ));
        }
        if self.timeout_ms == 0 {
            return Err(VsError::Validate(
                "effect timeout_ms must be greater than zero".into(),
            ));
        }
        if self.class == EffectClass::ExternalWrite
            && self.idempotency != IdempotencyPolicy::Required
        {
            return Err(VsError::Validate(
                "external write effects require idempotency".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectRequest {
    pub id: String,
    pub execution_id: String,
    pub task_id: u64,
    pub frame_depth: usize,
    pub sequence: u64,
    pub node_id: u32,
    pub spec: EffectSpec,
    pub inputs: Vec<Value>,
    pub config: BTreeMap<String, Value>,
    pub attempt: u16,
}

impl EffectRequest {
    pub(crate) fn stable_id(
        execution_id: &str,
        task_id: u64,
        frame_depth: usize,
        sequence: u64,
        node_id: u32,
    ) -> String {
        let mut digest = Sha256::new();
        digest.update(execution_id.as_bytes());
        digest.update(task_id.to_le_bytes());
        digest.update((frame_depth as u64).to_le_bytes());
        digest.update(sequence.to_le_bytes());
        digest.update(node_id.to_le_bytes());
        let digest = digest.finalize();
        let suffix = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        format!("effect:{suffix}")
    }

    fn payload_digest(&self) -> VsResult<[u8; 32]> {
        let encoded = serde_json::to_vec(&(
            &self.execution_id,
            self.task_id,
            self.frame_depth,
            self.sequence,
            self.node_id,
            &self.spec,
            &self.inputs,
            &self.config,
        ))
        .map_err(|error| VsError::Runtime(format!("effect digest failed: {error}")))?;
        Ok(Sha256::digest(encoded).into())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectResponse {
    pub outputs: Vec<Value>,
    pub next: ExecNext,
    #[serde(default)]
    pub chunks: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectError {
    pub message: String,
    pub retryable: bool,
}

impl EffectError {
    pub fn permanent(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
        }
    }

    pub fn retryable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
        }
    }
}

#[async_trait]
pub trait EffectExecutor: Send + Sync {
    async fn execute(&self, request: EffectRequest) -> Result<EffectResponse, EffectError>;
}

#[async_trait]
pub trait ExecutionClock: Send + Sync {
    fn now_millis(&self) -> u64;
    async fn sleep_millis(&self, millis: u64);
}

#[derive(Default)]
pub struct TokioClock;

#[async_trait]
impl ExecutionClock for TokioClock {
    fn now_millis(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
    }

    async fn sleep_millis(&self, millis: u64) {
        tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
    }
}

/// Deterministic clock for simulation and fast durable-workflow tests.
/// `sleep_millis` advances logical time immediately instead of blocking a thread.
#[derive(Debug, Default)]
pub struct VirtualClock {
    now_ms: std::sync::atomic::AtomicU64,
}

impl VirtualClock {
    pub const fn new(now_ms: u64) -> Self {
        Self {
            now_ms: std::sync::atomic::AtomicU64::new(now_ms),
        }
    }

    pub fn advance(&self, millis: u64) {
        self.now_ms
            .fetch_add(millis, std::sync::atomic::Ordering::SeqCst);
    }
}

#[async_trait]
impl ExecutionClock for VirtualClock {
    fn now_millis(&self) -> u64 {
        self.now_ms.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn sleep_millis(&self, millis: u64) {
        self.advance(millis);
    }
}

/// Serializing in-memory executor that replays committed responses by effect identity.
/// Production executors should provide the same atomic claim/commit contract in durable storage.
pub struct MemoizingEffectExecutor<E> {
    inner: E,
    committed: Mutex<HashMap<String, ([u8; 32], EffectResponse)>>,
    inflight: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl<E> MemoizingEffectExecutor<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            committed: Mutex::new(HashMap::new()),
            inflight: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl<E: EffectExecutor> EffectExecutor for MemoizingEffectExecutor<E> {
    async fn execute(&self, request: EffectRequest) -> Result<EffectResponse, EffectError> {
        let digest = request
            .payload_digest()
            .map_err(|error| EffectError::permanent(error.to_string()))?;
        let key = request.id.clone();
        let gate = {
            let mut inflight = self.inflight.lock().await;
            inflight
                .entry(key.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let _guard = gate.lock().await;
        let committed = self.committed.lock().await;
        if let Some((existing_digest, response)) = committed.get(&key) {
            if existing_digest != &digest {
                drop(committed);
                self.inflight.lock().await.remove(&key);
                return Err(EffectError::permanent(format!(
                    "effect identity {} was reused with a different payload",
                    request.id
                )));
            }
            let response = response.clone();
            drop(committed);
            self.inflight.lock().await.remove(&key);
            return Ok(response);
        }
        drop(committed);
        let result = self.inner.execute(request).await;
        match result {
            Ok(response) => {
                self.committed
                    .lock()
                    .await
                    .insert(key.clone(), (digest, response.clone()));
                self.inflight.lock().await.remove(&key);
                Ok(response)
            }
            Err(error) => {
                self.inflight.lock().await.remove(&key);
                Err(error)
            }
        }
    }
}

pub type SharedEffectExecutor = Arc<dyn EffectExecutor>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum DurableEffectState {
    Pending {
        payload_digest: [u8; 32],
        request: EffectRequest,
    },
    Committed {
        payload_digest: [u8; 32],
        response: EffectResponse,
    },
}

/// Durable at-least-once effect journal. A crash after the external operation
/// but before the commit may re-execute the request, so external writes must
/// honor the stable effect identity as an idempotency key.
pub struct DurableEffectExecutor<E> {
    inner: E,
    store: Arc<dyn DurableStore>,
    inflight: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl<E> DurableEffectExecutor<E> {
    pub fn new(inner: E, store: Arc<dyn DurableStore>) -> Self {
        Self {
            inner,
            store,
            inflight: Mutex::new(HashMap::new()),
        }
    }

    fn key(id: &str) -> Result<RecordKey, EffectError> {
        RecordKey::new("effects", id).map_err(|error| EffectError::permanent(error.to_string()))
    }

    fn load_state(
        &self,
        key: &RecordKey,
    ) -> Result<Option<(u64, DurableEffectState)>, EffectError> {
        let Some(record) = self
            .store
            .load(key)
            .map_err(|error| EffectError::retryable(error.to_string()))?
        else {
            return Ok(None);
        };
        let state = record
            .decode()
            .map_err(|error| EffectError::permanent(error.to_string()))?;
        Ok(Some((record.revision, state)))
    }
}

#[async_trait]
impl<E: EffectExecutor> EffectExecutor for DurableEffectExecutor<E> {
    async fn execute(&self, request: EffectRequest) -> Result<EffectResponse, EffectError> {
        let digest = request
            .payload_digest()
            .map_err(|error| EffectError::permanent(error.to_string()))?;
        let key = Self::key(&request.id)?;
        let gate = {
            let mut inflight = self.inflight.lock().await;
            inflight
                .entry(request.id.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let _guard = gate.lock().await;

        let revision = match self.load_state(&key)? {
            Some((
                _,
                DurableEffectState::Committed {
                    payload_digest,
                    response,
                },
            )) => {
                if payload_digest != digest {
                    self.inflight.lock().await.remove(&request.id);
                    return Err(EffectError::permanent(format!(
                        "effect identity {} was reused with a different payload",
                        request.id
                    )));
                }
                self.inflight.lock().await.remove(&request.id);
                return Ok(response);
            }
            Some((revision, DurableEffectState::Pending { payload_digest, .. })) => {
                if payload_digest != digest {
                    self.inflight.lock().await.remove(&request.id);
                    return Err(EffectError::permanent(format!(
                        "effect identity {} was reused with a different payload",
                        request.id
                    )));
                }
                revision
            }
            None => {
                let state = DurableEffectState::Pending {
                    payload_digest: digest,
                    request: request.clone(),
                };
                match self.store.commit(
                    &key,
                    RevisionGuard::Create,
                    DurableMutation::default()
                        .with_payload(&state)
                        .map_err(|error| EffectError::permanent(error.to_string()))?,
                ) {
                    Ok(record) => record.revision,
                    Err(_) => match self.load_state(&key)? {
                        Some((revision, DurableEffectState::Pending { payload_digest, .. }))
                            if payload_digest == digest =>
                        {
                            revision
                        }
                        Some((
                            _,
                            DurableEffectState::Committed {
                                payload_digest,
                                response,
                            },
                        )) if payload_digest == digest => {
                            self.inflight.lock().await.remove(&request.id);
                            return Ok(response);
                        }
                        Some(_) => {
                            self.inflight.lock().await.remove(&request.id);
                            return Err(EffectError::permanent(format!(
                                "effect identity {} was reused with a different payload",
                                request.id
                            )));
                        }
                        None => {
                            self.inflight.lock().await.remove(&request.id);
                            return Err(EffectError::retryable(
                                "effect journal create conflict could not be resolved",
                            ));
                        }
                    },
                }
            }
        };

        let response = match self.inner.execute(request.clone()).await {
            Ok(response) => response,
            Err(error) => {
                self.inflight.lock().await.remove(&request.id);
                return Err(error);
            }
        };
        let state = DurableEffectState::Committed {
            payload_digest: digest,
            response: response.clone(),
        };
        match self.store.commit(
            &key,
            RevisionGuard::Match(revision),
            DurableMutation::default()
                .with_payload(&state)
                .map_err(|error| EffectError::permanent(error.to_string()))?,
        ) {
            Ok(_) => {
                self.inflight.lock().await.remove(&request.id);
                Ok(response)
            }
            Err(error) => match self.load_state(&key)? {
                Some((
                    _,
                    DurableEffectState::Committed {
                        payload_digest,
                        response,
                    },
                )) if payload_digest == digest => {
                    self.inflight.lock().await.remove(&request.id);
                    Ok(response)
                }
                _ => {
                    self.inflight.lock().await.remove(&request.id);
                    Err(EffectError::retryable(format!(
                        "commit durable effect response: {error}"
                    )))
                }
            },
        }
    }
}
