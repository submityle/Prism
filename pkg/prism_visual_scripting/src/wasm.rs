//! Typed WASM effect boundary for the visual-scripting runtime.
//!
//! `prism_visual_scripting` owns the Effect contract; `prism_wasm` owns the
//! Component sandbox. This module deliberately does not expose a JSON ABI:
//! hosts receive typed Rust values and decide how to encode them for a
//! Component call. JSON remains an application-level adapter outside the
//! runtime hot path.

use crate::effect::{EffectError, EffectExecutor, EffectRequest, EffectResponse};
use crate::{
    BackendCallResult, BindingRegistry, CallBudget, ExecNext, ScriptBackend, ScriptModule, Value,
    VsError, VsResult,
};
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A typed request handed to a host-owned `prism_wasm` Component sandbox.
#[derive(Clone, Debug)]
pub struct WasmCall {
    pub component: String,
    pub operation: String,
    pub request_id: String,
    pub execution_id: String,
    pub attempt: u16,
    pub inputs: Vec<Value>,
    pub config: BTreeMap<String, Value>,
}

/// A typed Component result before it is committed as a visual-scripting Effect.
#[derive(Clone, Debug)]
pub struct WasmCallResult {
    pub outputs: Vec<Value>,
    pub next: ExecNext,
    pub chunks: Vec<Value>,
}

/// Host failure with an explicit retry contract.
#[derive(Clone, Debug)]
pub struct WasmHostError {
    pub message: String,
    pub retryable: bool,
}

impl WasmHostError {
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

/// Host-owned typed Component ABI.
///
/// Implementations normally wrap `prism_wasm::Sandbox::call_typed` or its
/// batch form. The trait does not prescribe a wire representation, allowing a
/// POD Component ABI without forcing JSON allocation into the Effect path.
pub trait WasmHost: Send + Sync + 'static {
    /// Validate and compile component bytes in the host-owned sandbox.
    fn validate_module(&self, _bytes: &[u8]) -> Result<(), WasmHostError> {
        Err(WasmHostError::permanent(
            "WASM host must validate component bytes",
        ))
    }
    fn call(&self, request: WasmCall) -> Result<WasmCallResult, WasmHostError>;
}

use std::sync::atomic::{AtomicU64, Ordering};

/// Adapter that turns a typed `WasmHost` into a `ScriptBackend`.
///
/// The compiled instance is retained as content-addressed bytes; actual
/// Component compilation/instance pooling remains owned by the host and can
/// be backed by `prism_wasm::Sandbox` without leaking wasmtime types here.
pub struct WasmScriptBackend {
    host: Arc<dyn WasmHost>,
    component: String,
    sequence: AtomicU64,
}

impl WasmScriptBackend {
    pub fn new(host: Arc<dyn WasmHost>, component: impl Into<String>) -> Self {
        Self {
            host,
            component: component.into(),
            sequence: AtomicU64::new(1),
        }
    }
}

impl ScriptBackend for WasmScriptBackend {
    type Instance = Vec<u8>;

    fn compile(
        &self,
        module: &ScriptModule,
        _bindings: &BindingRegistry,
    ) -> VsResult<Self::Instance> {
        if module.bytes.is_empty() {
            return Err(VsError::Runtime("WASM script module is empty".into()));
        }
        self.host
            .validate_module(&module.bytes)
            .map_err(|error| VsError::Compile(error.message))?;
        Ok(module.bytes.to_vec())
    }

    fn call(
        &self,
        instance: &Self::Instance,
        operation: &str,
        args: &[Value],
        budget: &mut CallBudget,
    ) -> VsResult<BackendCallResult> {
        budget.charge(1)?;
        if instance.is_empty() {
            return Err(VsError::Runtime("WASM script instance is empty".into()));
        }
        let id = self.sequence.fetch_add(1, Ordering::Relaxed).to_string();
        let result = self
            .host
            .call(WasmCall {
                component: self.component.clone(),
                operation: operation.to_string(),
                request_id: id.clone(),
                execution_id: format!("script:{id}"),
                attempt: 0,
                inputs: args.to_vec(),
                config: BTreeMap::new(),
            })
            .map_err(|error| VsError::Runtime(error.message))?;
        Ok(BackendCallResult {
            outputs: result.outputs,
            next: result.next,
            chunks: result.chunks,
        })
    }
}
pub struct WasmEffectExecutor {
    host: Arc<dyn WasmHost>,
    component: String,
}

impl WasmEffectExecutor {
    /// Construct an executor for a registered Component name.
    pub fn new(host: Arc<dyn WasmHost>, component: impl Into<String>) -> Self {
        Self {
            host,
            component: component.into(),
        }
    }

    pub fn component(&self) -> &str {
        &self.component
    }
}

#[async_trait]
impl EffectExecutor for WasmEffectExecutor {
    async fn execute(&self, request: EffectRequest) -> Result<EffectResponse, EffectError> {
        let host = Arc::clone(&self.host);
        let call = WasmCall {
            component: self.component.clone(),
            operation: request.spec.operation.clone(),
            request_id: request.id,
            execution_id: request.execution_id,
            attempt: request.attempt,
            inputs: request.inputs,
            config: request.config,
        };
        let result = tokio::task::spawn_blocking(move || host.call(call))
            .await
            .map_err(|error| EffectError::retryable(format!("wasm effect worker failed: {error}")))?
            .map_err(|error| EffectError {
                message: error.message,
                retryable: error.retryable,
            })?;
        Ok(EffectResponse {
            outputs: result.outputs,
            next: result.next,
            chunks: result.chunks,
        })
    }
}
