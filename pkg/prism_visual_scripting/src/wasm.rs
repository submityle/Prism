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
use std::time::Instant;

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
    pub fuel_remaining: u64,
    pub memory_pages_limit: u32,
    pub deadline: Instant,
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
    /// Compile validated bytes into a host-owned content-addressed instance.
    /// The default is deliberately rejected; production hosts should return a
    /// compiled handle representation understood by their `call` method.
    fn compile_module(&self, bytes: &[u8]) -> Result<Vec<u8>, WasmHostError> {
        self.validate_module(bytes)?;
        Err(WasmHostError::permanent(
            "WASM host must compile component bytes",
        ))
    }

    fn call(&self, request: WasmCall) -> Result<WasmCallResult, WasmHostError>;
}

/// Adapter that turns a typed `WasmHost` into a `ScriptBackend`.
///
/// The compiled instance is retained as a host-defined opaque byte handle;
/// actual Component compilation and instance pooling remain owned by the host
/// and can be backed by `prism_wasm::Sandbox` without leaking wasmtime types.
pub struct WasmScriptBackend {
    host: Arc<dyn WasmHost>,
    component: String,
}

impl WasmScriptBackend {
    pub fn new(host: Arc<dyn WasmHost>, component: impl Into<String>) -> Self {
        Self {
            host,
            component: component.into(),
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
            .compile_module(&module.bytes)
            .map_err(|error| VsError::Compile(error.message))
    }

    fn call(
        &self,
        instance: &Self::Instance,
        request: &crate::ScriptCall,
        budget: &mut CallBudget,
    ) -> VsResult<BackendCallResult> {
        budget.charge(1)?;
        if instance.is_empty() {
            return Err(VsError::Runtime("WASM script instance is empty".into()));
        }
        budget.check_deadline()?;
        let result = self
            .host
            .call(WasmCall {
                component: self.component.clone(),
                operation: request.operation.clone(),
                request_id: request.request_id.clone(),
                execution_id: request.execution_id.clone(),
                attempt: request.attempt,
                inputs: request.args.clone(),
                config: request.config.clone(),
                fuel_remaining: budget.remaining_fuel,
                memory_pages_limit: budget.limits.memory_pages,
                deadline: budget.deadline(),
            })
            .map_err(|error| {
                if error.retryable {
                    VsError::RetryableRuntime(error.message)
                } else {
                    VsError::Runtime(error.message)
                }
            })?;
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
            fuel_remaining: u64::MAX,
            memory_pages_limit: u32::MAX,
            deadline: Instant::now() + std::time::Duration::from_secs(10),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct RecordingHost {
        call: Mutex<Option<WasmCall>>,
    }

    impl WasmHost for RecordingHost {
        fn validate_module(&self, bytes: &[u8]) -> Result<(), WasmHostError> {
            if bytes == b"component" {
                Ok(())
            } else {
                Err(WasmHostError {
                    message: "invalid component".into(),
                    retryable: false,
                })
            }
        }

        fn compile_module(&self, bytes: &[u8]) -> Result<Vec<u8>, WasmHostError> {
            self.validate_module(bytes)?;
            Ok(bytes.to_vec())
        }

        fn call(&self, request: WasmCall) -> Result<WasmCallResult, WasmHostError> {
            *self.call.lock().unwrap() = Some(request);
            Ok(WasmCallResult {
                outputs: vec![Value::I64(9)],
                next: ExecNext::End,
                chunks: Vec::new(),
            })
        }
    }

    #[test]
    fn adapter_forwards_identity_config_and_budget() {
        let host = Arc::new(RecordingHost {
            call: Mutex::new(None),
        });
        let backend = WasmScriptBackend::new(host.clone(), "combat");
        let module = ScriptModule::from_bytes("combat", 1, b"component");
        let instance = backend
            .compile(&module, &BindingRegistry::default())
            .unwrap();
        let request = crate::ScriptCall {
            module: "combat".into(),
            operation: "tick".into(),
            request_id: "req-42".into(),
            execution_id: "exec-42".into(),
            attempt: 3,
            args: vec![Value::I64(1)],
            config: BTreeMap::from([("difficulty".into(), Value::I64(2))]),
        };
        let mut budget = CallBudget::new(crate::ResourceBudget {
            fuel: 100,
            memory_pages: 64,
            ..crate::ResourceBudget::sandboxed()
        });
        let result = backend.call(&instance, &request, &mut budget).unwrap();
        assert_eq!(result.outputs, vec![Value::I64(9)]);
        let captured = host.call.lock().unwrap().clone().unwrap();
        assert_eq!(captured.request_id, "req-42");
        assert_eq!(captured.execution_id, "exec-42");
        assert_eq!(captured.attempt, 3);
        assert_eq!(captured.config, request.config);
        assert_eq!(captured.fuel_remaining, 99);
        assert_eq!(captured.memory_pages_limit, 64);
        assert!(captured.deadline >= budget.started);
    }
}
