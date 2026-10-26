//! Optional WASM effect adapter boundary.
//!
//! The visual-scripting core does not own a WASM runtime. Hosts implement
//! [`WasmHost`] around their chosen sandbox and retain control of fuel,
//! deadlines, capabilities, and module admission.

use crate::effect::{EffectError, EffectExecutor, EffectRequest, EffectResponse};
use crate::{ExecNext, Value};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Host-owned JSON Component ABI implementation.
pub trait WasmHost: Send + Sync + 'static {
    /// Invoke one component operation and return its JSON response payload.
    fn call_json(&self, component: &str, operation: &str, input: String) -> Result<String, String>;
}

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct WasmEffectInput {
    request_id: String,
    execution_id: String,
    attempt: u16,
    inputs: Vec<Value>,
    config: std::collections::BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WasmEffectOutput {
    outputs: Vec<Value>,
    next: ExecNext,
    #[serde(default)]
    chunks: Vec<Value>,
    #[serde(default)]
    error: Option<WasmEffectError>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WasmEffectError {
    message: String,
    retryable: bool,
}

/// Effect executor backed by a host-provided WASM/component sandbox.
pub struct WasmEffectExecutor {
    host: Arc<dyn WasmHost>,
    component: String,
}

impl WasmEffectExecutor {
    /// Construct an executor for a registered component name.
    pub fn new(host: Arc<dyn WasmHost>, component: impl Into<String>) -> Self {
        Self {
            host,
            component: component.into(),
        }
    }

    /// Return the registered component name.
    pub fn component(&self) -> &str {
        &self.component
    }
}

#[async_trait]
impl EffectExecutor for WasmEffectExecutor {
    async fn execute(&self, request: EffectRequest) -> Result<EffectResponse, EffectError> {
        let operation = request.spec.operation.clone();
        let input = serde_json::to_string(&WasmEffectInput {
            request_id: request.id.clone(),
            execution_id: request.execution_id.clone(),
            attempt: request.attempt,
            inputs: request.inputs.clone(),
            config: request.config.clone(),
        })
        .map_err(|error| EffectError::permanent(format!("encode wasm effect input: {error}")))?;
        let host = Arc::clone(&self.host);
        let component = self.component.clone();
        let result =
            tokio::task::spawn_blocking(move || host.call_json(&component, &operation, input))
                .await
                .map_err(|error| {
                    EffectError::retryable(format!("wasm effect worker failed: {error}"))
                })?
                .map_err(|error| EffectError::retryable(format!("wasm effect failed: {error}")))?;
        let response = serde_json::from_str::<WasmEffectOutput>(&result).map_err(|error| {
            EffectError::permanent(format!("decode wasm effect output: {error}"))
        })?;
        if let Some(error) = response.error {
            return Err(EffectError {
                message: error.message,
                retryable: error.retryable,
            });
        }
        Ok(EffectResponse {
            outputs: response.outputs,
            next: response.next,
            chunks: response.chunks,
        })
    }
}
