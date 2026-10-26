//! Prism-facing integration for visual scripting.
//!
//! This module is intentionally small: the VM remains backend-neutral while
//! Prism hosts can exchange reflected values, structured diagnostics, and
//! cancellable asynchronous work.

use crate::analysis::{FindingSeverity, StaticFinding, StaticModelReport};
use crate::error::{VsError, VsResult};
use crate::registry::{ConcurrentNodeLibrary, NodeDefinition, NodeRegistry};
use crate::value::Value;
use futures::Future;
use prism_diagnostic::{Event, Level};
use prism_reflect::{impl_reflect_value, Reflect};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// JSON payload used when crossing the Prism reflection boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonValue(pub serde_json::Value);

impl_reflect_value!(JsonValue);

/// Convert a reflected Prism value into a visual-scripting [`Value`].
pub fn value_from_reflect(value: &dyn Reflect) -> Result<Value, ReflectValueError> {
    if let Some(value) = value.downcast_ref::<bool>() {
        return Ok(Value::Bool(*value));
    }
    if let Some(value) = value.downcast_ref::<i32>() {
        return Ok(Value::I32(*value));
    }
    if let Some(value) = value.downcast_ref::<i64>() {
        return Ok(Value::I64(*value));
    }
    if let Some(value) = value.downcast_ref::<f32>() {
        return Ok(Value::F32(*value));
    }
    if let Some(value) = value.downcast_ref::<f64>() {
        return Ok(Value::F64(*value));
    }
    if let Some(value) = value.downcast_ref::<String>() {
        return Ok(Value::String(Arc::from(value.as_str())));
    }
    if let Some(value) = value.downcast_ref::<JsonValue>() {
        return Ok(Value::Json(Arc::new(value.0.clone())));
    }
    Err(ReflectValueError::UnsupportedType(
        value.type_name().to_owned(),
    ))
}

/// Convert a visual-scripting [`Value`] to a reflected Prism value.
pub fn value_to_reflect(value: &Value) -> Result<Box<dyn Reflect>, ReflectValueError> {
    Ok(match value {
        Value::Bool(value) => Box::new(*value),
        Value::I32(value) => Box::new(*value),
        Value::I64(value) => Box::new(*value),
        Value::F32(value) => Box::new(*value),
        Value::F64(value) => Box::new(*value),
        Value::String(value) => Box::new(value.to_string()),
        Value::Json(value) => Box::new(JsonValue((**value).clone())),
        Value::Unit | Value::List(_) => {
            return Err(ReflectValueError::UnsupportedValue(value.ty()))
        }
    })
}

/// A conversion failure at the Prism reflection boundary.
#[derive(Debug, thiserror::Error)]
pub enum ReflectValueError {
    #[error("unsupported reflected type: {0}")]
    UnsupportedType(String),
    #[error("unsupported visual-scripting value type: {0:?}")]
    UnsupportedValue(crate::value::ValueType),
}

/// Map a visual-scripting error to Prism's structured diagnostic event model.
pub fn diagnostic_for_error(error: &VsError) -> Event {
    Event::new(Level::Error, "prism_visual_scripting", error.to_string())
}

/// Map one static-analysis finding to a Prism diagnostic event.
pub fn diagnostic_for_finding(finding: &StaticFinding) -> Event {
    let level = match finding.severity {
        FindingSeverity::Warning => Level::Warn,
        FindingSeverity::Error => Level::Error,
    };
    Event::new(
        level,
        "prism_visual_scripting",
        format!("{}: {}", finding.code, finding.message),
    )
    .with_field(
        "code",
        prism_diagnostic::FieldValue::Str(finding.code.clone()),
    )
}

/// Convert every finding in a static report to structured Prism diagnostics.
pub fn diagnostics_for_report(report: &StaticModelReport) -> Vec<Event> {
    report.findings.iter().map(diagnostic_for_finding).collect()
}

/// Stable identifier for work submitted through [`PrismHost`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TaskId(u64);

/// State returned by [`PrismHost::poll_task`].
#[derive(Debug)]
pub enum TaskPoll {
    Pending,
    Ready(VsResult<Value>),
    Cancelled,
    Unknown,
}

type TaskReceiver = oneshot::Receiver<VsResult<Value>>;

/// Backend-neutral Prism host boundary for node registration and cancellable work.
///
/// The host owns only Prism/UVS contracts: no ECS or Bevy types are required.
/// Tasks run on the caller's Tokio runtime and must cooperatively observe the
/// cancellation token supplied to their future.
pub struct PrismHost {
    registry: Arc<ConcurrentNodeLibrary>,
    next_task: AtomicU64,
    tasks: Mutex<HashMap<TaskId, (CancellationToken, TaskReceiver)>>,
}

impl Default for PrismHost {
    fn default() -> Self {
        Self::new()
    }
}

impl PrismHost {
    #[must_use]
    pub fn new() -> Self {
        Self {
            registry: ConcurrentNodeLibrary::new(),
            next_task: AtomicU64::new(1),
            tasks: Mutex::new(HashMap::new()),
        }
    }

    /// Register or replace a node definition and return its stable identity.
    pub fn register_node(&self, definition: NodeDefinition) -> String {
        let identity = definition.identity();
        self.registry.register(definition);
        identity
    }

    #[must_use]
    pub fn registry(&self) -> &Arc<ConcurrentNodeLibrary> {
        &self.registry
    }

    #[must_use]
    pub fn has_node(&self, id: &str) -> bool {
        self.registry.get(id).is_some()
    }

    /// Submit cancellable work. The future receives a token owned by the host.
    pub fn spawn_task<F, Fut>(&self, work: F) -> TaskId
    where
        F: FnOnce(CancellationToken) -> Fut + Send + 'static,
        Fut: Future<Output = VsResult<Value>> + Send + 'static,
    {
        let id = TaskId(self.next_task.fetch_add(1, Ordering::Relaxed));
        let cancel = CancellationToken::new();
        let child = cancel.clone();
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let result = work(child.clone()).await;
            if !child.is_cancelled() {
                let _ = tx.send(result);
            }
        });
        self.tasks
            .lock()
            .expect("task registry lock poisoned")
            .insert(id, (cancel, rx));
        id
    }

    /// Request cancellation. Work that ignores its token is still drained as cancelled.
    pub fn cancel_task(&self, id: TaskId) -> bool {
        self.tasks
            .lock()
            .expect("task registry lock poisoned")
            .get(&id)
            .map(|(token, _)| {
                token.cancel();
                true
            })
            .unwrap_or(false)
    }

    /// Poll without blocking; completed tasks are removed from the host.
    pub fn poll_task(&self, id: TaskId) -> TaskPoll {
        let mut tasks = self.tasks.lock().expect("task registry lock poisoned");
        let Some((token, receiver)) = tasks.get_mut(&id) else {
            return TaskPoll::Unknown;
        };
        if token.is_cancelled() {
            tasks.remove(&id);
            return TaskPoll::Cancelled;
        }
        match receiver.try_recv() {
            Ok(result) => {
                tasks.remove(&id);
                TaskPoll::Ready(result)
            }
            Err(oneshot::error::TryRecvError::Empty) => TaskPoll::Pending,
            Err(oneshot::error::TryRecvError::Closed) => {
                tasks.remove(&id);
                TaskPoll::Ready(Err(VsError::Cancelled))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Pin;
    use crate::registry::{ExecNext, FnRunner, NodeDefinition, Purity, RunnerKind};

    #[tokio::test]
    async fn host_polls_completed_task() {
        let host = PrismHost::new();
        let id = host.spawn_task(|_| async { Ok(Value::I64(7)) });
        tokio::task::yield_now().await;
        assert!(matches!(
            host.poll_task(id),
            TaskPoll::Ready(Ok(Value::I64(7)))
        ));
        assert!(matches!(host.poll_task(id), TaskPoll::Unknown));
    }

    #[tokio::test]
    async fn host_cancels_task_before_poll() {
        let host = PrismHost::new();
        let id = host.spawn_task(|cancel| async move {
            tokio::select! {
                _ = cancel.cancelled() => Err(VsError::Cancelled),
                _ = tokio::time::sleep(std::time::Duration::from_secs(60)) => Ok(Value::Unit),
            }
        });
        assert!(host.cancel_task(id));
        assert!(matches!(host.poll_task(id), TaskPoll::Cancelled));
    }

    #[test]
    fn host_registers_node_in_concurrent_registry() {
        let host = PrismHost::new();
        let definition = NodeDefinition {
            id: "test.node".into(),
            purity: Purity::Pure,
            inputs: Vec::<Pin>::new(),
            outputs: Vec::<Pin>::new(),
            runner: RunnerKind::sync(FnRunner(|_, _, _| Ok(ExecNext::End))),
        };
        let identity = host.register_node(definition);
        assert!(identity.starts_with("test.node@"));
        assert!(host.has_node("test.node"));
    }
}
