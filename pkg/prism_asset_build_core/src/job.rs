use std::fmt;
use std::sync::Arc;

/// Stable identity used to order jobs and refer to dependencies in a build graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct JobId(pub u128);

/// Failure returned by a local processor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessorError {
    message: String,
}

impl ProcessorError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ProcessorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProcessorError {}

/// Synchronous local transformation. Inputs are passed in the exact order
/// declared by [`BuildJob::dependencies`]. A processor should be a pure
/// function of those bytes and its captured, immutable configuration: it must
/// not read ambient state or produce nondeterministic output.
///
/// Processors are shared by the serial and parallel graph executors. They must
/// be thread-safe and deterministic: stateful interior-mutability can still
/// violate reproducibility and is the implementor's responsibility to avoid.
pub trait Processor: Send + Sync {
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError>;
}

impl<F> Processor for F
where
    F: for<'a, 'b> Fn(&'a [&'b [u8]]) -> Result<Vec<u8>, ProcessorError> + Send + Sync,
{
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        self(inputs)
    }
}

/// A single output-producing graph job.
pub struct BuildJob {
    id: JobId,
    dependencies: Vec<JobId>,
    processor: Arc<dyn Processor>,
}

impl BuildJob {
    /// Create a job. Dependency order is semantically significant and is
    /// preserved when the processor receives its input byte slices.
    pub fn new<P>(id: JobId, dependencies: impl Into<Vec<JobId>>, processor: P) -> Self
    where
        P: Processor + 'static,
    {
        Self {
            id,
            dependencies: dependencies.into(),
            processor: Arc::new(processor),
        }
    }

    pub fn id(&self) -> JobId {
        self.id
    }

    pub fn dependencies(&self) -> &[JobId] {
        &self.dependencies
    }

    pub(crate) fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        self.processor.process(inputs)
    }

    pub(crate) fn processor(&self) -> Arc<dyn Processor> {
        Arc::clone(&self.processor)
    }
}
