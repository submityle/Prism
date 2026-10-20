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
    resources: crate::resources::BuildResources,
    max_output_bytes: u64,
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
            resources: crate::resources::BuildResources::default(),
            max_output_bytes: u64::MAX,
        }
    }

    pub fn id(&self) -> JobId {
        self.id
    }

    /// Declare peak concurrent demand for resource-aware execution. This is
    /// scheduling policy and does not alter the semantic artifact key.
    pub fn with_resources(mut self, resources: crate::resources::BuildResources) -> Self {
        self.resources = resources;
        self
    }

    pub fn resources(&self) -> crate::resources::BuildResources {
        self.resources
    }

    /// Limits each encoded output, including verified cache hits and both
    /// determinism runs. Zero permits only an empty output. The default is
    /// `u64::MAX`, meaning no caller-specified bound. This does not constrain
    /// processor heap allocations or aggregate outputs, and does not change
    /// the artifact key; every cache reuse is checked against the current bound.
    pub fn with_max_output_bytes(mut self, max_output_bytes: u64) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    pub fn max_output_bytes(&self) -> u64 {
        self.max_output_bytes
    }

    pub fn dependencies(&self) -> &[JobId] {
        &self.dependencies
    }

    pub(crate) fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        let output = self.processor.process(inputs)?;
        if output.len() as u64 > self.max_output_bytes {
            return Err(ProcessorError::new(
                "processor output exceeds declared byte bound",
            ));
        }
        Ok(output)
    }

    pub(crate) fn processor(&self) -> Arc<dyn Processor> {
        Arc::clone(&self.processor)
    }
}
