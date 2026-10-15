//! Explicit host implementations of registered response programs.
//! This does not interpret SPIR-V: the application registers real CPU code
//! for a verified executable and owns the execution-table generation counter.
use crate::{response_output::ResponseOutput, LightingEnvironment, ShadingFrame, SurfaceSample};
use alloc::collections::BTreeMap;
use prism_render_material::{
    GpuMaterialHeader, MaterialExecutionBinding, MaterialExecutionTables, ProgramId, ProgramRef,
    ResponseProgramIndex,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuResponseError {
    StaleGeneration { expected: u64, actual: u64 },
    InvalidExecutionBinding,
    NotRegisteredProgram,
    DuplicateEvaluator(ProgramId),
    MissingEvaluator(ProgramId),
    ProgramMismatch,
    EvaluationFailed,
    InvalidOutput,
}

/// Inputs shared with builtin resolve, after geometry/generation validation,
/// textures, dynamic channels and normal mapping. The evaluator owns all
/// output channels, including emission; resolve does not add anything twice.
#[derive(Clone, Copy, Debug)]
pub struct CpuResponseContext<'a> {
    pub program_id: ProgramId,
    pub binding: MaterialExecutionBinding,
    pub execution_generation: u64,
    pub surface: &'a SurfaceSample,
    pub frame: ShadingFrame,
    pub position: [f32; 3],
    pub lighting: LightingEnvironment<'a>,
    pub screen_space_ao: f32,
}

pub trait CpuResponseEvaluator {
    fn evaluate(&self, context: CpuResponseContext<'_>)
        -> Result<ResponseOutput, CpuResponseError>;
}

impl<F> CpuResponseEvaluator for F
where
    F: for<'a> Fn(CpuResponseContext<'a>) -> Result<ResponseOutput, CpuResponseError>,
{
    fn evaluate(
        &self,
        context: CpuResponseContext<'_>,
    ) -> Result<ResponseOutput, CpuResponseError> {
        self(context)
    }
}

/// Registration is tied to an immutable verified table borrow. A rebuilt
/// table needs a new registry and generation, even if dense ids are reused.
pub struct CpuResponseRegistry<'a> {
    pub(crate) execution: &'a MaterialExecutionTables,
    generation: u64,
    evaluators: BTreeMap<ProgramId, &'a dyn CpuResponseEvaluator>,
}

impl<'a> CpuResponseRegistry<'a> {
    pub fn new(execution: &'a MaterialExecutionTables, generation: u64) -> Self {
        Self {
            execution,
            generation,
            evaluators: BTreeMap::new(),
        }
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn register(
        &mut self,
        program: ResponseProgramIndex,
        evaluator: &'a dyn CpuResponseEvaluator,
    ) -> Result<(), CpuResponseError> {
        let program = self
            .execution
            .program(program)
            .map_err(|_| CpuResponseError::InvalidExecutionBinding)?
            .program();
        let ProgramRef::Registered(id) = program else {
            return Err(CpuResponseError::NotRegisteredProgram);
        };
        if self.evaluators.contains_key(&id) {
            return Err(CpuResponseError::DuplicateEvaluator(id));
        }
        self.evaluators.insert(id, evaluator);
        Ok(())
    }

    pub(crate) fn validate_generation(&self, expected: u64) -> Result<(), CpuResponseError> {
        if expected != self.generation {
            return Err(CpuResponseError::StaleGeneration {
                expected,
                actual: self.generation,
            });
        }
        Ok(())
    }

    pub(crate) fn resolve(
        &self,
        header: &GpuMaterialHeader,
        expected: u64,
    ) -> Result<(MaterialExecutionBinding, &dyn CpuResponseEvaluator), CpuResponseError> {
        self.validate_generation(expected)?;
        let words = header.execution_binding_words();
        if words.contains(&GpuMaterialHeader::EXECUTION_BINDING_SENTINEL) {
            return Err(CpuResponseError::InvalidExecutionBinding);
        }
        let binding = self
            .execution
            .bind_raw(
                words[0],
                words[1],
                words[2],
                u64::from(words[3]) | (u64::from(words[4]) << 32),
            )
            .map_err(|_| CpuResponseError::InvalidExecutionBinding)?;
        let ProgramRef::Registered(id) = self
            .execution
            .program(binding.response_program_index)
            .map_err(|_| CpuResponseError::InvalidExecutionBinding)?
            .program()
        else {
            return Err(CpuResponseError::NotRegisteredProgram);
        };
        let evaluator = self
            .evaluators
            .get(&id)
            .copied()
            .ok_or(CpuResponseError::MissingEvaluator(id))?;
        Ok((binding, evaluator))
    }
}
