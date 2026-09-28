//! Driving `slangc` to produce per-backend artifacts.
//!
//! This is the only module that shells out to the compiler. Everything is
//! expressed as an explicit [`CompileRequest`] so the argument vector is
//! auditable and deterministic; nothing here interpolates untrusted strings
//! into a shell (we invoke the binary directly, never via `sh -c`).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{SlangError, SlangResult};
use crate::target::{ShaderStage, Target};
use crate::toolchain::Slangc;

/// A preprocessor define passed to `slangc` as `-D<name>[=<value>]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Define {
    /// Macro name.
    pub name: String,
    /// Optional macro value.
    pub value: Option<String>,
}

impl Define {
    /// A bare `-Dname` flag.
    pub fn flag(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: None,
        }
    }

    /// A `-Dname=value` flag.
    pub fn valued(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: Some(value.into()),
        }
    }

    /// Render as the exact `slangc` argument token.
    pub fn as_arg(&self) -> String {
        match &self.value {
            Some(value) => format!("-D{}={}", self.name, value),
            None => format!("-D{}", self.name),
        }
    }
}

/// A fully specified single-target compilation.
#[derive(Debug, Clone)]
pub struct CompileRequest {
    /// Path to the `.slang` source module.
    pub source: PathBuf,
    /// Entry-point name (for example `computeMain`).
    pub entry: String,
    /// Shader stage of the entry point.
    pub stage: ShaderStage,
    /// Backend target.
    pub target: Target,
    /// Specialization / feature defines.
    pub defines: Vec<Define>,
    /// Where the compiled artifact should be written.
    pub output: PathBuf,
    /// Optional reflection-JSON output path.
    pub reflection: Option<PathBuf>,
}

impl CompileRequest {
    /// Construct a minimal request; refine with the builder-style setters.
    pub fn new(
        source: impl Into<PathBuf>,
        entry: impl Into<String>,
        stage: ShaderStage,
        target: Target,
        output: impl Into<PathBuf>,
    ) -> Self {
        Self {
            source: source.into(),
            entry: entry.into(),
            stage,
            target,
            defines: Vec::new(),
            output: output.into(),
            reflection: None,
        }
    }

    /// Attach a define.
    pub fn with_define(mut self, define: Define) -> Self {
        self.defines.push(define);
        self
    }

    /// Request reflection JSON alongside the artifact.
    pub fn with_reflection(mut self, path: impl Into<PathBuf>) -> Self {
        self.reflection = Some(path.into());
        self
    }

    /// Build the ordered argument vector passed to `slangc`.
    ///
    /// Exposed (and unit-tested) independently of process execution so the
    /// command line can be verified without a compiler present.
    pub fn args(&self) -> Vec<String> {
        let mut args: Vec<String> = vec![
            self.source.to_string_lossy().into_owned(),
            "-target".to_string(),
            self.target.slangc_name().to_string(),
            "-stage".to_string(),
            self.stage.slangc_name().to_string(),
            "-entry".to_string(),
            self.entry.clone(),
        ];
        for define in &self.defines {
            args.push(define.as_arg());
        }
        if let Some(reflection) = &self.reflection {
            args.push("-reflection-json".to_string());
            args.push(reflection.to_string_lossy().into_owned());
        }
        args.push("-o".to_string());
        args.push(self.output.to_string_lossy().into_owned());
        args
    }
}

/// Artifacts produced by a successful compilation.
#[derive(Debug, Clone)]
pub struct CompileArtifacts {
    /// The primary output path (as requested).
    pub output: PathBuf,
    /// The reflection JSON path, if one was requested and produced.
    pub reflection: Option<PathBuf>,
}

/// Run a single [`CompileRequest`] through the located compiler.
pub fn compile(slangc: &Slangc, request: &CompileRequest) -> SlangResult<CompileArtifacts> {
    let output = Command::new(slangc.path())
        .args(request.args())
        .output()
        .map_err(|err| SlangError::io(slangc.path(), &err))?;

    if !output.status.success() {
        return Err(SlangError::CompileFailed {
            code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }

    ensure_exists(&request.output)?;
    if let Some(reflection) = &request.reflection {
        ensure_exists(reflection)?;
    }

    Ok(CompileArtifacts {
        output: request.output.clone(),
        reflection: request.reflection.clone(),
    })
}

fn ensure_exists(path: &Path) -> SlangResult<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(SlangError::MissingArtifact {
            path: path.to_path_buf(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn define_arg_rendering() {
        assert_eq!(Define::flag("USE_RT").as_arg(), "-DUSE_RT");
        assert_eq!(
            Define::valued("MAX_LAYERS", "4").as_arg(),
            "-DMAX_LAYERS=4"
        );
    }

    #[test]
    fn args_include_target_stage_entry_and_output_in_order() {
        let request = CompileRequest::new(
            "closure.slang",
            "computeMain",
            ShaderStage::Compute,
            Target::Wgsl,
            "out.wgsl",
        );
        let args = request.args();
        assert_eq!(args[0], "closure.slang");
        assert_eq!(args[1], "-target");
        assert_eq!(args[2], "wgsl");
        assert_eq!(args[3], "-stage");
        assert_eq!(args[4], "compute");
        assert_eq!(args[5], "-entry");
        assert_eq!(args[6], "computeMain");
        assert_eq!(args[args.len() - 2], "-o");
        assert_eq!(args[args.len() - 1], "out.wgsl");
    }

    #[test]
    fn args_include_defines_and_reflection() {
        let request = CompileRequest::new(
            "closure.slang",
            "computeMain",
            ShaderStage::Compute,
            Target::Spirv,
            "out.spv",
        )
        .with_define(Define::flag("USE_RT"))
        .with_reflection("out.reflect.json");
        let args = request.args();
        assert!(args.iter().any(|a| a == "-DUSE_RT"));
        assert!(args.iter().any(|a| a == "-reflection-json"));
        assert!(args.iter().any(|a| a == "out.reflect.json"));
    }
}
