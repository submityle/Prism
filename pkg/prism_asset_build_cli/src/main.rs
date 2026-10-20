//! Command-line entry point for the local Prism asset-build vertical slice.
//!
//! The CLI deliberately delegates identity, graph execution, caching, and
//! publication to `prism_asset_build_core`; it contains no second build
//! implementation or format-specific cache logic.

mod args;
mod cache;
mod commands;
mod editable;
mod error;
mod mapping;
mod output;
mod patch;
mod pipeline;
mod query;

use argh::FromArgs;
use std::ffi::OsString;
use std::io::{self, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            // A broken output stream cannot carry its own terminal event. Do
            // not rerun a command: publication may already have committed.
            let _ = writeln!(
                io::stderr().lock(),
                "prism asset build output failed: {error}"
            );
            ExitCode::FAILURE
        }
    }
}

fn run() -> io::Result<u8> {
    let arguments: Vec<OsString> = std::env::args_os().skip(1).collect();
    // --json is a top-level switch, before the subcommand. Inspect only that
    // prefix so an option value spelled "--json" never enables output mode.
    let json = arguments
        .iter()
        .take_while(|arg| {
            arg.to_str()
                .is_some_and(|value| value.starts_with('-') && value != "--")
        })
        .any(|arg| arg == "--json");
    let hint = command_hint(&arguments);
    let stdout = io::stdout();
    let mut output = output::Output::new(stdout.lock(), json);
    output.started(hint)?;

    let Some(strings) = arguments
        .iter()
        .map(|arg| arg.to_str())
        .collect::<Option<Vec<_>>>()
    else {
        output.failed(
            hint,
            "invalid_arguments",
            "command-line arguments must be valid UTF-8",
            2,
        )?;
        return Ok(2);
    };
    let cli = match args::PrismAssetBuild::from_args(&["prism_asset_build_cli"], &strings) {
        Ok(cli) => cli,
        Err(early) => {
            if early.status.is_ok() {
                output.succeeded(hint, &output::CommandResult::Help { help: early.output })?;
                return Ok(0);
            }
            output.failed(hint, "invalid_arguments", &early.output, 2)?;
            return Ok(2);
        }
    };
    debug_assert_eq!(json, cli.json);
    let command = Some(cli.command.name());
    match commands::run(cli.command) {
        Ok(result) => {
            output.succeeded(command, &result)?;
            Ok(0)
        }
        Err(error) => {
            output.failed(command, error.code(), &error.to_string(), 1)?;
            Ok(1)
        }
    }
}

fn command_hint(arguments: &[OsString]) -> Option<&'static str> {
    let mut remaining = arguments.iter().skip_while(|arg| *arg == "--json");
    let first = remaining.next()?;
    let command = if first == "--" {
        remaining.next()?
    } else {
        first
    };
    match command.to_str()? {
        "plan" => Some("plan"),
        "editable-plan" => Some("editable-plan"),
        "editable-build" => Some("editable-build"),
        "import" => Some("import"),
        "build" => Some("build"),
        "package" => Some("package"),
        "patch" => Some("patch"),
        "validate" => Some("validate"),
        "inspect" => Some("inspect"),
        "why" => Some("why"),
        "graph" => Some("graph"),
        "diff" => Some("diff"),
        "cache" => Some("cache"),
        _ => None,
    }
}
