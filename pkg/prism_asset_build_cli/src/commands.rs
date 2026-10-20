use crate::args::Command;
use crate::error::CommandError;
use crate::output::CommandResult;
use crate::pipeline;

/// Runs one parsed command.
pub fn run(command: Command) -> Result<CommandResult, CommandError> {
    match command {
        Command::Plan(command) => pipeline::plan(command).map(CommandResult::Plan),
        Command::Import(command) => pipeline::publish(
            command.source,
            command.logical_path,
            command.asset_id,
            command.store,
            command.verify_determinism,
        )
        .map(CommandResult::Publish),
        Command::Build(command) => pipeline::publish(
            command.source,
            command.logical_path,
            command.asset_id,
            command.store,
            command.verify_determinism,
        )
        .map(CommandResult::Publish),
        Command::Package(command) => pipeline::package(command).map(CommandResult::Package),
        Command::Patch(command) => crate::patch::run(command).map(CommandResult::Patch),
        Command::Validate(command) => pipeline::validate(command).map(CommandResult::Validate),
        Command::Inspect(command) => pipeline::inspect(command.store).map(CommandResult::Inspect),
        Command::Why(command) => pipeline::why(command).map(CommandResult::Why),
    }
}
