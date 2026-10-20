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
        Command::Graph(command) => crate::query::graph(command).map(CommandResult::Graph),
        Command::Diff(command) => crate::query::diff(command).map(CommandResult::Diff),
        Command::Cache(command) => match command.command {
            crate::args::CacheCommand::Status(command) => {
                crate::cache::status(&command.store, command.max_index_bytes)
                    .map(CommandResult::CacheStatus)
            }
            crate::args::CacheCommand::Verify(command) => {
                crate::cache::verify(&command.store, command.max_index_bytes)
                    .map(CommandResult::CacheVerify)
            }
            crate::args::CacheCommand::Prune(command) => {
                let capacity = prism_asset_build_core::DdcCapacity::new(
                    command.max_entries,
                    command.max_encoded_bytes,
                )?;
                crate::cache::prune(
                    &command.store,
                    capacity,
                    std::time::Duration::from_secs(command.grace_seconds),
                    command.max_index_bytes,
                )
                .map(CommandResult::CachePrune)
            }
        },
    }
}
