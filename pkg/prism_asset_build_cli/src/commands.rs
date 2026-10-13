use crate::args::{Command, PrismAssetBuild};
use crate::error::CommandError;
use crate::pipeline;

/// Runs one parsed command.
pub fn run(cli: PrismAssetBuild) -> Result<(), CommandError> {
    match cli.command {
        Command::Plan(command) => pipeline::plan(command),
        Command::Import(command) => pipeline::publish(
            command.source,
            command.logical_path,
            command.asset_id,
            command.store,
        ),
        Command::Build(command) => pipeline::publish(
            command.source,
            command.logical_path,
            command.asset_id,
            command.store,
        ),
        Command::Package(command) => pipeline::package(command),
        Command::Validate(command) => pipeline::validate(command),
        Command::Inspect(command) => pipeline::inspect(command.store),
    }
}
