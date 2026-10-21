//! Prism editor application entry point for strict headless and native runs.

#![allow(
    clippy::print_stdout,
    reason = "the command-line application emits deterministic run reports"
)]

use std::{
    ffi::{OsStr, OsString},
    fmt,
    path::PathBuf,
    process::ExitCode,
};

use prism_editor_app::{EditorAppConfig, EditorAppError, EditorApplication};

#[cfg(feature = "desktop")]
use std::time::Duration;

#[cfg(feature = "desktop")]
use prism_editor_app::{
    run_desktop_with_document, DesktopConfig, DesktopError, DesktopScene, DesktopUiError,
    EditorController, EditorDesktopUi, SceneProject, SceneProjectError,
};

const DEFAULT_FRAMES: u64 = 1;
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 800;
const MAX_DESKTOP_DIMENSION: u32 = 32_768;

fn main() -> ExitCode {
    match run(std::env::args_os().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("prism_editor_app: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: impl IntoIterator<Item = OsString>) -> Result<(), RunError> {
    let options = CliOptions::parse(args)?;
    if options.desktop {
        #[cfg(feature = "desktop")]
        {
            return run_native(options);
        }
        #[cfg(not(feature = "desktop"))]
        {
            return Err(RunError::DesktopUnavailable);
        }
    }
    run_headless(options.frames)
}

fn run_headless(frames: u64) -> Result<(), RunError> {
    let mut app = EditorApplication::new(EditorAppConfig::default())?;
    let report = app.run_headless(frames, |_| {})?;
    println!(
        "frames={} documents={} selection={} consoles={} viewports={} assets={} checksum={}",
        report.frames,
        report.snapshot.document_count,
        report.snapshot.selection_count,
        report.snapshot.console_count,
        report.snapshot.viewport_count,
        report.snapshot.asset_count,
        report.snapshot.checksum,
    );
    Ok(())
}

#[cfg(feature = "desktop")]
fn run_native(options: CliOptions) -> Result<(), RunError> {
    let (ui, document) = match options.project {
        Some(ProjectAction::Create(path)) => {
            let opened = SceneProject::create(path)?;
            let (ui, document) = opened.into_desktop();
            (ui, Some(document))
        }
        Some(ProjectAction::Open(path)) => {
            let opened = SceneProject::open(path)?;
            let (ui, document) = opened.into_desktop();
            (ui, Some(document))
        }
        None => {
            let mut ui = EditorDesktopUi::new(EditorController::new())?;
            ui.mount_content_browser(prism_editor_content::ContentCatalog::new())?;
            (ui, None)
        }
    };
    // The production ECS-to-scene adapter is a separate Editor slice. This
    // host starts with a valid empty renderer and never invents project types.
    let scene = DesktopScene::new(Vec::new())?;
    let outcome = run_desktop_with_document(
        DesktopConfig {
            logical_size: [options.width, options.height],
            frame_limit: Some(options.frames),
            time_limit: Some(Duration::from_secs(5)),
            ..DesktopConfig::default()
        },
        ui,
        scene,
        document,
    );
    println!(
        "native_frames={} native_input_events={} adapter={} backend={} error={:?}",
        outcome.report.presented_frames,
        outcome.report.native_input_events,
        outcome.report.adapter_name,
        outcome.report.backend,
        outcome.error,
    );
    if let Some(error) = outcome.error {
        return Err(error.into());
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CliOptions {
    desktop: bool,
    frames: u64,
    width: u32,
    height: u32,
    project: Option<ProjectAction>,
}

impl CliOptions {
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, CliError> {
        let mut options = Self {
            desktop: false,
            frames: DEFAULT_FRAMES,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            project: None,
        };
        let mut frames_seen = false;
        let mut width_seen = false;
        let mut height_seen = false;
        let mut arguments = args.into_iter();
        while let Some(argument) = arguments.next() {
            if argument == "--desktop" {
                if options.desktop {
                    return Err(CliError::Duplicate("--desktop"));
                }
                options.desktop = true;
            } else if argument == "--frames" {
                if frames_seen {
                    return Err(CliError::Duplicate("--frames"));
                }
                frames_seen = true;
                options.frames =
                    parse_positive_u64("--frames", required_value("--frames", arguments.next())?)?;
            } else if argument == "--width" {
                if width_seen {
                    return Err(CliError::Duplicate("--width"));
                }
                width_seen = true;
                options.width =
                    parse_dimension("--width", required_value("--width", arguments.next())?)?;
            } else if argument == "--height" {
                if height_seen {
                    return Err(CliError::Duplicate("--height"));
                }
                height_seen = true;
                options.height =
                    parse_dimension("--height", required_value("--height", arguments.next())?)?;
            } else if argument == "--create-project" {
                let path = PathBuf::from(required_value("--create-project", arguments.next())?);
                set_project(&mut options.project, ProjectAction::Create(path))?;
            } else if argument == "--open-project" {
                let path = PathBuf::from(required_value("--open-project", arguments.next())?);
                set_project(&mut options.project, ProjectAction::Open(path))?;
            } else {
                return Err(CliError::Unknown(argument));
            }
        }
        if options.project.is_some() && !options.desktop {
            return Err(CliError::RequiresDesktop("project options"));
        }
        if (width_seen || height_seen) && !options.desktop {
            return Err(CliError::RequiresDesktop("--width and --height"));
        }
        Ok(options)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ProjectAction {
    Create(PathBuf),
    Open(PathBuf),
}

fn required_value(name: &'static str, value: Option<OsString>) -> Result<OsString, CliError> {
    match value {
        Some(value) if !value.is_empty() && !looks_like_option(&value) => Ok(value),
        Some(value) => Err(CliError::MissingValueBefore { name, next: value }),
        None => Err(CliError::MissingValue(name)),
    }
}

fn looks_like_option(value: &OsStr) -> bool {
    value.as_encoded_bytes().starts_with(b"--")
}

fn parse_positive_u64(name: &'static str, value: OsString) -> Result<u64, CliError> {
    let parsed = value
        .to_str()
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| CliError::InvalidNumber {
            name,
            value: value.clone(),
        })?;
    if parsed == 0 {
        return Err(CliError::OutOfRange {
            name,
            minimum: 1,
            maximum: u64::MAX,
            value: parsed,
        });
    }
    Ok(parsed)
}

fn parse_dimension(name: &'static str, value: OsString) -> Result<u32, CliError> {
    let parsed = value
        .to_str()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| CliError::InvalidNumber {
            name,
            value: value.clone(),
        })?;
    if !(1..=MAX_DESKTOP_DIMENSION).contains(&parsed) {
        return Err(CliError::OutOfRange {
            name,
            minimum: 1,
            maximum: u64::from(MAX_DESKTOP_DIMENSION),
            value: u64::from(parsed),
        });
    }
    Ok(parsed)
}

fn set_project(
    destination: &mut Option<ProjectAction>,
    action: ProjectAction,
) -> Result<(), CliError> {
    if destination.is_some() {
        return Err(CliError::ProjectConflict);
    }
    *destination = Some(action);
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
enum CliError {
    Unknown(OsString),
    Duplicate(&'static str),
    MissingValue(&'static str),
    MissingValueBefore {
        name: &'static str,
        next: OsString,
    },
    InvalidNumber {
        name: &'static str,
        value: OsString,
    },
    OutOfRange {
        name: &'static str,
        minimum: u64,
        maximum: u64,
        value: u64,
    },
    ProjectConflict,
    RequiresDesktop(&'static str),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(argument) => {
                write!(formatter, "unknown argument `{}`", argument.to_string_lossy())
            }
            Self::Duplicate(name) => write!(formatter, "argument `{name}` was provided twice"),
            Self::MissingValue(name) => write!(formatter, "argument `{name}` requires a value"),
            Self::MissingValueBefore { name, next } => write!(
                formatter,
                "argument `{name}` requires a value before `{}`",
                next.to_string_lossy()
            ),
            Self::InvalidNumber { name, value } => write!(
                formatter,
                "argument `{name}` has invalid integer value `{}`",
                value.to_string_lossy()
            ),
            Self::OutOfRange {
                name,
                minimum,
                maximum,
                value,
            } => write!(
                formatter,
                "argument `{name}` value {value} is outside {minimum}..={maximum}"
            ),
            Self::ProjectConflict => formatter.write_str(
                "exactly one project action is allowed; --create-project and --open-project conflict",
            ),
            Self::RequiresDesktop(name) => write!(formatter, "{name} require --desktop"),
        }
    }
}

impl std::error::Error for CliError {}

#[derive(Debug)]
enum RunError {
    Cli(CliError),
    Application(EditorAppError),
    #[cfg(feature = "desktop")]
    Desktop(DesktopError),
    #[cfg(feature = "desktop")]
    Ui(DesktopUiError),
    #[cfg(feature = "desktop")]
    Project(SceneProjectError),
    #[cfg(not(feature = "desktop"))]
    DesktopUnavailable,
}

impl fmt::Display for RunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cli(error) => error.fmt(formatter),
            Self::Application(error) => error.fmt(formatter),
            #[cfg(feature = "desktop")]
            Self::Desktop(error) => error.fmt(formatter),
            #[cfg(feature = "desktop")]
            Self::Ui(error) => error.fmt(formatter),
            #[cfg(feature = "desktop")]
            Self::Project(error) => error.fmt(formatter),
            #[cfg(not(feature = "desktop"))]
            Self::DesktopUnavailable => {
                formatter.write_str("desktop support is not enabled in this build")
            }
        }
    }
}

impl std::error::Error for RunError {}

impl From<CliError> for RunError {
    fn from(value: CliError) -> Self {
        Self::Cli(value)
    }
}

impl From<EditorAppError> for RunError {
    fn from(value: EditorAppError) -> Self {
        Self::Application(value)
    }
}

#[cfg(feature = "desktop")]
impl From<DesktopError> for RunError {
    fn from(value: DesktopError) -> Self {
        Self::Desktop(value)
    }
}

#[cfg(feature = "desktop")]
impl From<DesktopUiError> for RunError {
    fn from(value: DesktopUiError) -> Self {
        Self::Ui(value)
    }
}

#[cfg(feature = "desktop")]
impl From<SceneProjectError> for RunError {
    fn from(value: SceneProjectError) -> Self {
        Self::Project(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> Result<CliOptions, CliError> {
        CliOptions::parse(arguments.iter().map(|argument| OsString::from(*argument)))
    }

    #[test]
    fn defaults_are_headless_and_bounded() {
        assert_eq!(
            parse(&[]),
            Ok(CliOptions {
                desktop: false,
                frames: DEFAULT_FRAMES,
                width: DEFAULT_WIDTH,
                height: DEFAULT_HEIGHT,
                project: None,
            })
        );
    }

    #[test]
    fn parses_one_desktop_project_action_and_dimensions() {
        assert_eq!(
            parse(&[
                "--desktop",
                "--create-project",
                "sample.prism",
                "--frames",
                "3",
                "--width",
                "1920",
                "--height",
                "1080",
            ]),
            Ok(CliOptions {
                desktop: true,
                frames: 3,
                width: 1920,
                height: 1080,
                project: Some(ProjectAction::Create(PathBuf::from("sample.prism"))),
            })
        );
    }

    #[test]
    fn rejects_unknown_missing_and_duplicate_arguments() {
        assert!(matches!(parse(&["positional"]), Err(CliError::Unknown(_))));
        assert_eq!(
            parse(&["--frames"]),
            Err(CliError::MissingValue("--frames"))
        );
        assert!(matches!(
            parse(&["--frames", "--desktop"]),
            Err(CliError::MissingValueBefore {
                name: "--frames",
                ..
            })
        ));
        assert_eq!(
            parse(&["--frames", "1", "--frames", "2"]),
            Err(CliError::Duplicate("--frames"))
        );
        assert_eq!(
            parse(&["--desktop", "--desktop"]),
            Err(CliError::Duplicate("--desktop"))
        );
    }

    #[test]
    fn rejects_invalid_or_out_of_range_numbers() {
        assert!(matches!(
            parse(&["--frames", "many"]),
            Err(CliError::InvalidNumber {
                name: "--frames",
                ..
            })
        ));
        assert!(matches!(
            parse(&["--frames", "0"]),
            Err(CliError::OutOfRange {
                name: "--frames",
                ..
            })
        ));
        assert!(matches!(
            parse(&["--desktop", "--width", "32769"]),
            Err(CliError::OutOfRange {
                name: "--width",
                ..
            })
        ));
    }

    #[test]
    fn rejects_conflicting_or_non_desktop_project_actions() {
        assert_eq!(
            parse(&["--desktop", "--create-project", "a", "--open-project", "b",]),
            Err(CliError::ProjectConflict)
        );
        assert_eq!(
            parse(&["--open-project", "a"]),
            Err(CliError::RequiresDesktop("project options"))
        );
        assert_eq!(
            parse(&["--width", "800"]),
            Err(CliError::RequiresDesktop("--width and --height"))
        );
    }
}
