//! Prism editor application entry point for strict headless and native runs.

#![allow(
    clippy::print_stdout,
    reason = "the command-line application emits deterministic run reports"
)]

use core::fmt;
use std::{
    ffi::{OsStr, OsString},
    io::Write,
    path::PathBuf,
    process::ExitCode,
};

use prism_editor_app::{EditorAppConfig, EditorAppError, EditorApplication};

#[cfg(any(feature = "desktop", test))]
use std::time::Duration;

#[cfg(feature = "desktop")]
use prism_editor_app::{
    editor_scene_mesh_catalog, install_scene_authoring_commands, run_desktop_with_document,
    DesktopConfig, DesktopError, DesktopUiError, EditorController, EditorDesktopUi,
    EditorSceneSession, SceneProject, SceneProjectConfig, SceneProjectError,
};

const DEFAULT_HEADLESS_FRAMES: u64 = 1;
#[cfg(any(feature = "desktop", test))]
const DEFAULT_SMOKE_TIME_LIMIT_SECONDS: u64 = 30;
const MAX_TIME_LIMIT_SECONDS: u64 = 86_400;
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 800;
const MAX_DESKTOP_DIMENSION: u32 = 32_768;

fn main() -> ExitCode {
    match run(std::env::args_os().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // An unavailable stderr must still produce a failing process exit.
            let _ = writeln!(std::io::stderr().lock(), "prism_editor_app: {error}");
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
    run_headless(options.frames.unwrap_or(DEFAULT_HEADLESS_FRAMES))
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
    validate_frame_limit(Some(frames), report.frames)
}

#[cfg(feature = "desktop")]
fn run_native(options: CliOptions) -> Result<(), RunError> {
    let time_limit = options.desktop_time_limit();
    let (mut ui, document) = match options.project {
        Some(ProjectAction::Create(path)) => {
            let opened =
                SceneProject::create_with(path, scene_catalog()?, SceneProjectConfig::default())?;
            let (ui, document) = opened.into_desktop();
            (ui, Some(document))
        }
        Some(ProjectAction::Open(path)) => {
            let opened =
                SceneProject::open_with(path, scene_catalog()?, SceneProjectConfig::default())?;
            let (ui, document) = opened.into_desktop();
            (ui, Some(document))
        }
        None => {
            let mut ui = EditorDesktopUi::new(EditorController::new())?;
            ui.mount_content_browser(prism_editor_content::ContentCatalog::new())?;
            (ui, None)
        }
    };
    install_scene_authoring_commands(ui.controller_mut()).map_err(DesktopUiError::from)?;
    ui.refresh()?;
    let scene = EditorSceneSession::new(ui.controller_mut())?;
    let outcome = run_desktop_with_document(
        DesktopConfig {
            logical_size: [options.width, options.height],
            frame_limit: options.frames,
            time_limit,
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
    validate_frame_limit(options.frames, outcome.report.presented_frames)
}

#[cfg(feature = "desktop")]
fn scene_catalog() -> Result<prism_editor_app::DocumentSchemaCatalog, RunError> {
    editor_scene_mesh_catalog().map_err(|error| DesktopError::Document(error.to_string()).into())
}

fn validate_frame_limit(limit: Option<u64>, completed: u64) -> Result<(), RunError> {
    if let Some(expected) = limit
        && completed < expected
    {
        return Err(RunError::FrameLimitNotReached {
            expected,
            completed,
        });
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CliOptions {
    desktop: bool,
    // None keeps desktop interactive; headless applies its own one-frame default.
    frames: Option<u64>,
    time_limit_seconds: Option<u64>,
    width: u32,
    height: u32,
    project: Option<ProjectAction>,
}

impl CliOptions {
    fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, CliError> {
        let mut options = Self {
            desktop: false,
            frames: None,
            time_limit_seconds: None,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            project: None,
        };
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
                if options.frames.is_some() {
                    return Err(CliError::Duplicate("--frames"));
                }
                options.frames = Some(parse_positive_u64(
                    "--frames",
                    required_value("--frames", arguments.next())?,
                    u64::MAX,
                )?);
            } else if argument == "--time-limit-seconds" {
                if options.time_limit_seconds.is_some() {
                    return Err(CliError::Duplicate("--time-limit-seconds"));
                }
                // Bound native deadlines even for representable u64 input.
                options.time_limit_seconds = Some(parse_positive_u64(
                    "--time-limit-seconds",
                    required_value("--time-limit-seconds", arguments.next())?,
                    MAX_TIME_LIMIT_SECONDS,
                )?);
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
        if options.time_limit_seconds.is_some() && !options.desktop {
            return Err(CliError::RequiresDesktop("--time-limit-seconds"));
        }
        Ok(options)
    }

    #[cfg(any(feature = "desktop", test))]
    fn desktop_time_limit(&self) -> Option<Duration> {
        self.time_limit_seconds
            // An explicit frame target must not wait forever when no frame
            // can be presented (for example, an occluded CI window).
            .or_else(|| self.frames.map(|_| DEFAULT_SMOKE_TIME_LIMIT_SECONDS))
            .map(Duration::from_secs)
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

fn parse_positive_u64(name: &'static str, value: OsString, maximum: u64) -> Result<u64, CliError> {
    let parsed = value
        .to_str()
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| CliError::InvalidNumber {
            name,
            value: value.clone(),
        })?;
    if !(1..=maximum).contains(&parsed) {
        return Err(CliError::OutOfRange {
            name,
            minimum: 1,
            maximum,
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
    FrameLimitNotReached {
        expected: u64,
        completed: u64,
    },
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
            Self::FrameLimitNotReached {
                expected,
                completed,
            } => write!(
                formatter,
                "frame limit not reached: expected {expected} frames, completed {completed}"
            ),
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
                frames: None,
                time_limit_seconds: None,
                width: DEFAULT_WIDTH,
                height: DEFAULT_HEIGHT,
                project: None,
            })
        );
        assert_eq!(
            parse(&[])
                .unwrap()
                .frames
                .unwrap_or(DEFAULT_HEADLESS_FRAMES),
            1
        );
        assert_eq!(parse(&["--frames", "7"]).unwrap().frames, Some(7));
    }

    #[test]
    fn desktop_defaults_are_interactive_without_a_deadline() {
        let options = parse(&["--desktop"]).unwrap();
        assert!(options.desktop);
        assert_eq!(options.frames, None);
        assert_eq!(options.time_limit_seconds, None);
        assert_eq!(options.desktop_time_limit(), None);
    }

    #[test]
    fn explicit_desktop_frames_always_have_a_watchdog() {
        for frames in ["1", "3", "18446744073709551615"] {
            let options = parse(&["--frames", frames, "--desktop"]).unwrap();
            assert_eq!(options.frames, Some(frames.parse::<u64>().unwrap()));
            assert_eq!(options.time_limit_seconds, None);
            assert_eq!(
                options.desktop_time_limit(),
                Some(Duration::from_secs(DEFAULT_SMOKE_TIME_LIMIT_SECONDS))
            );
        }
    }

    #[test]
    fn explicit_deadline_overrides_smoke_watchdog_in_any_argument_order() {
        for seconds in ["1", "60", "86400"] {
            for arguments in [
                [
                    "--desktop",
                    "--frames",
                    "3",
                    "--time-limit-seconds",
                    seconds,
                ],
                [
                    "--time-limit-seconds",
                    seconds,
                    "--frames",
                    "3",
                    "--desktop",
                ],
            ] {
                let options = parse(&arguments).unwrap();
                let seconds = seconds.parse::<u64>().unwrap();
                assert_eq!(options.frames, Some(3));
                assert_eq!(options.time_limit_seconds, Some(seconds));
                assert_eq!(
                    options.desktop_time_limit(),
                    Some(Duration::from_secs(seconds))
                );
            }
        }
    }

    #[test]
    fn explicit_deadline_does_not_imply_a_frame_target() {
        for seconds in ["1", "86400"] {
            let options = parse(&["--desktop", "--time-limit-seconds", seconds]).unwrap();
            assert_eq!(options.frames, None);
            assert_eq!(
                options.desktop_time_limit(),
                Some(Duration::from_secs(seconds.parse::<u64>().unwrap()))
            );
        }
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
                frames: Some(3),
                time_limit_seconds: None,
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
    fn rejects_missing_empty_duplicate_and_option_like_values() {
        for name in ["--frames", "--time-limit-seconds", "--width", "--height"] {
            assert_eq!(
                parse(&["--desktop", name]),
                Err(CliError::MissingValue(name))
            );
            for value in ["", "--desktop"] {
                assert_eq!(
                    parse(&[name, value]),
                    Err(CliError::MissingValueBefore {
                        name,
                        next: OsString::from(value),
                    })
                );
            }
            assert_eq!(
                parse(&["--desktop", name, "1", name, "2"]),
                Err(CliError::Duplicate(name))
            );
        }
    }

    #[test]
    fn rejects_invalid_zero_and_overflowing_limits() {
        for name in ["--frames", "--time-limit-seconds"] {
            for value in ["many", "-1", "1.5", " 1", "18446744073709551616"] {
                assert_eq!(
                    parse(&["--desktop", name, value]),
                    Err(CliError::InvalidNumber {
                        name,
                        value: OsString::from(value),
                    })
                );
            }
            assert_eq!(
                parse(&["--desktop", name, "0"]),
                Err(CliError::OutOfRange {
                    name,
                    minimum: 1,
                    maximum: if name == "--frames" {
                        u64::MAX
                    } else {
                        MAX_TIME_LIMIT_SECONDS
                    },
                    value: 0,
                })
            );
        }
        for value in [86_401, u64::MAX] {
            assert_eq!(
                parse(&["--desktop", "--time-limit-seconds", &value.to_string()]),
                Err(CliError::OutOfRange {
                    name: "--time-limit-seconds",
                    minimum: 1,
                    maximum: MAX_TIME_LIMIT_SECONDS,
                    value,
                })
            );
        }
    }

    #[test]
    fn headless_frame_parser_accepts_both_positive_u64_boundaries() {
        for (value, expected) in [("1", 1), ("18446744073709551615", u64::MAX)] {
            let options = parse(&["--frames", value]).unwrap();
            assert!(!options.desktop);
            assert_eq!(options.frames, Some(expected));
            assert_eq!(options.time_limit_seconds, None);
        }
    }

    #[test]
    fn rejects_invalid_zero_and_overflowing_dimensions() {
        for name in ["--width", "--height"] {
            for value in ["-1", "wide", "4294967296"] {
                assert_eq!(
                    parse(&["--desktop", name, value]),
                    Err(CliError::InvalidNumber {
                        name,
                        value: OsString::from(value),
                    })
                );
            }
            for value in [0, 32_769, u64::from(u32::MAX)] {
                assert_eq!(
                    parse(&["--desktop", name, &value.to_string()]),
                    Err(CliError::OutOfRange {
                        name,
                        minimum: 1,
                        maximum: u64::from(MAX_DESKTOP_DIMENSION),
                        value,
                    })
                );
            }
            for value in ["1", "32768"] {
                assert!(parse(&["--desktop", name, value]).is_ok());
            }
        }
    }

    #[test]
    fn rejects_time_limit_and_dimensions_outside_desktop_scope() {
        for arguments in [
            vec!["--time-limit-seconds", "2"],
            vec!["--frames", "1", "--time-limit-seconds", "2"],
        ] {
            assert_eq!(
                parse(&arguments),
                Err(CliError::RequiresDesktop("--time-limit-seconds"))
            );
        }
        for name in ["--width", "--height"] {
            assert_eq!(
                parse(&[name, "800"]),
                Err(CliError::RequiresDesktop("--width and --height"))
            );
        }
    }

    #[test]
    fn rejects_non_unicode_numbers_and_preserves_non_unicode_project_paths() {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;

            let value = OsString::from_vec(vec![0xff]);
            for name in ["--frames", "--time-limit-seconds", "--width", "--height"] {
                assert_eq!(
                    CliOptions::parse([OsString::from("--desktop"), name.into(), value.clone()]),
                    Err(CliError::InvalidNumber {
                        name,
                        value: value.clone(),
                    })
                );
            }
            let options = CliOptions::parse([
                OsString::from("--desktop"),
                OsString::from("--open-project"),
                value.clone(),
            ])
            .unwrap();
            assert_eq!(options.project, Some(ProjectAction::Open(value.into())));
        }
    }

    #[test]
    fn frame_target_must_be_reached_even_after_a_clean_native_close() {
        for completed in [0, 2] {
            let error = validate_frame_limit(Some(3), completed).unwrap_err();
            assert!(matches!(
                error,
                RunError::FrameLimitNotReached {
                    expected: 3,
                    completed: actual,
                } if actual == completed
            ));
            assert_eq!(
                error.to_string(),
                format!("frame limit not reached: expected 3 frames, completed {completed}")
            );
        }
        for completed in [3, 4, u64::MAX] {
            assert!(validate_frame_limit(Some(3), completed).is_ok());
        }
        assert!(validate_frame_limit(Some(u64::MAX), u64::MAX).is_ok());
        assert!(validate_frame_limit(None, 0).is_ok());
        assert!(validate_frame_limit(None, 10).is_ok());
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

    #[test]
    fn preserves_project_actions_and_strict_argument_grammar() {
        for name in ["--create-project", "--open-project"] {
            let options = parse(&[name, "sample project.prism", "--desktop"]).unwrap();
            let path = PathBuf::from("sample project.prism");
            assert_eq!(
                options.project,
                Some(if name == "--create-project" {
                    ProjectAction::Create(path)
                } else {
                    ProjectAction::Open(path)
                })
            );
            assert_eq!(options.frames, None);
            assert_eq!(options.desktop_time_limit(), None);
            assert_eq!(parse(&[name]), Err(CliError::MissingValue(name)));
            for value in ["", "--desktop"] {
                assert_eq!(
                    parse(&[name, value]),
                    Err(CliError::MissingValueBefore {
                        name,
                        next: value.into(),
                    })
                );
            }
            assert_eq!(
                parse(&["--desktop", name, "a", name, "b"]),
                Err(CliError::ProjectConflict)
            );
            let other = if name == "--create-project" {
                "--open-project"
            } else {
                "--create-project"
            };
            assert_eq!(
                parse(&["--desktop", name, "a", other, "b"]),
                Err(CliError::ProjectConflict)
            );
            assert_eq!(
                parse(&[name, "a"]),
                Err(CliError::RequiresDesktop("project options"))
            );
        }
        for argument in [
            "--help",
            "--",
            "--frames=3",
            "--time-limit-seconds=5",
            "--unknown",
        ] {
            assert_eq!(
                parse(&[argument]),
                Err(CliError::Unknown(OsString::from(argument)))
            );
        }
    }

    #[cfg(not(feature = "desktop"))]
    #[test]
    fn valid_desktop_requests_still_fail_without_the_feature() {
        assert!(matches!(
            run([OsString::from("--desktop")]),
            Err(RunError::DesktopUnavailable)
        ));
        assert!(matches!(
            run([OsString::from("--desktop"), "--frames".into(), "1".into()]),
            Err(RunError::DesktopUnavailable)
        ));
    }
}
