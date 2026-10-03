//! Common imports: `use prism_diagnostic::prelude::*;`.

pub use crate::filter::{max_level, set_max_level};
pub use crate::model::{Event, Field, FieldValue, Level};
pub use crate::sink::{set_sink, CaptureSink, ConsoleSink, FileSink, Sink};
pub use crate::{debug, error, event, info, trace, warn};
