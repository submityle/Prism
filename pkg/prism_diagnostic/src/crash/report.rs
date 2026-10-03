//! The crash-report container and its deterministic binary writer (`crash`
//! feature, design §12 / §21, M6).
//!
//! [`CrashReport`] wraps a [`CrashContext`] with a magic + format version and
//! serializes to a self-contained little-endian byte stream via the crate's
//! hand-rolled [`wire`](crate::wire) codec (no `serde`, no external minidump
//! crate). The encoding is fully ordered and allocation-order-preserving, so
//! the same report always produces identical bytes — a versioned on-disk
//! contract. Every decode path is total: a truncated or corrupt stream yields a
//! [`WireError`] instead of panicking.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::wire::{ByteReader, ByteWriter, WireError};

use super::context::{
    CrashContext, CrashReason, ModuleEntry, RegisterSnapshot, StackFrame, ThreadContext,
};

/// Magic marker (`"PCR1"`, little-endian) prefixing every serialized report.
pub const CRASH_REPORT_MAGIC: u32 = 0x3152_4350;

/// Current crash-report wire format version.
pub const CRASH_REPORT_VERSION: u16 = 1;

const REASON_SIGNAL: u8 = 1;
const REASON_EXCEPTION: u8 = 2;
const REASON_ABORT: u8 = 3;
const REASON_ASSERTION: u8 = 4;
const REASON_OOM: u8 = 5;
const REASON_OTHER: u8 = 6;

/// A versioned, serializable crash report wrapping a [`CrashContext`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrashReport {
    /// Format version the report was produced with.
    pub format_version: u16,
    /// Captured crash context.
    pub context: CrashContext,
}

impl CrashReport {
    /// Wrap `context` in a report stamped with the current format version.
    pub fn new(context: CrashContext) -> Self {
        Self {
            format_version: CRASH_REPORT_VERSION,
            context,
        }
    }

    /// Serialize to a self-contained byte stream (magic + version + context).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        w.put_u32(CRASH_REPORT_MAGIC);
        w.put_u16(self.format_version);
        write_context(&mut w, &self.context);
        w.into_vec()
    }

    /// Parse a report from bytes produced by [`Self::to_bytes`].
    ///
    /// Validates the magic and rejects a format version newer than
    /// [`CRASH_REPORT_VERSION`]; both surface as [`WireError::InvalidValue`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WireError> {
        let mut r = ByteReader::new(bytes);
        let magic = r.get_u32()?;
        if magic != CRASH_REPORT_MAGIC {
            return Err(WireError::InvalidValue);
        }
        let format_version = r.get_u16()?;
        if format_version == 0 || format_version > CRASH_REPORT_VERSION {
            return Err(WireError::InvalidValue);
        }
        let context = read_context(&mut r)?;
        Ok(Self {
            format_version,
            context,
        })
    }

    /// Write the serialized report to `writer` (`std` only).
    #[cfg(feature = "std")]
    pub fn write_to<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.to_bytes())
    }

    /// Write the serialized report to a file at `path` (`std` only).
    #[cfg(feature = "std")]
    pub fn write_to_path<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_to(&mut file)
    }

    /// Read and parse a report from a file at `path` (`std` only).
    #[cfg(feature = "std")]
    pub fn read_from_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
    }
}

fn write_opt_str(w: &mut ByteWriter, value: &Option<String>) {
    match value {
        Some(text) => {
            w.put_bool(true);
            w.put_str(text);
        }
        None => w.put_bool(false),
    }
}

fn read_opt_str(r: &mut ByteReader<'_>) -> Result<Option<String>, WireError> {
    if r.get_bool()? {
        Ok(Some(r.get_str()?))
    } else {
        Ok(None)
    }
}

fn write_reason(w: &mut ByteWriter, reason: &Option<CrashReason>) {
    match reason {
        None => w.put_u8(0),
        Some(CrashReason::Signal(sig)) => {
            w.put_u8(REASON_SIGNAL);
            w.put_i64(i64::from(*sig));
        }
        Some(CrashReason::Exception(code)) => {
            w.put_u8(REASON_EXCEPTION);
            w.put_u32(*code);
        }
        Some(CrashReason::Abort) => w.put_u8(REASON_ABORT),
        Some(CrashReason::Assertion) => w.put_u8(REASON_ASSERTION),
        Some(CrashReason::OutOfMemory) => w.put_u8(REASON_OOM),
        Some(CrashReason::Other(text)) => {
            w.put_u8(REASON_OTHER);
            w.put_str(text);
        }
    }
}

fn read_reason(r: &mut ByteReader<'_>) -> Result<Option<CrashReason>, WireError> {
    match r.get_u8()? {
        0 => Ok(None),
        REASON_SIGNAL => {
            let sig = i32::try_from(r.get_i64()?).map_err(|_| WireError::InvalidValue)?;
            Ok(Some(CrashReason::Signal(sig)))
        }
        REASON_EXCEPTION => Ok(Some(CrashReason::Exception(r.get_u32()?))),
        REASON_ABORT => Ok(Some(CrashReason::Abort)),
        REASON_ASSERTION => Ok(Some(CrashReason::Assertion)),
        REASON_OOM => Ok(Some(CrashReason::OutOfMemory)),
        REASON_OTHER => Ok(Some(CrashReason::Other(r.get_str()?))),
        other => Err(WireError::InvalidTag(other)),
    }
}

fn write_registers(w: &mut ByteWriter, regs: &RegisterSnapshot) {
    w.put_u64(regs.instruction_pointer);
    w.put_u64(regs.stack_pointer);
    w.put_u64(regs.frame_pointer);
    w.put_u32(regs.general.len() as u32);
    for (name, value) in &regs.general {
        w.put_str(name);
        w.put_u64(*value);
    }
}

fn read_registers(r: &mut ByteReader<'_>) -> Result<RegisterSnapshot, WireError> {
    let instruction_pointer = r.get_u64()?;
    let stack_pointer = r.get_u64()?;
    let frame_pointer = r.get_u64()?;
    let count = r.get_u32()? as usize;
    let mut general = Vec::with_capacity(count);
    for _ in 0..count {
        let name = r.get_str()?;
        let value = r.get_u64()?;
        general.push((name, value));
    }
    Ok(RegisterSnapshot {
        instruction_pointer,
        stack_pointer,
        frame_pointer,
        general,
    })
}

fn write_frame(w: &mut ByteWriter, frame: &StackFrame) {
    w.put_u32(frame.index);
    w.put_u64(frame.instruction_pointer);
    w.put_u64(frame.module_offset);
    write_opt_str(w, &frame.symbol);
    write_opt_str(w, &frame.module);
}

fn read_frame(r: &mut ByteReader<'_>) -> Result<StackFrame, WireError> {
    Ok(StackFrame {
        index: r.get_u32()?,
        instruction_pointer: r.get_u64()?,
        module_offset: r.get_u64()?,
        symbol: read_opt_str(r)?,
        module: read_opt_str(r)?,
    })
}

fn write_module(w: &mut ByteWriter, module: &ModuleEntry) {
    w.put_str(&module.name);
    w.put_u64(module.base_address);
    w.put_u64(module.size);
    write_opt_str(w, &module.build_id);
}

fn read_module(r: &mut ByteReader<'_>) -> Result<ModuleEntry, WireError> {
    Ok(ModuleEntry {
        name: r.get_str()?,
        base_address: r.get_u64()?,
        size: r.get_u64()?,
        build_id: read_opt_str(r)?,
    })
}

fn write_thread(w: &mut ByteWriter, thread: &ThreadContext) {
    w.put_u64(thread.thread_id);
    write_opt_str(w, &thread.thread_name);
    write_registers(w, &thread.registers);
    w.put_u32(thread.frames.len() as u32);
    for frame in &thread.frames {
        write_frame(w, frame);
    }
}

fn read_thread(r: &mut ByteReader<'_>) -> Result<ThreadContext, WireError> {
    let thread_id = r.get_u64()?;
    let thread_name = read_opt_str(r)?;
    let registers = read_registers(r)?;
    let count = r.get_u32()? as usize;
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        frames.push(read_frame(r)?);
    }
    Ok(ThreadContext {
        thread_id,
        thread_name,
        registers,
        frames,
    })
}

fn write_context(w: &mut ByteWriter, context: &CrashContext) {
    write_reason(w, &context.reason);
    write_thread(w, &context.crashing_thread);
    w.put_u32(context.other_threads.len() as u32);
    for thread in &context.other_threads {
        write_thread(w, thread);
    }
    w.put_u32(context.modules.len() as u32);
    for module in &context.modules {
        write_module(w, module);
    }
    w.put_u64(context.timestamp_nanos);
    w.put_str(&context.build_id);
    w.put_u32(context.notes.len() as u32);
    for (key, value) in &context.notes {
        w.put_str(key);
        w.put_str(value);
    }
}

fn read_context(r: &mut ByteReader<'_>) -> Result<CrashContext, WireError> {
    let reason = read_reason(r)?;
    let crashing_thread = read_thread(r)?;

    let other_count = r.get_u32()? as usize;
    let mut other_threads = Vec::with_capacity(other_count);
    for _ in 0..other_count {
        other_threads.push(read_thread(r)?);
    }

    let module_count = r.get_u32()? as usize;
    let mut modules = Vec::with_capacity(module_count);
    for _ in 0..module_count {
        modules.push(read_module(r)?);
    }

    let timestamp_nanos = r.get_u64()?;
    let build_id = r.get_str()?;

    let note_count = r.get_u32()? as usize;
    let mut notes = Vec::with_capacity(note_count);
    for _ in 0..note_count {
        let key = r.get_str()?;
        let value = r.get_str()?;
        notes.push((key, value));
    }

    Ok(CrashContext {
        reason,
        crashing_thread,
        other_threads,
        modules,
        timestamp_nanos,
        build_id,
        notes,
    })
}
