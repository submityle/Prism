//! Version 1 diagnostic wire format, little endian and length prefixed.
use super::*;

const MAGIC: &[u8; 8] = b"PRFG\x01\0\0\0";
fn invalid() -> ProfilerError {
    ProfilerError::InvalidRange {
        field: "capture wire format",
    }
}

struct Writer {
    bytes: Option<Vec<u8>>,
    size: usize,
    limit: usize,
}
impl Writer {
    fn raw(&mut self, value: &[u8]) -> Result<(), ProfilerError> {
        self.size = self
            .size
            .checked_add(value.len())
            .ok_or(ProfilerError::ArithmeticOverflow)?;
        validate_limit(self.size, self.limit, "capture wire bytes")?;
        if let Some(bytes) = &mut self.bytes {
            bytes.extend_from_slice(value);
        }
        Ok(())
    }
    fn n(&mut self, value: u64) -> Result<(), ProfilerError> {
        self.raw(&value.to_le_bytes())
    }
    fn count(&mut self, value: usize) -> Result<(), ProfilerError> {
        self.n(u64::try_from(value).map_err(|_| ProfilerError::ArithmeticOverflow)?)
    }
    fn text(&mut self, value: &str) -> Result<(), ProfilerError> {
        self.count(value.len())?;
        self.raw(value.as_bytes())
    }
    fn option(&mut self, value: Option<u64>) -> Result<(), ProfilerError> {
        self.n(u64::from(value.is_some()))?;
        if let Some(value) = value {
            self.n(value)?;
        }
        Ok(())
    }
}
pub(super) fn encode(
    report: &GraphCaptureReport,
    limits: GraphCaptureLimits,
) -> Result<Vec<u8>, ProfilerError> {
    let mut counter = Writer {
        bytes: None,
        size: 0,
        limit: limits.wire_bytes,
    };
    write(report, &mut counter)?;
    let mut writer = Writer {
        bytes: Some(Vec::with_capacity(counter.size)),
        size: 0,
        limit: limits.wire_bytes,
    };
    write(report, &mut writer)?;
    Ok(writer.bytes.expect("encoding writer"))
}
#[cfg(feature = "render-graph")]
pub(super) fn encoded_size(
    report: &GraphCaptureReport,
    limits: GraphCaptureLimits,
) -> Result<usize, ProfilerError> {
    let mut counter = Writer {
        bytes: None,
        size: 0,
        limit: limits.wire_bytes,
    };
    write(report, &mut counter)?;
    Ok(counter.size)
}
fn write(r: &GraphCaptureReport, w: &mut Writer) -> Result<(), ProfilerError> {
    w.raw(MAGIC)?;
    w.n(r.frame)?;
    w.n(match r.source {
        GraphCaptureSource::Declaration => 0,
        GraphCaptureSource::PreparedRuntime => 1,
    })?;
    w.n(match r.timing_source {
        None => 0,
        Some(GraphTimingSource::Cpu) => 1,
        Some(GraphTimingSource::Gpu) => 2,
    })?;
    w.n(r.alias_heap_bytes)?;
    w.count(r.resources.len())?;
    for a in &r.resources {
        w.option(a.owner.map(u64::from))?;
        w.n(u64::from(a.slot))?;
        w.n(a.frame)?;
        w.text(&a.name)?;
        w.n(u64::from(a.lifetime))?;
        w.n(a.size)?;
        w.n(a.alignment)?;
        match &a.layout {
            CapturedResourceLayout::Buffer { bytes, usage } => {
                w.n(0)?;
                w.n(*bytes)?;
                w.n(u64::from(*usage))?;
            }
            CapturedResourceLayout::Texture {
                extent,
                dimension,
                format,
                mip_levels,
                samples,
                usage,
            } => {
                w.n(1)?;
                for &axis in extent {
                    w.n(u64::from(axis))?;
                }
                w.n(u64::from(*dimension))?;
                w.text(format)?;
                w.n(u64::from(*mip_levels))?;
                w.n(u64::from(*samples))?;
                w.n(u64::from(*usage))?;
            }
        }
    }
    w.count(r.passes.len())?;
    for p in &r.passes {
        w.n(u64::from(p.key))?;
        w.n(u64::from(p.feature))?;
        w.n(u64::from(p.variant))?;
        w.text(&p.name)?;
        w.n(u64::from(p.queue))?;
        w.option(p.duration_ns)?;
        w.count(p.accesses.len())?;
        for a in &p.accesses {
            w.n(u64::from(a.resource))?;
            w.n(u64::from(a.kind))?;
            w.n(u64::from(a.input_version))?;
            w.option(a.output_version.map(u64::from))?;
        }
    }
    w.count(r.order.len())?;
    for &id in &r.order {
        w.n(u64::from(id))?;
    }
    w.count(r.dependencies.len())?;
    for e in &r.dependencies {
        w.n(u64::from(e.before))?;
        w.n(u64::from(e.after))?;
        match e.kind {
            CapturedDependencyKind::Explicit => w.n(0)?,
            CapturedDependencyKind::Hazard {
                resource,
                before_access,
                after_access,
                queue_transfer,
            } => {
                w.n(1)?;
                w.n(u64::from(resource))?;
                w.n(u64::from(before_access))?;
                w.n(u64::from(after_access))?;
                w.n(u64::from(queue_transfer))?;
            }
        }
    }
    w.count(r.budgets.len())?;
    for b in &r.budgets {
        w.text(&b.category)?;
        w.n(b.consumed)?;
        w.n(b.limit)?;
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn raw(&mut self, count: usize) -> Result<&'a [u8], ProfilerError> {
        let end = self.offset.checked_add(count).ok_or_else(invalid)?;
        let result = self.bytes.get(self.offset..end).ok_or_else(invalid)?;
        self.offset = end;
        Ok(result)
    }
    fn n(&mut self) -> Result<u64, ProfilerError> {
        Ok(u64::from_le_bytes(
            self.raw(8)?.try_into().map_err(|_| invalid())?,
        ))
    }
    fn u32(&mut self) -> Result<u32, ProfilerError> {
        self.n()?.try_into().map_err(|_| invalid())
    }
    fn u8(&mut self) -> Result<u8, ProfilerError> {
        self.n()?.try_into().map_err(|_| invalid())
    }
    fn boolean(&mut self) -> Result<bool, ProfilerError> {
        match self.n()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid()),
        }
    }
    fn option(&mut self) -> Result<Option<u64>, ProfilerError> {
        if self.boolean()? {
            self.n().map(Some)
        } else {
            Ok(None)
        }
    }
    // Count checked against both configured cap and remaining wire bytes BEFORE allocation.
    fn count(&mut self, cap: usize, minimum_bytes: usize) -> Result<usize, ProfilerError> {
        let count: usize = self.n()?.try_into().map_err(|_| invalid())?;
        validate_limit(count, cap, "capture wire collection")?;
        if count > (self.bytes.len() - self.offset) / minimum_bytes {
            return Err(invalid());
        }
        Ok(count)
    }
    fn text(&mut self, cap: usize) -> Result<String, ProfilerError> {
        let n = self.count(cap, 1)?;
        Ok(std::str::from_utf8(self.raw(n)?)
            .map_err(|_| invalid())?
            .to_owned())
    }
}
pub(super) fn decode(
    bytes: &[u8],
    l: GraphCaptureLimits,
) -> Result<GraphCaptureReport, ProfilerError> {
    validate_limit(bytes.len(), l.wire_bytes, "capture wire bytes")?;
    let mut r = Reader { bytes, offset: 0 };
    if r.raw(8)? != MAGIC {
        return Err(invalid());
    }
    let frame = r.n()?;
    let source = match r.n()? {
        0 => GraphCaptureSource::Declaration,
        1 => GraphCaptureSource::PreparedRuntime,
        _ => return Err(invalid()),
    };
    let timing_source = match r.n()? {
        0 => None,
        1 => Some(GraphTimingSource::Cpu),
        2 => Some(GraphTimingSource::Gpu),
        _ => return Err(invalid()),
    };
    let alias_heap_bytes = r.n()?;
    let count = r.count(l.resources.min(u32::MAX as usize), 80)?;
    let mut resources = Vec::with_capacity(count);
    for _ in 0..count {
        let owner = r
            .option()?
            .map(u32::try_from)
            .transpose()
            .map_err(|_| invalid())?;
        let slot = r.n()?.try_into().map_err(|_| invalid())?;
        let frame = r.n()?;
        let name = r.text(l.text_bytes)?;
        let lifetime = r.u8()?;
        let size = r.n()?;
        let alignment = r.n()?;
        let layout = match r.n()? {
            0 => CapturedResourceLayout::Buffer {
                bytes: r.n()?,
                usage: r.u32()?,
            },
            1 => CapturedResourceLayout::Texture {
                extent: [r.u32()?, r.u32()?, r.u32()?],
                dimension: r.u8()?,
                format: r.text(l.text_bytes)?,
                mip_levels: r.u32()?,
                samples: r.u32()?,
                usage: r.u32()?,
            },
            _ => return Err(invalid()),
        };
        resources.push(CapturedGraphResource {
            owner,
            slot,
            frame,
            name,
            lifetime,
            size,
            alignment,
            layout,
        });
    }
    let count = r.count(l.passes.min(u32::MAX as usize), 56)?;
    let mut passes = Vec::with_capacity(count);
    let mut total = 0usize;
    for _ in 0..count {
        let key = r.u32()?;
        let feature = r.u32()?;
        let variant = r.u32()?;
        let name = r.text(l.text_bytes)?;
        let queue = r.u8()?;
        let duration_ns = r.option()?;
        let count = r.count(l.accesses.saturating_sub(total), 32)?;
        total += count;
        let mut accesses = Vec::with_capacity(count);
        for _ in 0..count {
            accesses.push(CapturedGraphAccess {
                resource: r.u32()?,
                kind: r.u8()?,
                input_version: r.u32()?,
                output_version: r
                    .option()?
                    .map(u32::try_from)
                    .transpose()
                    .map_err(|_| invalid())?,
            });
        }
        passes.push(CapturedGraphPass {
            key,
            feature,
            variant,
            name,
            queue,
            duration_ns,
            accesses,
        });
    }
    let count = r.count(l.passes, 8)?;
    let mut order = Vec::with_capacity(count);
    for _ in 0..count {
        order.push(r.u32()?);
    }
    let count = r.count(l.dependencies, 24)?;
    let mut dependencies = Vec::with_capacity(count);
    for _ in 0..count {
        let before = r.u32()?;
        let after = r.u32()?;
        let kind = match r.n()? {
            0 => CapturedDependencyKind::Explicit,
            1 => CapturedDependencyKind::Hazard {
                resource: r.u32()?,
                before_access: r.u32()?,
                after_access: r.u32()?,
                queue_transfer: r.boolean()?,
            },
            _ => return Err(invalid()),
        };
        dependencies.push(CapturedGraphDependency {
            before,
            after,
            kind,
        });
    }
    let count = r.count(l.budgets, 24)?;
    let mut budgets = Vec::with_capacity(count);
    for _ in 0..count {
        budgets.push(CapturedGraphBudget {
            category: r.text(l.text_bytes)?,
            consumed: r.n()?,
            limit: r.n()?,
        });
    }
    if r.offset != bytes.len() {
        return Err(invalid());
    }
    let report = GraphCaptureReport {
        frame,
        source,
        resources,
        passes,
        order,
        dependencies,
        budgets,
        timing_source,
        alias_heap_bytes,
    };
    report.validate(l)?;
    Ok(report)
}
