//! Manifest-checked archive routing shared by pollable IO backends.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::ToString;
use core::task::{Context, Poll};
use prism_world_io::IoError;
use prism_world_stream::SectorId;

use crate::{CellIo, CellReadRequest, IoCompletion, RequestId, SubmitError};

/// Resolves current cell manifests to registered archive ranges before admission.
///
/// Registration changes affect future submissions only. An admitted request keeps
/// its original metadata, offset, and checksum until terminal retirement.
pub struct RoutedCellIo<I: CellIo> {
    io: I,
    sources: BTreeMap<SectorId, CellReadRequest>,
}

impl<I: CellIo> RoutedCellIo<I> {
    /// Wraps a backend with an initially empty source registry.
    pub fn new(io: I) -> Self {
        Self {
            io,
            sources: BTreeMap::new(),
        }
    }

    /// Registers a complete manifest and range without performing IO.
    ///
    /// Overflow is rejected before replacing an existing source.
    pub fn register(
        &mut self,
        source: CellReadRequest,
    ) -> Result<Option<CellReadRequest>, IoError> {
        let request = source.io_request();
        request
            .offset()
            .checked_add(request.length())
            .ok_or(IoError::RangeOverflow)?;
        Ok(self.sources.insert(source.sector(), source))
    }

    /// Resolves an exact manifest while preserving the caller's scheduling priority.
    pub fn resolve(&self, request: CellReadRequest) -> Result<CellReadRequest, IoError> {
        let source = self
            .sources
            .get(&request.sector())
            .ok_or_else(|| IoError::Provider {
                message: format!("no file range registered for sector {:?}", request.sector()),
            })?;
        if source.manifest() != request.manifest() {
            return Err(IoError::Provider {
                message: format!("stale file range for sector {:?}", request.sector()),
            });
        }
        Ok(CellReadRequest {
            priority: request.priority(),
            ..*source
        })
    }
}

impl<I: CellIo> CellIo for RoutedCellIo<I> {
    type Error = I::Error;

    fn submit(&mut self, request: CellReadRequest) -> Result<RequestId, SubmitError> {
        let request = self
            .resolve(request)
            .map_err(|error| SubmitError::Provider {
                message: error.to_string(),
            })?;
        self.io.submit(request)
    }

    fn cancel(&mut self, id: RequestId) -> bool {
        self.io.cancel(id)
    }

    fn in_flight(&self) -> usize {
        self.io.in_flight()
    }

    fn poll_completion(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<IoCompletion<Self::Error>, Self::Error>> {
        self.io.poll_completion(cx)
    }
}

#[cfg(feature = "tasks")]
impl<I: crate::PriorityCellIo> crate::PriorityCellIo for RoutedCellIo<I> {
    fn submit_with_priority(
        &mut self,
        request: CellReadRequest,
        priority: alloc::sync::Arc<prism_tasks::PriorityCell>,
    ) -> Result<RequestId, SubmitError> {
        let request = self
            .resolve(request)
            .map_err(|error| SubmitError::Provider {
                message: error.to_string(),
            })?;
        self.io.submit_with_priority(request, priority)
    }
}
