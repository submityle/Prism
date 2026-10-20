//! Bounded navigation over an immutable authoritative frame-graph capture.

use std::{collections::BTreeSet, sync::Arc};

use prism_editor_profiler::{CriticalPath, GraphCaptureLimits, GraphCaptureReport};

use crate::{frame_graph_view::BoundFrameGraphPanel, UiEvent};

/// Maximum rows materialized in each independent panel page.
pub const MAX_FRAME_GRAPH_PAGE_ROWS: usize = 32;

/// Independent page sizes; report admission also uses `GraphCaptureLimits::default`.
#[derive(Clone, Copy, Debug)]
pub struct FrameGraphPanelLimits {
    /// Pass boxes on the queue-lane graph.
    pub passes: usize,
    /// Resource rows.
    pub resources: usize,
    /// Dependency rows and maximum painted dependency wires.
    pub dependencies: usize,
    /// Access/SSA relationship rows.
    pub accesses: usize,
    /// Budget gauges.
    pub budgets: usize,
}
impl Default for FrameGraphPanelLimits {
    fn default() -> Self {
        Self {
            passes: 8,
            resources: 6,
            dependencies: 8,
            accesses: 8,
            budgets: 6,
        }
    }
}

/// Independently paged report table; offsets refer to the displayed filtered list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameGraphPage {
    /// Compiler execution order.
    Passes,
    /// Original resource table.
    Resources,
    /// All or selection-related dependency edges.
    Dependencies,
    /// Selected pass accesses or resource users, in declaration order.
    Accesses,
    /// Canonically sorted real budget categories.
    Budgets,
}
impl FrameGraphPage {
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Passes => 0,
            Self::Resources => 1,
            Self::Dependencies => 2,
            Self::Accesses => 3,
            Self::Budgets => 4,
        }
    }
}

/// IDs are positions in this exact published report, never semantic pass keys.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameGraphSelection {
    /// Original pass index.
    Pass(u32),
    /// Original resource index.
    Resource(u32),
    /// Canonical dependency index.
    Dependency(usize),
}

/// Host commands and exact bound semantic activations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameGraphPanelAction {
    /// Select a real pass and reveal its accesses/relationships.
    Pass(u32),
    /// Select a real resource and reveal its users/relationships.
    Resource(u32),
    /// Inspect one explicit or hazard edge; preserve the relationship filter.
    Dependency(usize),
    /// Show the next page of a table.
    Next(FrameGraphPage),
    /// Show the preceding page of a table.
    Previous(FrameGraphPage),
    /// Restrict dependencies to the last selected pass/resource.
    RelatedOnly(bool),
    /// Clear the selection and relationship filter.
    ClearSelection,
}

/// Rejection before publication or navigation state changes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameGraphPanelError {
    /// Page sizes must be nonzero and no larger than the hard row cap.
    InvalidLimits,
    /// Published report exceeds independent UI admission bounds.
    ReportCapacity,
    /// Report analysis failed; the previous publication remains intact.
    InvalidReport,
    /// Event belongs to another controller, publication or navigation revision.
    StaleView,
    /// Target or event is absent from the exact displayed tree.
    UnboundEvent,
    /// Command references an absent report/table entry.
    MissingTarget,
}
impl std::fmt::Display for FrameGraphPanelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "frame graph panel: {self:?}")
    }
}
impl std::error::Error for FrameGraphPanelError {}

/// UI state only: no recorder, renderer, GPU buffer or capture device is owned.
pub struct FrameGraphPanelController {
    pub(crate) report: Option<Arc<GraphCaptureReport>>,
    pub(crate) limits: FrameGraphPanelLimits,
    pub(crate) offsets: [usize; 5],
    pub(crate) selection: Option<FrameGraphSelection>,
    pub(crate) focus: Option<FrameGraphSelection>,
    pub(crate) related_only: bool,
    pub(crate) critical: Option<CriticalPath>,
    pub(crate) critical_passes: BTreeSet<u32>,
    pub(crate) stamp: Arc<()>,
}
impl Default for FrameGraphPanelController {
    fn default() -> Self {
        Self::new(FrameGraphPanelLimits::default()).expect("valid default page limits")
    }
}
impl FrameGraphPanelController {
    /// Create an empty bounded panel.
    pub fn new(limits: FrameGraphPanelLimits) -> Result<Self, FrameGraphPanelError> {
        if [
            limits.passes,
            limits.resources,
            limits.dependencies,
            limits.accesses,
            limits.budgets,
        ]
        .iter()
        .any(|&n| n == 0 || n > MAX_FRAME_GRAPH_PAGE_ROWS)
        {
            return Err(FrameGraphPanelError::InvalidLimits);
        }
        Ok(Self {
            report: None,
            limits,
            offsets: [0; 5],
            selection: None,
            focus: None,
            related_only: true,
            critical: None,
            critical_passes: BTreeSet::new(),
            stamp: Arc::new(()),
        })
    }
    /// Publish an actual completed capture or clear it. Validate before mutation.
    /// Every accepted publication fences old events, even for equal frame IDs or
    /// the same Arc. Selection indices never carry across capture publications.
    pub fn publish(
        &mut self,
        report: Option<Arc<GraphCaptureReport>>,
    ) -> Result<(), FrameGraphPanelError> {
        let critical = if let Some(r) = &report {
            admit(r)?;
            r.critical_path(GraphCaptureLimits::default())
                .map_err(|_| FrameGraphPanelError::InvalidReport)?
        } else {
            None
        };
        let critical_passes = critical
            .as_ref()
            .map(|p| p.node_ids().iter().map(|id| id.raw() as u32).collect())
            .unwrap_or_default();
        self.report = report;
        self.critical = critical;
        self.critical_passes = critical_passes;
        self.selection = None;
        self.focus = None;
        self.offsets = [0; 5];
        self.stamp = Arc::new(());
        Ok(())
    }
    /// Exact immutable publication. Retaining it prevents mutation via Arc.
    pub fn report(&self) -> Option<&Arc<GraphCaptureReport>> {
        self.report.as_ref()
    }
    /// Selection qualified by the current publication.
    pub const fn selection(&self) -> Option<FrameGraphSelection> {
        self.selection
    }
    /// Only a complete measured path; partial/missing timings remain None.
    pub fn critical_path(&self) -> Option<&CriticalPath> {
        self.critical.as_ref()
    }
    /// Materialize bounded real Loom controls, boxes, wires and budget gauges.
    pub fn build(&self) -> BoundFrameGraphPanel {
        crate::frame_graph_view::build(self)
    }
    /// Resolve only a bound event from this exact controller revision AND Arc.
    pub fn handle_event(
        &mut self,
        view: &BoundFrameGraphPanel,
        target: &str,
        event: &UiEvent,
    ) -> Result<FrameGraphPanelAction, FrameGraphPanelError> {
        let same_report = match (&self.report, &view.report) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        if !Arc::ptr_eq(&self.stamp, &view.stamp) || !same_report {
            return Err(FrameGraphPanelError::StaleView);
        }
        let action = view.dispatch(target, event)?;
        self.apply(action)?;
        Ok(action)
    }
    /// Apply validated host navigation. Errors leave both state and fence intact.
    pub fn apply(&mut self, action: FrameGraphPanelAction) -> Result<(), FrameGraphPanelError> {
        use FrameGraphPanelAction as A;
        let r = self
            .report
            .as_deref()
            .ok_or(FrameGraphPanelError::MissingTarget)?;
        match action {
            A::Pass(id) if id as usize >= r.passes().len() => {
                return Err(FrameGraphPanelError::MissingTarget)
            }
            A::Resource(id) if id as usize >= r.resources().len() => {
                return Err(FrameGraphPanelError::MissingTarget)
            }
            A::Dependency(id) if id >= r.dependencies().len() => {
                return Err(FrameGraphPanelError::MissingTarget)
            }
            _ => {}
        }
        match action {
            A::Pass(id) => {
                self.selection = Some(FrameGraphSelection::Pass(id));
                self.focus = self.selection;
                self.offsets[0] = reveal(
                    r.order()
                        .iter()
                        .position(|&p| p == id)
                        .expect("validated order"),
                    self.limits.passes,
                );
                self.offsets[2] = 0;
                self.offsets[3] = 0;
            }
            A::Resource(id) => {
                self.selection = Some(FrameGraphSelection::Resource(id));
                self.focus = self.selection;
                self.offsets[1] = reveal(id as usize, self.limits.resources);
                self.offsets[2] = 0;
                self.offsets[3] = 0;
            }
            A::Dependency(id) => self.selection = Some(FrameGraphSelection::Dependency(id)),
            A::Next(page) => {
                let (count, size) = self.page_info(page);
                let offset = &mut self.offsets[page.index()];
                *offset = (*offset + size).min(reveal(count.saturating_sub(1), size));
            }
            A::Previous(page) => {
                let (_, size) = self.page_info(page);
                let offset = &mut self.offsets[page.index()];
                *offset = offset.saturating_sub(size);
            }
            A::RelatedOnly(value) => {
                self.related_only = value;
                self.offsets[2] = 0;
            }
            A::ClearSelection => {
                self.selection = None;
                self.focus = None;
                self.offsets[2] = 0;
                self.offsets[3] = 0;
            }
        }
        self.stamp = Arc::new(());
        Ok(())
    }
    pub(crate) fn dependency_matches(
        &self,
        edge: &prism_editor_profiler::CapturedGraphDependency,
    ) -> bool {
        if !self.related_only {
            return true;
        }
        match self.focus {
            Some(FrameGraphSelection::Pass(id)) => edge.before == id || edge.after == id,
            Some(FrameGraphSelection::Resource(id)) => {
                matches!(edge.kind, prism_editor_profiler::CapturedDependencyKind::Hazard { resource, .. } if resource == id)
            }
            _ => true,
        }
    }
    pub(crate) fn access_matches(&self, pass: u32, resource: u32) -> bool {
        match self.focus {
            Some(FrameGraphSelection::Pass(id)) => pass == id,
            Some(FrameGraphSelection::Resource(id)) => resource == id,
            _ => false,
        }
    }
    pub(crate) fn page_info(&self, page: FrameGraphPage) -> (usize, usize) {
        let Some(r) = self.report.as_deref() else {
            return (0, 1);
        };
        match page {
            FrameGraphPage::Passes => (r.passes().len(), self.limits.passes),
            FrameGraphPage::Resources => (r.resources().len(), self.limits.resources),
            FrameGraphPage::Dependencies => (
                r.dependencies()
                    .iter()
                    .filter(|e| self.dependency_matches(e))
                    .count(),
                self.limits.dependencies,
            ),
            FrameGraphPage::Accesses => (
                r.passes()
                    .iter()
                    .enumerate()
                    .flat_map(|(p, pass)| pass.accesses.iter().map(move |a| (p as u32, a.resource)))
                    .filter(|&(p, r)| self.access_matches(p, r))
                    .count(),
                self.limits.accesses,
            ),
            FrameGraphPage::Budgets => (r.budgets().len(), self.limits.budgets),
        }
    }
}
fn reveal(index: usize, size: usize) -> usize {
    index / size * size
}

fn admit(r: &GraphCaptureReport) -> Result<(), FrameGraphPanelError> {
    let l = GraphCaptureLimits::default();
    let mut accesses = 0usize;
    if r.passes().len() > l.passes
        || r.resources().len() > l.resources
        || r.dependencies().len() > l.dependencies
        || r.budgets().len() > l.budgets
    {
        return Err(FrameGraphPanelError::ReportCapacity);
    }
    let text = |s: &str| {
        if s.len() > l.text_bytes {
            Err(FrameGraphPanelError::ReportCapacity)
        } else {
            Ok(())
        }
    };
    for p in r.passes() {
        text(&p.name)?;
        accesses = accesses
            .checked_add(p.accesses.len())
            .ok_or(FrameGraphPanelError::ReportCapacity)?;
        if accesses > l.accesses {
            return Err(FrameGraphPanelError::ReportCapacity);
        }
    }
    for resource in r.resources() {
        text(&resource.name)?;
        if let prism_editor_profiler::CapturedResourceLayout::Texture { format, .. } =
            &resource.layout
        {
            text(format)?;
        }
    }
    for b in r.budgets() {
        text(&b.category)?;
    }
    Ok(())
}
