//! Contributed asset checks and snapshot-fenced save, commit and cook gates.
//! Callbacks are trusted host code: budgets bound invocations and retained data,
//! not callback execution time or allocations inside a callback.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::fmt;

use crate::{AssetId, AssetRecord, ContentCatalog, DiagnosticCode, DiagnosticSeverity};

/// Operation for which the host requests validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationOperation {
    /// Persist authoring changes.
    Save,
    /// Publish changes to version control.
    Commit,
    /// Produce runtime artifacts.
    Cook,
}

/// Explicit bounds for a complete validation run.
#[derive(Clone, Copy, Debug)]
pub struct ValidationLimits {
    /// Maximum contributed rules.
    pub max_rules: usize,
    /// Maximum assets examined by built-in and contributed checks.
    pub max_assets: usize,
    /// Maximum dependency edges examined by built-in checks.
    pub max_dependencies: usize,
    /// Maximum asset/rule callback invocations.
    pub max_evaluations: usize,
    /// Maximum contributed finding emissions, including repeated findings.
    pub max_findings: usize,
    /// Maximum diagnostic rows retained for presentation.
    pub max_diagnostics: usize,
    /// Maximum UTF-8 bytes per contributed message or callback failure.
    pub max_message_bytes: usize,
    /// Maximum bytes per rule identity or contributed finding code.
    pub max_identifier_bytes: usize,
}

impl Default for ValidationLimits {
    fn default() -> Self {
        Self {
            max_rules: 64,
            max_assets: 100_000,
            max_dependencies: 1_000_000,
            max_evaluations: 1_000_000,
            max_findings: 100_000,
            max_diagnostics: 1024,
            max_message_bytes: 4096,
            max_identifier_bytes: 128,
        }
    }
}

/// Context shared by contributed checks; catalog access never clones its data.
pub struct ValidationContext<'a> {
    /// Complete catalog snapshot being checked.
    pub catalog: &'a ContentCatalog,
    /// Requested operation, allowing stricter cook or commit policy.
    pub operation: ValidationOperation,
}

/// A project or domain validation contribution. No business policy is built in.
/// Configuration must remain immutable while registered; re-register after
/// policy changes to revoke old receipts. Checks using external state must
/// arrange their own revision fence when the operation is published.
pub trait AssetValidationRule: Send + Sync {
    /// Stable namespaced identity, for example project.texture-budget.
    fn id(&self) -> &str;
    /// Checks one asset. Return an error if this check could not complete.
    /// Findings emitted before failure remain visible, but cannot authorize work.
    fn validate(
        &self,
        context: &ValidationContext<'_>,
        asset: &AssetRecord,
        diagnostics: &mut ValidationSink<'_>,
    ) -> Result<(), String>;
}

/// A navigable finding with stable rule/code identity and exact asset path.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ValidationIssue {
    /// Severity used by the requested gate policy.
    pub severity: DiagnosticSeverity,
    /// Built-in or host contribution identity.
    pub rule: String,
    /// Stable finding code within that rule.
    pub code: String,
    /// Owning asset, when applicable.
    pub asset: Option<AssetId>,
    /// Exact source path for navigation.
    pub path: Option<String>,
    /// Bounded presentation message.
    pub message: String,
}

/// Registration or execution rejection; no authorization is produced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationError {
    /// Invalid rule identity or contribution code.
    InvalidIdentifier,
    /// A contribution uses the reserved built-in namespace.
    ReservedIdentifier,
    /// An identity is already registered.
    DuplicateRule(String),
    /// A named execution or retained-data budget was exceeded.
    Limit(&'static str),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier => f.write_str("invalid validation identifier"),
            Self::ReservedIdentifier => f.write_str("reserved validation identifier"),
            Self::DuplicateRule(id) => write!(f, "validation rule already registered: {id}"),
            Self::Limit(name) => write!(f, "validation exceeded {name}"),
        }
    }
}

impl core::error::Error for ValidationError {}

struct Findings {
    limits: ValidationLimits,
    issues: BTreeSet<ValidationIssue>,
    errors: usize,
    warnings: usize,
    emissions: usize,
    complete: bool,
    truncated: bool,
}

impl Findings {
    fn retain(&mut self, issue: ValidationIssue) {
        match issue.severity {
            DiagnosticSeverity::Error => self.errors = self.errors.saturating_add(1),
            DiagnosticSeverity::Warning => self.warnings = self.warnings.saturating_add(1),
            DiagnosticSeverity::Info => {}
        }
        self.issues.insert(issue);
        if self.issues.len() > self.limits.max_diagnostics {
            self.issues.pop_last();
            self.truncated = true;
        }
    }
}

/// A bounded contribution output for one rule and asset. The host supplies
/// identity/path; callbacks cannot redirect a finding to an unrelated asset.
pub struct ValidationSink<'a> {
    findings: &'a mut Findings,
    rule: &'a str,
    asset: &'a AssetRecord,
    rejected: Option<ValidationError>,
}

impl ValidationSink<'_> {
    /// Emits a finding. Rejection is latched even if a callback ignores the error.
    pub fn emit(
        &mut self,
        severity: DiagnosticSeverity,
        code: &str,
        message: &str,
    ) -> Result<(), ValidationError> {
        if let Some(error) = &self.rejected {
            return Err(error.clone());
        }
        let result = self.check(severity, code, message);
        if let Err(error) = &result {
            self.rejected = Some(error.clone());
        }
        result
    }

    fn check(
        &mut self,
        severity: DiagnosticSeverity,
        code: &str,
        message: &str,
    ) -> Result<(), ValidationError> {
        check_identifier(code, self.findings.limits)?;
        if message.len() > self.findings.limits.max_message_bytes {
            return Err(ValidationError::Limit("message bytes"));
        }
        if self.findings.emissions >= self.findings.limits.max_findings {
            return Err(ValidationError::Limit("finding emissions"));
        }
        self.findings.emissions += 1;
        self.findings.retain(ValidationIssue {
            severity,
            rule: self.rule.to_string(),
            code: code.to_string(),
            asset: Some(self.asset.id),
            path: Some(self.asset.path.clone()),
            message: message.to_string(),
        });
        Ok(())
    }
}

/// Rule owner with deterministic identity order and revocable snapshot identity.
pub struct ValidationRegistry {
    limits: ValidationLimits,
    rules: BTreeMap<String, Arc<dyn AssetValidationRule>>,
    identity: Arc<()>,
}

impl Default for ValidationRegistry {
    fn default() -> Self {
        Self::new(ValidationLimits::default())
    }
}

impl ValidationRegistry {
    /// Creates an empty contribution set. Built-in graph/VCS checks always run.
    #[must_use]
    pub fn new(limits: ValidationLimits) -> Self {
        Self {
            limits,
            rules: BTreeMap::new(),
            identity: Arc::new(()),
        }
    }

    /// Registers a complete contribution or leaves the registry unchanged.
    pub fn register(&mut self, rule: Arc<dyn AssetValidationRule>) -> Result<(), ValidationError> {
        let id = rule.id();
        check_identifier(id, self.limits)?;
        if id == "builtin" || id.starts_with("builtin.") {
            return Err(ValidationError::ReservedIdentifier);
        }
        if self.rules.contains_key(id) {
            return Err(ValidationError::DuplicateRule(id.to_string()));
        }
        if self.rules.len() >= self.limits.max_rules {
            return Err(ValidationError::Limit("rule count"));
        }
        self.rules.insert(id.to_string(), rule);
        self.identity = Arc::new(());
        Ok(())
    }

    /// Removes a contribution and revokes previously generated receipts.
    pub fn unregister(&mut self, id: &str) -> bool {
        if self.rules.remove(id).is_none() {
            return false;
        }
        self.identity = Arc::new(());
        true
    }

    /// Validates an immutable complete catalog. Callback input is bounded before
    /// any callback runs; failed checks cannot produce a passing receipt.
    pub fn validate(
        &self,
        catalog: &ContentCatalog,
        operation: ValidationOperation,
    ) -> Result<ValidationReceipt, ValidationError> {
        if catalog.len() > self.limits.max_assets {
            return Err(ValidationError::Limit("asset count"));
        }
        if catalog
            .len()
            .checked_mul(self.rules.len())
            .is_none_or(|count| count > self.limits.max_evaluations)
        {
            return Err(ValidationError::Limit("rule evaluations"));
        }
        let mut dependencies = 0usize;
        for record in catalog.assets.values() {
            dependencies = dependencies
                .checked_add(record.dependencies.len())
                .filter(|count| *count <= self.limits.max_dependencies)
                .ok_or(ValidationError::Limit("dependency count"))?;
        }
        let builtin = catalog.validate_with_diagnostic_limit(
            self.limits
                .max_diagnostics
                .min(catalog.limits.max_diagnostics),
        );
        let mut findings = Findings {
            limits: self.limits,
            issues: BTreeSet::new(),
            errors: 0,
            warnings: 0,
            emissions: 0,
            complete: builtin.is_complete(),
            truncated: builtin.truncated,
        };
        for diagnostic in builtin.diagnostics() {
            let code = match diagnostic.code {
                DiagnosticCode::MissingDependency => "missing-dependency",
                DiagnosticCode::DependencyCycle => "dependency-cycle",
                DiagnosticCode::DependencyDepthExceeded => "dependency-depth",
                DiagnosticCode::VersionControlConflict => "vcs-conflict",
            };
            findings.retain(ValidationIssue {
                severity: diagnostic.severity,
                rule: "builtin".to_string(),
                code: code.to_string(),
                asset: diagnostic.asset,
                path: diagnostic.path.clone(),
                message: diagnostic.message.clone(),
            });
        }
        // Built-in output may already have been truncated. Its complete error
        // count remains authoritative even when no error row survived display.
        findings.errors = builtin.error_count();
        findings.warnings = builtin.warning_count();
        for (id, rule) in &self.rules {
            for asset in catalog.assets.values() {
                let mut sink = ValidationSink {
                    findings: &mut findings,
                    rule: id,
                    asset,
                    rejected: None,
                };
                let result =
                    rule.validate(&ValidationContext { catalog, operation }, asset, &mut sink);
                if let Some(error) = sink.rejected {
                    return Err(error);
                }
                if let Err(message) = result {
                    if message.len() > self.limits.max_message_bytes {
                        return Err(ValidationError::Limit("message bytes"));
                    }
                    findings.complete = false;
                    findings.retain(ValidationIssue {
                        severity: DiagnosticSeverity::Error,
                        rule: id.clone(),
                        code: "rule-failed".to_string(),
                        asset: Some(asset.id),
                        path: Some(asset.path.clone()),
                        message,
                    });
                }
            }
        }
        Ok(ValidationReceipt {
            catalog_identity: catalog.snapshot_identity.clone(),
            registry_identity: self.identity.clone(),
            operation,
            issues: findings.issues.into_iter().collect(),
            errors: findings.errors,
            warnings: findings.warnings,
            complete: findings.complete,
            truncated: findings.truncated,
        })
    }
}

fn check_identifier(id: &str, limits: ValidationLimits) -> Result<(), ValidationError> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-/".contains(&byte))
    {
        return Err(ValidationError::InvalidIdentifier);
    }
    if id.len() > limits.max_identifier_bytes {
        return Err(ValidationError::Limit("identifier bytes"));
    }
    Ok(())
}

/// Gate policy. Errors and incomplete checks always block the operation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ValidationPolicy {
    /// Also reject warnings, including those omitted by diagnostic display bounds.
    pub deny_warnings: bool,
}

/// An immutable validation result tied to exact catalog and rule snapshots.
pub struct ValidationReceipt {
    catalog_identity: Arc<()>,
    registry_identity: Arc<()>,
    operation: ValidationOperation,
    issues: Vec<ValidationIssue>,
    errors: usize,
    warnings: usize,
    complete: bool,
    /// At least one finding was omitted from the bounded display list.
    pub truncated: bool,
}

impl ValidationReceipt {
    /// Deterministically sorted, bounded findings for report UI/navigation.
    #[must_use]
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }

    /// Error finding emissions including those not retained for presentation.
    #[must_use]
    pub const fn error_count(&self) -> usize {
        self.errors
    }

    /// Warning finding emissions including those omitted from presentation.
    #[must_use]
    pub const fn warning_count(&self) -> usize {
        self.warnings
    }

    /// Whether all checks completed within their bounds.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete
    }

    /// Borrows the exact checked snapshot for immediate save/commit/cook
    /// preparation. The immutable borrow prevents mutation while preparing it.
    /// Asynchronous publication still needs the destination's revision fence.
    pub fn authorize<'a>(
        &self,
        catalog: &'a ContentCatalog,
        registry: &'a ValidationRegistry,
        operation: ValidationOperation,
        policy: ValidationPolicy,
    ) -> Result<ValidatedCatalog<'a>, ValidationRejection> {
        if !Arc::ptr_eq(&self.catalog_identity, &catalog.snapshot_identity) {
            return Err(ValidationRejection::StaleCatalog);
        }
        if !Arc::ptr_eq(&self.registry_identity, &registry.identity) {
            return Err(ValidationRejection::StaleRules);
        }
        if self.operation != operation {
            return Err(ValidationRejection::WrongOperation);
        }
        if !self.complete {
            return Err(ValidationRejection::Incomplete);
        }
        if self.errors != 0 {
            return Err(ValidationRejection::Errors);
        }
        if policy.deny_warnings && self.warnings != 0 {
            return Err(ValidationRejection::Warnings);
        }
        Ok(ValidatedCatalog {
            catalog,
            _registry: registry,
            operation,
        })
    }
}

/// Failure to authorize an operation from a validation receipt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationRejection {
    /// The catalog changed or belongs to an unrelated snapshot.
    StaleCatalog,
    /// The contribution set changed or belongs to another registry.
    StaleRules,
    /// A receipt was prepared for another operation.
    WrongOperation,
    /// A traversal or contributed check did not finish.
    Incomplete,
    /// At least one error was emitted.
    Errors,
    /// Strict policy rejected at least one warning, displayed or omitted.
    Warnings,
}

impl fmt::Display for ValidationRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "validation gate rejected: {self:?}")
    }
}

impl core::error::Error for ValidationRejection {}

/// Borrowed checked snapshot. It cannot be constructed or retargeted by callers.
pub struct ValidatedCatalog<'a> {
    catalog: &'a ContentCatalog,
    _registry: &'a ValidationRegistry,
    operation: ValidationOperation,
}

impl ValidatedCatalog<'_> {
    /// Catalog from which a host prepares the authorized operation.
    #[must_use]
    pub const fn catalog(&self) -> &ContentCatalog {
        self.catalog
    }

    /// Operation that passed this gate.
    #[must_use]
    pub const fn operation(&self) -> ValidationOperation {
        self.operation
    }
}
