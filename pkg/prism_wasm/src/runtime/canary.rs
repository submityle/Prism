//! 金丝雀发布框架与自愈机制。
//!
//! # 金丝雀发布
//! [`CanaryRouter`] 在同一个 [`Sandbox`] 上管理稳定版本与金丝雀版本，
//! 通过原子计数器取模来路由流量，支持：
//! - **影子模式（Shadow Mode）**：金丝雀流量同时在稳定版本上静默再跑一次，
//!   比较两份 Receipt 的 `commitment` 是否一致；
//! - **自动回滚**：金丝雀错误率超过阈值时自动将流量切回稳定版本；
//! - **自动晋升**：金丝雀连续低错误率达到阈值时自动将其提升为稳定版本。
//!
//! # 自愈机制
//! [`SelfHealing`] 为每个模块维护一个**滑动窗口**错误率统计，
//! 当失败率超过 `failure_threshold` 时自动调用
//! [`SandboxEngine::pin_version`] 将指针拨回上一个稳定摘要，
//! 无需人工介入即可降级恢复。

use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use wasmtime::component::{ComponentNamedList, Lift, Lower};

use crate::runtime::engine::SandboxEngine;
use crate::runtime::sandbox::{CallReceipt, Sandbox};

// ─────────────────────────── VersionStats ────────────────────────────────────

/// 金丝雀路由器的双版本统计快照。
#[derive(Default, Clone, Debug)]
pub struct VersionStats {
    /// 稳定版本总调用次数。
    pub stable_calls: u64,
    /// 稳定版本错误次数。
    pub stable_errors: u64,
    /// 金丝雀版本总调用次数。
    pub canary_calls: u64,
    /// 金丝雀版本错误次数。
    pub canary_errors: u64,
    /// 影子模式下稳定/金丝雀输出 Receipt 不一致的次数。
    pub divergent_receipts: u64,
}

impl VersionStats {
    /// 金丝雀版本错误率（0.0–1.0）。无调用时返回 0.0。
    pub fn canary_error_rate(&self) -> f64 {
        if self.canary_calls == 0 {
            0.0
        } else {
            self.canary_errors as f64 / self.canary_calls as f64
        }
    }

    /// 稳定版本错误率（0.0–1.0）。无调用时返回 0.0。
    pub fn stable_error_rate(&self) -> f64 {
        if self.stable_calls == 0 {
            0.0
        } else {
            self.stable_errors as f64 / self.stable_calls as f64
        }
    }
}

// ─────────────────────────── RouteDecision ───────────────────────────────────

/// 单次调用的路由决策。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDecision {
    Stable,
    Canary,
}

// ─────────────────────────── CanaryRouter ────────────────────────────────────

/// 金丝雀路由器。
///
/// # 使用流程
/// 1. 将稳定版本安装为 `"<base>:stable"`、金丝雀版本安装为 `"<base>:canary"`；
/// 2. 构造 `CanaryRouter::new(sandbox, "<base>")`；
/// 3. 调用 [`deploy_canary`](CanaryRouter::deploy_canary) 设置初始流量比；
/// 4. 用 [`call_typed`](CanaryRouter::call_typed) 替代原来的
///    `sandbox.call_typed()`——路由逻辑自动处理。
///
/// # 影子模式
/// 开启 `.with_shadow_mode(true)` 后，所有路由到金丝雀版本的请求会**同时**在
/// 稳定版本上静默再跑一次，比较两份 Receipt 的 commitment，差异计入
/// [`VersionStats::divergent_receipts`]。
pub struct CanaryRouter {
    sandbox: Arc<Sandbox>,
    stable_name: String,
    /// 当前金丝雀版本的逻辑名，`None` 表示尚未部署金丝雀。
    canary_name: RwLock<Option<String>>,
    /// 金丝雀流量百分比（0–100）。
    canary_percent: AtomicUsize,
    /// 单调递增请求计数器，用于取模路由（避免引入 rand 依赖）。
    counter: AtomicU64,
    stats: Mutex<VersionStats>,
    /// 金丝雀错误率超过此值时自动回滚（默认 0.10）。
    auto_rollback_threshold: f64,
    /// 金丝雀成功率超过此值时自动晋升（默认 0.99）。
    auto_promote_threshold: f64,
    /// 触发自动晋升/回滚所需的最小金丝雀调用次数（默认 20）。
    min_canary_calls: u64,
    /// 是否开启影子模式。
    shadow_mode: bool,
}

impl CanaryRouter {
    /// 构造路由器。初始无金丝雀版本，所有流量走稳定版本。
    ///
    /// `module_base_name` 是模块的基础名称，路由器会自动派生：
    /// - 稳定版本名：`"<base>:stable"`
    /// - 金丝雀版本名：`"<base>:canary"`（由 [`deploy_canary`] 指定）
    pub fn new(sandbox: Arc<Sandbox>, module_base_name: impl Into<String>) -> Self {
        let base = module_base_name.into();
        Self {
            stable_name: format!("{base}:stable"),
            canary_name: RwLock::new(None),
            sandbox,
            canary_percent: AtomicUsize::new(0),
            counter: AtomicU64::new(0),
            stats: Mutex::new(VersionStats::default()),
            auto_rollback_threshold: 0.10,
            auto_promote_threshold: 0.99,
            min_canary_calls: 20,
            shadow_mode: false,
        }
    }

    /// 设置自动回滚的错误率阈值（0.0–1.0，默认 0.10）。
    pub fn with_rollback_threshold(mut self, t: f64) -> Self {
        self.auto_rollback_threshold = t;
        self
    }

    /// 设置自动晋升的成功率阈值（0.0–1.0，默认 0.99）。
    pub fn with_promote_threshold(mut self, t: f64) -> Self {
        self.auto_promote_threshold = t;
        self
    }

    /// 设置触发自动决策所需的最小金丝雀调用次数（默认 20）。
    pub fn with_min_canary_calls(mut self, n: u64) -> Self {
        self.min_canary_calls = n;
        self
    }

    /// 开启/关闭影子模式。
    pub fn with_shadow_mode(mut self, enabled: bool) -> Self {
        self.shadow_mode = enabled;
        self
    }

    /// 稳定版本的逻辑模块名（`"<base>:stable"`）。
    pub fn stable_name(&self) -> &str {
        &self.stable_name
    }

    /// 部署新的金丝雀版本，并设置初始流量比（0–100）。
    ///
    /// 调用此方法前需先把金丝雀字节安装到 `SandboxEngine`：
    /// ```rust,ignore
    /// let canary_src = ModuleSource::new("adder:canary", bytes);
    /// engine.install(canary_src, &policy)?;
    /// router.deploy_canary("adder:canary", 5); // 5% 流量
    /// ```
    pub fn deploy_canary(&self, canary_name: impl Into<String>, percent: usize) {
        *self.canary_name.write() = Some(canary_name.into());
        self.canary_percent
            .store(percent.min(100), Ordering::Relaxed);
        *self.stats.lock() = VersionStats::default(); // 重置统计
    }

    /// 调整金丝雀流量百分比（0–100）。
    pub fn set_canary_percent(&self, percent: usize) {
        self.canary_percent
            .store(percent.min(100), Ordering::Relaxed);
    }

    /// 暂停金丝雀流量（保留部署，流量归零）。
    pub fn pause_canary(&self) {
        self.canary_percent.store(0, Ordering::Relaxed);
    }

    /// 手动回滚：将金丝雀流量归零。
    pub fn rollback(&self) {
        self.canary_percent.store(0, Ordering::Relaxed);
    }

    /// 手动晋升：将金丝雀流量归零（调用方还需更新引擎的版本指针）。
    ///
    /// 完整晋升示例：
    /// ```rust,ignore
    /// let canary_digest = engine.current_digest("adder:canary").unwrap();
    /// engine.pin_version("adder:stable", canary_digest)?;
    /// router.promote();
    /// ```
    pub fn promote(&self) {
        self.canary_percent.store(0, Ordering::Relaxed);
    }

    /// 读取当前统计快照。
    pub fn stats(&self) -> VersionStats {
        self.stats.lock().clone()
    }

    /// 当前金丝雀流量百分比。
    pub fn canary_percent(&self) -> usize {
        self.canary_percent.load(Ordering::Relaxed)
    }

    // 检查是否触发自动晋升/回滚。
    fn maybe_auto_adjust(&self) {
        let stats = self.stats.lock();
        if stats.canary_calls < self.min_canary_calls {
            return;
        }
        let error_rate = stats.canary_error_rate();
        let success_rate = 1.0 - error_rate;
        drop(stats);

        if error_rate > self.auto_rollback_threshold {
            self.rollback();
        } else if success_rate >= self.auto_promote_threshold {
            self.promote();
        }
    }

    /// 类型化调用（自动路由 + 统计 + 影子模式 + 自动晋升/回滚）。
    ///
    /// 返回 `(CallReceipt<R>, RouteDecision)`；调用方可根据 `RouteDecision`
    /// 决定是否记录路由信息。
    ///
    /// `P` 需实现 `Clone`，影子模式下会额外克隆一份参数用于稳定版本比对。
    pub fn call_typed<P, R>(&self, func: &str, args: P) -> Result<(CallReceipt<R>, RouteDecision)>
    where
        P: ComponentNamedList + Lower + Send + Sync + Clone + std::fmt::Debug + 'static,
        R: ComponentNamedList + Lift + Send + Sync + 'static + Clone + std::fmt::Debug,
    {
        let percent = self.canary_percent.load(Ordering::Relaxed);
        let cnt = self.counter.fetch_add(1, Ordering::Relaxed);

        // 确定路由目标（读锁只持有到字符串克隆完毕，不跨越 WASM 执行）
        let (use_canary, target_name) = {
            let guard = self.canary_name.read();
            let route_to_canary =
                guard.is_some() && percent > 0 && ((cnt % 100) as usize) < percent;
            let name = if route_to_canary {
                guard.as_deref().unwrap().to_string()
            } else {
                self.stable_name.clone()
            };
            (route_to_canary, name)
        };

        let decision = if use_canary {
            RouteDecision::Canary
        } else {
            RouteDecision::Stable
        };

        // 影子模式：提前克隆参数，用于稳定版本的静默执行
        let shadow_args = if self.shadow_mode && use_canary {
            Some(args.clone())
        } else {
            None
        };

        let result = self.sandbox.call_typed::<P, R>(&target_name, func, args);

        // 影子模式：比对 stable 和 canary 的 Receipt commitment
        if let Some(s_args) = shadow_args {
            if let Ok(canary_r) = &result {
                if let Ok(stable_r) =
                    self.sandbox
                        .call_typed::<P, R>(&self.stable_name, func, s_args)
                {
                    if canary_r.receipt.commitment != stable_r.receipt.commitment {
                        self.stats.lock().divergent_receipts += 1;
                    }
                }
            }
        }

        // 更新统计
        {
            let mut s = self.stats.lock();
            if use_canary {
                s.canary_calls += 1;
                if result.is_err() {
                    s.canary_errors += 1;
                }
            } else {
                s.stable_calls += 1;
                if result.is_err() {
                    s.stable_errors += 1;
                }
            }
        }

        self.maybe_auto_adjust();
        result.map(|r| (r, decision))
    }
}

// ─────────────────────────── SelfHealing ─────────────────────────────────────

struct HealthWindow {
    /// 上一个已知稳定版本的摘要（版本升级前由外部调用 `mark_stable` 保存）。
    stable_digest: Option<[u8; 32]>,
    /// 滑动窗口：`true` = 成功，`false` = 失败。
    window: VecDeque<bool>,
    window_size: usize,
    /// 是否已触发过回滚（避免重复操作，等待人工确认后由 `reset` 解除）。
    rolled_back: bool,
}

impl HealthWindow {
    fn new(window_size: usize) -> Self {
        Self {
            stable_digest: None,
            window: VecDeque::with_capacity(window_size),
            window_size,
            rolled_back: false,
        }
    }

    fn push(&mut self, success: bool) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(success);
    }

    fn failure_rate(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let failures = self.window.iter().filter(|&&ok| !ok).count();
        failures as f64 / self.window.len() as f64
    }
}

/// 自愈管理器。
///
/// 为每个模块名维护一个滑动窗口错误率，当失败率超过 `failure_threshold` 时
/// 自动调用 [`SandboxEngine::pin_version`] 回滚到上一个稳定摘要。
///
/// # 使用示例
/// ```rust,ignore
/// let healer = Arc::new(SelfHealing::new(engine.clone())
///     .failure_threshold(0.15)
///     .window_size(50));
///
/// // 版本升级前：记录当前稳定摘要
/// healer.mark_stable("adder", engine.current_digest("adder").unwrap());
///
/// // 每次调用后：记录结果
/// match sandbox.call_typed::<P, R>("adder", "fn", args) {
///     Ok(_)  => healer.record_success("adder"),
///     Err(_) => {
///         if let Some(d) = healer.record_failure("adder") {
///             eprintln!("自愈回滚到摘要 {}", hex::encode(d));
///         }
///     }
/// }
/// ```
pub struct SelfHealing {
    engine: Arc<SandboxEngine>,
    modules: Mutex<HashMap<String, HealthWindow>>,
    /// 触发回滚的失败率阈值（默认 0.15 = 15%）。
    failure_threshold: f64,
    /// 滑动窗口大小（默认 50 次调用）。
    window_size: usize,
    /// 触发判断所需的最小样本数（默认 window_size / 2）。
    min_samples: usize,
}

impl SelfHealing {
    pub fn new(engine: Arc<SandboxEngine>) -> Self {
        Self {
            engine,
            modules: Mutex::new(HashMap::new()),
            failure_threshold: 0.15,
            window_size: 50,
            min_samples: 10,
        }
    }

    pub fn failure_threshold(mut self, t: f64) -> Self {
        self.failure_threshold = t;
        self
    }

    pub fn window_size(mut self, n: usize) -> Self {
        self.min_samples = n / 2;
        self.window_size = n;
        self
    }

    pub fn min_samples(mut self, n: usize) -> Self {
        self.min_samples = n;
        self
    }

    /// 记录当前稳定摘要，通常在每次热插拔升级**前**调用。
    /// 若发生回滚，引擎会将 `current[name]` 恢复为此摘要。
    pub fn mark_stable(&self, module_name: &str, digest: [u8; 32]) {
        let mut mods = self.modules.lock();
        let w = mods
            .entry(module_name.to_string())
            .or_insert_with(|| HealthWindow::new(self.window_size));
        w.stable_digest = Some(digest);
        w.rolled_back = false; // 新版本部署，重置回滚锁
    }

    /// 记录一次成功调用。
    pub fn record_success(&self, module_name: &str) {
        self.push(module_name, true);
    }

    /// 记录一次失败调用，并检查是否触发自愈回滚。
    ///
    /// 返回 `Some(stable_digest)` 表示已触发回滚并恢复到该摘要。
    pub fn record_failure(&self, module_name: &str) -> Option<[u8; 32]> {
        self.push(module_name, false);
        self.maybe_heal(module_name)
    }

    /// 手动重置回滚锁，允许再次触发自愈（通常在人工确认新版本稳定后调用）。
    pub fn reset(&self, module_name: &str) {
        if let Some(w) = self.modules.lock().get_mut(module_name) {
            w.rolled_back = false;
            w.window.clear();
        }
    }

    /// 查询某模块当前窗口内的失败率（0.0–1.0）。
    pub fn failure_rate(&self, module_name: &str) -> f64 {
        self.modules
            .lock()
            .get(module_name)
            .map(|w| w.failure_rate())
            .unwrap_or(0.0)
    }

    /// 查询某模块是否处于已回滚状态（等待人工确认）。
    pub fn is_rolled_back(&self, module_name: &str) -> bool {
        self.modules
            .lock()
            .get(module_name)
            .map(|w| w.rolled_back)
            .unwrap_or(false)
    }

    fn push(&self, module_name: &str, success: bool) {
        let mut mods = self.modules.lock();
        let w = mods
            .entry(module_name.to_string())
            .or_insert_with(|| HealthWindow::new(self.window_size));
        w.push(success);
    }

    fn maybe_heal(&self, module_name: &str) -> Option<[u8; 32]> {
        let mut mods = self.modules.lock();
        let w = mods.get_mut(module_name)?;
        if w.rolled_back {
            return None;
        }
        if w.window.len() < self.min_samples {
            return None;
        }
        if w.failure_rate() < self.failure_threshold {
            return None;
        }
        let digest = w.stable_digest?;
        if self.engine.pin_version(module_name, digest).is_ok() {
            w.rolled_back = true;
            w.window.clear(); // 清空窗口，避免立即再次触发
            Some(digest)
        } else {
            None
        }
    }
}
