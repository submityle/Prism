//! 高层 [`Sandbox`] —— Component Model + WASI p2 + 多沙箱。
//!
//! 与核心模块版本相比的重点变化：
//! - 使用 [`wasmtime::component::Linker`] 与 [`wasmtime::component::InstancePre`]；
//! - 默认调用 [`wasmtime_wasi::p2::add_to_linker_sync`] 把 WASI p2 接口接入；
//! - 不再用 `Func::call(&[Val], &mut [Val])`，而用 component 的 typed 调用；
//!   出于通用性，本实现暴露泛型 `call_typed::<P, R>`。

use anyhow::{anyhow, Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use wasmtime::component::{ComponentNamedList, InstancePre, Lift, Linker, Lower};
use wasmtime::Store;
use wasmtime_wasi::WasiCtx;

use crate::runtime::engine::{default_wasi_ctx, SandboxEngine, StoreState};
use crate::security::attestation::{Attestor, Receipt};
use crate::security::policy::{Policy, ResourceCaps};

/// 一次执行的完整结果。
pub struct CallReceipt<R> {
    pub returns: R,
    pub fuel_consumed: Option<u64>,
    pub elapsed_ms: u128,
    pub receipt: Receipt,
}

/// 沙箱门面。
///
/// 一个 `Sandbox` ≈ 一份「策略 + Linker 配置 + 证明者」。
/// 多个 `Sandbox` 可共享同一个 [`SandboxEngine`]，从而做到「一次编译、多沙箱共享」。
///
/// `InstancePre` 缓存由每个 `Sandbox` 独立持有：不同沙箱可能拥有不同的宿主导入
/// （`configure_linker` 不同），放在引擎层共享会导致宿主函数错配。
pub struct Sandbox {
    name: String,
    engine: Arc<SandboxEngine>,
    policy: Policy,
    attestor: Arc<Attestor>,
    /// 用户提供的 Linker 配置回调。
    /// WASI p2 默认已被注册，此回调用于追加 / 覆盖额外的宿主导入。
    configure_linker: Box<dyn Fn(&mut Linker<StoreState>, &Policy) -> Result<()> + Send + Sync>,
    /// 是否在 Linker 上自动注入 WASI p2（默认 true）。
    enable_wasi: bool,
    /// WasiCtx 工厂函数，每次调用时用于构造新的 WasiCtx。
    ///
    /// 默认为 [`default_wasi_ctx`]（继承宿主 stdio）。
    /// 对于**纯计算**沙箱，可切换为 [`no_wasi_ctx`] 以省去每次调用的 3 次 `dup()` 系统调用：
    ///
    /// ```rust,ignore
    /// Sandbox::new(...).with_wasi_ctx_fn(|| prism_wasm::no_wasi_ctx())
    /// ```
    wasi_ctx_fn: Arc<dyn Fn() -> WasiCtx + Send + Sync>,
    /// 每个 Sandbox 独立持有的 InstancePre 缓存，key = component digest。
    /// 避免跨沙箱共享 Linker 配置不同的 InstancePre。
    instance_pre_cache: RwLock<HashMap<[u8; 32], Arc<InstancePre<StoreState>>>>,
    active_instances: AtomicUsize,
}

impl Sandbox {
    pub fn new(
        name: impl Into<String>,
        engine: Arc<SandboxEngine>,
        policy: Policy,
        attestor: Arc<Attestor>,
    ) -> Self {
        Self {
            name: name.into(),
            engine,
            policy,
            attestor,
            configure_linker: Box::new(|_, _| Ok(())),
            enable_wasi: true,
            wasi_ctx_fn: Arc::new(default_wasi_ctx),
            instance_pre_cache: RwLock::new(HashMap::new()),
            active_instances: AtomicUsize::new(0),
        }
    }

    pub fn with_linker<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut Linker<StoreState>, &Policy) -> Result<()> + Send + Sync + 'static,
    {
        self.configure_linker = Box::new(f);
        self
    }

    pub fn with_wasi(mut self, enable: bool) -> Self {
        self.enable_wasi = enable;
        self
    }

    /// 自定义 WasiCtx 工厂。
    ///
    /// 纯计算场景建议传入 `|| prism_wasm::no_wasi_ctx()`，
    /// 避免每次调用进行 3 次 `dup()` 系统调用。
    pub fn with_wasi_ctx_fn<F>(mut self, f: F) -> Self
    where
        F: Fn() -> WasiCtx + Send + Sync + 'static,
    {
        self.wasi_ctx_fn = Arc::new(f);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn engine(&self) -> &SandboxEngine {
        &self.engine
    }
    pub fn policy(&self) -> &Policy {
        &self.policy
    }
    pub fn attestor(&self) -> &Attestor {
        &self.attestor
    }

    // ── 内部：解析 InstancePre（双重检查锁） ─────────────────────────────

    fn resolve_pre(&self, digest: [u8; 32]) -> Result<Arc<InstancePre<StoreState>>> {
        // 快路径：读锁，绝大多数调用在此命中。
        if let Some(p) = self.instance_pre_cache.read().get(&digest).cloned() {
            return Ok(p);
        }
        // 慢路径：先构建，再写锁下二次检查插入。
        let cached = self
            .engine
            .get_component(&digest)
            .ok_or_else(|| anyhow!("摘要对应的 component 缓存丢失"))?;
        let mut linker: Linker<StoreState> = Linker::new(self.engine.engine());
        if self.enable_wasi
            && self
                .policy
                .allows(&crate::security::policy::Capability::Wasi)
        {
            wasmtime_wasi::p2::add_to_linker_sync(&mut linker)
                .context("注入 WASI p2 到 Linker 失败")?;
        }
        (self.configure_linker)(&mut linker, &self.policy).context("用户回调配置 Linker 失败")?;
        let built = linker
            .instantiate_pre(&cached.component)
            .context("预实例化 component 失败")?;

        let mut map = self.instance_pre_cache.write();
        Ok(map.entry(digest).or_insert_with(|| Arc::new(built)).clone())
    }

    // ── 内部：构造单次 Store 并执行一次调用 ──────────────────────────────

    fn execute_one<P, R>(
        &self,
        pre: &Arc<InstancePre<StoreState>>,
        digest: [u8; 32],
        func: &str,
        args: P,
    ) -> Result<CallReceipt<R>>
    where
        P: ComponentNamedList + Lower + Send + Sync + 'static + std::fmt::Debug,
        R: ComponentNamedList + Lift + Send + Sync + 'static + Clone + std::fmt::Debug,
    {
        let previous = self.active_instances.fetch_add(1, Ordering::AcqRel);
        if previous >= self.policy.max_concurrent_instances {
            self.active_instances.fetch_sub(1, Ordering::AcqRel);
            anyhow::bail!("沙箱 `{}` 并发实例配额已满", self.name);
        }
        struct ActiveGuard<'a>(&'a AtomicUsize);
        impl Drop for ActiveGuard<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let _active = ActiveGuard(&self.active_instances);
        let max_memory_bytes = self.policy.max_memory_pages.map(|p| p as usize * 64 * 1024);
        let limiter = ResourceCaps {
            max_memory_bytes,
            max_table_elements: self.policy.max_table_elements,
        };
        let state = StoreState::new(limiter, (self.wasi_ctx_fn)());
        let mut store = Store::new(self.engine.engine(), state);
        store.limiter(|s| &mut s.limiter);

        if let Some(fuel) = self.policy.fuel {
            store.set_fuel(fuel).context("注入燃料失败")?;
        }
        if let Some(deadline) = self.policy.deadline {
            let tick_ns = self.engine.epoch_tick().as_nanos();
            let ticks = deadline.as_nanos().div_ceil(tick_ns).max(1) as u64;
            store.set_epoch_deadline(ticks);
        }

        let started = Instant::now();
        let instance = pre
            .instantiate(&mut store)
            .context("实例化 component 失败")?;
        let exported = instance
            .get_func(&mut store, func)
            .ok_or_else(|| anyhow!("component 没有导出 `{func}`"))?;
        let typed = exported
            .typed::<P, R>(&store)
            .with_context(|| format!("`{func}` 的类型与期望不匹配"))?;

        let input_repr = format!("{args:?}");
        let returns = typed.call(&mut store, args).context("调用失败")?;
        typed.post_return(&mut store).ok();

        let elapsed_ms = started.elapsed().as_millis();
        let fuel_consumed = self
            .policy
            .fuel
            .map(|f| f.saturating_sub(store.get_fuel().unwrap_or(0)));

        let output_repr = format!("{:?}", &returns);
        anyhow::ensure!(
            output_repr.len() <= self.policy.max_output_bytes,
            "输出超过策略上限"
        );
        let trace = std::mem::take(&mut store.data_mut().trace);
        anyhow::ensure!(
            trace.len() <= self.policy.max_trace_bytes,
            "审计 trace 超过策略上限"
        );
        let receipt = self.attestor.issue(
            digest,
            input_repr.as_bytes(),
            output_repr.as_bytes(),
            &trace,
        );

        Ok(CallReceipt {
            returns,
            fuel_consumed,
            elapsed_ms,
            receipt,
        })
    }

    // ── 公开 API ──────────────────────────────────────────────────────────

    /// 类型化调用（单次）。每次调用创建一个全新的 `Store`（micro-sandbox 边界）。
    pub fn call_typed<P, R>(&self, name: &str, func: &str, args: P) -> Result<CallReceipt<R>>
    where
        P: ComponentNamedList + Lower + Send + Sync + 'static + std::fmt::Debug,
        R: ComponentNamedList + Lift + Send + Sync + 'static + Clone + std::fmt::Debug,
    {
        let digest = self
            .engine
            .current_digest(name)
            .ok_or_else(|| anyhow!("没有名为 `{name}` 的 component 被安装"))?;
        let pre = self.resolve_pre(digest)?;
        self.execute_one(&pre, digest, func, args)
    }

    /// 批量类型化调用：在**同一个** `spawn_blocking` 内顺序执行 `args_list` 中的所有调用。
    ///
    /// # 性能说明
    ///
    /// 对比逐个 `call_typed`，`call_typed_many` 把以下固定开销**摊薄到整个批次**：
    ///
    /// | 开销项 | 逐个调用 | 批量调用 |
    /// |---|---|---|
    /// | `current_digest` RwLock read | ×N | ×1 |
    /// | `instance_pre_cache` RwLock read | ×N | ×1 |
    /// | 策略常量计算 | ×N | ×1 |
    ///
    /// 每次调用仍然创建独立的 `Store`（隔离边界不变）。
    pub fn call_typed_many<P, R>(
        &self,
        name: &str,
        func: &str,
        args_list: Vec<P>,
    ) -> Vec<Result<CallReceipt<R>>>
    where
        P: ComponentNamedList + Lower + Send + Sync + 'static + std::fmt::Debug,
        R: ComponentNamedList + Lift + Send + Sync + 'static + Clone + std::fmt::Debug,
    {
        let digest = match self.engine.current_digest(name) {
            Some(d) => d,
            None => {
                let msg = format!("没有名为 `{name}` 的 component 被安装");
                return args_list
                    .into_iter()
                    .map(|_| Err(anyhow!("{msg}")))
                    .collect();
            }
        };
        let pre = match self.resolve_pre(digest) {
            Ok(p) => p,
            Err(e) => {
                let msg = format!("{e}");
                return args_list
                    .into_iter()
                    .map(|_| Err(anyhow!("{msg}")))
                    .collect();
            }
        };
        args_list
            .into_iter()
            .map(|args| self.execute_one(&pre, digest, func, args))
            .collect()
    }

    /// 异步单次调用：在 tokio 阻塞线程池中执行，不阻塞 async 执行器。
    ///
    /// > **高吞吐场景请使用 [`call_many_async`](Sandbox::call_many_async)**，
    /// > 避免每次请求都独占一个 `spawn_blocking` 调度槽。
    pub async fn call_typed_async<P, R>(
        self: Arc<Self>,
        name: &str,
        func: &str,
        args: P,
    ) -> Result<CallReceipt<R>>
    where
        P: ComponentNamedList + Lower + Send + Sync + 'static + std::fmt::Debug,
        R: ComponentNamedList + Lift + Send + Sync + 'static + Clone + std::fmt::Debug,
    {
        let name = name.to_string();
        let func = func.to_string();
        tokio::task::spawn_blocking(move || self.call_typed(&name, &func, args))
            .await
            .map_err(|e| anyhow::anyhow!("spawn_blocking panicked: {e}"))?
    }

    /// 异步批量调用：`args_list` 中的所有调用在**同一个** `spawn_blocking` 中顺序执行。
    ///
    /// # 推荐用法（百万级并发）
    ///
    /// ```rust,ignore
    /// // 把 1,000,000 个请求分成 chunk_size 大小的批次，每批一个 spawn_blocking
    /// let chunk_size = 2000;
    /// let mut tasks = tokio::task::JoinSet::new();
    /// for chunk in all_args.chunks(chunk_size) {
    ///     let sb = sandbox.clone();
    ///     let batch = chunk.to_vec();
    ///     tasks.spawn(async move {
    ///         sb.call_many_async("mod", "fn", batch).await
    ///     });
    /// }
    /// ```
    pub async fn call_many_async<P, R>(
        self: Arc<Self>,
        name: &str,
        func: &str,
        args_list: Vec<P>,
    ) -> Result<Vec<Result<CallReceipt<R>>>>
    where
        P: ComponentNamedList + Lower + Send + Sync + 'static + std::fmt::Debug,
        R: ComponentNamedList + Lift + Send + Sync + 'static + Clone + std::fmt::Debug,
    {
        let name = name.to_string();
        let func = func.to_string();
        tokio::task::spawn_blocking(move || self.call_typed_many(&name, &func, args_list))
            .await
            .map_err(|e| anyhow::anyhow!("spawn_blocking panicked: {e}"))
    }
}
