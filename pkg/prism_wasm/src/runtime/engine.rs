//! 沙箱引擎 —— 升级到 **Component Model + WASI Preview 2**。
//!
//! # 关键变化（相对核心模块版本）
//! - 缓存对象从 [`wasmtime::Module`] 换成 [`wasmtime::component::Component`]；
//! - `InstancePre` 来自 `wasmtime::component`；
//! - `StoreState` 内嵌 [`wasmtime_wasi::WasiCtx`] 与 [`ResourceTable`]，
//!   并实现 [`wasmtime_wasi::WasiView`]，从而能直接接入 WASI p2 主机实现。

use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use wasmtime::component::{Component, ResourceTable};
use wasmtime::{Config, Engine};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

use crate::security::policy::Policy;

/// 注入到每个 `Store<StoreState>` 中的状态。
///
/// 同时充当：
///   - 资源 limiter 宿主（`limiter`）
///   - 宿主 trace 收集器（`trace`）
///   - WASI p2 上下文（`ctx` / `table`）
pub struct StoreState {
    pub limiter: crate::security::policy::ResourceCaps,
    pub trace: Vec<u8>,
    pub ctx: WasiCtx,
    pub table: ResourceTable,
}

impl StoreState {
    pub fn new(limiter: crate::security::policy::ResourceCaps, ctx: WasiCtx) -> Self {
        Self {
            limiter,
            trace: Vec::new(),
            ctx,
            table: ResourceTable::new(),
        }
    }
}

// 让 wasmtime-wasi 的主机实现能从 Store 中拿到 ctx + table。
impl WasiView for StoreState {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

/// 一个已编译的 Component 缓存项。
#[derive(Clone)]
pub struct CachedComponent {
    pub digest: [u8; 32],
    pub name: String,
    pub component: Component,
}

/// 沙箱引擎 —— 一个进程通常只需要一份，多沙箱共享之。
///
/// 只负责缓存**编译产物**（`Component`）；`InstancePre` 由各 `Sandbox` 独立持有，
/// 因为不同沙箱的 Linker 配置可能不同，共享 `InstancePre` 会导致宿主函数错乱。
///
/// # 并发编译去重
/// `compile_gates` 为每个 digest 维护一个 [`tokio::sync::OnceCell`]，确保即使
/// N 个沙箱并发安装相同字节，JIT 编译也只发生一次，其余 N-1 个调用直接等待结果。
pub struct SandboxEngine {
    pub(crate) engine: Engine,
    /// 已完成编译的 Component 缓存（content-addressed by SHA-256 digest）。
    components: RwLock<HashMap<[u8; 32], CachedComponent>>,
    /// 模块逻辑名 → 当前 digest 映射（支持热插拔）。
    pub(crate) current: RwLock<HashMap<String, [u8; 32]>>,
    /// 每 digest 一个 OnceCell，保证并发安装相同字节时编译只触发一次。
    compile_gates:
        parking_lot::Mutex<HashMap<[u8; 32], Arc<tokio::sync::OnceCell<CachedComponent>>>>,
    _watchdog: WatchdogHandle,
    epoch_tick: Duration,
}

impl SandboxEngine {
    pub fn new() -> Result<Self> {
        Self::with_epoch_tick(Duration::from_millis(50))
    }

    pub fn with_epoch_tick(epoch_tick: Duration) -> Result<Self> {
        anyhow::ensure!(!epoch_tick.is_zero(), "epoch tick 必须大于零");
        let mut cfg = Config::new();
        cfg.consume_fuel(true);
        cfg.epoch_interruption(true);
        cfg.wasm_backtrace(true);
        cfg.wasm_component_model(true); // 显式启用 component model
        cfg.cranelift_opt_level(wasmtime::OptLevel::Speed);
        let engine = Engine::new(&cfg).context("创建 wasmtime 引擎失败")?;

        let watchdog = WatchdogHandle::spawn(engine.clone(), epoch_tick);

        Ok(Self {
            engine,
            components: Default::default(),
            current: Default::default(),
            compile_gates: Default::default(),
            _watchdog: watchdog,
            epoch_tick,
        })
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn epoch_tick(&self) -> Duration {
        self.epoch_tick
    }

    /// 同步安装：在调用方线程直接编译 component（适用于启动期单线程初始化）。
    ///
    /// 热路径请使用 [`install_async`]，避免阻塞调用线程。
    pub fn install(
        &self,
        src: crate::loader::ModuleSource,
        policy: &Policy,
    ) -> Result<CachedComponent> {
        if !policy.digest_allowed(&src.digest) {
            anyhow::bail!(
                "Component 摘要 {} 不在策略白名单中",
                hex::encode(src.digest)
            );
        }

        // 快路径：缓存命中，仅在 digest 变化时才取写锁更新 current。
        if let Some(c) = self.components.read().get(&src.digest).cloned() {
            self.update_current_if_changed(&src.name, src.digest);
            return Ok(c);
        }

        let component = Component::new(&self.engine, &**src.bytes)
            .with_context(|| format!("编译 component `{}` 失败", src.name))?;

        let cached = CachedComponent {
            digest: src.digest,
            name: src.name.clone(),
            component,
        };
        self.components.write().insert(src.digest, cached.clone());
        self.current.write().insert(src.name, src.digest);
        Ok(cached)
    }

    /// 异步安装：编译在 tokio 阻塞线程池中完成，不阻塞 async 执行器。
    ///
    /// # 并发去重
    /// 多个沙箱并发安装相同字节时，JIT 编译只会被触发**一次**；
    /// 其余并发调用通过 `OnceCell` 等待第一次编译的结果，避免重复编译浪费。
    pub async fn install_async(
        &self,
        src: crate::loader::ModuleSource,
        policy: &Policy,
    ) -> Result<CachedComponent> {
        if !policy.digest_allowed(&src.digest) {
            anyhow::bail!(
                "Component 摘要 {} 不在策略白名单中",
                hex::encode(src.digest)
            );
        }

        // 超快路径：已完成编译，直接返回缓存。
        if let Some(c) = self.components.read().get(&src.digest).cloned() {
            self.update_current_if_changed(&src.name, src.digest);
            return Ok(c);
        }

        // 取或创建该 digest 专属的 OnceCell —— 保证并发时只有一个 init 闭包被执行。
        let gate = {
            let mut map = self.compile_gates.lock();
            map.entry(src.digest)
                .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new()))
                .clone()
        };

        // get_or_try_init: 第一个到达的 caller 执行 init；其余并发 caller 等待结果。
        // 若 init 失败，OnceCell 重置，下次调用重试。
        let cached = gate
            .get_or_try_init(|| async {
                let engine_clone = self.engine.clone();
                let bytes = src.bytes.clone();
                let name_for_err = src.name.clone();
                let digest = src.digest;

                let component = tokio::task::spawn_blocking(move || {
                    Component::new(&engine_clone, &**bytes)
                        .with_context(|| format!("编译 component `{}` 失败", name_for_err))
                })
                .await
                .map_err(|e| anyhow::anyhow!("spawn_blocking panicked: {e}"))??;

                let cached = CachedComponent {
                    digest,
                    name: src.name.clone(),
                    component,
                };
                // 写入全局 components 缓存，使 get_component / call_typed 可见。
                self.components.write().insert(digest, cached.clone());
                Ok::<CachedComponent, anyhow::Error>(cached)
            })
            .await?
            .clone();

        self.update_current_if_changed(&src.name, src.digest);
        Ok(cached)
    }

    pub fn current_digest(&self, name: &str) -> Option<[u8; 32]> {
        self.current.read().get(name).copied()
    }

    pub fn get_component(&self, digest: &[u8; 32]) -> Option<CachedComponent> {
        self.components.read().get(digest).cloned()
    }

    /// 将指定名称的「当前版本」强制指向某个已缓存的摘要。
    ///
    /// 用于回滚场景：当金丝雀版本出现问题时，将指针拨回上一个稳定摘要。
    /// 若该摘要从未被编译过（不在缓存中），返回错误。
    pub fn pin_version(&self, name: &str, digest: [u8; 32]) -> anyhow::Result<()> {
        if !self.components.read().contains_key(&digest) {
            anyhow::bail!(
                "摘要 {} 不在 component 缓存中，无法回滚（该版本可能从未被安装过）",
                hex::encode(digest)
            );
        }
        self.current.write().insert(name.to_string(), digest);
        Ok(())
    }

    /// 仅在 digest 确实改变时才取写锁，减少热路径写竞争。
    fn update_current_if_changed(&self, name: &str, digest: [u8; 32]) {
        if self.current.read().get(name).copied() == Some(digest) {
            return; // 已是最新，无需写锁
        }
        self.current.write().insert(name.to_string(), digest);
    }
}

/// 默认的 WASI Ctx 构造器：标准输入输出继承自宿主。
/// 如要更严格隔离，由调用方自行传入定制好的 `WasiCtx`。
pub fn default_wasi_ctx() -> WasiCtx {
    WasiCtxBuilder::new().inherit_stdio().build()
}

/// 最小化 WASI Ctx：不继承宿主 stdio，省去 3 次 `dup()` 系统调用。
///
/// 适合**纯计算**沙箱（无文件 I/O 需求）。在百万级调用场景下，
/// 使用此函数替代 [`default_wasi_ctx`] 可显著降低 per-call 系统调用开销：
///
/// ```rust,ignore
/// Sandbox::new(...).with_wasi_ctx_fn(prism_wasm::no_wasi_ctx)
/// ```
pub fn no_wasi_ctx() -> WasiCtx {
    WasiCtxBuilder::new().build()
}

struct WatchdogHandle {
    stop: Arc<std::sync::atomic::AtomicBool>,
}

impl WatchdogHandle {
    fn spawn(engine: Engine, tick: Duration) -> Self {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let s2 = stop.clone();
        thread::spawn(move || {
            while !s2.load(std::sync::atomic::Ordering::Relaxed) {
                thread::sleep(tick);
                engine.increment_epoch();
            }
        });
        Self { stop }
    }
}

impl Drop for WatchdogHandle {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}
