//! 零信任策略（Zero-Trust Policy）。
//!
//! 设计原则：**默认拒绝（deny by default）**，所有能力必须被显式授权。
//! - 宿主导入函数：默认全部拒绝，除非通过 [`Capability::HostImport`] 放行；
//! - 文件 / 网络 / 环境变量：同上；
//! - 资源（内存页、表元素、燃料、墙钟时间）：必须给上限；
//! - 模块身份：可选的 SHA-256 摘要白名单，未列入即不予加载。
//!
//! 该策略在 [`crate::sandbox::Sandbox`] 与 [`crate::engine::SandboxEngine`]
//! 中作为唯一「准入开关」使用。

use std::collections::BTreeSet;
use std::time::Duration;

/// 一项可被授予 / 拒绝的能力。
///
/// 使用 `BTreeSet` 存放，所以必须实现 `Ord`；这里用最简单的派生即可。
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Capability {
    /// 允许调用宿主导入函数 `module::field`，例如 `host::log`。
    HostImport(String, String),
    /// 允许读取指定环境变量。
    EnvRead(String),
    /// 允许读 / 写指定预开放目录（与 WASI preopens 配合使用）。
    FsRead(String),
    FsWrite(String),
    /// 允许向 `host:port` 发起网络连接（与 wasmtime-wasi sockets 配合使用）。
    NetConnect(String, u16),
    /// 允许链接 WASI Preview 2 接口。具体资源仍由受限 WasiCtx 控制。
    Wasi,
}

/// 策略本体 —— 所有上限 + 能力白名单 + 摘要白名单。
#[derive(Clone, Debug)]
pub struct Policy {
    /// 已授权能力集合。
    pub caps: BTreeSet<Capability>,
    /// 线性内存最大页数（每页 64 KiB）；`None` 表示用引擎默认值。
    pub max_memory_pages: Option<u32>,
    /// 表（function table 等）最大元素数量。
    pub max_table_elements: Option<u32>,
    /// 单次调用的墙钟超时；触发后通过 epoch 中断打断执行。
    pub deadline: Option<Duration>,
    /// 单次调用的燃料预算；`None` 表示关闭燃料计量。
    pub fuel: Option<u64>,
    /// 模块摘要白名单（零信任准入）。**空集合 = 不限制**。
    pub allowed_digests: BTreeSet<[u8; 32]>,
    /// 同一租户允许同时存活的调用/实例数；超过时立即施加背压。
    pub max_concurrent_instances: usize,
    /// 回执输出表示与宿主审计 trace 的最大字节数。
    pub max_output_bytes: usize,
    pub max_trace_bytes: usize,
}

impl Default for Policy {
    /// 一份「足够安全」的默认策略：
    /// - 不授予任何宿主能力；
    /// - 内存上限 16 MiB（256 页 × 64 KiB）；
    /// - 表上限 10000；
    /// - 1 秒墙钟超时；
    /// - 1000 万燃料；
    /// - 不限制模块摘要。
    fn default() -> Self {
        Self {
            caps: BTreeSet::new(),
            max_memory_pages: Some(256),
            max_table_elements: Some(10_000),
            deadline: Some(Duration::from_secs(1)),
            fuel: Some(10_000_000),
            allowed_digests: BTreeSet::new(),
            max_concurrent_instances: 64,
            max_output_bytes: 1024 * 1024,
            max_trace_bytes: 64 * 1024,
        }
    }
}

impl Policy {
    /// 取得一个 [`PolicyBuilder`]，用流式 API 构造策略。
    pub fn builder() -> PolicyBuilder {
        PolicyBuilder { p: Self::default() }
    }

    /// 检查某项能力是否被授予。
    pub fn allows(&self, cap: &Capability) -> bool {
        self.caps.contains(cap)
    }

    /// 检查某模块摘要是否被允许加载。
    /// 白名单为空时视为「全部放行」。
    pub fn digest_allowed(&self, digest: &[u8; 32]) -> bool {
        self.allowed_digests.is_empty() || self.allowed_digests.contains(digest)
    }
}

/// 策略构造器（建造者模式）。
///
/// 用法示例：
/// ```ignore
/// let p = Policy::builder()
///     .allow_import("host", "log")
///     .fuel(2_000_000)
///     .deadline(Duration::from_millis(500))
///     .pin_digest(known_hash)
///     .build();
/// ```
pub struct PolicyBuilder {
    p: Policy,
}

impl PolicyBuilder {
    /// 授予一项任意能力。
    pub fn allow(mut self, cap: Capability) -> Self {
        self.p.caps.insert(cap);
        self
    }
    /// 快捷方法：授予一个宿主导入函数。
    pub fn allow_import(self, module: &str, field: &str) -> Self {
        self.allow(Capability::HostImport(module.into(), field.into()))
    }
    /// 设置线性内存最大页数。
    pub fn memory_pages(mut self, n: u32) -> Self {
        self.p.max_memory_pages = Some(n);
        self
    }
    /// 设置燃料预算。
    pub fn fuel(mut self, n: u64) -> Self {
        self.p.fuel = Some(n);
        self
    }
    /// 设置墙钟超时。
    pub fn deadline(mut self, d: Duration) -> Self {
        self.p.deadline = Some(d);
        self
    }
    pub fn max_concurrent_instances(mut self, n: usize) -> Self {
        self.p.max_concurrent_instances = n;
        self
    }
    pub fn output_limit(mut self, bytes: usize) -> Self {
        self.p.max_output_bytes = bytes;
        self
    }
    pub fn trace_limit(mut self, bytes: usize) -> Self {
        self.p.max_trace_bytes = bytes;
        self
    }
    /// 把一个模块摘要加入白名单（多次调用即可加入多个）。
    pub fn pin_digest(mut self, digest: [u8; 32]) -> Self {
        self.p.allowed_digests.insert(digest);
        self
    }
    /// 构造最终的不可变 [`Policy`]。
    pub fn build(self) -> Policy {
        self.p
    }
}

/// 把策略中的内存 / 表上限映射成 wasmtime 的 [`wasmtime::ResourceLimiter`]。
///
/// 引擎会在每次内存 / 表增长前回调 `memory_growing` / `table_growing`，
/// 我们返回 `false` 即可拒绝增长，从而让分配失败的运行时错误冒泡出去。
pub struct ResourceCaps {
    /// 字节为单位的内存上限。
    pub max_memory_bytes: Option<usize>,
    /// 表元素上限。
    pub max_table_elements: Option<u32>,
}

impl wasmtime::ResourceLimiter for ResourceCaps {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> anyhow::Result<bool> {
        Ok(match self.max_memory_bytes {
            Some(cap) => desired <= cap,
            None => true,
        })
    }

    fn table_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> anyhow::Result<bool> {
        Ok(match self.max_table_elements {
            Some(cap) => desired <= cap as usize,
            None => true,
        })
    }
}
