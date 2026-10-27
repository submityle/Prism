//! 热插拔（Hot Swap）—— 在不中断现有调用的前提下替换模块版本。
//!
//! # 实现策略
//! - **mtime 轮询**：对登记了文件路径的模块，定时比较 `metadata.modified()`；
//!   时间戳变了就触发重载（也支持手动 `reload`）。
//! - **原子指针**：所有「当前版本」信息都集中在
//!   [`crate::runtime::SandboxEngine::current`] 这张表里，安装新版本 ≡ 翻指针。
//! - **零中断**：正在执行的旧调用持有的是旧 `Arc<InstancePre>`，
//!   不会被打断；新进的调用读到新指针，自动用新版本。

use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use crate::loader::Loader;
use crate::runtime::engine::SandboxEngine;
use crate::security::policy::Policy;

/// 一个被追踪的逻辑模块。
struct Tracked {
    /// 引擎中登记的逻辑名。
    name: String,
    /// 可选的文件路径；为 `Some` 时 `poll()` 会比较其 mtime。
    path: Option<PathBuf>,
    /// 上一次观察到的 mtime；首次建议设成「文件当前 mtime」，避免立即触发一次重载。
    last_mtime: Option<SystemTime>,
}

/// 热插拔管理器。
///
/// 通常由上层应用持有一个长生命周期实例，并把它注入到调度循环 / 定时任务中
/// 周期性调用 [`HotSwap::poll`]。
pub struct HotSwap {
    engine: Arc<SandboxEngine>,
    loader: Arc<dyn Loader>,
    policy: Policy,
    tracked: Mutex<HashMap<String, Tracked>>,
}

impl HotSwap {
    /// 构造热插拔管理器。
    ///
    /// `loader` 应当与初次安装时使用的加载器一致，否则重载可能拿到不同来源的字节。
    pub fn new(engine: Arc<SandboxEngine>, loader: Arc<dyn Loader>, policy: Policy) -> Self {
        Self {
            engine,
            loader,
            policy,
            tracked: Mutex::new(HashMap::new()),
        }
    }

    /// 首次登记一个逻辑模块。
    ///
    /// - `watch_path`：传 `Some(path)` 即可启用 mtime 轮询；
    ///   传 `None` 则只能用 [`HotSwap::reload`] 手动触发。
    pub fn track(&self, name: &str, watch_path: Option<PathBuf>) -> Result<[u8; 32]> {
        // 1) 通过加载器读取字节并安装到引擎。
        let src = self.loader.load(name)?;
        let cached = self.engine.install(src, &self.policy)?;

        // 2) 记录初始 mtime，避免 track 后立即 poll 触发一次伪重载。
        let mtime = watch_path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok().and_then(|m| m.modified().ok()));
        self.tracked.lock().insert(
            name.to_string(),
            Tracked {
                name: name.to_string(),
                path: watch_path,
                last_mtime: mtime,
            },
        );
        Ok(cached.digest)
    }

    /// 强制重载一个模块（不看 mtime）。
    ///
    /// # 返回值
    /// - `Ok(Some(new_digest))`：内容变了，已切换到新版本；
    /// - `Ok(None)`：内容字节完全一致，无需切换；
    /// - `Err(_)`：加载或编译失败，保持当前版本不变（旧调用不受影响）。
    pub fn reload(&self, name: &str) -> Result<Option<[u8; 32]>> {
        let src = self.loader.load(name)?;
        let prev = self.engine.current_digest(name);
        if prev == Some(src.digest) {
            return Ok(None);
        }
        let cached = self.engine.install(src, &self.policy)?;
        Ok(Some(cached.digest))
    }

    /// 轮询所有登记了文件路径的模块。
    /// 凡是 mtime 与上次记录不同的，都尝试重载一次。
    ///
    /// # 返回
    /// 实际发生切换（摘要变化）的逻辑名列表，便于上层做日志 / 通知。
    pub fn poll(&self) -> Result<Vec<String>> {
        let mut swapped = Vec::new();
        let mut tracked = self.tracked.lock();
        for t in tracked.values_mut() {
            // 没有路径的模块跳过（只能手动 reload）。
            let Some(path) = t.path.as_ref() else {
                continue;
            };
            // 文件被删了 / 暂时拿不到元数据：保守跳过，等下次再说。
            let Ok(meta) = std::fs::metadata(path) else {
                continue;
            };
            let Ok(modified) = meta.modified() else {
                continue;
            };
            if t.last_mtime != Some(modified) {
                if self.reload(&t.name)?.is_some() {
                    swapped.push(t.name.clone());
                }
                // 不论是否真的切换，都记录新的 mtime，避免反复触发。
                t.last_mtime = Some(modified);
            }
        }
        Ok(swapped)
    }
}
