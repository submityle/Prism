//! 模块加载器（Loader）—— 沙箱的「零信任入口」。
//!
//! 所有进入引擎的字节流，都必须经过某个 [`Loader`] 实现，并在落盘 / 编译
//! 之前完成 SHA-256 指纹计算。这样：
//!   - 上层 [`crate::security::Policy`] 可以基于摘要做白名单准入；
//!   - 上层 [`crate::runtime::SandboxEngine`] 可以用摘要做内容寻址缓存；
//!   - 上层 [`crate::security::attestation`] 可以把摘要写入回执便于审计。
//!
//! 想接入 HTTP / IPFS / OCI 等远程源，只需自行实现 [`Loader`] trait。
//!
//! 子模块：
//!   - [`hotswap`] —— 基于 mtime + 原子指针的热插拔管理器。

pub mod hotswap;
pub use hotswap::HotSwap;

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 加载产出的「带指纹的模块字节」。
///
/// `bytes` 用 `Arc` 包裹是为了让多个组件（编译器、缓存、回执）能共享同一份
/// 内存，而不需要拷贝。
#[derive(Clone)]
pub struct ModuleSource {
    /// 上层逻辑名（业务侧用来引用模块的别名，例如 `"adder"`）。
    pub name: String,
    /// 模块原始字节（可能是 `.wasm` 二进制，也可能是 `.wat` 文本）。
    pub bytes: Arc<Vec<u8>>,
    /// 模块字节的 SHA-256 摘要 —— 内容寻址 ID。
    pub digest: [u8; 32],
}

impl ModuleSource {
    /// 构造一个 `ModuleSource`，构造时自动计算摘要。
    pub fn new(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        Self {
            name: name.into(),
            bytes: Arc::new(bytes),
            digest,
        }
    }

    /// 把摘要转成十六进制字符串，方便日志输出。
    pub fn digest_hex(&self) -> String {
        hex::encode(self.digest)
    }
}

/// 可插拔的 WASM 字节来源 trait。
///
/// 只要实现 `load`，就能把任意来源接入沙箱引擎。
pub trait Loader: Send + Sync {
    /// 根据逻辑名加载模块字节。
    fn load(&self, name: &str) -> Result<ModuleSource>;
}

/// 文件系统加载器：在多个搜索根目录下查找模块。
///
/// 查找顺序：
///   1. 依次拼接 `root/name`；
///   2. 依次拼接 `root/name.{ext}`（默认尝试 `.wasm`、`.wat`）。
///
/// `name` 仅接受相对、无 `..` 的逻辑模块名；绝对路径和越界路径一律拒绝。
pub struct FileLoader {
    /// 搜索根目录列表。
    roots: Vec<PathBuf>,
    /// 自动尝试的扩展名集合。
    extensions: Vec<&'static str>,
}

impl FileLoader {
    /// 用一组搜索根构造文件加载器。
    pub fn new<I, P>(roots: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        Self {
            roots: roots
                .into_iter()
                .map(|root| {
                    let path = root.into();
                    fs::canonicalize(&path).unwrap_or(path)
                })
                .collect(),
            extensions: vec!["wasm", "wat"],
        }
    }

    /// 运行期追加一个搜索根 —— 支持「自定义加载路径」的核心入口。
    pub fn add_root(&mut self, p: impl Into<PathBuf>) {
        let path = p.into();
        self.roots.push(fs::canonicalize(&path).unwrap_or(path));
    }

    /// 按上述顺序解析逻辑名到磁盘路径，找不到返回 `None`。
    fn resolve(&self, name: &str) -> Option<PathBuf> {
        let requested = Path::new(name);
        if requested.is_absolute()
            || requested.components().any(|component| {
                matches!(
                    component,
                    std::path::Component::ParentDir
                        | std::path::Component::Prefix(_)
                        | std::path::Component::RootDir
                )
            })
        {
            return None;
        }
        for root in &self.roots {
            let candidates = std::iter::once(root.join(name)).chain(
                self.extensions
                    .iter()
                    .map(|ext| root.join(format!("{name}.{ext}"))),
            );
            for candidate in candidates {
                let canonical = match fs::canonicalize(&candidate) {
                    Ok(path) => path,
                    Err(_) => continue,
                };
                if canonical.starts_with(root) && canonical.is_file() {
                    return Some(canonical);
                }
            }
        }
        None
    }
}

impl Loader for FileLoader {
    fn load(&self, name: &str) -> Result<ModuleSource> {
        let path = self
            .resolve(name)
            .ok_or_else(|| anyhow!("在所有搜索根中均未找到模块 `{name}`"))?;
        let bytes = fs::read(&path).with_context(|| format!("读取文件 {}", path.display()))?;
        Ok(ModuleSource::new(name, bytes))
    }
}

/// 内存加载器：把字节直接塞进 HashMap，用于测试 / 动态注入 / 热替换。
///
/// 内部用 `RwLock` 保证并发安全；写入用独占锁、读取用共享锁。
pub struct MemoryLoader {
    entries: parking_lot::RwLock<std::collections::HashMap<String, Arc<Vec<u8>>>>,
}

impl MemoryLoader {
    pub fn new() -> Self {
        Self {
            entries: Default::default(),
        }
    }

    /// 注入或覆盖一个内存模块。
    /// 配合 [`crate::loader::HotSwap::reload`] 即可实现「无文件热插拔」。
    pub fn insert(&self, name: impl Into<String>, bytes: Vec<u8>) {
        self.entries.write().insert(name.into(), Arc::new(bytes));
    }
}

impl Default for MemoryLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Loader for MemoryLoader {
    fn load(&self, name: &str) -> Result<ModuleSource> {
        let bytes = self
            .entries
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("内存加载器中没有模块 `{name}`"))?;
        // `(*bytes).clone()` 复制内层 `Vec<u8>`：因为 `ModuleSource::new`
        // 会在内部重新 Arc，所以这里必须给一个独立的 Vec。
        Ok(ModuleSource::new(name, (*bytes).clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn file_loader_rejects_escape_paths() {
        let root = std::env::temp_dir().join(format!(
            "prism-wasm-loader-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("safe.wasm"), b"safe").unwrap();
        let loader = FileLoader::new([root.clone()]);
        assert!(loader.load("safe").is_ok());
        assert!(loader.load("../safe.wasm").is_err());
        assert!(loader.load(root.to_string_lossy().as_ref()).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

/// 链式加载器：按顺序依次尝试，第一个成功的胜出。
///
/// 典型用法：把内存加载器放前面（用于运行期注入），文件加载器放后面（用于
/// 持久化模块）。任何一个失败都会被记录为 `last_err`，全部失败时把最后一个
/// 错误返回给调用方。
pub struct ChainLoader {
    inner: Vec<Arc<dyn Loader>>,
}

impl ChainLoader {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// 把一个加载器追加到链尾，支持链式调用。
    pub fn push(mut self, l: Arc<dyn Loader>) -> Self {
        self.inner.push(l);
        self
    }
}

impl Default for ChainLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Loader for ChainLoader {
    fn load(&self, name: &str) -> Result<ModuleSource> {
        let mut last_err: Option<anyhow::Error> = None;
        for l in &self.inner {
            match l.load(name) {
                Ok(s) => return Ok(s),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("链上没有任何加载器")))
    }
}
