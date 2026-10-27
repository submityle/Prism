//! 多沙箱注册表（`SandboxRegistry`）。
//!
//! 用一个共享的 [`SandboxEngine`] 承载多个互相隔离的沙箱（多租户）：
//!   - 每个沙箱有独立的 `Policy` / `Attestor` / Linker 配置；
//!   - Component 安装到引擎的命名空间下：内部用 `tenant::name` 做隔离；
//!   - 编译产物 / `InstancePre` 仍然按摘要在引擎层共享（同一份字节
//!     不论被哪个租户安装，编译只发生一次）。

use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::loader::{Loader, ModuleSource};
use crate::runtime::engine::SandboxEngine;
use crate::runtime::sandbox::{CallReceipt, Sandbox};

/// 多沙箱管理器。
pub struct SandboxRegistry {
    engine: Arc<SandboxEngine>,
    tenants: RwLock<HashMap<String, Arc<Sandbox>>>,
}

impl SandboxRegistry {
    pub fn new(engine: Arc<SandboxEngine>) -> Self {
        Self {
            engine,
            tenants: RwLock::new(HashMap::new()),
        }
    }

    /// 注册一个新沙箱（租户）。
    pub fn register(&self, sandbox: Sandbox) -> Result<Arc<Sandbox>> {
        let arc = Arc::new(sandbox);
        let mut tenants = self.tenants.write();
        if tenants.contains_key(arc.name()) {
            anyhow::bail!("租户 `{}` 已注册", arc.name());
        }
        tenants.insert(arc.name().to_string(), arc.clone());
        Ok(arc)
    }

    /// 按名字取出某租户沙箱。
    pub fn get(&self, name: &str) -> Option<Arc<Sandbox>> {
        self.tenants.read().get(name).cloned()
    }

    /// 注销一个租户。返回是否真有此租户。
    pub fn remove(&self, name: &str) -> bool {
        self.tenants.write().remove(name).is_some()
    }

    /// 当前所有租户名（拷贝出来，避免借用）。
    pub fn tenants(&self) -> Vec<String> {
        self.tenants.read().keys().cloned().collect()
    }

    pub fn engine(&self) -> &Arc<SandboxEngine> {
        &self.engine
    }

    /// 便捷方法：把字节以 `tenant::name` 的形式安装到引擎。
    /// 同一份字节即便被多个租户安装，引擎层也只会编译一次。
    pub fn install_for(&self, tenant: &str, module_name: &str, bytes: Vec<u8>) -> Result<[u8; 32]> {
        let sb = self
            .get(tenant)
            .ok_or_else(|| anyhow!("租户 `{tenant}` 未注册"))?;
        let qualified = format!("{tenant}::{module_name}");
        let src = ModuleSource::new(qualified, bytes);
        let cached = self.engine.install(src, sb.policy())?;
        Ok(cached.digest)
    }

    /// 便捷方法：通过加载器为某租户加载。
    pub fn install_for_loader(
        &self,
        tenant: &str,
        module_name: &str,
        loader: &dyn Loader,
    ) -> Result<[u8; 32]> {
        let src = loader.load(module_name)?;
        let sb = self
            .get(tenant)
            .ok_or_else(|| anyhow!("租户 `{tenant}` 未注册"))?;
        let qualified = format!("{tenant}::{module_name}");
        let src = ModuleSource {
            name: qualified,
            ..src
        };
        let cached = self.engine.install(src, sb.policy())?;
        Ok(cached.digest)
    }

    /// 便捷调用：等价于 `registry.get(tenant)?.call_typed("tenant::module", func, args)`。
    pub fn call<P, R>(
        &self,
        tenant: &str,
        module_name: &str,
        func: &str,
        args: P,
    ) -> Result<CallReceipt<R>>
    where
        P: wasmtime::component::ComponentNamedList
            + wasmtime::component::Lower
            + Send
            + Sync
            + 'static
            + std::fmt::Debug,
        R: wasmtime::component::ComponentNamedList
            + wasmtime::component::Lift
            + Send
            + Sync
            + 'static
            + Clone
            + std::fmt::Debug,
    {
        let sb = self
            .get(tenant)
            .ok_or_else(|| anyhow!("租户 `{tenant}` 未注册"))?;
        let qualified = format!("{tenant}::{module_name}");
        sb.call_typed(&qualified, func, args)
    }

    /// 异步便捷安装：编译在阻塞线程池完成，不阻塞 async 执行器。
    pub async fn install_for_async(
        &self,
        tenant: &str,
        module_name: &str,
        bytes: Vec<u8>,
    ) -> Result<[u8; 32]> {
        let sb = self
            .get(tenant)
            .ok_or_else(|| anyhow!("租户 `{tenant}` 未注册"))?;
        let qualified = format!("{tenant}::{module_name}");
        let src = ModuleSource::new(qualified, bytes);
        let cached = self.engine.install_async(src, sb.policy()).await?;
        Ok(cached.digest)
    }

    /// 异步便捷安装（通过加载器）：编译在阻塞线程池完成，不阻塞 async 执行器。
    pub async fn install_for_loader_async(
        &self,
        tenant: &str,
        module_name: &str,
        loader: &dyn Loader,
    ) -> Result<[u8; 32]> {
        let src = loader.load(module_name)?;
        let sb = self
            .get(tenant)
            .ok_or_else(|| anyhow!("租户 `{tenant}` 未注册"))?;
        let qualified = format!("{tenant}::{module_name}");
        let src = ModuleSource {
            name: qualified,
            ..src
        };
        let cached = self.engine.install_async(src, sb.policy()).await?;
        Ok(cached.digest)
    }

    /// 异步批量调用（同一租户 + 同一模块）：所有 args 在同一个 `spawn_blocking` 中执行。
    ///
    /// 适合百万级调用场景，配合分块使用：
    ///
    /// ```rust,ignore
    /// // 把 1_000_000 个请求按 chunk_size 分批，每批一个 spawn_blocking
    /// for chunk in all_args.chunks(chunk_size) {
    ///     registry.call_many_async("tenant", "module", "func", chunk.to_vec());
    /// }
    /// ```
    pub async fn call_many_async<P, R>(
        &self,
        tenant: &str,
        module_name: &str,
        func: &str,
        args_list: Vec<P>,
    ) -> Result<Vec<Result<CallReceipt<R>>>>
    where
        P: wasmtime::component::ComponentNamedList
            + wasmtime::component::Lower
            + Send
            + Sync
            + 'static
            + std::fmt::Debug,
        R: wasmtime::component::ComponentNamedList
            + wasmtime::component::Lift
            + Send
            + Sync
            + 'static
            + Clone
            + std::fmt::Debug,
    {
        let sb = self
            .get(tenant)
            .ok_or_else(|| anyhow!("租户 `{tenant}` 未注册"))?;
        let qualified = format!("{tenant}::{module_name}");
        sb.call_many_async(&qualified, func, args_list).await
    }

    /// 异步便捷调用：WASM 执行在阻塞线程池完成，不阻塞 async 执行器。
    pub async fn call_async<P, R>(
        &self,
        tenant: &str,
        module_name: &str,
        func: &str,
        args: P,
    ) -> Result<CallReceipt<R>>
    where
        P: wasmtime::component::ComponentNamedList
            + wasmtime::component::Lower
            + Send
            + Sync
            + 'static
            + std::fmt::Debug,
        R: wasmtime::component::ComponentNamedList
            + wasmtime::component::Lift
            + Send
            + Sync
            + 'static
            + Clone
            + std::fmt::Debug,
    {
        let sb = self
            .get(tenant)
            .ok_or_else(|| anyhow!("租户 `{tenant}` 未注册"))?;
        let qualified = format!("{tenant}::{module_name}");
        sb.call_typed_async(&qualified, func, args).await
    }
}

/// 引用计数版本的便捷别名。
pub type SharedRegistry = Arc<SandboxRegistry>;
