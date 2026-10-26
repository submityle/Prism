//! 节点执行体抽象。
//!
//! 提供两套 trait：
//!
//! - [`NodeRunner`]（同步）—— 旧版 API；实现简单、无运行时依赖。
//! - [`AsyncNodeRunner`]（异步）—— 节点内可 `.await`；可通过
//!   [`InvokeCtx::chunk_tx`] 发送 [`Chunk`] 流式输出。
//!
//! 同一个 [`crate::registry::NodeDefinition`] 上挂的 [`RunnerKind`] 决定 VM 走
//! 同步还是异步执行路径。
//! - 同步 VM ([`crate::Vm::run_entry`] / [`crate::Vm::run_all`]) 只能跑
//!   [`RunnerKind::Sync`]；遇到 [`RunnerKind::Async`] 会返回
//!   [`crate::VsError::AsyncRunnerInSyncVm`]。
//! - 异步 VM ([`crate::Vm::run_entry_async`] / [`crate::Vm::run_all_async`])
//!   两种 runner 都能跑。

use crate::effect::EffectSpec;
use crate::error::VsResult;
use crate::registry::host::{Chunk, InvokeCtx};
use crate::registry::library::ExecNext;
use crate::value::Value;
use async_trait::async_trait;
use std::sync::Arc;

/// 同步节点执行体。Pure 节点应忽略 `exec_next` 返回值并返回 [`ExecNext::End`]。
pub trait NodeRunner: Send + Sync {
    fn invoke(
        &self,
        inputs: &[Value],
        outputs: &mut [Value],
        ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext>;
}

/// 异步节点执行体。
///
/// 与同步版同形：返回 [`ExecNext`] 决定 exec 流走向；通过 `outputs` 写回 data。
/// 长时操作应定期检查 `ctx.cancel.is_cancelled()`，并在流式模式下通过
/// `ctx.chunk_tx` 推 [`Chunk::Delta`] / [`Chunk::Progress`]。
#[async_trait]
pub trait AsyncNodeRunner: Send + Sync {
    async fn invoke(
        &self,
        inputs: &[Value],
        outputs: &mut [Value],
        ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext>;
}

/// 同步闭包适配器。
pub struct FnRunner<F>(pub F)
where
    F: Fn(&[Value], &mut [Value], &mut InvokeCtx<'_>) -> VsResult<ExecNext> + Send + Sync;

impl<F> NodeRunner for FnRunner<F>
where
    F: Fn(&[Value], &mut [Value], &mut InvokeCtx<'_>) -> VsResult<ExecNext> + Send + Sync,
{
    fn invoke(
        &self,
        inputs: &[Value],
        outputs: &mut [Value],
        ctx: &mut InvokeCtx<'_>,
    ) -> VsResult<ExecNext> {
        (self.0)(inputs, outputs, ctx)
    }
}

/// 节点持有的执行体形态。
#[derive(Clone)]
pub enum RunnerKind {
    Sync(Arc<dyn NodeRunner>),
    Async(Arc<dyn AsyncNodeRunner>),
    /// A declarative external side effect executed by the host effect runtime.
    Effect(EffectSpec),
}

impl RunnerKind {
    pub fn sync<R: NodeRunner + 'static>(r: R) -> Self {
        Self::Sync(Arc::new(r))
    }
    pub fn r#async<R: AsyncNodeRunner + 'static>(r: R) -> Self {
        Self::Async(Arc::new(r))
    }
    pub fn effect(spec: EffectSpec) -> Self {
        Self::Effect(spec)
    }
    pub fn is_async(&self) -> bool {
        matches!(self, Self::Async(_))
    }
}

/// 兼容别名（旧版本 API）。
pub type SharedRunner = Arc<dyn NodeRunner>;

/// 流式辅助：在 `ctx.chunk_tx` 上发送 chunk；非流式模式下静默丢弃。
pub async fn send_chunk(ctx: &InvokeCtx<'_>, chunk: Chunk) {
    if let Some(tx) = ctx.chunk_tx {
        // 通道关闭意味着上游取消订阅，节点不必报错。
        let _ = tx.send(chunk).await;
    }
}
