//! eBPF + WASM 双重可信链验证。
//!
//! # 架构
//! ```text
//!   WASM 执行
//!     │
//!     ├─ 用户态宿主函数调用 ──→ HostCallEvent[]（用户态声明）
//!     │                                │
//!     │                                ▼
//!     │                        EbpfBridge::verify()
//!     │                                ▲
//!     └─ 内核态 eBPF / ptrace ──→ SyscallEvent[]（内核观测）
//!
//! 一致 → VerifyResult { ok: true }
//! 不一致 → VerifyResult { ok: false, checks: [...] }
//! ```
//!
//! # 平台说明
//! - **Linux**：通过 eBPF `tracepoint/syscalls/sys_enter_*` 捕获真实系统调用，
//!   推荐配合 [`aya`](https://github.com/aya-rs/aya) crate 将事件注入本结构。
//! - **macOS / 其他**：eBPF 不可用，`ebpf_available()` 返回 `false`，
//!   仅做用户态一致性校验（`observed` 传空切片）。
//!
//! # 使用示例
//! ```rust,ignore
//! let bridge = EbpfBridge::new()
//!     .allow("host", "log",     &["write"])
//!     .allow("host", "env_get", &["read"])
//!     .allow("wasi", "fd_write",&["write", "writev"]);
//!
//! // 执行完毕后，从宿主层收集宿主调用事件
//! let declared = vec![
//!     HostCallEvent::new("host", "log", b"hello"),
//! ];
//!
//! // Linux 上通过 eBPF ring buffer 收集内核事件；其他平台传 &[]
//! let observed: Vec<SyscallEvent> = collect_from_ebpf(); // 用户实现
//!
//! let result = bridge.verify(&declared, &observed);
//! assert!(result.ok);
//! println!("声明链哈希: {}", hex::encode(result.declared_chain));
//! ```

use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

/// 宿主调用事件（用户态侧）。
///
/// 在宿主函数实现中，每次被 WASM 调用时向 `StoreState::trace` 追加一条记录，
/// 执行完毕后将其解析为 `HostCallEvent` 列表。
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HostCallEvent {
    /// 宿主模块名，例如 `"host"` 或 `"wasi_snapshot_preview1"`。
    pub module: String,
    /// 函数名，例如 `"log"` 或 `"fd_write"`。
    pub func: String,
    /// 参数内容的 SHA-256 摘要（避免在事件中携带原始参数）。
    pub args_digest: [u8; 32],
}

impl HostCallEvent {
    /// 构造一条宿主调用事件，自动计算参数摘要。
    pub fn new(module: impl Into<String>, func: impl Into<String>, args: &[u8]) -> Self {
        Self {
            module: module.into(),
            func: func.into(),
            args_digest: Sha256::digest(args).into(),
        }
    }
}

/// 系统调用事件（内核态侧，来自 eBPF / ptrace 捕获）。
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SyscallEvent {
    /// 系统调用名称，例如 `"write"`、`"read"`、`"openat"`。
    pub syscall: String,
    /// 触发该系统调用的进程 PID（多进程场景用于过滤）。
    pub pid: u32,
    /// 系统调用返回值（负数通常表示错误码）。
    pub retval: i64,
}

impl SyscallEvent {
    pub fn new(syscall: impl Into<String>, pid: u32, retval: i64) -> Self {
        Self {
            syscall: syscall.into(),
            pid,
            retval,
        }
    }
}

/// 单项检查结果。
#[derive(Clone, Debug)]
pub struct CheckItem {
    /// 检查类别，例如 `"policy"` / `"kernel-verify"` / `"chain-integrity"`。
    pub check: &'static str,
    /// 是否通过。
    pub passed: bool,
    /// 详情描述（用于日志/审计）。
    pub detail: String,
}

/// 完整的验证结果。
#[derive(Clone, Debug)]
pub struct VerifyResult {
    /// 所有检查均通过时为 `true`。
    pub ok: bool,
    /// 逐项检查结果。
    pub checks: Vec<CheckItem>,
    /// 宿主调用声明链的 SHA-256（对所有 `HostCallEvent` 序列哈希）。
    ///
    /// 可写入 [`Receipt`](crate::security::attestation::Receipt) 或上链，
    /// 用于事后审计时重算验证。
    pub declared_chain: [u8; 32],
    /// 内核观测事件序列的 SHA-256（无内核数据时为全零）。
    pub observed_chain: [u8; 32],
}

impl VerifyResult {
    /// 声明链的前 8 字节十六进制（用于简短日志）。
    pub fn declared_chain_short(&self) -> String {
        hex::encode(&self.declared_chain[..8])
    }
}

/// eBPF + WASM 双重可信链验证器。
///
/// 通过 [`allow`](EbpfBridge::allow) 注册「宿主函数 → 允许的系统调用集合」策略，
/// 然后调用 [`verify`](EbpfBridge::verify) 进行交叉校验。
pub struct EbpfBridge {
    /// 策略映射：`(module, func)` → 允许的系统调用名集合。
    policy: HashMap<(String, String), HashSet<String>>,
    /// 当前平台是否支持 eBPF。
    ebpf_available: bool,
}

impl EbpfBridge {
    pub fn new() -> Self {
        Self {
            policy: HashMap::new(),
            ebpf_available: Self::detect_ebpf(),
        }
    }

    /// 注册一个宿主函数允许触发的系统调用白名单（构造器链式调用）。
    ///
    /// 例如：
    /// ```rust,ignore
    /// bridge.allow("host", "log", &["write"])
    ///       .allow("host", "read_file", &["openat", "read", "close"])
    /// ```
    pub fn allow(
        mut self,
        module: impl Into<String>,
        func: impl Into<String>,
        syscalls: &[&str],
    ) -> Self {
        self.policy
            .entry((module.into(), func.into()))
            .or_default()
            .extend(syscalls.iter().map(|s| s.to_string()));
        self
    }

    /// 当前平台是否探测到 eBPF 支持。
    pub fn ebpf_available(&self) -> bool {
        self.ebpf_available
    }

    /// 执行双重可信链验证。
    ///
    /// # 参数
    /// - `declared`：从 WASM 宿主层收集到的 [`HostCallEvent`] 列表（用户态声明）；
    /// - `observed`：从 eBPF / ptrace / dtrace 捕获到的 [`SyscallEvent`] 列表（内核观测）。
    ///   在非 Linux 平台或无权限时，传入 `&[]` 即可——仅执行用户态策略校验。
    ///
    /// # 检查项
    /// 1. **policy**：每个宿主调用是否在策略白名单中；
    /// 2. **kernel-verify**：内核观测到的系统调用是否与对应宿主函数的允许集合一致；
    /// 3. **chain-integrity**：声明链与观测链的摘要（供外部审计使用）。
    pub fn verify(&self, declared: &[HostCallEvent], observed: &[SyscallEvent]) -> VerifyResult {
        let mut checks = Vec::new();

        // 1. 计算声明链哈希（无论是否有策略都计算）
        let declared_chain = self.declared_chain_hash(declared);
        let observed_chain = if observed.is_empty() {
            [0u8; 32]
        } else {
            self.syscall_chain_hash(observed)
        };

        // 2. 策略校验：每个宿主调用是否在白名单中
        if self.policy.is_empty() {
            checks.push(CheckItem {
                check: "policy",
                passed: true,
                detail: format!(
                    "未配置策略，跳过白名单检查（{} 个宿主调用）",
                    declared.len()
                ),
            });
        } else {
            let mut violations = Vec::new();
            for ev in declared {
                let key = (ev.module.clone(), ev.func.clone());
                if !self.policy.contains_key(&key) {
                    violations.push(format!("{}::{}", ev.module, ev.func));
                }
            }
            if violations.is_empty() {
                checks.push(CheckItem {
                    check: "policy",
                    passed: true,
                    detail: format!("全部 {} 个宿主调用均在策略白名单内", declared.len()),
                });
            } else {
                checks.push(CheckItem {
                    check: "policy",
                    passed: false,
                    detail: format!("以下宿主调用不在白名单中: {}", violations.join(", ")),
                });
            }
        }

        // 3. 内核校验：观测到的系统调用是否与声明一致
        if observed.is_empty() {
            checks.push(CheckItem {
                check: "kernel-verify",
                passed: true,
                detail: if self.ebpf_available {
                    "eBPF 可用但未提供内核侧事件（跳过内核校验）".to_string()
                } else {
                    format!(
                        "当前平台不支持 eBPF（{}），仅做用户态校验",
                        std::env::consts::OS
                    )
                },
            });
        } else {
            // 收集所有观测到的系统调用名
            let observed_set: HashSet<&str> = observed.iter().map(|e| e.syscall.as_str()).collect();

            let mut violations = Vec::new();
            for ev in declared {
                let key = (ev.module.clone(), ev.func.clone());
                if let Some(allowed) = self.policy.get(&key) {
                    // 找出观测到但不在该函数允许集合中的系统调用
                    for &sc in &observed_set {
                        if !allowed.contains(sc) {
                            violations.push(format!(
                                "syscall `{}` 被观测到，但 {}::{} 不允许此调用",
                                sc, ev.module, ev.func
                            ));
                        }
                    }
                }
            }

            if violations.is_empty() {
                checks.push(CheckItem {
                    check: "kernel-verify",
                    passed: true,
                    detail: format!(
                        "内核观测 {} 个独立 syscall 与用户态声明一致",
                        observed_set.len()
                    ),
                });
            } else {
                checks.push(CheckItem {
                    check: "kernel-verify",
                    passed: false,
                    detail: violations.join("; "),
                });
            }
        }

        // 4. 链完整性：输出两条链的摘要供外部审计
        checks.push(CheckItem {
            check: "chain-integrity",
            passed: true,
            detail: format!(
                "declared_chain={}, observed_chain={}",
                hex::encode(&declared_chain[..8]),
                if observed.is_empty() {
                    "N/A (no kernel events)".to_string()
                } else {
                    hex::encode(&observed_chain[..8])
                }
            ),
        });

        let ok = checks.iter().all(|c| c.passed);
        VerifyResult {
            ok,
            checks,
            declared_chain,
            observed_chain,
        }
    }

    /// 对宿主调用声明序列求哈希（顺序敏感）。
    fn declared_chain_hash(&self, events: &[HostCallEvent]) -> [u8; 32] {
        let mut h = Sha256::new();
        for ev in events {
            h.update(ev.module.as_bytes());
            h.update(b"::");
            h.update(ev.func.as_bytes());
            h.update(&ev.args_digest);
        }
        h.finalize().into()
    }

    /// 对系统调用观测序列求哈希（顺序敏感）。
    fn syscall_chain_hash(&self, events: &[SyscallEvent]) -> [u8; 32] {
        let mut h = Sha256::new();
        for ev in events {
            h.update(ev.syscall.as_bytes());
            h.update(&ev.pid.to_le_bytes());
            h.update(&ev.retval.to_le_bytes());
        }
        h.finalize().into()
    }

    /// 探测当前平台是否支持 eBPF。
    fn detect_ebpf() -> bool {
        #[cfg(target_os = "linux")]
        {
            // 简单探测：/proc/sys/kernel/bpf_stats_enabled 存在则内核有 BPF 子系统
            std::path::Path::new("/proc/sys/kernel/bpf_stats_enabled").exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
}

impl Default for EbpfBridge {
    fn default() -> Self {
        Self::new()
    }
}
