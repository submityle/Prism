//! 执行证明（Execution Attestation）—— 「零知识风格」回执模块。
//!
//! # 设计目标
//! 让一个 *验证者*（verifier）不需要重新执行 WASM 模块，就能确认：
//!   - 是 *哪一个* 模块（用 SHA-256 摘要唯一标识）
//!   - 在 *哪个输入* 下
//!   - 产生了 *哪个输出*
//!   - 期间发生了 *哪些宿主交互*（trace）
//!
//! # 与真正 ZKP 的关系
//! 这里 **没有** 实现完整的 SNARK / STARK 电路证明，而是采用最轻量的
//! 「承诺 + 签名」方案：把四个摘要拼起来再哈希一次得到 `commitment`，
//! 再由 `Attestor` 用密钥签名。它具备 ZKP 的两条核心性质中的一条：
//!   - ✅ **完整性 / 不可伪造性**：没有密钥的人无法伪造合法回执；
//!   - ❌ **零知识性**：本实现暴露了输入/输出摘要，但没有暴露原文。
//!
//! 想要升级为真正的 ZKP（risc0、Nova、SP1、Halo2 等），只需替换
//! [`Attestor::sign`] 与 [`Attestor::verify`]，承诺结构本身可以保留。

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

/// 一次沙箱调用的可验证回执。
///
/// 字段全部使用 32 字节定长哈希，便于序列化、上链或写入审计日志。
#[derive(Clone, Debug)]
pub struct Receipt {
    /// 被调用模块的 SHA-256 摘要（内容寻址 ID）。
    pub module_digest: [u8; 32],
    /// 输入参数序列化后的 SHA-256 摘要。
    pub input_digest: [u8; 32],
    /// 返回值序列化后的 SHA-256 摘要。
    pub output_digest: [u8; 32],
    /// 宿主调用 trace 的 SHA-256 摘要（用于审计宿主侧副作用）。
    pub trace_digest: [u8; 32],
    /// 聚合承诺 = H(module ‖ input ‖ output ‖ trace)。
    /// 验证者只需校验本字段即可同时锁定四要素。
    pub commitment: [u8; 32],
    /// 证明者对 `commitment` 的 HMAC-SHA256 签名（长度 32 字节）。
    pub signature: [u8; 32],
}

impl Receipt {
    /// 把承诺哈希转成十六进制串，方便日志输出 / 跨网络传输。
    pub fn commitment_hex(&self) -> String {
        hex::encode(self.commitment)
    }
}

/// 证明者（Attestor）—— 持有密钥、负责签发并验证回执。
///
/// 当前为 **对称密钥** 实现（同一个 `secret` 既用于签名也用于验证），
/// 仅适合单进程 / 受信任运行时场景。要做跨主机互信，应替换为
/// 非对称签名（如 ed25519、ECDSA）或真正的零知识证明系统。
pub struct Attestor {
    /// 证明者私密种子。生产环境务必替换为非对称签名密钥。
    secret: [u8; 32],
}

impl Attestor {
    /// 使用调用方提供的 32 字节密钥构造证明者。
    ///
    /// 如果把同一个 `secret` 注入到多个进程，它们便共享一个「信任域」，
    /// 可以互相验证对方签发的回执。
    pub fn new(secret: [u8; 32]) -> Self {
        Self { secret }
    }

    /// 进程级临时证明者。
    ///
    /// 用进程 PID + 固定盐派生密钥，**仅适用于演示 / 单进程内自验**。
    /// 进程重启后密钥会变，旧回执将无法再被验证。
    pub fn ephemeral() -> Self {
        let mut h = Sha256::new();
        h.update(std::process::id().to_le_bytes());
        h.update(b"prism-attestor");
        Self {
            secret: h.finalize().into(),
        }
    }

    /// 为一次执行结果签发回执。
    ///
    /// # 参数
    /// * `module_digest` —— 已通过加载器 / 引擎计算好的模块摘要；
    /// * `input`         —— 输入参数的字节序列化结果；
    /// * `output`        —— 返回值的字节序列化结果；
    /// * `trace`         —— 调用过程中宿主侧记录的副作用字节流。
    ///
    /// # 流程
    /// 1. 分别对 input / output / trace 求 SHA-256；
    /// 2. 把 `module || input || output || trace` 四段摘要再做一次 SHA-256，
    ///    得到聚合承诺 `commitment`；
    /// 3. 调用 [`sign`](Self::sign) 对 `commitment` 签名。
    pub fn issue(
        &self,
        module_digest: [u8; 32],
        input: &[u8],
        output: &[u8],
        trace: &[u8],
    ) -> Receipt {
        // 第 1 步：分别求三段摘要。
        let input_digest: [u8; 32] = Sha256::digest(input).into();
        let output_digest: [u8; 32] = Sha256::digest(output).into();
        let trace_digest: [u8; 32] = Sha256::digest(trace).into();

        // 第 2 步：拼接 4 段 32 字节摘要再哈希一次，作为聚合承诺。
        // 顺序固定为 module → input → output → trace，验证端必须保持一致。
        let mut h = Sha256::new();
        h.update(module_digest);
        h.update(input_digest);
        h.update(output_digest);
        h.update(trace_digest);
        let commitment: [u8; 32] = h.finalize().into();

        // 第 3 步：对承诺签名。
        let signature = self.sign(&commitment);

        Receipt {
            module_digest,
            input_digest,
            output_digest,
            trace_digest,
            commitment,
            signature,
        }
    }

    /// 校验回执签名是否合法。
    ///
    /// 注意：本函数 **只校验签名**，不会重算 `commitment`；
    /// 如果调用方想防御「篡改摘要后重新打包」的攻击，应在外部
    /// 用相同字段重新计算 `commitment` 并比较。
    pub fn verify(&self, r: &Receipt) -> bool {
        let commitment = Self::commitment(
            r.module_digest,
            r.input_digest,
            r.output_digest,
            r.trace_digest,
        );
        commitment == r.commitment && self.verify_signature(&r.commitment, &r.signature)
    }

    fn commitment(
        module: [u8; 32],
        input: [u8; 32],
        output: [u8; 32],
        trace: [u8; 32],
    ) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(module);
        h.update(input);
        h.update(output);
        h.update(trace);
        h.finalize().into()
    }

    fn sign(&self, msg: &[u8; 32]) -> [u8; 32] {
        let mut mac =
            Hmac::<Sha256>::new_from_slice(&self.secret).expect("HMAC accepts keys of any size");
        mac.update(msg);
        mac.finalize().into_bytes().into()
    }

    fn verify_signature(&self, msg: &[u8; 32], signature: &[u8; 32]) -> bool {
        let mut mac =
            Hmac::<Sha256>::new_from_slice(&self.secret).expect("HMAC accepts keys of any size");
        mac.update(msg);
        mac.verify_slice(signature).is_ok()
    }
}
