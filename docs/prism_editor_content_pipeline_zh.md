# Prism 编辑器内容管线适配器

本模块是编辑器与真实 prism_asset_build_cli 之间的有界异步进程适配层。它只构造固定的 --json import、build、package、validate、inspect 参数，不经过 shell，也不解析人类可读输出。

## 接线与授权

主模块应添加 pub mod pipeline，并在 crate 依赖中提供 serde = { workspace = true } 与 serde_json = { workspace = true }。UI 的变更入口必须先由 ValidationReceipt::authorize 得到 ValidatedCatalog（操作为 Cook），再调用 Pipeline::submit_checked。该入口按 AssetId 和 catalog 中的完整 logical path 精确核对请求，拒绝授权后的路径替换。submit_unvalidated 仅供诊断或故障测试，不是 UI 入口。

异步完成后调用 PipelineCompletion::ensure_current；它用 catalog 的不可伪造快照 token 做 revision fence。它不能撤销 CLI 已经写入的 store/package 副作用，因此宿主仍应在提交前决定如何处理过期结果。

## 边界与协议

PipelineLimits 限制并发句柄、请求/路径字节、stdout+stderr、JSONL 行和事件字节、事件数、结果数组项、PNG 源文件大小以及总超时。每次读取每个通道最多一个固定缓冲块，避免输出洪泛阻塞取消或超时检查。JSONL 必须按 sequence 从 command_started 开始并以单个 terminal event 结束；终态 payload、命令名、schema、CLI exit code 和操作系统 exit status 都必须一致，成功结果只在全部条件成立时发布。

Unix 上 stdout/stderr 使用非阻塞 socket 对，监督线程在取消、超时、协议错误、输出超限和 drop 时 kill 并 wait 直接子进程。此层只保证直接子进程回收；CLI 当前不派生外部孙进程，若未来改变该约束需由宿主增加进程组策略。取消不承诺回滚已经完成的资产发布。

请求只接受项目根目录下由 catalog logical path 派生的 PNG 源路径；目录组件中的 symlink 会被拒绝。包 mapping 文件、签名参数、任意 argv 和任意环境变量不属于受检 API，避免绕过 catalog 身份核对。

## 验证边界

tests/pipeline.rs 的真实测试使用真实临时 PNG、真实 CLI 二进制（由 PRISM_ASSET_BUILD_CLI 指定，默认 /private/tmp/prism-asset-build-check/debug/prism_asset_build_cli），顺序执行 import、package、validate、inspect，并检查强类型终态。没有用 echo/mock 冒充成功链路。测试另检查授权后的 logical path 替换被拒绝。

建议在依赖已构建且磁盘充足时运行：

    CARGO_TARGET_DIR=/private/tmp/prism-editor-joint-cr1ODK CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_BUILD_JOBS=2 PRISM_ASSET_BUILD_CLI=/private/tmp/prism-asset-build-check/debug/prism_asset_build_cli cargo test -p prism_editor_content --test pipeline

故障 fixture、取消/reap 和严格 Clippy 应在主模块导出与依赖接线后运行；本实现不把未运行的这些证据宣称为通过。
