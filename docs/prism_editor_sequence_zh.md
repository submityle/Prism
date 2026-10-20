# Timeline 嵌套 Sequence

本切片实现真实文档的嵌套时钟、通道采样和事件调度。文件范围仅为 timeline 的 src/sequence.rs、tests/sequence.rs 和本文。没有新增依赖、Bevy 类型或共享根文件修改。

## 公开集成接口

根模块由主线程增加：

```rust
mod sequence;
pub use sequence::{
    SequenceClip, SequenceClipId, SequenceError, SequenceEvent,
    SequenceFrame, SequenceId, SequenceInstance, SequenceLimits,
    SequencePlayer, SequenceProvider, SequenceRange, SequenceRegistry,
    SequenceSample,
};
```

`SequenceRegistry<'doc>` 拥有有界 BTreeMap 拓扑及 clip 数据，借用不可变的 TimelineDocument。insert(id, &document) 用实际 document codec 和资源策略验证，丢弃临时编码，不克隆或保存替代文档。SequenceId 和 SequenceClipId 保留宿主提供的 u128 身份；clip ID 在整个 registry 中唯一。通过 document(id) 得到原始借用，clips(id) 按 clip ID 顺序读取。插入重复身份、缺失引用、零重复次数、环、过深路径或映射溢出均返回错误；失败不会保留部分边或占用 clip ID。

构建流程为 new(limits) → insert 各文档 → add_clip(parent_id, clip) → SequencePlayer::new(&registry, root_id, range, mode)。文档先注册，随后接边；全部节点（包括非活动节点）都必须无环。player 借用 registry，Rust 借用约束阻止播放中编辑拓扑。加载新版本或编辑文档后，由宿主重建 registry/player 并明确恢复位置。

player 提供 tick/iteration/status/rate/fractional_tick、set_rate/play/pause、seek(tick)、advance(Duration)。seek 和 advance 返回 SequenceFrame；没有内部线程、系统时钟、文件或图形后端。root elapsed 精确分数与 Once/Loop 行为使用已有真实 Playback，并调用实际 timeline 采样；嵌套层也直接调用原始 document.timeline.sample/events_in/events_at。

## 区间与整数时钟

SequenceRange::new(start, end) 表示非空半开区间 [start, end)。end 是 u64，因而本接口最后可表示的活动 tick 是 u64::MAX - 1。clip 的 parent 区间采用父文档 tick，source_range 采用子文档 tick；两者无需同起点或同长度。

对活动父 tick p，计算：

```text
A = child.ticks_per_second × rate.numerator
B = parent.ticks_per_second × rate.denominator
q = floor((p - parent.start) × A / B)
L = source_range.end - source_range.start
iteration = q / L
local_tick = source_range.start + q % L
```

只在 p 位于 parent 区间且 q < repetitions × L 时活动。repetitions 必须为正有限 u64，父区间可提前裁剪重复；有限重复结束后即使父区间仍存在也不采样、不会末帧保持。每层先 floor 成整数再映射下一层，不能把多层倍率合并后只 floor 一次。重复来源区间的每次出现都携带明确 iteration。

乘除使用 u128 商/余数递推，避免中间乘积超过 u128 但正确商仍可表示时的假溢出。无法表示的真实商、事件逆映射或 root elapsed 运算明确报错。注册时验证 clip 全父区间的映射，因此无效极端倍率在播放前即可被拒绝。反向、零速和无限重复没有隐式解释；暂停使用 pause，速度使用现有正 PlaybackRate。

## 实际采样、事件与 provider

SequenceSample 含 sequence ID、原始 document 引用、完整实例路径、真实 local tick 和实际轨道值。相同 source 多次出现以及重叠 clip 保持独立路径，不自动混合或制造相机/音频规则。实例路径的每一项为稳定 clip ID 加本次 repetition；root 路径为空。

创建或成功 seek 后，下一次 Playing advance（可以是零 Duration）只派发目的地各活动实例的真实本地事件。seek 自身只采样、不扫过跳转路径；暂停期间 advance 不消费时间或派发事件。恢复播放不会重复派发已消费的起点。Once 到末 tick 后为 Finished，seek 后可重新开始；root Loop 的发生次数与每层有限 clip repetition 分别记录。

后续 Playing advance 按 (旧位置, 新位置] 遍历来源区间；快放即使跳过多个本地 tick、跨过完整子 clip 或重复边界，也收集实际越过的源事件。减速多次采样同一本地 tick 不会重复派发。事件向外逐层用 ceil 逆映射得到最早可观察 root tick；整数 root tick 相同的事件按完整路径、源 tick、源 event ID 确定排序。root 循环代次在最外层排序，因此大步 advance 与相同总时长的小步 advances 具有相同事件序列。

SequenceEvent 保存原始 TimelineEvent 引用（实际 id/name/tick）、源 document 引用、实例路径及 root_tick/root_iteration。没有生成替代事件或 camera/audio 特例。SequenceFrame::dispatch(&mut provider) 显式调用 SequenceProvider：先按路径派发目的地样本，再派发排序后的事件。provider 可根据通道及原始事件接入相机、音频或其他领域；这些领域行为不由 kernel 猜测。

provider 的 Error 与外部副作用归宿主所有；回调失败立即停止，之前的外部效果无法自动回滚。frame 评估成功后 player 已推进，宿主必须制定失败恢复与重复 dispatch 的去重策略。此切片没有宣称完成 native app 或真实音频/相机设备接线；测试中的接收器验证实际通道和事件负载经过公开回调。

## 有界工作与原子失败

SequenceLimits 对 registry 规模、图深度、单次访问数、事件出现次数、被采样轨道数及单文档 codec 施加上限。硬上限为 4096 个 sequence、65536 个 clip、64 层、每次一百万 visits/events/sample tracks、每文档 64 MiB 编码。默认分别为 256/4096/16/16384/4096/16384/8 MiB。零事件/轨道预算有效，零 visits 无效。

一次访问计入进入的节点与检查的 clip（包含不活动 clip）。逐个遍历完整重复的节点也消耗 visits，所以无事件的极端重复不会无限循环；超额返回 Limit。sample budget 对所有活动实例的源轨道数量计费，包含尚未产生首关键帧值的轨道，避免用空结果绕过工作预算。进入已有 root Playback 前也先检查 root 采样预算。输出实例数和实例路径内存同时受到 visits/depth 限制，实际通道维度和源关键帧数再受 document 策略限制。

advance/seek 在临时 root transport 和临时完整输出上验证全部运算、预算与真实采样；失败保留位置、循环代次、精确分数和起点事件游标，不返回部分事件批次。宿主可缩短 Duration、提高经过验证的预算或停止；不应静默丢弃超额事件。

## 验证证据与剩余接线

13 项实际回归覆盖三层跨频率采样与引用身份、快放跳过事件、减速去重、有限重复与父区间裁剪、嵌套来源已不活动但路径事件仍派发、seek/暂停/单 tick root loop、相同总时长分步与整步一致性、全局 ID/缺失引用/环/深度、registry/codec/采样/事件/工作预算、真实 cubic 溢出、宽有理乘除与真实溢出，以及显式 provider 负载与拒绝传播。独立对照测试逐个枚举源事件并向外投影，核对 27 组频率/倍率组合；生产代码按输入区间向内遍历，未共用测试 oracle。

验证命令为 cargo test --offline --manifest-path <owned-harness>/Cargo.toml --test sequence --jobs 2。独立 harness 直接编译本仓库实际 sequence.rs/tests/sequence.rs，并依赖真实 prism_editor_timeline 路径 API；无复制 kernel、替代文档或伪后端。13 项全通过，严格 cargo clippy --lib --test sequence --no-deps -- -D warnings 通过，harness 配置包含工作区显式 Clippy lint。自有隔离 target 验证后清理，不修改或清理其他任务目录。

主线程负责上述根导出。native 项目需选择文档快照生命周期、构建子序列拓扑、驱动 elapsed、接入领域 provider、处理显式错误，以及管理保存格式。当前 TimelineDocument 文件 codec 不包含 sequence 拓扑；本次实现不添加未分配的拓扑持久化或共享 UI 操作。
