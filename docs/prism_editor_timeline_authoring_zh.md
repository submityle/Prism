# 时间轴命令、文档与 Loom 编辑面板

更新日期：2026-10-21。对应设计 §8、§14、§15。`prism_editor_timeline` 使用整数 tick 和稳定 u128 身份，`prism_editor_ui` 提供可挂载的真实轨道、关键帧、曲线与播放控件；全部依赖 Prism，不依赖 Bevy。

## 权威模型与可撤销命令

`TimelineEditor` 独占模型，外部只读。`TimelineEdit` 覆盖增加/删除轨道、按 ID 插入或替换/删除关键帧、增加/删除事件。`execute` 按顺序原子执行，记录逆操作；中途失败反向恢复已执行前缀。Undo/Redo 同样执行命令数据，不为每次编辑复制完整时间轴。删除轨道的逆必须保留该轨道全部关键帧，这是实际受影响的数据。

默认限制为 4096 轨道、全局 100 万关键帧、每轨 64 通道、65536 事件、每文本 4096 UTF-8 字节、每命令 4096 项编辑、256 项历史和 16 MiB 历史逻辑负载。历史为正向/逆向负载的较大值预留，确保后续 Undo/Redo 可以执行；不是进程 RSS 计量。新命令确定有效且自身放得下后才淘汰最旧历史。失败不改变数据、索引、revision、精确 token 或 redo。完全相同的 PutKey 命令不制造历史。

`TimelineSnapshotToken` 使用不可复用的 Arc 身份，不能以同值 revision 代替。独立文档、分叉和旧视图即使数字修订相同也不能互相授权。关键帧按 tick 排序并维护 ID 索引；事件按 tick/ID 排序，轨道有稳定 ID 范围查询。新 ID 在当前最大值后检查递增，耗尽显式报错。

Step、Linear 和 Cubic Hermite 都对真实通道采样。Hermite 切线是每 tick 的值斜率，左右关键帧分别提供 outgoing/incoming；向量维度和有限性先检查。线性插值采用加权端点，避免先计算端点差导致溢出。不能表示为有限 f64 的样本报错；极端 Hermite 中间表达式溢出也会拒绝，不把无穷值传入绘制或运行时。

## 可回放命令与完整保存格式

`TimelineCommand` 的 magic 是 `PRSTLN01`，包含 label 和六种编辑。`TimelineDocument` 的 magic 是 `PRSTLD01`，包含非零 u64 ticks_per_second、完整轨道及事件。两种格式均为确定性小端版本 1：u32 数量/UTF-8 长度、u128 身份、u64 tick 和精确有限 f64 位模式，包括负零。完整文档不会被伪装成一条 AddTrack/AddEvent 命令，因此文档大小和对象数独立于历史字节与每命令编辑数。

文档 `encode/decode(limits, max_bytes)` 单独限制文件字节；编码借用轨道，解码在孤立候选中完成。长度/剩余字节检查先于分配，拒绝未知版本/tag、截断、尾随数据、坏 UTF-8、NaN/Inf、重复关键帧 ID/tick、事件 ID 和零频率。加载成功后由宿主显式安装候选，不对活动模型部分覆盖。实际异步磁盘存储另见对应 storage 模块的专用文档。

## Loom 操作与真实图形

`TimelinePanelController::build(&editor)` 返回 `BoundTimelinePanel`。宿主挂载其 `element()`，从实际布局节点取得命中 key，再调用 `handle_event`。view 同时绑定控制器身份和文档 token；另一控制器、旧状态和新文档的事件都拒绝。解析或事务失败保留原视图可用性。

面板已有轨道名/通道数、轨道增删、当前位置插帧、事件增删、关键帧选择、整数时间/向量/插值/双侧切线编辑、通道切换、成组移动/删除及 Undo/Redo。成组移动先撤出全部选中键再放到目标 tick，允许使用组内腾出的时间；外部碰撞使整组回滚。选择上限为 1024，Extend 只查询同轨 anchor 与目标间的区间。

每视图最多 32 轨、1024 个关键帧标记、128 个事件和 128 个曲线样本。轨道范围/时间窗口可操作，超出标记容量会显式显示提示。key marker、playhead 和 curve point 均为真实 keyed Loom 盒，采用现有样式 token。u64 时间先整数相减再投影像素，避免接近 u64::MAX 时相邻 tick 被浮点舍入合并。曲线采样使用真实 Timeline 值，有限点才投影。当前曲线用有界采样点绘制；直接鼠标拖键和切线手柄尚未实现。

## 播放与宿主时钟

Play/Pause/Stop、Once/Loop、起止 tick、频率和正有理数速率直接驱动 `Playback`。默认范围为完整 u64 区间、60 ticks/s、Once、每次最多 1024 个事件；实际序列宿主应显式设置结束位置。默认大范围保留完整作者时间的 Seek 能力，不推断项目片段长度。

`advance(&editor, Duration)` 由宿主单调时钟驱动，返回真实样本及按循环代次排序的借用事件，播放不改作者模型/token/history。暂停保留小数 tick；Stop 原子采样起点后暂停并重置起点事件；改配置验证后夹紧当前位置并暂停。修改速率重置小数余量，相同速率保留。Once 到终点后 Finished，Stop/Seek 后才能重新开始。详细整数余量和闭区间循环语义见 `prism_editor_timeline_playback_zh.md`。

事件超限、时间计算溢出或采样错误均不消耗 elapsed；调用者应分片重试或显式停止，不能跳过失败区间继续伪称已完整派发。原生时钟唤醒、运行时消费、文件存储、子序列/相机/音频域轨道仍需各自接入与验收。

## 已执行验证

2026-10-21 在主仓库独立 target、offline、jobs=2、debug=0、incremental=false 执行：

```sh
cargo test --offline -p prism_editor_timeline -p prism_editor_ui --all-targets
cargo clippy --offline -p prism_editor_timeline -p prism_editor_ui --all-targets --no-deps -- -D warnings
```

时间轴 37 项通过（3 单元、8 authoring、4 document、17 playback、5 wire），UI 76 项通过，其中 timeline_panel 9 项。覆盖事务/历史预算和回滚、全部编码截断、完整大事件文档独立于命令限制、精确身份和浮点位、实际布局及蓝色选择/曲线像素、u64 高位时间精度、真实循环事件与样本、失败重试和 pause/stop/速率。严格 Clippy 通过。此次是默认 features 测试，不能作为 GPU、原生窗口事件或所有具体域序列器完成的证据。
