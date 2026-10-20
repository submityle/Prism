# Timeline 整数 tick 播放与运行时消费契约

本切片实现调用方驱动的通用正向播放器，源数据直接来自现有 `Timeline`，采样调用 `Timeline::sample`，事件查询调用 `Timeline::events_in`。不读取系统时钟，不依赖 Bevy，也不规定轨道对应何种游戏对象、组件或事件动作。调用方传入实际经过的 `Duration`，可以使用自己的单调时钟或固定步长。

## 主线程集成

本切片只拥有 `src/playback.rs`、`tests/playback.rs` 和本文档。共享 `lib.rs`、Cargo、editor、wire、UI、app 由主线程管理。需要在 `pkg/prism_editor_timeline/src/lib.rs` 添加：

```rust
mod playback;
pub use playback::{
    Playback, PlaybackConfig, PlaybackError, PlaybackEvent, PlaybackFrame,
    PlaybackMode, PlaybackRate, PlaybackStatus,
};
```

没有新增 Cargo 依赖；实现只使用 `core` 和 `alloc`。本文不将独立 harness 的通过等同于真实 crate、UI 或 app 已完成挂接。

## API 与时间模型

`Playback::new(PlaybackConfig)` 验证正整数 `ticks_per_second` 及闭区间 `[start_tick, end_tick]`，初始暂停、位于起点、速率为一。区间允许单 tick，也允许完整 `[0, u64::MAX]`。`PlaybackMode::Once` 到终点后进入 `Finished`；`PlaybackMode::Loop` 按该区间重复。`max_events` 是每次 advance 可返回的事件出现次数上限，重复循环中的同一事件分别计数；零预算只允许不遇到事件的推进。

`PlaybackRate::new(numerator, denominator)` 要求分子分母均非零并约分。只有正向正速率；暂停使用 `pause()`。固定速率下，每次消费时间的精确公式为：

```text
D = 1_000_000_000 * rate.denominator
N = elapsed.as_nanos() * ticks_per_second * rate.numerator + old_remainder
ticks = N / D
new_remainder = N % D
```

实现用商/余数长乘法计算该公式，不要求中间的纳秒乘积能够放入 u128；最终 tick 商、位置计算和循环代次若超出 u128，明确返回 `ArithmeticOverflow`。不会饱和、环绕或偷偷丢时间。`fractional_tick()` 返回余数分子与分母。固定速率下，未遇到错误且按相同策略消费事件时，拆分 Duration 与合并 Duration 得到相同位置、代次、余数和事件序列。

`set_rate(rate)` 是明确的时间重定相操作：更换速率时清零旧的不足一 tick 余数，整数位置及事件游标保持；设置相同的约分速率保留余数。需要跨速率边界保存亚 tick 相位的上层应在整数 tick 边界切换速率；本实现不承诺变速点的分数相位连续。

`play()` 开始或恢复播放；Finished 状态返回 false，需先 seek。`pause()` 保留位置、余数、代次和待发事件状态。暂停或 Finished 时调用 advance 仍返回当前位置真实 samples，但忽略传入的时间、不发送事件；调用方应传每次播放更新的增量时间，不能恢复时再补入暂停期间的墙钟时间。

## 事件边界、循环和 seek

创建后，首个成功的 playing advance 先发送起点事件，即使 Duration 为零；随后只发送新访问 tick 的事件。后续零时间调用不会重复发送。暂停调用不会消耗这次起点交付资格。

普通前进的事件窗口为 `(old_tick, destination_tick]`。循环周期是 `end - start + 1` 个 tick，先访问终点，经过一个 tick 再访问下一圈起点。例如区间 [5, 7] 从起点推进七 tick 的访问序列为：

```text
tick:       5 6 7 | 5 6 7 | 5 6
iteration:  0 0 0 | 1 1 1 | 2 2
```

每次访问按 tick 顺序输出事件，同 tick 按现有稳定事件 ID 排序。区间外事件不会发送。输出的 `PlaybackEvent { event, iteration }` 引用真实事件，保留 ID、tick、name；循环代次区分同一事件的不同出现。单 tick 循环周期为一，每推进一 tick 就产生该 tick 的下一代事件。

`seek(&timeline, tick)` 先验证边界并真实采样；成功后重置余数和循环代次，不发送跳过路径中的事件。它将目标 tick 事件重新设为待发，由下一次成功的 playing advance 交付。Playing 保持 Playing，Paused 保持 Paused，Finished 变成 Paused。失败不改变任何状态。重复 seek 是显式的新播放路径，可以再次发送相同目标事件。

Once 到终点会交付终点事件一次、清零余数并进入 Finished；超出终点的时间不累计到后续播放。单 tick Once 在首个成功 playing advance 即完成。

## 预算、原子性与成本

一次 advance 先计算候选时间、位置和代次，将事件路径拆成最多四段：待发起点、当前圈后缀、完整中间圈、最终圈前缀。使用有界扫描预检每段事件数量，并在展开前检查整圈重复次数是否超出剩余预算。没有事件的整圈直接跳过；不会按每个 tick 或每个空圈迭代。超预算立即返回 `EventBudgetExceeded { limit }`。

预检通过后，调用真实 sample；只有采样和所有检查成功，才生成完整事件输出并提交候选播放器。预算不足、整数溢出、插值产生非有限值等错误，均不改变整数 tick、余数、代次、状态或起点事件游标，不返回部分事件。失败后本次 Duration 仍未消费；调用方必须保存它并重试、拆成可接受的小步，或者显式停止。不能当作成功而丢弃时间。若单 tick 的事件本身超过预算，拆时间也无法解决，需要新的预算配置或修改源数据。

事件 Vec 最多 `max_events` 个借用项，不复制事件名称。事件工作量受返回预算约束，再加固定次数的 BTree 区间查询。samples 是当前 Timeline 的真实有序 `BTreeMap<TrackId, Vec<f64>>`：首 key 之前的轨道不出现，末 key 之后保持末值，插值错误透传为 `PlaybackError::Sample(TimelineError)`。samples 的工作和内存仍由源轨道数、维度决定，事件预算不限制采样总量；需要使用 authoring 的 `TimelineLimits` 或调用方输入限制约束源数据。本播放器不包含异步队列，也不承诺进程内存分配失败后的恢复。

## 泛用运行时消费

```rust
use core::time::Duration;
use prism_editor_timeline::{Playback, PlaybackConfig, PlaybackMode, Timeline};

fn update(timeline: &Timeline) -> Result<(), prism_editor_timeline::PlaybackError> {
    let mut transport = Playback::new(PlaybackConfig {
        ticks_per_second: 48_000,
        start_tick: 0,
        end_tick: 95_999,
        mode: PlaybackMode::Loop,
        max_events: 256,
    })?;
    transport.play();
    let frame = transport.advance(timeline, Duration::from_millis(16))?;
    for (track_id, channels) in &frame.samples {
        // 由运行时自己的绑定表将稳定 track_id 映射到消费端。
        let _ = (track_id, channels);
    }
    for occurrence in &frame.events {
        // 保持输出顺序，将 event.id/name 和 iteration 交给领域适配器。
        let _ = (occurrence.event, occurrence.iteration);
    }
    Ok(())
}
```

真实循环应长期持有同一个 Playback，逐次传入实际时间增量。上例仅演示一次调用。运行时负责稳定 track ID 与目标通道的绑定、样本应用以及事件执行，播放器不认识固定游戏类型，也不直接写入任何外部状态。

一次成功 advance 的输出应由消费端按顺序交付一次；播放器状态提交和外部系统写入并非跨系统事务。若消费端要重试写入，需要自己的事务/幂等账本，不能再次推进播放器后声称同一批已被重放。seek 会重置 iteration，clone 也会复制播放游标，因此跨 seek、重启、克隆分支的幂等键还需要调用方的 transport/document epoch，不能只使用 (event ID, iteration)。

Frame 的事件借用源 Timeline，借用存续期间 Rust 不允许可变编辑该源。播放器自身不保存文档 revision/snapshot token。跨帧换文档或编辑源数据前，运行时应在自己的稳定快照边界暂停并显式 seek/建立新 epoch，或使用自己的不可变快照；它不会追补在已消费 tick 上后来新增的事件，也不宣称提供文档代际围栏。

## 验证与边界

独立 harness 以当前 timeline lib 快照及真实 editor/wire/playback 源文件编译，仅在 harness 补上述导出；不修改共享 lib。playback 集成测试覆盖真实二维插值和事件引用、正有理速率余数、Duration 拆分、暂停恢复、seek、闭区间多圈、逐 tick 事件 oracle、预算回滚及分步重试、Once 完成、单 tick 周期、u64 全区间与最大端点、海量空圈、整数溢出及 sample 失败的完整回滚。

本切片运行的验证结果：17 个 playback 测试、3 个现有内核单元测试通过；复制 workspace lint 配置后，独立 harness 的 `cargo clippy --offline --all-targets -- -D warnings` 通过。验证使用单 job、关闭 debug/incremental 的独占小 target。真实 crate 与 UI 的整体测试由主线程添加导出后执行；本文不将该集成工作标为完成。
