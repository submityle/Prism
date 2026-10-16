# WGPU 同宿主设备接入与命令交接

## 实现契约

`WgpuDevice::adopt(adapter, device, queue)` 接收宿主原始设备请求返回的三元组的 clone，不创建第二个设备，不替换宿主错误或 device-loss callback。能力来自该 device 实际启用的 features / limits，而非 adapter 的能力上限。

每次 adoption 建立独立 RHI 资源表。资源 ID 只在该表及共享该表的 clone 内有效；不同 adoption 即便使用同一个原生 device，也不能交换 ID。wgpu 的原生设备比较不构成跨 Instance 的所有权证明，此接口不接纳任意原生资源注册。

`WgpuQueue::encode` 复用 `submit_tracked` 的命令 replay，只返回原生 command buffer，不调用 submit、write_buffer、poll 或 wait。宿主把它加入自己的提交，负责上传次序及真实完成通知。录制完成和提交完成都不是 GPU 完成，图分配必须保留到对应提交真正完成后再回收。

后端导出十类精确原生 handle：buffer、texture、texture view、sampler、shader module、bind-group layout、bind group、pipeline layout、render pipeline、compute pipeline。导出不分配替代资源，也不改变资源表；移除表项不会主动 destroy 已导出的原生 clone，stale ID 则不能解析到复用后的槽位。显式销毁仍被宿主或 RHI 使用的 buffer / texture 不符合该契约。

宿主负责 adopted device 的失效观测和恢复；旧帧完成回收后，为替换设备建立新的 bridge / 资源表。由 Prism 自行创建的设备继续使用既有 `take_device_lost` 路径。

## 硬件回归

2026-10-16，Apple M2，Metal，IntegratedGpu，`class=Hardware`，沙盒外执行：

```sh
PRISM_REQUIRE_GPU=1 cargo test -p prism_render_driver_wgpu \
  --test interop --test native_handles -- --nocapture --test-threads=1
```

结果：`interop` 9 passed，`native_handles` 11 passed，0 failed，0 ignored。严格模式下无适配器或非硬件适配器均失败，不计为跳过后的通过。

覆盖内容：

- 原始 handle 与实际启用能力保持不变，重复 adoption 的资源表互不影响。
- 宿主 device-loss callback 在接入及 bridge drop 后仍有效；独立设备路径仍报告 Prism device loss。
- transfer、compute、render 命令经宿主提交后才产生结果；提交前 staging 读回验证没有隐藏提交。
- 同一 Instance 中另一 logical device 的 queue 被原生验证拒绝。
- 十类 handle 的准确解析、移除 / stale generation 拒绝及原生 clone 生命周期。
- RHI 表释放后，保留的原生对象仍实际绘制、被 compute 采样，并读回预期像素。

这些测试证明同设备桥接与原生命令交接，不替代 Scene 全帧 Fabric 装配、外部 history 管理或跨后端性能验收。

## Scene composition 回归

同日严格硬件复跑发现：`resident_composition_ecs_prepares_executes_and_invalidates_targets` 的测试设备使用默认 features，生产 `prepare_visibility_buffers` 创建的 `Rg16Float` motion storage texture 因而被拒绝。

修复测试设备准入：显式要求 `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`，并检查 HDR、visibility、motion 的实际纹理用途与 kernel workgroup 限制；device request 与 Bevy 的格式能力请求对齐。没有修改 motion ABI、删减输出或移除原有绘制 / 读回断言。新增无特性与无 motion storage usage 的负向回归。

```sh
PRISM_REQUIRE_GPU=1 cargo test -p prism_render_scene --lib \
  shading::composition::ecs_tests -- --nocapture --test-threads=1
```

结果：3 passed，0 failed；其中真实 GPU 测试覆盖 HDR 读回、同帧仅消费一次、resize 使旧 bind group 失效及 stale-frame 拒绝。
