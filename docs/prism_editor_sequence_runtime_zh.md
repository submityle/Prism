# Prism 编辑器序列 Runtime 适配（§8）

本切片把序列内核的真实 SequenceFrame 接到 Prism 原生相机 ECS 和驻留音频图路径。它不是新的 gameplay 系统，也不解析事件名称。

## 接线

主模块启用 sequence-runtime feature 后公开 sequence_runtime 模块。主模块需要添加以下可选路径依赖，默认 feature 关闭：prism_ecs、prism_camera、prism_camera_ecs、prism_audio_assets、prism_audio_ecs、prism_math。测试还需要 prism_app、prism_audio_core；prism_audio_ecs 使用 default-features = false，避免强制 CPAL 设备。

宿主先创建 SequenceRuntimeAdapter，使用稳定的 SequenceId、完整 SequenceInstance path、TrackId、事件 ID 和宿主拥有的 Entity 建立绑定：

* 相机绑定写入真实 CameraPoseComponent、CameraIntentComponent、ViewHistoryComponent。相机切换可以绑定稳定事件，也可绑定 selector track，使 seek 不依赖跨越事件；seek/stop 会清除 temporal history。
* 音频绑定检查 AudioPlayer.asset 和 AudioAssetStore 的 decoder metadata。只接受有界、已驻留且有 frame count 的 mono/stereo 源；streaming 或未知长度返回 AudioUnavailable。时间换算为整数 tick * sample_rate / ticks_per_second + offset，不能表示、越界或溢出都失败。
* 音频 Start/Pause/Stop/Seek 是显式事件 action。暂停设置 AudioPlayer.playing = false 并保留 playhead；PlaybackSettings.paused 只是静音，不能替代这里的时间语义。应用后由现有 sync_audio_source_routes 负责真实 PCM graph 发布和 DSP 输出。

## 原子边界

RuntimeFence 同时保存 SequenceProjectToken、项目 revision 和 WorldId。prepare 只读世界并产生完整 before/after plan；apply 先逐项比较组件快照，再以无失败的 ECS 写入提交。rollback 只在 publication barrier 前、且组件仍等于 after-image 时恢复 before-image。相机 publication 或音频 graph hand-off 之后调用 publication_barrier，回滚会明确返回 Published，因为已对外部运行时可见的音频不能假装可逆。

RuntimeTransport::Seek 不分发 crossed events，只从 samples 和 selector tracks 重建状态；Paused 保留音频 source frame；Stopped 停止所有已绑定播放器并停用绑定相机。所有更新受 max changes/events 限制。

## 不在本切片内

没有实现相机算法、音频解码器、音频设备、streaming backend、项目 gameplay 角色映射或自定义 DSP。Erdos 的原生 app 只需在宿主调度中：播放器推进得到 SequenceFrame → prepare → apply → 相机 Publication/audio source-route 系统 → publication_barrier。若 graph hand-off 失败，其错误由既有 AudioGraphRouteState.last_error() 保持可见。
