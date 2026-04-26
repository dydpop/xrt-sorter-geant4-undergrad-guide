# 符号与变量说明

| 符号/变量 | 含义 |
| --- | --- |
| XRT | X-ray Transmission，X 射线透射 |
| `event_id` | 仿真事件编号 |
| `detector_edep_keV` | 单个 event 中探测器能量沉积，单位 keV |
| `detector_gamma_entries` | 单个 event 中 gamma 进入探测器的计数 |
| `primary_gamma_entries` | 单个 event 中 primary gamma 到达探测器的计数 |
| `PHOTONS_PER_SAMPLE` | 每个虚拟样本包含的 event 数，当前为 100 |
| `sample_id` | 虚拟样本编号 |
| `primary_transmission_rate` | 主 gamma 透射率 |
| `mean_detector_edep_keV` | 样本级平均探测器能量沉积 |
| `detector_gamma_rate` | 样本级探测器 gamma 命中率 |
| `group_label` | 粗粒度吸收组标签 |
| `low_absorption` | 低吸收组 |
| `high_absorption` | 高吸收组 |
| accuracy | 测试集中预测正确样本的比例 |
