"""采样节点子包。

包含采样相关的节点：
    SamplerNode     - 通用采样器（任务 3.6）
    SchedulerNode   - 调度器选择（后续阶段）

导入子包即触发节点注册。
"""

from .sampler import SamplerNode

__all__ = ["SamplerNode"]
