"""通用镜像选择器（v0.2.7+ 通用下载器重构）。

统一 core/.../hf_mirrors.py::HFMirrorSelector 和
sdk/.../downloader.py::_EngineHFMirrorSelector 的重复实现（R1）。

设计原则：
    - 支持多种镜像类型（HF 镜像、PyTorch 镜像等）
    - 速度监控 + 自动切换
    - 不可达冷却
    - 进程级单例

规范来源：universal-downloader-refactor-analysis.md §3.2.3
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ===== 速度监控参数 =====

SLOW_SPEED_THRESHOLD_BPS = 100 * 1024  # 100 KB/s
SLOW_DURATION_SECONDS = 60.0  # 持续 60 秒慢则切换
UNAVAILABLE_COOLDOWN_SECONDS = 600.0  # 不可达冷却 10 分钟
SPEED_WINDOW_SIZE = 20  # 滑动窗口采样数


@dataclass
class MirrorSource:
    """镜像源数据类。

    Attributes:
        url: 镜像 URL
        name: 镜像名称（日志用）
        available: 是否可用
        unavailable_until: 不可用截止时间戳（0 表示可用）
        speed_samples: 速度采样窗口
    """

    url: str
    name: str = ""
    available: bool = True
    unavailable_until: float = 0.0
    speed_samples: list = field(default_factory=list)

    def is_available(self) -> bool:
        """检查是否可用（过了冷却期）。"""
        if self.unavailable_until > 0 and time.time() < self.unavailable_until:
            return False
        return True

    def mark_unavailable(self, cooldown: float = UNAVAILABLE_COOLDOWN_SECONDS) -> None:
        """标记为不可用。"""
        self.available = False
        self.unavailable_until = time.time() + cooldown
        logger.warning("镜像 %s 标记为不可用，冷却 %d 秒", self.name or self.url, int(cooldown))

    def report_speed(self, speed_bps: float) -> None:
        """上报速度采样。"""
        self.speed_samples.append(speed_bps)
        if len(self.speed_samples) > SPEED_WINDOW_SIZE:
            self.speed_samples = self.speed_samples[-SPEED_WINDOW_SIZE:]

    def get_avg_speed(self) -> float:
        """获取平均速度。"""
        if not self.speed_samples:
            return 0.0
        return sum(self.speed_samples) / len(self.speed_samples)

    def is_slow(self) -> bool:
        """检查是否持续慢速。"""
        if len(self.speed_samples) < SPEED_WINDOW_SIZE:
            return False
        avg = self.get_avg_speed()
        return avg < SLOW_SPEED_THRESHOLD_BPS


class MirrorSelector:
    """通用镜像选择器。

    统一 HFMirrorSelector 和 _EngineHFMirrorSelector，支持多种镜像类型。

    使用示例：
        # HF 镜像
        hf_selector = MirrorSelector(
            mirrors=[
                MirrorSource(url="https://huggingface.co", name="HF 官方"),
                MirrorSource(url="https://hf-mirror.com", name="HF 镜像"),
            ]
        )

        # PyTorch 镜像
        torch_selector = MirrorSelector(
            mirrors=[
                MirrorSource(url="https://download.pytorch.org/whl", name="PyTorch 官方"),
                MirrorSource(url="https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl", name="清华镜像"),
            ]
        )
    """

    def __init__(
        self,
        mirrors: list = None,
        slow_threshold: float = SLOW_SPEED_THRESHOLD_BPS,
        slow_duration: float = SLOW_DURATION_SECONDS,
    ):
        """
        Args:
            mirrors: 镜像源列表（MirrorSource 对象）
            slow_threshold: 慢速阈值（bytes/sec）
            slow_duration: 持续慢速时长（秒）
        """
        self._mirrors: list[MirrorSource] = mirrors or []
        self._current_index: int = 0
        self._slow_threshold = slow_threshold
        self._slow_duration = slow_duration
        self._slow_since: Optional[float] = None

    @property
    def mirrors(self) -> list:
        """镜像列表。"""
        return self._mirrors

    @property
    def current(self) -> Optional[MirrorSource]:
        """当前镜像。"""
        if not self._mirrors:
            return None
        return self._mirrors[self._current_index]

    def get_current_url(self) -> str:
        """获取当前镜像 URL。"""
        source = self.current
        return source.url if source else ""

    def get_preferred(self) -> Optional[MirrorSource]:
        """获取首选可用镜像（跳过不可用的）。"""
        for i, mirror in enumerate(self._mirrors):
            if mirror.is_available():
                self._current_index = i
                return mirror
        # 所有镜像都不可用，返回第一个（让调用方尝试）
        return self._mirrors[0] if self._mirrors else None

    def report_speed(self, speed_bps: float) -> None:
        """上报当前镜像速度。"""
        if self.current:
            self.current.report_speed(speed_bps)

        # 检查是否需要切换
        if self.current and self.current.is_slow():
            if self._slow_since is None:
                self._slow_since = time.time()
            elif time.time() - self._slow_since >= self._slow_duration:
                logger.warning(
                    "镜像 %s 持续慢速（平均 %.1f KB/s），触发切换",
                    self.current.name or self.current.url,
                    self.current.get_avg_speed() / 1024,
                )
                self.switch_to_next()
        else:
            self._slow_since = None

    def switch_to_next(self) -> Optional[MirrorSource]:
        """切换到下一个镜像。"""
        if not self._mirrors:
            return None

        # 标记当前为不可用
        if self.current:
            self.current.mark_unavailable()

        # 找下一个可用的
        for i in range(len(self._mirrors)):
            idx = (self._current_index + 1 + i) % len(self._mirrors)
            if self._mirrors[idx].is_available():
                self._current_index = idx
                self._slow_since = None
                logger.info(
                    "切换到镜像: %s",
                    self._mirrors[idx].name or self._mirrors[idx].url,
                )
                return self._mirrors[idx]

        # 所有镜像都不可用，重置为第一个
        self._current_index = 0
        for mirror in self._mirrors:
            mirror.available = True
            mirror.unavailable_until = 0.0
        logger.warning("所有镜像都不可用，重置并使用第一个")
        return self.current

    def reset(self) -> None:
        """重置所有镜像状态。"""
        for mirror in self._mirrors:
            mirror.available = True
            mirror.unavailable_until = 0.0
            mirror.speed_samples.clear()
        self._current_index = 0
        self._slow_since = None


# ===== 预定义镜像列表 =====

DEFAULT_HF_MIRRORS: list = [
    MirrorSource(url="https://huggingface.co", name="HF 官方"),
    MirrorSource(url="https://hf-mirror.com", name="HF 国内镜像"),
]

DEFAULT_PYTORCH_MIRRORS: list = [
    MirrorSource(url="https://download.pytorch.org/whl", name="PyTorch 官方"),
    MirrorSource(url="https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl", name="清华镜像"),
    MirrorSource(url="https://mirrors.aliyun.com/pytorch-wheels", name="阿里镜像"),
]


# ===== 进程级单例 =====

_hf_selector: Optional[MirrorSelector] = None
_pytorch_selector: Optional[MirrorSelector] = None


def get_hf_mirror_selector() -> MirrorSelector:
    """获取 HF 镜像选择器单例。"""
    global _hf_selector
    if _hf_selector is None:
        _hf_selector = MirrorSelector(mirrors=DEFAULT_HF_MIRRORS)
    return _hf_selector


def get_pytorch_mirror_selector() -> MirrorSelector:
    """获取 PyTorch 镜像选择器单例。"""
    global _pytorch_selector
    if _pytorch_selector is None:
        _pytorch_selector = MirrorSelector(mirrors=DEFAULT_PYTORCH_MIRRORS)
    return _pytorch_selector
