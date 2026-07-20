"""SDK 测试共享 fixtures。

主要职责：
    1. 把项目根目录加入 sys.path，让兼容层 ``from engines.base.xxx import`` 可用
    2. 提供跨测试文件复用的 fixtures（tmp_log_dir / mock_engine / event_loop）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ===== sys.path 设置 =====
# 项目根目录（d:\myProInstallDir\AIReliable），让 engines.base 兼容层可被 import
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===== 共享 fixtures =====


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """临时日志目录（每个测试函数独立）。"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture
def mock_engine():
    """构造一个最小可用的 BaseEngine 子类实例。

    所有抽象方法都返回 mock 响应，便于上层逻辑测试。
    """
    from aireliable_engine import (
        BaseEngine,
        CapabilitiesResponse,
        LoadResponse,
        LoraLoadResponse,
        LoraUnloadResponse,
        StatusResponse,
        UnloadResponse,
        make_final_frame,
        make_frame,
    )

    class _MockEngine(BaseEngine):
        """最小可用的 BaseEngine 子类，用于测试。"""

        def __init__(self):
            super().__init__(
                engine_id="mock",
                engine_name="Mock Engine",
                capability="text-to-image",
                engine_type="image",
            )
            self.load_called = False
            self.unload_called = False
            self.infer_called = False
            self.load_lora_called = False
            self.unload_lora_called = False

        async def _capabilities_impl(self):
            return CapabilitiesResponse(
                engine_id=self.engine_id,
                engine_name=self.engine_name,
                capability=self.capability,
                type=self.engine_type,
            )

        async def _load_impl(self, variant):
            self.load_called = True
            return LoadResponse(variant=variant, state="ready", vram_used="1GB")

        async def _unload_impl(self):
            self.unload_called = True
            return UnloadResponse(state="idle")

        async def _infer_impl(self, request, stream):
            self.infer_called = True
            await stream.put(make_frame(0.5, "denoising"))
            await stream.put(
                make_final_frame(
                    result={"image_urls": ["http://example.com/a.png"]},
                    timing={"total": 1.0},
                )
            )

        async def _status_impl(self):
            return StatusResponse(state=self.state_machine.state.value)

        async def _load_lora_impl(self, name, weight):
            self.load_lora_called = True
            return LoraLoadResponse(name=name, weight=weight, loaded_loras=[name])

        async def _unload_lora_impl(self, name):
            self.unload_lora_called = True
            return LoraUnloadResponse(name=name, loaded_loras=[])

    return _MockEngine()
