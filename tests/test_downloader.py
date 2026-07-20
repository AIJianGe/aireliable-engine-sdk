"""下载器测试。

mock huggingface_hub / httpx，不实际下载。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aireliable_engine.downloader import (
    BaseEngineDownloader,
    DEFAULT_HF_MIRRORS,
    DownloadResult,
    DownloadSpec,
    HFSnapshotDownloader,
    _EngineHFMirrorSelector,
)


# ===== 数据类测试 =====


class TestDownloadSpec:
    """DownloadSpec 数据类字段测试。"""

    def test_required_fields(self):
        """必填字段。"""
        spec = DownloadSpec(
            file_id="f1",
            filename="model.bin",
            target_dir=Path("/data"),
            source_type="huggingface_file",
        )
        assert spec.file_id == "f1"
        assert spec.filename == "model.bin"
        assert spec.target_dir == Path("/data")
        assert spec.source_type == "huggingface_file"

    def test_default_values(self):
        """默认值。"""
        spec = DownloadSpec(
            file_id="f1",
            filename="",
            target_dir=Path("/data"),
            source_type="manual",
        )
        assert spec.source_config == {}
        assert spec.expected_size == 0
        assert spec.description == ""

    def test_with_all_fields(self):
        """所有字段。"""
        spec = DownloadSpec(
            file_id="f1",
            filename="model.bin",
            target_dir=Path("/data"),
            source_type="huggingface_snapshot",
            source_config={"repo_id": "Baidu/ERNIE"},
            expected_size=1024 * 1024 * 100,
            description="ERNIE 模型",
        )
        assert spec.source_config == {"repo_id": "Baidu/ERNIE"}
        assert spec.expected_size == 1024 * 1024 * 100
        assert spec.description == "ERNIE 模型"


class TestDownloadResult:
    """DownloadResult 数据类字段测试。"""

    def test_required_fields(self):
        """必填字段。"""
        r = DownloadResult(file_id="f1", local_path=Path("/data/a.bin"))
        assert r.file_id == "f1"
        assert r.local_path == Path("/data/a.bin")
        assert r.bytes_downloaded == 0
        assert r.success is True
        assert r.error is None

    def test_failure_result(self):
        """失败结果。"""
        r = DownloadResult(
            file_id="f1",
            local_path=Path("/data/a.bin"),
            bytes_downloaded=100,
            success=False,
            error="network error",
        )
        assert r.success is False
        assert r.error == "network error"
        assert r.bytes_downloaded == 100


# ===== BaseEngineDownloader 测试 =====


class _ConcreteDownloader(BaseEngineDownloader):
    """具体子类，用于测试。"""

    async def list_download_specs(self) -> list[DownloadSpec]:
        return []

    async def download_one(
        self,
        spec: DownloadSpec,
        progress_cb=None,
        cancel_event=None,
    ) -> DownloadResult:
        return DownloadResult(file_id=spec.file_id, local_path=spec.target_dir / spec.filename)


class TestBaseEngineDownloader:
    """BaseEngineDownloader 抽象基类测试。"""

    def test_cannot_instantiate_abstract(self):
        """BaseEngineDownloader 是 ABC，不能直接实例化。"""
        with pytest.raises(TypeError):
            BaseEngineDownloader()  # type: ignore[abstract]

    def test_incomplete_subclass_raises(self):
        """子类必须实现 list_download_specs / download_one。"""

        class _Incomplete(BaseEngineDownloader):
            async def list_download_specs(self):
                return []

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """完整子类可实例化。"""
        d = _ConcreteDownloader()
        assert isinstance(d, BaseEngineDownloader)

    @pytest.mark.asyncio
    async def test_get_hf_endpoint_default_none(self):
        """get_hf_endpoint 默认返回 None。"""
        d = _ConcreteDownloader()
        assert d.get_hf_endpoint() is None

    @pytest.mark.asyncio
    async def test_ensure_pipeline_config_default_empty(self):
        """ensure_pipeline_config 默认返回空字符串。"""
        d = _ConcreteDownloader()
        result = await d.ensure_pipeline_config("some_model")
        assert result == ""

    @pytest.mark.asyncio
    async def test_concrete_methods_callable(self):
        """具体方法可调用。"""
        d = _ConcreteDownloader()
        specs = await d.list_download_specs()
        assert specs == []
        result = await d.download_one(
            DownloadSpec(file_id="f", filename="x", target_dir=Path("/tmp"), source_type="manual")
        )
        assert result.file_id == "f"


# ===== HFSnapshotDownloader 测试 =====


class TestHFSnapshotDownloaderConstruction:
    """HFSnapshotDownloader 构造测试。"""

    def test_basic_construction(self):
        """构造（用 repo_id / allow_patterns）。"""
        d = HFSnapshotDownloader(
            repo_id="Baidu/ERNIE-Image",
            allow_patterns=["*.json", "vae/*"],
        )
        assert d.repo_id == "Baidu/ERNIE-Image"
        assert d.allow_patterns == ["*.json", "vae/*"]
        assert d.ignore_patterns == []
        assert d.max_workers == 1

    def test_default_mirrors(self):
        """未指定镜像时用 DEFAULT_HF_MIRRORS。"""
        d = HFSnapshotDownloader(repo_id="r", allow_patterns=["*"])
        assert d.mirror_endpoints == DEFAULT_HF_MIRRORS
        # mirror_endpoint 兼容字段
        assert d.mirror_endpoint == DEFAULT_HF_MIRRORS[0]

    def test_mirror_endpoints_list(self):
        """mirror_endpoints 列表正确。"""
        endpoints = ["https://a.com", "https://b.com"]
        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=endpoints,
        )
        assert d.mirror_endpoints == endpoints
        assert d.mirror_endpoint == endpoints[0]

    def test_v01_single_mirror_endpoint_compat(self):
        """兼容 v0.1 单一 mirror_endpoint 参数。"""
        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoint="https://hf-mirror.com",
        )
        # 应被包装为单元素列表
        assert d.mirror_endpoints == ["https://hf-mirror.com"]
        assert d.mirror_endpoint == "https://hf-mirror.com"

    def test_mirror_endpoints_overrides_single(self):
        """mirror_endpoints 优先级高于 mirror_endpoint。"""
        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoint="https://old.com",
            mirror_endpoints=["https://new1.com", "https://new2.com"],
        )
        assert d.mirror_endpoints == ["https://new1.com", "https://new2.com"]

    def test_ignore_patterns(self):
        """ignore_patterns 正确存储。"""
        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            ignore_patterns=["*.bin", "transformer/*"],
        )
        assert d.ignore_patterns == ["*.bin", "transformer/*"]

    def test_max_workers(self):
        """max_workers 正确存储。"""
        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            max_workers=4,
        )
        assert d.max_workers == 4


class TestHFSnapshotDownloaderCheckReachable:
    """check_reachable 测试（mock httpx）。"""

    @pytest.mark.asyncio
    async def test_all_unreachable_returns_false(self, monkeypatch):
        """所有镜像都不可达时返回 False。"""

        class _FakeResponse:
            def __init__(self, status_code: int):
                self.status_code = status_code

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def head(self, url):
                raise ConnectionError("unreachable")

        # mock httpx.AsyncClient
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com", "https://b.com"],
        )
        result = await d.check_reachable(timeout=1.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_first_reachable_returns_true(self, monkeypatch):
        """第一个镜像可达时返回 True。"""

        class _FakeResponse:
            def __init__(self, status_code: int):
                self.status_code = status_code

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                self.head_count = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def head(self, url):
                self.head_count += 1
                return _FakeResponse(200)

        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        result = await d.check_reachable(timeout=1.0)
        assert result is True


# ===== _EngineHFMirrorSelector 测试 =====


class TestEngineHFMirrorSelectorInit:
    """_EngineHFMirrorSelector 初始化测试。"""

    def test_default_mirrors(self):
        """默认用 DEFAULT_HF_MIRRORS。"""
        sel = _EngineHFMirrorSelector()
        assert sel.mirrors == DEFAULT_HF_MIRRORS

    def test_custom_mirrors(self):
        """自定义镜像列表。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        assert sel.mirrors == ["https://a.com", "https://b.com"]

    def test_empty_mirrors_falls_back_to_default(self):
        """空镜像列表时回退到 DEFAULT_HF_MIRRORS。

        源码 ``list(mirrors or DEFAULT_HF_MIRRORS)`` 中空列表是 falsy，
        会被替换为默认镜像，所以 ValueError 是死代码（保留覆盖以记录此事实）。
        """
        sel = _EngineHFMirrorSelector(mirrors=[])
        assert sel.mirrors == DEFAULT_HF_MIRRORS

    def test_get_current_endpoint_initial(self):
        """初始 endpoint 是镜像列表第一个。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        assert sel.get_current_endpoint() == "https://a.com"


class TestEngineHFMirrorSelectorSpeed:
    """速度上报 + 切换测试。"""

    def test_report_speed_fast_no_switch(self):
        """速度快时不切换。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        # 上报高速
        sel.report_speed(1024 * 1024)  # 1 MB/s
        assert sel.should_switch() is False

    def test_report_speed_slow_triggers_switch(self):
        """速度持续慢速触发切换。"""
        import time as _t

        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        # 上报慢速并模拟时间流逝
        # 通过修改 _slow_since 模拟已持续 60s+
        sel.report_speed(10 * 1024)  # 10 KB/s 远低于阈值
        # should_switch 现在应是 False（时间还没到）
        assert sel.should_switch() is False
        # 手动把 _slow_since 设为 61s 前
        cur = sel.mirrors[sel._current_idx]
        sel._slow_since[cur] = _t.time() - 61.0
        # 现在应该 True
        assert sel.should_switch() is True

    def test_report_speed_recovery_resets(self):
        """速度恢复后 should_switch 返回 False。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        # 先慢速
        sel.report_speed(10 * 1024)
        cur = sel.mirrors[sel._current_idx]
        assert sel._slow_since[cur] is not None
        # 再高速
        sel.report_speed(1024 * 1024)
        assert sel._slow_since[cur] is None
        assert sel.should_switch() is False


class TestEngineHFMirrorSelectorFailure:
    """失败上报测试。"""

    def test_report_failure_switches_immediately(self):
        """report_failure 立即切换。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        assert sel.get_current_endpoint() == "https://a.com"
        new_ep = sel.report_failure("conn error")
        assert new_ep == "https://b.com"
        assert sel.get_current_endpoint() == "https://b.com"

    def test_report_failure_marks_unavailable(self):
        """report_failure 标记当前镜像不可用。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        sel.report_failure("err")
        assert sel._marked_unavailable["https://a.com"] is True

    def test_report_result_success_clears_unavailable(self):
        """report_result 成功时清除不可用标记。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        sel.report_failure("err")  # 标记 a.com 不可用
        assert sel._marked_unavailable["https://a.com"] is True
        # 切到 b.com 后报成功
        sel.report_result(success=True)
        # b.com 没有被标记
        assert sel._marked_unavailable["https://b.com"] is False

    def test_report_result_failure_no_change(self):
        """report_result 失败时不改变标记。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        sel.report_result(success=False)
        # 没有镜像被标记
        assert not any(sel._marked_unavailable.values())


class TestEngineHFMirrorSelectorSwitchToNext:
    """switch_to_next 测试。"""

    def test_switch_to_next_advances(self):
        """switch_to_next 推进到下一个可用镜像。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com", "https://c.com"])
        assert sel.get_current_endpoint() == "https://a.com"
        new_ep = sel.switch_to_next()
        assert new_ep == "https://b.com"
        new_ep = sel.switch_to_next()
        assert new_ep == "https://c.com"

    def test_switch_to_next_marks_old_unavailable(self):
        """switch_to_next 标记旧镜像不可用。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        sel.switch_to_next()
        assert sel._marked_unavailable["https://a.com"] is True


# ===== download_to_cache / download_subdir mock 测试 =====


class TestDownloadToCacheMocked:
    """download_to_cache 测试（mock snapshot_download）。"""

    @pytest.mark.asyncio
    async def test_download_to_cache_success(self, monkeypatch, tmp_path: Path):
        """成功下载返回路径。"""

        def fake_snapshot_download(**kwargs):
            return str(tmp_path / "snapshot")

        # mock huggingface_hub.snapshot_download
        import sys
        fake_module = MagicMock()
        fake_module.snapshot_download = fake_snapshot_download
        fake_module.constants = MagicMock()
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*.json"],
            mirror_endpoints=["https://a.com"],
        )
        result = await d.download_to_cache()
        assert result == str(tmp_path / "snapshot")

    @pytest.mark.asyncio
    async def test_download_to_cache_all_fail_raises(self, monkeypatch):
        """所有镜像都失败时抛 RuntimeError。"""

        def fake_snapshot_download(**kwargs):
            raise ConnectionError("connection refused")

        import sys
        fake_module = MagicMock()
        fake_module.snapshot_download = fake_snapshot_download
        fake_module.constants = MagicMock()
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*.json"],
            mirror_endpoints=["https://a.com", "https://b.com"],
        )
        with pytest.raises(RuntimeError, match="下载 HF 仓库 snapshot 失败"):
            await d.download_to_cache()

    @pytest.mark.asyncio
    async def test_download_to_cache_no_huggingface_hub(self, monkeypatch):
        """未安装 huggingface_hub 时抛 RuntimeError。"""
        import sys
        # 模拟 ImportError
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*.json"],
            mirror_endpoints=["https://a.com"],
        )
        with pytest.raises((RuntimeError, ImportError)):
            await d.download_to_cache()

    @pytest.mark.asyncio
    async def test_download_to_cache_cancel_before_start(self, monkeypatch):
        """启动前取消抛 RuntimeError。"""
        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*.json"],
            mirror_endpoints=["https://a.com"],
        )
        cancel_event = asyncio.Event()
        cancel_event.set()
        with pytest.raises(RuntimeError, match="下载已取消"):
            await d.download_to_cache(cancel_event=cancel_event)


# ===== download_subdir mock 测试 =====


class TestDownloadSubdirMocked:
    """download_subdir 测试（mock huggingface_hub）。"""

    @pytest.mark.asyncio
    async def test_download_subdir_success(self, monkeypatch, tmp_path: Path):
        """download_subdir 成功下载子目录。"""

        # 准备 mock
        class _FakeTreeItem:
            def __init__(self, path, size):
                self.path = path
                self.size = size

        class _FakeHfApi:
            def __init__(self, endpoint=None):
                self.endpoint = endpoint

            def list_repo_files(self, repo_id):
                return ["vae/config.json", "vae/model.safetensors", "other.txt"]

            def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False):
                return [
                    _FakeTreeItem("vae/config.json", 1000),
                    _FakeTreeItem("vae/model.safetensors", 5000),
                ]

        # 创建实际存在的本地文件，避免触发 fallback copy
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        def fake_hf_hub_download(repo_id, filename):
            local = cache_dir / filename.replace("/", "_")
            local.write_bytes(b"fake content")
            return str(local)

        import sys
        fake_module = MagicMock()
        fake_module.HfApi = _FakeHfApi
        fake_module.hf_hub_download = fake_hf_hub_download
        fake_module.constants = MagicMock()
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        d = HFSnapshotDownloader(
            repo_id="Baidu/ERNIE",
            allow_patterns=["vae/*"],
            mirror_endpoints=["https://a.com"],
        )
        paths = await d.download_subdir("vae")
        assert len(paths) == 2
        assert "vae_config.json" in paths[0]
        assert "vae_model.safetensors" in paths[1]

    @pytest.mark.asyncio
    async def test_download_subdir_no_files_returns_empty(self, monkeypatch):
        """子目录无文件返回空列表。"""

        class _FakeHfApi:
            def __init__(self, endpoint=None):
                self.endpoint = endpoint

            def list_repo_files(self, repo_id):
                return ["other.txt"]  # 不含 vae/ 前缀

            def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False):
                return []

        def fake_hf_hub_download(repo_id, filename):
            return f"/cache/{filename}"

        import sys
        fake_module = MagicMock()
        fake_module.HfApi = _FakeHfApi
        fake_module.hf_hub_download = fake_hf_hub_download
        fake_module.constants = MagicMock()
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        paths = await d.download_subdir("vae")
        assert paths == []

    @pytest.mark.asyncio
    async def test_download_subdir_cancel_midway(self, monkeypatch, tmp_path: Path):
        """下载过程中取消抛 RuntimeError。"""

        class _FakeTreeItem:
            def __init__(self, path, size):
                self.path = path
                self.size = size

        class _FakeHfApi:
            def __init__(self, endpoint=None):
                self.endpoint = endpoint

            def list_repo_files(self, repo_id):
                return ["vae/a.bin", "vae/b.bin", "vae/c.bin"]

            def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False):
                return [_FakeTreeItem("vae/a.bin", 100), _FakeTreeItem("vae/b.bin", 200)]

        # 实际创建文件，避免触发 fallback
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        def fake_hf_hub_download(repo_id, filename):
            local = cache_dir / filename.replace("/", "_")
            local.write_bytes(b"fake")
            return str(local)

        import sys
        fake_module = MagicMock()
        fake_module.HfApi = _FakeHfApi
        fake_module.hf_hub_download = fake_hf_hub_download
        fake_module.constants = MagicMock()
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        cancel_event = asyncio.Event()
        # 在 progress_cb 中 set 取消事件，第一个文件下载完成后会触发
        def cancel_cb(downloaded, total, speed):
            cancel_event.set()

        with pytest.raises(RuntimeError, match="下载被取消"):
            await d.download_subdir("vae", progress_cb=cancel_cb, cancel_event=cancel_event)

    @pytest.mark.asyncio
    async def test_download_subdir_no_huggingface_hub(self, monkeypatch):
        """未安装 huggingface_hub 时抛 RuntimeError。"""
        import sys
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        with pytest.raises((RuntimeError, ImportError)):
            await d.download_subdir("vae")

    @pytest.mark.asyncio
    async def test_download_subdir_list_repo_files_all_fail(self, monkeypatch):
        """list_repo_files 所有镜像都失败抛 RuntimeError。"""

        class _FakeHfApi:
            def __init__(self, endpoint=None):
                self.endpoint = endpoint

            def list_repo_files(self, repo_id):
                raise ConnectionError("conn refused")

            def list_repo_tree(self, repo_id, **kwargs):
                return []

        def fake_hf_hub_download(repo_id, filename):
            return f"/cache/{filename}"

        import sys
        fake_module = MagicMock()
        fake_module.HfApi = _FakeHfApi
        fake_module.hf_hub_download = fake_hf_hub_download
        fake_module.constants = MagicMock()
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com", "https://b.com"],
        )
        with pytest.raises(RuntimeError, match="list_repo_files 所有镜像均失败"):
            await d.download_subdir("vae")

    @pytest.mark.asyncio
    async def test_download_subdir_tree_fails_degrades_to_zero_size(self, monkeypatch, tmp_path: Path):
        """list_repo_tree 失败时降级用 0 字节估算（不抛错）。"""

        class _FakeHfApi:
            def __init__(self, endpoint=None):
                self.endpoint = endpoint

            def list_repo_files(self, repo_id):
                return ["vae/a.bin"]

            def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False):
                raise RuntimeError("tree failed")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        def fake_hf_hub_download(repo_id, filename):
            local = cache_dir / filename.replace("/", "_")
            local.write_bytes(b"fake")
            return str(local)

        import sys
        fake_module = MagicMock()
        fake_module.HfApi = _FakeHfApi
        fake_module.hf_hub_download = fake_hf_hub_download
        fake_module.constants = MagicMock()
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        paths = await d.download_subdir("vae")
        assert len(paths) == 1


# ===== _download_file_with_retry 测试 =====


class TestDownloadFileWithRetry:
    """_download_file_with_retry 测试。"""

    @pytest.mark.asyncio
    async def test_symlink_failure_triggers_fallback(self, monkeypatch, tmp_path: Path):
        """Windows symlink 失败时触发 fallback copy。"""
        # 构造假的 HF 缓存目录（路径与 _fallback_copy_blob_to_snapshot 一致：
        # Path.home() / ".cache" / "huggingface" / "hub" / "models--{repo_id}")
        repo_id = "Baidu/ERNIE-Image"
        cache_root = tmp_path / ".cache" / "huggingface" / "hub"
        cache_repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
        blobs_dir = cache_repo_dir / "blobs"
        snapshots_dir = cache_repo_dir / "snapshots" / "abc123"
        refs_main = cache_repo_dir / "refs" / "main"
        blobs_dir.mkdir(parents=True)
        snapshots_dir.mkdir(parents=True)
        refs_main.parent.mkdir(parents=True)
        # 写入 blob 和 commit hash
        blob_path = blobs_dir / "blob1"
        blob_path.write_bytes(b"hello world")
        refs_main.write_text("abc123")

        # mock Path.home() 返回我们的 tmp_path（用 classmethod 包裹）
        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        def fake_hf_hub_download(repo_id, filename):
            # 模拟 symlink 失败
            raise FileNotFoundError("symlink failed")

        d = HFSnapshotDownloader(
            repo_id=repo_id,
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        result = await d._download_file_with_retry(fake_hf_hub_download, "config.json")
        # 应触发 fallback copy，返回 snapshot 路径
        assert "config.json" in result

    @pytest.mark.asyncio
    async def test_download_file_mirror_issue_triggers_retry(self, monkeypatch, tmp_path: Path):
        """下载文件遇到镜像问题（308/ConnectError）时切换镜像重试。"""
        call_count = {"n": 0}
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        def fake_hf_hub_download(repo_id, filename):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # 第一次抛 ConnectError 模拟镜像问题
                raise ConnectionError("ConnectError: failed")
            # 第二次成功 - 返回真实存在的文件
            local = cache_dir / filename
            local.write_bytes(b"fake")
            return str(local)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com", "https://b.com"],
        )
        result = await d._download_file_with_retry(fake_hf_hub_download, "config.json")
        assert "config.json" in result
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_download_file_non_mirror_issue_no_retry(self, monkeypatch):
        """非镜像问题（如 ValueError）不重试，直接抛错。"""

        def fake_hf_hub_download(repo_id, filename):
            raise ValueError("invalid filename")

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com", "https://b.com"],
        )
        with pytest.raises(RuntimeError, match="下载文件失败"):
            await d._download_file_with_retry(fake_hf_hub_download, "config.json")

    @pytest.mark.asyncio
    async def test_download_file_all_mirrors_fail(self, monkeypatch):
        """所有镜像都失败抛 RuntimeError。"""

        def fake_hf_hub_download(repo_id, filename):
            raise ConnectionError("ConnectError: always fails")

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com", "https://b.com"],
        )
        with pytest.raises(RuntimeError, match="下载文件失败"):
            await d._download_file_with_retry(fake_hf_hub_download, "config.json")

    @pytest.mark.asyncio
    async def test_download_file_canceled(self, monkeypatch):
        """取消时抛 RuntimeError。"""

        def fake_hf_hub_download(repo_id, filename):
            return f"/cache/{filename}"

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        cancel_event = asyncio.Event()
        cancel_event.set()
        with pytest.raises(RuntimeError, match="下载已取消"):
            await d._download_file_with_retry(
                fake_hf_hub_download, "config.json", cancel_event=cancel_event
            )

    @pytest.mark.asyncio
    async def test_download_file_returns_nonexistent_path_triggers_fallback(self, monkeypatch, tmp_path: Path):
        """hf_hub_download 返回路径但文件不存在时触发 fallback copy。"""
        # 构造假的 HF 缓存目录（用于 fallback copy 成功）
        repo_id = "r"
        cache_repo_dir = tmp_path / ".cache" / "huggingface" / "hub" / "models--r"
        blobs_dir = cache_repo_dir / "blobs"
        snapshots_dir = cache_repo_dir / "snapshots" / "abc123"
        refs_main = cache_repo_dir / "refs" / "main"
        blobs_dir.mkdir(parents=True)
        snapshots_dir.mkdir(parents=True)
        refs_main.parent.mkdir(parents=True)
        (blobs_dir / "blob1").write_bytes(b"hello")
        refs_main.write_text("abc123")

        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        def fake_hf_hub_download(repo_id, filename):
            # 返回不存在的路径
            return str(tmp_path / "nonexistent" / "config.json")

        d = HFSnapshotDownloader(
            repo_id=repo_id,
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        result = await d._download_file_with_retry(fake_hf_hub_download, "config.json")
        # 应触发 fallback copy，返回 snapshot 路径
        assert "config.json" in result


# ===== _fallback_copy_blob_to_snapshot 测试 =====


class TestFallbackCopyBlobToSnapshot:
    """_fallback_copy_blob_to_snapshot 测试。"""

    def test_blobs_dir_not_exists_raises(self, tmp_path: Path, monkeypatch):
        """blobs 目录不存在时抛 RuntimeError。"""
        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        with pytest.raises(RuntimeError, match="blobs 目录不存在"):
            d._fallback_copy_blob_to_snapshot("config.json")

    def test_blobs_empty_raises(self, tmp_path: Path, monkeypatch):
        """blobs 目录为空时抛 RuntimeError。"""
        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        repo_id = "r"
        cache_repo_dir = tmp_path / ".cache" / "huggingface" / "hub" / "models--r"
        (cache_repo_dir / "blobs").mkdir(parents=True)

        d = HFSnapshotDownloader(
            repo_id=repo_id,
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        with pytest.raises(RuntimeError, match="blobs 目录为空"):
            d._fallback_copy_blob_to_snapshot("config.json")

    def test_no_commit_hash_raises(self, tmp_path: Path, monkeypatch):
        """无法确定 commit hash 时抛 RuntimeError。"""
        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        repo_id = "r"
        cache_repo_dir = tmp_path / ".cache" / "huggingface" / "hub" / "models--r"
        blobs_dir = cache_repo_dir / "blobs"
        blobs_dir.mkdir(parents=True)
        # 写一个 blob
        (blobs_dir / "blob1").write_bytes(b"data")
        # 不创建 refs/main 或 snapshots

        d = HFSnapshotDownloader(
            repo_id=repo_id,
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        with pytest.raises(RuntimeError, match="无法确定 commit hash"):
            d._fallback_copy_blob_to_snapshot("config.json")

    def test_fallback_copy_success(self, tmp_path: Path, monkeypatch):
        """fallback copy 成功返回路径。"""
        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        repo_id = "r"
        cache_repo_dir = tmp_path / ".cache" / "huggingface" / "hub" / "models--r"
        blobs_dir = cache_repo_dir / "blobs"
        snapshots_dir = cache_repo_dir / "snapshots" / "abc123"
        refs_main = cache_repo_dir / "refs" / "main"
        blobs_dir.mkdir(parents=True)
        snapshots_dir.mkdir(parents=True)
        refs_main.parent.mkdir(parents=True)
        (blobs_dir / "blob1").write_bytes(b"hello world")
        refs_main.write_text("abc123")

        d = HFSnapshotDownloader(
            repo_id=repo_id,
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        result = d._fallback_copy_blob_to_snapshot("config.json")
        assert "config.json" in result
        # 文件应已被复制
        result_path = pathlib.Path(result)
        assert result_path.exists()
        assert result_path.read_bytes() == b"hello world"

    def test_fallback_copy_existing_same_size_skips(self, tmp_path: Path, monkeypatch):
        """目标已存在且大小相同时跳过 copy。"""
        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        repo_id = "r"
        cache_repo_dir = tmp_path / ".cache" / "huggingface" / "hub" / "models--r"
        blobs_dir = cache_repo_dir / "blobs"
        snapshots_dir = cache_repo_dir / "snapshots" / "abc123"
        refs_main = cache_repo_dir / "refs" / "main"
        blobs_dir.mkdir(parents=True)
        snapshots_dir.mkdir(parents=True)
        refs_main.parent.mkdir(parents=True)
        (blobs_dir / "blob1").write_bytes(b"hello world")
        refs_main.write_text("abc123")
        # 目标已存在且大小相同
        existing = snapshots_dir / "config.json"
        existing.write_bytes(b"hello world")

        d = HFSnapshotDownloader(
            repo_id=repo_id,
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        result = d._fallback_copy_blob_to_snapshot("config.json")
        assert "config.json" in result
        # 应直接返回，不抛错

    def test_fallback_copy_commit_from_snapshots_dir(self, tmp_path: Path, monkeypatch):
        """refs/main 不存在时从 snapshots 目录推断 commit hash。"""
        import pathlib
        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        repo_id = "r"
        cache_repo_dir = tmp_path / ".cache" / "huggingface" / "hub" / "models--r"
        blobs_dir = cache_repo_dir / "blobs"
        snapshots_dir = cache_repo_dir / "snapshots" / "def456"  # 任意 hash
        blobs_dir.mkdir(parents=True)
        snapshots_dir.mkdir(parents=True)
        # 不创建 refs/main，让代码从 snapshots 目录推断 commit
        (blobs_dir / "blob1").write_bytes(b"data")

        d = HFSnapshotDownloader(
            repo_id=repo_id,
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        result = d._fallback_copy_blob_to_snapshot("config.json")
        assert "def456" in result
        assert "config.json" in result


# ===== _setup_hf_endpoint 测试 =====


class TestSetupHfEndpoint:
    """_setup_hf_endpoint 测试。"""

    def test_sets_env_var(self, monkeypatch):
        """设置 HF_ENDPOINT 环境变量。"""
        monkeypatch.delenv("HF_ENDPOINT", raising=False)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        d._setup_hf_endpoint()
        import os
        assert os.environ.get("HF_ENDPOINT") == "https://a.com"

    def test_sets_constants(self, monkeypatch):
        """设置 huggingface_hub.constants.HF_ENDPOINT。"""
        import sys
        from unittest.mock import MagicMock

        fake_module = MagicMock()
        fake_module.constants = MagicMock()
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        d._setup_hf_endpoint()
        assert fake_module.constants.HF_ENDPOINT == "https://a.com"

    def test_constants_set_failure_logged(self, monkeypatch):
        """constants 设置失败时只 warning 不抛错。"""
        import sys
        from unittest.mock import MagicMock

        fake_module = MagicMock()
        fake_module.constants = MagicMock()
        # 触发异常
        type(fake_module.constants).HF_ENDPOINT = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

        d = HFSnapshotDownloader(
            repo_id="r",
            allow_patterns=["*"],
            mirror_endpoints=["https://a.com"],
        )
        # 不应抛异常
        d._setup_hf_endpoint()


# ===== MirrorSelector 边界场景补充测试 =====


class TestMirrorSelectorEdgeCases:
    """_EngineHFMirrorSelector 边界场景测试。"""

    def test_get_current_endpoint_cooldown_passed(self):
        """当前镜像冷却已过，恢复可用。"""
        import time as _t

        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        # 标记 a.com 不可用
        sel._marked_unavailable["https://a.com"] = True
        sel._unavailable_since["https://a.com"] = _t.time() - (
            _EngineHFMirrorSelector.UNAVAILABLE_COOLDOWN_SECONDS + 1
        )
        # 当前 idx 是 0（a.com），冷却过了应恢复
        ep = sel.get_current_endpoint()
        assert ep == "https://a.com"
        assert sel._marked_unavailable["https://a.com"] is False

    def test_get_current_endpoint_cooldown_active_advances(self):
        """当前镜像冷却未过，前进到下一个。"""
        import time as _t

        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        # 标记 a.com 不可用，冷却刚开始
        sel._marked_unavailable["https://a.com"] = True
        sel._unavailable_since["https://a.com"] = _t.time()
        # get_current_endpoint 应前进到 b.com
        ep = sel.get_current_endpoint()
        assert ep == "https://b.com"

    def test_report_speed_caps_sample_list(self):
        """report_speed 采样列表上限 20。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com"])
        for _ in range(30):
            sel.report_speed(1024 * 1024)
        cur = sel.mirrors[sel._current_idx]
        assert len(sel._speed_samples[cur]) <= 20

    def test_should_switch_when_marked_unavailable(self):
        """should_switch 在当前镜像被标记不可用时返回 True。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        sel._marked_unavailable["https://a.com"] = True
        assert sel.should_switch() is True

    def test_advance_to_next_all_unavailable_resets(self):
        """所有镜像不可用时重置所有标记。"""
        import time as _t

        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        # 全部标记为不可用，且冷却未过
        sel._marked_unavailable["https://a.com"] = True
        sel._marked_unavailable["https://b.com"] = True
        sel._unavailable_since["https://a.com"] = _t.time()
        sel._unavailable_since["https://b.com"] = _t.time()
        # 调用 _advance_to_next_available 应重置
        sel._advance_to_next_available()
        assert sel._current_idx == 0
        assert not any(sel._marked_unavailable.values())

    def test_report_failure_with_empty_msg(self):
        """report_failure 不传 msg 时使用默认。"""
        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        sel.report_failure()
        assert sel.get_current_endpoint() == "https://b.com"

    def test_report_result_success_clears_unavailable_current(self):
        """report_result 成功时清除当前镜像的不可用标记。"""
        import time as _t

        sel = _EngineHFMirrorSelector(mirrors=["https://a.com", "https://b.com"])
        # 切换到 b.com，标记 a.com 不可用
        sel.report_failure("err")
        # b.com 是当前镜像，标记 b.com 不可用
        sel._marked_unavailable["https://b.com"] = True
        sel.report_result(success=True)
        assert sel._marked_unavailable["https://b.com"] is False
