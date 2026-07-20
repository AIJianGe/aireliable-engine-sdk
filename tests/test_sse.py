"""SSE 流测试。

覆盖 SSEStream / make_frame / make_final_frame / make_error_frame。
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from aireliable_engine.sse import (
    SSEStream,
    make_error_frame,
    make_final_frame,
    make_frame,
)


class TestSSEStreamBasic:
    """SSEStream 基础流式行为测试。"""

    @pytest.mark.asyncio
    async def test_put_and_events(self):
        """put + events 应正确流式输出。"""
        stream = SSEStream()
        await stream.put({"progress": 0.5, "stage": "denoising"})
        stream.close()

        frames = []
        async for frame in stream.events():
            frames.append(frame)

        assert len(frames) == 1
        assert "denoising" in frames[0]
        assert "0.5" in frames[0]
        assert frames[0].startswith("data: ")
        assert frames[0].endswith("\n\n")

    @pytest.mark.asyncio
    async def test_multiple_frames_order(self):
        """多帧输出顺序正确。"""
        stream = SSEStream()
        await stream.put({"progress": 0.1, "step": 1})
        await stream.put({"progress": 0.5, "step": 2})
        await stream.put({"progress": 1.0, "step": 3})
        stream.close()

        frames = []
        async for frame in stream.events():
            frames.append(frame)

        assert len(frames) == 3
        # 顺序应与 put 一致
        parsed = [json.loads(f[len("data: "):].strip()) for f in frames]
        assert parsed[0]["step"] == 1
        assert parsed[1]["step"] == 2
        assert parsed[2]["step"] == 3

    @pytest.mark.asyncio
    async def test_close_terminates_events(self):
        """close 后 events 迭代器正常结束。"""
        stream = SSEStream()
        await stream.put({"progress": 0.5})
        stream.close()

        frames = []
        async for frame in stream.events():
            frames.append(frame)

        # 迭代器应该结束
        assert len(frames) == 1

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """多次 close 不报错。"""
        stream = SSEStream()
        stream.close()
        stream.close()  # 第二次应静默返回
        stream.close()  # 第三次也应静默返回

    @pytest.mark.asyncio
    async def test_empty_stream_closes_cleanly(self):
        """空流 close 后 events 立即结束。"""
        stream = SSEStream()
        stream.close()

        frames = []
        async for frame in stream.events():
            frames.append(frame)

        assert frames == []


class TestSSEStreamPutAfterClose:
    """put 在 close 后行为测试。"""

    @pytest.mark.asyncio
    async def test_put_after_close_silently_dropped(self):
        """put 在 close 后静默丢弃（不抛错）。"""
        stream = SSEStream()
        stream.close()
        # 不应抛异常
        await stream.put({"progress": 0.5})
        # 数据应被丢弃
        frames = []
        async for frame in stream.events():
            frames.append(frame)
        assert frames == []

    def test_put_threadsafe_after_close_silently_dropped(self):
        """put_threadsafe 在 close 后静默丢弃。"""

        async def _run():
            stream = SSEStream()
            stream.close()
            # 不应抛异常
            stream.put_threadsafe({"progress": 0.5})

        asyncio.run(_run())


class TestSSEStreamThreadsafe:
    """put_threadsafe 跨线程写入测试。"""

    def test_put_threadsafe_from_another_thread(self):
        """put_threadsafe 跨线程写入应能被主线程消费。"""

        async def _run():
            stream = SSEStream(loop=asyncio.get_event_loop())

            def worker():
                for i in range(3):
                    stream.put_threadsafe({"progress": (i + 1) / 3, "step": i})

            t = threading.Thread(target=worker)
            t.start()
            t.join()

            stream.close()

            frames = []
            async for frame in stream.events():
                frames.append(frame)
            return frames

        frames = asyncio.run(_run())
        assert len(frames) == 3
        parsed = [json.loads(f[len("data: "):].strip()) for f in frames]
        assert parsed[0]["step"] == 0
        assert parsed[2]["step"] == 2


class TestSSESerialization:
    """SSE 帧序列化格式测试。"""

    @pytest.mark.asyncio
    async def test_frame_serialization_format(self):
        """帧序列化为 `data: {json}\\n\\n` 格式。"""
        stream = SSEStream()
        await stream.put({"progress": 0.5, "stage": "denoising"})
        stream.close()

        frames = []
        async for frame in stream.events():
            frames.append(frame)

        assert len(frames) == 1
        # 严格校验前缀和后缀
        assert frames[0].startswith("data: ")
        assert frames[0].endswith("\n\n")
        # 中间是合法 JSON
        body = frames[0][len("data: "):-2]
        data = json.loads(body)
        assert data == {"progress": 0.5, "stage": "denoising"}

    @pytest.mark.asyncio
    async def test_unicode_preserved(self):
        """中文 / Unicode 字符正确序列化（ensure_ascii=False）。"""
        stream = SSEStream()
        await stream.put({"progress": 0.5, "stage": "去噪中", "msg": "中文测试 ✓"})
        stream.close()

        frames = []
        async for frame in stream.events():
            frames.append(frame)

        assert len(frames) == 1
        # 中文应原样保留，不转为 \uXXXX
        assert "去噪中" in frames[0]
        assert "中文测试" in frames[0]
        assert "✓" in frames[0]
        # 不应出现 \\u 转义
        assert "\\u" not in frames[0]

    @pytest.mark.asyncio
    async def test_json_special_chars(self):
        """JSON 特殊字符应正确转义。"""
        stream = SSEStream()
        await stream.put({"msg": 'quote " and backslash \\ and newline \n'})
        stream.close()

        frames = []
        async for frame in stream.events():
            frames.append(frame)

        assert len(frames) == 1
        body = frames[0][len("data: "):-2]
        # 应能反序列化回原始内容
        data = json.loads(body)
        assert data["msg"] == 'quote " and backslash \\ and newline \n'


class TestFrameHelpers:
    """make_frame / make_final_frame / make_error_frame 测试。"""

    def test_make_frame_basic(self):
        """make_frame 基本字段。"""
        frame = make_frame(0.5, "denoising")
        assert frame["progress"] == 0.5
        assert frame["stage"] == "denoising"

    def test_make_frame_with_extra_fields(self):
        """make_frame 含 extra 字段。"""
        frame = make_frame(0.5, "denoising", step=25, total=50, info="halfway")
        assert frame["progress"] == 0.5
        assert frame["stage"] == "denoising"
        assert frame["step"] == 25
        assert frame["total"] == 50
        assert frame["info"] == "halfway"

    def test_make_frame_progress_is_float(self):
        """make_frame 强制 progress 为 float。"""
        frame = make_frame(1, "loading")
        assert frame["progress"] == 1.0
        assert isinstance(frame["progress"], float)

    def test_make_final_frame(self):
        """make_final_frame 应 progress=1.0。"""
        frame = make_final_frame(
            result={"image_url": "http://example.com/a.png"},
            timing={"total": 18.5},
        )
        assert frame["progress"] == 1.0
        assert frame["stage"] == "completed"
        assert frame["result"]["image_url"] == "http://example.com/a.png"
        assert frame["timing"]["total"] == 18.5

    def test_make_error_frame_default_type(self):
        """make_error_frame 默认 error_type=engine_error。"""
        frame = make_error_frame("模型未加载")
        assert frame["stage"] == "error"
        assert frame["progress"] == 0.0
        assert frame["error"]["type"] == "engine_error"
        assert frame["error"]["message"] == "模型未加载"

    def test_make_error_frame_custom_type(self):
        """make_error_frame 自定义 error_type。"""
        frame = make_error_frame("模型未加载", "model_not_loaded")
        assert frame["error"]["type"] == "model_not_loaded"
        assert frame["error"]["message"] == "模型未加载"

    def test_make_error_frame_structure(self):
        """make_error_frame 错误结构完整。"""
        frame = make_error_frame("xxx", "yyy")
        assert set(frame.keys()) == {"progress", "stage", "error"}
        assert set(frame["error"].keys()) == {"type", "message"}
