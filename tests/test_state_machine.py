"""状态机测试。

覆盖 RuntimeState / StateMachine 的所有公开行为。
"""

from __future__ import annotations

import time

import pytest

from aireliable_engine.state_machine import RuntimeState, StateMachine


class TestStateMachineBasic:
    """StateMachine 基础行为测试。"""

    def test_initial_state_is_idle(self):
        """初始状态应为 IDLE。"""
        sm = StateMachine()
        assert sm.state == RuntimeState.IDLE

    def test_initial_state_can_be_overridden(self):
        """初始状态可由参数指定。"""
        sm = StateMachine(initial=RuntimeState.READY)
        assert sm.state == RuntimeState.READY

    def test_valid_transition_chain(self):
        """合法状态转换链：IDLE → LOADING → READY → INFERRING → READY。"""
        sm = StateMachine()
        sm.transition(RuntimeState.LOADING)
        assert sm.state == RuntimeState.LOADING
        sm.transition(RuntimeState.READY)
        assert sm.state == RuntimeState.READY
        sm.transition(RuntimeState.INFERRING)
        assert sm.state == RuntimeState.INFERRING
        sm.transition(RuntimeState.READY)
        assert sm.state == RuntimeState.READY

    def test_invalid_transition_idle_to_inferring(self):
        """非法转换：IDLE 直接到 INFERRING 应抛 ValueError。"""
        sm = StateMachine(initial=RuntimeState.IDLE)
        with pytest.raises(ValueError, match="非法状态转换"):
            sm.transition(RuntimeState.INFERRING)

    def test_invalid_transition_ready_to_loading(self):
        """非法转换：READY → LOADING 不在允许集合里应抛 ValueError。"""
        sm = StateMachine(initial=RuntimeState.READY)
        with pytest.raises(ValueError, match="非法状态转换"):
            sm.transition(RuntimeState.LOADING)

    def test_same_state_transition_no_error(self):
        """转换到相同状态不应抛错。"""
        sm = StateMachine(initial=RuntimeState.READY)
        # 同状态转换：当前代码允许 new_state == old 时直接通过
        sm.transition(RuntimeState.READY)
        assert sm.state == RuntimeState.READY


class TestStateMachineReset:
    """reset 行为测试。"""

    def test_reset_from_error_to_idle(self):
        """ERROR 状态可 reset 回 IDLE。"""
        sm = StateMachine(initial=RuntimeState.ERROR)
        sm.reset()
        assert sm.state == RuntimeState.IDLE

    def test_reset_from_idle_raises(self):
        """非 ERROR 状态不能 reset。"""
        sm = StateMachine(initial=RuntimeState.IDLE)
        with pytest.raises(ValueError, match="reset 仅在 ERROR 状态可用"):
            sm.reset()

    def test_reset_from_ready_raises(self):
        """READY 状态不能 reset。"""
        sm = StateMachine(initial=RuntimeState.READY)
        with pytest.raises(ValueError, match="reset 仅在 ERROR 状态可用"):
            sm.reset()

    def test_reset_from_loading_raises(self):
        """LOADING 状态不能 reset。"""
        sm = StateMachine(initial=RuntimeState.LOADING)
        with pytest.raises(ValueError, match="reset 仅在 ERROR 状态可用"):
            sm.reset()


class TestStateMachineQueries:
    """can_infer / is_busy 查询方法测试。"""

    def test_can_infer_only_when_ready(self):
        """can_infer 仅 READY 时为 True。"""
        sm = StateMachine()
        assert not sm.can_infer()
        sm.transition(RuntimeState.LOADING)
        assert not sm.can_infer()
        sm.transition(RuntimeState.READY)
        assert sm.can_infer()
        sm.transition(RuntimeState.INFERRING)
        assert not sm.can_infer()

    def test_is_busy_when_loading_or_inferring(self):
        """is_busy 在 LOADING / INFERRING 时为 True。"""
        sm = StateMachine()
        assert not sm.is_busy()
        sm.transition(RuntimeState.LOADING)
        assert sm.is_busy()
        sm.transition(RuntimeState.READY)
        assert not sm.is_busy()
        sm.transition(RuntimeState.INFERRING)
        assert sm.is_busy()

    def test_is_busy_false_when_error(self):
        """ERROR 状态 is_busy 为 False。"""
        sm = StateMachine(initial=RuntimeState.ERROR)
        assert not sm.is_busy()

    def test_can_infer_false_when_error(self):
        """ERROR 状态 can_infer 为 False。"""
        sm = StateMachine(initial=RuntimeState.ERROR)
        assert not sm.can_infer()


class TestStateMachineTimestamps:
    """touch / last_used_at / uptime 测试。"""

    def test_touch_updates_last_used(self):
        """touch 应更新 last_used_at。"""
        sm = StateMachine()
        old = sm.last_used_at
        time.sleep(0.01)
        sm.touch()
        assert sm.last_used_at > old

    def test_uptime_increases_over_time(self):
        """uptime 应随时间增加。"""
        sm = StateMachine()
        u1 = sm.uptime
        time.sleep(0.01)
        u2 = sm.uptime
        assert u2 > u1

    def test_transition_to_ready_updates_last_used(self):
        """转换到 READY 时应更新 last_used_at。"""
        sm = StateMachine(initial=RuntimeState.IDLE)
        old = sm.last_used_at
        time.sleep(0.01)
        sm.transition(RuntimeState.LOADING)
        sm.transition(RuntimeState.READY)
        assert sm.last_used_at > old

    def test_transition_to_inferring_updates_last_used(self):
        """转换到 INFERRING 时应更新 last_used_at。"""
        sm = StateMachine(initial=RuntimeState.READY)
        old = sm.last_used_at
        time.sleep(0.01)
        sm.transition(RuntimeState.INFERRING)
        assert sm.last_used_at > old


class TestStateMachineListeners:
    """监听器测试。"""

    def test_listener_called_on_transition(self):
        """add_listener 后状态变化触发监听器。"""
        calls: list[tuple] = []
        sm = StateMachine()
        sm.add_listener(lambda old, new: calls.append((old, new)))
        sm.transition(RuntimeState.LOADING)
        assert len(calls) == 1
        assert calls[0] == (RuntimeState.IDLE, RuntimeState.LOADING)

    def test_multiple_listeners_all_called(self):
        """多个监听器都被调用。"""
        calls_a: list[tuple] = []
        calls_b: list[tuple] = []
        sm = StateMachine()
        sm.add_listener(lambda old, new: calls_a.append((old, new)))
        sm.add_listener(lambda old, new: calls_b.append((old, new)))
        sm.transition(RuntimeState.LOADING)
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    def test_listener_exception_does_not_block_transition(self):
        """监听器异常不影响状态转换。"""
        def bad_listener(old, new):
            raise RuntimeError("监听器炸了")

        sm = StateMachine()
        sm.add_listener(bad_listener)
        # 不应抛异常
        sm.transition(RuntimeState.LOADING)
        # 状态转换仍生效
        assert sm.state == RuntimeState.LOADING

    def test_listener_called_multiple_times(self):
        """多次转换应多次触发监听器。"""
        calls: list[tuple] = []
        sm = StateMachine()
        sm.add_listener(lambda old, new: calls.append((old, new)))
        sm.transition(RuntimeState.LOADING)
        sm.transition(RuntimeState.READY)
        sm.transition(RuntimeState.INFERRING)
        assert len(calls) == 3
        assert calls[0] == (RuntimeState.IDLE, RuntimeState.LOADING)
        assert calls[1] == (RuntimeState.LOADING, RuntimeState.READY)
        assert calls[2] == (RuntimeState.READY, RuntimeState.INFERRING)
