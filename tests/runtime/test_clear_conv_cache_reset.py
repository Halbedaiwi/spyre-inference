# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``SPYRE_CLEAR_CONV_CACHE``: idle timer, engine-thread kernel clear, server cache reset."""

import threading
import types
import urllib.error
from unittest import mock

import pytest
import torch.nn as nn
from vllm.config import CompilationMode
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference import envs
from spyre_inference.custom_ops import lazy_compile
from spyre_inference.custom_ops.lazy_compile import CompileOutermost
from spyre_inference.v1.worker import spyre_model_runner as smr


@pytest.fixture
def runner():
    r = smr.TorchSpyreModelRunner.__new__(smr.TorchSpyreModelRunner)
    r._kernel_reset_lock = threading.Lock()
    r._kernel_reset_timer = None
    r._kernel_reset_due = threading.Event()
    return r


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("SPYRE_CLEAR_CONV_CACHE", "1")
    envs.clear_env_cache()


@pytest.fixture
def timers():
    """Replace ``threading.Timer`` so nothing waits; yields the timers created."""
    created: list = []

    def make(interval, fn):
        t = mock.Mock(interval=interval, fn=fn)
        created.append(t)
        return t

    with mock.patch.object(smr.threading, "Timer", side_effect=make):
        yield created


def _sched(finished=(), tokens=0):
    return types.SimpleNamespace(finished_req_ids=set(finished), total_num_scheduled_tokens=tokens)


def _resp(status: int):
    cm = mock.MagicMock()
    cm.__enter__.return_value.status = status
    return cm


def _reset() -> bool:
    return smr.TorchSpyreModelRunner._reset_server_caches(mock.Mock())


def test_posts_all_three_endpoints_in_order():
    with mock.patch.object(smr.urllib.request, "urlopen", return_value=_resp(200)) as urlopen:
        assert _reset() is True
    urls = [call.args[0].full_url for call in urlopen.call_args_list]
    assert urls == [f"{smr._SERVER_BASE_URL}/{e}" for e in smr._SERVER_CACHE_RESET_ENDPOINTS]
    assert [call.args[0].get_method() for call in urlopen.call_args_list] == ["POST"] * 3
    assert all(
        call.kwargs["timeout"] == smr._SERVER_RESET_TIMEOUT_S for call in urlopen.call_args_list
    )


def test_stops_at_first_http_error():
    err = urllib.error.HTTPError("u", 404, "Not Found", {}, None)  # ty: ignore[invalid-argument-type]
    with mock.patch.object(smr.urllib.request, "urlopen", side_effect=err) as urlopen:
        assert _reset() is False
    assert urlopen.call_count == 1


def test_connection_failure_returns_false():
    with mock.patch.object(
        smr.urllib.request, "urlopen", side_effect=urllib.error.URLError("refused")
    ):
        assert _reset() is False


def test_timeout_returns_false():
    with mock.patch.object(smr.urllib.request, "urlopen", side_effect=TimeoutError()):
        assert _reset() is False


def test_finished_request_with_no_work_arms_timer(runner, timers):
    runner._track_idle_for_kernel_reset(_sched(finished={"a"}))
    assert [t.interval for t in timers] == [5.0]
    timers[0].start.assert_called_once()
    assert timers[0].daemon is True
    assert runner._kernel_reset_timer is timers[0]


def test_rearming_cancels_previous_timer(runner, timers):
    runner._track_idle_for_kernel_reset(_sched(finished={"a"}))
    runner._track_idle_for_kernel_reset(_sched(finished={"b"}))
    timers[0].cancel.assert_called_once()
    assert runner._kernel_reset_timer is timers[1]


def test_scheduled_tokens_cancel_pending_timer(runner, timers):
    runner._track_idle_for_kernel_reset(_sched(finished={"a"}))
    runner._track_idle_for_kernel_reset(_sched(tokens=8))
    timers[0].cancel.assert_called_once()
    assert runner._kernel_reset_timer is None


def test_idle_step_without_finished_request_does_nothing(runner, timers):
    runner._track_idle_for_kernel_reset(_sched())
    assert timers == []


def test_timer_fire_marks_due_without_clearing_kernels(runner, timers):
    runner._schedule_kernel_reset()
    with (
        mock.patch.object(smr, "get_world_group", return_value=types.SimpleNamespace(rank=1)),
        mock.patch.object(smr.torch._dynamo, "reset") as dynamo_reset,
        mock.patch.object(smr, "reset_compiled_kernels") as clear,
    ):
        timers[0].fn()
    assert runner._kernel_reset_due.is_set()
    assert runner._kernel_reset_timer is None
    dynamo_reset.assert_not_called()
    clear.assert_not_called()


def test_stale_timer_does_nothing(runner, timers):
    runner._schedule_kernel_reset()
    runner._schedule_kernel_reset()
    with mock.patch.object(smr, "get_world_group") as world:
        timers[0].fn()
    assert not runner._kernel_reset_due.is_set()
    assert runner._kernel_reset_timer is timers[1]
    world.assert_not_called()


def test_cancelled_timer_does_nothing(runner, timers):
    runner._schedule_kernel_reset()
    runner._cancel_kernel_reset_timer()
    timers[0].fn()
    assert not runner._kernel_reset_due.is_set()


@pytest.mark.parametrize("rank, resets", [(0, True), (1, False)])
def test_only_rank_zero_resets_server_caches(runner, timers, rank, resets):
    runner._reset_server_caches = mock.Mock(return_value=True)
    runner._schedule_kernel_reset()
    with mock.patch.object(smr, "get_world_group", return_value=types.SimpleNamespace(rank=rank)):
        timers[0].fn()
    assert runner._reset_server_caches.called is resets
    assert runner._kernel_reset_due.is_set()


@pytest.mark.parametrize("ok, level", [(True, "info"), (False, "error")])
def test_reset_outcome_is_logged(runner, timers, ok, level):
    runner._reset_server_caches = mock.Mock(return_value=ok)
    runner._schedule_kernel_reset()
    with (
        mock.patch.object(smr, "get_world_group", return_value=types.SimpleNamespace(rank=0)),
        mock.patch.object(smr, "logger") as log,
    ):
        timers[0].fn()
    assert log.error.called is (level == "error")
    assert any("ready for next run" in c.args[0] for c in log.info.call_args_list) is ok


def test_execute_model_clears_kernels_once_when_due(runner, flag_on):
    runner._kernel_reset_due.set()
    with (
        mock.patch.object(smr.torch._dynamo, "reset") as dynamo_reset,
        mock.patch.object(smr, "reset_compiled_kernels") as clear,
        mock.patch.object(GPUModelRunner, "execute_model", return_value="out") as base,
    ):
        assert runner.execute_model(_sched(tokens=4), "x", k=1) == "out"
        runner.execute_model(_sched(tokens=4))
    dynamo_reset.assert_called_once()
    clear.assert_called_once()
    assert not runner._kernel_reset_due.is_set()
    assert base.call_count == 2
    assert base.call_args_list[0].args[1:] == ("x",)
    assert base.call_args_list[0].kwargs == {"k": 1}


def test_execute_model_does_not_clear_when_not_due(runner, flag_on):
    with (
        mock.patch.object(smr.torch._dynamo, "reset") as dynamo_reset,
        mock.patch.object(smr, "reset_compiled_kernels") as clear,
        mock.patch.object(GPUModelRunner, "execute_model", return_value="out"),
    ):
        runner.execute_model(_sched(tokens=4))
    dynamo_reset.assert_not_called()
    clear.assert_not_called()


def test_flag_off_neither_tracks_nor_clears(runner, timers):
    runner._kernel_reset_due.set()
    with (
        mock.patch.object(smr.torch._dynamo, "reset") as dynamo_reset,
        mock.patch.object(smr, "reset_compiled_kernels") as clear,
        mock.patch.object(GPUModelRunner, "execute_model", return_value="out") as base,
    ):
        assert runner.execute_model(_sched(finished={"a"})) == "out"
    assert timers == []
    dynamo_reset.assert_not_called()
    clear.assert_not_called()
    base.assert_called_once()
    assert runner._kernel_reset_due.is_set()


def test_reset_compiled_kernels_clears_every_live_instance(monkeypatch):
    monkeypatch.setattr(
        lazy_compile,
        "get_cached_compilation_config",
        lambda: types.SimpleNamespace(mode=CompilationMode.NONE),
    )

    class _Layer(CompileOutermost, nn.Module):
        pass

    layers = [_Layer(), _Layer()]
    for layer in layers:
        layer.spyre_compiled_kernel = lambda *a: None
    lazy_compile.reset_compiled_kernels()
    assert all(layer.spyre_compiled_kernel is None for layer in layers)
