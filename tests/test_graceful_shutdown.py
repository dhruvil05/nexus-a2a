"""
tests/test_graceful_shutdown.py

Tests for nexus_a2a/core/graceful_shutdown.py — GracefulShutdown.

Coverage:
  - Properties: is_draining, is_drained, received_signal, drain_duration_seconds
  - install() / uninstall() signal handler registration
  - _trigger(): sets event, records signal name, idempotent on double-signal
  - wait(): suspends until signal received
  - drain(): clean drain (tasks finish), forced drain (timeout + cancellation)
  - drain() idempotency — safe to call twice
  - Async context manager (__aenter__ / __aexit__)
  - _count_active_tasks() / _cancel_active_tasks()
  - AgentServer.stop() called during drain
  - TaskManager.stop_watchdog() called during drain
  - network.bus.publish('network.shutdown') called during drain
  - Windows SIGTERM warning path
"""

from __future__ import annotations

import asyncio
import signal
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus_a2a.core.graceful_shutdown import GracefulShutdown
from nexus_a2a.models.task import TaskState

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_task(state: TaskState) -> MagicMock:
    t = MagicMock()
    t.state = state
    t.id = f"task-{state.value}"
    return t


def make_mock_network(tasks: list[MagicMock] | None = None) -> MagicMock:
    network = MagicMock()
    network.task_manager.list_all = AsyncMock(return_value=tasks or [])
    network.task_manager.cancel = AsyncMock()
    network.task_manager.stop_watchdog = AsyncMock()
    network.bus.publish = AsyncMock()
    return network


def make_mock_server(running: bool = True) -> MagicMock:
    server = MagicMock()
    server.is_running = running
    server.stop = AsyncMock()
    return server


# ── Properties ────────────────────────────────────────────────────────────────


class TestProperties:
    def test_is_draining_false_initially(self):
        sd = GracefulShutdown(network=make_mock_network())
        assert sd.is_draining is False

    def test_is_drained_false_initially(self):
        sd = GracefulShutdown(network=make_mock_network())
        assert sd.is_drained is False

    def test_received_signal_none_initially(self):
        sd = GracefulShutdown(network=make_mock_network())
        assert sd.received_signal is None

    def test_drain_duration_none_before_drain(self):
        sd = GracefulShutdown(network=make_mock_network())
        assert sd.drain_duration_seconds is None


# ── _trigger() ────────────────────────────────────────────────────────────────


class TestTrigger:
    def test_trigger_sets_shutdown_event(self):
        sd = GracefulShutdown(network=make_mock_network())
        assert not sd._shutdown_event.is_set()
        sd._trigger("SIGTERM")
        assert sd._shutdown_event.is_set()

    def test_trigger_records_signal_name(self):
        sd = GracefulShutdown(network=make_mock_network())
        sd._trigger("SIGTERM")
        assert sd.received_signal == "SIGTERM"

    def test_trigger_sigint(self):
        sd = GracefulShutdown(network=make_mock_network())
        sd._trigger("SIGINT")
        assert sd.received_signal == "SIGINT"

    def test_trigger_idempotent_first_signal_wins(self):
        sd = GracefulShutdown(network=make_mock_network())
        sd._trigger("SIGTERM")
        sd._trigger("SIGINT")  # second signal ignored
        assert sd.received_signal == "SIGTERM"


# ── install() / uninstall() ───────────────────────────────────────────────────


class TestInstallUninstall:
    @pytest.mark.asyncio
    async def test_install_registers_handlers(self):
        sd = GracefulShutdown(network=make_mock_network())
        loop = asyncio.get_running_loop()
        registered = []

        def capture(sig, *args, **kwargs):
            registered.append(sig)

        with patch.object(loop, "add_signal_handler", side_effect=capture):
            sd.install()
        # On POSIX both SIGTERM and SIGINT should be registered
        if sys.platform != "win32":
            assert signal.SIGTERM in registered
            assert signal.SIGINT in registered

    @pytest.mark.asyncio
    async def test_uninstall_removes_handlers(self):
        sd = GracefulShutdown(network=make_mock_network())
        sd.install()
        loop = asyncio.get_running_loop()
        removed = []

        def capture_remove(sig):
            removed.append(sig)

        with patch.object(loop, "remove_signal_handler", side_effect=capture_remove):
            sd.uninstall()
        # Verify at least SIGINT was removed (SIGTERM on non-Windows)
        assert signal.SIGINT in removed

    @pytest.mark.asyncio
    async def test_uninstall_without_install_is_safe(self):
        sd = GracefulShutdown(network=make_mock_network())
        sd.uninstall()  # should not raise

    @pytest.mark.asyncio
    async def test_install_outside_event_loop_raises(self):
        sd = GracefulShutdown(network=make_mock_network())
        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
            with pytest.raises(RuntimeError, match="running asyncio event loop"):
                sd.install()

    @pytest.mark.asyncio
    async def test_install_survives_not_implemented_error(self):
        """Some event loops (pytest, Windows WSL) raise NotImplementedError."""
        sd = GracefulShutdown(network=make_mock_network())
        loop = asyncio.get_running_loop()
        with patch.object(loop, "add_signal_handler", side_effect=NotImplementedError):
            sd.install()  # must not raise

    @pytest.mark.asyncio
    async def test_uninstall_survives_not_implemented_error(self):
        """Some event loops raise NotImplementedError on remove_signal_handler."""
        sd = GracefulShutdown(network=make_mock_network())
        loop = asyncio.get_running_loop()
        with patch.object(
            loop, "remove_signal_handler", side_effect=NotImplementedError
        ):
            sd.uninstall()  # must not raise

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
    async def test_windows_only_registers_sigint(self):
        sd = GracefulShutdown(network=make_mock_network())
        loop = asyncio.get_running_loop()
        registered = []

        def capture(sig, *args, **kwargs):
            registered.append(sig)

        with patch.object(loop, "add_signal_handler", side_effect=capture):
            sd.install()
        assert signal.SIGINT in registered
        assert signal.SIGTERM not in registered


# ── wait() ────────────────────────────────────────────────────────────────────


class TestWait:
    @pytest.mark.asyncio
    async def test_wait_returns_after_trigger(self):
        sd = GracefulShutdown(network=make_mock_network())

        async def fire():
            await asyncio.sleep(0.01)
            sd._trigger("SIGTERM")

        await asyncio.gather(sd.wait(), fire())
        assert sd.received_signal == "SIGTERM"

    @pytest.mark.asyncio
    async def test_wait_blocks_until_signal(self):
        sd = GracefulShutdown(network=make_mock_network())
        done = False

        async def waiter():
            nonlocal done
            await sd.wait()
            done = True

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.02)
        assert done is False
        sd._trigger("SIGINT")
        await task
        assert done is True


# ── drain(): clean path ───────────────────────────────────────────────────────


class TestDrainClean:
    @pytest.mark.asyncio
    async def test_drain_sets_is_drained(self):
        sd = GracefulShutdown(network=make_mock_network())
        await sd.drain()
        assert sd.is_drained is True

    @pytest.mark.asyncio
    async def test_drain_is_draining_false_after_completion(self):
        sd = GracefulShutdown(network=make_mock_network())
        await sd.drain()
        assert sd.is_draining is False

    @pytest.mark.asyncio
    async def test_drain_duration_set_after_drain(self):
        sd = GracefulShutdown(network=make_mock_network())
        await sd.drain()
        assert sd.drain_duration_seconds is not None
        assert sd.drain_duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_drain_stops_watchdog(self):
        network = make_mock_network()
        sd = GracefulShutdown(network=network)
        await sd.drain()
        network.task_manager.stop_watchdog.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drain_publishes_shutdown_event(self):
        network = make_mock_network()
        sd = GracefulShutdown(network=network)
        await sd.drain()
        network.bus.publish.assert_awaited()
        args = network.bus.publish.call_args
        assert args[0][0] == "network.shutdown"

    @pytest.mark.asyncio
    async def test_drain_shutdown_event_includes_signal(self):
        network = make_mock_network()
        sd = GracefulShutdown(network=network)
        sd._trigger("SIGTERM")
        await sd.drain()
        payload = network.bus.publish.call_args[0][1]
        assert payload["signal"] == "SIGTERM"

    @pytest.mark.asyncio
    async def test_drain_stops_server_if_running(self):
        network = make_mock_network()
        server = make_mock_server(running=True)
        sd = GracefulShutdown(network=network, server=server)
        await sd.drain()
        server.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_drain_skips_server_stop_if_not_running(self):
        network = make_mock_network()
        server = make_mock_server(running=False)
        sd = GracefulShutdown(network=network, server=server)
        await sd.drain()
        server.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_drain_skips_server_stop_if_no_server(self):
        network = make_mock_network()
        sd = GracefulShutdown(network=network, server=None)
        await sd.drain()  # should not raise

    @pytest.mark.asyncio
    async def test_drain_with_no_active_tasks_is_immediate(self):
        """No active tasks → drain should return without polling."""
        network = make_mock_network(tasks=[])
        sd = GracefulShutdown(network=network, poll_interval_sec=0.01)
        import time

        t0 = time.monotonic()
        await sd.drain()
        elapsed = time.monotonic() - t0
        # Should finish well under the drain timeout
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_drain_waits_for_active_tasks(self):
        """Tasks transition from WORKING to COMPLETED during drain window."""
        task = make_task(TaskState.WORKING)
        call_count = 0

        async def list_all_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return [task]  # first 2 polls: task still active
            task.state = TaskState.COMPLETED
            return [task]  # 3rd poll: task done

        network = make_mock_network()
        network.task_manager.list_all = AsyncMock(side_effect=list_all_side_effect)

        sd = GracefulShutdown(
            network=network,
            drain_timeout_sec=5.0,
            poll_interval_sec=0.01,
        )
        await sd.drain()
        assert sd.is_drained is True
        network.task_manager.cancel.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_completed_tasks_not_counted_as_active(self):
        tasks = [
            make_task(TaskState.COMPLETED),
            make_task(TaskState.FAILED),
            make_task(TaskState.CANCELLED),
        ]
        network = make_mock_network(tasks=tasks)
        sd = GracefulShutdown(network=network, poll_interval_sec=0.01)
        await sd.drain()
        network.task_manager.cancel.assert_not_awaited()


# ── drain(): forced cancellation path ────────────────────────────────────────


class TestDrainForced:
    @pytest.mark.asyncio
    async def test_drain_cancels_tasks_on_timeout(self):
        """Tasks still WORKING after drain window → force-cancelled."""
        task = make_task(TaskState.WORKING)
        network = make_mock_network(tasks=[task])
        sd = GracefulShutdown(
            network=network,
            drain_timeout_sec=0.05,  # very short timeout
            poll_interval_sec=0.01,
        )
        await sd.drain()
        network.task_manager.cancel.assert_awaited_with(task.id)

    @pytest.mark.asyncio
    async def test_drain_cancels_submitted_and_input_required(self):
        tasks = [
            make_task(TaskState.SUBMITTED),
            make_task(TaskState.INPUT_REQUIRED),
        ]
        network = make_mock_network(tasks=tasks)
        sd = GracefulShutdown(
            network=network,
            drain_timeout_sec=0.05,
            poll_interval_sec=0.01,
        )
        await sd.drain()
        assert network.task_manager.cancel.await_count == 2

    @pytest.mark.asyncio
    async def test_drain_still_completes_after_cancel_error(self):
        """Individual task cancel errors should not abort the drain."""
        task = make_task(TaskState.WORKING)
        network = make_mock_network(tasks=[task])
        network.task_manager.cancel = AsyncMock(side_effect=Exception("cancel failed"))
        sd = GracefulShutdown(
            network=network,
            drain_timeout_sec=0.05,
            poll_interval_sec=0.01,
        )
        await sd.drain()
        assert sd.is_drained is True  # drain completed despite error


# ── drain() idempotency ───────────────────────────────────────────────────────


class TestDrainIdempotency:
    @pytest.mark.asyncio
    async def test_drain_twice_only_runs_once(self):
        network = make_mock_network()
        sd = GracefulShutdown(network=network)
        await sd.drain()
        await sd.drain()
        # stop_watchdog should only be called once
        assert network.task_manager.stop_watchdog.await_count == 1

    @pytest.mark.asyncio
    async def test_second_drain_call_returns_immediately(self):
        network = make_mock_network()
        sd = GracefulShutdown(network=network)
        await sd.drain()
        first_duration = sd.drain_duration_seconds
        await sd.drain()
        # Duration should not change on second call
        assert sd.drain_duration_seconds == first_duration


# ── Resilience: errors in helpers ────────────────────────────────────────────


class TestDrainResilience:
    @pytest.mark.asyncio
    async def test_drain_survives_list_all_error(self):
        network = make_mock_network()
        network.task_manager.list_all = AsyncMock(side_effect=Exception("store down"))
        sd = GracefulShutdown(network=network, poll_interval_sec=0.01)
        await sd.drain()
        assert sd.is_drained is True

    @pytest.mark.asyncio
    async def test_drain_survives_stop_watchdog_error(self):
        network = make_mock_network()
        network.task_manager.stop_watchdog = AsyncMock(
            side_effect=Exception("watchdog error")
        )
        sd = GracefulShutdown(network=network)
        await sd.drain()
        assert sd.is_drained is True

    @pytest.mark.asyncio
    async def test_drain_survives_publish_error(self):
        network = make_mock_network()
        network.bus.publish = AsyncMock(side_effect=Exception("bus error"))
        sd = GracefulShutdown(network=network)
        await sd.drain()
        assert sd.is_drained is True

    @pytest.mark.asyncio
    async def test_drain_survives_server_stop_error(self):
        network = make_mock_network()
        server = make_mock_server(running=True)
        server.stop = AsyncMock(side_effect=Exception("server error"))
        sd = GracefulShutdown(network=network, server=server)
        await sd.drain()
        assert sd.is_drained is True


# ── Async context manager ─────────────────────────────────────────────────────


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_aenter_installs_handlers(self):
        sd = GracefulShutdown(network=make_mock_network())
        with (
            patch.object(sd, "install") as mock_install,
            patch.object(sd, "uninstall"),
            patch.object(sd, "drain", new_callable=AsyncMock),
        ):
            async with sd:
                mock_install.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_calls_drain(self):
        sd = GracefulShutdown(network=make_mock_network())
        with (
            patch.object(sd, "install"),
            patch.object(sd, "uninstall"),
            patch.object(sd, "drain", new_callable=AsyncMock) as mock_drain,
        ):
            async with sd:
                pass
            mock_drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aexit_uninstalls_handlers(self):
        sd = GracefulShutdown(network=make_mock_network())
        with (
            patch.object(sd, "install"),
            patch.object(sd, "uninstall") as mock_uninstall,
            patch.object(sd, "drain", new_callable=AsyncMock),
        ):
            async with sd:
                pass
            mock_uninstall.assert_called_once()

    @pytest.mark.asyncio
    async def test_aexit_does_not_double_drain_if_already_drained(self):
        network = make_mock_network()
        sd = GracefulShutdown(network=network)
        await sd.drain()
        first_count = network.task_manager.stop_watchdog.await_count

        with patch.object(sd, "install"), patch.object(sd, "uninstall"):
            async with sd:
                pass

        # stop_watchdog count must not increase — drain already ran
        assert network.task_manager.stop_watchdog.await_count == first_count

    @pytest.mark.asyncio
    async def test_full_context_manager_workflow(self):
        """End-to-end: signal fires inside the context, drain happens on exit."""
        network = make_mock_network()
        server = make_mock_server(running=True)

        with (
            patch.object(
                asyncio.get_running_loop(), "add_signal_handler", return_value=None
            ),
            patch.object(
                asyncio.get_running_loop(), "remove_signal_handler", return_value=None
            ),
        ):
            async with GracefulShutdown(network=network, server=server) as sd:
                sd._trigger("SIGTERM")  # simulate signal

        assert sd.is_drained is True
        assert sd.received_signal == "SIGTERM"
        server.stop.assert_awaited_once()
        network.task_manager.stop_watchdog.assert_awaited_once()


# ── _count_active_tasks / _cancel_active_tasks ────────────────────────────────


class TestHelpers:
    @pytest.mark.asyncio
    async def test_count_active_tasks_zero_with_no_tasks(self):
        sd = GracefulShutdown(network=make_mock_network(tasks=[]))
        assert await sd._count_active_tasks() == 0

    @pytest.mark.asyncio
    async def test_count_active_tasks_counts_working(self):
        tasks = [make_task(TaskState.WORKING), make_task(TaskState.SUBMITTED)]
        sd = GracefulShutdown(network=make_mock_network(tasks=tasks))
        assert await sd._count_active_tasks() == 2

    @pytest.mark.asyncio
    async def test_count_active_tasks_excludes_terminal(self):
        tasks = [
            make_task(TaskState.COMPLETED),
            make_task(TaskState.FAILED),
            make_task(TaskState.CANCELLED),
            make_task(TaskState.WORKING),
        ]
        sd = GracefulShutdown(network=make_mock_network(tasks=tasks))
        assert await sd._count_active_tasks() == 1

    @pytest.mark.asyncio
    async def test_cancel_active_tasks_cancels_all_active(self):
        tasks = [make_task(TaskState.WORKING), make_task(TaskState.SUBMITTED)]
        network = make_mock_network(tasks=tasks)
        sd = GracefulShutdown(network=network)
        n = await sd._cancel_active_tasks()
        assert n == 2
        assert network.task_manager.cancel.await_count == 2

    @pytest.mark.asyncio
    async def test_cancel_active_tasks_skips_terminal(self):
        tasks = [make_task(TaskState.COMPLETED), make_task(TaskState.WORKING)]
        network = make_mock_network(tasks=tasks)
        sd = GracefulShutdown(network=network)
        n = await sd._cancel_active_tasks()
        assert n == 1

    @pytest.mark.asyncio
    async def test_count_returns_zero_on_store_error(self):
        network = make_mock_network()
        network.task_manager.list_all = AsyncMock(side_effect=Exception("boom"))
        sd = GracefulShutdown(network=network)
        assert await sd._count_active_tasks() == 0
