"""Unit tests for system environment detection."""

from zmodel.core.system_info import get_system_environment


def test_get_system_environment():
    env = get_system_environment()
    assert env.os_name != ""
    assert env.architecture != ""
    assert env.python_version != ""
    assert env.cpu_count_logical is not None and env.cpu_count_logical > 0
    assert env.memory_total_gb > 0
    assert env.onnx_version is not None
    assert env.onnxruntime_version is not None
    assert isinstance(env.host_providers, list)
    assert len(env.host_providers) > 0
    assert "CPUExecutionProvider" in env.host_providers
    # Verification checks
    assert env.host_runtime_verified is True
    assert env.target_hardware_verified is False
    assert "Not verified" in env.target_architecture_status
