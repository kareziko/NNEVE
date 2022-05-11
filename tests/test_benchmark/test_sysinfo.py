from nneve.benchmark import SysInfo, get_sys_info
from nneve.benchmark.sysinfo import (
    CpuInfo,
    GpuInfo,
    PlatformInfo,
    PythonInfo,
    SysMemInfo,
    TfInfo,
)


def test_get_sys_info():
    info = get_sys_info()
    assert isinstance(info, SysInfo)
    assert isinstance(info.platform, PlatformInfo)
    assert isinstance(info.python, PythonInfo)
    assert isinstance(info.cpu, CpuInfo)
    assert isinstance(info.memory, SysMemInfo)
    assert isinstance(info.tensorflow, TfInfo)
    assert isinstance(info.gpu, list)
    if len(info.gpu) != 0:
        assert isinstance(info.gpu[0], GpuInfo)
