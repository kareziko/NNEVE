from nneve.benchmark import SysInfo, get_sys_info


def test_get_sys_info():
    info = get_sys_info()
    assert isinstance(info, SysInfo)
