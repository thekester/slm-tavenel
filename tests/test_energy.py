import sys
import time
import types

from slm.energy import GpuEnergyMeter


def fake_nvml(counter: bool, watts: float = 100.0):
    state = {"mj": 1_000_000}
    module = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: object(),
        nvmlDeviceGetPowerUsage=lambda h: int(watts * 1000),
    )
    if counter:
        module.nvmlDeviceGetTotalEnergyConsumption = lambda h: state["mj"]
    else:
        def unsupported(h):
            raise RuntimeError("not supported")
        module.nvmlDeviceGetTotalEnergyConsumption = unsupported
    return module, state


def test_counter_mode_reports_energy_since_start(monkeypatch):
    module, state = fake_nvml(counter=True)
    monkeypatch.setitem(sys.modules, "pynvml", module)
    meter = GpuEnergyMeter()
    assert meter.available and meter.method == "nvml_total_energy_counter"
    state["mj"] += 3_600_000  # 3,600 J = 1 Wh
    assert meter.energy_wh() == 1.0
    assert meter.power_w() == 100.0


def test_sampling_mode_integrates_power(monkeypatch):
    module, _ = fake_nvml(counter=False, watts=3600.0)
    monkeypatch.setitem(sys.modules, "pynvml", module)
    meter = GpuEnergyMeter(sample_seconds=0.05)
    assert meter.method == "nvml_power_sampling_1hz"
    time.sleep(0.3)
    meter.close()
    # 3,600 W for ~0.25-0.3 s is ~0.25-0.3 Wh.
    assert 0.1 < meter.energy_wh() < 0.5


def test_absent_nvml_means_no_measurement(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import fails
    meter = GpuEnergyMeter()
    assert not meter.available and meter.energy_wh() is None and meter.power_w() is None
