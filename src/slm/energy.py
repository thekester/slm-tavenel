"""GPU energy measurement for training runs, through NVIDIA's management library (NVML).

On GPUs that expose it (Volta and later), NVML's cumulative energy counter
(``nvmlDeviceGetTotalEnergyConsumption``, millijoules since the driver loaded) is read at the start and
at each logging step; otherwise board power (``nvmlDeviceGetPowerUsage``) is sampled once per second in a
background thread and integrated. Limits, also written into the run records: only the GPU board is
measured (CPU, RAM, disks and power-supply losses are not), and the counter includes every process using
the same GPU during the run.
"""
from __future__ import annotations

import threading
import time


class GpuEnergyMeter:
    def __init__(self, device_index: int = 0, sample_seconds: float = 1.0):
        self.available, self.method, self._nvml = False, None, None
        self._lock = threading.Lock()
        self._sampled_j = 0.0
        self._stop = threading.Event()
        try:
            import pynvml  # provided by the nvidia-ml-py package

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:  # noqa: BLE001 - no NVML, no NVIDIA GPU or no driver: measurement is simply absent
            return
        try:
            self._start_mj = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            self.method = "nvml_total_energy_counter"
        except Exception:  # noqa: BLE001
            self.method = "nvml_power_sampling_1hz"
            self._thread = threading.Thread(target=self._sample, args=(sample_seconds,), daemon=True)
            self._thread.start()
        self.available = True

    def _sample(self, period: float) -> None:
        last = time.perf_counter()
        while not self._stop.wait(period):
            now = time.perf_counter()
            watts = self.power_w()
            with self._lock:
                self._sampled_j += (watts or 0.0) * (now - last)
            last = now

    def power_w(self) -> float | None:
        if self._nvml is None:
            return None
        try:
            return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:  # noqa: BLE001
            return None

    def energy_wh(self) -> float | None:
        """GPU energy since this meter started, in watt-hours."""
        if not self.available:
            return None
        if self.method == "nvml_total_energy_counter":
            return (self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle) - self._start_mj) / 3.6e6
        with self._lock:
            return self._sampled_j / 3600.0

    def close(self) -> None:
        self._stop.set()
