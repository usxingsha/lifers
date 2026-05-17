"""
Lifers HAL — 机器人硬件抽象层
传感器/执行器/总线统一接口，仿真回退
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Device abstractions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeviceInfo:
    name: str
    dev_type: str
    port: str = ""
    props: Dict[str, Any] = field(default_factory=dict)
    status: str = "unknown"  # unknown, ok, error, disconnected


# ── Sensors ───────────────────────────────────────────────────────────────────

class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._backend = None

    def _lazy_init(self) -> None:
        if self._backend is not None:
            return
        try:
            import cv2
            self._backend = cv2
        except ImportError:
            pass

    def capture(self) -> Optional[np.ndarray]:
        self._lazy_init()
        if self._backend:
            cap = self._backend.VideoCapture(self.index)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
        # Simulation fallback
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)

    def info(self) -> DeviceInfo:
        return DeviceInfo(name=f"camera_{self.index}", dev_type="camera",
                          props={"width": self.width, "height": self.height},
                          status="ok" if self._backend else "simulated")


class Microphone:
    def __init__(self, sr: int = 16000, chunk_ms: int = 100) -> None:
        self.sr = sr
        self.chunk_size = sr * chunk_ms // 1000
        self._stream = None

    def read_chunk(self) -> np.ndarray:
        # Real mic would use pyaudio; fallback to silence
        return np.random.randn(self.chunk_size).astype(np.float32) * 0.001

    def info(self) -> DeviceInfo:
        return DeviceInfo(name="microphone_0", dev_type="microphone",
                          props={"sample_rate": self.sr}, status="simulated")


class Lidar:
    def __init__(self, scan_points: int = 360, max_range: float = 10.0) -> None:
        self.scan_points = scan_points
        self.max_range = max_range

    def scan(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (angles, ranges)."""
        angles = np.linspace(0, 2 * math.pi, self.scan_points, dtype=np.float32)
        ranges = np.random.uniform(0.5, self.max_range, self.scan_points).astype(np.float32)
        return angles, ranges

    def info(self) -> DeviceInfo:
        return DeviceInfo(name="lidar_0", dev_type="lidar", status="simulated")


class GPS:
    def __init__(self) -> None:
        self._lat = 0.0
        self._lon = 0.0

    def read(self) -> Tuple[float, float]:
        self._lat += np.random.randn() * 0.0001
        self._lon += np.random.randn() * 0.0001
        return self._lat, self._lon

    def info(self) -> DeviceInfo:
        return DeviceInfo(name="gps_0", dev_type="gps", status="simulated")


# ── Actuators ─────────────────────────────────────────────────────────────────

class Motor:
    def __init__(self, name: str, max_speed: float = 1.0) -> None:
        self.name = name
        self.max_speed = max_speed
        self._speed = 0.0
        self._position = 0.0

    def set_speed(self, speed: float) -> None:
        self._speed = np.clip(speed, -self.max_speed, self.max_speed)
        self._position += self._speed * 0.05  # simulate dt

    def stop(self) -> None:
        self._speed = 0.0

    @property
    def position(self) -> float:
        return self._position


class Servo:
    def __init__(self, name: str, min_angle: float = -90, max_angle: float = 90) -> None:
        self.name = name
        self.min_angle = min_angle
        self.max_angle = max_angle
        self._angle = 0.0

    def set_angle(self, degrees: float) -> None:
        self._angle = np.clip(degrees, self.min_angle, self.max_angle)

    @property
    def angle(self) -> float:
        return self._angle


class Speaker:
    def __init__(self, sr: int = 16000) -> None:
        self.sr = sr

    def play(self, audio: np.ndarray) -> None:
        # Real speaker would use pyaudio/sounddevice; stub
        pass

    def info(self) -> DeviceInfo:
        return DeviceInfo(name="speaker_0", dev_type="speaker", status="simulated")


class Display:
    def __init__(self, width: int = 800, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._buffer = np.zeros((height, width, 3), dtype=np.uint8)

    def show(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        self._buffer[:min(h, self.height), :min(w, self.width)] = image[:min(h, self.height), :min(w, self.width)]


# ═══════════════════════════════════════════════════════════════════════════════
# Bus System (I2C/SPI/Serial abstraction)
# ═══════════════════════════════════════════════════════════════════════════════

class I2CBus:
    """Simulated I2C bus."""
    def __init__(self, bus_id: int = 1) -> None:
        self.bus_id = bus_id
        self._devices: Dict[int, Dict[str, Any]] = {}

    def register(self, address: int, name: str) -> None:
        self._devices[address] = {"name": name, "registers": {}}

    def write(self, address: int, register: int, value: int) -> None:
        if address in self._devices:
            self._devices[address]["registers"][register] = value

    def read(self, address: int, register: int) -> Optional[int]:
        return self._devices.get(address, {}).get("registers", {}).get(register)


class SerialBus:
    """Simulated serial/UART."""
    def __init__(self, port: str = "COM1", baud: int = 115200) -> None:
        self.port = port
        self.baud = baud
        self._buffer = b""

    def write(self, data: bytes) -> None:
        pass

    def read(self, n: int = 1024) -> bytes:
        return b""


class GPIOBus:
    """Simulated GPIO."""
    def __init__(self) -> None:
        self._pins: Dict[int, int] = {}

    def setup(self, pin: int, mode: str = "OUT") -> None:
        self._pins[pin] = 0

    def write(self, pin: int, value: int) -> None:
        self._pins[pin] = value

    def read(self, pin: int) -> int:
        return self._pins.get(pin, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Robot HAL — unified interface
# ═══════════════════════════════════════════════════════════════════════════════

class RobotHAL:
    """Unified Hardware Abstraction Layer for Lifers robot."""

    def __init__(self) -> None:
        # Sensors
        self.camera = Camera()
        self.microphone = Microphone()
        self.lidar = Lidar()
        self.gps = GPS()
        # Actuators
        self.left_motor = Motor("left_wheel")
        self.right_motor = Motor("right_wheel")
        self.head_servo = Servo("head_pan", -90, 90)
        self.arm_servo = Servo("arm_lift", -45, 45)
        self.speaker = Speaker()
        self.display = Display()
        # Buses
        self.i2c = I2CBus()
        self.serial = SerialBus()
        self.gpio = GPIOBus()
        # State
        self._pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # x, y, theta

    def sense_all(self) -> Dict[str, Any]:
        """Read all sensors."""
        return {
            "camera": self.camera.capture().shape if self.camera.capture() is not None else None,
            "lidar": {"angles": self.lidar.scan()[0].tolist()[:5] + ["..."], "ranges_min": float(np.min(self.lidar.scan()[1])), "ranges_max": float(np.max(self.lidar.scan()[1]))},
            "gps": self.gps.read(),
            "imu": {"pose": self._pose.tolist()},
            "ts_ms": int(time.time() * 1000),
        }

    def move(self, linear: float, angular: float, dt: float = 0.1) -> np.ndarray:
        """Differential drive: move and update pose."""
        self.left_motor.set_speed(linear - angular * 0.3)
        self.right_motor.set_speed(linear + angular * 0.3)
        self._pose[0] += linear * math.cos(self._pose[2]) * dt
        self._pose[1] += linear * math.sin(self._pose[2]) * dt
        self._pose[2] += angular * dt
        return self._pose.copy()

    def stop(self) -> None:
        self.left_motor.stop()
        self.right_motor.stop()

    def device_list(self) -> List[DeviceInfo]:
        return [
            self.camera.info(), self.microphone.info(), self.lidar.info(),
            self.gps.info(), self.speaker.info(),
        ]

    def status(self) -> Dict[str, Any]:
        return {
            "pose": self._pose.tolist(),
            "motors": {"left": self.left_motor.position, "right": self.right_motor.position},
            "servos": {"head": self.head_servo.angle, "arm": self.arm_servo.angle},
            "devices": [d.status for d in self.device_list()],
        }
