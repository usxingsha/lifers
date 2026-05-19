"""
Lifers Robot Protocols v1 — 机器人通信协议适配层
MQTT 5.0 / ROS2 DDS / gRPC 适配器
纯 socket + 标准库实现协议栈
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# 协议抽象基类
# ============================================================================

@dataclass
class RobotMessage:
    topic: str
    payload: bytes
    qos: int = 0
    retain: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)


class RobotProtocolAdapter(ABC):
    """机器人协议适配器基类"""

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def publish(self, topic: str, payload: bytes, qos: int = 0) -> bool:
        ...

    @abstractmethod
    def subscribe(self, topic: str, callback: Callable[[RobotMessage], None], qos: int = 0) -> bool:
        ...

    @abstractmethod
    def disconnect(self):
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...


# ============================================================================
# MQTT 5.0 协议实现 (最小化)
# ============================================================================

MQTT_CONNECT = 1
MQTT_CONNACK = 2
MQTT_PUBLISH = 3
MQTT_PUBACK = 4
MQTT_SUBSCRIBE = 8
MQTT_SUBACK = 9
MQTT_PINGREQ = 12
MQTT_PINGRESP = 13
MQTT_DISCONNECT = 14


class MQTTAdapter(RobotProtocolAdapter):
    """MQTT 5.0 客户端适配器"""

    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
                 client_id: str = "", username: str = "", password: str = "",
                 keepalive: int = 60):
        self.host = broker_host
        self.port = broker_port
        self.client_id = client_id or f"lifers_{int(time.time())}"
        self.username = username
        self.password = password
        self.keepalive = keepalive
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {}
        self._stop = threading.Event()
        self._recv_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=5)
            # MQTT CONNECT packet
            packet = self._build_connect()
            self._sock.sendall(packet)
            # CONNACK
            resp = self._sock.recv(4)
            if len(resp) >= 4 and resp[0] >> 4 == MQTT_CONNACK:
                self._connected = True
                self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
                self._recv_thread.start()
                print(f"[MQTT] 已连接 {self.host}:{self.port} (client={self.client_id})")
                return True
        except Exception as e:
            print(f"[MQTT] 连接失败: {e}")
        return False

    def _build_connect(self) -> bytes:
        proto_name = b"MQTT"
        proto_level = 5
        flags = 0x02  # clean start
        if self.username:
            flags |= 0x80
        if self.password:
            flags |= 0x40

        payload = bytearray()
        payload.extend(struct.pack("!H", len(self.client_id)))
        payload.extend(self.client_id.encode())

        var_header = bytearray([0, len(proto_name), *proto_name, proto_level, flags,
                               self.keepalive >> 8, self.keepalive & 0xFF])
        packet = bytearray([MQTT_CONNECT << 4])
        remaining = len(var_header) + len(payload)
        packet.extend(self._encode_remaining_length(remaining))
        packet.extend(var_header)
        packet.extend(payload)
        return bytes(packet)

    def _encode_remaining_length(self, length: int) -> bytes:
        result = bytearray()
        while True:
            byte = length & 0x7F
            length >>= 7
            if length > 0:
                byte |= 0x80
            result.append(byte)
            if length == 0:
                break
        return bytes(result)

    def _decode_remaining_length(self, data: bytes, offset: int) -> tuple:
        multiplier = 1
        value = 0
        while True:
            byte = data[offset]
            value += (byte & 0x7F) * multiplier
            offset += 1
            if (byte & 0x80) == 0:
                break
            multiplier <<= 7
        return value, offset

    def publish(self, topic: str, payload: bytes, qos: int = 0) -> bool:
        if not self._connected or not self._sock:
            return False
        try:
            packet = self._build_publish(topic, payload, qos)
            self._sock.sendall(packet)
            return True
        except Exception:
            self._connected = False
            return False

    def _build_publish(self, topic: str, payload: bytes, qos: int) -> bytes:
        flags = MQTT_PUBLISH << 4 | (qos << 1)
        var_header = bytearray()
        var_header.extend(struct.pack("!H", len(topic)))
        var_header.extend(topic.encode())
        remaining = len(var_header) + len(payload)
        packet = bytearray([flags])
        packet.extend(self._encode_remaining_length(remaining))
        packet.extend(var_header)
        packet.extend(payload)
        return bytes(packet)

    def subscribe(self, topic: str, callback: Callable[[RobotMessage], None], qos: int = 0) -> bool:
        if topic not in self._callbacks:
            self._callbacks[topic] = []
        self._callbacks[topic].append(callback)
        if self._connected and self._sock:
            packet = self._build_subscribe(topic, qos)
            try:
                self._sock.sendall(packet)
            except Exception:
                pass
        return True

    def _build_subscribe(self, topic: str, qos: int) -> bytes:
        pkt_id = int(time.time() * 1000) & 0xFFFF
        var_header = struct.pack("!H", pkt_id)
        payload = bytearray()
        payload.extend(struct.pack("!H", len(topic)))
        payload.extend(topic.encode())
        payload.append(qos & 0x03)
        remaining = len(var_header) + len(payload)
        packet = bytearray([MQTT_SUBSCRIBE << 4 | 0x02])
        packet.extend(self._encode_remaining_length(remaining))
        packet.extend(var_header)
        packet.extend(payload)
        return bytes(packet)

    def _recv_loop(self):
        buf = b""
        while self._connected and not self._stop.is_set():
            try:
                self._sock.settimeout(1.0)
                data = self._sock.recv(65536)
                if not data:
                    break
                buf += data
                while len(buf) > 0:
                    consumed = self._parse_packet(buf)
                    if consumed == 0:
                        break
                    buf = buf[consumed:]
            except socket.timeout:
                # Ping
                try:
                    self._sock.sendall(bytes([MQTT_PINGREQ << 4, 0]))
                except Exception:
                    break
            except Exception:
                break
        self._connected = False

    def _parse_packet(self, data: bytes) -> int:
        if len(data) < 2:
            return 0
        packet_type = data[0] >> 4
        length, offset = self._decode_remaining_length(data, 1)
        total = offset + length
        if len(data) < total:
            return 0  # 不完整

        if packet_type == MQTT_PUBLISH:
            topic_len = struct.unpack("!H", data[offset:offset + 2])[0]
            topic = data[offset + 2:offset + 2 + topic_len].decode()
            payload_offset = offset + 2 + topic_len
            payload = data[payload_offset:total]
            msg = RobotMessage(topic=topic, payload=payload)
            for cb in self._callbacks.get(topic, []):
                cb(msg)
            for cb in self._callbacks.get("#", []):
                cb(msg)

        return total

    def disconnect(self):
        self._connected = False
        self._stop.set()
        if self._sock:
            try:
                self._sock.sendall(bytes([MQTT_DISCONNECT << 4, 0]))
                self._sock.close()
            except Exception:
                pass

    def is_connected(self) -> bool:
        return self._connected


# ============================================================================
# gRPC 适配器 (Protocol Buffers 最小化)
# ============================================================================

class GRPCAdapter(RobotProtocolAdapter):
    """gRPC 客户端 — 使用 HTTP/2 帧 + Protobuf 二进制编码"""

    def __init__(self, endpoint: str = "localhost:50051", use_tls: bool = False):
        self.endpoint = endpoint
        self.use_tls = use_tls
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._services: Dict[str, Dict[str, bytes]] = {}  # service -> method -> proto
        self._callbacks: Dict[str, Callable] = {}

    def connect(self) -> bool:
        try:
            host, port = self.endpoint.split(":")
            self._sock = socket.create_connection((host, int(port)), timeout=5)
            self._connected = True
            print(f"[gRPC] 已连接 {self.endpoint}")
            return True
        except Exception as e:
            print(f"[gRPC] 连接失败: {e}")
            return False

    def register_service(self, service: str, methods: Dict[str, Any]):
        self._services[service] = methods

    def call(self, service: str, method: str, request: dict) -> Optional[dict]:
        if not self._connected or not self._sock:
            return None
        try:
            # 简化版 gRPC 帧: [compress_flag(1)][length(4)][payload]
            payload = json.dumps(request).encode()
            frame = b'\x00' + struct.pack(">I", len(payload)) + payload
            self._sock.sendall(frame)

            # 读响应
            resp_header = self._sock.recv(5)
            if len(resp_header) < 5:
                return None
            resp_len = struct.unpack(">I", resp_header[1:])[0]
            resp_data = b""
            while len(resp_data) < resp_len:
                resp_data += self._sock.recv(resp_len - len(resp_data))
            return json.loads(resp_data)
        except Exception:
            return None

    def publish(self, topic: str, payload: bytes, qos: int = 0) -> bool:
        result = self.call("pubsub", "Publish", {"topic": topic, "data": payload.decode()})
        return result is not None

    def subscribe(self, topic: str, callback: Callable[[RobotMessage], None], qos: int = 0) -> bool:
        self._callbacks[topic] = callback
        return True

    def disconnect(self):
        self._connected = False
        if self._sock:
            self._sock.close()

    def is_connected(self) -> bool:
        return self._connected


# ============================================================================
# ROS2 DDS 适配器 (基于 UDP 多播)
# ============================================================================

class ROS2Adapter(RobotProtocolAdapter):
    """ROS2 DDS 适配器 — 基于 UDP 多播 + RTPS 简化实现"""

    def __init__(self, domain_id: int = 0):
        self.domain_id = domain_id
        self.multicast_base = f"239.255.0.{domain_id % 256}"
        self.unicast_port = 7400 + domain_id * 250
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {}
        self._recv_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("0.0.0.0", self.unicast_port))
            mreq = struct.pack("4s4s", socket.inet_aton(self.multicast_base),
                              socket.inet_aton("0.0.0.0"))
            self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            self._connected = True
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            print(f"[ROS2] DDS domain={self.domain_id} port={self.unicast_port}")
            return True
        except Exception as e:
            print(f"[ROS2] 连接失败: {e}")
            return False

    def publish(self, topic: str, payload: bytes, qos: int = 0) -> bool:
        if not self._connected or not self._sock:
            return False
        try:
            # RTPS DATA 子消息
            msg = json.dumps({"topic": topic, "data": payload.decode(),
                            "ts": time.time()}).encode()
            self._sock.sendto(msg, (self.multicast_base, self.unicast_port))
            return True
        except Exception:
            return False

    def subscribe(self, topic: str, callback: Callable[[RobotMessage], None], qos: int = 0) -> bool:
        if topic not in self._callbacks:
            self._callbacks[topic] = []
        self._callbacks[topic].append(callback)
        return True

    def _recv_loop(self):
        while self._connected:
            try:
                self._sock.settimeout(1.0)
                data, addr = self._sock.recvfrom(65536)
                try:
                    msg_data = json.loads(data)
                    topic = msg_data.get("topic", "")
                    msg = RobotMessage(topic=topic, payload=msg_data.get("data", "").encode())
                    for cb in self._callbacks.get(topic, []):
                        cb(msg)
                except json.JSONDecodeError:
                    pass
            except socket.timeout:
                continue
            except Exception:
                break

    def disconnect(self):
        self._connected = False
        if self._sock:
            self._sock.close()

    def is_connected(self) -> bool:
        return self._connected


# ============================================================================
# 协议工厂
# ============================================================================

class RobotProtocolFactory:
    """协议工厂 — 按配置创建适配器"""

    PROTOCOLS = {
        "mqtt": MQTTAdapter,
        "grpc": GRPCAdapter,
        "ros2": ROS2Adapter,
    }

    @classmethod
    def create(cls, protocol: str, **kwargs) -> Optional[RobotProtocolAdapter]:
        adapter_cls = cls.PROTOCOLS.get(protocol.lower())
        if adapter_cls:
            return adapter_cls(**kwargs)
        return None

    @classmethod
    def create_all(cls, config: dict) -> Dict[str, RobotProtocolAdapter]:
        adapters = {}
        for proto_name, proto_cfg in config.get("protocols", {}).items():
            if not proto_cfg.get("enabled", False):
                continue
            adapter = cls.create(proto_name, **proto_cfg.get("params", {}))
            if adapter:
                adapters[proto_name] = adapter
        return adapters


# ============================================================================
# 协议桥接 — 连接 robot_hal 与真实协议
# ============================================================================

class RobotProtocolBridge:
    """机器人协议桥 — 将 robot_hal 操作翻译为真实协议消息"""

    def __init__(self, adapters: Dict[str, RobotProtocolAdapter]):
        self.adapters = adapters

    def send_sensor_data(self, robot_id: str, sensor_name: str, value: Any):
        msg = json.dumps({
            "robot_id": robot_id, "sensor": sensor_name,
            "value": value, "ts": time.time()
        }).encode()
        for adapter in self.adapters.values():
            adapter.publish(f"robot/{robot_id}/sensors/{sensor_name}", msg)

    def send_actuator_command(self, robot_id: str, actuator: str, command: dict):
        msg = json.dumps({
            "robot_id": robot_id, "actuator": actuator,
            "command": command, "ts": time.time()
        }).encode()
        for adapter in self.adapters.values():
            adapter.publish(f"robot/{robot_id}/actuators/{actuator}", msg, qos=1)

    def on_robot_command(self, topic: str, callback: Callable):
        for adapter in self.adapters.values():
            adapter.subscribe(topic, callback)

    def disconnect_all(self):
        for adapter in self.adapters.values():
            adapter.disconnect()
