# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from dataclasses import dataclass, is_dataclass, asdict
from io import BytesIO
from typing import Any, Callable, Dict

import torch
import zmq

import msgpack
import numpy as np
import io

from gr00t.data.dataset import ModalityConfig

def default_encoder(obj):
    """Custom encoder for msgpack to handle numpy arrays and dataclasses."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if is_dataclass(obj):
        return asdict(obj)
    # 2. 添加对 ModalityConfig 的显式处理
    if isinstance(obj, ModalityConfig):
        # 将 ModalityConfig 对象转换为字典。
        # 我们假设它的属性存储在 __dict__ 中。
        return obj.__dict__
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    raise TypeError(f"can not serialize {type(obj).__name__!r} object")


class MsgSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        # --- START OF FIX ---
        # 使用 encode_custom_classes 作为 default 处理器，
        # 而不是外部的 default_encoder。
        # 这会正确地序列化 NumPy 数组，而不是将它们转换为列表。
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes, use_bin_type=True)
        # --- END OF FIX ---


    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if "__ndarray_class__" in obj:
            obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        
        # --- ADD THIS PART ---
        # 将 default_encoder 中的其他逻辑也整合进来，以处理其他自定义类型
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, ModalityConfig):
            return obj.__dict__
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist() # 或者也用 np.save 的方式处理
        # --- END OF ADDED PART ---

        return obj
    

class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555, api_token: str = None, use_msgpack: bool = False):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}
        self.api_token = api_token
        self.use_msgpack = use_msgpack

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def _validate_token(self, request: dict) -> bool:
        """
        Validate the API token in the request.
        """
        if self.api_token is None:
            return True  # No token required
        return request.get("api_token") == self.api_token

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            if self.use_msgpack :
                try:
                    message = self.socket.recv()

                    #import pdb; pdb.set_trace()
                    request = MsgSerializer.from_bytes(message)

                    # Validate token before processing request
                    if not self._validate_token(request):
                        self.socket.send(
                            MsgSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                        )
                        continue

                    endpoint = request.get("endpoint", "get_action")

                    if endpoint not in self._endpoints:
                        raise ValueError(f"Unknown endpoint: {endpoint}")

                    handler = self._endpoints[endpoint]
                    result = (
                        handler.handler(request.get("data", {}))
                        if handler.requires_input
                        else handler.handler()
                    )
                    self.socket.send(MsgSerializer.to_bytes(result))
                except Exception as e:
                    print(f"Error in server: {e}")
                    import traceback

                    print(traceback.format_exc())
                    self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))
            else:
                try:
                    message = self.socket.recv()
                    request = TorchSerializer.from_bytes(message)

                    # Validate token before processing request
                    if not self._validate_token(request):
                        self.socket.send(
                            TorchSerializer.to_bytes({"error": "Unauthorized: Invalid API token"})
                        )
                        continue

                    endpoint = request.get("endpoint", "get_action")

                    if endpoint not in self._endpoints:
                        raise ValueError(f"Unknown endpoint: {endpoint}")

                    handler = self._endpoints[endpoint]
                    result = (
                        handler.handler(request.get("data", {}))
                        if handler.requires_input
                        else handler.handler()
                    )
                    self.socket.send(TorchSerializer.to_bytes(result))
                except Exception as e:
                    print(f"Error in server: {e}")
                    import traceback

                    print(traceback.format_exc())
                    self.socket.send(TorchSerializer.to_bytes({"error": str(e)}))


class BaseInferenceClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        use_msgpack : bool = False
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.use_msgpack = use_msgpack
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        
        if self.use_msgpack:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
            response = MsgSerializer.from_bytes(message)

        else:
            self.socket.send(TorchSerializer.to_bytes(request))
            message = self.socket.recv()
            response = TorchSerializer.from_bytes(message)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", observations)
