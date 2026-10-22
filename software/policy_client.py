"""Minimal client for the Isaac-GR00T policy server (gr00t/eval/run_gr00t_server.py),
so the laptop doesn't need the whole GR00T install. Speaks the same protocol as
gr00t.policy.server_client.PolicyClient: ZeroMQ REQ/REP, msgpack messages,
numpy arrays as {"__ndarray_class__": True, "as_npy": <np.save bytes>}.
"""
import io

import msgpack
import numpy as np
import zmq


def _encode(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            raise TypeError("object arrays can't be sent")
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode(obj):
    if not isinstance(obj, dict):
        return obj
    get = lambda k: obj.get(k, obj.get(k.encode()))
    if get("__ndarray_class__"):
        return np.load(io.BytesIO(get("as_npy")), allow_pickle=False)
    # the server replies in msgpack_numpy's layout: {nd, type, kind, shape, data}
    if get("nd") is True and get("kind") in (b"", ""):
        dtype = get("type")
        dtype = dtype.decode() if isinstance(dtype, bytes) else dtype
        return np.frombuffer(get("data"), dtype=np.dtype(dtype)).reshape(get("shape"))
    return obj


class PolicyClient:
    def __init__(self, host="localhost", port=5555, timeout_ms=15000, api_token=None):
        self.addr = f"tcp://{host}:{port}"
        self.timeout_ms, self.api_token = timeout_ms, api_token
        self.ctx = zmq.Context()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, endpoint, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        if self.api_token:
            req["api_token"] = self.api_token
        try:
            self.sock.send(msgpack.packb(req, default=_encode, use_bin_type=True))
            raw = self.sock.recv()
        except zmq.error.Again:
            self.sock.close()
            self._connect()          # a REQ socket is stuck after a timeout
            raise TimeoutError(f"no reply from GR00T server at {self.addr}")
        if raw == b"ERROR":
            raise RuntimeError("server error")
        resp = msgpack.unpackb(raw, object_hook=_decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"server error: {resp['error']}")
        return resp

    def ping(self):
        try:
            self.call("ping", requires_input=False)
            return True
        except (TimeoutError, zmq.error.ZMQError):
            return False

    def get_action(self, observation, options=None):
        action, info = self.call("get_action", {"observation": observation, "options": options})
        return action, info

    def reset(self, options=None):
        return self.call("reset", {"options": options})

    def close(self):
        self.sock.close()
        self.ctx.term()


def make_observation(frame_rgb, servo_deg, task):
    """Observation in the layout the GR00T server expects: every leaf has
    (batch=1, time=1, ...) dims; keys follow gr00t/gr00t_arm_config.py."""
    s = np.asarray(servo_deg, np.float32)
    return {
        "video": {"front": frame_rgb[None, None].astype(np.uint8)},
        "state": {"single_arm": s[None, None, :4], "gripper": s[None, None, 4:5]},
        "language": {"annotation.human.task_description": [[task]]},
    }


def action_steps(action):
    """GR00T action chunk -> list of 5-servo commands (one per time step)."""
    arm = np.asarray(action["single_arm"])[0]       # (T, 4)
    grip = np.asarray(action["gripper"])[0]         # (T, 1)
    return [list(map(float, np.concatenate([a, g]))) for a, g in zip(arm, grip)]
