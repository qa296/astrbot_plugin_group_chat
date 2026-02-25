#!/usr/bin/env python3
"""
AstrBot Group Chat Message Flow Simulator
Connects to AstrBot via reverse WebSocket to simulate group chat messages
for testing the flow state machine plugin.

Usage:
    python message_flow_simulator.py

No external dependencies - uses Python standard library only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
import struct
import time
from dataclasses import dataclass
from typing import Any, TypedDict


# ==================== Type Definitions ====================
class UserType(TypedDict):
    user_id: int
    nickname: str
    card: str
    role: str


class ConfigType(TypedDict):
    ws_host: str
    ws_port: int
    ws_token: str
    group_id: int
    bot_id: int
    user_count: int
    messages_per_minute: int
    include_at_bot: bool
    at_bot_probability: float


# ==================== Configuration ====================
CONFIG: ConfigType = {
    # Reverse WebSocket configuration
    # Use 127.0.0.1 or localhost to connect (0.0.0.0 is for server binding only)
    # NOTE: Make sure this matches your AstrBot aiocqhttp platform config:
    # - ws_reverse_host and ws_reverse_port in AstrBot config
    # - Default AstrBot port is 6199, not 6184
    "ws_host": "127.0.0.1",
    "ws_port": 6184,  # Change this to match your AstrBot config
    "ws_token": "~KiR6.3kJaW5rrcW",  # Change this to match your AstrBot config
    # Simulation configuration
    "group_id": 123456789,  # Target group ID
    "bot_id": 12345678,  # Bot ID (must match X-Self-ID header)
    "user_count": 7,  # Number of virtual users
    "messages_per_minute": 10,  # Messages per minute
    # Message content configuration
    "include_at_bot": True,  # Occasionally @mention the bot
    "at_bot_probability": 0.1,  # Probability of @mention
}

# ==================== Virtual User Pool ====================
VIRTUAL_USERS: list[UserType] = [
    {"user_id": 100001, "nickname": "XiaoMing", "card": "", "role": "member"},
    {"user_id": 100002, "nickname": "XiaoHong", "card": "HongHong", "role": "admin"},
    {"user_id": 100003, "nickname": "ZhangSan", "card": "", "role": "member"},
    {"user_id": 100004, "nickname": "LiSi", "card": "XiaoLi", "role": "member"},
    {"user_id": 100005, "nickname": "WangWu", "card": "", "role": "member"},
    {"user_id": 100006, "nickname": "ZhaoLiu", "card": "", "role": "member"},
    {"user_id": 100007, "nickname": "AhQi", "card": "QiZai", "role": "member"},
    {"user_id": 100008, "nickname": "LaoBa", "card": "", "role": "member"},
    {"user_id": 100009, "nickname": "XiaoJiu", "card": "", "role": "member"},
    {"user_id": 100010, "nickname": "ShiShi", "card": "ShiGe", "role": "owner"},
]

# ==================== Message Templates ====================
MESSAGE_TEMPLATES: dict[str, list[str]] = {
    "daily_chat": [
        "Nice weather today",
        "Anyone up for games?",
        "Just finished eating, so sleepy",
        "Any plans for the weekend?",
        "Watching a new show, it's amazing",
        "Need to wake up early tomorrow, good night",
        "Anyone still here?",
        "Hahaha I'm dying of laughter",
        "Don't you think so?",
        "I think we should look at it this way",
        "Really?!",
        "No way, no way",
        "So true",
        "Can't stop laughing folks",
        "That move was epic",
    ],
    "tech_discussion": [
        "How do I fix this bug?",
        "Has anyone used this framework?",
        "Can't write any more code",
        "Stepped into another pitfall today",
        "Is this approach feasible?",
        "Recommend a good library please",
        "Any ideas for performance optimization?",
        "How's this architecture design?",
        "Has anyone done something similar?",
        "The documentation is terrible",
    ],
    "gaming": [
        "Anyone for ranked tonight?",
        "This game is too hard",
        "Any pros to carry me?",
        "Got bad pulls again",
        "This version is ridiculous",
        "Climbing ranks is so hard",
        "Teammates are throwing",
        "Was that play cool or what?",
        "Is the new character strong?",
        "Event rewards are too stingy",
    ],
    "life_sharing": [
        "Encountered something weird today",
        "Sharing my lunch",
        "Been working out lately, so tired",
        "Any good song recommendations?",
        "Where did you go this weekend?",
        "Not in a good mood lately",
        "Finally Friday!",
        "Working overtime till now",
        "Good luck today",
        "Bought something new, happy",
    ],
    "interaction": [
        "What do you all think?",
        "Anyone feel the same?",
        "What's your take?",
        "Share your thoughts",
        "Anyone know about this?",
        "Need some help",
        "Has anyone encountered this?",
        "Suggestions please",
        "Let's chat",
        "Tell me",
    ],
}

# Emoji pool
EMOJIS: list[str] = [
    " grin ",
    " lol ",
    " rofl ",
    " smile ",
    " heart ",
    " think ",
    " smirk ",
    " thumbsup ",
    " clap ",
    " party ",
    " fire ",
    " love ",
    " 100 ",
    " pray ",
    " sweat ",
]


class WebSocketClient:
    """Simple WebSocket client implementation using standard library."""

    OPCODE_CONTINUATION: int = 0x0
    OPCODE_TEXT: int = 0x1
    OPCODE_BINARY: int = 0x2
    OPCODE_CLOSE: int = 0x8
    OPCODE_PING: int = 0x9
    OPCODE_PONG: int = 0xA
    HEARTBEAT_INTERVAL: float = 5.0  # 5 seconds heartbeat interval

    def __init__(
        self,
        host: str,
        port: int,
        token: str = "",
        self_id: int = 0,
    ) -> None:
        self.host = host
        self.port = port
        self.token = token
        self.self_id = self_id
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.connected: bool = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._receive_task: asyncio.Task[None] | None = None

    def _generate_websocket_key(self) -> str:
        """Generate WebSocket handshake key."""
        random_bytes = bytes(random.randint(0, 255) for _ in range(16))
        return base64.b64encode(random_bytes).decode()

    async def connect(self, path: str = "/ws") -> bool:
        """Connect to WebSocket server."""
        try:
            # Establish TCP connection
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port
            )

            # Generate WebSocket key
            ws_key = self._generate_websocket_key()

            # Build handshake request with OneBot V11 reverse WebSocket headers
            headers: dict[str, str] = {
                "Host": f"{self.host}:{self.port}",
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Sec-WebSocket-Key": ws_key,
                "Sec-WebSocket-Version": "13",
                # OneBot V11 reverse WebSocket required headers
                "X-Client-Role": "Universal",  # Universal = Event + API
                "X-Self-ID": str(self.self_id),
            }

            # OneBot V11 uses access_token query parameter or header
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            # Send handshake request
            request = f"GET {path} HTTP/1.1\r\n"
            for key, value in headers.items():
                request += f"{key}: {value}\r\n"
            request += "\r\n"

            self.writer.write(request.encode())
            await self.writer.drain()

            # Read handshake response
            response = await self.reader.read(4096)
            response_str = response.decode()

            # Verify handshake success
            # HTTP 101 indicates protocol switching (WebSocket upgrade)
            if "101" in response_str.split("\r\n")[0]:
                self.connected = True
                # Start heartbeat and receive tasks
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                self._receive_task = asyncio.create_task(self._receive_loop())
                return True
            else:
                print(f"Handshake failed: {response_str[:200]}")
                return False

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def _encode_frame(self, data: str, opcode: int = 0x1) -> bytes:
        """Encode WebSocket frame with mask (required for client)."""
        payload = data.encode("utf-8")
        length = len(payload)

        # First byte: FIN(1) + RSV1-3(000) + OPCODE(4bit)
        frame = bytearray([0x80 | opcode])

        # Payload length with MASK bit set (0x80)
        if length <= 125:
            frame.append(0x80 | length)
        elif length <= 65535:
            frame.append(0x80 | 126)
            frame.extend(struct.pack(">H", length))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack(">Q", length))

        # Generate random mask (required for client-to-server frames)
        mask = bytes(random.randint(0, 255) for _ in range(4))
        frame.extend(mask)

        # Mask the payload
        masked_payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        frame.extend(masked_payload)

        return bytes(frame)

    async def send(self, data: str) -> bool:
        """Send text data."""
        if not self.connected or not self.writer:
            return False

        try:
            frame = self._encode_frame(data)
            self.writer.write(frame)
            await self.writer.drain()
            return True
        except Exception as e:
            print(f"Send failed: {e}")
            return False

    async def recv(self) -> str | None:
        """Receive data."""
        if not self.connected or not self.reader:
            return None

        try:
            # Read frame header
            header = await self.reader.read(2)
            if len(header) < 2:
                return None

            opcode = header[0] & 0x0F
            masked = (header[1] & 0x80) != 0
            length = header[1] & 0x7F

            # Read extended length
            if length == 126:
                ext_len = await self.reader.read(2)
                length = struct.unpack(">H", ext_len)[0]
            elif length == 127:
                ext_len = await self.reader.read(8)
                length = struct.unpack(">Q", ext_len)[0]

            # Read mask
            mask: bytes | None = None
            if masked:
                mask = await self.reader.read(4)

            # Read payload data
            payload = await self.reader.read(length)

            # Unmask
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            # Handle control frames
            if opcode == self.OPCODE_PING:
                await self._send_pong(payload)
                return await self.recv()
            elif opcode == self.OPCODE_CLOSE:
                await self.close()
                return None

            return payload.decode("utf-8")

        except Exception as e:
            print(f"Receive failed: {e}")
            return None

    async def _send_pong(self, payload: bytes) -> None:
        """Send PONG frame."""
        if not self.writer:
            return
        frame = bytearray([0x80 | self.OPCODE_PONG])
        frame.append(len(payload))
        frame.extend(payload)
        self.writer.write(bytes(frame))
        await self.writer.drain()

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat (PING) frames."""
        while self.connected:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                if self.connected and self.writer:
                    # Send PING frame
                    ping_frame = bytearray([0x80 | self.OPCODE_PING, 0])
                    self.writer.write(bytes(ping_frame))
                    await self.writer.drain()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat error: {e}")
                break

    async def _receive_loop(self) -> None:
        """Background task to receive and handle incoming frames."""
        while self.connected:
            try:
                data = await self.recv()
                if data is None:
                    # Connection closed
                    self.connected = False
                    break
                # We don't need to process incoming data for this simulator
                # Just keep the connection alive
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Receive loop error: {e}")
                break

    async def close(self) -> None:
        """Close connection."""
        self.connected = False

        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self.writer:
            try:
                # Send close frame
                close_frame = bytearray([0x80 | self.OPCODE_CLOSE, 0])
                self.writer.write(bytes(close_frame))
                await self.writer.drain()
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
        self.reader = None
        self.writer = None


@dataclass
class SimulatedMessage:
    """Simulated message."""

    user: UserType
    content: str
    timestamp: int
    message_id: str
    include_at: bool = False


class MessageFlowSimulator:
    """Message flow simulator."""

    def __init__(self, config: ConfigType) -> None:
        self.config = config
        self.users: list[UserType] = VIRTUAL_USERS[: config["user_count"]]
        self.ws: WebSocketClient | None = None
        self.running: bool = False
        self.message_counter: int = 0

    def generate_message_id(self) -> str:
        """Generate message ID."""
        self.message_counter += 1
        return f"sim_{int(time.time())}_{self.message_counter}"

    def generate_content(self) -> str:
        """Generate message content."""
        # Randomly select topic
        topic = random.choice(list(MESSAGE_TEMPLATES.keys()))
        template = random.choice(MESSAGE_TEMPLATES[topic])

        # Randomly add emoji
        if random.random() < 0.3:
            template += f" {random.choice(EMOJIS)}"

        return template

    def create_onebot_message(self, sim_msg: SimulatedMessage) -> dict[str, Any]:
        """Create OneBot V11 format message."""
        message_content: list[dict[str, Any]] = []

        # If need to @mention bot
        if sim_msg.include_at:
            message_content.append(
                {"type": "at", "data": {"qq": str(self.config["bot_id"])}}
            )

        # Add text content
        message_content.append({"type": "text", "data": {"text": sim_msg.content}})

        # Build OneBot V11 message format
        return {
            "post_type": "message",
            "message_type": "group",
            "time": sim_msg.timestamp,
            "self_id": self.config["bot_id"],
            "sub_type": "normal",
            "user_id": sim_msg.user["user_id"],
            "group_id": self.config["group_id"],
            "message_id": sim_msg.message_id,
            "message": message_content,
            "raw_message": sim_msg.content,
            "font": 0,
            "sender": {
                "user_id": sim_msg.user["user_id"],
                "nickname": sim_msg.user["nickname"],
                "card": sim_msg.user["card"],
                "sex": "unknown",
                "age": random.randint(18, 30),
                "area": "unknown",
                "level": str(random.randint(1, 100)),
                "role": sim_msg.user["role"],
            },
        }

    async def connect(self) -> bool:
        """Connect to AstrBot reverse WebSocket."""
        print(
            f"Connecting to ws://{self.config['ws_host']}:{self.config['ws_port']}/ws ..."
        )

        self.ws = WebSocketClient(
            host=self.config["ws_host"],
            port=self.config["ws_port"],
            token=self.config["ws_token"],
            self_id=self.config["bot_id"],
        )

        success = await self.ws.connect("/ws")
        if success:
            print("Connection successful!")
        return success

    async def send_message(self, message: dict[str, Any]) -> bool:
        """Send message."""
        if self.ws and self.ws.connected:
            return await self.ws.send(json.dumps(message))
        return False

    async def simulate_message_flow(self) -> None:
        """Simulate message flow."""
        interval: float = 60.0 / self.config["messages_per_minute"]

        print("\nStarting message flow simulation...")
        print(f"  Group ID: {self.config['group_id']}")
        print(f"  Virtual users: {len(self.users)}")
        print(f"  Message rate: {self.config['messages_per_minute']} per minute")
        print(f"  Send interval: {interval:.1f} seconds")
        print("-" * 40)

        while self.running:
            try:
                # Randomly select user
                user = random.choice(self.users)

                # Generate message content
                content = self.generate_content()

                # Determine whether to @mention bot
                include_at: bool = (
                    self.config["include_at_bot"]
                    and random.random() < self.config["at_bot_probability"]
                )

                # Create simulated message
                sim_msg = SimulatedMessage(
                    user=user,
                    content=content,
                    timestamp=int(time.time()),
                    message_id=self.generate_message_id(),
                    include_at=include_at,
                )

                # Convert to OneBot format
                onebot_msg = self.create_onebot_message(sim_msg)

                # Send message
                success = await self.send_message(onebot_msg)

                if success:
                    at_indicator = " [@Bot]" if include_at else ""
                    print(
                        f"[{sim_msg.timestamp}] {user['nickname']}: {content}{at_indicator}"
                    )

                # Wait for next message interval (with random jitter)
                jitter: float = random.uniform(-interval * 0.3, interval * 0.3)
                await asyncio.sleep(interval + jitter)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Simulation error: {e}")
                await asyncio.sleep(1)

    async def run(self) -> None:
        """Run simulator."""
        self.running = True

        # Connect to AstrBot
        if not await self.connect():
            return

        # Start message flow simulation
        try:
            await self.simulate_message_flow()
        except KeyboardInterrupt:
            print("\n\nSimulator stopped")
        finally:
            self.running = False
            if self.ws:
                await self.ws.close()
                print("Connection closed")


async def main() -> None:
    """Main function."""
    print("=" * 50)
    print("  AstrBot Group Chat Message Flow Simulator")
    print("  For testing flow state machine plugin")
    print("  No external dependencies version")
    print("=" * 50)

    simulator = MessageFlowSimulator(CONFIG)

    try:
        await simulator.run()
    except KeyboardInterrupt:
        print("\n\nProgram exited")


if __name__ == "__main__":
    asyncio.run(main())
