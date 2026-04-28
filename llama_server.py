"""
Lifecycle manager for the standalone llama.cpp HTTP server (`llama-server.exe`).

Why this exists
---------------
On Windows the prebuilt CUDA wheels of ``llama-cpp-python`` from PyPI / abetlen
are stuck at version 0.3.4, whose bundled llama.cpp does not understand the
``qwen3`` GGUF architecture (added upstream in 2025-04). The maintained
standalone binaries from the official ``llama.cpp`` releases DO support qwen3
and ship CUDA 13 runtimes that match recent NVIDIA drivers.

So: instead of fighting Python bindings, we run the server binary as a
subprocess and talk to its OpenAI-compatible HTTP API. The chain code uses
``langchain_openai.ChatOpenAI`` pointed at ``http://127.0.0.1:<port>/v1``.
"""
from __future__ import annotations

import atexit
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("llama_server")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_SERVER_EXE = Path(
    os.environ.get(
        "LLAMA_SERVER_EXE",
        r"C:\Users\25289\Downloads\files\runtime\llama-server.exe",
    )
)
DEFAULT_MODEL_PATH = Path(
    os.environ.get("LLAMA_MODEL_PATH", "./models/model.gguf")
)


@dataclass
class ServerConfig:
    server_exe: Path = DEFAULT_SERVER_EXE
    model_path: Path = DEFAULT_MODEL_PATH
    host: str = "127.0.0.1"
    port: int = 8765
    # n_ctx is the TOTAL context across all slots; per-slot ctx = n_ctx / parallel.
    # 16384 / 4 = 4096 per slot, fine for typical queries and for ragas judges.
    n_ctx: int = 16384
    n_gpu_layers: int = -1   # -1 = all layers on GPU; 0 = CPU only
    # parallel >= 4 lets ragas's answer_relevancy fan out N completions in
    # a single request (it asks for paraphrased questions for cosine sim);
    # n=1 fails with 'n_cmpl > slots'. 4 is enough for the default ragas
    # configuration on a 4GB VRAM card.
    parallel: int = 4
    extra_args: list[str] = field(default_factory=list)
    startup_timeout_s: float = 90.0
    log_path: Path = Path("./llama_server.log")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"


# ---------------------------------------------------------------------------
# Lifecycle manager
# ---------------------------------------------------------------------------
class LlamaServer:
    """Spawn / health-check / shutdown a llama-server.exe process."""

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        self.config = config or ServerConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None

    # ------------------------------------------------------------------ probe
    def is_running(self) -> bool:
        try:
            r = requests.get(f"{self.config.base_url}/health", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # -------------------------------------------------------------- start API
    def start(self) -> None:
        """Start the server if not already up. Idempotent."""
        if self.is_running():
            logger.info("Reusing already-running llama-server at %s",
                        self.config.base_url)
            return

        cfg = self.config
        if not cfg.server_exe.exists():
            raise FileNotFoundError(
                f"llama-server.exe not found at {cfg.server_exe}. "
                "Set LLAMA_SERVER_EXE env var to override."
            )
        if not cfg.model_path.exists():
            raise FileNotFoundError(f"GGUF model not found at {cfg.model_path}")

        cmd = [
            str(cfg.server_exe),
            "--model", str(cfg.model_path.resolve()),
            "--host", cfg.host,
            "--port", str(cfg.port),
            "--ctx-size", str(cfg.n_ctx),
            "--n-gpu-layers", str(cfg.n_gpu_layers),
            "--parallel", str(cfg.parallel),
            *cfg.extra_args,
        ]
        logger.info("Launching: %s", " ".join(cmd))

        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(cfg.log_path, "w", encoding="utf-8", errors="replace")

        # DETACHED_PROCESS on Windows keeps the child from sharing our console
        # and dying when we hit Ctrl-C in the parent.
        creationflags = 0x00000008 if os.name == "nt" else 0
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=self._log_fh,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except Exception as exc:
            logger.exception("Failed to spawn llama-server.")
            raise RuntimeError(f"Failed to spawn llama-server: {exc}") from exc

        atexit.register(self.stop)

        # Wait for /health to come up.
        deadline = time.time() + cfg.startup_timeout_s
        while time.time() < deadline:
            if self.is_running():
                elapsed = cfg.startup_timeout_s - (deadline - time.time())
                logger.info("llama-server ready after %.1fs at %s",
                            elapsed, cfg.base_url)
                return
            if self._proc.poll() is not None:
                tail = self._read_log_tail()
                raise RuntimeError(
                    "llama-server exited during startup. "
                    f"Last log output:\n{tail}"
                )
            time.sleep(0.5)

        # Timed out
        tail = self._read_log_tail()
        self.stop()
        raise TimeoutError(
            f"llama-server did not become ready within "
            f"{cfg.startup_timeout_s:.0f}s.\nLast log:\n{tail}"
        )

    # --------------------------------------------------------------- stop API
    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            logger.info("Stopping llama-server (pid=%d)", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._log_fh and not self._log_fh.closed:
            self._log_fh.close()
        self._log_fh = None

    # ------------------------------------------------------------------ utils
    def _read_log_tail(self, n_chars: int = 1500) -> str:
        try:
            text = Path(self.config.log_path).read_text(
                encoding="utf-8", errors="replace"
            )
            return text[-n_chars:]
        except Exception:
            return "(no log available)"

    # --------------------------------------------------------- context helper
    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Singleton helper for the rest of the app
# ---------------------------------------------------------------------------
_SHARED: Optional[LlamaServer] = None


def get_shared_server(config: Optional[ServerConfig] = None) -> LlamaServer:
    """Return a process-wide singleton server, starting it on first call."""
    global _SHARED
    if _SHARED is None:
        _SHARED = LlamaServer(config)
        _SHARED.start()
    elif not _SHARED.is_running():
        _SHARED.start()
    return _SHARED
