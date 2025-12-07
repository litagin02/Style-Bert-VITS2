"""
Simplified pyopenjtalk worker using multiprocessing.managers.BaseManager.
Runs pyopenjtalk in a separate process to avoid user dictionary access errors.
Maintains a single shared server so dictionary updates are visible to all clients.

Port discovery:
- Server binds to port 0 (OS auto-selects available port)
- Server writes actual port to a temp file
- Client reads port from file and connects
- authkey validates it's our server (not another app)

Idle timeout:
- Server tracks last activity time
- After 30 seconds of no requests, server exits automatically
- This ensures no orphan processes when user closes the app
"""

import atexit
import os
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.managers import BaseManager
from typing import Any, Optional, Union

from style_bert_vits2.logging import logger


_AUTH_KEY = b"sbv2_pyopenjtalk_worker"
_IDLE_TIMEOUT = 300  # seconds (5 minutes - allows for slow model loading)
_manager: Optional[BaseManager] = None


def _get_port_file() -> str:
    """Get path to port file (user-specific to avoid conflicts)."""
    return os.path.join(
        tempfile.gettempdir(),
        f"sbv2_pyopenjtalk_worker_{os.getuid() if hasattr(os, 'getuid') else os.getlogin()}.port",
    )


class _PyopenjtalkManager(BaseManager):
    pass


# Global activity tracker for server process
_last_activity: float = 0.0
_server_instance: Optional[Any] = None


def _update_activity() -> None:
    """Update last activity timestamp."""
    global _last_activity
    _last_activity = time.time()


def _idle_monitor(timeout: int) -> None:
    """Background thread that monitors idle time and shuts down server."""
    global _server_instance
    while True:
        time.sleep(5)  # Check every 5 seconds
        idle_time = time.time() - _last_activity
        if idle_time > timeout:
            logger.info(
                f"No activity for {timeout}s, shutting down pyopenjtalk worker server"
            )
            # Clean up port file
            try:
                os.remove(_get_port_file())
            except OSError:
                pass
            # Exit the process
            os._exit(0)


def _start_server() -> None:
    """Start the manager server in this process (called by subprocess)."""
    import pyopenjtalk

    global _last_activity, _server_instance
    _last_activity = time.time()

    class PyopenjtalkServer(BaseManager):
        pass

    # Wrap functions to track activity
    def wrap_with_activity(func):
        def wrapper(*args, **kwargs):
            _update_activity()
            return func(*args, **kwargs)

        return wrapper

    # Register actual pyopenjtalk functions with activity tracking
    PyopenjtalkServer.register(
        "run_frontend",
        wrap_with_activity(lambda text: pyopenjtalk.run_frontend(text)),
    )
    PyopenjtalkServer.register(
        "make_label",
        wrap_with_activity(lambda njd_features: pyopenjtalk.make_label(njd_features)),
    )
    PyopenjtalkServer.register(
        "mecab_dict_index",
        wrap_with_activity(
            lambda path, out_path, dn_mecab=None: pyopenjtalk.mecab_dict_index(
                path, out_path, dn_mecab
            )
        ),
    )
    PyopenjtalkServer.register(
        "update_global_jtalk_with_user_dict",
        wrap_with_activity(
            lambda paths: pyopenjtalk.update_global_jtalk_with_user_dict(paths)
        ),
    )
    PyopenjtalkServer.register(
        "unset_user_dict",
        wrap_with_activity(lambda: pyopenjtalk.unset_user_dict()),
    )

    # Bind to port 0 - OS will assign an available port
    manager = PyopenjtalkServer(address=("localhost", 0), authkey=_AUTH_KEY)
    server = manager.get_server()
    _server_instance = server

    # Get the actual port assigned by OS
    actual_port = server.address[1]

    # Write port to file
    port_file = _get_port_file()
    with open(port_file, "w") as f:
        f.write(str(actual_port))

    # Start idle monitor thread
    monitor_thread = threading.Thread(
        target=_idle_monitor,
        args=(_IDLE_TIMEOUT,),
        daemon=True,
    )
    monitor_thread.start()

    logger.info(
        f"pyopenjtalk worker server started on port {actual_port} (idle timeout: {_IDLE_TIMEOUT}s)"
    )
    server.serve_forever()


# Register callable proxies for client side
_PyopenjtalkManager.register("run_frontend")
_PyopenjtalkManager.register("make_label")
_PyopenjtalkManager.register("mecab_dict_index")
_PyopenjtalkManager.register("update_global_jtalk_with_user_dict")
_PyopenjtalkManager.register("unset_user_dict")


def _read_port_from_file() -> Optional[int]:
    """Read port from port file if it exists."""
    port_file = _get_port_file()
    try:
        with open(port_file, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _try_connect(port: int) -> Optional[BaseManager]:
    """Try to connect to server at given port. Returns manager if successful."""
    try:
        manager = _PyopenjtalkManager(address=("localhost", port), authkey=_AUTH_KEY)
        manager.connect()
        return manager
    except (ConnectionRefusedError, EOFError):
        # Server not running on this port
        return None
    except Exception as e:
        # Auth error or other issue - not our server
        logger.debug(f"Connection to port {port} failed: {e}")
        return None


def initialize_worker() -> None:
    """Initialize connection to pyopenjtalk worker server, starting it if needed."""
    global _manager
    if _manager is not None:
        return

    # Try connecting to existing server using port file
    port = _read_port_from_file()
    if port is not None:
        manager = _try_connect(port)
        if manager is not None:
            _manager = manager
            logger.debug(
                f"Connected to existing pyopenjtalk worker server on port {port}"
            )
            return
        else:
            logger.debug(
                "Port file exists but server not responding, starting new server"
            )

    # Start new server process
    logger.debug("Starting new pyopenjtalk worker server")

    # Start server as subprocess
    server_code = f"""
import sys
sys.path.insert(0, {repr(str(sys.path[0]))})
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker import _start_server
_start_server()
"""

    if sys.platform.startswith("win"):
        cf = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        subprocess.Popen(
            [sys.executable, "-c", server_code],
            creationflags=cf,
            startupinfo=si,
        )
    else:
        subprocess.Popen(
            [sys.executable, "-c", server_code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Wait for server to start and write port file
    for i in range(20):
        time.sleep(0.5)
        port = _read_port_from_file()
        if port is not None:
            manager = _try_connect(port)
            if manager is not None:
                _manager = manager
                logger.debug(f"pyopenjtalk worker server started on port {port}")
                atexit.register(terminate_worker)
                return

    raise TimeoutError("サーバーに接続できませんでした")


def terminate_worker() -> None:
    """Terminate the worker connection."""
    global _manager
    if _manager is None:
        return

    logger.debug("pyopenjtalk worker connection closed")
    _manager = None


# pyopenjtalk interface functions


def run_frontend(text: str, jtalk: Any = None) -> list[dict[str, Any]]:
    """Run OpenJTalk frontend. The jtalk parameter is ignored when using worker."""
    global _manager
    if _manager is not None:
        try:
            return _manager.run_frontend(text)._getvalue()
        except (ConnectionRefusedError, EOFError, BrokenPipeError):
            # Server has shut down, reconnect
            logger.debug("pyopenjtalk worker server connection lost, reconnecting...")
            _manager = None
            initialize_worker()
            return _manager.run_frontend(text)._getvalue()
    else:
        import pyopenjtalk

        return pyopenjtalk.run_frontend(text, jtalk=jtalk)


def make_label(njd_features: Any, jtalk: Any = None) -> list[str]:
    """Make label from NJD features. The jtalk parameter is ignored when using worker."""
    global _manager
    if _manager is not None:
        try:
            return _manager.make_label(njd_features)._getvalue()
        except (ConnectionRefusedError, EOFError, BrokenPipeError):
            # Server has shut down, reconnect
            logger.debug("pyopenjtalk worker server connection lost, reconnecting...")
            _manager = None
            initialize_worker()
            return _manager.make_label(njd_features)._getvalue()
    else:
        import pyopenjtalk

        return pyopenjtalk.make_label(njd_features, jtalk=jtalk)


def mecab_dict_index(path: str, out_path: str, dn_mecab: Optional[str] = None) -> None:
    global _manager
    if _manager is not None:
        try:
            _manager.mecab_dict_index(path, out_path, dn_mecab)._getvalue()
        except (ConnectionRefusedError, EOFError, BrokenPipeError):
            logger.debug("pyopenjtalk worker server connection lost, reconnecting...")
            _manager = None
            initialize_worker()
            _manager.mecab_dict_index(path, out_path, dn_mecab)._getvalue()
    else:
        import pyopenjtalk

        pyopenjtalk.mecab_dict_index(path, out_path, dn_mecab)


def update_global_jtalk_with_user_dict(paths: Union[str, list[str]]) -> None:
    global _manager
    if _manager is not None:
        try:
            _manager.update_global_jtalk_with_user_dict(paths)._getvalue()
        except (ConnectionRefusedError, EOFError, BrokenPipeError):
            logger.debug("pyopenjtalk worker server connection lost, reconnecting...")
            _manager = None
            initialize_worker()
            _manager.update_global_jtalk_with_user_dict(paths)._getvalue()
    else:
        import pyopenjtalk

        pyopenjtalk.update_global_jtalk_with_user_dict(paths)


def unset_user_dict() -> None:
    global _manager
    if _manager is not None:
        try:
            _manager.unset_user_dict()._getvalue()
        except (ConnectionRefusedError, EOFError, BrokenPipeError):
            logger.debug("pyopenjtalk worker server connection lost, reconnecting...")
            _manager = None
            initialize_worker()
            _manager.unset_user_dict()._getvalue()
    else:
        import pyopenjtalk

        pyopenjtalk.unset_user_dict()
