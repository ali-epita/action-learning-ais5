"""PointCast application bootstrap: QApplication + backend + global hotkey.

The global hotkey runs on a pynput listener thread; it emits a Qt signal so the
interaction starts on the UI thread. The model is warmed up in a background
thread so the app is responsive immediately. Logging is configured here so the
whole pipeline narrates itself to the console.
"""

from __future__ import annotations

import sys

from ..utils.logging import get_logger, setup_logging
from .backends import get_backend
from .config import PointCastConfig
from .engine import GroundingEngine


def run(cfg: PointCastConfig | None = None) -> int:
    cfg = cfg or PointCastConfig()
    setup_logging(cfg.log_level, cfg.log_file)
    log = get_logger("pointcast")

    from PySide6 import QtWidgets

    from .controller import Controller

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # transient windows must not quit the app
    _install_sigint_quit(app)  # make Ctrl+C in the terminal exit cleanly

    log.info("starting PointCast: backend=%s model=%s dry_run=%s", cfg.backend, cfg.model_path, cfg.dry_run)
    log.info("confirm=%s countdown=%.1fs try_twice=%s tts=%s", cfg.confirm_mode, cfg.countdown_seconds, cfg.use_try_twice, cfg.tts_engine)

    backend = get_backend(cfg.backend, model_path=cfg.model_path, max_tokens=cfg.max_tokens)
    engine = GroundingEngine(backend, cfg)
    controller = Controller(cfg, engine)

    # Warm up on the controller's single inference thread (all MLX work runs
    # there; see controller threading note). This also binds the GPU stream.
    controller.warmup_async()
    _prewarm_macos_accessibility(log)
    _install_hotkey(cfg.hotkey, controller, log)
    if cfg.enable_voice:
        if cfg.enable_ptt_key:
            _install_ptt(cfg.ptt_key, controller, log)
            log.info("voice enabled: press %s for the search bar, then hold %s or click the mic to speak",
                     cfg.hotkey, cfg.ptt_key)
        else:
            log.info("voice enabled: press %s for the search bar, then click the mic to speak", cfg.hotkey)

    log.info("ready - press %s to point (Ctrl+C to quit)", cfg.hotkey)
    try:
        return app.exec()
    finally:
        controller.shutdown()


def _install_hotkey(hotkey: str, controller, log) -> None:
    from PySide6 import QtCore

    class _Bridge(QtCore.QObject):
        triggered = QtCore.Signal()

    bridge = _Bridge()
    bridge.triggered.connect(controller.start_interaction)  # queued: pynput thread -> UI thread
    controller._bridge = bridge  # keep a reference alive

    def on_activate() -> None:
        log.info("hotkey pressed (%s)", hotkey)
        bridge.triggered.emit()

    try:
        from pynput import keyboard

        hk = keyboard.GlobalHotKeys({hotkey: on_activate})
        hk.daemon = True
        hk.start()
        controller._hotkey = hk  # keep a reference alive
    except Exception as e:  # noqa: BLE001
        log.warning("global hotkey unavailable (%r) - grant Accessibility permission", e)


def _install_sigint_quit(app) -> None:
    """Make Ctrl+C in the launching terminal quit PointCast cleanly.

    Qt's event loop runs in C++ and rarely yields to the Python interpreter, so
    the default SIGINT handler never runs and Ctrl+C appears to hang. We install
    a handler that calls ``app.quit()`` and a periodic no-op timer that wakes the
    interpreter often enough to actually run it. ``app.quit()`` unwinds
    ``app.exec()`` so the ``finally: controller.shutdown()`` path runs.
    """
    import signal

    from PySide6 import QtCore

    try:
        signal.signal(signal.SIGINT, lambda *_: app.quit())
    except (ValueError, OSError):
        return  # not on the main thread (e.g. embedded) - skip
    timer = QtCore.QTimer(app)  # parented to app so it stays alive
    timer.timeout.connect(lambda: None)
    timer.start(200)


def _prewarm_macos_accessibility(log) -> None:
    """Resolve pyobjc's lazily-loaded ``AXIsProcessTrusted`` symbol on the main
    thread before any pynput listener starts.

    pynput's macOS listeners each call ``HIServices.AXIsProcessTrusted()`` once
    at thread startup. pyobjc resolves that symbol lazily via a non-thread-safe
    ``funcmap.pop(name)``. With ``--voice`` the global-hotkey listener and the
    push-to-talk listener start concurrently and race that first resolution; the
    loser dies with ``KeyError: 'AXIsProcessTrusted'`` and its key events never
    fire. Touching the symbol once here caches it as a normal module attribute,
    so the listener threads never hit the lazy path. No-op off macOS.
    """
    if sys.platform != "darwin":
        return
    try:
        import HIServices

        HIServices.AXIsProcessTrusted()
    except Exception as e:  # noqa: BLE001
        log.debug("accessibility prewarm skipped (%r)", e)


def _resolve_key(keyboard, name: str):
    name = name.lower()
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    return keyboard.KeyCode.from_char(name)


def _key_matches(key, target) -> bool:
    try:
        if key == target:
            return True
    except Exception:
        pass
    kc = getattr(key, "char", None)
    tc = getattr(target, "char", None)
    return kc is not None and kc == tc


def _install_ptt(ptt_key: str, controller, log) -> None:
    """Hold-to-talk: a key-down starts recording, key-up transcribes."""
    from PySide6 import QtCore

    class _PttBridge(QtCore.QObject):
        pressed = QtCore.Signal()
        released = QtCore.Signal()

    bridge = _PttBridge()
    bridge.pressed.connect(controller.on_ptt_press)  # queued: listener thread -> UI thread
    bridge.released.connect(controller.on_ptt_release)
    controller._ptt_bridge = bridge

    try:
        from pynput import keyboard

        target = _resolve_key(keyboard, ptt_key)
        state = {"down": False}

        def on_press(key) -> None:
            if _key_matches(key, target) and not state["down"]:
                state["down"] = True
                bridge.pressed.emit()

        def on_release(key) -> None:
            if _key_matches(key, target) and state["down"]:
                state["down"] = False
                bridge.released.emit()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        controller._ptt_listener = listener  # keep a reference alive
    except Exception as e:  # noqa: BLE001
        log.warning("push-to-talk unavailable (%r) - grant Accessibility/Microphone permission", e)
