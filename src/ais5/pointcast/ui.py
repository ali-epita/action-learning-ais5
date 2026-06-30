"""PySide6 widgets for PointCast.

  CrosshairOverlay — full-screen, click-through, translucent; paints the
                     high-contrast crosshair (and numbered markers in M3).
  ConfirmHUD       — small focusable panel: spoken/printed locator + countdown,
                     handles Enter/Esc/R.
  InputBar         — frameless line edit to type the target.

Qt paints in device-independent (logical) coordinates — the same space
pyautogui clicks in — so points from ``CoordinateMapper.ground_to_logical`` are
used directly here and for the click.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .config import PointCastConfig
from .locator import countdown_hint

Qt = QtCore.Qt


class CrosshairOverlay(QtWidgets.QWidget):
    def __init__(self, cfg: PointCastConfig):
        super().__init__()
        self.cfg = cfg
        self._point: tuple[float, float] | None = None
        self._markers: list[tuple[float, float]] = []
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._origin = QtCore.QPoint(0, 0)
        self._fit_primary_screen()

    def _fit_primary_screen(self) -> None:
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.geometry()
            self._origin = geo.topLeft()
            self.setGeometry(geo)

    def show_point(self, lx: float, ly: float, markers: list[tuple[float, float]] | None = None) -> None:
        ox, oy = self._origin.x(), self._origin.y()
        self._point = (lx - ox, ly - oy)
        self._markers = [(mx - ox, my - oy) for (mx, my) in (markers or [])]
        self.show()
        self.raise_()
        self.update()

    def show_markers_only(self, markers: list[tuple[float, float]]) -> None:
        ox, oy = self._origin.x(), self._origin.y()
        self._point = None
        self._markers = [(mx - ox, my - oy) for (mx, my) in markers]
        self.show()
        self.raise_()
        self.update()

    def clear(self) -> None:
        self._point = None
        self._markers = []
        self.update()
        self.hide()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        if self._point is None and not self._markers:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        for i, (mx, my) in enumerate(self._markers, start=1):
            self._draw_marker(p, mx, my, i)
        if self._point is not None:
            self._draw_crosshair(p, self._point[0], self._point[1])
        p.end()

    def _draw_crosshair(self, p: QtGui.QPainter, x: float, y: float) -> None:
        r = float(self.cfg.crosshair_radius)
        t = self.cfg.crosshair_thickness
        c = QtCore.QPointF(x, y)
        # dark halo for contrast on any background
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 170), t + 3))
        p.drawEllipse(c, r, r)
        p.drawLine(QtCore.QPointF(x - r - 14, y), QtCore.QPointF(x + r + 14, y))
        p.drawLine(QtCore.QPointF(x, y - r - 14), QtCore.QPointF(x, y + r + 14))
        col = QtGui.QColor(self.cfg.crosshair_color)
        p.setPen(QtGui.QPen(col, t))
        p.drawEllipse(c, r, r)
        p.drawLine(QtCore.QPointF(x - r - 14, y), QtCore.QPointF(x + r + 14, y))
        p.drawLine(QtCore.QPointF(x, y - r - 14), QtCore.QPointF(x, y + r + 14))
        p.setBrush(col)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(c, 3.5, 3.5)

    def _draw_marker(self, p: QtGui.QPainter, x: float, y: float, n: int) -> None:
        r = 18.0
        c = QtCore.QPointF(x, y)
        p.setBrush(QtGui.QColor(self.cfg.marker_color))
        p.setPen(QtGui.QPen(QtGui.QColor("#1f2937"), 2))
        p.drawEllipse(c, r, r)
        f = p.font()
        f.setBold(True)
        f.setPointSize(14)
        p.setFont(f)
        p.setPen(QtGui.QColor("#1f2937"))
        p.drawText(QtCore.QRectF(x - r, y - r, 2 * r, 2 * r), Qt.AlignmentFlag.AlignCenter, str(n))


class ConfirmHUD(QtWidgets.QWidget):
    confirmed = QtCore.Signal()
    canceled = QtCore.Signal()
    retry = QtCore.Signal()

    def __init__(self, cfg: PointCastConfig):
        super().__init__()
        self.cfg = cfg
        self._remaining = 0.0
        self._mode = "countdown"
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)

    def _build(self) -> None:
        panel = QtWidgets.QFrame(self)
        panel.setStyleSheet(f"background:{self.cfg.hud_bg}; border-radius:16px;")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(6)
        self._title = QtWidgets.QLabel("")
        self._title.setStyleSheet(f"color:{self.cfg.hud_fg}; font-size:21px; font-weight:600;")
        self._hint = QtWidgets.QLabel("")
        self._hint.setStyleSheet("color:#9ca3af; font-size:13px;")
        self._badge = QtWidgets.QLabel("")
        self._badge.setStyleSheet("color:#34d399; font-size:12px; font-weight:600;")
        for w in (self._badge, self._title, self._hint):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(w)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

    def show_confirm(self, phrase: str, badge: str, seconds: float, mode: str) -> None:
        self._mode = mode
        self._remaining = seconds
        self._title.setText(phrase)
        self._badge.setText(badge.upper())
        self._hint.setText(countdown_hint(seconds, confirm_mode=mode))
        self.adjustSize()
        self._position()
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()
        if mode == "countdown":
            self._timer.start()

    def _tick(self) -> None:
        self._remaining -= 0.1
        if self._remaining <= 0.05:
            self._timer.stop()
            self.confirmed.emit()
            return
        self._hint.setText(countdown_hint(self._remaining, confirm_mode=self._mode))

    def _position(self) -> None:
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        g = screen.geometry()
        self.move(int(g.center().x() - self.width() / 2), int(g.bottom() - self.height() - 90))

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        k = e.key()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._timer.stop()
            self.confirmed.emit()
        elif k == Qt.Key.Key_Escape:
            self._timer.stop()
            self.canceled.emit()
        elif k == Qt.Key.Key_R:
            self._timer.stop()
            self.retry.emit()
        else:
            super().keyPressEvent(e)

    def dismiss(self) -> None:
        self._timer.stop()
        self.hide()


class MicButton(QtWidgets.QAbstractButton):
    """A microphone button that pulses with sonar rings while listening.

    Click toggles voice input; the same listening animation is driven whether
    the user clicked it or is holding the push-to-talk key (the controller calls
    ``set_listening`` in both cases).
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(46, 46)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # never steal focus from the text field
        self.setToolTip("Hold Right Ctrl or click to speak")
        self._listening = False
        self._phase = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._animate)

    def set_listening(self, on: bool) -> None:
        if on == self._listening:
            return
        self._listening = on
        if on:
            self._phase = 0.0
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _animate(self) -> None:
        self._phase = (self._phase + 0.045) % 1.0
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(46, 46)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        accent = QtGui.QColor("#ef4444") if self._listening else QtGui.QColor("#60a5fa")

        # expanding sonar rings while listening (two, offset in phase)
        if self._listening:
            for k in (0, 1):
                ph = (self._phase + 0.5 * k) % 1.0
                rr = 15.0 + ph * 8.0
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QtGui.QPen(QtGui.QColor(239, 68, 68, int(130 * (1.0 - ph))), 2))
                p.drawEllipse(QtCore.QPointF(cx, cy), rr, rr)

        # round button base
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(69, 10, 10) if self._listening else QtGui.QColor("#111827"))
        p.drawEllipse(QtCore.QPointF(cx, cy), 15.0, 15.0)

        # microphone glyph
        pen = QtGui.QPen(accent, 2.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(accent)
        p.drawRoundedRect(QtCore.QRectF(cx - 4.5, cy - 10.0, 9.0, 14.0), 4.5, 4.5)  # head
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QtCore.QRectF(cx - 8.0, cy - 7.0, 16.0, 17.0), 200 * 16, 140 * 16)  # holder
        p.drawLine(QtCore.QPointF(cx, cy + 10.0), QtCore.QPointF(cx, cy + 13.0))  # stem
        p.drawLine(QtCore.QPointF(cx - 5.0, cy + 13.0), QtCore.QPointF(cx + 5.0, cy + 13.0))  # base
        p.end()


class InputBar(QtWidgets.QWidget):
    submitted = QtCore.Signal(str)
    canceled = QtCore.Signal()
    mic_clicked = QtCore.Signal()

    def __init__(self, cfg: PointCastConfig):
        super().__init__()
        self.cfg = cfg
        self._mic: MicButton | None = None
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._build()

    def _build(self) -> None:
        self._panel = QtWidgets.QFrame(self)
        lay = QtWidgets.QVBoxLayout(self._panel)
        lay.setContentsMargins(20, 16, 20, 14)
        lay.setSpacing(8)

        self._label = QtWidgets.QLabel("PointCast — describe where to click")
        self._label.setStyleSheet(f"color:{self.cfg.hud_fg}; font-size:14px; font-weight:600;")

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        self._edit = QtWidgets.QLineEdit()
        self._edit.setPlaceholderText("e.g. the search bar")
        self._edit.setStyleSheet(
            "QLineEdit{background:#1f2937; color:#f9fafb; border:1px solid #374151;"
            "border-radius:9px; padding:11px; font-size:16px;}"
        )
        self._edit.setMinimumWidth(480)
        self._edit.returnPressed.connect(self._submit)
        row.addWidget(self._edit, 1)
        if self.cfg.enable_voice:
            self._mic = MicButton()
            self._mic.clicked.connect(lambda: self.mic_clicked.emit())
            row.addWidget(self._mic, 0)

        self._hint = QtWidgets.QLabel(self._idle_hint())
        self._hint.setStyleSheet("color:#9ca3af; font-size:12px;")

        lay.addWidget(self._label)
        lay.addLayout(row)
        lay.addWidget(self._hint)
        self._set_border(None)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._panel)

    def _idle_hint(self) -> str:
        if not self.cfg.enable_voice:
            return "Enter to point  ·  Esc to cancel"
        if self.cfg.enable_ptt_key:
            return "Hold the talk key or click the mic to speak  ·  Enter to point  ·  Esc to cancel"
        return "Click the mic to speak  ·  Enter to point  ·  Esc to cancel"

    def _set_border(self, color: str | None) -> None:
        border = f"border:2px solid {color};" if color else "border:2px solid transparent;"
        self._panel.setStyleSheet(f"QFrame{{background:{self.cfg.hud_bg}; border-radius:16px; {border}}}")

    def set_state(self, state: str) -> None:
        """state in {'idle', 'listening', 'transcribing'} — drives label, border, mic."""
        if state == "listening":
            self._label.setText("Listening…")
            self._hint.setText(
                "Release the talk key or click the mic to stop"
                if self.cfg.enable_ptt_key else "Click the mic to stop"
            )
            self._set_border("#ef4444")
        elif state == "transcribing":
            self._label.setText("Transcribing…")
            self._hint.setText("one moment…")
            self._set_border("#60a5fa")
        else:
            self._label.setText("PointCast — describe where to click")
            self._hint.setText(self._idle_hint())
            self._set_border(None)
        if self._mic is not None:
            self._mic.set_listening(state == "listening")

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)

    def set_text(self, text: str) -> None:
        self._edit.setText(text)
        self._edit.setFocus()
        self._edit.selectAll()

    def text(self) -> str:
        return self._edit.text().strip()

    def _submit(self) -> None:
        text = self._edit.text().strip()
        if text:
            self.submitted.emit(text)

    def prompt(self, prefill: str = "") -> None:
        self.set_state("idle")
        self._edit.setText(prefill)
        self.adjustSize()
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            g = screen.geometry()
            self.move(int(g.center().x() - self.width() / 2), int(g.top() + g.height() * 0.16))
        self.show()
        self.raise_()
        self.activateWindow()
        self._edit.setFocus()
        self._edit.selectAll()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.hide()
            self.canceled.emit()
        else:
            super().keyPressEvent(e)

    def dismiss(self) -> None:
        if self._mic is not None:
            self._mic.set_listening(False)
        self.hide()


class DisambiguateHUD(QtWidgets.QWidget):
    """Shown when the gate is uncertain and there are multiple candidates.
    The numbered markers live on the overlay; this panel takes the key/voice pick."""

    picked = QtCore.Signal(int)  # 0-based candidate index
    canceled = QtCore.Signal()
    retry = QtCore.Signal()

    def __init__(self, cfg: PointCastConfig):
        super().__init__()
        self.cfg = cfg
        self._count = 0
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build()

    def _build(self) -> None:
        panel = QtWidgets.QFrame(self)
        panel.setStyleSheet(f"background:{self.cfg.hud_bg}; border-radius:16px;")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(6)
        self._title = QtWidgets.QLabel("I'm not sure — did you mean one of these?")
        self._title.setStyleSheet(f"color:{self.cfg.hud_fg}; font-size:19px; font-weight:600;")
        self._hint = QtWidgets.QLabel("")
        self._hint.setStyleSheet("color:#9ca3af; font-size:13px;")
        for w in (self._title, self._hint):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(w)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

    def show_options(self, count: int) -> None:
        self._count = count
        self._hint.setText(f"Press 1–{count}  ·  Esc cancel  ·  R retry")
        self.adjustSize()
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            g = screen.geometry()
            self.move(int(g.center().x() - self.width() / 2), int(g.bottom() - self.height() - 90))
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        digit = int(e.key()) - int(Qt.Key.Key_0)
        if 1 <= digit <= 9 and digit - 1 < self._count:
            self.picked.emit(digit - 1)
            return
        if e.key() == Qt.Key.Key_Escape:
            self.canceled.emit()
            return
        if e.key() == Qt.Key.Key_R:
            self.retry.emit()
            return
        super().keyPressEvent(e)

    def dismiss(self) -> None:
        self.hide()


class StatusHUD(QtWidgets.QWidget):
    """Transient, click-through status banner (e.g. 'Listening...')."""

    def __init__(self, cfg: PointCastConfig):
        super().__init__()
        self.cfg = cfg
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        panel = QtWidgets.QFrame(self)
        panel.setStyleSheet(f"background:{self.cfg.hud_bg}; border-radius:14px;")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(22, 12, 22, 12)
        self._label = QtWidgets.QLabel("")
        self._label.setStyleSheet(f"color:{self.cfg.hud_fg}; font-size:18px; font-weight:600;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._label)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

    def show_status(self, text: str) -> None:
        self._label.setText(text)
        self.adjustSize()
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            g = screen.geometry()
            self.move(int(g.center().x() - self.width() / 2), int(g.top() + g.height() * 0.10))
        self.show()
        self.raise_()

    def dismiss(self) -> None:
        self.hide()
