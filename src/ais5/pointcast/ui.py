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

import re

from PySide6 import QtCore, QtGui, QtWidgets

from .command import detect_verb
from .config import PointCastConfig
from .locator import countdown_hint

Qt = QtCore.Qt

# ── design tokens: every surface shares one language ─────────────────────────
PANEL_BG = "#15181e"          # one panel background everywhere
PANEL_BORDER = "#2a2f3a"      # hairline border
FIELD_BG = "#1c2027"          # input field background
TEXT = "#e8eaee"              # primary text
MUTED = "#8b93a1"             # secondary text
RADIUS = 12                   # one corner radius
# Per-verb accent colours for the command keyword.
VERB_COLORS = {"click": "#60a5fa", "read": "#a78bfa", "type": "#34d399", "press": "#fbbf24"}


def rgba(hex_color: str, alpha: float) -> str:
    """Qt stylesheets read 8-digit hex as #AARRGGBB, not CSS's #RRGGBBAA —
    build an explicit rgba() instead so translucent accents stay the right hue."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def panel_qss(name: str, *, border: str = PANEL_BORDER, radius: int = RADIUS) -> str:
    """The shared panel style, scoped by objectName so it never cascades onto
    child labels (QLabel subclasses QFrame)."""
    return f"#{name}{{background:{PANEL_BG}; border:1px solid {border}; border-radius:{radius}px;}}"


def _make_panel(parent: QtWidgets.QWidget, name: str, *, border: str = PANEL_BORDER) -> QtWidgets.QFrame:
    panel = QtWidgets.QFrame(parent)
    panel.setObjectName(name)
    panel.setStyleSheet(panel_qss(name, border=border))
    return panel

# The leading keyword to colour: a verb, or a read-style question opener.
_KEYWORD_RE = re.compile(
    r"^(\s*)(right[\s-]?click|click|read|type|press"
    r"|what(?:'s| is| are| does)?|how (?:much|many)|tell me)\b",
    re.IGNORECASE,
)


def _keyword_verb(word: str) -> str:
    w = word.strip().lower()
    if w.startswith("type"):
        return "type"
    if w.startswith("press"):
        return "press"
    if w.startswith(("click", "right")):
        return "click"  # click / right click
    return "read"  # read / what… / how… / tell me


class _VerbHighlighter(QtGui.QSyntaxHighlighter):
    """Colours ONLY the leading command keyword; the rest stays default."""

    def highlightBlock(self, text: str) -> None:  # noqa: N802 (Qt override)
        m = _KEYWORD_RE.match(text)
        if not m:
            return
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(VERB_COLORS[_keyword_verb(m.group(2))]))
        fmt.setFontWeight(QtGui.QFont.Weight.Bold)
        self.setFormat(m.start(2), len(m.group(2)), fmt)


class CommandEdit(QtWidgets.QTextEdit):
    """Single-line input that colours the leading command keyword. Behaves like
    a QLineEdit for the caller (text/setText/returnPressed/escapePressed)."""

    returnPressed = QtCore.Signal()
    escapePressed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setWordWrapMode(QtGui.QTextOption.WrapMode.NoWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabChangesFocus(True)
        # Set the font on the widget (not the stylesheet) so font metrics are
        # correct, then size the field to exactly one text line -> the text
        # sits vertically centered instead of hugging the top.
        f = self.font()
        f.setPixelSize(16)
        self.setFont(f)
        self.document().setDocumentMargin(2)
        self.setFixedHeight(self.fontMetrics().height() + 22)
        self._hl = _VerbHighlighter(self.document())

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:  # noqa: N802
        k = e.key()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.returnPressed.emit()
            e.accept()
            return
        if k == Qt.Key.Key_Escape:
            self.escapePressed.emit()
            e.accept()
            return
        super().keyPressEvent(e)

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, t: str) -> None:  # noqa: N802
        self.setPlainText(t)

    def deselect(self) -> None:
        c = self.textCursor()
        c.clearSelection()
        self.setTextCursor(c)

    def end(self, *_a) -> None:  # match QLineEdit.end(mark) signature loosely
        self.moveCursor(QtGui.QTextCursor.MoveOperation.End)


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
        self._deadline = 0.0  # monotonic time when the countdown auto-confirms
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
        panel = _make_panel(self, "confirmPanel")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(4)
        self._badge = QtWidgets.QLabel("")
        self._badge.setStyleSheet("color:#34d399; font-size:10px; font-weight:700; letter-spacing:1.5px;")
        self._title = QtWidgets.QLabel("")
        self._title.setStyleSheet(f"color:{TEXT}; font-size:16px; font-weight:600;")
        self._hint = QtWidgets.QLabel("")
        self._hint.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        for w in (self._badge, self._title, self._hint):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(w)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

    def show_confirm(self, phrase: str, badge: str, seconds: float, mode: str) -> None:
        import time

        self._mode = mode
        self._deadline = time.monotonic() + seconds
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
        import time

        # Wall-clock countdown: QTimer ticks drift under load; the deadline is
        # authoritative so "3 seconds" is really 3 seconds.
        remaining = self._deadline - time.monotonic()
        if remaining <= 0.05:
            self._timer.stop()
            self.confirmed.emit()
            return
        self._hint.setText(countdown_hint(remaining, confirm_mode=self._mode))

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
    settings_clicked = QtCore.Signal()
    live_clicked = QtCore.Signal()

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
        self._panel = _make_panel(self, "inputPanel")
        lay = QtWidgets.QVBoxLayout(self._panel)
        lay.setContentsMargins(14, 12, 14, 10)
        lay.setSpacing(7)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)

        self._edit = CommandEdit()
        self._edit.setPlaceholderText("click the search bar  ·  read my grade  ·  type hello into the box")
        self._edit.setMinimumWidth(560)
        self._edit.returnPressed.connect(self._submit)
        self._edit.escapePressed.connect(self._on_escape)
        self._edit.textChanged.connect(lambda: self._update_verb_style(self._edit.text()))
        # Soft breathing glow that only appears when a verb is recognised.
        self._edit_glow = QtWidgets.QGraphicsDropShadowEffect(self._edit)
        self._edit_glow.setOffset(0, 0)
        self._edit_glow.setBlurRadius(0)
        self._edit.setGraphicsEffect(self._edit_glow)
        self._pulse = QtCore.QPropertyAnimation(self._edit_glow, b"blurRadius", self)
        self._pulse.setDuration(1500)
        self._pulse.setLoopCount(-1)
        self._pulse.setKeyValueAt(0.0, 6)
        self._pulse.setKeyValueAt(0.5, 16)
        self._pulse.setKeyValueAt(1.0, 6)
        self._pulse.setEasingCurve(QtCore.QEasingCurve.Type.InOutSine)
        self._current_verb: str | None = None
        row.addWidget(self._edit, 1)
        self._mic = MicButton()
        self._mic.clicked.connect(lambda: self.mic_clicked.emit())
        self._mic.setVisible(self.cfg.enable_voice)
        row.addWidget(self._mic, 0)

        def _icon_btn(glyph: str, tip: str, hover_color: str) -> QtWidgets.QPushButton:
            b = QtWidgets.QPushButton(glyph)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setFixedSize(34, 34)
            b.setToolTip(tip)
            b.setStyleSheet(
                f"QPushButton{{background:transparent; color:{MUTED}; border:none; font-size:18px;}}"
                f"QPushButton:hover{{color:{hover_color};}}"
            )
            return b

        # ◉ live: starts a hands-free session (echoes the LiveHUD's ● dot).
        # Shown only when live is enabled; hover foreshadows the red live mode.
        self._live_btn = _icon_btn("◉", "Live session — hands-free until you say “done”", "#ef4444")
        self._live_btn.setVisible(self.cfg.enable_live_voice)
        self._live_btn.clicked.connect(lambda: self.live_clicked.emit())
        row.addWidget(self._live_btn, 0)

        self._gear = _icon_btn("⚙", "Settings", TEXT)
        self._gear.clicked.connect(lambda: self.settings_clicked.emit())
        row.addWidget(self._gear, 0)

        self._hint = QtWidgets.QLabel(self._idle_hint())
        self._hint.setStyleSheet(f"color:{MUTED}; font-size:11px;")

        # keep a hidden label handle for voice-state text (no title row anymore)
        self._label = QtWidgets.QLabel("")
        self._label.hide()

        lay.addLayout(row)
        lay.addWidget(self._hint)
        self._set_border(None)
        self._apply_edit_style(None)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._panel)

    def _apply_edit_style(self, verb: str | None) -> None:
        """Neutral text (the keyword is coloured by the highlighter); only the
        border picks up a faint verb accent."""
        border = VERB_COLORS.get(verb) if verb else PANEL_BORDER
        self._edit.setStyleSheet(
            f"QTextEdit{{background:{FIELD_BG}; color:{TEXT}; border:1px solid {border};"
            f"border-radius:9px; padding:8px 10px;}}"
        )

    def _update_verb_style(self, text: str) -> None:
        verb = detect_verb(text)
        if verb == self._current_verb:
            return
        self._current_verb = verb
        self._apply_edit_style(verb)
        if verb is None:
            self._pulse.stop()
            self._edit_glow.setBlurRadius(0)
        else:
            self._edit_glow.setColor(QtGui.QColor(VERB_COLORS[verb]))
            self._pulse.start()

    def _idle_hint(self) -> str:
        verbs = "read  ·  click  ·  type  ·  press"
        if not self.cfg.enable_voice:
            return f"{verbs}     Enter to run  ·  Esc to cancel"
        if self.cfg.enable_ptt_key:
            return f"{verbs}     hold the talk key or click the mic  ·  Esc to cancel"
        return f"{verbs}     click the mic to speak  ·  Esc to cancel"

    def _set_border(self, color: str | None) -> None:
        self._panel.setStyleSheet(panel_qss("inputPanel", border=color or PANEL_BORDER))

    def set_state(self, state: str) -> None:
        """state in {'idle', 'listening', 'transcribing'} — drives hint, border, mic."""
        if state == "listening":
            self._hint.setText(
                "listening — release the talk key or click the mic to stop"
                if self.cfg.enable_ptt_key else "listening — click the mic to stop"
            )
            self._set_border("#ef4444")
        elif state == "transcribing":
            self._hint.setText("transcribing…")
            self._set_border("#60a5fa")
        else:
            self._hint.setText(self._idle_hint())
            self._set_border(None)
        if self._mic is not None:
            self._mic.set_listening(state == "listening")

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)

    def set_voice_enabled(self, on: bool) -> None:
        """Show/hide the mic when voice input is toggled in settings."""
        self._mic.setVisible(on)
        self._hint.setText(self._idle_hint())

    def set_live_enabled(self, on: bool) -> None:
        """Show/hide the ◉ live button when live voice is toggled in settings."""
        self._live_btn.setVisible(on)

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

    def _on_escape(self) -> None:
        # Escape is consumed by the focused CommandEdit and routed here.
        self.hide()
        self.canceled.emit()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self._on_escape()
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
        panel = _make_panel(self, "disambigPanel")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(4)
        self._title = QtWidgets.QLabel("Not sure — did you mean one of these?")
        self._title.setStyleSheet(f"color:{TEXT}; font-size:15px; font-weight:600;")
        self._hint = QtWidgets.QLabel("")
        self._hint.setStyleSheet(f"color:{MUTED}; font-size:12px;")
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


class ResultCard(QtWidgets.QWidget):
    """Compact strip that shows a read-back answer, so the result is available
    by ear AND by eye: a small violet dot + the query, then the answer. Sits in
    the lower third (not over the content being read). Click-through; hidden
    before any screenshot; auto-dismisses (the controller manages timing)."""

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
        violet = VERB_COLORS["read"]
        panel = _make_panel(self, "resultPanel", border=rgba(violet, 0.4))
        glow = QtWidgets.QGraphicsDropShadowEffect(panel)
        glow.setOffset(0, 0)
        glow.setBlurRadius(22)
        glow.setColor(QtGui.QColor(violet))
        panel.setGraphicsEffect(glow)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(20, 12, 20, 13)
        lay.setSpacing(3)
        self._query = QtWidgets.QLabel("")
        self._query.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        self._query.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._answer = QtWidgets.QLabel("")
        self._answer.setStyleSheet(f"color:{TEXT}; font-size:19px; font-weight:650;")
        self._answer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._answer.setWordWrap(True)
        self._answer.setMinimumWidth(360)
        self._answer.setMaximumWidth(640)
        lay.addWidget(self._query)
        lay.addWidget(self._answer)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)
        self._violet = violet

    def show_answer(self, query: str, answer: str) -> None:
        q = query.strip()
        self._query.setText(f"●  {q}" if q else "●")
        self._query.setStyleSheet(f"color:{self._violet}; font-size:11px; font-weight:600;")
        self._answer.setText(answer.strip())
        self.adjustSize()
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            g = screen.geometry()
            self.move(int(g.center().x() - self.width() / 2), int(g.bottom() - self.height() - 110))
        self.show()
        self.raise_()

    def dismiss(self) -> None:
        self.hide()


class RecordHUD(QtWidgets.QWidget):
    """Recording controls shown while a recipe is being demonstrated: the
    recipe name, a step counter, and Save / Cancel buttons. The typed/spoken
    "save recipe" and "cancel recording" commands still work; these are the
    pointer-friendly equivalents. Hidden before each screenshot (like the
    status banner) so the model never grounds these controls."""

    save = QtCore.Signal()
    cancel = QtCore.Signal()

    def __init__(self, cfg: PointCastConfig):
        super().__init__()
        self.cfg = cfg
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        panel = _make_panel(self, "recordPanel")
        lay = QtWidgets.QHBoxLayout(panel)
        lay.setContentsMargins(16, 10, 12, 10)
        lay.setSpacing(12)

        col = QtWidgets.QVBoxLayout()
        col.setSpacing(2)
        self._title = QtWidgets.QLabel("")
        self._title.setStyleSheet(f"color:{TEXT}; font-size:13px; font-weight:600;")
        self._hint = QtWidgets.QLabel("")
        self._hint.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        col.addWidget(self._title)
        col.addWidget(self._hint)
        lay.addLayout(col, 1)

        def _ghost(label: str, accent: str) -> QtWidgets.QPushButton:
            b = QtWidgets.QPushButton(label)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton{{background:transparent; color:{accent}; border:1px solid {rgba(accent, 0.4)};"
                f"border-radius:8px; padding:5px 12px; font-size:12px; font-weight:600;}}"
                f"QPushButton:hover{{background:{rgba(accent, 0.13)}; border-color:{accent};}}"
            )
            return b

        self._cancel_btn = _ghost("Cancel", MUTED)
        self._cancel_btn.clicked.connect(lambda: self.cancel.emit())
        self._save_btn = _ghost("Save recipe", "#34d399")
        self._save_btn.clicked.connect(lambda: self.save.emit())
        lay.addWidget(self._cancel_btn, 0)
        lay.addWidget(self._save_btn, 0)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

    def show_recording(self, name: str, steps: int) -> None:
        self._title.setText(f"<span style='color:#ef4444'>●</span>  Recording: {name}")
        self._hint.setText(
            f"{steps} step{'s' if steps != 1 else ''}  ·  or say “save recipe” / “cancel recording”"
        )
        self.adjustSize()
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            g = screen.geometry()
            self.move(int(g.center().x() - self.width() / 2), int(g.top() + g.height() * 0.10))
        self.show()
        self.raise_()

    def dismiss(self) -> None:
        self.hide()


class LiveHUD(QtWidgets.QWidget):
    """The live-session surface: replaces the command bar while a hands-free
    voice session runs. Shows a pulsing mic state, the last thing you said, and
    what PointCast is doing, plus a scrolling short history — so the user can
    SEE the conversation, not just hear it. Dismissed only when the session
    ends ('done'/'thank you'/hotkey)."""

    STATE_COLORS = {"listening": "#ef4444", "thinking": "#60a5fa", "acting": "#34d399"}

    def __init__(self, cfg: PointCastConfig):
        super().__init__()
        self.cfg = cfg
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        panel = _make_panel(self, "livePanel")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(9)
        self._dot = QtWidgets.QLabel("●")
        self._dot.setStyleSheet("color:#ef4444; font-size:13px;")
        self._state = QtWidgets.QLabel("Listening")
        self._state.setStyleSheet(f"color:{TEXT}; font-size:15px; font-weight:700; letter-spacing:0.3px;")
        header.addWidget(self._dot, 0)
        header.addWidget(self._state, 0)
        header.addStretch(1)
        endhint = QtWidgets.QLabel("say “done” to finish")
        endhint.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        header.addWidget(endhint, 0)
        lay.addLayout(header)

        # the current transcript, large
        self._transcript = QtWidgets.QLabel("")
        self._transcript.setStyleSheet(f"color:{TEXT}; font-size:20px; font-weight:600;")
        self._transcript.setWordWrap(True)
        self._transcript.setMinimumWidth(460)
        self._transcript.setMaximumWidth(560)
        lay.addWidget(self._transcript)

        # a short rolling history of recent turns
        self._history = QtWidgets.QLabel("")
        self._history.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        self._history.setWordWrap(True)
        lay.addWidget(self._history)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

        self._turns: list[str] = []
        self._pulse = 0
        self._levels = (1.0, 0.6, 0.28, 0.6)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._tick)

    def _place(self) -> None:
        self.layout().activate()
        self.adjustSize()
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            g = screen.geometry()
            self.move(int(g.center().x() - self.width() / 2), int(g.top() + g.height() * 0.14))
        self.show()
        self.raise_()

    def _tick(self) -> None:
        self._pulse = (self._pulse + 1) % len(self._levels)
        c = self.STATE_COLORS.get(self._cur_state, "#ef4444")
        self._dot.setStyleSheet(f"color:{rgba(c, self._levels[self._pulse])}; font-size:13px;")

    def begin(self) -> None:
        self._turns = []
        self._history.setText("")
        self.set_state("listening")
        self.set_transcript("")
        self._place()
        self._timer.start()

    def set_state(self, state: str, detail: str = "") -> None:
        self._cur_state = state
        label = {"listening": "Listening", "thinking": "Thinking", "acting": "Working"}.get(state, state.title())
        self._state.setText(f"{label}{('  ·  ' + detail) if detail else ''}")
        c = self.STATE_COLORS.get(state, "#ef4444")
        self._dot.setStyleSheet(f"color:{c}; font-size:13px;")
        self._place()

    def set_transcript(self, text: str) -> None:
        # the live/just-heard command, shown prominently
        self._transcript.setText(f"“{text}”" if text else "…")
        self._place()

    def add_turn(self, said: str, outcome: str = "") -> None:
        line = f"“{said}”" + (f"  →  {outcome}" if outcome else "")
        self._turns.append(line)
        self._turns = self._turns[-3:]  # keep it short
        self._history.setText("\n".join(self._turns))
        self._place()

    def dismiss(self) -> None:
        self._timer.stop()
        self.hide()


class SettingsPanel(QtWidgets.QWidget):
    """Compact in-app settings: the runtime toggles that previously required
    CLI flags. Emits ``changed(key, value)``; the controller applies + persists.
    Hidden before every screenshot like every other PointCast surface."""

    changed = QtCore.Signal(str, object)
    closed = QtCore.Signal()

    _TOGGLES = (
        ("enable_voice", "Voice input", "mic button in the command bar"),
        ("enable_live_voice", "Live voice", "adds a ◉ live button for a hands-free session"),
        ("enable_tasks", "Task recipes", "multi-step recipes like “check my storage”"),
        ("enable_agent", "Agent mode", "“do <goal>” — the model plans each click"),
        ("use_deep_links", "Deep links", "recipes jump straight to Settings panes"),
        ("use_try_twice", "Try-Twice", "self-check ladder before committing a click"),
        ("retina_crop", "Retina crops", "re-check zooms use native resolution"),
        ("show_trust_panel", "Trust panel", "on-device proof card, bottom right"),
        ("dry_run", "Dry run", "never dispatch a real click (safe rehearsal)"),
    )

    def __init__(self, cfg: PointCastConfig):
        super().__init__()
        self.cfg = cfg
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        panel = _make_panel(self, "settingsPanel")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(6)

        title = QtWidgets.QLabel("SETTINGS")
        title.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:700; letter-spacing:1.5px;")
        lay.addWidget(title)

        self._boxes: dict[str, QtWidgets.QCheckBox] = {}
        box_qss = (
            f"QCheckBox{{color:{TEXT}; font-size:13px; font-weight:600; spacing:8px;}}"
            f"QCheckBox::indicator{{width:15px; height:15px; border:1px solid {PANEL_BORDER};"
            f"border-radius:4px; background:{FIELD_BG};}}"
            "QCheckBox::indicator:checked{background:#34d399; border-color:#34d399;}"
        )
        for key, label, tip in self._TOGGLES:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(8)
            cb = QtWidgets.QCheckBox(label)
            cb.setStyleSheet(box_qss)
            cb.setCursor(Qt.CursorShape.PointingHandCursor)
            cb.toggled.connect(lambda v, k=key: self.changed.emit(k, bool(v)))
            self._boxes[key] = cb
            hint = QtWidgets.QLabel(tip)
            hint.setStyleSheet(f"color:{MUTED}; font-size:11px;")
            row.addWidget(cb, 0)
            row.addStretch(1)
            row.addWidget(hint, 0)
            lay.addLayout(row)

        # TTS engine choice
        tts_row = QtWidgets.QHBoxLayout()
        tts_label = QtWidgets.QLabel("Voice output")
        tts_label.setStyleSheet(f"color:{TEXT}; font-size:13px; font-weight:600;")
        self._tts = QtWidgets.QComboBox()
        self._tts.addItems(["say", "neural", "off"])
        self._tts.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tts.setStyleSheet(
            f"QComboBox{{background:{FIELD_BG}; color:{TEXT}; border:1px solid {PANEL_BORDER};"
            f"border-radius:6px; padding:3px 10px; font-size:12px;}}"
        )
        self._tts.currentTextChanged.connect(lambda v: self.changed.emit("tts_engine", v))
        tts_row.addWidget(tts_label, 0)
        tts_row.addStretch(1)
        tts_row.addWidget(self._tts, 0)
        lay.addLayout(tts_row)

        note = QtWidgets.QLabel("model, hotkey and resolution are set at launch (CLI)")
        note.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        lay.addWidget(note)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

    def sync_from(self, cfg: PointCastConfig) -> None:
        """Reflect the current config without re-emitting change signals."""
        for key, cb in self._boxes.items():
            cb.blockSignals(True)
            cb.setChecked(bool(getattr(cfg, key)))
            cb.blockSignals(False)
        self._tts.blockSignals(True)
        self._tts.setCurrentText(cfg.tts_engine)
        self._tts.blockSignals(False)

    def open_near(self, anchor: QtWidgets.QWidget) -> None:
        self.sync_from(self.cfg)
        self.adjustSize()
        g = anchor.geometry()
        self.move(g.x() + g.width() - self.width(), g.y() + g.height() + 8)
        self.show()
        self.raise_()
        self.activateWindow()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape:
            self.dismiss()
            return
        super().keyPressEvent(e)

    def dismiss(self) -> None:
        if self.isVisible():
            self.hide()
            self.closed.emit()


class TrustPanel(QtWidgets.QWidget):
    """Egress-proof trust card (M5): model identity, live peak memory, weights
    sha256, and the offline statement. Click-through, never takes focus, and
    the controller hides it before every screenshot so the model never grounds
    our own panel."""

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
        panel = _make_panel(self, "trustPanel")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(3)
        self._title = QtWidgets.QLabel("ON-DEVICE · SCREEN NEVER LEAVES THIS MAC")
        self._title.setStyleSheet("color:#34d399; font-size:9px; font-weight:700; letter-spacing:1px;")
        self._model = QtWidgets.QLabel("")
        self._memory = QtWidgets.QLabel("peak memory: …")
        self._hash = QtWidgets.QLabel("weights sha256: computing…")
        for w in (self._model, self._memory, self._hash):
            w.setStyleSheet(f"color:{MUTED}; font-size:10px; font-family:Menlo,monospace;")
        for w in (self._title, self._model, self._memory, self._hash):
            lay.addWidget(w)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)

    def set_model_line(self, text: str) -> None:
        self._model.setText(text)
        self._reposition()

    def set_memory(self, nbytes: object) -> None:
        if isinstance(nbytes, (int, float)) and nbytes > 0:
            self._memory.setText(f"peak memory: {nbytes / 1e9:.2f} GB (8 GB budget)")
        else:
            self._memory.setText("peak memory: n/a")
        self._reposition()

    def set_hash(self, text: str) -> None:
        self._hash.setText(f"weights sha256: {text}")
        self._reposition()

    def _reposition(self) -> None:
        self.adjustSize()
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        g = screen.geometry()
        self.move(int(g.right() - self.width() - 24), int(g.bottom() - self.height() - 24))

    def show_panel(self) -> None:
        self._reposition()
        self.show()
        self.raise_()

    def dismiss(self) -> None:
        self.hide()


class StatusHUD(QtWidgets.QWidget):
    """Slim, click-through status pill. Two modes:

      show_status(text)          — static line ("Step 2 of 3: …")
      show_busy(text, accent)    — thinking state: an accent dot + the text +
                                   a softly cycling ellipsis, so the app never
                                   looks dead while the model works.
    """

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
        panel = _make_panel(self, "statusPanel")
        lay = QtWidgets.QHBoxLayout(panel)
        lay.setContentsMargins(14, 8, 14, 8)
        self._label = QtWidgets.QLabel("")
        self._label.setStyleSheet(f"color:{TEXT}; font-size:13px; font-weight:600;")
        lay.addWidget(self._label, 0, Qt.AlignmentFlag.AlignCenter)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(panel)
        # Busy = a gently pulsing accent dot before constant text. Only the
        # dot's opacity changes, so the pill's geometry never shifts.
        self._busy_text = ""
        self._accent = "#60a5fa"
        self._phase = 0
        self._pulse_levels = (1.0, 0.65, 0.3, 0.65)
        self._busy_timer = QtCore.QTimer(self)
        self._busy_timer.setInterval(380)
        self._busy_timer.timeout.connect(self._busy_tick)

    def _place(self) -> None:
        # Size EXACTLY to content every time: a frameless tool window keeps its
        # old (possibly wider) size otherwise, which strands the text mid-pill.
        self.layout().activate()
        self.setFixedSize(self.sizeHint())
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            g = screen.geometry()
            self.move(int(g.center().x() - self.width() / 2), int(g.top() + g.height() * 0.10))
        self.show()
        self.raise_()

    def _busy_html(self, alpha: float) -> str:
        return (
            f"<span style='color:{rgba(self._accent, alpha)}'>●</span>"
            f"<span style='color:{TEXT}'>&nbsp;&nbsp;{self._busy_text}…</span>"
        )

    def show_status(self, text: str) -> None:
        self._busy_timer.stop()
        self._label.setText(text)
        self._place()

    def show_busy(self, text: str, accent: str = "#60a5fa") -> None:
        self._busy_text = text
        self._accent = accent
        self._phase = 0
        self._label.setText(self._busy_html(1.0))
        self._busy_timer.start()
        self._place()

    def _busy_tick(self) -> None:
        self._phase = (self._phase + 1) % len(self._pulse_levels)
        self._label.setText(self._busy_html(self._pulse_levels[self._phase]))

    def dismiss(self) -> None:
        self._busy_timer.stop()
        self.hide()
