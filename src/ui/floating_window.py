"""
FloatingWindow — always-on-top translation overlay.
Drag to position, resize to cover text. Toggle ON to auto-translate.
"""

import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QApplication, QShortcut,
)
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QFontMetrics, QPen,
    QKeySequence, QMouseEvent, QTextLayout,
)

from src.utils.constants import *
from src.engine.capture import ScreenCapture
from src.engine.ocr import OCREngine
from src.engine.translator import TranslationEngine


class _OCRLoader(QThread):
    """Load the OCR model in a background thread so the UI appears instantly."""
    loaded = pyqtSignal(object)

    def run(self):
        engine = OCREngine()
        engine._get_engine()  # triggers model loading
        self.loaded.emit(engine)


_DPI_SCALE_CACHE = None

def _dpi_scale() -> float:
    """Get the DPI scaling factor (physical / logical pixels). Cached."""
    global _DPI_SCALE_CACHE
    if _DPI_SCALE_CACHE is not None:
        return _DPI_SCALE_CACHE
    try:
        from ctypes import windll
        windll.user32.SetProcessDPIAware()
        hdc = windll.user32.GetDC(0)
        dpi = windll.gdi32.GetDeviceCaps(hdc, 88)
        windll.user32.ReleaseDC(0, hdc)
        _DPI_SCALE_CACHE = dpi / 96.0
    except Exception:
        _DPI_SCALE_CACHE = 1.0
    return _DPI_SCALE_CACHE


class ResizeEdge:
    NONE = 0; TOP = 1; BOTTOM = 2; LEFT = 3; RIGHT = 4
    TOP_LEFT = 5; TOP_RIGHT = 6; BOTTOM_LEFT = 7; BOTTOM_RIGHT = 8


class FloatingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._resize_edge = ResizeEdge.NONE
        self._resize_start_geom = QRect()
        self._resize_start_pos = QPoint()
        self._opacity = OPACITY_DEFAULT
        self._border_mode = 0  # 0=black, 1=white; toggled by border btn

        # Engines
        self._capture = ScreenCapture()
        self._translator = TranslationEngine()
        self._ocr = OCREngine()              # placeholder, loaded in background
        self._ocr_ready = False              # gate for translate toggle
        print("[启动] OCR 引擎后台加载中...", flush=True)

        # State
        self._enabled = False           # Master toggle
        self._translating = False       # Actively polling
        self._processing = False        # Mid OCR/translate cycle
        self._last_hash = None          # Hash of last processed image
        self._last_dirty_hash = None    # Hash of last "dirty" capture (no flash)
        self._last_texts = None
        self._last_captured_img = None
        self._overlays = []
        self._last_ocr_results = None    # Cache OCR results for lang switch
        self._last_translated_text = ""  # Full translated text
        self._status_text = ""
        self._capturing = False          # True during clean screenshot
        self._dpi_scale = 1.0           # Physical/logical pixel ratio

        # Timers
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._on_stable)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._tick)

        # Stable-poll counter: require N consecutive unchanged polls
        # before OCR+translate.  Each poll IS a confirmation, so a
        # premature timer fire between polls can never happen.
        try:
            from config import STABLE_POLLS_REQUIRED, POLL_INTERVAL_MS
        except ImportError:
            STABLE_POLLS_REQUIRED = 2
            POLL_INTERVAL_MS = 1500
        self._stable_count = 0
        self._stable_polls_required = STABLE_POLLS_REQUIRED
        self._poll_interval = POLL_INTERVAL_MS

        self._init_window()
        self._build_ui()
        self._init_shortcuts()

        # Kick off background OCR model loading
        self._ocr_loader = _OCRLoader()
        self._ocr_loader.loaded.connect(self._on_ocr_ready)
        self._ocr_loader.start()
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.setToolTip("OCR 引擎加载中，请稍候...")
        self._status_label.setText("OCR 加载中...")
        self._status_label.setStyleSheet("color: #ff9800; font-size: 10px; background: transparent;")

    def _on_ocr_ready(self, engine: OCREngine):
        """Called from background thread when OCR model finishes loading."""
        self._ocr = engine
        self._ocr_ready = True
        self._toggle_btn.setEnabled(True)
        self._toggle_btn.setToolTip("开启翻译 (Ctrl+L)")
        self._status_label.setText("就绪 — 点击 ▶ 开始")
        self._status_label.setStyleSheet("color: #4caf50; font-size: 10px; background: transparent;")
        print("[启动] 准备就绪", flush=True)

    # ── Window ─────────────────────────────────────────────

    def _init_window(self):
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        screen = QApplication.primaryScreen()
        if screen:
            sg = screen.availableGeometry()
            self.setGeometry(
                (sg.width() - DEFAULT_WIDTH) // 2,
                (sg.height() - DEFAULT_HEIGHT) // 2,
                DEFAULT_WIDTH, DEFAULT_HEIGHT,
            )
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        self._apply_opacity()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Floating square buttons ──
        BTN = 34  # square button size

        btn_bar = QWidget(self)
        btn_bar.setFixedHeight(BTN + 10)
        btn_bar.setStyleSheet("background: transparent;")

        bh = QHBoxLayout(btn_bar)
        bh.setContentsMargins(6, 5, 6, 5)
        bh.setSpacing(5)

        btn_style = """
            QPushButton {
                color: #ccc; background: rgba(30,30,30,180);
                border: none; border-radius: 6px;
                font-size: 14px; font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(60,60,60,200);
            }
        """

        # ── Drag handle ──
        self._drag_btn = QPushButton("⋮⋮")
        self._drag_btn.setFixedSize(BTN, BTN)
        self._drag_btn.setCursor(Qt.SizeAllCursor)
        self._drag_btn.setStyleSheet(btn_style)
        self._drag_btn.setToolTip("按住拖动窗口")
        self._drag_btn._drag_start = QPoint()
        self._drag_btn._dragging = False

        def drag_press(e):
            if e.button() == Qt.LeftButton:
                self._drag_btn._drag_start = e.globalPos()
                self._drag_btn._dragging = True
                self._drag_btn.setCursor(Qt.ClosedHandCursor)

        def drag_move(e):
            if self._drag_btn._dragging:
                d = e.globalPos() - self._drag_btn._drag_start
                self.move(self.pos() + d)
                self._drag_btn._drag_start = e.globalPos()

        def drag_release(e):
            if e.button() == Qt.LeftButton and self._drag_btn._dragging:
                self._drag_btn._dragging = False
                self._drag_btn.setCursor(Qt.SizeAllCursor)
                self._on_interaction()

        self._drag_btn.mousePressEvent = drag_press
        self._drag_btn.mouseMoveEvent = drag_move
        self._drag_btn.mouseReleaseEvent = drag_release
        bh.addWidget(self._drag_btn)

        # ── Language popup button ──
        self._lang_names = list(TARGET_LANGUAGES.keys())
        self._lang_codes = list(TARGET_LANGUAGES.values())
        self._lang_idx = 0  # 中文 (简体)
        self._lang_btn = QPushButton("中")
        self._lang_btn.setFixedSize(BTN, BTN)
        self._lang_btn.setStyleSheet(btn_style)
        self._lang_btn.setToolTip("选择目标语言")

        from PyQt5.QtWidgets import QMenu
        self._lang_menu = QMenu(self)
        self._lang_menu.setStyleSheet("""
            QMenu {
                background: #2a2a2a; color: #eee; border: 1px solid #555;
                border-radius: 4px; padding: 4px 0;
            }
            QMenu::item {
                padding: 5px 24px; font-size: 12px;
            }
            QMenu::item:selected {
                background: #0078d4; border-radius: 2px;
            }
        """)
        for i, name in enumerate(self._lang_names):
            action = self._lang_menu.addAction(name)
            action.setData(i)

        def show_lang_menu():
            # Pause all timers while menu is open — menu overlaps content area
            was_polling = self._poll_timer.isActive()
            self._poll_timer.stop()
            self._debounce_timer.stop()
            pos = self._lang_btn.mapToGlobal(
                QPoint(0, self._lang_btn.height() + 2))
            chosen = self._lang_menu.exec_(pos)
            if chosen is not None:
                idx = chosen.data()
                self._lang_idx = idx
                code = self._lang_codes[idx]
                short = {"zh-CN": "中", "zh-TW": "繁", "en": "EN", "ja": "日",
                         "ko": "한", "fr": "FR", "de": "DE", "es": "ES",
                         "pt": "PT", "ru": "RU", "it": "IT", "ar": "عر",
                         "vi": "VI", "th": "ไทย"}.get(code, code[:2])
                self._lang_btn.setText(short)
                self._lang_btn.setToolTip(f"目标语言: {name}")
                if self._enabled and self._translating:
                    self._retranslate()
            # Resume polling if it was running
            if was_polling:
                self._poll_timer.start(self._poll_interval)

        self._lang_btn.clicked.connect(show_lang_menu)
        bh.addWidget(self._lang_btn)

        # ── Border color button ──
        self._border_btn = QPushButton("⬛")
        self._border_btn.setFixedSize(BTN, BTN)
        self._border_btn.setStyleSheet(btn_style)
        self._border_btn.setToolTip("切换边框颜色 (黑/白)")
        self._border_btn.clicked.connect(self._on_toggle_border)
        bh.addWidget(self._border_btn)

        # ── Toggle button ──
        self._toggle_btn = QPushButton("▶")
        self._toggle_btn.setFixedSize(BTN, BTN)
        self._toggle_btn.setStyleSheet(btn_style + """
            QPushButton { background: rgba(0,120,212,200); }
            QPushButton:hover { background: rgba(0,140,240,220); }
        """)
        self._toggle_btn.setToolTip("开启翻译 (Ctrl+L)")
        self._toggle_btn.clicked.connect(self._on_toggle)
        bh.addWidget(self._toggle_btn)

        bh.addStretch()

        # ── Status label (compact, below buttons) ──
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            "color: #aaa; font-size: 10px; background: transparent; padding: 0 4px;"
        )
        bh.addWidget(self._status_label)
        bh.addStretch()

        # ── Close button ──
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(BTN, BTN)
        close_btn.setStyleSheet(btn_style + """
            QPushButton:hover { background: rgba(192,57,43,200); border-color: #c0392b; }
        """)
        close_btn.setToolTip("关闭")
        close_btn.clicked.connect(self.close)
        bh.addWidget(close_btn)

        layout.addWidget(btn_bar)
        layout.addStretch(1)

        self._btn_bar = btn_bar

    def _init_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+L"), self, self._on_toggle)
        QShortcut(QKeySequence("Escape"), self, self._clear_translation)
        QShortcut(QKeySequence("F5"), self, self._force_refresh)

    # ── Toggle ──────────────────────────────────────────────

    def _on_toggle(self):
        if not self._ocr_ready:
            return  # OCR model not loaded yet
        if self._enabled:
            self._disable()
        else:
            self._enable()

    def _enable(self):
        self._enabled = True
        self._toggle_btn.setText("■")
        self._set_toggle_color("rgba(76,175,80,200)")
        self._toggle_btn.setToolTip("关闭翻译 (Ctrl+L)")
        self._status_label.setText("停稳 1 秒后开始...")
        self._status_label.setStyleSheet("color: #aaa; font-size: 10px; background: transparent;")
        self._overlays = []
        self._last_hash = None
        self._last_dirty_hash = None
        self._last_texts = None
        self._last_ocr_results = None
        self._last_captured_img = None
        self._translating = False
        self._processing = False
        self.update()
        print("[主窗口] 翻译已开启，等待窗口稳定...", flush=True)
        self._debounce_timer.start(1000)

    def _disable(self):
        self._enabled = False
        self._translating = False
        self._processing = False
        self._poll_timer.stop()
        self._debounce_timer.stop()
        self._toggle_btn.setText("▶")
        self._set_toggle_color("rgba(0,120,212,200)")
        self._toggle_btn.setToolTip("开启翻译 (Ctrl+L)")
        self._status_label.setText("")
        self._overlays = []
        self._last_hash = None
        self._last_dirty_hash = None
        self._last_texts = None
        self._status_text = ""
        self._last_ocr_results = None
        self._last_captured_img = None
        self.update()

    def _clear_translation(self):
        """Stop translating and clear overlay, but leave toggle ON."""
        if not self._enabled:
            return
        self._poll_timer.stop()
        self._debounce_timer.stop()
        self._stable_count = 0
        self._translating = False
        self._processing = False
        self._overlays = []
        self._last_hash = None
        self._last_dirty_hash = None
        self._last_texts = None
        self._last_ocr_results = None
        self._last_captured_img = None
        self._status_text = ""
        self._status_label.setText("停稳 1 秒后重试...")
        self._status_label.setStyleSheet("color: #aaa; font-size: 10px; background: transparent;")
        self._debounce_timer.start(1000)
        self.update()

    def _force_refresh(self):
        if self._enabled:
            self._processing = False
            self._last_hash = None
            self._last_dirty_hash = None
            self._last_texts = None
            self._overlays = []
            self._start_translating()

    # ── Opacity ────────────────────────────────────────────

    def _on_toggle_border(self):
        """Cycle border color: black → white → black."""
        self._border_mode = 1 if self._border_mode == 0 else 0
        if self._border_mode == 0:
            self._border_btn.setText("⬛")
            self._border_btn.setToolTip("切换边框颜色 (当前: 黑色)")
        else:
            self._border_btn.setText("⬜")
            self._border_btn.setToolTip("切换边框颜色 (当前: 白色)")
        self.update()

    def _apply_opacity(self):
        self.setWindowOpacity(self._opacity / 100.0)

    def _retranslate(self):
        """Re-translate cached OCR results with current language (no re-capture)."""
        if not self._enabled or self._processing:
            return
        if not self._last_ocr_results or self._last_captured_img is None:
            return
        self._processing = True
        self._set_interactive(False)
        try:
            results = self._last_ocr_results
            img = self._last_captured_img
            lang = self._lang_codes[self._lang_idx]
            self._translator.clear_cache()
            texts = [r.original_text for r in results]
            print(f"[翻译] 语言切换，重新翻译 {len(results)} 段 -> {lang}", flush=True)
            try:
                translated_text = self._translator.translate(texts, lang)
            except Exception as e:
                print(f"[翻译] 重新翻译失败: {e}", flush=True)
                return
            self._last_translated_text = translated_text
            self._compute_overlays(translated_text, img)
            self._status_label.setText("⚡ 已翻译")
            self._status_label.setStyleSheet("color: #4caf50; font-size: 10px; background: transparent;")
            self.update()
        finally:
            self._processing = False
            self._set_interactive(True)

    def _set_interactive(self, enabled: bool):
        """Enable/disable all buttons except close during processing."""
        self._drag_btn.setEnabled(enabled)
        self._lang_btn.setEnabled(enabled)
        self._toggle_btn.setEnabled(enabled)

    def _set_toggle_color(self, bg):
        """Update toggle button background, keeping base style intact."""
        self._toggle_btn.setStyleSheet(
            "QPushButton { color: #ccc; background: %s;"
            " border: none; border-radius: 6px;"
            " font-size: 14px; font-weight: bold; }"
            " QPushButton:hover { background: %s; }" % (bg, bg)
        )

    # ── Debounce ────────────────────────────────────────────

    def _on_stable(self):
        """Fires when window hasn't moved/resized for 1 second."""
        if not self._enabled:
            return
        if self._translating:
            return
        self._start_translating()

    def _start_translating(self):
        if not self._enabled:
            return
        self._stable_count = 0
        self._translating = True
        self._processing = False
        self._last_hash = None
        self._last_dirty_hash = None
        self._last_texts = None
        self._last_ocr_results = None
        self._last_captured_img = None
        self._overlays = []
        self._status_label.setText("⟳ 识别中...")
        self._status_label.setStyleSheet("color: #0078d4; font-size: 10px; background: transparent;")
        self.update()
        QTimer.singleShot(50, self._tick)

    # ── Move/Resize → restart debounce ─────────────────────

    def moveEvent(self, event):
        super().moveEvent(event)
        self._on_interaction()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._on_interaction()

    def _on_interaction(self):
        """User is moving/resizing. Clear results, restart debounce."""
        had_overlays = bool(self._translating or self._processing)
        if had_overlays:
            self._poll_timer.stop()
            self._stable_count = 0
            self._translating = False
            self._processing = False
            self._overlays = []
            self._last_hash = None
            self._last_dirty_hash = None
            self._last_texts = None
            self._last_ocr_results = None
            self._last_captured_img = None
            self._status_text = ""
            self.update()
        if self._enabled:
            if had_overlays or not self._translating:
                self._status_label.setText("停稳 1 秒后继续...")
                self._status_label.setStyleSheet("color: #aaa; font-size: 10px; background: transparent;")
            self._debounce_timer.start(1000)

    # ── Pipeline tick ──────────────────────────────────────

    def _tick(self):
        """One cycle: capture → OCR → translate → render."""
        if not self._enabled or not self._translating:
            return
        # Guard: skip if previous cycle hasn't finished
        if self._processing:
            return

        self._set_interactive(False)
        try:
            self._do_tick()
        except Exception as e:
            # Catch ALL errors — never let the app crash
            self._status_text = ""
            self._status_label.setText(f"✗ 出错: {e}")
            self._status_label.setStyleSheet("color: #f44336; font-size: 10px;")
            self._set_toggle_color("rgba(244,67,54,200)")
            self._processing = False
            self._translating = False
            self.update()
            # Retry after delay
            QTimer.singleShot(3000, self._start_translating)
        finally:
            if not self._processing:
                self._set_interactive(True)

    def _do_tick(self):
        """Dirty-check poll: detect change → reset stable counter.

        Each poll IS a confirmation.  Translation only fires after N
        consecutive polls see no change — this guarantees at least one
        full poll cycle of stability.  No timer can fire prematurely
        between checks.
        """
        self._poll_timer.stop()
        self._processing = True
        self._set_toggle_color("rgba(255,152,0,200)")

        # ── Compute capture geometry ──
        geom = self.geometry()
        scale = _dpi_scale()
        self._dpi_scale = scale

        btn_h = self._btn_bar.height()
        x = int(geom.x() * scale)
        y = int((geom.y() + btn_h) * scale)
        w = max(1, int(geom.width() * scale))
        h = max(1, int((geom.height() - btn_h) * scale))

        # ── Dirty capture ──
        dirty_img = self._capture.capture(x, y, w, h)
        if dirty_img is None or dirty_img.size == 0:
            self._processing = False
            self._poll_timer.start(self._poll_interval)
            return

        dirty_hash = self._compute_hash(dirty_img)

        # ── Compare against previous dirty capture ──
        if self._last_dirty_hash is not None:
            if np.sum(self._last_dirty_hash != dirty_hash) <= 3:
                # Unchanged → increment stable counter
                self._stable_count += 1
                req = self._stable_polls_required
                if self._stable_count >= req:
                    # Content confirmed stable — schedule OCR+translate
                    # on the event loop to avoid nesting inside _tick()
                    self._stable_count = 0
                    self._processing = False
                    self._set_toggle_color("rgba(76,175,80,200)")
                    QTimer.singleShot(0, self._on_content_stable)
                    return
                else:
                    # Still waiting for more confirmations
                    self._processing = False
                    self._set_toggle_color("rgba(76,175,80,200)")
                    self._status_label.setText(
                        f"等待稳定... ({self._stable_count}/{req})")
                    self._status_label.setStyleSheet(
                        "color: #aaa; font-size: 10px; background: transparent;")
                    self._poll_timer.start(self._poll_interval)
                    return

        # ── Content changed (or first capture) — reset counter ──
        self._last_dirty_hash = dirty_hash
        self._stable_count = 0
        self._status_label.setText("检测到变化，等待稳定...")
        self._status_label.setStyleSheet(
            "color: #ff9800; font-size: 10px; background: transparent;")
        self._processing = False
        self._poll_timer.start(self._poll_interval)

    def _on_content_stable(self):
        """Content confirmed stable across enough polls → OCR + translate."""
        if not self._enabled or not self._translating:
            return
        if self._processing:
            return

        self._poll_timer.stop()
        self._processing = True
        self._set_interactive(False)

        geom = self.geometry()
        scale = self._dpi_scale
        btn_h = self._btn_bar.height()
        x = int(geom.x() * scale)
        y = int((geom.y() + btn_h) * scale)
        w = max(1, int(geom.width() * scale))
        h = max(1, int((geom.height() - btn_h) * scale))

        try:
            # ── Clean capture (hide overlays) ──
            print(f"[截图] 文字已稳定，执行干净截屏...", flush=True)
            self._capturing = True
            self.repaint()
            try:
                img = self._capture.capture(x, y, w, h)
            except Exception as e:
                self._capturing = False
                self.update()
                raise Exception(f"截取失败: {e}")
            self._capturing = False
            self.update()

            if img is None or img.size == 0:
                print("[截图] 截取为空", flush=True)
                return

            print(f"[截图] 成功: {img.shape}", flush=True)

            # ── Clean hash check ──
            current_hash = self._compute_hash(img)
            if self._last_hash is not None:
                if np.sum(self._last_hash != current_hash) <= 2:
                    return
            self._last_hash = current_hash

            # ── OCR ──
            self._status_label.setText("⟳ 识别文字...")
            self._status_label.setStyleSheet(
                "color: #0078d4; font-size: 10px; background: transparent;")
            self._status_text = ""
            print("[OCR] 开始识别...", flush=True)
            try:
                raw_results = self._ocr.recognize(img)
            except Exception as e:
                raise Exception(f"OCR失败: {e}")
            results = self._ocr.merge_paragraphs(raw_results)

            if not results:
                self._overlays = []
                self._status_text = "未检测到文字"
                self._status_label.setText("⚡ 就绪 (监控中)")
                self._status_label.setStyleSheet(
                    "color: #4caf50; font-size: 10px; background: transparent;")
                self._set_toggle_color("rgba(76,175,80,200)")
                self.update()
                # Capture reference for next dirty compare
                try:
                    ref_img = self._capture.capture(x, y, w, h)
                    self._last_dirty_hash = self._compute_hash(ref_img)
                except Exception:
                    pass
                return

            # ── Text change check ──
            current_texts = {r.original_text for r in results}
            if current_texts == self._last_texts:
                self._set_toggle_color("rgba(76,175,80,200)")
                return
            self._last_texts = current_texts

            # ── Translate ──
            lang = self._lang_codes[self._lang_idx]
            texts = [r.original_text for r in results]
            print(f"[翻译] 目标语言: {lang}, 待翻译: {len(results)} 段", flush=True)
            self._status_label.setText("⟳ 翻译中...")
            self._status_label.setStyleSheet(
                "color: #ff9800; font-size: 10px; background: transparent;")
            try:
                translated_text = self._translator.translate(texts, lang)
            except Exception as e:
                raise Exception(f"翻译失败: {e}")

            # ── Render overlays ──
            self._last_ocr_results = results
            self._last_captured_img = img
            self._last_translated_text = translated_text
            self._compute_overlays(translated_text, img)
            print(f"[渲染] 覆盖层: {len(self._overlays)} 个", flush=True)
            for i, ov in enumerate(self._overlays):
                r = ov['rect']
                font_pt = ov['font'].pointSize()
                print(f"  [{i}] ({r.x()},{r.y()}) {r.width()}x{r.height()} "
                      f"\"{ov['text'][:40]}\" {font_pt}pt",
                      flush=True)

            # ── Done ──
            self._status_text = ""
            self._status_label.setText("⚡ 已翻译")
            self._status_label.setStyleSheet(
                "color: #4caf50; font-size: 10px; background: transparent;")
            self._set_toggle_color("rgba(76,175,80,200)")
            self.update()

            # Capture reference dirty image (with overlays) for next compare
            try:
                ref_img = self._capture.capture(x, y, w, h)
                self._last_dirty_hash = self._compute_hash(ref_img)
            except Exception:
                pass

        except Exception:
            raise
        finally:
            self._processing = False
            self._set_interactive(True)
            self._poll_timer.start(self._poll_interval)

    # ── Overlay computation ─────────────────────────────────

    @staticmethod
    def _layout_lines(text: str, font: 'QFont', max_w: int) -> list:
        """Use QTextLayout for accurate line breaking and measurement.

        Returns list of (line_text, width, height).
        """
        layout = QTextLayout(text, font)
        layout.beginLayout()
        lines = []
        y = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max_w)
            line.setPosition(QPointF(0, y))
            start = line.textStart()
            length = line.textLength()
            line_text = text[start:start + length]
            # naturalTextRect gives accurate rendered bounds
            lw = int(line.naturalTextRect().width())
            lh = int(line.height())
            lines.append((line_text, lw, lh))
            y += lh
        layout.endLayout()
        return lines

    @staticmethod
    def _layout_height(text: str, font: 'QFont', max_w: int) -> float:
        """Get total height of text laid out with QTextLayout."""
        lines = FloatingWindow._layout_lines(text, font, max_w)
        return sum(lh for _, _, lh in lines)

    def _compute_overlays(self, translated_text, img):
        """Flow translated text to fill the content area with consistent sizing."""
        self._overlays = []
        if not translated_text or not translated_text.strip():
            return

        h, w = img.shape[:2]
        scale = getattr(self, '_dpi_scale', 1.0) or 1.0
        lw, lh = int(w / scale), int(h / scale)
        bar_h = self._btn_bar.height()
        margin = 8

        mean_brightness = float(img[:, :, ::-1].mean())
        if mean_brightness < 128:
            text_color = QColor(255, 255, 255)
            bg_color = QColor(0, 0, 0, 170)
        else:
            text_color = QColor(30, 30, 36)
            bg_color = QColor(255, 255, 255, 180)

        text = translated_text.strip()
        avail_w = lw - margin * 2
        avail_h = lh - margin * 2
        text_w = avail_w - 16  # match drawText area (adjusted ±8px)

        # Binary search for largest font where all text fits, capped at 18pt
        MAX_FONT = 18
        lo, hi = MIN_FONT_SIZE, MAX_FONT
        best_pt = MIN_FONT_SIZE

        test_font = QFont(FONT_FAMILY, MAX_FONT)
        if self._layout_height(text, test_font, text_w) <= avail_h:
            best_pt = MAX_FONT
        else:
            while lo <= hi:
                mid = (lo + hi) // 2
                test_font = QFont(FONT_FAMILY, mid)
                if self._layout_height(text, test_font, text_w) <= avail_h:
                    best_pt = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

        font = QFont(FONT_FAMILY, best_pt)
        lines = self._layout_lines(text, font, text_w)

        # Lay out lines, flowing top to bottom
        y = bar_h + margin
        for line_text, line_w, line_h in lines:
            if y + line_h > bar_h + margin + avail_h:
                break
            # Full-width background for uniform look
            self._overlays.append({
                'rect': QRect(margin, y, avail_w, line_h),
                'text': line_text,
                'font': font,
                'text_color': text_color,
                'bg_color': bg_color,
            })
            y += line_h + 2

        print(f"[渲染] {len(self._overlays)} 行, {best_pt}pt, "
              f"布局 {lw}x{lh}", flush=True)




    def _compute_hash(self, img):
        """Compute a perceptual hash (24×24 = 576-bit) for change detection.

        Uses a larger grid than the typical 16×16 pHash so small text
        changes (a single character) flip enough bits to clear the threshold.
        """
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
        return (small > small.mean()).flatten()

    # ── Paint ───────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        bar_h = self._btn_bar.height()

        # ── Overlay text with background ──
        if not self._capturing:
            for ov in self._overlays:
                r = ov['rect']
                painter.setPen(Qt.NoPen)
                painter.setBrush(ov['bg_color'])
                painter.drawRoundedRect(r.adjusted(-2, -1, 2, 1), 3, 3)
                painter.setPen(ov['text_color'])
                painter.setFont(ov['font'])
                painter.drawText(
                    r.adjusted(8, 0, -8, 0),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    ov['text'],
                )

        # ── Content area border (drawn after overlays so it stays on top) ──
        if self._border_mode == 0:
            border_color = QColor(0, 0, 0, 200)     # black
        else:
            border_color = QColor(255, 255, 255, 200)  # white
        painter.setPen(QPen(border_color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(0, bar_h, w - 1, h - bar_h - 1)

        # ── Resize grip (bottom-right) ──
        grip_size = 20
        grip_margin = 4
        grip_color = QColor(128, 128, 128, 200)
        painter.setPen(QPen(grip_color, 3))
        for i in range(3):
            o = i * 7 + grip_margin
            painter.drawLine(
                w - grip_size + o, h - grip_margin,
                w - grip_margin, h - grip_size + o,
            )

        # ── "No text" hint ──
        if self._status_text == "未检测到文字":
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 140))
            sr = QRect(w // 2 - 100, bar_h + (h - bar_h) // 2 - 22, 200, 44)
            painter.drawRoundedRect(sr, 8, 8)
            painter.setPen(QColor(255, 255, 255, 200))
            painter.setFont(QFont(FONT_FAMILY, 12))
            painter.drawText(sr, Qt.AlignCenter, self._status_text)

    # ── Mouse: resize ────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        pos = event.pos()
        if self._is_br_corner(pos):
            self._resize_edge = ResizeEdge.BOTTOM_RIGHT
            self._resize_start_geom = self.geometry()
            self._resize_start_pos = event.globalPos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._resize_edge != ResizeEdge.NONE:
            self._do_resize(event.globalPos())
            return
        if self._is_br_corner(event.pos()):
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self._resize_edge != ResizeEdge.NONE:
                self._resize_edge = ResizeEdge.NONE
                self._on_interaction()

    def _is_br_corner(self, pos: QPoint) -> bool:
        m = 24  # generous hit area for resize corner
        return pos.x() >= self.width() - m and pos.y() >= self.height() - m

    def _do_resize(self, gp: QPoint):
        d = gp - self._resize_start_pos
        g = QRect(self._resize_start_geom)
        e = self._resize_edge
        if e in (ResizeEdge.LEFT, ResizeEdge.TOP_LEFT, ResizeEdge.BOTTOM_LEFT):
            g.setLeft(min(g.left() + d.x(), g.right() - MIN_WIDTH))
        if e in (ResizeEdge.RIGHT, ResizeEdge.TOP_RIGHT, ResizeEdge.BOTTOM_RIGHT):
            g.setRight(max(g.right() + d.x(), g.left() + MIN_WIDTH))
        if e in (ResizeEdge.TOP, ResizeEdge.TOP_LEFT, ResizeEdge.TOP_RIGHT):
            g.setTop(min(g.top() + d.y(), g.bottom() - MIN_HEIGHT))
        if e in (ResizeEdge.BOTTOM, ResizeEdge.BOTTOM_LEFT, ResizeEdge.BOTTOM_RIGHT):
            g.setBottom(max(g.bottom() + d.y(), g.top() + MIN_HEIGHT))
        self.setGeometry(g)

    # ── Cleanup ─────────────────────────────────────────────

    def closeEvent(self, event):
        os._exit(0)
