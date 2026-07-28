import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QAction, QDragEnterEvent, QDropEvent, QPalette, QColor,
    QBrush, QKeyEvent
)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QCheckBox, QStatusBar,
    QDialog, QDialogButtonBox, QAbstractItemView, QMenu, QInputDialog, QTextEdit
)

from backend import BatchProcessor, normalize_text

# ------------------ Dark Theme ------------------
def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    dark = QPalette()
    dark.setColor(QPalette.Window, QColor(53, 53, 53))
    dark.setColor(QPalette.WindowText, Qt.white)
    dark.setColor(QPalette.Base, QColor(35, 35, 35))
    dark.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark.setColor(QPalette.ToolTipBase, Qt.white)
    dark.setColor(QPalette.ToolTipText, Qt.white)
    dark.setColor(QPalette.Text, Qt.white)
    dark.setColor(QPalette.Button, QColor(53, 53, 53))
    dark.setColor(QPalette.ButtonText, Qt.white)
    dark.setColor(QPalette.BrightText, Qt.red)
    dark.setColor(QPalette.Link, QColor(42, 130, 218))
    dark.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(dark)

# ------------------ Replace Dialog ------------------
class ReplaceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Replace Text in Selected Ads")
        layout = QVBoxLayout(self)
        label = QLabel(
            "Enter replacement texts (one per line).\n"
            "If you provide several lines, each ad will be replaced with a random one."
        )
        layout.addWidget(label)
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Replacement text...")
        self.text_edit.setMinimumHeight(100)
        layout.addWidget(self.text_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def replacements(self) -> List[str]:
        return [line.strip() for line in self.text_edit.toPlainText().splitlines() if line.strip()]


# ------------------ Main Window ------------------
class AdManagerWindow(QMainWindow):
    HIGHLIGHT_COLOR = QColor(255, 165, 0, 80)
    DEFAULT_BG = QColor(0, 0, 0, 0)

    def __init__(self):
        super().__init__()
        self.aggressive_cb = None
        self.file_edit = None
        self.files_lbl = None
        self.blocks_lbl = None
        self.ads_lbl = None
        self.selected_lbl = None
        self.table = None
        self.status_bar = None
        self.processor = BatchProcessor()
        self.candidate_map: Dict[int, Tuple[str, int]] = {}
        self.candidate_map = {}
        self.model = None
        self.threshold = 0.5
        self._load_ml_model()
        self.init_ui()

    def _load_ml_model(self):
        model_path = Path("model/ad_classifier.pkl")
        threshold_path = Path("model/threshold.txt")
        if model_path.exists():
            try:
                self.model = joblib.load(str(model_path))
                if threshold_path.exists():
                    self.threshold = float(threshold_path.read_text().strip())
                else:
                    self.threshold = 0.5
                print(f"ML model loaded. Threshold: {self.threshold:.4f}")
            except Exception as e:
                QMessageBox.warning(self, "ML model load error", f"Could not load model: {e}")
                self.model = None
                self.threshold = 0.5

    def init_ui(self):
        self.setWindowTitle("Subtitle Ad Manager")
        self.setMinimumSize(1200, 750)
        apply_dark_theme(QApplication.instance())

        # Menu
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction(QAction("Load Files...", self, shortcut="Ctrl+O", triggered=self.open_files))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, shortcut="Ctrl+Q", triggered=self.close))
        help_menu = menu.addMenu("Help")
        help_menu.addAction(QAction("About", self, triggered=lambda: QMessageBox.about(self, "About", "Batch Subtitle Ad Manager with AI")))

        # Toolbar for options
        toolbar = self.addToolBar("Options")
        self.aggressive_cb = QCheckBox("Aggressive mode (letters only)")
        self.aggressive_cb.setToolTip("Detect ads using only letters, ignoring numbers/symbols.")
        toolbar.addWidget(self.aggressive_cb)

        # ML model toggle
        self.ml_mode_cb = QCheckBox("Use AI Model (ML)")
        self.ml_mode_cb.setChecked(True)
        self.ml_mode_cb.setToolTip("Use trained machine learning model for detection (high accuracy).")
        if self.model is None:
            self.ml_mode_cb.setEnabled(False)
            self.ml_mode_cb.setToolTip("ML model not found. Please train and place in 'model/' folder.")
        toolbar.addWidget(self.ml_mode_cb)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # File row
        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Drag & drop subtitle files, or click Load Files...")
        self.file_edit.setReadOnly(True)
        file_row.addWidget(self.file_edit)
        load_btn = QPushButton("Load Files...")
        load_btn.clicked.connect(self.open_files)
        file_row.addWidget(load_btn)
        clear_btn = QPushButton("🗑 Clear Files")
        clear_btn.setToolTip("Remove all loaded files and reset the table")
        clear_btn.clicked.connect(self.clear_files)
        file_row.addWidget(clear_btn)
        main_layout.addLayout(file_row)

        # Row 2: Info labels
        info_row = QHBoxLayout()
        self.files_lbl = QLabel("Files: 0")
        self.blocks_lbl = QLabel("Blocks: 0")
        self.ads_lbl = QLabel("Ads: 0")
        self.selected_lbl = QLabel("Selected: 0")
        info_row.addWidget(self.files_lbl)
        info_row.addWidget(self.blocks_lbl)
        info_row.addWidget(self.ads_lbl)
        info_row.addWidget(self.selected_lbl)
        info_row.addStretch()
        main_layout.addLayout(info_row)

        # Row 3: Action buttons
        action_row = QHBoxLayout()
        find_btn = QPushButton("🔍 Find Ads in All Files")
        find_btn.clicked.connect(self.find_ads)
        action_row.addWidget(find_btn)
        action_row.addWidget(QPushButton("☑ Select All", clicked=self.select_all))
        action_row.addWidget(QPushButton("☐ Deselect All", clicked=self.deselect_all))
        action_row.addWidget(QPushButton("☑ Toggle Highlighted", clicked=self.toggle_highlighted_selection))
        action_row.addWidget(QPushButton("↔ Invert All", clicked=self.invert_selection))
        action_row.addStretch()
        main_layout.addLayout(action_row)

        # Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Ad?", "File", "Index", "Time", "Text"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemChanged.connect(self.on_item_changed)
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.on_context_menu)
        main_layout.addWidget(self.table, stretch=1)

        # Row 4: Bottom actions
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QPushButton("🗑 Remove Checked", clicked=self.remove_checked))
        bottom_row.addWidget(QPushButton("✏ Replace Checked...", clicked=self.replace_checked))
        bottom_row.addWidget(QPushButton("📋 Export List", clicked=self.export_list))
        bottom_row.addWidget(QPushButton("💾 Save All Clean Files", clicked=self.save_all))
        bottom_row.addStretch()
        main_layout.addLayout(bottom_row)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.setAcceptDrops(True)

    # ----- Keyboard: Enter toggles checkbox of current row -----
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.table.isActiveWindow() or self.table.hasFocus():
                current_row = self.table.currentRow()
                if current_row >= 0:
                    item = self.table.item(current_row, 0)
                    if item:
                        new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                        item.setCheckState(new_state)
                        return
        super().keyPressEvent(event)

    # ------------------ Drag & Drop ------------------
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            paths = [url.toLocalFile() for url in urls if Path(url.toLocalFile()).is_file()]
            if paths:
                self._load_files(paths)

    # ------------------ File Loading ------------------
    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Load Subtitle Files", "",
                                                "Subtitle Files (*.srt *.vtt *.ass *.ssa);;All Files (*)")
        if files:
            self._load_files(files)

    def _load_files(self, paths: list):
        try:
            total = self.processor.load_files(paths)
            self.file_edit.setText(f"{len(paths)} file(s) loaded")
            self.files_lbl.setText(f"Files: {len(self.processor.files)}")
            self.blocks_lbl.setText(f"Blocks: {total}")
            self.ads_lbl.setText("Ads: 0")
            self.selected_lbl.setText("Selected: 0")
            self.table.setRowCount(0)
            self.candidate_map.clear()
            self.status_bar.showMessage(f"Loaded {len(paths)} files, {total} blocks")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def clear_files(self):
        self.processor.clear()
        self.file_edit.clear()
        self.files_lbl.setText("Files: 0")
        self.blocks_lbl.setText("Blocks: 0")
        self.ads_lbl.setText("Ads: 0")
        self.selected_lbl.setText("Selected: 0")
        self.table.setRowCount(0)
        self.candidate_map.clear()
        self.status_bar.showMessage("All files cleared.")

    # ------------------ Detection & Table ------------------
    def find_ads(self):
        if not self.processor.files:
            QMessageBox.warning(self, "No files", "Load subtitle files first.")
            return
        aggressive = self.aggressive_cb.isChecked()
        use_ml = self.ml_mode_cb.isChecked() and self.model is not None

        ads = []
        if use_ml:
            for fpath, proc in self.processor.processors.items():
                for idx, blk in enumerate(proc.blocks):
                    norm_text = normalize_text(blk.text, aggressive=aggressive)
                    prob = self.model.predict_proba([norm_text])[0][1]
                    if prob >= self.threshold:
                        ads.append((fpath, idx, blk))
        else:
            ads = self.processor.detect_all_ads(aggressive=aggressive)

        self.table.setRowCount(0)
        self.candidate_map.clear()
        self.table.itemChanged.disconnect(self.on_item_changed)
        for row, (fpath, idx, blk) in enumerate(ads):
            self.table.insertRow(row)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked)
            self.table.setItem(row, 0, chk)
            fname = Path(fpath).name
            self.table.setItem(row, 1, QTableWidgetItem(fname))
            self.table.setItem(row, 2, QTableWidgetItem(blk.index))
            self.table.setItem(row, 3, QTableWidgetItem(f"{blk.start} --> {blk.end}"))
            text_item = QTableWidgetItem(blk.text)
            text_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap)
            self.table.setItem(row, 4, text_item)
            self.candidate_map[row] = (fpath, idx)

        self.table.resizeRowsToContents()
        self.table.itemChanged.connect(self.on_item_changed)
        for row in range(self.table.rowCount()):
            self.update_row_background(row, True)
        self.ads_lbl.setText(f"Ads: {len(ads)}")
        mode_str = "ML" if use_ml else "regex"
        self.status_bar.showMessage(f"{len(ads)} potential ads found using {mode_str} detection.")

    # ------------------ Row Highlighting ------------------
    def update_row_background(self, row: int, checked: bool):
        color = self.HIGHLIGHT_COLOR if checked else self.DEFAULT_BG
        brush = QBrush(color)
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item:
                item.setBackground(brush)

    def on_item_changed(self, item: QTableWidgetItem):
        if item.column() == 0:
            row = item.row()
            self.update_row_background(row, item.checkState() == Qt.Checked)
            self.update_selected_count()

    def on_cell_double_clicked(self, row: int, col: int):
        chk_item = self.table.item(row, 0)
        if chk_item:
            current = chk_item.checkState()
            new_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
            chk_item.setCheckState(new_state)

    # ------------------ Context menu: Edit single row text ------------------
    def on_context_menu(self, pos):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        menu = QMenu()
        edit_action = menu.addAction("Edit Text...")
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == edit_action:
            self.edit_row_text(row)

    def edit_row_text(self, row: int):
        if row not in self.candidate_map:
            return
        fpath, idx = self.candidate_map[row]
        current_text = self.table.item(row, 4).text()
        text, ok = QInputDialog.getMultiLineText(self, "Edit Text", "New text:", current_text)
        if ok and text:
            self.processor.replace_text_for_file(fpath, idx, text)
            self.table.item(row, 4).setText(text)
            self.table.resizeRowToContents(row)

    # ------------------ Selection Helpers ------------------
    def _set_all_checked(self, state: Qt.CheckState):
        self.table.itemChanged.disconnect(self.on_item_changed)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setCheckState(state)
        self.table.itemChanged.connect(self.on_item_changed)
        for row in range(self.table.rowCount()):
            self.update_row_background(row, state == Qt.Checked)
        self.update_selected_count()

    def select_all(self):
        self._set_all_checked(Qt.Checked)

    def deselect_all(self):
        self._set_all_checked(Qt.Unchecked)

    def toggle_highlighted_selection(self):
        """تغییر وضعیت چک‌باکس سطرهایی که کاربر با ماوس روی آن‌ها هایلایت کرده است."""
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            return

        rows_to_toggle = set()
        for r in selected_ranges:
            for row in range(r.topRow(), r.bottomRow() + 1):
                rows_to_toggle.add(row)

        self.table.itemChanged.disconnect(self.on_item_changed)
        for row in rows_to_toggle:
            item = self.table.item(row, 0)
            if item:
                current = item.checkState()
                new_state = Qt.Unchecked if current == Qt.Checked else Qt.Checked
                item.setCheckState(new_state)
        self.table.itemChanged.connect(self.on_item_changed)

        for row in rows_to_toggle:
            item = self.table.item(row, 0)
            self.update_row_background(row, item and item.checkState() == Qt.Checked)

        self.update_selected_count()

    def invert_selection(self):
        self.table.itemChanged.disconnect(self.on_item_changed)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                current = item.checkState()
                item.setCheckState(Qt.Unchecked if current == Qt.Checked else Qt.Checked)
        self.table.itemChanged.connect(self.on_item_changed)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            self.update_row_background(row, item and item.checkState() == Qt.Checked)
        self.update_selected_count()

    def update_selected_count(self):
        count = 0
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                count += 1
        self.selected_lbl.setText(f"Selected: {count}")

    def _checked_items(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                if row in self.candidate_map:
                    fpath, idx = self.candidate_map[row]
                    groups.setdefault(fpath, []).append(idx)
        return groups

    # ------------------ Actions ------------------
    def remove_checked(self):
        groups = self._checked_items()
        if not groups:
            QMessageBox.information(self, "Nothing checked", "No ads selected.")
            return
        total = sum(len(v) for v in groups.values())
        if QMessageBox.question(self, "Confirm", f"Remove {total} blocks permanently?") == QMessageBox.Yes:
            for fpath, indices in groups.items():
                self.processor.remove_blocks_for_file(fpath, indices)
            self.find_ads()
            self.blocks_lbl.setText(f"Blocks: {self.processor.total_blocks}")
            self.status_bar.showMessage(f"Removed {total} blocks. Files not saved yet.")

    def replace_checked(self):
        groups = self._checked_items()
        if not groups:
            QMessageBox.information(self, "Nothing checked", "No ads selected.")
            return
        dlg = ReplaceDialog(self)
        if dlg.exec() == QDialog.Accepted:
            replacements = dlg.replacements()
            if not replacements:
                return
            single_text = replacements[0] if len(replacements) == 1 else ""
            for fpath, indices in groups.items():
                for idx in indices:
                    if len(replacements) == 1:
                        self.processor.replace_text_for_file(fpath, idx, single_text)
                    else:
                        new_text = random.choice(replacements)
                        self.processor.replace_text_for_file(fpath, idx, new_text)
            self.find_ads()
            self.status_bar.showMessage(f"Replaced text in {sum(len(v) for v in groups.values())} blocks.")

    def export_list(self):
        groups = self._checked_items()
        if not groups:
            QMessageBox.information(self, "Nothing to export", "No ads checked.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Ad List", "ad_list.txt", "Text Files (*.txt)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                for fpath, indices in groups.items():
                    proc = self.processor.processors.get(fpath)
                    if not proc:
                        continue
                    f.write(f"File: {Path(fpath).name}\n")
                    for idx in indices:
                        blk = proc.blocks[idx]
                        f.write(f"  Index: {blk.index}  Time: {blk.start} --> {blk.end}\n")
                        f.write(f"  Text: {blk.text}\n\n")
                    f.write("\n")
            self.status_bar.showMessage(f"Exported {sum(len(v) for v in groups.values())} ads to {path}")

    def save_all(self):
        if not self.processor.processors:
            QMessageBox.warning(self, "No files", "Nothing to save.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory for clean files")
        if output_dir:
            try:
                saved = self.processor.save_all(output_dir=output_dir, suffix="_clean")
                QMessageBox.information(self, "Saved", f"Saved {len(saved)} file(s) to\n{output_dir}")
                self.status_bar.showMessage("All files saved.")
            except Exception as e:
                QMessageBox.critical(self, "Save error", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdManagerWindow()
    window.show()
    sys.exit(app.exec())