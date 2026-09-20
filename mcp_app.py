from __future__ import annotations

import sys
import traceback
from pathlib import Path

try:
    from PySide6.QtCore import QThread, Signal, Qt
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
        QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit, QMessageBox, QProgressBar,
        QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QButtonGroup, QAbstractSpinBox
    )
except ModuleNotFoundError as exc:
    raise SystemExit("缺少 PySide6。请运行 START_MCP.cmd（会自动安装依赖），或先运行 INSTALL_DEPS.cmd。") from exc

import pandas as pd
from mcp_engine import (
    RunConfig, read_table, detect_time_column, numeric_candidates,
    suggest_speed_column, suggest_direction_column, suggest_target_direction_column, resolve_time_column, run,
)


class Worker(QThread):
    message = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, config: RunConfig):
        super().__init__()
        self.config = config

    def run(self):
        try:
            result = run(self.config, progress=self.message.emit)
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SegmentedChoice(QWidget):
    """Lightweight pill-style selector used instead of the native Windows combo box."""
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self._items = list(items)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        for i, (text, data) in enumerate(self._items):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setProperty("segmented", True)
            btn.setProperty("choiceData", data)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(31)
            self._group.addButton(btn, i)
            layout.addWidget(btn, 1)
            if i == 0:
                btn.setChecked(True)

    def currentData(self):
        btn = self._group.checkedButton()
        return btn.property("choiceData") if btn is not None else None

    def setCurrentData(self, data):
        for btn in self._group.buttons():
            if btn.property("choiceData") == data:
                btn.setChecked(True)
                return True
        return False




class ToggleOption(QPushButton):
    """Wide pill toggle used instead of a tiny native checkbox."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setProperty("toggleOption", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(31)
        self.setStyleSheet("text-align:left; padding-left:12px;")


def flatten_numeric(widget):
    """Remove native Windows stepper boxes while preserving typing/wheel/keyboard control."""
    widget.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    widget.setMinimumHeight(31)
    return widget


class FilePicker(QWidget):
    selected = Signal(str)
    def __init__(self, button_text: str):
        super().__init__()
        layout = QHBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.edit = QLineEdit(); self.edit.setPlaceholderText("选择 CSV/TXT/XLS/XLSX")
        self.btn = QPushButton(button_text)
        layout.addWidget(self.edit, 1); layout.addWidget(self.btn)
        self.btn.clicked.connect(self.pick)
    def pick(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "数据文件 (*.csv *.txt *.dat *.tsv *.xls *.xlsx *.xlsm);;所有文件 (*.*)")
        if path:
            self.edit.setText(path); self.selected.emit(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCP独立拟合工具 V1.0.59")
        self.resize(1180, 760)
        self.measured_df = None; self.era_df = None; self.worker = None
        root = QWidget(); self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(12, 10, 12, 10)
        main.setSpacing(8)

        self.setStyleSheet("""
            QMainWindow { background:#f5f7fa; }
            QWidget { color:#20252b; font-size:13px; }
            QGroupBox {
                background:#ffffff; border:1px solid #e0e5ea; border-radius:10px;
                margin-top:12px; padding-top:10px; font-weight:600;
            }
            QGroupBox::title { subcontrol-origin: margin; left:12px; padding:0 5px; color:#343a40; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height:31px; padding:1px 11px; background:#fbfcfe;
                border:1px solid #d8dee7; border-radius:9px; selection-background-color:#dce8ff;
            }
            QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { background:#ffffff; border-color:#c5ced9; }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { background:#ffffff; border:1px solid #7da5ef; }
            QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width:0px; height:0px; border:none; background:transparent; }
            QSpinBox::up-arrow, QSpinBox::down-arrow, QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow { image:none; width:0px; height:0px; }
            QComboBox { padding-right:12px; }
            QComboBox::drop-down { border:none; width:0px; }
            QComboBox::down-arrow { image:none; width:0px; height:0px; }
            QComboBox QAbstractItemView {
                background:#ffffff; border:1px solid #d8dde3; border-radius:7px;
                padding:5px; outline:0; selection-background-color:#eaf1ff; selection-color:#20252b;
            }
            QPushButton {
                min-height:30px; padding:2px 11px; background:#ffffff;
                border:1px solid #d3d9df; border-radius:7px;
            }
            QPushButton:hover { background:#f4f7fb; border-color:#b9c4d0; }
            QPushButton[segmented="true"] {
                min-height:31px; padding:1px 14px; background:#f3f5f8;
                border:1px solid transparent; border-radius:8px; color:#5b6470; font-weight:600;
            }
            QPushButton[segmented="true"]:hover { background:#e9edf3; color:#303740; }
            QPushButton[segmented="true"]:checked {
                background:#e8f0ff; border:1px solid #a9c0ef; color:#2459a9; font-weight:700;
            }
            QPushButton[toggleOption="true"] {
                min-height:31px; padding:1px 12px; background:#f4f6f9;
                border:1px solid transparent; border-radius:9px; color:#4f5965; font-weight:600; text-align:left;
            }
            QPushButton[toggleOption="true"]:hover { background:#eceff4; color:#303740; }
            QPushButton[toggleOption="true"]:checked {
                background:#e9f1ff; border:1px solid #aac2f1; color:#2459a9; font-weight:700;
            }
            QTextEdit, QTableWidget { background:#ffffff; border:1px solid #e0e5ea; border-radius:9px; }
        """)

        title_row = QHBoxLayout()
        title = QLabel("MCP 独立拟合工具")
        title.setStyleSheet("font-size:22px;font-weight:700;color:#20252b;")
        subtitle = QLabel("八种MCP方法 · 独立10min / 1h训练")
        subtitle.setStyleSheet("color:#6b737c;")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(subtitle)
        main.addLayout(title_row)

        # 1. 输入与输出
        data_box = QGroupBox("1. 输入与输出")
        grid = QGridLayout(data_box)
        grid.setHorizontalSpacing(10); grid.setVerticalSpacing(8)
        self.measured_picker = FilePicker("选择测风数据")
        self.era_picker = FilePicker("选择ERA数据")
        self.meas_time = QComboBox(); self.meas_speed = QComboBox(); self.meas_dir = QComboBox()
        self.era_time = QComboBox(); self.era_speed = QComboBox(); self.era_dir = QComboBox()
        self.output_edit = QLineEdit(); self.output_edit.setPlaceholderText("默认：测风文件旁边的 MCP独立拟合结果")
        out_btn = QPushButton("选择输出目录"); out_btn.clicked.connect(self.pick_output)
        grid.addWidget(QLabel("测风数据"),0,0); grid.addWidget(self.measured_picker,0,1,1,5)
        grid.addWidget(QLabel("测风时间列"),1,0); grid.addWidget(self.meas_time,1,1)
        grid.addWidget(QLabel("目标风速列"),1,2); grid.addWidget(self.meas_speed,1,3)
        grid.addWidget(QLabel("目标风向列"),1,4); grid.addWidget(self.meas_dir,1,5)
        grid.addWidget(QLabel("ERA数据"),2,0); grid.addWidget(self.era_picker,2,1,1,5)
        grid.addWidget(QLabel("ERA时间列"),3,0); grid.addWidget(self.era_time,3,1)
        grid.addWidget(QLabel("ERA风速列"),3,2); grid.addWidget(self.era_speed,3,3)
        grid.addWidget(QLabel("ERA风向列"),3,4); grid.addWidget(self.era_dir,3,5)
        grid.addWidget(QLabel("输出目录"),4,0); grid.addWidget(self.output_edit,4,1,1,4); grid.addWidget(out_btn,4,5)
        self.measured_picker.selected.connect(self.load_measured)
        self.era_picker.selected.connect(self.load_era)
        self.meas_speed.currentTextChanged.connect(self._sync_target_direction)
        main.addWidget(data_box)

        # 2. 核心拟合设置
        core_box = QGroupBox("2. 核心拟合设置")
        core = QGridLayout(core_box)
        core.setHorizontalSpacing(10); core.setVerticalSpacing(8)
        self.fit_resolution = SegmentedChoice([("10 min", "10min"), ("1 h", "1h")])
        self.fit_resolution.setToolTip("真正控制八种方法的训练与预测分辨率：10min会用原始严格并发10min重新拟合并直接生成10min Fit；1h会先聚合成小时训练集再重新拟合，随后将1h Fit保持到10min输出。两种模式不是同一模型的平均/展开。")
        self.training_scope = SegmentedChoice([("全部有效数据", "all_valid"), ("仅评价年", "evaluation_year")])
        self.training_scope.setToolTip("全部有效数据=测风与ERA所有有效同期重叠；仅评价年=只取评价年内有效同期。")
        self.fit_min = flatten_numeric(QDoubleSpinBox()); self.fit_min.setRange(0,30); self.fit_min.setDecimals(2); self.fit_min.setValue(2.5); self.fit_min.setSuffix(" m/s")
        self.fit_min.setToolTip("仅BSR/LLS/TLS/VR在50/50抽样评价训练半样本执行；正式最终模型保留低风速。")
        self.shift_min = flatten_numeric(QSpinBox()); self.shift_min.setRange(-48,0); self.shift_min.setValue(-12)
        self.shift_max = flatten_numeric(QSpinBox()); self.shift_max.setRange(0,48); self.shift_max.setValue(12)
        self.export10 = ToggleOption("输出完整评价年10min拟合 / 补齐结果"); self.export10.setChecked(True)
        self.train_start = QLineEdit(); self.train_start.setPlaceholderText("留空=所选训练范围起点")
        self.train_end = QLineEdit(); self.train_end.setPlaceholderText("留空=所选训练范围终点")
        core.addWidget(QLabel("拟合分辨率"),0,0); core.addWidget(self.fit_resolution,0,1,1,2)
        core.addWidget(QLabel("训练数据范围"),0,3); core.addWidget(self.training_scope,0,4,1,2)
        core.addWidget(QLabel("抽样训练低风速阈值"),1,0); core.addWidget(self.fit_min,1,1)
        core.addWidget(QLabel("ERA平移搜索(h)"),1,2); core.addWidget(self.shift_min,1,3); core.addWidget(QLabel("至"),1,4); core.addWidget(self.shift_max,1,5)
        core.addWidget(QLabel("训练限制开始"),2,0); core.addWidget(self.train_start,2,1,1,2)
        core.addWidget(QLabel("训练限制结束"),2,3); core.addWidget(self.train_end,2,4,1,2)
        core.addWidget(self.export10,3,0,1,6)
        main.addWidget(core_box)

        # 3. 高级 / 随机参数
        advanced_row = QHBoxLayout()
        random_box = QGroupBox("3A. 稳定性与抽样")
        rp = QGridLayout(random_box); rp.setHorizontalSpacing(10); rp.setVerticalSpacing(8)
        self.ss_seed = flatten_numeric(QSpinBox()); self.ss_seed.setRange(0,2147483647); self.ss_seed.setValue(20260907)
        self.mts_seed = flatten_numeric(QSpinBox()); self.mts_seed.setRange(0,2147483647); self.mts_seed.setValue(20260831)
        self.stochastic_runs = flatten_numeric(QSpinBox()); self.stochastic_runs.setRange(1,50); self.stochastic_runs.setValue(9); self.stochastic_runs.setSuffix(" 次")
        self.stochastic_runs.setToolTip("SpeedSort用于稳定性诊断；MTS按自身随机实现逻辑使用。")
        self.det_sampling_enable = ToggleOption("启用确定性方法 50% / 50% 重复抽样评价")
        self.det_sampling_enable.setChecked(True)
        self.det_sampling_runs = flatten_numeric(QSpinBox()); self.det_sampling_runs.setRange(1,500); self.det_sampling_runs.setValue(50); self.det_sampling_runs.setSuffix(" 次")
        self.det_sampling_mode = QComboBox()
        self.det_sampling_mode.addItem("月内按天抽", "day")
        self.det_sampling_mode.addItem("月内按小时抽", "hour")
        self.det_sampling_enable.toggled.connect(self.det_sampling_runs.setEnabled)
        self.det_sampling_enable.toggled.connect(self.det_sampling_mode.setEnabled)
        rp.addWidget(QLabel("SS主Seed"),0,0); rp.addWidget(self.ss_seed,0,1)
        rp.addWidget(QLabel("MTS主Seed"),1,0); rp.addWidget(self.mts_seed,1,1)
        rp.addWidget(QLabel("SS稳定性 / MTS随机实现"),2,0); rp.addWidget(self.stochastic_runs,2,1)
        rp.addWidget(self.det_sampling_enable,3,0,1,2)
        rp.addWidget(QLabel("抽样方式"),4,0); rp.addWidget(self.det_sampling_mode,4,1)
        rp.addWidget(QLabel("重复次数"),5,0); rp.addWidget(self.det_sampling_runs,5,1)
        advanced_row.addWidget(random_box, 1)

        mts_box = QGroupBox("3B. MTS / WBL")
        mp = QGridLayout(mts_box); mp.setHorizontalSpacing(10); mp.setVerticalSpacing(8)
        self.mts_ma = flatten_numeric(QSpinBox()); self.mts_ma.setRange(0,24); self.mts_ma.setValue(3); self.mts_ma.setSuffix(" h")
        self.mts_tol = flatten_numeric(QDoubleSpinBox()); self.mts_tol.setRange(0.0,5.0); self.mts_tol.setDecimals(2); self.mts_tol.setSingleStep(0.25); self.mts_tol.setValue(1.5); self.mts_tol.setSuffix(" states")
        self.mts_attempts = flatten_numeric(QSpinBox()); self.mts_attempts.setRange(10,5000); self.mts_attempts.setValue(500)
        self.mts_edge_mode = QComboBox()
        self.mts_edge_mode.addItem("WG兼容：右端拒绝整条重抽（默认）", "right_rejection")
        self.mts_edge_mode.addItem("旧版：48候选双端择优（对照）", "legacy_best_of_n")
        self.wbl_mode = QComboBox()
        self.wbl_mode.addItem("Windographer 4.2.25 / Openwind 兼容（当前主线）", "windographer_compat")
        self.wbl_mode.addItem("WBL-Adaptive（增强自适应）", "adaptive")
        mp.addWidget(QLabel("MTS移动平均"),0,0); mp.addWidget(self.mts_ma,0,1)
        mp.addWidget(QLabel("右端容差"),1,0); mp.addWidget(self.mts_tol,1,1)
        mp.addWidget(QLabel("最大重抽"),2,0); mp.addWidget(self.mts_attempts,2,1)
        mp.addWidget(QLabel("边缘模式"),3,0); mp.addWidget(self.mts_edge_mode,3,1)
        mp.addWidget(QLabel("WBL主结果模式"),4,0); mp.addWidget(self.wbl_mode,4,1)
        advanced_row.addWidget(mts_box, 2)
        main.addLayout(advanced_row)

        # 4. 运行与诊断
        action = QHBoxLayout()
        self.run_btn = QPushButton("开始八法拟合")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("font-size:15px;font-weight:700;background:#e8f0ff;color:#2459a9;border:1px solid #a9c0ef;border-radius:9px;")
        self.open_btn = QPushButton("打开输出目录"); self.open_btn.setEnabled(False); self.open_btn.clicked.connect(self.open_output)
        action.addWidget(self.run_btn,1); action.addWidget(self.open_btn)
        self.run_btn.clicked.connect(self.start_run)
        main.addLayout(action)
        self.progress = QProgressBar(); self.progress.setRange(0,0); self.progress.hide(); main.addWidget(self.progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        log_box = QGroupBox("运行日志"); ll = QVBoxLayout(log_box)
        self.log = QTextEdit(); self.log.setReadOnly(True); ll.addWidget(self.log)
        metric_box = QGroupBox("八法拟合诊断"); ml = QVBoxLayout(metric_box)
        self.table = QTableWidget(); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        ml.addWidget(self.table)
        splitter.addWidget(log_box); splitter.addWidget(metric_box)
        splitter.setStretchFactor(0, 2); splitter.setStretchFactor(1, 3)
        splitter.setSizes([420, 700])
        main.addWidget(splitter, 1)
        self.statusBar().showMessage("请选择测风数据和ERA数据")

    def _fill_combo(self, combo, values, preferred=None):
        combo.clear(); combo.addItems([str(v) for v in values])
        if preferred and preferred in values: combo.setCurrentText(preferred)

    def load_measured(self, path):
        try:
            df = read_table(path); self.measured_df = df
            detected_time = detect_time_column(df)
            detected_speed = suggest_speed_column(df)
            detected_dir = suggest_target_direction_column(df, detected_speed)
            self._fill_combo(self.meas_time, list(df.columns), detected_time)
            nums = numeric_candidates(df)
            self._fill_combo(self.meas_speed, nums, detected_speed)
            self._fill_combo(self.meas_dir, nums, detected_dir)
            self.log.append(f"测风数据已读取：{len(df):,} 行，{len(df.columns)} 列；自动时间列={detected_time}；建议目标风速列={detected_speed}；建议目标风向列={detected_dir}")
            if not self.output_edit.text().strip():
                self.output_edit.setText(str(Path(path).resolve().parent / "MCP独立拟合结果"))
        except Exception as e:
            QMessageBox.critical(self,"读取失败",str(e))

    def _sync_target_direction(self, speed_col):
        if self.measured_df is None or not speed_col:
            return
        preferred = suggest_target_direction_column(self.measured_df, speed_col)
        if preferred and self.meas_dir.findText(preferred) >= 0:
            self.meas_dir.setCurrentText(preferred)

    def load_era(self, path):
        try:
            df = read_table(path); self.era_df = df
            detected_time = detect_time_column(df)
            detected_speed = suggest_speed_column(df)
            detected_dir = suggest_direction_column(df)
            self._fill_combo(self.era_time, list(df.columns), detected_time)
            nums = numeric_candidates(df)
            self._fill_combo(self.era_speed, nums, detected_speed)
            self._fill_combo(self.era_dir, nums, detected_dir)
            self.log.append(f"ERA数据已读取：{len(df):,} 行，{len(df.columns)} 列；自动时间列={detected_time}；ERA风速={detected_speed}；ERA风向={detected_dir}")
        except Exception as e:
            QMessageBox.critical(self,"读取失败",str(e))

    def pick_output(self):
        path = QFileDialog.getExistingDirectory(self,"选择输出目录")
        if path: self.output_edit.setText(path)

    def start_run(self):
        try:
            if not self.measured_picker.edit.text().strip() or not self.era_picker.edit.text().strip():
                raise ValueError("请先选择测风数据和ERA数据。")
            for combo,name in [(self.meas_time,"测风时间列"),(self.meas_speed,"目标风速列"),(self.meas_dir,"目标风向列"),(self.era_time,"ERA时间列"),(self.era_speed,"ERA风速列"),(self.era_dir,"ERA风向列")]:
                if not combo.currentText(): raise ValueError(f"请指定{name}。")
            out = self.output_edit.text().strip()
            if not out: raise ValueError("请指定输出目录。")
            cfg = RunConfig(
                measured_path=self.measured_picker.edit.text().strip(), era_path=self.era_picker.edit.text().strip(),
                measured_time_col=self.meas_time.currentText(), measured_speed_col=self.meas_speed.currentText(), measured_direction_col=self.meas_dir.currentText(),
                era_time_col=self.era_time.currentText(), era_speed_col=self.era_speed.currentText(), era_direction_col=self.era_dir.currentText(),
                output_dir=out, fit_min_speed=float(self.fit_min.value()), fit_resolution=str(self.fit_resolution.currentData()), training_scope=str(self.training_scope.currentData()), shift_min=int(self.shift_min.value()), shift_max=int(self.shift_max.value()),
                export_10min=self.export10.isChecked(),
                concurrent_start=self.train_start.text().strip() or None,
                concurrent_end=self.train_end.text().strip() or None,
                ss_random_seed=int(self.ss_seed.value()),
                wbl_mode=str(self.wbl_mode.currentData()),
                mts_moving_average_hours=int(self.mts_ma.value()),
                mts_random_seed=int(self.mts_seed.value()),
                mts_edge_tolerance_states=float(self.mts_tol.value()),
                mts_edge_max_attempts=int(self.mts_attempts.value()),
                mts_edge_mode=str(self.mts_edge_mode.currentData()),
                deterministic_sampling_enabled=bool(self.det_sampling_enable.isChecked()),
                deterministic_sampling_runs=int(self.det_sampling_runs.value()),
                deterministic_sampling_seed=42,
                deterministic_sampling_mode=str(self.det_sampling_mode.currentData()),
                stochastic_realizations=int(self.stochastic_runs.value()),
            )
            self.run_btn.setEnabled(False); self.progress.show(); self.open_btn.setEnabled(False); self.table.clear(); self.table.setRowCount(0); self.table.setColumnCount(0)
            self.log.append("\n========== 新任务 ==========")
            self.worker = Worker(cfg); self.worker.message.connect(self.on_message); self.worker.finished_ok.connect(self.on_done); self.worker.failed.connect(self.on_failed); self.worker.start()
        except Exception as e:
            QMessageBox.warning(self,"参数不完整",str(e))

    def on_message(self,msg):
        self.log.append(msg); self.statusBar().showMessage(msg)

    def on_done(self,result):
        self.progress.hide(); self.run_btn.setEnabled(True); self.open_btn.setEnabled(True)
        self._last_output = result["output_dir"]
        self.show_metrics(result["metrics"])
        s = result["summary"]
        self.statusBar().showMessage(f"完成：评价年{s['评价年起点'][:10]}~{s['评价年终点'][:10]}，ERA平移{s['最佳ERA时间平移(h)']:+d}h，R²={s['最佳R²']:.4f}")
        QMessageBox.information(self,"计算完成",f"八种MCP方法已全部计算。\n\n输出目录：\n{result['output_dir']}")

    def on_failed(self,detail):
        self.progress.hide(); self.run_btn.setEnabled(True)
        self.log.append(detail)
        last = detail.strip().splitlines()[-1] if detail.strip() else "未知错误"
        self.statusBar().showMessage("计算失败")
        QMessageBox.critical(self,"计算失败",last)

    def show_metrics(self,df: pd.DataFrame):
        self.table.setRowCount(len(df)); self.table.setColumnCount(len(df.columns)); self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for i,row in df.iterrows():
            for j,c in enumerate(df.columns):
                v = row[c]
                if isinstance(v,float): text = "" if pd.isna(v) else f"{v:.5f}"
                else: text = str(v)
                self.table.setItem(i,j,QTableWidgetItem(text))

    def open_output(self):
        import os, subprocess
        path = getattr(self,"_last_output",self.output_edit.text().strip())
        if not path: return
        if sys.platform.startswith("win"): os.startfile(path)
        elif sys.platform == "darwin": subprocess.Popen(["open",path])
        else: subprocess.Popen(["xdg-open",path])


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(); w.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
