QSS = """
* { font-family:"Segoe UI","Yu Gothic UI","Meiryo UI",sans-serif; color:#ECECEC; font-size:13px; }
#Root { background:#191919; border:1px solid #343434; border-radius:0px; }
#Header { background:transparent; }
#Logo { font-size:17px; font-weight:700; color:#FFFFFF; letter-spacing:1px; }
#Sidebar { background:#1D1D1D; border:1px solid #303030; border-radius:16px; }
QPushButton#NavBtn { background:transparent; border:none; border-radius:11px; padding:9px 12px; margin:2px 0; color:#B5B5B5; font-size:13.5px; font-weight:600; text-align:left; }
QPushButton#NavBtn:hover { color:#FFFFFF; background:#2A2A2A; }
QPushButton#NavBtn:checked { color:#FFFFFF; background:#D84040; }
QLineEdit#BadgeEdit { background:#141414; border:2px solid #D84040; border-radius:9px; padding:4px 8px; color:#FFFFFF; font-family:"Consolas",monospace; font-size:13px; }
QPushButton:disabled { color:#666666; }
QPushButton#Power { background:#262626; border:1px solid #404040; border-radius:18px; color:#D8D8D8; font-size:16px; }
QPushButton#Power:hover { color:#FFFFFF; border-color:#666666; background:#2E2E2E; }
QPushButton#WinBtn { background:transparent; border:none; border-radius:14px; min-width:30px; max-width:30px; min-height:30px; max-height:30px; color:#B5B5B5; font-size:13px; }
QPushButton#WinBtn:hover { background:#2C2C2C; color:#FFFFFF; }
#Card { background:#202020; border:1px solid #303030; border-radius:16px; }
#CardTitle { font-size:14.5px; font-weight:700; color:#FFFFFF; }
#CtlLabel { font-size:13px; font-weight:600; color:#F2F2F2; }
#CtlDesc { font-size:11.5px; color:#A8A8A8; }
#Muted { color:#A8A8A8; font-size:12px; }
#Badge { background:#262626; border:1px solid #383838; border-radius:9px; padding:3px 9px; color:#FFFFFF; font-family:"Consolas",monospace; font-size:12.5px; }
#Badge:hover { border-color:#D84040; }
QSlider::groove:horizontal { height:6px; background:#343434; border-radius:3px; }
QSlider::sub-page:horizontal { background:#D84040; border-radius:3px; }
QSlider::handle:horizontal { width:18px; height:18px; margin:-6px 0; border-radius:9px; background:#FFFFFF; border:2px solid #555555; }
QSlider::handle:horizontal:hover { border-color:#D84040; }
QLineEdit,QComboBox { background:#161616; border:1px solid #383838; border-radius:10px; padding:8px 11px; color:#F0F0F0; font-size:13px; selection-background-color:#D84040; }
QLineEdit:focus,QComboBox:focus { border:2px solid #D84040; padding:7px 10px; }
QLineEdit:hover,QComboBox:hover { border-color:#505050; }
QComboBox::drop-down { border:none; width:22px; }
QComboBox QAbstractItemView { background:#232323; border:1px solid #3A3A3A; border-radius:10px; padding:4px; selection-background-color:#D84040; outline:none; }
QPushButton#Ghost { background:#262626; border:1px solid #3D3D3D; border-radius:11px; padding:9px 16px; color:#EDEDED; font-size:12.5px; font-weight:600; }
QPushButton#Ghost:hover { border-color:#666666; color:#FFFFFF; background:#2D2D2D; }
QPushButton#Ghost:pressed { background:#1C1C1C; }
QPushButton#Ghost:checked { background:#D84040; color:#FFFFFF; border-color:#D84040; }
#StatusBar { background:#1D1D1D; border:1px solid #303030; border-radius:16px; }
#StatusText { color:#BDBDBD; font-family:"Consolas",monospace; font-size:12px; }
#StatusMode { background:transparent; border:none; padding:4px 6px; color:#FFFFFF; font-size:12px; font-weight:700; }
#SectionLbl { color:#A0A0A0; font-size:11px; font-weight:600; }
QPushButton#PresetNum { background:#262626; border:1px solid #3D3D3D; border-radius:16px; min-width:32px; max-width:32px; min-height:32px; max-height:32px; color:#BDBDBD; font-size:12.5px; font-weight:700; }
QPushButton#PresetNum:hover { color:#FFFFFF; border-color:#666666; }
QPushButton#PresetNum:checked { background:#D84040; color:#FFFFFF; border-color:#D84040; }
QTextEdit { background:#141414; border:1px solid #2E2E2E; border-radius:12px; color:#BDBDBD; padding:10px; font-family:"Consolas",monospace; font-size:11px; }
QScrollBar:vertical { background:transparent; width:10px; margin:2px; }
QScrollBar::handle:vertical { background:#3D3D3D; border-radius:5px; min-height:28px; }
QScrollBar::handle:vertical:hover { background:#555555; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QScrollBar:horizontal { height:0; }
QScrollArea { background:transparent; border:none; }
QScrollArea > QWidget > QWidget { background:transparent; }
#OverlayPreview { background:#141414; border:1px solid #2E2E2E; border-radius:14px; }
#MetricVal { font-size:22px; font-weight:700; color:#FFFFFF; font-family:"Consolas",monospace; }
#MetricLbl { color:#A8A8A8; font-size:11.5px; font-weight:600; }
QPushButton#QuickPower { background:#D84040; border:none; border-radius:14px; color:#FFFFFF; font-size:15px; font-weight:700; padding:10px 26px; }
QPushButton#QuickPower:hover { background:#E85050; }
QPushButton#QuickPower:pressed { background:#B93333; }
QPushButton#Ghost:focus { border-color:#D84040; }
QPushButton#NavBtn:focus { color:#FFFFFF; background:#2A2A2A; }
QPushButton#PresetNum:focus { border-color:#D84040; color:#FFFFFF; }
#PerfGraph { background:#141414; border:1px solid #2E2E2E; border-radius:12px; }
#StateDot { border-radius:5px; background:#7E8499; }
#QuickState { font-size:14px; font-weight:700; color:#FFFFFF; }
QToolTip { background:#2A2A2A; color:#FFFFFF; border:1px solid #484848; border-radius:8px; padding:6px 10px; font-size:12px; }
"""
