# wind-MCP

## Windows 免 Python 便携版

普通用户无需安装 Python 或 Anaconda：

1. 下载 [Wind-MCP-Windows.zip](https://github.com/yunbeiwang517-dot/wind-MCP/releases/download/windows-portable/Wind-MCP-Windows.zip)
2. 解压整个 ZIP
3. 双击 `Wind-MCP.exe`

> 请保留解压后的整个文件夹，不要只单独复制 EXE。

风资源 MCP（Measure-Correlate-Predict）独立拟合工具。

## Windows 运行

直接双击：

- `启动MCP独立拟合工具.cmd`
- 或 `START_MCP.cmd`

启动脚本会自动寻找 Python 3，并检查依赖；缺少依赖时会根据 `requirements.txt` 自动安装。

## Python 依赖

- numpy >= 1.26
- pandas >= 2.0
- openpyxl >= 3.1
- PySide6 >= 6.7

## 主要文件

- `mcp_app.py`：桌面程序入口
- `mcp_engine.py`：MCP 主流程
- `mcp_algorithms/`：各 MCP 算法实现
- `算法文件SHA256核对.txt`：算法文件哈希核对

## MCP 算法文件

当前源码包含 Adaptive WBL、Weibull Fit、BSR、LLS、MTS、SpeedSort、TLS、VR、VS、WBL 等实现。

## 安装依赖

也可以单独运行：

- `安装依赖.cmd`
- 或 `INSTALL_DEPS.cmd`

> 本仓库为 Windows 本地风资源 MCP 工具源码。
