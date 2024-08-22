


## 项目简介

该项目是一个自动瞄准和射击的脚本，主要用验证Android模拟蓝牙鼠标在AI自瞄领域的可行性。它使用计算机视觉技术检测屏幕中目标的位置，并根据用户设置的参数自动调整瞄准和射击。脚本支持自定义配置文件，用户可以根据需要调整瞄准强度、目标调整等参数。



## 功能特性

- **屏幕捕获**: 截取屏幕中心区域的图像用于目标检测。
- **目标检测与分类**: 使用 YOLOv8 模型检测游戏中的敌人并分类。
- **自动瞄准与射击**: 根据检测结果计算瞄准向量并自动发起射击。
- **UDP 通信**: 通过 UDP 协议发送鼠标移动和点击指令，实现远程控制。
- **多线程支持**: 通过线程池管理多线程任务，提高系统响应速度。

## 系统要求

- **操作系统**: Windows
- **编译器**: 支持 C++17 的编译器
- **依赖库**:
    - [OpenCV](https://opencv.org/)
    - [ONNX Runtime](https://onnxruntime.ai/)
    - [Boost Asio](https://www.boost.org/doc/libs/1_76_0/doc/html/boost_asio.html)
    - [CUDA](https://developer.nvidia.com/cuda-toolkit) (可选，用于加速推理)
    - [CMake](https://cmake.org/) (构建工具)

## 安装步骤

1. **克隆项目仓库**:
   ```bash
   git clone https://github.com/yourusername/auto-aim-shoot.git
   cd auto-aim-shoot
   ```

2. **安装依赖**:
   使用包管理器安装以下依赖库：
    - **OpenCV**: 请参考 [OpenCV 官方文档](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html) 进行安装。
    - **ONNX Runtime**: 请参考 [ONNX Runtime 官方文档](https://onnxruntime.ai/docs/build/) 进行安装。
    - **Boost Asio**: 请参考 [Boost 官方文档](https://www.boost.org/doc/libs/1_76_0/more/getting_started/windows.html) 进行安装。

3. **配置与编译**:
   使用 CMake 进行构建：
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

4. **准备模型文件**:
   将预训练的 YOLOv8 模型文件放置在项目目录下的 `models` 文件夹中，并确保路径与 `inference.h` 中的定义一致。

5. **运行程序**:
   在构建目录下运行生成的可执行文件：
   ```bash
   ./auto-aim-shoot
   ```

## 使用说明

1. **配置文件**:
   在根目录下创建 `config.json` 文件，用于配置 UDP 端口、IP 地址等参数。示例如下：
   ```json
   {
       "udp_ip": "192.168.1.100",
       "udp_port": 12345,
       "aim_strength": 1.0,
       "shoot_range": 5
   }
   ```

2. **运行程序**:
   程序启动后将自动开始捕获屏幕并进行目标检测。一旦检测到目标，将自动调整鼠标位置并发起射击。

3. **快捷键**:
    - **PageUp**: 显示 GUI 窗口
    - **PageDown**: 隐藏 GUI 窗口
    - **侧键1**: 启动自动瞄准和射击
    - **侧键2**: 启动仅自动瞄准

4. **调试模式**:
    - **debugAI**: 打开 AI 推理的调试信息输出。
    - **debugCapture**: 打开屏幕捕获的调试信息输出。

## 贡献指南

如果你想为此项目做出贡献，请按照以下步骤操作：

1. **Fork 仓库**: 点击右上角的 `Fork` 按钮将项目仓库 Fork 到你的 GitHub 账户。
2. **创建分支**: 在本地克隆你的 Fork 仓库，并在新分支上进行开发。
   ```bash
   git checkout -b new-feature
   ```
3. **提交更改**: 推送你的更改到你的 Fork 仓库，并创建 Pull Request。
   ```bash
   git push origin new-feature
   ```
4. **创建 Pull Request**: 在 GitHub 上提交 Pull Request，项目维护者会尽快进行代码审查和合并。

## 许可协议

该项目使用 [MIT 许可证](LICENSE) 开源，详情请查阅 LICENSE 文件。

## 注意事项

- 本项目仅供学习和研究使用，请勿用于非法用途或违反游戏协议的行为。
- 使用该脚本可能会导致你的游戏账号被封禁，请谨慎使用。
