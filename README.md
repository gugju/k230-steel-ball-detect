# K230 Steel Ball Detection System

Real-time steel ball (钢珠) detection and tracking system running on Canaan K230 development boards with CanMV firmware. Designed for robotic ball-chasing applications — the K230 detects balls via YOLOv8, displays results on LCD, streams video over WiFi, and sends coordinates to an MCU (MSPM0/STM32) via UART for closed-loop motor control.

## 中文技术文档

面向使用者的中文文档在 [docs/](docs/)：

| 文档 | 内容 |
|------|------|
| [docs/01-硬件选型.md](docs/01-硬件选型.md) | 两种相机（亚博 / 庐山派）对照、选脚本、接线 |
| [docs/02-图传方案.md](docs/02-图传方案.md) | 两种图传（AP+MJPEG / RTSP）对照与选用 |
| [docs/03-部署与运行.md](docs/03-部署与运行.md) | 固件、模型、SD 卡、运行与排错 |
| [docs/04-串口协议与MCU对接.md](docs/04-串口协议与MCU对接.md) | 11 字节坐标帧、与 MCU 联调 |
| [docs/05-版本与调参.md](docs/05-版本与调参.md) | 版本历史、CONFIG 速查、现场调参 |

MCU 固件侧见 [ball-balance-1350](https://github.com/gugju/ball-balance-1350)。

## Supported Hardware

| Board | Variant Suffix | Camera | Display | UART | Notes |
|-------|---------------|--------|---------|------|-------|
| Yahboom K230 CanMV v1.8 | *(no suffix)* | GC2093, default CSI | ST7701 640×480 | UART1 GPIO9(TX)/10(RX) | Original development board |
| LCKFB 庐山派 K230 (LushanPi) | `_lsp` | GC2093, CSI2 `Sensor(id=2)` | ST7701 800×480 | UART2 GPIO11(TX)/12(RX) | 立创开发板, 40Pin RPi-compatible |

## File Naming Convention

```
gangqiujiance_ap_mjpeg_v{N}.py       → Yahboom board, WiFi AP + MJPEG streaming
gangqiujiance_ap_mjpeg_v{N}_lsp.py   → 庐山派 board, WiFi AP + MJPEG streaming
gangqiujiance_v{N}_rtsp.py           → Yahboom board, RTSP streaming
gangqiujiance_v{N}_rtsp_lsp.py       → 庐山派 board, RTSP streaming
```

## Version History

| Version | Key Changes |
|---------|-------------|
| v5 | Base architecture: PipeLine + AIBase + aidemo, WiFi AP MJPEG streaming, touch exposure control, 3-channel sensor (display/MJPEG/AI), square 480×480 crop view, binary UART protocol |
| v6 | URL display moved to top-center with black background + white text (fix green transparency) |
| v7 | Exposure persistence — saves to `/sdcard/exposure.conf`, auto-restores on boot |
| v8 | Sensor fps 30→60 |
| v9 | Added inference time display `AI:xxms` |
| v10 | URL position shifted right |
| v11 | Sensor fps 60→120; `_lsp` variant removes square crop for full-screen 800×480 |
| v12 | Display changed to inference FPS `AI:xxfps xxms N:x` |
| v13 | `_lsp` only: fix garbled screen (800×480 resolution + Display.init before bind_layer) |
| v14 | Frame rate limiter (target_fps=75), Sensor fps matched to 75 |
| v15 | Removed all FPS display |
| v16 | Restored screen refresh rate display only `FPS:xx` |
| v17 | Replaced all `draw_string_advanced` → `draw_string` (fix Deprecated warning on new firmware) |
| v18 | Restored original display `FPS:xx N:x` |
| v19 | Display format `FPS:xx AI:xxms N:x` (v10 style) with draw_string |
| v20 | Simplified to `FPS:xx N:x` + centered URL only |
| v54_rtsp | RTSP streaming variant (replaces MJPEG with H.264 RTSP) |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    K230 CanMV Script                      │
├─────────────────────────────────────────────────────────┤
│  Sensor (3 channels)                                     │
│  ├─ CHN_0: Display feed (YUV420SP → LCD via bind_layer) │
│  ├─ CHN_1: MJPEG/RTSP stream (RGB565 → WiFi)           │
│  └─ CHN_2: AI inference (RGB888P → ai2d → NPU)         │
├─────────────────────────────────────────────────────────┤
│  AI Pipeline                                             │
│  ├─ ai2d: letterbox pad(128) + bilinear resize → 224×224│
│  ├─ nncase_runtime: INT8 kmodel inference (~12ms)       │
│  └─ aidemo.yolov8_det_postprocess: NMS + coord mapping  │
├─────────────────────────────────────────────────────────┤
│  Output                                                  │
│  ├─ LCD OSD: detection boxes + FPS + URL                │
│  ├─ WiFi: MJPEG HTTP stream (AP mode, port 80)          │
│  └─ UART: binary frame to MCU                           │
└─────────────────────────────────────────────────────────┘
```

## UART Protocol

Binary frame sent at ~33Hz (every 30ms):

```
Header:  FA FB FC FD FE
Payload: ",1," + capture_x (1 byte, 0-223) + "\r\n"
No ball: (nothing sent)
```

`capture_x` is the ball center X coordinate in the AI input space (0–223 for 224×224 model). The MCU uses this to compute steering error relative to center (112).

## WiFi Streaming

The K230 creates a WiFi Access Point:

| Parameter | Value |
|-----------|-------|
| SSID | `FJPITYBK2301` |
| Password | `znznnb666` |
| IP | `192.168.4.1` (default) |
| Port | `80` |
| URL | `http://192.168.4.1:80/` |

Connect your phone/laptop to the AP, then open the URL in a browser to see the live MJPEG stream (320×180, ~8fps).

## Prerequisites

### Firmware

- **Yahboom**: Standard CanMV K230 firmware (factory default works)
- **庐山派**: `CanMV_K230_LCKFB_*.img` from [wiki.lckfb.com](https://wiki.lckfb.com/zh-hans/lushan-pi-k230/quick-start.html)

### IDE

- **Yahboom**: CanMV IDE (desktop)
- **庐山派**: VS Code + CanMV extension, or CanMV IDE Web

### Model File

Place `best_v21.kmodel` (224×224 input, INT8, single class "gangzhu") on the SD card root:
```
/sdcard/best_v21.kmodel
```

To convert your own model, see the [k230-yolov8-kmodel-convert](#) skill or use:
```bash
# Export ONNX
yolo export model=best.pt format=onnx imgsz=224 opset=12 simplify=True

# Convert with nncase (requires conda env with nncase + nncase-kpu)
python convert_to_kmodel.py  # see scripts in D:\00software\k230\
```

### SD Card Layout

```
/sdcard/
├── best_v21.kmodel          # AI model
├── exposure.conf            # Saved exposure (auto-created)
└── main.py                  # Script (rename for auto-boot)
```

## Deployment

### Quick Start (IDE)

1. Connect K230 via USB (Type-C for 庐山派, micro-USB for Yahboom)
2. Open CanMV IDE / VS Code CanMV extension
3. Connect to board (COM port)
4. Open the desired script version
5. Click Run

### Standalone (Auto-boot)

1. Rename your chosen script to `main.py`
2. Copy to SD card root alongside `best_v21.kmodel`
3. Power on — script runs automatically

**庐山派 power note**: The 3.1" LCD draws ~700mA total. A standard USB-A port may not supply enough current. Use a powered USB hub or the GH1.25 2P auxiliary power connector.

## Wiring (K230 → MCU)

### Yahboom K230

| K230 Pin | Function | MCU Connection |
|----------|----------|----------------|
| GPIO9 | UART1 TX | MCU RX |
| GPIO10 | UART1 RX | MCU TX (optional) |
| GND | Ground | MCU GND |

### 庐山派 K230

| K230 Pin | Function | MCU Connection |
|----------|----------|----------------|
| GPIO11 | UART2 TX (GH1.25 串口2) | MCU RX |
| GPIO12 | UART2 RX (GH1.25 串口2) | MCU TX (optional) |
| GND | Ground | MCU GND |

**CRITICAL**: Signal level is 3.3V. NEVER connect 5V UART signals to K230 GPIO — it will damage the board.

## Configuration Reference

All tunable parameters are in the `CONFIG` dict at the top of each script:

| Key | Default | Description |
|-----|---------|-------------|
| `kmodel_path` | `/sdcard/best_v21.kmodel` | Model file path |
| `model_input_size` | `[224, 224]` | Must match kmodel |
| `rgb888p_size` | `[224, 224]` | AI channel capture size |
| `confidence_threshold` | `0.30` | Detection confidence cutoff |
| `nms_threshold` | `0.40` | NMS IoU threshold |
| `max_boxes_num` | `5` | Max detections per frame |
| `target_fps` | `75` | Frame rate limiter (v14+) |
| `ap_ssid` | `FJPITYBK2301` | WiFi AP name |
| `ap_password` | `znznnb666` | WiFi AP password |
| `mjpeg_size` | `[320, 180]` | Stream resolution |
| `mjpeg_fps` | `8` | Stream frame rate |
| `exposure_initial_us` | `10000` | Initial exposure (µs) |
| `exposure_step_us` | `1000` | Touch adjustment step |
| `exposure_conf_path` | `/sdcard/exposure.conf` | Exposure save file |
| `square_view_size` | `[480, 480]` / `[800, 480]` | Display crop size |
| `square_view_offset` | `[80, 0]` / `[0, 0]` | Display crop offset |
| `display_source_crop` | `[420, 0, 1080, 1080]` | Sensor crop region |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Green flashing screen (庐山派) | PipeLine uses default Sensor (not id=2) | Use `_lsp` version with `Sensor(id=2)` |
| Garbled/corrupted display (庐山派) | Display initialized at 640×480 instead of 800×480 | Use `_lsp` version (display_size=[800,480]) |
| `draw_string_advanced Deprecated` spam | New firmware deprecated the function | Use v17+ (draw_string) or roll back firmware |
| No camera image | Sensor ribbon cable misaligned | Reseat FPC connector (庐山派 cable sits at an angle by design) |
| WiFi stream not loading | Phone not connected to K230 AP | Connect to SSID `FJPITYBK2301`, then browse to `http://192.168.4.1:80/` |
| UART not received by MCU | Wrong pins or baud mismatch | Yahboom=GPIO9/UART1, 庐山派=GPIO11/UART2, both 115200 8N1 |
| Board won't boot | Wrong firmware or bad SD card | 庐山派 needs `LCKFB` firmware; use quality SD card |
| Exposure resets on reboot | Using version < v7 | Upgrade to v7+ for exposure persistence |

## Performance

Measured on Yahboom K230 with GC2093 @ 224×224 model:

| Metric | Value |
|--------|-------|
| NPU inference | ~12ms per frame |
| Achieved FPS | ~70fps (unlimited), 75fps (with limiter) |
| MJPEG stream | 320×180 @ 8fps |
| UART report rate | ~33Hz (30ms interval) |
| Power draw | ~700mA with LCD (庐山派) |

## License

Private project — 电子设计竞赛 (Electronic Design Competition) use.
