# 04 · 串口协议与 MCU 对接

K230 检测到钢球后，把中心 X 坐标经串口发给 MCU（MSPM0/STM32），做闭环控制。
与 [1350 固件工程](https://github.com/gugju/ball-balance-1350) 的 `docs/04-视觉通信协议.md` 对应。

## 帧格式

有球时每 ~30ms 发一帧（约 33Hz）；**无球不发**（MCU 侧靠超时判失效）。

```
字节:   [0][1][2][3][4]  [5] [6]  [7]  [8]           [9][10]
内容:    FA FB FC FD FE   ,   1    ,   capture_x       \r \n
        └────帧头────┘  └有效┘   └─坐标 0–223（单字节）─┘
```

- 帧头固定 `FA FB FC FD FE`（5 字节）。
- `,1,` 为固定分隔（3 字节）。
- `capture_x`：钢球中心 X，AI 输入空间 0–223（224×224 模型）。
- 以 `\r\n` 结束。总长 **11 字节**。

构造代码（各版本一致）：

```python
data = bytes((0xFA, 0xFB, 0xFC, 0xFD, 0xFE)) + b",1," + \
       bytes((best["capture_x"] & 0xFF,)) + b"\r\n"
uart.write(data)
```

## 坐标系

- `capture_x` 是 224×224 输入图上的横坐标，**中心 = 112**。
- 范围 0–223，超界在脚本内钳位。
- 对应摆杆物理坐标：1350 固件把视觉 0–223 映射为 `pos`（**100 = O 点**），
  Q4 分段目标 139≈+5cm / 65≈−5cm。

## MCU 侧解析（1350 为例）

UART3（PB3=RX）中断里流式解析：遇 `0xFA` 同步，收满 11 字节且以 `\n` 结尾则取帧。

```c
if (c == 0xFA) { rx_idx = 1; rx_buf[0] = 0xFA; return; }
if (rx_idx == 0) return;
rx_buf[rx_idx++] = c;
if (rx_idx >= BALL_PKT_LEN) {          /* 11 */
    rx_idx = 0;
    if (c == '\n') {
        ball_rx.valid = rx_buf[6];      /* '1' */
        ball_rx.pos   = rx_buf[8];      /* capture_x */
        ball_new      = 1;
    }
}
```

容错：150ms 无新帧（`BALL_FRAME_TIMEOUT`）判 `Camera:WAIT`，清 PID 积分。

## 接线

| K230 | 亚博 | 庐山派 | MCU (1350) |
|------|------|--------|------------|
| TX | GPIO9 (UART1) | GPIO11 (UART2) | PB3 (UART3 RX) |
| RX（可选） | GPIO10 | GPIO12 | PB2 (UART3 TX) |
| GND | GND | GND | GND |

- 波特率 **115200 8N1**（`uart_baudrate`）。
- 电平 **3.3V**，禁 5V。
- `uart_enabled=False` 可整体关闭串口输出（纯图传模式）。

## 联调自检

1. USB 串口助手接 K230 TX，应见周期性 11 字节帧（有球时）。
2. 对着镜头放球/移走：有球出帧、无球停发。
3. 接 MCU 后 OLED 显示坐标随球移动；`Camera:OK` 后方可开跑。
4. 坐标不对先查两头波特率与 TX/RX 交叉。
