# ============================================================
# gangqiujiance_v31.py
# 功能：K230 钢珠检测 + 串口发送距离 + 中线显示
# v31：保持 v30 方形显示；AI 检测完整画面，ROI 仅绘制边框
# ============================================================

# ---- K230 官方库 ----
from libs.PipeLine import PipeLine, ScopedTiming  # 摄像头→显示的视频管线
from libs.AIBase import AIBase                     # AI 推理基类
from libs.AI2D import Ai2d                         # 硬件图像预处理（缩放/填充）
from libs.Utils import *                           # 工具函数（画框颜色等）

# ---- K230 运行时库 ----
import nncase_runtime as nn   # NPU 推理引擎
import ulab.numpy as np       # 精简版 numpy
import aidemo                 # YOLO 后处理函数
import time                   # 时间函数
import gc                     # 垃圾回收
import os                     # 系统函数
import sys                    # 系统函数
from media.sensor import (
    Sensor,
    PIXEL_FORMAT_RGB_565,
    PIXEL_FORMAT_RGB_888_PLANAR,
    PIXEL_FORMAT_YUV_SEMIPLANAR_420,
)
from media.media import CAM_CHN_ID_0, CAM_CHN_ID_1, CAM_CHN_ID_2, MediaManager, VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, VENC_CHN_ID_0
import uctypes
from media.display import Display
from media.vencoder import Encoder, ChnAttrStr, StreamData
import network
import _thread
import multimedia as mm
import image

# ---- 串口模块（没有也不影响运行） ----
try:
    from ybUtils.YbUart import YbUart  # Yahboom 串口驱动
except ImportError:
    YbUart = None

try:
    from machine import TOUCH
except ImportError:
    TOUCH = None


# ============================================================
# 全局配置（修改这里改变程序行为）
# ============================================================
CONFIG = {
    "kmodel_path": "/sdcard/best_v21.kmodel",  # AI 模型文件路径
    "labels": ["gangzhu"],                       # 类别名：钢珠
    "model_input_size": [224, 224],              # 模型输入尺寸（宽×高）
    "rgb888p_size": [224, 224],                  # 摄像头采集尺寸
    "display_mode": "lcd",                       # "lcd" 或 "hdmi"
    "confidence_threshold": 0.30,                # 低于 0.30 的结果丢弃
    "nms_threshold": 0.40,                       # 重叠度超 0.40 的框合并
    "max_boxes_num": 5,                          # 钢珠任务只保留最多 5 个目标
    "gc_interval_frames": 30,                   # 每 30 帧回收一次，降低周期性卡顿
    "startup_delay_ms": 1000,                   # 开机自启动时等待系统和 USB 就绪
    "square_view_size": [480, 480],             # LCD 中央显示的正方形视频尺寸
    "square_view_offset": [80, 0],              # 正方形视频在 640x480 LCD 的位置
    "sensor_native_size": [1920, 1080],         # GC2093 当前输出尺寸（见启动日志）
    "display_source_crop": [420, 0, 1080, 1080],# LCD 方形画面对应的原始图像区域
    "uart_enabled": True,                        # 启用串口
    "uart_baudrate": 115200,                     # 波特率
    "wifi_ssid": "Xiaomi 15",
    "wifi_password": "11234567",
    # Keep the proven standalone RTSP geometry/codec path exactly unchanged.
    "rtsp_size": [1280, 720],
    "rtsp_bitrate_kbps": 512,
    "exposure_initial_us": 10000,
    "exposure_step_us": 1000,
    "debug_mode": 0,                             # 0=关闭调试
}


def align_up(value, alignment):
    """向上取整到 alignment 的倍数（K230 硬件要求宽度是16的倍数）"""
    return (value + alignment - 1) // alignment * alignment


class SteelBallDetector(AIBase):
    def __init__(
        self,
        kmodel_path,
        labels,
        model_input_size,
        rgb888p_size,
        display_size,
        detection_display_size,
        detection_display_offset,
        source_image_size,
        display_source_crop,
        confidence_threshold=0.5,
        nms_threshold=0.4,
        max_boxes_num=100,
        debug_mode=0,
    ):
        super().__init__(
            kmodel_path,
            model_input_size,
            rgb888p_size,
            debug_mode,
        )
        self.labels = labels
        self.model_input_size = model_input_size
        self.rgb888p_size = [
            align_up(rgb888p_size[0], 16),
            rgb888p_size[1],
        ]
        self.display_size = [
            align_up(display_size[0], 16),
            display_size[1],
        ]
        self.detection_display_size = detection_display_size
        self.detection_display_offset = detection_display_offset
        self.source_image_size = source_image_size
        self.display_source_crop = display_source_crop
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes_num = max_boxes_num
        self.debug_mode = debug_mode
        self.colors = get_colors(len(labels))

        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8,
        )

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = (
                input_image_size
                if input_image_size
                else self.rgb888p_size
            )
            top, bottom, left, right, self.scale = letterbox_pad_param(
                self.rgb888p_size,
                self.model_input_size,
            )
            self.ai2d.pad(
                [0, 0, 0, 0, top, bottom, left, right],
                0,
                [128, 128, 128],
            )
            self.ai2d.resize(
                nn.interp_method.tf_bilinear,
                nn.interp_mode.half_pixel,
            )
            self.ai2d.build(
                [
                    1,
                    3,
                    ai2d_input_size[1],
                    ai2d_input_size[0],
                ],
                [
                    1,
                    3,
                    self.model_input_size[1],
                    self.model_input_size[0],
                ],
            )

    def preprocess(self, input_np):
        return [nn.from_numpy(input_np)]

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            if not results or len(results) < 1:
                return []

            actual_shape = results[0].shape
            if len(actual_shape) != 3:
                raise ValueError("Unexpected output0 rank: " + str(actual_shape))

            # v2: 自动适配检测模型 [1, 4+Ncls, N] 和分割模型 [1, 4+Ncls+32, N]
            detection_channels = 4 + len(self.labels)
            detection_output = (
                results[0][0][0:detection_channels, :]
                .transpose()
                .copy()
            )

            return aidemo.yolov8_det_postprocess(
                detection_output,
                [self.rgb888p_size[1], self.rgb888p_size[0]],
                [self.model_input_size[1], self.model_input_size[0]],
                [
                    self.source_image_size[1],
                    self.source_image_size[0],
                ],
                len(self.labels),
                self.confidence_threshold,
                self.nms_threshold,
                self.max_boxes_num,
            )

    def get_detections(self, dets):
        detections = []
        if dets is None or len(dets) < 3:
            return detections

        for i in range(len(dets[0])):
            class_id = int(dets[1][i])
            if class_id < 0 or class_id >= len(self.labels):
                continue

            box = dets[0][i]
            # 后处理结果先处在完整摄像头坐标系；仅保留与 LCD 方形画面
            # 重叠的部分，并映射到该画面的显示坐标。
            source_x = int(round(box[0], 0))
            source_y = int(round(box[1], 0))
            source_w = int(round(box[2], 0))
            source_h = int(round(box[3], 0))
            source_center_x = source_x + source_w // 2
            capture_x = int(
                source_center_x * self.rgb888p_size[0]
                / self.source_image_size[0]
            )
            if capture_x < 0:
                capture_x = 0
            elif capture_x >= self.rgb888p_size[0]:
                capture_x = self.rgb888p_size[0] - 1
            crop_x, crop_y, crop_w, crop_h = self.display_source_crop
            left = max(source_x, crop_x)
            top = max(source_y, crop_y)
            right = min(source_x + source_w, crop_x + crop_w)
            bottom = min(source_y + source_h, crop_y + crop_h)
            if right <= left or bottom <= top:
                continue

            view_w, view_h = self.detection_display_size
            view_x, view_y = self.detection_display_offset
            x = view_x + int((left - crop_x) * view_w / crop_w)
            y = view_y + int((top - crop_y) * view_h / crop_h)
            w = int((right - left) * view_w / crop_w)
            h = int((bottom - top) * view_h / crop_h)

            if x < 0:
                w += x
                x = 0
            if y < 0:
                h += y
                y = 0
            if x >= self.display_size[0] or y >= self.display_size[1]:
                continue
            if x + w > self.display_size[0]:
                w = self.display_size[0] - x
            if y + h > self.display_size[1]:
                h = self.display_size[1] - y
            if w <= 1 or h <= 1:
                continue

            detections.append({
                "class_id": class_id,
                "label": self.labels[class_id],
                "score": float(dets[2][i]),
                "box": [x, y, w, h],
                "center": [x + w // 2, y + h // 2],
                "capture_x": capture_x,
                "color": self.colors[class_id],
            })

        return detections

    def draw_detections(self, pl, detections, fps, center_x):
        pl.osd_img.clear()

        # 物理 LCD 是 640x480；视频层仅绑定中央 480x480 的裁切画面。
        # 此处补上两侧黑边，避免底层未使用区域显示残留图像。
        view_x, view_y = self.detection_display_offset
        view_w, view_h = self.detection_display_size
        if view_x > 0:
            pl.osd_img.draw_rectangle(
                0, 0, view_x, self.display_size[1],
                color=(0, 0, 0, 255), thickness=1, fill=True,
            )
        right_x = view_x + view_w
        if right_x < self.display_size[0]:
            pl.osd_img.draw_rectangle(
                right_x,
                0,
                self.display_size[0] - right_x,
                self.display_size[1],
                color=(0, 0, 0, 255), thickness=1, fill=True,
            )

        pl.osd_img.draw_rectangle(
            view_x,
            view_y,
            view_w,
            view_h,
            color=(255, 165, 0, 255),
            thickness=1,
        )

        # 画竖直中线（红色，宽4像素）
        pl.osd_img.draw_rectangle(
            center_x - 2, 0, 4, self.display_size[1],
            color=(255, 0, 0, 255), thickness=2,
        )

        # 左上角: 帧率 + 目标数
        txt = "FPS:" + str(int(fps)) + " N:" + str(len(detections))
        pl.osd_img.draw_string_advanced(
            view_x + 8,
            view_y + 8,
            24,
            txt,
            color=(255, 255, 0, 255),
        )

        for index, detection in enumerate(detections):
            x, y, w, h = detection["box"]
            cx, cy = detection["center"]
            score = detection["score"]
            color = detection["color"]

            pl.osd_img.draw_rectangle(
                x,
                y,
                w,
                h,
                color=color,
                thickness=3,
            )
            pl.osd_img.draw_circle(
                cx,
                cy,
                5,
                color=color,
                thickness=3,
            )

            text_y = y - 26
            if text_y < 34:
                text_y = y + 4

            text = (
                "#"
                + str(index + 1)
                + " "
                + str(round(score, 2))
                + " X:"
                + str(detection["capture_x"])
            )
            pl.osd_img.draw_string_advanced(
                x,
                text_y,
                20,
                text,
                color=color,
            )


def setup_exposure(sensor, config):
    state = {"value": config["exposure_initial_us"], "min": 1, "max": 1000000}
    try:
        exp_range = sensor.get_exposure_time_range()
        if exp_range:
            state["max"], state["min"] = exp_range
        state["value"] = max(state["min"], min(state["max"], state["value"]))
        sensor.exposure(state["value"])
        print("Manual exposure:", state["value"], "us")
    except Exception as error:
        print("Exposure control unavailable:", error)
    return state


def open_touch():
    if TOUCH is None:
        print("Touch unavailable: machine.TOUCH is missing")
        return None
    try:
        return TOUCH(0)
    except Exception as error:
        print("Touch unavailable:", error)
        return None


def handle_exposure_touch(touch, sensor, state, config):
    if touch is None:
        return
    try:
        points = touch.read()
        for point in points:
            if point.event != TOUCH.EVENT_DOWN:
                continue
            if 10 <= point.x < 300 and 420 <= point.y < 475:
                state["value"] = max(state["min"], state["value"] - config["exposure_step_us"])
            elif 340 <= point.x < 630 and 420 <= point.y < 475:
                state["value"] = min(state["max"], state["value"] + config["exposure_step_us"])
            else:
                continue
            sensor.exposure(state["value"])
            print("Exposure:", state["value"], "us")
    except Exception as error:
        print("Touch read failed:", error)


def draw_exposure_controls(pipeline, state):
    osd = pipeline.osd_img
    osd.draw_rectangle(0, 415, 640, 65, color=(0, 0, 0, 220), thickness=1, fill=True)
    osd.draw_rectangle(10, 422, 290, 50, color=(0, 120, 255, 255), thickness=2)
    osd.draw_rectangle(340, 422, 290, 50, color=(0, 120, 255, 255), thickness=2)
    osd.draw_string_advanced(105, 432, 24, "EXP -", color=(255, 255, 255, 255))
    osd.draw_string_advanced(435, 432, 24, "EXP +", color=(255, 255, 255, 255))
    osd.draw_string_advanced(250, 390, 20, "EXP:" + str(int(state["value"])) + "us", color=(255, 255, 0, 255))


def draw_rtsp_url(pipeline, url):
    """在亚博实体屏幕上显示当前 DHCP 分配的 RTSP 地址。"""
    osd = pipeline.osd_img
    osd.draw_rectangle(
        75, 42, 555, 30,
        color=(0, 0, 0, 210), thickness=1, fill=True,
    )
    osd.draw_string_advanced(
        82, 47, 15, url,
        color=(0, 255, 255, 255),
    )


def create_pipeline(config, display_size):
    pipeline = PipeLine(
        rgb888p_size=config["rgb888p_size"],
        display_size=display_size,
        display_mode=config["display_mode"],
    )

    # 不调用旧 PipeLine.create()：它会将原始画面铺满整块 LCD。
    # 通道 0/1 使用 1080x1080 中央裁切来显示方形画面；通道 2 则保留
    # 完整原始画面给 AI 检测，ROI 只在显示阶段作为坐标映射和边框使用。
    nn.shrink_memory_pool()
    native_w, native_h = config["sensor_native_size"]
    # Channel 0 is dedicated to VENC and channel 2 is dedicated to AI.
    # The LCD camera layer is deliberately disabled in this diagnostic build:
    # direct CAM0 -> VENC is kept identical to the known-good RTSP example.
    pipeline.sensor = Sensor()
    pipeline.sensor.reset()
    pipeline.sensor.set_framesize(
        width=config["rtsp_size"][0],
        height=config["rtsp_size"][1],
        alignment=12,
    )
    pipeline.sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    # Physical LCD uses its own square, center-cropped camera channel.
    view_w, view_h = config["square_view_size"]
    view_x, view_y = config["square_view_offset"]
    pipeline.sensor.set_framesize(
        w=view_w,
        h=view_h,
        chn=CAM_CHN_ID_1,
        crop=config["display_source_crop"],
    )
    pipeline.sensor.set_pixformat(
        PIXEL_FORMAT_YUV_SEMIPLANAR_420, chn=CAM_CHN_ID_1
    )
    pipeline.sensor.set_framesize(
        w=config["rgb888p_size"][0],
        h=config["rgb888p_size"][1],
        chn=CAM_CHN_ID_2,
    )
    pipeline.sensor.set_pixformat(
        PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2
    )

    pipeline.osd_img = image.Image(
        display_size[0], display_size[1], image.ARGB8888
    )
    pipeline.display_size = display_size

    # Create the proven CAM0 -> VENC link first.
    pipeline.rtsp = RtspChannel(pipeline.sensor, config)
    pipeline.rtsp.prepare(config)

    # Bind only CAM1 to the physical LCD. IDE image mirroring is disabled;
    # it was the remaining media-buffer difference between v50 and v51.
    lcd_bind = pipeline.sensor.bind_info(
        x=view_x, y=view_y, chn=CAM_CHN_ID_1
    )
    Display.bind_layer(**lcd_bind, layer=Display.LAYER_VIDEO1)
    Display.init(
        Display.ST7701,
        width=display_size[0],
        height=display_size[1],
        osd_num=pipeline.osd_layer_num,
        to_ide=False,
    )
    pipeline.display_size = [Display.width(), Display.height()]
    MediaManager.init()
    pipeline.rtsp.start()
    try:
        pipeline.sensor.auto_exposure(False)
    except Exception as error:
        print("Manual exposure setup failed:", error)
    pipeline.sensor.run()
    pipeline.rtsp.start_thread()
    return pipeline


def create_detector(config, display_size):
    detector = SteelBallDetector(
        kmodel_path=config["kmodel_path"],
        labels=config["labels"],
        model_input_size=config["model_input_size"],
        rgb888p_size=config["rgb888p_size"],
        display_size=display_size,
        detection_display_size=config["square_view_size"],
        detection_display_offset=config["square_view_offset"],
        source_image_size=config["sensor_native_size"],
        display_source_crop=config["display_source_crop"],
        confidence_threshold=config["confidence_threshold"],
        nms_threshold=config["nms_threshold"],
        max_boxes_num=config["max_boxes_num"],
        debug_mode=config["debug_mode"],
    )
    detector.config_preprocess()
    return detector


def open_uart(config):
    if not config["uart_enabled"]:
        return None
    if YbUart is None:
        print("UART disabled: ybUtils.YbUart is unavailable")
        return None

    try:
        return YbUart(baudrate=config["uart_baudrate"])
    except Exception as error:
        print("UART disabled:", error)
        return None


def report_result(detections, uart):
    """
    串口发送格式：@BALL,valid,posh,posl\\r\\n

    ┌──────────────────────────────────────────────────────┐
    │  @BALL,1,0,50\\r\\n                                    │
    │  ─────┬───── ── ──                                   │
    │       │      │  └─ posl: 距离低字节                   │
    │       │      └──── posh: 距离高字节                   │
    │       └────────── valid: 0=没球 1=有球                │
    │                                                      │
    │  距离 = 球心X - 中线X（有符号16位）                   │
    │                                                      │
    │  偏右（球在中线右边）：距离为正                       │
    │    例：距离=50  → posh=0,  posl=50                   │
    │    发送：@BALL,1,0,50\\r\\n                            │
    │                                                      │
    │  偏左（球在中线左边）：距离为负（补码）               │
    │    例：距离=-30 → 补码=65506=0xFFE2                   │
    │    posh=255(0xFF), posl=226(0xE2)                    │
    │    发送：@BALL,1,255,226\\r\\n                         │
    │                                                      │
    │  未检测到球：@BALL,0,0,0\\r\\n                         │
    └──────────────────────────────────────────────────────┘
    """
    if not detections:
        # 二进制帧：40 42 41 4C 4C 00 00 00 0D 0A
        return

    # 取置信度最高的目标
    best = detections[0]
    for det in detections[1:]:
        if det["score"] > best["score"]:
            best = det

    # 帧头固定为 FA FB FC FD FE；坐标范围为 0～223，只发送原始低字节。
    data = bytes((0xFA, 0xFB, 0xFC, 0xFD, 0xFE)) + b",1," + bytes((best["capture_x"] & 0xFF,)) + b"\r\n"
    if uart is not None:
        print("UART TX HEX:", data.hex())
        uart.send(data)


class RtspChannel:
    def __init__(self, sensor, config):
        self.sensor = sensor
        self.w, self.h = config["rtsp_size"]
        self.server = mm.rtsp_server()
        self.encoder = Encoder()
        self.running = False
        self.thread_over = True
        self.link = None

    def prepare(self, config):
        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)
        wlan.connect(config["wifi_ssid"], config["wifi_password"])
        while not wlan.isconnected():
            time.sleep_ms(100)
        self.encoder.SetOutBufs(VENC_CHN_ID_0, 8, self.w, self.h)
        attr = ChnAttrStr(
            self.encoder.PAYLOAD_TYPE_H265,
            self.encoder.H265_PROFILE_MAIN,
            self.w,
            self.h,
            bit_rate=config["rtsp_bitrate_kbps"],
            gopLen=30,
        )
        self.encoder.Create(VENC_CHN_ID_0, attr)
        # Hardware zero-copy link. Do not snapshot channel 0 from Python.
        self.link = MediaManager.link(
            self.sensor.bind_info()["src"],
            (VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, VENC_CHN_ID_0),
        )
        self.server.rtspserver_init(8554)
        self.server.rtspserver_createsession(
            "ball", mm.multi_media_type.media_h265, False
        )
        self.url = "rtsp://%s:8554/ball" % wlan.ifconfig()[0]

    def start(self):
        self.encoder.Start(VENC_CHN_ID_0)
        self.server.rtspserver_start()
        print("RTSP:", self.url)

    def start_thread(self):
        self.running = True
        self.thread_over = False
        _thread.start_new_thread(self._loop, ())

    def _loop(self):
        stream = StreamData()
        try:
            while self.running:
                self.encoder.GetStream(VENC_CHN_ID_0, stream)
                for i in range(stream.pack_cnt):
                    data = bytes(uctypes.bytearray_at(stream.data[i], stream.data_size[i]))
                    self.server.rtspserver_sendvideodata("ball", data, stream.data_size[i], 1000)
                self.encoder.ReleaseStream(VENC_CHN_ID_0, stream)
        except Exception as error:
            print("RTSP stream error:", error)
            import sys
            sys.print_exception(error)
            self.running = False
        finally:
            self.thread_over = True

    def stop(self):
        self.running = False
        wait_start = time.ticks_ms()
        while not self.thread_over and time.ticks_diff(time.ticks_ms(), wait_start) < 2000:
            time.sleep_ms(20)
        try:
            self.encoder.Stop(VENC_CHN_ID_0)
        except Exception:
            pass
        try:
            self.server.rtspserver_stop()
            self.server.rtspserver_deinit()
        except Exception:
            pass
        try:
            if self.link is not None:
                self.link.destroy()
                self.link = None
        except Exception:
            pass
        try:
            self.encoder.Destroy(VENC_CHN_ID_0)
        except Exception:
            pass

def main():
    """主函数：初始化硬件 → 主循环（取图→推理→画框→显示→串口） → 清理退出"""
    print("=== gangqiujiance_v53_rtsp (RTSP URL on Yahboom LCD) ===")

    # 根据显示模式设置屏幕尺寸
    if CONFIG["display_mode"] == "hdmi":
        display_size = [1920, 1080]
    else:
        display_size = [640, 480]

    # 计算中线X坐标（屏幕宽度的一半，用于计算球心到中线的距离）
    center_x = align_up(display_size[0], 16) // 2

    # 初始化变量
    pipeline = None    # 视频管线（摄像头→显示）
    detector = None    # 检测器（AI推理）
    uart = None        # 串口
    rtsp = None
    touch = None
    exposure_state = None
    frame_count = 0                     # 帧计数
    clock = None                         # 显示管线创建后再初始化
    fps_val = 0.0
    running = True  # 主循环标志，设为 False 可退出

    try:
        # 作为 main.py 开机自启动时，不能立刻占用摄像头/显示媒体资源。
        # 预留时间让系统服务、USB 和 IDE 通信接口完成初始化。
        print("[v31] waiting for system:", CONFIG["startup_delay_ms"], "ms")
        time.sleep_ms(CONFIG["startup_delay_ms"])
        os.exitpoint()

        print("[v31] creating pipeline...")
        pipeline = create_pipeline(CONFIG, display_size)
        print("[v31] pipeline ready")
        rtsp = pipeline.rtsp
        exposure_state = setup_exposure(pipeline.sensor, CONFIG)
        touch = open_touch()

        print("[v31] creating detector...")
        detector = create_detector(CONFIG, display_size)
        print("[v31] detector ready")

        uart = open_uart(CONFIG)
        clock = time.clock()

        # 在媒体初始化完成后再启用 IDE 退出点，避免影响启动阶段。
        os.exitpoint(os.EXITPOINT_ENABLE)
        print("Steel-ball detector started (v31)")
        print("Model:", CONFIG["kmodel_path"])

        while running:
            try:
                # 与 main2.py 一致：每帧开始时更新帧率计时器。
                clock.tick()
                os.exitpoint()  # K230 退出检查点，IDE 停止命令通过这里生效

                # 步骤1：获取摄像头画面
                frame = pipeline.get_frame()
                # 步骤2：AI推理（~66ms，最耗时的步骤）
                result = detector.run(frame)
                # 步骤3：解析检测结果（过滤+去重）
                detections = detector.get_detections(result)

                # 步骤4：读取 CanMV 维护的实时帧率。
                fps_val = clock.fps()

                handle_exposure_touch(touch, pipeline.sensor, exposure_state, CONFIG)
                detector.draw_detections(pipeline, detections, fps_val, center_x)
                draw_exposure_controls(pipeline, exposure_state)
                draw_rtsp_url(pipeline, rtsp.url)
                pipeline.show_image()

                # 步骤7：每40ms最多串口上报一次
                report_result(detections, uart)

                # 步骤8：降低垃圾回收频率，减少 FPS 周期性抖动。
                frame_count += 1
                if frame_count % CONFIG["gc_interval_frames"] == 0:
                    gc.collect()

                # 步骤9：让出1ms控制权（让IDE能发送停止命令）
                time.sleep_ms(1)

            # 收到停止信号（IDE点停止 或 Ctrl+C）
            except KeyboardInterrupt:
                print("[v31] 停止信号收到，退出中...")
                running = False  # 退出主循环
            except SystemExit:
                running = False

    # 启动失败或其他异常
    except BaseException as error:
        print("Steel-ball detector error:", error)
        sys.print_exception(error)

    # 无论正常还是异常，都会执行清理（释放硬件资源）
    finally:
        print("[v31] 清理资源...")
        if rtsp is not None:
            rtsp.stop()
        if detector is not None:
            detector.deinit()      # 释放检测器
        if pipeline is not None:
            try:
                pipeline.sensor.stop()
            except Exception:
                pass
            try:
                Display.deinit()
                time.sleep_ms(50)
            except Exception:
                pass
            try:
                MediaManager.deinit()
            except Exception:
                pass
        if uart is not None:
            uart.deinit()          # 关闭串口
        if touch is not None:
            touch.deinit()
        gc.collect()               # 最终垃圾回收
        print("[v31] 已退出")


if __name__ == "__main__":
    main()
