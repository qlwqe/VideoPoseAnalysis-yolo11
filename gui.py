import os
import sys
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from analyze import PoseAnalyzer

class WorkoutDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("健身动作检测")
        self.root.geometry("1200x700")
        self.font_config()
        # 初始化变量
        self.video_path = ""
        self.cap = None
        self.detection_running = False
        self.selected_exercise = tk.StringVar(value="左手")
        self.exercise_kpts = {
            "左手": [5, 7, 9], 
            "右手": [6, 8, 10],
            "左腿": [11, 13, 15], 
            "右腿": [12, 14, 16],
            "左侧身体和大腿": [0, 11, 13],
            "右侧身体和大腿": [0, 12, 14],
        }

        # 用户数据
        self.height_cm = tk.DoubleVar(value=175.0)  # 默认身高175cm
        self.weight_kg = tk.DoubleVar(value=70.0)   # 默认体重70kg

        # 存储控件引用的变量
        self.start_button = None
        self.stop_button = None

        # 显示区域尺寸缓存
        self.display_sizes = {
            "original": {"label_size": None, "frame_size": None},
            "processed": {"label_size": None, "frame_size": None}
        }

        # 视频帧缓存
        self.current_frames = {
            "original": None,
            "processed": None
        }

        # 姿势分析器
        self.pose_analyzer = None

        # 创建界面
        self.create_widgets()

        # 绑定窗口大小变化事件
        self.root.bind("<Configure>", self.on_window_resize)

        # 为视频显示区域添加事件绑定
        self.original_label.bind("<Configure>", lambda e: self.on_display_resize(e, "original"))
        self.processed_label.bind("<Configure>", lambda e: self.on_display_resize(e, "processed"))

        # 初始更新显示
        self.update_display()

    def font_config(self):
        # 尝试设置多种中文字体，确保在不同系统上都能正常显示中文
        if sys.platform.startswith('win'):
            default_font = ('SimHei', 10)
        elif sys.platform.startswith('darwin'):
            default_font = ('Heiti TC', 10)
        else:
            default_font = ('WenQuanYi Micro Hei', 10)

        self.root.option_add('*Font', default_font)

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 使用grid布局管理左右两部分
        main_frame.columnconfigure(0, weight=1)  # 左侧控制面板
        main_frame.columnconfigure(1, weight=100)  # 右侧视频显示区

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 视频选择部分
        ttk.Label(control_frame, text="选择视频文件:").pack(anchor=tk.W, pady=5)
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)

        self.video_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.video_path_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="浏览...", command=self.select_video).pack(side=tk.LEFT, padx=5)

        # 用户数据输入
        ttk.Label(control_frame, text="用户数据").pack(anchor=tk.W, pady=10)
        
        user_frame = ttk.Frame(control_frame)
        user_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(user_frame, text="身高(cm):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(user_frame, textvariable=self.height_cm, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(user_frame, text="体重(kg):").grid(row=0, column=2, padx=(10, 0), sticky=tk.W)
        ttk.Entry(user_frame, textvariable=self.weight_kg, width=10).grid(row=0, column=3, sticky=tk.W)

        # 动作类型选择
        ttk.Label(control_frame, text="检测部位:").pack(anchor=tk.W, pady=5)
        exercise_frame = ttk.Frame(control_frame)
        exercise_frame.pack(fill=tk.X, pady=5)

        for exercise in self.exercise_kpts.keys():
            ttk.Radiobutton(
                exercise_frame,
                text=exercise,
                variable=self.selected_exercise,
                value=exercise
            ).pack(anchor=tk.W)

        # 模型选择
        ttk.Label(control_frame, text="选择模型:").pack(anchor=tk.W, pady=5)
        self.model_var = tk.StringVar(value="yolo11x-pose.pt")
        ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=["yolo11x-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt"],
            state="readonly",
            width=25
        ).pack(fill=tk.X, pady=5)

        # 按钮控制区
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=20)

        # 保存按钮引用
        self.start_button = ttk.Button(button_frame, text="开始检测", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="停止检测", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # 状态信息
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(control_frame, textvariable=self.status_var, justify=tk.LEFT).pack(anchor=tk.SW, fill=tk.X, pady=10)

        # 右侧视频显示区
        display_frame = ttk.LabelFrame(main_frame, text="视频显示", padding=10)
        display_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # 使用grid布局管理左右两个视频显示区域
        display_frame.rowconfigure(0, weight=1)
        display_frame.columnconfigure(0, weight=1)  # 左侧原始视频
        display_frame.columnconfigure(1, weight=5)  # 右侧处理后视频

        # 原始视频
        original_frame = ttk.LabelFrame(display_frame, text="原始视频", padding=5)
        original_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.original_label = ttk.Label(original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)

        # 处理后视频
        processed_frame = ttk.LabelFrame(display_frame, text="处理后视频", padding=5)
        processed_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.processed_label = ttk.Label(processed_frame)
        self.processed_label.pack(fill=tk.BOTH, expand=True)

        data_frame = ttk.LabelFrame(control_frame, text="运动数据", padding=10)
        data_frame.pack(fill=tk.X, pady=10, side=tk.BOTTOM, anchor=tk.SW)

        # 初始化数据变量
        self.count_var = tk.StringVar(value="动作计数: 0")
        self.angle_var = tk.StringVar(value="当前角度: --")
        self.power_var = tk.StringVar(value="功率: --")
        self.velocity_var = tk.StringVar(value="速度: --")

        # 显示数据的标签
        ttk.Label(data_frame, textvariable=self.count_var).pack(anchor=tk.W)
        ttk.Label(data_frame, textvariable=self.angle_var).pack(anchor=tk.W)
        ttk.Label(data_frame, textvariable=self.velocity_var).pack(anchor=tk.W)
        ttk.Label(data_frame, textvariable=self.power_var).pack(anchor=tk.W)

    def on_window_resize(self, event):
        # 窗口大小变化时，只在根窗口变化时重置显示
        if event.widget == self.root:
            self.update_display()

    def on_display_resize(self, event, display_type):
        # 视频显示区域大小变化时，更新对应区域的尺寸缓存
        label = self.original_label if display_type == "original" else self.processed_label
        if label.winfo_width() > 1 and label.winfo_height() > 1:
            self.display_sizes[display_type]["label_size"] = (label.winfo_width(), label.winfo_height())
            self.update_display(display_type)

    def select_video(self):
        """选择视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv")]
        )
        if file_path:
            self.video_path = file_path
            self.video_path_var.set(file_path)
            self.status_var.set(f"已选择视频: {os.path.basename(file_path)}")

    def start_detection(self):
        """开始动作检测"""
        if not self.video_path:
            messagebox.showerror("错误", "请先选择视频文件")
            return

        if self.detection_running:
            return

        # 获取用户数据
        height = self.height_cm.get()
        weight = self.weight_kg.get()
        exercise = self.selected_exercise.get()

        # 创建姿势分析器
        self.pose_analyzer = PoseAnalyzer(
            height_cm=height,
            weight_kg=weight,
            exercise_type=exercise
        )

        # 打开视频文件
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return

        # 重置显示尺寸缓存
        self.display_sizes = {
            "original": {"label_size": None, "frame_size": None},
            "processed": {"label_size": None, "frame_size": None}
        }

        # 更新UI状态
        self.detection_running = True
        self.status_var.set(f"正在检测: {exercise}")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # 使用after方法处理下一帧
        self.process_next_frame()

    def process_next_frame(self):
        """处理下一帧"""
        if not self.detection_running or not self.cap:
            return

        success, frame = self.cap.read()
        if success:
            # 获取当前帧号
            frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # 保存原始帧
            original_frame = frame.copy()
            
            # 处理帧
            if self.pose_analyzer:
                processed_frame, data = self.pose_analyzer.process_frame(frame, frame_num, fps)
                
                # 更新运动数据显示
                self.count_var.set(f"动作计数: {data['count']}")
                self.angle_var.set(f"当前角度: {data['angle']:.1f}°")
                self.velocity_var.set(f"速度: {data['velocity']:.2f} m/s")
                self.power_var.set(f"功率: {data['power']:.2f} W")
                
                # 如果关键点缺失，在状态栏显示警告
                if data['keypoints_missing']:
                    self.status_var.set("警告: 关键点缺失！无法计算完整数据")
                else:
                    self.status_var.set(f"正在检测: {self.selected_exercise.get()}")
            
            # 保存处理后的帧
            self.current_frames = {
                "original": original_frame,
                "processed": processed_frame
            }
            
            # 更新显示
            self.update_display()
            
            # 继续处理下一帧
            self.root.after(30, self.process_next_frame)
        else:
            # 视频处理完成
            self.status_var.set("检测完成")
             # 生成报告
            if self.pose_analyzer and self.video_path:
                self.pose_analyzer.generate_reports(self.video_path)
                self.status_var.set("检测完成 - 报告已生成")

            self.stop_detection()

    def update_display(self, display_type=None):
        """更新视频显示"""
        # 如果没有指定显示类型，更新所有显示
        if display_type is None:
            for display_type, frame in self.current_frames.items():
                if frame is not None:
                    label = self.original_label if display_type == "original" else self.processed_label
                    self.display_frame(frame, label, display_type)
        else:
            # 只更新指定类型的显示
            if display_type in self.current_frames and self.current_frames[display_type] is not None:
                label = self.original_label if display_type == "original" else self.processed_label
                self.display_frame(self.current_frames[display_type], label, display_type)

    def display_frame(self, frame, label, display_type):
        """在Tkinter标签中显示OpenCV帧"""
        # 转换BGR为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 获取标签尺寸
        label_width = label.winfo_width()
        label_height = label.winfo_height()

        # 如果标签尺寸有效
        if label_width > 1 and label_height > 1:
            # 获取当前显示类型的缓存尺寸
            current_size = self.display_sizes[display_type]["label_size"]

            # 检查是否需要重新计算缩放
            if current_size != (label_width, label_height):
                # 获取原始帧的宽高比
                height, width = rgb_frame.shape[:2]
                ratio = width / height

                # 计算适合标签的尺寸
                new_width, new_height = self.calculate_fit_size(width, height, label_width, label_height)

                # 缓存新的显示尺寸
                self.display_sizes[display_type] = {
                    "label_size": (label_width, label_height),
                    "frame_size": (new_width, new_height)
                }

                # 调整帧大小
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
            else:
                # 如果尺寸未变化，使用缓存的缩放结果
                new_width, new_height = self.display_sizes[display_type]["frame_size"]
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))

            # 转换为PhotoImage
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)

            # 更新标签
            label.config(image=photo)
            label.image = photo  # 保持引用，防止被垃圾回收

    def calculate_fit_size(self, src_width, src_height, dest_width, dest_height):
        """计算适合目标区域的尺寸，保持宽高比"""
        src_ratio = src_width / src_height
        dest_ratio = dest_width / dest_height

        if src_ratio > dest_ratio:
            # 源图像更宽，以宽度为基准
            new_width = dest_width
            new_height = int(dest_width / src_ratio)
        else:
            # 源图像更高，以高度为基准
            new_height = dest_height
            new_width = int(dest_height * src_ratio)

        return new_width, new_height

    def stop_detection(self):
        """停止动作检测"""
        self.detection_running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        # 重置姿势分析器
        self.pose_analyzer = None

        # 更新UI状态
        self.status_var.set("就绪")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = WorkoutDetectorApp(root)
    root.mainloop()