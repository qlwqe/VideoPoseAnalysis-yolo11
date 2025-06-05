import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from jinja2 import Template
import datetime
import matplotlib

# 设置matplotlib使用英文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 12

class PoseAnalyzer:
    def __init__(self, height_cm, weight_kg, exercise_type, save_folder="runs"):
        """
        初始化姿势分析器
        
        参数:
        height_cm: 身高(厘米)
        weight_kg: 体重(公斤)
        exercise_type: 检测部位名称 (如"左手", "右腿"等)
        save_folder: 保存结果的文件夹
        """
        self.height_cm = height_cm
        self.weight_kg = weight_kg
        self.exercise_type = exercise_type
        self.save_folder = save_folder
        
        # 创建保存目录
        self.create_directories()
        
        # 加载YOLO姿势模型
        self.model = YOLO('yolo11x-pose.pt')
        
        # 初始化变量
        self.pixel_height = None
        self.scale_factor = None
        self.prev_keypoint_y = None
        self.prev_time = None
        self.count = 0
        self.current_angle = 0
        self.current_power = 0
        self.current_velocity = 0
        self.current_displacement = 0
        self.max_velocity = 0
        self.max_power = 0
        self.work_done = 0  # 做的功（焦耳）
        
        # 历史数据记录
        self.history = {
            "time": [],
            "angle": [],
            "velocity": [],
            "power": [],
            "displacement": [],
            "action": []
        }
        
        # 存储每次运动的统计数据
        self.workout_stats = []
        
        # 部位对应的关键点索引
        self.exercise_kpts = {
            "左手": [5, 7, 9], 
            "右手": [6, 8, 10],
            "左腿": [11, 13, 15], 
            "右腿": [12, 14, 16],
            "左侧身体和大腿": [0, 11, 13],
            "右侧身体和大腿": [0, 12, 14],
        }
        
        # 当前检测的关键点
        self.keypoint_indices = self.exercise_kpts.get(exercise_type, [5, 7, 9])
        
        # 动作检测状态
        self.state = 0  # 0: 高角度状态, 1: 低角度状态
        self.max_angle = 0
        self.min_angle = 180
        self.start_time = None
        self.velocities = []
        
        # 当前运动统计
        self.current_workout = {
            "start_time": None,
            "min_angle": 180,
            "max_angle": 0,
            "max_power": 0,
            "max_velocity": 0,
        }
        
        # 状态标志
        self.keypoints_missing = False
        
        # 视频输出相关
        self.video_writer = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        
    def create_directories(self):
        """创建保存结果的目录"""
        os.makedirs(os.path.join(self.save_folder, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "videos"), exist_ok=True)
        
    def setup_video_writer(self, frame):
        """设置视频输出"""
        if self.video_writer is None:
            # 获取视频尺寸
            self.frame_height, self.frame_width = frame.shape[:2]
            
            # 创建视频写入器
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.save_folder, "videos", f"output_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, 
                                               (self.frame_width, self.frame_height))
            
    def process_frame(self, frame, frame_num, fps):
        """
        处理视频帧并返回分析结果
        
        参数:
        frame: 视频帧 (numpy数组)
        frame_num: 当前帧号
        fps: 视频帧率
        
        返回:
        processed_frame: 处理后的帧 (带标注)
        data: 包含运动数据的字典
        """
        # 设置帧率（如果尚未设置）
        if self.fps is None:
            self.fps = fps
            
        # 重置关键点缺失状态
        self.keypoints_missing = False
        
        # 使用YOLO模型进行姿势检测
        results = self.model(frame, stream=True)
        
        # 创建帧的副本用于绘制
        processed_frame = frame.copy()
        
        # 处理检测结果
        for result in results:
            if result.keypoints is not None:
                pose_landmarks = result.keypoints.xy.cpu().numpy()
                
                if len(pose_landmarks) > 0:
                    # 获取第一个检测到的人体
                    person_landmarks = pose_landmarks[0]
                    
                    # 检查关键点是否可用
                    if len(person_landmarks) >= 17:
                        # 检查头部和脚部关键点是否存在
                        head_index = 0
                        left_foot_index = 15
                        right_foot_index = 16
                        
                        if (len(person_landmarks) > max(head_index, left_foot_index, right_foot_index) and
                            not np.isnan(person_landmarks[head_index][0]) and 
                            not np.isnan(person_landmarks[left_foot_index][0]) and 
                            not np.isnan(person_landmarks[right_foot_index][0])):
                            
                            # 计算身高像素比例（仅第一次执行）
                            if self.pixel_height is None:
                                head = person_landmarks[head_index]
                                left_foot = person_landmarks[left_foot_index]
                                right_foot = person_landmarks[right_foot_index]
                                foot_y = max(left_foot[1], right_foot[1])
                                self.pixel_height = abs(head[1] - foot_y)
                                self.scale_factor = (self.height_cm / 100) / self.pixel_height if self.pixel_height > 0 else 1
                            
                            # 检测特定部位的动作
                            self._detect_movement(person_landmarks, frame_num, fps, processed_frame)
                        else:
                            # 关键点缺失
                            self.keypoints_missing = True
                        
                        # 绘制关键点
                        for i, lm in enumerate(person_landmarks):
                            if not np.isnan(lm[0]) and not np.isnan(lm[1]):
                                cv2.circle(processed_frame, (int(lm[0]), int(lm[1])), 5, (0, 0, 255), -1)
                    else:
                        self.keypoints_missing = True
                else:
                    self.keypoints_missing = True
            else:
                self.keypoints_missing = True
        
        # 更新历史数据
        self.update_history(frame_num, fps)
        
        # 准备返回数据
        data = {
            'count': self.count,
            'angle': self.current_angle,
            'power': self.current_power,
            'velocity': self.current_velocity,
            'displacement': self.current_displacement,
            'horsepower': self.power_to_horsepower(self.current_power),
            'max_velocity': self.max_velocity,
            'keypoints_missing': self.keypoints_missing
        }
        
        # 设置视频写入器
        self.setup_video_writer(processed_frame)
        
        # 写入视频帧
        if self.video_writer is not None:
            self.video_writer.write(processed_frame)
        
        return processed_frame, data
    
    def update_history(self, frame_num, fps):
        """更新历史数据记录"""
        current_time = frame_num / fps
        self.history["time"].append(current_time)
        self.history["angle"].append(self.current_angle)
        self.history["velocity"].append(self.current_velocity)
        self.history["power"].append(self.current_power)
        self.history["displacement"].append(self.current_displacement)
        
        # 记录动作事件（0无事件，1动作开始，2动作结束）
        action_event = 0
        if self.state == 1:  # 动作进行中
            action_event = 1
        elif self.history["action"] and self.history["action"][-1] == 1:  # 动作刚结束
            action_event = 2
        
        self.history["action"].append(action_event)
    
    def power_to_horsepower(self, power_watts):
        """将功率从瓦特转换为马力"""
        return power_watts / 745.7 if power_watts > 0 else 0
    
    def _detect_movement(self, landmarks, frame_num, fps, frame):
        """
        检测特定部位的运动并计算相关数据
        
        参数:
        landmarks: 人体关键点数组
        frame_num: 当前帧号
        fps: 视频帧率
        frame: 用于绘制的帧
        """
        # 提取当前部位的关键点
        try:
            kpt1, kpt2, kpt3 = [landmarks[i] for i in self.keypoint_indices]
        except (IndexError, TypeError):
            self.keypoints_missing = True
            return
            
        # 计算当前角度
        self.current_angle = self._calculate_angle(kpt1, kpt2, kpt3)
        
        # 使用中间关键点的y坐标进行位移检测
        current_y = kpt2[1]
        current_time = frame_num / fps
        
        # 绘制关键点之间的连线
        cv2.line(frame, (int(kpt1[0]), int(kpt1[1])), 
                 (int(kpt2[0]), int(kpt2[1])), (0, 255, 0), 2)
        cv2.line(frame, (int(kpt2[0]), int(kpt2[1])), 
                 (int(kpt3[0]), int(kpt3[1])), (0, 255, 0), 2)

        
        # 动作检测逻辑（基于角度变化）
        if self.prev_keypoint_y is not None:
            # 计算实际位移（米）
            pixel_displacement = current_y - self.prev_keypoint_y
            self.current_displacement = pixel_displacement * self.scale_factor if self.scale_factor else 0
            
            delta_time = current_time - self.prev_time
            
            # 计算速度（米/秒）
            velocity = self.current_displacement / delta_time if delta_time > 0 else 0
            self.current_velocity = velocity
            
            # 更新最大速度
            if abs(velocity) > abs(self.max_velocity):
                self.max_velocity = velocity
            
            # 更新最大和最小角度
            if self.state == 0:  # 高角度状态
                if self.current_angle > self.max_angle:
                    self.max_angle = self.current_angle
            else:  # 低角度状态
                if self.current_angle < self.min_angle:
                    self.min_angle = self.current_angle
            
            # 状态转换逻辑
            if self.state == 0:  # 高角度状态
                # 如果角度小于90度，进入低角度状态
                if self.current_angle < 90:
                    self.state = 1
                    self.min_angle = self.current_angle
                    
                    # 开始记录当前运动
                    self.current_workout = {
                        "start_time": current_time,
                        "min_angle": self.min_angle,
                        "max_angle": self.max_angle,
                        "max_power": 0,
                        "max_velocity": 0,
                    }
            else:  # 低角度状态
                # 如果角度大于135度，完成一次动作
                if self.current_angle >= 135:
                    self.state = 0
                    self.count += 1
                    
                    # 计算实际位移（使用角度差作为位移指标）
                    angle_displacement = self.max_angle - self.min_angle
                    if self.scale_factor is not None and not self.keypoints_missing:
                        # 使用角度位移估计实际位移
                        real_displacement = angle_displacement * 0.005  # 每度对应0.005米
                        elapsed_time = current_time - self.current_workout["start_time"]
                        
                        # 计算功率
                        force = self.weight_kg * 9.8
                        work_done = force * real_displacement
                        self.work_done += work_done  # 累加做功
                        
                        if elapsed_time > 0:
                            # 平均功率 = 功 / 时间
                            self.current_power = work_done / elapsed_time
                            
                            # 更新最大功率
                            if self.current_power > self.max_power:
                                self.max_power = self.current_power
                    
                    # 记录当前运动数据
                    self.workout_stats.append({
                        "duration": elapsed_time,
                        "displacement": real_displacement,
                        "max_power": self.current_power,
                        "max_velocity": self.max_velocity
                    })
                    
                    # 重置最大最小角度
                    self.max_angle = 0
                    self.min_angle = 180
        
        # 更新前一帧数据
        self.prev_keypoint_y = current_y
        self.prev_time = current_time

    def _calculate_angle(self, a, b, c):
        """
        计算三点之间的角度
        
        参数:
        a, b, c: 三个关键点的坐标
        
        返回:
        angle: 三点之间的角度（度）
        """
        # 计算向量
        ba = np.array([a[0]-b[0], a[1]-b[1]])
        bc = np.array([c[0]-b[0], c[1]-b[1]])
        
        # 计算点积
        dot_product = np.dot(ba, bc)
        
        # 计算模
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        # 避免除以零
        if norm_ba == 0 or norm_bc == 0:
            return 0
        
        # 计算角度（弧度）
        cos_angle = dot_product / (norm_ba * norm_bc)
        # 确保cos_angle在有效范围内
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        # 转换为角度
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def finalize(self):
        """完成处理，释放资源"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
    
    def generate_reports(self, video_path):
        """生成图表和报告"""
        if not self.workout_stats:
            print("没有检测到任何运动，无法生成报告")
            return None
        
        # 生成图表
        chart_paths = self.generate_charts()
        
        # 生成报告
        report_path = self.generate_html_report(chart_paths, video_path)
        
        return report_path

    def generate_charts(self):
        """生成分析图表"""
        chart_paths = []
        
        # 生成每次运动的统计图表（离散数据点）
        if not self.workout_stats:
            return chart_paths
        
        # 提取每次运动的数据
        workout_numbers = range(1, len(self.workout_stats) + 1)
        durations = [s['duration'] for s in self.workout_stats]
        displacements = [s['displacement'] for s in self.workout_stats]
        max_powers = [s['max_power'] for s in self.workout_stats]
        max_velocities = [s['max_velocity'] for s in self.workout_stats]
        
        # 1. 运动时长折线图
        plt.figure(figsize=(10, 6))
        plt.plot(workout_numbers, durations, 'bo-', markersize=8)
        plt.xlabel('Workout Number')
        plt.ylabel('Duration (s)')
        plt.title('Duration per Workout')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(workout_numbers)
        duration_chart = os.path.join(self.save_folder, "charts", "workout_duration.png")
        plt.savefig(duration_chart, bbox_inches='tight')
        plt.close()
        chart_paths.append(duration_chart)
        
        # 2. 位移折线图
        plt.figure(figsize=(10, 6))
        plt.plot(workout_numbers, displacements, 'go-', markersize=8)
        plt.xlabel('Workout Number')
        plt.ylabel('Displacement (m)')
        plt.title('Displacement per Workout')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(workout_numbers)
        displacement_chart = os.path.join(self.save_folder, "charts", "workout_displacement.png")
        plt.savefig(displacement_chart, bbox_inches='tight')
        plt.close()
        chart_paths.append(displacement_chart)
        
        # 3. 最大功率折线图
        plt.figure(figsize=(10, 6))
        plt.plot(workout_numbers, max_powers, 'ro-', markersize=8)
        plt.xlabel('Workout Number')
        plt.ylabel('Max Power (W)')
        plt.title('Max Power per Workout')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(workout_numbers)
        power_chart = os.path.join(self.save_folder, "charts", "workout_power.png")
        plt.savefig(power_chart, bbox_inches='tight')
        plt.close()
        chart_paths.append(power_chart)
        
        return chart_paths

    def generate_html_report(self, chart_paths, video_path):
        """生成HTML报告"""
        # 准备报告数据
        report_data = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "exercise_type": self.exercise_type,
            "user_height": self.height_cm,
            "user_weight": self.weight_kg,
            "total_reps": self.count,
            "max_velocity": self.max_velocity,
            "max_power": self.max_power,
            "max_horsepower": self.power_to_horsepower(self.max_power),
            "total_work": self.work_done,
            "chart_duration": os.path.basename(chart_paths[0]) if len(chart_paths) > 0 else "",
            "chart_displacement": os.path.basename(chart_paths[1]) if len(chart_paths) > 1 else "",
            "chart_power": os.path.basename(chart_paths[2]) if len(chart_paths) > 2 else "",
            "video_name": os.path.basename(video_path)
        }
        
        # 准备每次运动的详细数据
        detailed_data = []
        for i, stats in enumerate(self.workout_stats):
            detailed_data.append({
                "number": i + 1,
                "duration": stats['duration'],
                "displacement": stats['displacement'],
                "max_power": stats['max_power'],
                "max_velocity": stats['max_velocity'],
                "horsepower": self.power_to_horsepower(stats['max_power'])
            })
        
        # HTML报告模板
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>运动检测报告</title>

            <style>
                body { font-family: Arial, sans-serif; margin: 30px; }
                .container { max-width: 1200px; margin: auto; }
                .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
                .section { margin-bottom: 40px; }
                .section-title { font-size: 24px; color: #333; border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-bottom: 20px; }
                .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .stat-item { background-color: #f9f9f9; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .stat-label { font-weight: bold; margin-bottom: 10px; }
                .chart-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .chart-container { background-color: #fff; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .chart-container img { max-width: 100%; height: auto; display: block; }
                .footer { margin-top: 40px; text-align: center; color: #777; font-size: 0.9em; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                th { background-color: #f2f2f2; font-weight: bold; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>

        </head>

        <body>
            <div class="container">
                <div class="header">
                    <h1>运动检测报告</h1>

                    <p>日期: {{ report_data.date }}</p>

                </div>

                
                <div class="section">
                    <h2 class="section-title">用户信息</h2>

                    <p>检测部位: {{ report_data.exercise_type }}</p>

                    <p>身高: {{ report_data.user_height }} cm</p>

                    <p>体重: {{ report_data.user_weight }} kg</p>

                </div>

                
                <div class="section">
                    <h2 class="section-title">运动统计</h2>

                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-label">动作总数</div>
                            <div class="stat-value">{{ report_data.total_reps }} 个</div>

                        </div>

                        <div class="stat-item">
                            <div class="stat-label">最大速度</div>
                            <div class="stat-value">{{ "%.2f"|format(report_data.max_velocity) }} m/s</div>

                        </div>

                        <div class="stat-item">
                            <div class="stat-label">最大功率</div>
                            <div class="stat-value">{{ "%.2f"|format(report_data.max_power) }} W</div>

                        </div>

                        <div class="stat-item">
                            <div class="stat-label">最大马力</div>
                            <div class="stat-value">{{ "%.3f"|format(report_data.max_horsepower) }} HP</div>

                        </div>

                        <div class="stat-item">
                            <div class="stat-label">总功</div>
                            <div class="stat-value">{{ "%.2f"|format(report_data.total_work) }} J</div>
                        </div>
                    </div>
                </div>
                <div class="section">
                    <h2 class="section-title">运动统计图</h2>
                    <div class="chart-row">
                        {% if report_data.chart_duration %}
                        <div class="chart-container">
                            <img src="../charts/{{ report_data.chart_duration }}" alt="Duration Chart">
                        </div>
                        {% endif %}
                        {% if report_data.chart_displacement %}
                        <div class="chart-container">
                            <img src="../charts/{{ report_data.chart_displacement }}" alt="Displacement Chart">
                        </div>

                        {% endif %}
                        {% if report_data.chart_power %}
                        <div class="chart-container">
                            <img src="../charts/{{ report_data.chart_power }}" alt="Power Chart">
                        </div>

                        {% endif %}
                    </div>

                    
                    <h3>详细动作数据</h3>

                    <table>
                        <tr>
                            <th>动作计数 #</th>

                            <th>持续时间 (s)</th>

                            <th>位移 (m)</th>

                            <th>最大功率 (W)</th>

                            <th>最大速度 (m/s)</th>

                            <th>最大马力 (HP)</th>

                        </tr>

                        {% for item in detailed_data %}
                        <tr>
                            <td>{{ item.number }}</td>

                            <td>{{ "%.2f"|format(item.duration) }}</td>

                            <td>{{ "%.2f"|format(item.displacement) }}</td>

                            <td>{{ "%.2f"|format(item.max_velocity) }}</td>

                            <td>{{ "%.4f"|format(item.horsepower) }}</td>

                        </tr>

                        {% endfor %}
                    </table>

                </div>

                
                <div class="footer">
                    <p>分析视频: {{ report_data.video_name }}</p>

                    <p>报告生成日期: {{ report_data.date }}</p>

                </div>

            </div>

        </body>

        </html>

        """
        
        # 生成HTML
        template = Template(html_template)
        html_content = template.render(report_data=report_data, detailed_data=detailed_data)
        
        # 保存报告
        report_file = os.path.join(self.save_folder, "reports", f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"报告已生成: {report_file}")
        return report_file