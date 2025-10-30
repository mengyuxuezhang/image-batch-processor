import sys
import os
import json
import logging
import tempfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTabWidget, QGroupBox, QCheckBox, QPushButton, QLabel, 
                            QLineEdit, QFileDialog, QProgressBar, QTextEdit, QSplitter,
                            QScrollArea, QGridLayout, QSlider, QSpinBox, QDoubleSpinBox,
                            QComboBox, QFrame, QMessageBox, QListWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPixmap

from image_processor import ImageProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProcessingThread(QThread):
    """图片处理线程"""
    progress_signal = pyqtSignal(int)  # 进度信号
    log_signal = pyqtSignal(str)      # 日志信号
    finished_signal = pyqtSignal()    # 完成信号
    
    def __init__(self, processor, input_folder, output_folder, options):
        super().__init__()
        self.processor = processor
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.options = options
        self.is_running = False
        
    def run(self):
        """运行图片处理"""
        try:
            self.is_running = True
            self.log_signal.emit(f"开始处理文件夹: {self.input_folder}")
            
            # 获取所有图片文件
            image_files = self.processor.get_image_files(self.input_folder)
            total_files = len(image_files)
            
            if total_files == 0:
                self.log_signal.emit("未找到任何图片文件")
                self.finished_signal.emit()
                return
                
            self.log_signal.emit(f"找到 {total_files} 个图片文件")
            
            # 处理每个图片文件
            for i, file_path in enumerate(image_files):
                if not self.is_running:
                    self.log_signal.emit("处理已取消")
                    break
                    
                # 获取文件名
                file_name = os.path.basename(file_path)
                self.log_signal.emit(f"正在处理: {file_name}")
                
                # 构建输出路径
                output_path = os.path.join(self.output_folder, file_name)
                
                # 处理图片
                self.processor.process_single_image(file_path, output_path, self.options)
                
                # 更新进度
                progress = int((i + 1) / total_files * 100)
                self.progress_signal.emit(progress)
                
                self.log_signal.emit(f"完成: {file_name}")
                
            if self.is_running:
                self.log_signal.emit(f"所有图片处理完成，共处理 {total_files} 个文件")
                
        except Exception as e:
            self.log_signal.emit(f"处理过程中出错: {str(e)}")
            
        self.finished_signal.emit()
        
    def stop(self):
        """停止处理"""
        self.is_running = False

class ImageProcessorPreviewGUI(QMainWindow):
    """带预览功能的图片处理GUI主界面"""
    
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.processing_thread = None
        self.settings_file = "image_processor_settings.json"
        self.options = self.load_settings()
        self.preview_image_path = None  # 预览图片路径
        self.temp_dir = None  # 临时目录用于存储预览图片
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("高级图片处理器 - 带预览功能")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧：功能选择区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        
        # 基础功能选项卡
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # 基础功能组
        basic_group = QGroupBox("基础功能")
        basic_group_layout = QGridLayout()
        
        # 水印
        self.watermark_checkbox = QCheckBox("添加水印")
        self.watermark_checkbox.setChecked(self.options.get('watermark_text', '') != '')
        basic_group_layout.addWidget(self.watermark_checkbox, 0, 0)
        
        self.watermark_text = QLineEdit()
        self.watermark_text.setText(self.options.get('watermark_text', ''))
        self.watermark_text.setEnabled(self.options.get('watermark_text', '') != '')
        basic_group_layout.addWidget(QLabel("水印文字:"), 0, 1)
        basic_group_layout.addWidget(self.watermark_text, 0, 2)
        
        # 亮度
        self.brightness_checkbox = QCheckBox("调节亮度")
        self.brightness_checkbox.setChecked(self.options.get('brightness_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(self.brightness_checkbox, 1, 0)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(50, 150)
        self.brightness_slider.setValue(int(self.options.get('brightness_factor', 1.0) * 100))
        self.brightness_slider.setEnabled(self.options.get('brightness_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(QLabel("亮度:"), 1, 1)
        basic_group_layout.addWidget(self.brightness_slider, 1, 2)
        
        self.brightness_label = QLabel(f"{self.options.get('brightness_factor', 1.0):.2f}")
        basic_group_layout.addWidget(self.brightness_label, 1, 3)
        
        # 对比度
        self.contrast_checkbox = QCheckBox("调节对比度")
        self.contrast_checkbox.setChecked(self.options.get('contrast_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(self.contrast_checkbox, 2, 0)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 150)
        self.contrast_slider.setValue(int(self.options.get('contrast_factor', 1.0) * 100))
        self.contrast_slider.setEnabled(self.options.get('contrast_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(QLabel("对比度:"), 2, 1)
        basic_group_layout.addWidget(self.contrast_slider, 2, 2)
        
        self.contrast_label = QLabel(f"{self.options.get('contrast_factor', 1.0):.2f}")
        basic_group_layout.addWidget(self.contrast_label, 2, 3)
        
        # 饱和度
        self.saturation_checkbox = QCheckBox("调节饱和度")
        self.saturation_checkbox.setChecked(self.options.get('saturation_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(self.saturation_checkbox, 3, 0)
        
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(int(self.options.get('saturation_factor', 1.0) * 100))
        self.saturation_slider.setEnabled(self.options.get('saturation_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(QLabel("饱和度:"), 3, 1)
        basic_group_layout.addWidget(self.saturation_slider, 3, 2)
        
        self.saturation_label = QLabel(f"{self.options.get('saturation_factor', 1.0):.2f}")
        basic_group_layout.addWidget(self.saturation_label, 3, 3)
        
        # 锐化
        self.sharpness_checkbox = QCheckBox("锐化")
        self.sharpness_checkbox.setChecked(self.options.get('sharpness_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(self.sharpness_checkbox, 4, 0)
        
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 200)
        self.sharpness_slider.setValue(int(self.options.get('sharpness_factor', 1.0) * 100))
        self.sharpness_slider.setEnabled(self.options.get('sharpness_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(QLabel("锐化:"), 4, 1)
        basic_group_layout.addWidget(self.sharpness_slider, 4, 2)
        
        self.sharpness_label = QLabel(f"{self.options.get('sharpness_factor', 1.0):.2f}")
        basic_group_layout.addWidget(self.sharpness_label, 4, 3)
        
        # 模糊
        self.blur_checkbox = QCheckBox("模糊")
        self.blur_checkbox.setChecked(self.options.get('blur_radius', 0.0) > 0.0)
        basic_group_layout.addWidget(self.blur_checkbox, 5, 0)
        
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 50)
        self.blur_slider.setValue(int(self.options.get('blur_radius', 0.0) * 10))
        self.blur_slider.setEnabled(self.options.get('blur_radius', 0.0) > 0.0)
        basic_group_layout.addWidget(QLabel("模糊:"), 5, 1)
        basic_group_layout.addWidget(self.blur_slider, 5, 2)
        
        self.blur_label = QLabel(f"{self.options.get('blur_radius', 0.0):.1f}")
        basic_group_layout.addWidget(self.blur_label, 5, 3)
        
        # 尺寸调整
        self.resize_checkbox = QCheckBox("调整尺寸")
        self.resize_checkbox.setChecked(self.options.get('resize_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(self.resize_checkbox, 6, 0)
        
        self.resize_slider = QSlider(Qt.Horizontal)
        self.resize_slider.setRange(10, 200)
        self.resize_slider.setValue(int(self.options.get('resize_factor', 1.0) * 100))
        self.resize_slider.setEnabled(self.options.get('resize_factor', 1.0) != 1.0)
        basic_group_layout.addWidget(QLabel("尺寸:"), 6, 1)
        basic_group_layout.addWidget(self.resize_slider, 6, 2)
        
        self.resize_label = QLabel(f"{self.options.get('resize_factor', 1.0):.2f}")
        basic_group_layout.addWidget(self.resize_label, 6, 3)
        
        # 裁剪
        self.crop_checkbox = QCheckBox("裁剪")
        self.crop_checkbox.setChecked(self.options.get('crop_percentage', 0.0) > 0.0)
        basic_group_layout.addWidget(self.crop_checkbox, 7, 0)
        
        self.crop_slider = QSlider(Qt.Horizontal)
        self.crop_slider.setRange(0, 30)
        self.crop_slider.setValue(int(self.options.get('crop_percentage', 0.0) * 100))
        self.crop_slider.setEnabled(self.options.get('crop_percentage', 0.0) > 0.0)
        basic_group_layout.addWidget(QLabel("裁剪(%):"), 7, 1)
        basic_group_layout.addWidget(self.crop_slider, 7, 2)
        
        self.crop_label = QLabel(f"{self.options.get('crop_percentage', 0.0) * 100:.0f}%")
        basic_group_layout.addWidget(self.crop_label, 7, 3)
        
        # 边框
        self.border_checkbox = QCheckBox("添加边框")
        self.border_checkbox.setChecked(self.options.get('add_border', False))
        basic_group_layout.addWidget(self.border_checkbox, 8, 0)
        
        # 噪点
        self.noise_checkbox = QCheckBox("添加噪点")
        self.noise_checkbox.setChecked(self.options.get('add_noise', False))
        basic_group_layout.addWidget(self.noise_checkbox, 9, 0)
        
        # 镜像
        mirror_group = QGroupBox("镜像")
        mirror_layout = QHBoxLayout()
        
        self.mirror_horizontal_checkbox = QCheckBox("左右镜像")
        self.mirror_horizontal_checkbox.setChecked(self.options.get('mirror_horizontal', False))
        mirror_layout.addWidget(self.mirror_horizontal_checkbox)
        
        self.mirror_vertical_checkbox = QCheckBox("上下镜像")
        self.mirror_vertical_checkbox.setChecked(self.options.get('mirror_vertical', False))
        mirror_layout.addWidget(self.mirror_vertical_checkbox)
        
        mirror_group.setLayout(mirror_layout)
        
        # 删除元数据
        self.remove_metadata_checkbox = QCheckBox("删除元数据")
        self.remove_metadata_checkbox.setChecked(self.options.get('remove_metadata', True))
        basic_group_layout.addWidget(self.remove_metadata_checkbox, 10, 0)
        
        basic_group.setLayout(basic_group_layout)
        basic_layout.addWidget(basic_group)
        basic_layout.addWidget(mirror_group)
        basic_layout.addStretch()
        
        # 添加基础功能选项卡
        tab_widget.addTab(basic_tab, "基础功能")
        
        # 高级功能选项卡
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        # 几何变换组
        geometry_group = QGroupBox("几何变换")
        geometry_layout = QGridLayout()
        
        # 旋转
        self.rotation_checkbox = QCheckBox("轻微旋转")
        self.rotation_checkbox.setChecked(self.options.get('rotation_angle', 0.0) != 0.0)
        geometry_layout.addWidget(self.rotation_checkbox, 0, 0)
        
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-45, 45)
        self.rotation_slider.setValue(int(self.options.get('rotation_angle', 0.0)))
        self.rotation_slider.setEnabled(self.options.get('rotation_angle', 0.0) != 0.0)
        geometry_layout.addWidget(QLabel("角度:"), 0, 1)
        geometry_layout.addWidget(self.rotation_slider, 0, 2)
        
        self.rotation_label = QLabel(f"{self.options.get('rotation_angle', 0.0)}°")
        geometry_layout.addWidget(self.rotation_label, 0, 3)
        
        # 透视调整
        self.perspective_checkbox = QCheckBox("透视调整")
        self.perspective_checkbox.setChecked(self.options.get('perspective_distortion', 0.0) != 0.0)
        geometry_layout.addWidget(self.perspective_checkbox, 1, 0)
        
        self.perspective_slider = QSlider(Qt.Horizontal)
        self.perspective_slider.setRange(-50, 50)
        self.perspective_slider.setValue(int(self.options.get('perspective_distortion', 0.0) * 100))
        self.perspective_slider.setEnabled(self.options.get('perspective_distortion', 0.0) != 0.0)
        geometry_layout.addWidget(QLabel("透视:"), 1, 1)
        geometry_layout.addWidget(self.perspective_slider, 1, 2)
        
        self.perspective_label = QLabel(f"{self.options.get('perspective_distortion', 0.0):.2f}")
        geometry_layout.addWidget(self.perspective_label, 1, 3)
        
        geometry_group.setLayout(geometry_layout)
        advanced_layout.addWidget(geometry_group)
        
        # 色彩调整组
        color_group = QGroupBox("色彩调整")
        color_layout = QGridLayout()
        
        # 色相
        self.hue_checkbox = QCheckBox("色相调整")
        self.hue_checkbox.setChecked(self.options.get('hue_factor', 0.0) != 0.0)
        color_layout.addWidget(self.hue_checkbox, 0, 0)
        
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setRange(-180, 180)
        self.hue_slider.setValue(int(self.options.get('hue_factor', 0.0)))
        self.hue_slider.setEnabled(self.options.get('hue_factor', 0.0) != 0.0)
        color_layout.addWidget(QLabel("色相:"), 0, 1)
        color_layout.addWidget(self.hue_slider, 0, 2)
        
        self.hue_label = QLabel(f"{self.options.get('hue_factor', 0.0)}°")
        color_layout.addWidget(self.hue_label, 0, 3)
        
        # 色温
        self.color_temp_checkbox = QCheckBox("色温调整")
        self.color_temp_checkbox.setChecked(self.options.get('color_temp_factor', 0.0) != 0.0)
        color_layout.addWidget(self.color_temp_checkbox, 1, 0)
        
        self.color_temp_slider = QSlider(Qt.Horizontal)
        self.color_temp_slider.setRange(-100, 100)
        self.color_temp_slider.setValue(int(self.options.get('color_temp_factor', 0.0)))
        self.color_temp_slider.setEnabled(self.options.get('color_temp_factor', 0.0) != 0.0)
        color_layout.addWidget(QLabel("色温:"), 1, 1)
        color_layout.addWidget(self.color_temp_slider, 1, 2)
        
        self.color_temp_label = QLabel(f"{self.options.get('color_temp_factor', 0.0)}")
        color_layout.addWidget(self.color_temp_label, 1, 3)
        
        # 色彩平衡
        self.color_balance_checkbox = QCheckBox("色彩平衡")
        self.color_balance_checkbox.setChecked(
            self.options.get('red_balance', 1.0) != 1.0 or 
            self.options.get('green_balance', 1.0) != 1.0 or 
            self.options.get('blue_balance', 1.0) != 1.0)
        color_layout.addWidget(self.color_balance_checkbox, 2, 0)
        
        self.red_balance_slider = QSlider(Qt.Horizontal)
        self.red_balance_slider.setRange(50, 150)
        self.red_balance_slider.setValue(int(self.options.get('red_balance', 1.0) * 100))
        self.red_balance_slider.setEnabled(
            self.options.get('red_balance', 1.0) != 1.0 or 
            self.options.get('green_balance', 1.0) != 1.0 or 
            self.options.get('blue_balance', 1.0) != 1.0)
        color_layout.addWidget(QLabel("红:"), 2, 1)
        color_layout.addWidget(self.red_balance_slider, 2, 2)
        
        self.red_balance_label = QLabel(f"{self.options.get('red_balance', 1.0):.2f}")
        color_layout.addWidget(self.red_balance_label, 2, 3)
        
        self.green_balance_slider = QSlider(Qt.Horizontal)
        self.green_balance_slider.setRange(50, 150)
        self.green_balance_slider.setValue(int(self.options.get('green_balance', 1.0) * 100))
        self.green_balance_slider.setEnabled(
            self.options.get('red_balance', 1.0) != 1.0 or 
            self.options.get('green_balance', 1.0) != 1.0 or 
            self.options.get('blue_balance', 1.0) != 1.0)
        color_layout.addWidget(QLabel("绿:"), 3, 1)
        color_layout.addWidget(self.green_balance_slider, 3, 2)
        
        self.green_balance_label = QLabel(f"{self.options.get('green_balance', 1.0):.2f}")
        color_layout.addWidget(self.green_balance_label, 3, 3)
        
        self.blue_balance_slider = QSlider(Qt.Horizontal)
        self.blue_balance_slider.setRange(50, 150)
        self.blue_balance_slider.setValue(int(self.options.get('blue_balance', 1.0) * 100))
        self.blue_balance_slider.setEnabled(
            self.options.get('red_balance', 1.0) != 1.0 or 
            self.options.get('green_balance', 1.0) != 1.0 or 
            self.options.get('blue_balance', 1.0) != 1.0)
        color_layout.addWidget(QLabel("蓝:"), 4, 1)
        color_layout.addWidget(self.blue_balance_slider, 4, 2)
        
        self.blue_balance_label = QLabel(f"{self.options.get('blue_balance', 1.0):.2f}")
        color_layout.addWidget(self.blue_balance_label, 4, 3)
        
        # 黑白转换
        self.grayscale_checkbox = QCheckBox("黑白转换")
        self.grayscale_checkbox.setChecked(self.options.get('convert_to_grayscale', False))
        color_layout.addWidget(self.grayscale_checkbox, 5, 0)
        
        self.grayscale_combo = QComboBox()
        self.grayscale_combo.addItems(['weighted', 'average', 'luminosity', 'desaturation'])
        self.grayscale_combo.setCurrentText(self.options.get('grayscale_method', 'weighted'))
        self.grayscale_combo.setEnabled(self.options.get('convert_to_grayscale', False))
        color_layout.addWidget(QLabel("方法:"), 5, 1)
        color_layout.addWidget(self.grayscale_combo, 5, 2)
        
        color_group.setLayout(color_layout)
        advanced_layout.addWidget(color_group)
        
        # 光影效果组
        lighting_group = QGroupBox("光影效果")
        lighting_layout = QGridLayout()
        
        # 阴影/高光调整
        self.shadows_highlights_checkbox = QCheckBox("阴影/高光调整")
        self.shadows_highlights_checkbox.setChecked(
            self.options.get('shadow_factor', 0.0) != 0.0 or 
            self.options.get('highlight_factor', 0.0) != 0.0)
        lighting_layout.addWidget(self.shadows_highlights_checkbox, 0, 0)
        
        self.shadow_slider = QSlider(Qt.Horizontal)
        self.shadow_slider.setRange(-100, 100)
        self.shadow_slider.setValue(int(self.options.get('shadow_factor', 0.0) * 100))
        self.shadow_slider.setEnabled(
            self.options.get('shadow_factor', 0.0) != 0.0 or 
            self.options.get('highlight_factor', 0.0) != 0.0)
        lighting_layout.addWidget(QLabel("阴影:"), 0, 1)
        lighting_layout.addWidget(self.shadow_slider, 0, 2)
        
        self.shadow_label = QLabel(f"{self.options.get('shadow_factor', 0.0):.2f}")
        lighting_layout.addWidget(self.shadow_label, 0, 3)
        
        self.highlight_slider = QSlider(Qt.Horizontal)
        self.highlight_slider.setRange(-100, 100)
        self.highlight_slider.setValue(int(self.options.get('highlight_factor', 0.0) * 100))
        self.highlight_slider.setEnabled(
            self.options.get('shadow_factor', 0.0) != 0.0 or 
            self.options.get('highlight_factor', 0.0) != 0.0)
        lighting_layout.addWidget(QLabel("高光:"), 1, 1)
        lighting_layout.addWidget(self.highlight_slider, 1, 2)
        
        self.highlight_label = QLabel(f"{self.options.get('highlight_factor', 0.0):.2f}")
        lighting_layout.addWidget(self.highlight_label, 1, 3)
        
        # 曝光调整
        self.exposure_checkbox = QCheckBox("曝光调整")
        self.exposure_checkbox.setChecked(self.options.get('exposure_factor', 0.0) != 0.0)
        lighting_layout.addWidget(self.exposure_checkbox, 2, 0)
        
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-100, 100)
        self.exposure_slider.setValue(int(self.options.get('exposure_factor', 0.0) * 100))
        self.exposure_slider.setEnabled(self.options.get('exposure_factor', 0.0) != 0.0)
        lighting_layout.addWidget(QLabel("曝光:"), 2, 1)
        lighting_layout.addWidget(self.exposure_slider, 2, 2)
        
        self.exposure_label = QLabel(f"{self.options.get('exposure_factor', 0.0):.2f}")
        lighting_layout.addWidget(self.exposure_label, 2, 3)
        
        # 晕影效果
        self.vignette_checkbox = QCheckBox("晕影效果")
        self.vignette_checkbox.setChecked(self.options.get('vignette_strength', 0.0) > 0.0)
        lighting_layout.addWidget(self.vignette_checkbox, 3, 0)
        
        self.vignette_slider = QSlider(Qt.Horizontal)
        self.vignette_slider.setRange(0, 100)
        self.vignette_slider.setValue(int(self.options.get('vignette_strength', 0.0) * 100))
        self.vignette_slider.setEnabled(self.options.get('vignette_strength', 0.0) > 0.0)
        lighting_layout.addWidget(QLabel("晕影:"), 3, 1)
        lighting_layout.addWidget(self.vignette_slider, 3, 2)
        
        self.vignette_label = QLabel(f"{self.options.get('vignette_strength', 0.0):.2f}")
        lighting_layout.addWidget(self.vignette_label, 3, 3)
        
        lighting_group.setLayout(lighting_layout)
        advanced_layout.addWidget(lighting_group)
        
        # 细节处理组
        detail_group = QGroupBox("细节处理")
        detail_layout = QGridLayout()
        
        # 智能锐化
        self.smart_sharpen_checkbox = QCheckBox("智能锐化")
        self.smart_sharpen_checkbox.setChecked(self.options.get('smart_sharpen_amount', 0.0) > 0.0)
        detail_layout.addWidget(self.smart_sharpen_checkbox, 0, 0)
        
        self.smart_sharpen_slider = QSlider(Qt.Horizontal)
        self.smart_sharpen_slider.setRange(0, 100)
        self.smart_sharpen_slider.setValue(int(self.options.get('smart_sharpen_amount', 0.0) * 100))
        self.smart_sharpen_slider.setEnabled(self.options.get('smart_sharpen_amount', 0.0) > 0.0)
        detail_layout.addWidget(QLabel("强度:"), 0, 1)
        detail_layout.addWidget(self.smart_sharpen_slider, 0, 2)
        
        self.smart_sharpen_label = QLabel(f"{self.options.get('smart_sharpen_amount', 0.0):.2f}")
        detail_layout.addWidget(self.smart_sharpen_label, 0, 3)
        
        # 噪点减少
        self.reduce_noise_checkbox = QCheckBox("噪点减少")
        self.reduce_noise_checkbox.setChecked(self.options.get('reduce_noise_strength', 0.0) > 0.0)
        detail_layout.addWidget(self.reduce_noise_checkbox, 1, 0)
        
        self.reduce_noise_slider = QSlider(Qt.Horizontal)
        self.reduce_noise_slider.setRange(0, 100)
        self.reduce_noise_slider.setValue(int(self.options.get('reduce_noise_strength', 0.0) * 100))
        self.reduce_noise_slider.setEnabled(self.options.get('reduce_noise_strength', 0.0) > 0.0)
        detail_layout.addWidget(QLabel("降噪:"), 1, 1)
        detail_layout.addWidget(self.reduce_noise_slider, 1, 2)
        
        self.reduce_noise_label = QLabel(f"{self.options.get('reduce_noise_strength', 0.0):.2f}")
        detail_layout.addWidget(self.reduce_noise_label, 1, 3)
        
        # 颗粒效果
        self.grain_checkbox = QCheckBox("颗粒效果")
        self.grain_checkbox.setChecked(self.options.get('grain_strength', 0.0) > 0.0)
        detail_layout.addWidget(self.grain_checkbox, 2, 0)
        
        self.grain_slider = QSlider(Qt.Horizontal)
        self.grain_slider.setRange(0, 100)
        self.grain_slider.setValue(int(self.options.get('grain_strength', 0.0) * 100))
        self.grain_slider.setEnabled(self.options.get('grain_strength', 0.0) > 0.0)
        detail_layout.addWidget(QLabel("颗粒:"), 2, 1)
        detail_layout.addWidget(self.grain_slider, 2, 2)
        
        self.grain_label = QLabel(f"{self.options.get('grain_strength', 0.0):.2f}")
        detail_layout.addWidget(self.grain_label, 2, 3)
        
        # 细节增强
        self.enhance_details_checkbox = QCheckBox("细节增强")
        self.enhance_details_checkbox.setChecked(self.options.get('enhance_details_strength', 0.0) > 0.0)
        detail_layout.addWidget(self.enhance_details_checkbox, 3, 0)
        
        self.enhance_details_slider = QSlider(Qt.Horizontal)
        self.enhance_details_slider.setRange(0, 100)
        self.enhance_details_slider.setValue(int(self.options.get('enhance_details_strength', 0.0) * 100))
        self.enhance_details_slider.setEnabled(self.options.get('enhance_details_strength', 0.0) > 0.0)
        detail_layout.addWidget(QLabel("增强:"), 3, 1)
        detail_layout.addWidget(self.enhance_details_slider, 3, 2)
        
        self.enhance_details_label = QLabel(f"{self.options.get('enhance_details_strength', 0.0):.2f}")
        detail_layout.addWidget(self.enhance_details_label, 3, 3)
        
        detail_group.setLayout(detail_layout)
        advanced_layout.addWidget(detail_group)
        
        # 特殊效果组
        effect_group = QGroupBox("特殊效果")
        effect_layout = QGridLayout()
        
        # 色差效果
        self.chromatic_aberration_checkbox = QCheckBox("色差效果")
        self.chromatic_aberration_checkbox.setChecked(self.options.get('chromatic_aberration', 0.0) > 0.0)
        effect_layout.addWidget(self.chromatic_aberration_checkbox, 0, 0)
        
        self.chromatic_aberration_slider = QSlider(Qt.Horizontal)
        self.chromatic_aberration_slider.setRange(0, 100)
        self.chromatic_aberration_slider.setValue(int(self.options.get('chromatic_aberration', 0.0) * 100))
        self.chromatic_aberration_slider.setEnabled(self.options.get('chromatic_aberration', 0.0) > 0.0)
        effect_layout.addWidget(QLabel("色差:"), 0, 1)
        effect_layout.addWidget(self.chromatic_aberration_slider, 0, 2)
        
        self.chromatic_aberration_label = QLabel(f"{self.options.get('chromatic_aberration', 0.0):.2f}")
        effect_layout.addWidget(self.chromatic_aberration_label, 0, 3)
        
        # 柔光效果
        self.soft_glow_checkbox = QCheckBox("柔光效果")
        self.soft_glow_checkbox.setChecked(self.options.get('soft_glow_strength', 0.0) > 0.0)
        effect_layout.addWidget(self.soft_glow_checkbox, 1, 0)
        
        self.soft_glow_slider = QSlider(Qt.Horizontal)
        self.soft_glow_slider.setRange(0, 100)
        self.soft_glow_slider.setValue(int(self.options.get('soft_glow_strength', 0.0) * 100))
        self.soft_glow_slider.setEnabled(self.options.get('soft_glow_strength', 0.0) > 0.0)
        effect_layout.addWidget(QLabel("柔光:"), 1, 1)
        effect_layout.addWidget(self.soft_glow_slider, 1, 2)
        
        self.soft_glow_label = QLabel(f"{self.options.get('soft_glow_strength', 0.0):.2f}")
        effect_layout.addWidget(self.soft_glow_label, 1, 3)
        
        effect_group.setLayout(effect_layout)
        advanced_layout.addWidget(effect_group)
        
        advanced_layout.addStretch()
        
        # 添加高级功能选项卡
        tab_widget.addTab(advanced_tab, "高级功能")
        
        left_layout.addWidget(tab_widget)
        
        # 右侧：文件夹选择、预览和处理区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 文件夹选择组
        folder_group = QGroupBox("文件夹选择")
        folder_layout = QGridLayout()
        
        # 输入文件夹
        folder_layout.addWidget(QLabel("输入文件夹:"), 0, 0)
        self.input_folder_edit = QLineEdit()
        folder_layout.addWidget(self.input_folder_edit, 0, 1)
        
        self.input_folder_button = QPushButton("浏览...")
        self.input_folder_button.clicked.connect(self.select_input_folder)
        folder_layout.addWidget(self.input_folder_button, 0, 2)
        
        # 输出文件夹
        folder_layout.addWidget(QLabel("输出文件夹:"), 1, 0)
        self.output_folder_edit = QLineEdit()
        folder_layout.addWidget(self.output_folder_edit, 1, 1)
        
        self.output_folder_button = QPushButton("浏览...")
        self.output_folder_button.clicked.connect(self.select_output_folder)
        folder_layout.addWidget(self.output_folder_button, 1, 2)
        
        folder_group.setLayout(folder_layout)
        right_layout.addWidget(folder_group)
        
        # 预览组
        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout()
        
        # 预览按钮
        preview_button_layout = QHBoxLayout()
        self.select_preview_button = QPushButton("选择预览图片")
        self.select_preview_button.clicked.connect(self.select_preview_image)
        self.generate_preview_button = QPushButton("生成预览")
        self.generate_preview_button.clicked.connect(self.generate_preview)
        preview_button_layout.addWidget(self.select_preview_button)
        preview_button_layout.addWidget(self.generate_preview_button)
        preview_layout.addLayout(preview_button_layout)
        
        # 预览图像显示区域
        self.preview_images_layout = QHBoxLayout()
        self.original_preview_label = QLabel("原始图片")
        self.original_preview_label.setAlignment(Qt.AlignCenter)
        self.original_preview_label.setMinimumSize(300, 300)
        self.original_preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        
        self.processed_preview_label = QLabel("处理后图片")
        self.processed_preview_label.setAlignment(Qt.AlignCenter)
        self.processed_preview_label.setMinimumSize(300, 300)
        self.processed_preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        
        self.preview_images_layout.addWidget(self.original_preview_label)
        self.preview_images_layout.addWidget(self.processed_preview_label)
        preview_layout.addLayout(self.preview_images_layout)
        
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group)
        
        # 处理按钮组
        button_group = QGroupBox("处理控制")
        button_layout = QHBoxLayout()
        
        self.process_button = QPushButton("开始处理")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.process_button)
        
        self.stop_button = QPushButton("停止处理")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        button_layout.addWidget(self.stop_button)
        
        button_group.setLayout(button_layout)
        right_layout.addWidget(button_group)
        
        # 进度条
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("就绪")
        progress_layout.addWidget(self.progress_label)
        
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)
        
        # 日志输出
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # 设置左右部件比例
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 800])
        
        # 连接信号和槽
        self.connect_signals()
        
        # 设置初始文件夹路径
        if not self.input_folder_edit.text():
            self.input_folder_edit.setText(os.getcwd())
        if not self.output_folder_edit.text():
            self.output_folder_edit.setText(os.path.join(os.getcwd(), "output"))
            
    def connect_signals(self):
        """连接信号和槽"""
        # 基础功能
        self.watermark_checkbox.stateChanged.connect(self.on_watermark_checkbox_changed)
        self.watermark_text.textChanged.connect(self.on_watermark_text_changed)
        
        self.brightness_checkbox.stateChanged.connect(self.on_brightness_checkbox_changed)
        self.brightness_slider.valueChanged.connect(self.on_brightness_slider_changed)
        
        self.contrast_checkbox.stateChanged.connect(self.on_contrast_checkbox_changed)
        self.contrast_slider.valueChanged.connect(self.on_contrast_slider_changed)
        
        self.saturation_checkbox.stateChanged.connect(self.on_saturation_checkbox_changed)
        self.saturation_slider.valueChanged.connect(self.on_saturation_slider_changed)
        
        self.sharpness_checkbox.stateChanged.connect(self.on_sharpness_checkbox_changed)
        self.sharpness_slider.valueChanged.connect(self.on_sharpness_slider_changed)
        
        self.blur_checkbox.stateChanged.connect(self.on_blur_checkbox_changed)
        self.blur_slider.valueChanged.connect(self.on_blur_slider_changed)
        
        self.resize_checkbox.stateChanged.connect(self.on_resize_checkbox_changed)
        self.resize_slider.valueChanged.connect(self.on_resize_slider_changed)
        
        self.crop_checkbox.stateChanged.connect(self.on_crop_checkbox_changed)
        self.crop_slider.valueChanged.connect(self.on_crop_slider_changed)
        
        self.border_checkbox.stateChanged.connect(self.on_border_checkbox_changed)
        self.noise_checkbox.stateChanged.connect(self.on_noise_checkbox_changed)
        self.mirror_horizontal_checkbox.stateChanged.connect(self.on_mirror_horizontal_checkbox_changed)
        self.mirror_vertical_checkbox.stateChanged.connect(self.on_mirror_vertical_checkbox_changed)
        self.remove_metadata_checkbox.stateChanged.connect(self.on_remove_metadata_checkbox_changed)
        
        # 高级功能
        self.rotation_checkbox.stateChanged.connect(self.on_rotation_checkbox_changed)
        self.rotation_slider.valueChanged.connect(self.on_rotation_slider_changed)
        
        self.perspective_checkbox.stateChanged.connect(self.on_perspective_checkbox_changed)
        self.perspective_slider.valueChanged.connect(self.on_perspective_slider_changed)
        
        self.hue_checkbox.stateChanged.connect(self.on_hue_checkbox_changed)
        self.hue_slider.valueChanged.connect(self.on_hue_slider_changed)
        
        self.color_temp_checkbox.stateChanged.connect(self.on_color_temp_checkbox_changed)
        self.color_temp_slider.valueChanged.connect(self.on_color_temp_slider_changed)
        
        self.color_balance_checkbox.stateChanged.connect(self.on_color_balance_checkbox_changed)
        self.red_balance_slider.valueChanged.connect(self.on_red_balance_slider_changed)
        self.green_balance_slider.valueChanged.connect(self.on_green_balance_slider_changed)
        self.blue_balance_slider.valueChanged.connect(self.on_blue_balance_slider_changed)
        
        self.grayscale_checkbox.stateChanged.connect(self.on_grayscale_checkbox_changed)
        self.grayscale_combo.currentTextChanged.connect(self.on_grayscale_combo_changed)
        
        self.shadows_highlights_checkbox.stateChanged.connect(self.on_shadows_highlights_checkbox_changed)
        self.shadow_slider.valueChanged.connect(self.on_shadow_slider_changed)
        self.highlight_slider.valueChanged.connect(self.on_highlight_slider_changed)
        
        self.exposure_checkbox.stateChanged.connect(self.on_exposure_checkbox_changed)
        self.exposure_slider.valueChanged.connect(self.on_exposure_slider_changed)
        
        self.vignette_checkbox.stateChanged.connect(self.on_vignette_checkbox_changed)
        self.vignette_slider.valueChanged.connect(self.on_vignette_slider_changed)
        
        self.smart_sharpen_checkbox.stateChanged.connect(self.on_smart_sharpen_checkbox_changed)
        self.smart_sharpen_slider.valueChanged.connect(self.on_smart_sharpen_slider_changed)
        
        self.reduce_noise_checkbox.stateChanged.connect(self.on_reduce_noise_checkbox_changed)
        self.reduce_noise_slider.valueChanged.connect(self.on_reduce_noise_slider_changed)
        
        self.grain_checkbox.stateChanged.connect(self.on_grain_checkbox_changed)
        self.grain_slider.valueChanged.connect(self.on_grain_slider_changed)
        
        self.enhance_details_checkbox.stateChanged.connect(self.on_enhance_details_checkbox_changed)
        self.enhance_details_slider.valueChanged.connect(self.on_enhance_details_slider_changed)
        
        self.chromatic_aberration_checkbox.stateChanged.connect(self.on_chromatic_aberration_checkbox_changed)
        self.chromatic_aberration_slider.valueChanged.connect(self.on_chromatic_aberration_slider_changed)
        
        self.soft_glow_checkbox.stateChanged.connect(self.on_soft_glow_checkbox_changed)
        self.soft_glow_slider.valueChanged.connect(self.on_soft_glow_slider_changed)
        
    # 基础功能信号处理
    def on_watermark_checkbox_changed(self, state):
        """水印复选框状态改变"""
        enabled = state == Qt.Checked
        self.watermark_text.setEnabled(enabled)
        if enabled:
            self.options['watermark_text'] = self.watermark_text.text() if self.watermark_text.text() else "Watermark"
        else:
            self.options['watermark_text'] = ""
        self.save_settings()
        
    def on_watermark_text_changed(self, text):
        """水印文字改变"""
        self.options['watermark_text'] = text
        self.save_settings()
        
    def on_brightness_checkbox_changed(self, state):
        """亮度复选框状态改变"""
        enabled = state == Qt.Checked
        self.brightness_slider.setEnabled(enabled)
        if enabled:
            self.options['brightness_factor'] = self.brightness_slider.value() / 100.0
        else:
            self.options['brightness_factor'] = 1.0
            self.brightness_slider.setValue(100)
        self.save_settings()
        
    def on_brightness_slider_changed(self, value):
        """亮度滑块值改变"""
        factor = value / 100.0
        self.options['brightness_factor'] = factor
        self.brightness_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_contrast_checkbox_changed(self, state):
        """对比度复选框状态改变"""
        enabled = state == Qt.Checked
        self.contrast_slider.setEnabled(enabled)
        if enabled:
            self.options['contrast_factor'] = self.contrast_slider.value() / 100.0
        else:
            self.options['contrast_factor'] = 1.0
            self.contrast_slider.setValue(100)
        self.save_settings()
        
    def on_contrast_slider_changed(self, value):
        """对比度滑块值改变"""
        factor = value / 100.0
        self.options['contrast_factor'] = factor
        self.contrast_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_saturation_checkbox_changed(self, state):
        """饱和度复选框状态改变"""
        enabled = state == Qt.Checked
        self.saturation_slider.setEnabled(enabled)
        if enabled:
            self.options['saturation_factor'] = self.saturation_slider.value() / 100.0
        else:
            self.options['saturation_factor'] = 1.0
            self.saturation_slider.setValue(100)
        self.save_settings()
        
    def on_saturation_slider_changed(self, value):
        """饱和度滑块值改变"""
        factor = value / 100.0
        self.options['saturation_factor'] = factor
        self.saturation_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_sharpness_checkbox_changed(self, state):
        """锐化复选框状态改变"""
        enabled = state == Qt.Checked
        self.sharpness_slider.setEnabled(enabled)
        if enabled:
            self.options['sharpness_factor'] = self.sharpness_slider.value() / 100.0
        else:
            self.options['sharpness_factor'] = 1.0
            self.sharpness_slider.setValue(100)
        self.save_settings()
        
    def on_sharpness_slider_changed(self, value):
        """锐化滑块值改变"""
        factor = value / 100.0
        self.options['sharpness_factor'] = factor
        self.sharpness_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_blur_checkbox_changed(self, state):
        """模糊复选框状态改变"""
        enabled = state == Qt.Checked
        self.blur_slider.setEnabled(enabled)
        if enabled:
            self.options['blur_radius'] = self.blur_slider.value() / 10.0
        else:
            self.options['blur_radius'] = 0.0
            self.blur_slider.setValue(0)
        self.save_settings()
        
    def on_blur_slider_changed(self, value):
        """模糊滑块值改变"""
        radius = value / 10.0
        self.options['blur_radius'] = radius
        self.blur_label.setText(f"{radius:.1f}")
        self.save_settings()
        
    def on_resize_checkbox_changed(self, state):
        """尺寸复选框状态改变"""
        enabled = state == Qt.Checked
        self.resize_slider.setEnabled(enabled)
        if enabled:
            self.options['resize_factor'] = self.resize_slider.value() / 100.0
        else:
            self.options['resize_factor'] = 1.0
            self.resize_slider.setValue(100)
        self.save_settings()
        
    def on_resize_slider_changed(self, value):
        """尺寸滑块值改变"""
        factor = value / 100.0
        self.options['resize_factor'] = factor
        self.resize_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_crop_checkbox_changed(self, state):
        """裁剪复选框状态改变"""
        enabled = state == Qt.Checked
        self.crop_slider.setEnabled(enabled)
        if enabled:
            self.options['crop_percentage'] = self.crop_slider.value() / 100.0
        else:
            self.options['crop_percentage'] = 0.0
            self.crop_slider.setValue(0)
        self.save_settings()
        
    def on_crop_slider_changed(self, value):
        """裁剪滑块值改变"""
        percentage = value / 100.0
        self.options['crop_percentage'] = percentage
        self.crop_label.setText(f"{percentage * 100:.0f}%")
        self.save_settings()
        
    def on_border_checkbox_changed(self, state):
        """边框复选框状态改变"""
        self.options['add_border'] = state == Qt.Checked
        self.save_settings()
        
    def on_noise_checkbox_changed(self, state):
        """噪点复选框状态改变"""
        self.options['add_noise'] = state == Qt.Checked
        self.save_settings()
        
    def on_mirror_horizontal_checkbox_changed(self, state):
        """水平镜像复选框状态改变"""
        self.options['mirror_horizontal'] = state == Qt.Checked
        self.save_settings()
        
    def on_mirror_vertical_checkbox_changed(self, state):
        """垂直镜像复选框状态改变"""
        self.options['mirror_vertical'] = state == Qt.Checked
        self.save_settings()
        
    def on_remove_metadata_checkbox_changed(self, state):
        """删除元数据复选框状态改变"""
        self.options['remove_metadata'] = state == Qt.Checked
        self.save_settings()
        
    # 高级功能信号处理
    def on_rotation_checkbox_changed(self, state):
        """旋转复选框状态改变"""
        enabled = state == Qt.Checked
        self.rotation_slider.setEnabled(enabled)
        if enabled:
            self.options['rotation_angle'] = self.rotation_slider.value()
        else:
            self.options['rotation_angle'] = 0.0
            self.rotation_slider.setValue(0)
        self.save_settings()
        
    def on_rotation_slider_changed(self, value):
        """旋转滑块值改变"""
        self.options['rotation_angle'] = value
        self.rotation_label.setText(f"{value}°")
        self.save_settings()
        
    def on_perspective_checkbox_changed(self, state):
        """透视复选框状态改变"""
        enabled = state == Qt.Checked
        self.perspective_slider.setEnabled(enabled)
        if enabled:
            self.options['perspective_distortion'] = self.perspective_slider.value() / 100.0
        else:
            self.options['perspective_distortion'] = 0.0
            self.perspective_slider.setValue(0)
        self.save_settings()
        
    def on_perspective_slider_changed(self, value):
        """透视滑块值改变"""
        factor = value / 100.0
        self.options['perspective_distortion'] = factor
        self.perspective_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_hue_checkbox_changed(self, state):
        """色相复选框状态改变"""
        enabled = state == Qt.Checked
        self.hue_slider.setEnabled(enabled)
        if enabled:
            self.options['hue_factor'] = self.hue_slider.value()
        else:
            self.options['hue_factor'] = 0.0
            self.hue_slider.setValue(0)
        self.save_settings()
        
    def on_hue_slider_changed(self, value):
        """色相滑块值改变"""
        self.options['hue_factor'] = value
        self.hue_label.setText(f"{value}°")
        self.save_settings()
        
    def on_color_temp_checkbox_changed(self, state):
        """色温复选框状态改变"""
        enabled = state == Qt.Checked
        self.color_temp_slider.setEnabled(enabled)
        if enabled:
            self.options['color_temp_factor'] = self.color_temp_slider.value()
        else:
            self.options['color_temp_factor'] = 0.0
            self.color_temp_slider.setValue(0)
        self.save_settings()
        
    def on_color_temp_slider_changed(self, value):
        """色温滑块值改变"""
        self.options['color_temp_factor'] = value
        self.color_temp_label.setText(f"{value}")
        self.save_settings()
        
    def on_color_balance_checkbox_changed(self, state):
        """色彩平衡复选框状态改变"""
        enabled = state == Qt.Checked
        self.red_balance_slider.setEnabled(enabled)
        self.green_balance_slider.setEnabled(enabled)
        self.blue_balance_slider.setEnabled(enabled)
        if enabled:
            self.options['red_balance'] = self.red_balance_slider.value() / 100.0
            self.options['green_balance'] = self.green_balance_slider.value() / 100.0
            self.options['blue_balance'] = self.blue_balance_slider.value() / 100.0
        else:
            self.options['red_balance'] = 1.0
            self.options['green_balance'] = 1.0
            self.options['blue_balance'] = 1.0
            self.red_balance_slider.setValue(100)
            self.green_balance_slider.setValue(100)
            self.blue_balance_slider.setValue(100)
        self.save_settings()
        
    def on_red_balance_slider_changed(self, value):
        """红色平衡滑块值改变"""
        factor = value / 100.0
        self.options['red_balance'] = factor
        self.red_balance_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_green_balance_slider_changed(self, value):
        """绿色平衡滑块值改变"""
        factor = value / 100.0
        self.options['green_balance'] = factor
        self.green_balance_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_blue_balance_slider_changed(self, value):
        """蓝色平衡滑块值改变"""
        factor = value / 100.0
        self.options['blue_balance'] = factor
        self.blue_balance_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_grayscale_checkbox_changed(self, state):
        """黑白转换复选框状态改变"""
        enabled = state == Qt.Checked
        self.grayscale_combo.setEnabled(enabled)
        self.options['convert_to_grayscale'] = enabled
        self.save_settings()
        
    def on_grayscale_combo_changed(self, text):
        """黑白转换下拉框改变"""
        self.options['grayscale_method'] = text
        self.save_settings()
        
    def on_shadows_highlights_checkbox_changed(self, state):
        """阴影/高光复选框状态改变"""
        enabled = state == Qt.Checked
        self.shadow_slider.setEnabled(enabled)
        self.highlight_slider.setEnabled(enabled)
        if enabled:
            self.options['shadow_factor'] = self.shadow_slider.value() / 100.0
            self.options['highlight_factor'] = self.highlight_slider.value() / 100.0
        else:
            self.options['shadow_factor'] = 0.0
            self.options['highlight_factor'] = 0.0
            self.shadow_slider.setValue(0)
            self.highlight_slider.setValue(0)
        self.save_settings()
        
    def on_shadow_slider_changed(self, value):
        """阴影滑块值改变"""
        factor = value / 100.0
        self.options['shadow_factor'] = factor
        self.shadow_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_highlight_slider_changed(self, value):
        """高光滑块值改变"""
        factor = value / 100.0
        self.options['highlight_factor'] = factor
        self.highlight_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_exposure_checkbox_changed(self, state):
        """曝光复选框状态改变"""
        enabled = state == Qt.Checked
        self.exposure_slider.setEnabled(enabled)
        if enabled:
            self.options['exposure_factor'] = self.exposure_slider.value() / 100.0
        else:
            self.options['exposure_factor'] = 0.0
            self.exposure_slider.setValue(0)
        self.save_settings()
        
    def on_exposure_slider_changed(self, value):
        """曝光滑块值改变"""
        factor = value / 100.0
        self.options['exposure_factor'] = factor
        self.exposure_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_vignette_checkbox_changed(self, state):
        """晕影复选框状态改变"""
        enabled = state == Qt.Checked
        self.vignette_slider.setEnabled(enabled)
        if enabled:
            self.options['vignette_strength'] = self.vignette_slider.value() / 100.0
        else:
            self.options['vignette_strength'] = 0.0
            self.vignette_slider.setValue(0)
        self.save_settings()
        
    def on_vignette_slider_changed(self, value):
        """晕影滑块值改变"""
        factor = value / 100.0
        self.options['vignette_strength'] = factor
        self.vignette_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_smart_sharpen_checkbox_changed(self, state):
        """智能锐化复选框状态改变"""
        enabled = state == Qt.Checked
        self.smart_sharpen_slider.setEnabled(enabled)
        if enabled:
            self.options['smart_sharpen_amount'] = self.smart_sharpen_slider.value() / 100.0
        else:
            self.options['smart_sharpen_amount'] = 0.0
            self.smart_sharpen_slider.setValue(0)
        self.save_settings()
        
    def on_smart_sharpen_slider_changed(self, value):
        """智能锐化滑块值改变"""
        factor = value / 100.0
        self.options['smart_sharpen_amount'] = factor
        self.smart_sharpen_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_reduce_noise_checkbox_changed(self, state):
        """噪点减少复选框状态改变"""
        enabled = state == Qt.Checked
        self.reduce_noise_slider.setEnabled(enabled)
        if enabled:
            self.options['reduce_noise_strength'] = self.reduce_noise_slider.value() / 100.0
        else:
            self.options['reduce_noise_strength'] = 0.0
            self.reduce_noise_slider.setValue(0)
        self.save_settings()
        
    def on_reduce_noise_slider_changed(self, value):
        """噪点减少滑块值改变"""
        factor = value / 100.0
        self.options['reduce_noise_strength'] = factor
        self.reduce_noise_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_grain_checkbox_changed(self, state):
        """颗粒效果复选框状态改变"""
        enabled = state == Qt.Checked
        self.grain_slider.setEnabled(enabled)
        if enabled:
            self.options['grain_strength'] = self.grain_slider.value() / 100.0
        else:
            self.options['grain_strength'] = 0.0
            self.grain_slider.setValue(0)
        self.save_settings()
        
    def on_grain_slider_changed(self, value):
        """颗粒效果滑块值改变"""
        factor = value / 100.0
        self.options['grain_strength'] = factor
        self.grain_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_enhance_details_checkbox_changed(self, state):
        """细节增强复选框状态改变"""
        enabled = state == Qt.Checked
        self.enhance_details_slider.setEnabled(enabled)
        if enabled:
            self.options['enhance_details_strength'] = self.enhance_details_slider.value() / 100.0
        else:
            self.options['enhance_details_strength'] = 0.0
            self.enhance_details_slider.setValue(0)
        self.save_settings()
        
    def on_enhance_details_slider_changed(self, value):
        """细节增强滑块值改变"""
        factor = value / 100.0
        self.options['enhance_details_strength'] = factor
        self.enhance_details_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_chromatic_aberration_checkbox_changed(self, state):
        """色差效果复选框状态改变"""
        enabled = state == Qt.Checked
        self.chromatic_aberration_slider.setEnabled(enabled)
        if enabled:
            self.options['chromatic_aberration'] = self.chromatic_aberration_slider.value() / 100.0
        else:
            self.options['chromatic_aberration'] = 0.0
            self.chromatic_aberration_slider.setValue(0)
        self.save_settings()
        
    def on_chromatic_aberration_slider_changed(self, value):
        """色差效果滑块值改变"""
        factor = value / 100.0
        self.options['chromatic_aberration'] = factor
        self.chromatic_aberration_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def on_soft_glow_checkbox_changed(self, state):
        """柔光效果复选框状态改变"""
        enabled = state == Qt.Checked
        self.soft_glow_slider.setEnabled(enabled)
        if enabled:
            self.options['soft_glow_strength'] = self.soft_glow_slider.value() / 100.0
        else:
            self.options['soft_glow_strength'] = 0.0
            self.soft_glow_slider.setValue(0)
        self.save_settings()
        
    def on_soft_glow_slider_changed(self, value):
        """柔光效果滑块值改变"""
        factor = value / 100.0
        self.options['soft_glow_strength'] = factor
        self.soft_glow_label.setText(f"{factor:.2f}")
        self.save_settings()
        
    def select_input_folder(self):
        """选择输入文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹", self.input_folder_edit.text())
        if folder:
            self.input_folder_edit.setText(folder)
            
    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.output_folder_edit.text())
        if folder:
            self.output_folder_edit.setText(folder)
            
    def select_preview_image(self):
        """选择预览图片"""
        input_folder = self.input_folder_edit.text()
        if not input_folder or not os.path.exists(input_folder):
            QMessageBox.warning(self, "警告", "请先选择有效的输入文件夹")
            return
            
        # 打开文件选择对话框，只显示图片文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择预览图片", 
            input_folder,
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.gif)"
        )
        
        if file_path:
            self.preview_image_path = file_path
            self.show_preview_image(file_path, self.original_preview_label)
            self.log_text.append(f"已选择预览图片: {os.path.basename(file_path)}")
            
    def show_preview_image(self, image_path, label):
        """在指定标签中显示图片"""
        if not os.path.exists(image_path):
            return
            
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            label.setText("无法加载图片")
            return
            
        # 缩放图片以适应标签
        pixmap = pixmap.scaled(
            label.width() - 20, 
            label.height() - 20, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)
        
    def generate_preview(self):
        """生成预览图片"""
        if not self.preview_image_path:
            QMessageBox.warning(self, "警告", "请先选择预览图片")
            return
            
        try:
            # 创建临时目录用于存储预览图片
            if not self.temp_dir:
                self.temp_dir = tempfile.mkdtemp()
                
            # 生成处理后的预览图片路径
            preview_output_path = os.path.join(self.temp_dir, "preview_processed.jpg")
            
            # 处理预览图片
            self.log_text.append("正在生成预览...")
            QApplication.processEvents()  # 更新界面
            
            result_path = self.processor.process_single_image(
                self.preview_image_path, 
                preview_output_path, 
                self.options
            )
            
            if result_path and os.path.exists(result_path):
                self.show_preview_image(result_path, self.processed_preview_label)
                self.log_text.append("预览生成完成")
            else:
                QMessageBox.warning(self, "错误", "预览生成失败")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成预览时出错: {str(e)}")
            self.log_text.append(f"预览生成失败: {str(e)}")
            
    def start_processing(self):
        """开始处理"""
        input_folder = self.input_folder_edit.text()
        output_folder = self.output_folder_edit.text()
        
        if not input_folder or not os.path.exists(input_folder):
            QMessageBox.warning(self, "警告", "请选择有效的输入文件夹")
            return
            
        if not output_folder:
            QMessageBox.warning(self, "警告", "请选择输出文件夹")
            return
            
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 更新UI状态
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("处理中...")
        self.log_text.clear()
        
        # 创建处理线程
        self.processing_thread = ImageProcessingThread(
            self.processor, input_folder, output_folder, self.options)
        
        # 连接信号
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.log_signal.connect(self.append_log)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        
        # 开始处理
        self.processing_thread.start()
        
    def stop_processing(self):
        """停止处理"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
            
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"处理进度: {value}%")
        
    def append_log(self, message):
        """添加日志"""
        self.log_text.append(message)
        
    def processing_finished(self):
        """处理完成"""
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_label.setText("处理完成")
        
    def load_settings(self):
        """加载设置"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载设置失败: {str(e)}")
                
        # 默认设置
        return {
            'watermark_text': "",
            'brightness_factor': 1.0,
            'contrast_factor': 1.0,
            'saturation_factor': 1.0,
            'sharpness_factor': 1.0,
            'blur_radius': 0.0,
            'resize_factor': 1.0,
            'crop_percentage': 0.0,
            'add_border': False,
            'add_noise': False,
            'mirror_horizontal': False,
            'mirror_vertical': False,
            'remove_metadata': True,
            'change_md5': False,
            # 新增功能选项
            'hue_factor': 0.0,
            'color_temp_factor': 0.0,
            'red_balance': 1.0,
            'green_balance': 1.0,
            'blue_balance': 1.0,
            'convert_to_grayscale': False,
            'grayscale_method': 'weighted',
            'shadow_factor': 0.0,
            'highlight_factor': 0.0,
            'local_brightness_center_x': 0.5,
            'local_brightness_center_y': 0.5,
            'local_brightness_radius': 0.3,
            'local_brightness_factor': 1.0,
            'gradient_type': None,
            'gradient_direction': 0,
            'gradient_intensity': 0.3,
            'exposure_factor': 0.0,
            'smart_sharpen_amount': 0.0,
            'smart_sharpen_radius': 1.0,
            'smart_sharpen_threshold': 0,
            'reduce_noise_strength': 0.0,
            'grain_size': 0.5,
            'grain_strength': 0.0,
            'enhance_details_strength': 0.0,
            'rotation_angle': 0.0,
            'perspective_distortion': 0.0,
            'distortion_type': None,
            'distortion_strength': 0.0,
            'deformation_center_x': 0.5,
            'deformation_center_y': 0.5,
            'deformation_radius': 0.2,
            'deformation_strength': 0.0,
            'vignette_strength': 0.0,
            'chromatic_aberration': 0.0,
            'soft_glow_strength': 0.0,
            'texture_type': None,
            'texture_intensity': 0.0
        }
        
    def save_settings(self):
        """保存设置"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.options, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存设置失败: {str(e)}")
            
    def closeEvent(self, event):
        """关闭事件"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, '确认退出', 
                '正在处理图片，确定要退出吗？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                self.processing_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            # 清理临时目录
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    logger.warning(f"清理临时目录失败: {str(e)}")
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorPreviewGUI()
    window.show()
    sys.exit(app.exec_())