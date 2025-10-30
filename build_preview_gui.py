import sys
import os
import PyInstaller.__main__

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义要打包的主文件
main_script = os.path.join(current_dir, 'image_processor_gui_preview.py')

# 定义其他需要包含的文件
datas = [
    (os.path.join(current_dir, 'image_processor.py'), '.'),
    (os.path.join(current_dir, 'image_processor_settings.json'), '.'),
]

# 构建PyInstaller参数
pyinstaller_args = [
    main_script,  # 主程序文件
    '--name=图片处理器预览版',  # 可执行文件名称
    '--windowed',  # Windows下禁用控制台窗口
    '--onefile',  # 打包成单个exe文件
    '--clean',  # 清理临时文件
]

# 添加数据文件
for src, dst in datas:
    if os.path.exists(src):
        pyinstaller_args.append(f'--add-data={src};{dst}')

# 如果有图标文件，也添加进去
icon_path = os.path.join(current_dir, 'icon.ico')
if os.path.exists(icon_path):
    pyinstaller_args.append(f'--icon={icon_path}')

# 运行PyInstaller
if __name__ == '__main__':
    PyInstaller.__main__.run(pyinstaller_args)