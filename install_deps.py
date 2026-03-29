"""
安装评估所需的依赖包
Install dependencies for evaluation

使用方法:
    python setup_dependencies.py
"""

import subprocess
import sys
import io
import importlib.util

# 修复Windows控制台编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def install_package(package):
    """安装单个包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"[OK] {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {package} 安装失败: {e}")
        return False


def main():
    """安装所有必需的依赖"""
    print("=" * 60)
    print("安装评估依赖包")
    print("=" * 60)

    # 必需的包
    packages = [
        "pesq==0.0.4",
        "pystoi==0.4.1",
        "scipy",
        "librosa",
        "soundfile",
        "resampy",
        "numpy",
    ]

    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1

    print("-" * 60)
    print(f"安装完成: {success_count}/{len(packages)}")

    # 验证安装
    print("\n验证安装...")
    if importlib.util.find_spec("pesq") is not None:
        print("[OK] pesq 可用")
    else:
        print("[FAIL] pesq 不可用，可能需要重新安装")

    if importlib.util.find_spec("pystoi") is not None:
        print("[OK] pystoi 可用")
    else:
        print("[FAIL] pystoi 不可用，可能需要重新安装")


if __name__ == "__main__":
    main()
