"""
一键启动整合后的服务（API + 前端静态文件，单进程）。

用法：
    python scripts/start_api.py            # 直接启动（需已构建前端）
    python scripts/start_api.py --build    # 先构建前端再启动
    python scripts/start_api.py --port 9000

启动后访问 http://localhost:8000 即为完整应用（聊天界面 + API + /docs）。
"""

import sys
import argparse
import subprocess
from pathlib import Path

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from config.settings import settings  # noqa: E402


def build_frontend() -> bool:
    """构建 Vue 前端为静态文件"""
    frontend_dir = PROJECT_ROOT / "frontend"
    if not (frontend_dir / "package.json").exists():
        print("[WARN] 未找到 frontend/package.json，跳过前端构建。")
        return False

    print("[BUILD] 构建前端 (npm run build) ...")
    try:
        subprocess.run(["npm", "run", "build"], cwd=str(frontend_dir), check=True)
        print("[BUILD] ✅ 前端构建完成 -> frontend/dist/")
        return True
    except FileNotFoundError:
        print("[ERROR] 未找到 npm，请先安装 Node.js (建议 18+)。")
        return False
    except subprocess.CalledProcessError:
        print("[ERROR] 前端构建失败，请查看上方 npm 输出。")
        return False


def main():
    parser = argparse.ArgumentParser(description="启动整合服务（API + 前端单进程）")
    parser.add_argument("--build", action="store_true", help="启动前先构建前端")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=settings.api_port, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    dist_dir = PROJECT_ROOT / "frontend" / "dist"

    if args.build or not (dist_dir / "index.html").exists():
        if not (dist_dir / "index.html").exists() and not args.build:
            print("[INFO] 未检测到 frontend/dist，自动构建前端 ...")
        build_frontend()

    if not (dist_dir / "index.html").exists():
        print("[WARN] frontend/dist 不存在，将以「仅 API」模式启动（访问 /docs 查看接口）。")

    print("=" * 60)
    print(" 智能医疗问诊助手 · 整合服务（单进程）")
    print("=" * 60)
    print(f"  地址: http://localhost:{args.port}")
    print(f"  界面: http://localhost:{args.port}/        （Vue 聊天界面）")
    print(f"  文档: http://localhost:{args.port}/docs    （Swagger UI）")
    print(f"  接口: http://localhost:{args.port}/api/consult")
    print("=" * 60)

    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
