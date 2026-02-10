import subprocess
import sys
from pathlib import Path

SKILLS_ROOT = Path("skills").resolve()
WORKDIR = Path("workdir").resolve()


def run_python_script(skill: str, script: str, args=None):
    args = args or []

    skill_dir = (SKILLS_ROOT / skill / "scripts").resolve()
    script_path = (skill_dir / script).resolve()

    if not str(script_path).startswith(str(skill_dir)):
        return {"ok": False, "error": "Invalid path"}

    if not script_path.exists():
        return {"ok": False, "error": "Script not found"}

    cmd = [sys.executable, str(script_path), *args]

    try:
        p = subprocess.run(
            cmd,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120
        )

        return {
            "ok": p.returncode == 0,
            "stdout": p.stdout[-15000:],
            "stderr": p.stderr[-15000:],
            "code": p.returncode
        }

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Timeout"}
