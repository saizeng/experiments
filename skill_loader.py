from pathlib import Path


def load_skill_registry(root="skills"):
    root = Path(root)
    docs = []

    for skill_dir in root.iterdir():
        md = skill_dir / "SKILL.md"
        if md.exists():
            docs.append(f"# Skill: {skill_dir.name}\n\n{md.read_text()}")

    return "\n\n".join(docs)
