from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def get_path_from_root(path) -> str:
    return str(Path.joinpath(get_project_root(),path))
