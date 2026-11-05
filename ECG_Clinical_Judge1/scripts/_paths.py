"""Keep user inputs and generated artifacts outside publishable source trees."""
from pathlib import Path


def private_output(path):
    root=Path(__file__).resolve().parents[1]
    resolved=path.resolve()
    if resolved.is_relative_to(root) and not any(resolved.is_relative_to(root/name) for name in ['outputs','artifacts','private']):
        raise ValueError('Local generated files must be outside the repository or under outputs/, artifacts/ or private/')
    return resolved
