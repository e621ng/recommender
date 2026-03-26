#!/usr/bin/env python
"""Bump the project version and create a git tag."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def read_version() -> str:
    """Read the current version from recommender/__init__.py."""
    init_file = Path(__file__).parent.parent / "recommender" / "__init__.py"
    with open(init_file) as f:
        content = f.read()

    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError(f"Could not find __version__ in {init_file}")

    return match.group(1)


def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse semantic version string into (major, minor, patch)."""
    parts = version_str.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version_str}. Expected X.Y.Z")

    try:
        return tuple(int(p) for p in parts)  # type: ignore
    except ValueError:
        raise ValueError(f"Invalid version format: {version_str}. Expected X.Y.Z")


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version tuple into string."""
    return f"{major}.{minor}.{patch}"


def bump_version(current_version: str, bump_type: str) -> str:
    """Bump version according to SemVer rules."""
    major, minor, patch = parse_version(current_version)

    if bump_type == "major":
        return format_version(major + 1, 0, 0)
    elif bump_type == "minor":
        return format_version(major, minor + 1, 0)
    elif bump_type == "patch":
        return format_version(major, minor, patch + 1)
    else:
        # Allow explicit version
        parse_version(bump_type)  # Validate format
        return bump_type


def write_version(version_str: str) -> None:
    """Write the version to recommender/__init__.py."""
    init_file = Path(__file__).parent.parent / "recommender" / "__init__.py"
    with open(init_file) as f:
        content = f.read()

    new_content = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{version_str}"',
        content
    )

    if new_content == content:
        raise ValueError(f"Could not update version in {init_file}")

    with open(init_file, "w") as f:
        f.write(new_content)


def run_git_commit(version: str) -> None:
    """Commit the version change."""
    try:
        subprocess.run(
            ["git", "add", "recommender/__init__.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"[Release] Version {version}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git commit failed: {e.stderr}")


def run_git_tag(version: str) -> None:
    """Create an annotated git tag."""
    tag_name = f"{version}"
    try:
        subprocess.run(
            [
                "git",
                "tag",
                "-a",
                tag_name,
                "-m",
                f"Release version {version}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git tag failed: {e.stderr}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bump the project version and create a git tag."
    )
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        nargs="?",
        default=None,
        help="Version component to bump: major, minor, or patch",
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="Print the current version and exit",
    )

    args = parser.parse_args()

    try:
        current_version = read_version()

        if args.current:
            print(current_version)
            return

        if args.bump_type is None:
            parser.print_help()
            sys.exit(1)

        new_version = bump_version(current_version, args.bump_type)

        if new_version == current_version:
            print(f"Version is already {current_version}")
            return

        print(f"Bumping version: {current_version} -> {new_version}")

        write_version(new_version)
        print(f"Updated recommender/__init__.py")

        run_git_commit(new_version)
        print(f"Created commit: [Release] Version {new_version}")

        run_git_tag(new_version)
        print(f"Created tag: {new_version}")

        print("✓ Version bumped successfully!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
