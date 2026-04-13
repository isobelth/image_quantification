"""Utility helpers shared across cell-death analysis modules."""

import os
import sys
from pathlib import Path


def running_in_notebook():
    """Return True if the code is executing inside a Jupyter notebook."""
    if "ipykernel" in sys.modules:
        return True
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


def configure_java_home():
    """Set JAVA_HOME from jdk4py or conda prefix if not already set.

    Searches for a JVM shared library in known locations and configures
    the environment variables JAVA_HOME and PATH so that ImageJ / scyjava
    can find a working JDK at runtime.

    Returns
    -------
    str or None
        The resolved JAVA_HOME path, or None if no JDK was found.
    """
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    candidates = []
    try:
        import jdk4py
        candidates.append(str(jdk4py.JAVA_HOME))
    except Exception:
        pass
    candidates.append(str(Path(sys.prefix) / "Library"))
    for home in candidates:
        if (Path(home) / "bin" / "server" / "jvm.dll").exists():
            os.environ["JAVA_HOME"] = str(home)
            os.environ["PATH"] = (
                str(Path(home) / "bin") + os.pathsep + os.environ.get("PATH", "")
            )
            return str(home)
    return None
