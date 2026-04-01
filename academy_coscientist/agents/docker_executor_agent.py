# academy_coscientist/agents/docker_executor_agent.py
"""
DockerExecutorAgent — runs arbitrary Python code inside an isolated Docker
container and returns stdout, stderr, and the exit code.

Responsibilities
----------------
- Spin up a fresh, ephemeral container for each code execution request.
- Stream the generated code via stdin (no host filesystem mounts needed).
- Enforce resource limits (memory, CPU) and a wall-clock timeout.
- Return a plain dict with ``stdout``, ``stderr``, and ``returncode`` so
  the caller (ProofOfConceptAgent) can inspect results and decide whether to
  retry or fix the code.

The agent does **not** do any code generation or interpretation; it is a
pure execution sandbox.

Dynamic dependency resolution
------------------------------
Before running code, the agent extracts every ``import`` name and explicit
``pip install`` directive from the script and checks which packages are not
already present in the base image.  If extra packages are needed it builds a
thin Docker image that extends the base image with a single
``RUN pip install …`` layer (network access is allowed only during the build
step) and then runs the experiment in that derived image with
``--network none``.

Derived images are tagged ``<base>-ext-<hash8>`` and cached locally by
Docker, so identical dependency sets are never rebuilt.

Configuration
-------------
All parameters can be set at construction time and overridden at launch via
the ``poc.docker_*`` keys in ``simulator_config.yaml``.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import re
import subprocess
import tempfile
import textwrap
from typing import Any

from academy.agent import Agent, action

from academy_coscientist.utils.utils_logging import log_action, make_struct_logger

# ---------------------------------------------------------------------------
# Packages pre-installed in the default academy-poc-runner base image.
# Used to skip rebuilds when the script only uses already-available libraries.
# ---------------------------------------------------------------------------
_BASE_IMAGE_PACKAGES: frozenset[str] = frozenset({
    # stdlib — always available
    "os", "sys", "re", "json", "math", "time", "random", "itertools",
    "collections", "functools", "pathlib", "io", "csv", "copy", "string",
    "datetime", "hashlib", "struct", "typing", "dataclasses", "abc",
    "contextlib", "traceback", "warnings", "logging", "argparse", "unittest",
    "subprocess", "threading", "multiprocessing", "concurrent", "asyncio",
    "socket", "http", "urllib", "email", "base64", "uuid", "enum", "pprint",
    # third-party packages baked into the base image (keep in sync with docker/Dockerfile)
    "numpy", "np",
    "scipy",
    "pandas", "pd",
    "matplotlib", "plt",
    "seaborn", "sns",
    "sklearn", "scikit_learn", "scikit-learn",
    "statsmodels",
    "sympy",
    "xgboost",
    "lightgbm",
    "pyarrow",
    "h5py",
    "openpyxl",
    "networkx", "nx",
    "skimage", "scikit_image", "scikit-image",
    "PIL", "Pillow",
    "tqdm",
    "joblib",
    "requests",
    "tabulate",
})

# Mapping from common import aliases / top-level names to pip package names.
# Only entries *not* in _BASE_IMAGE_PACKAGES need entries here.
_IMPORT_TO_PIP: dict[str, str] = {
    "cv2":          "opencv-python-headless",
    "PIL":          "Pillow",
    "skimage":      "scikit-image",
    "Bio":          "biopython",
    "pymetis":      "pymetis",
    "igraph":       "python-igraph",
    "graph_tool":   "graph-tool",
    "torch":        "torch",
    "tensorflow":   "tensorflow",
    "tf":           "tensorflow",
    "keras":        "keras",
    "jax":          "jax",
    "xgboost":      "xgboost",
    "lightgbm":     "lightgbm",
    "statsmodels":  "statsmodels",
    "sympy":        "sympy",
    "numba":        "numba",
    "dask":         "dask",
    "h5py":         "h5py",
    "yaml":         "PyYAML",
    "toml":         "tomli",
    "requests":     "requests",
    "httpx":        "httpx",
    "aiohttp":      "aiohttp",
    "fastapi":      "fastapi",
    "flask":        "flask",
    "sqlalchemy":   "sqlalchemy",
    "pymongo":      "pymongo",
    "redis":        "redis",
    "celery":       "celery",
    "pydantic":     "pydantic",
    "attrs":        "attrs",
    "tqdm":         "tqdm",
    "rich":         "rich",
    "click":        "click",
    "typer":        "typer",
    "dotenv":       "python-dotenv",
    "psutil":       "psutil",
    "plotly":       "plotly",
    "seaborn":      "seaborn",
    "bokeh":        "bokeh",
    "altair":       "altair",
    "pyarrow":      "pyarrow",
    "polars":       "polars",
    "numexpr":      "numexpr",
    "lxml":         "lxml",
    "bs4":          "beautifulsoup4",
    "nltk":         "nltk",
    "spacy":        "spacy",
    "gensim":       "gensim",
    "transformers": "transformers",
    "datasets":     "datasets",
    "evaluate":     "evaluate",
    "accelerate":   "accelerate",
    "diffusers":    "diffusers",
    "peft":         "peft",
    "bitsandbytes": "bitsandbytes",
    "einops":       "einops",
    "timm":         "timm",
    "albumentations": "albumentations",
    "mmcv":         "mmcv",
    "open3d":       "open3d",
    "trimesh":      "trimesh",
    "shapely":      "shapely",
    "geopandas":    "geopandas",
    "folium":       "folium",
    "pyproj":       "pyproj",
    "rtree":        "rtree",
    "pyomo":        "pyomo",
    "cvxpy":        "cvxpy",
    "gurobipy":     "gurobipy",
    "pulp":         "PuLP",
    "deap":         "deap",
    "pymoo":        "pymoo",
    "optuna":       "optuna",
    "hyperopt":     "hyperopt",
    "mlflow":       "mlflow",
    "wandb":        "wandb",
    "comet_ml":     "comet-ml",
    "ray":          "ray",
    "joblib":       "joblib",
    "more_itertools": "more-itertools",
    "toolz":        "toolz",
    "cytoolz":      "cytoolz",
    "funcy":        "funcy",
    "boltons":      "boltons",
    "sortedcontainers": "sortedcontainers",
    "bitarray":     "bitarray",
    "pyzmq":        "pyzmq",
    "zmq":          "pyzmq",
    "paramiko":     "paramiko",
    "cryptography": "cryptography",
    "nacl":         "PyNaCl",
    "jwt":          "PyJWT",
    "bcrypt":       "bcrypt",
}


def _extract_packages(code: str) -> list[str]:
    """
    Return a deduplicated list of pip package names that the script requires
    but that are NOT already in the base image.

    Heuristic: parse the AST for ``import X`` / ``from X import …`` and
    scan for ``pip install …`` shell calls in comments or subprocess calls.
    """
    needed: set[str] = set()

    # --- AST-based import extraction ---
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    needed.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    needed.add(top)
    except SyntaxError:
        # Fall back to regex if the code isn't valid Python yet
        for m in re.finditer(r"^\s*(?:import|from)\s+([\w]+)", code, re.MULTILINE):
            needed.add(m.group(1))

    # --- Explicit pip install directives (comments or subprocess calls) ---
    # Matches: # pip install foo bar, subprocess("pip install foo"), os.system("pip install foo")
    for m in re.finditer(
        r"pip\s+install\s+((?:[\w\-\.]+\s*)+)", code, re.IGNORECASE
    ):
        for pkg in m.group(1).split():
            pkg = pkg.strip()
            if pkg and not pkg.startswith("-"):
                # Normalise: convert to import name for dedup check, keep pip name
                needed.add(pkg.replace("-", "_").split("==")[0].split(">=")[0])

    # Remove packages that are already in the base image
    extra_imports = needed - _BASE_IMAGE_PACKAGES

    # Map import names → pip names; keep only those we know about
    pip_packages: list[str] = []
    seen: set[str] = set()
    for imp in sorted(extra_imports):
        pip_name = _IMPORT_TO_PIP.get(imp, imp)  # fall back to import name
        if pip_name not in seen:
            pip_packages.append(pip_name)
            seen.add(pip_name)

    return pip_packages


def _derived_image_tag(base_image: str, packages: list[str]) -> str:
    """Stable tag for a derived image with a specific set of extra packages."""
    pkg_hash = hashlib.md5(
        " ".join(sorted(packages)).encode()
    ).hexdigest()[:8]
    # Sanitise base image name for use as a Docker tag prefix
    safe_base = re.sub(r"[^a-z0-9]", "-", base_image.lower()).strip("-")
    return f"{safe_base}-ext-{pkg_hash}"


def _image_exists_locally(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0


def _build_derived_image(base_image: str, packages: list[str], tag: str) -> tuple[bool, str]:
    """
    Build a Docker image extending *base_image* with *packages* installed.

    Returns (success, stderr_output).
    Network access is allowed during the build so pip can reach PyPI.
    """
    dockerfile_content = textwrap.dedent(f"""\
        FROM {base_image}
        RUN pip install --no-cache-dir {' '.join(packages)}
    """)
    with tempfile.TemporaryDirectory() as ctx:
        dockerfile_path = f"{ctx}/Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        result = subprocess.run(
            ["docker", "build", "-t", tag, ctx],
            capture_output=True,
            timeout=300,  # 5 min build limit
        )
    return result.returncode == 0, result.stderr.decode("utf-8", errors="replace")


class DockerExecutorAgent(Agent):
    """
    Executes Python code inside a Docker container.

    Parameters
    ----------
    image : str
        Base Docker image.  Must have a Python interpreter available as
        ``python``.  Default: ``python:3.11-slim``.
    timeout : int
        Wall-clock limit in seconds for each execution.  Default: 90.
    memory_limit : str
        Docker ``--memory`` flag value (e.g. ``"512m"``, ``"1g"``).
    network : str
        Docker ``--network`` value used for the *experiment* container.
        Use ``"none"`` (default) for full network isolation.
        The image-build step always uses bridge networking so pip can reach
        PyPI, regardless of this setting.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 90,
        memory_limit: str = "512m",
        network: str = "none",
    ) -> None:
        super().__init__()
        self.logger = make_struct_logger("DockerExecutorAgent")
        self._image = str(image)
        self._timeout = max(10, int(timeout))
        self._memory_limit = str(memory_limit)
        self._network = str(network)

    # ------------------------------------------------------------------
    # Public actions
    # ------------------------------------------------------------------

    @action
    async def run_code(self, code: str) -> dict[str, Any]:
        """
        Execute *code* inside a fresh Docker container.

        Before running, extra packages imported by the script are detected
        and a derived Docker image is built (if not already cached) that has
        those packages pre-installed.  The experiment itself runs with
        ``--network none`` regardless of which packages were needed.

        The code is passed to the container via stdin (``python -``), so no
        host file-system mounts are needed.

        Returns
        -------
        dict with keys:
            ``stdout``     — captured standard output (str)
            ``stderr``     — captured standard error (str)
            ``returncode`` — process exit code (int; 0 = success)
        """
        loop = asyncio.get_running_loop()

        # Resolve which image to use (may build a derived one)
        run_image, extra_pkgs = await loop.run_in_executor(
            None, self._resolve_image, code
        )

        log_action(
            self.logger,
            "run_code_start",
            {
                "image": run_image,
                "base_image": self._image,
                "extra_packages": extra_pkgs,
                "code_len": len(code),
            },
            {},
        )

        result = await loop.run_in_executor(None, self._execute_sync, code, run_image)

        log_action(
            self.logger,
            "run_code_done",
            {"image": run_image},
            {
                "returncode": result["returncode"],
                "stdout_len": len(result["stdout"]),
                "stderr_len": len(result["stderr"]),
            },
        )
        return result

    @action
    async def is_available(self) -> bool:
        """
        Return True if the Docker daemon is reachable and the configured
        image exists locally (or can be pulled).

        Runs ``docker version`` as a quick liveness check.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._check_docker)

    # ------------------------------------------------------------------
    # Internal helpers (run in thread-pool executor)
    # ------------------------------------------------------------------

    def _resolve_image(self, code: str) -> tuple[str, list[str]]:
        """
        Return ``(image_tag, extra_packages)`` to use for running *code*.

        If the script needs packages not in the base image, a derived image
        is built (or reused from the local Docker cache).
        """
        extra_pkgs = _extract_packages(code)

        if not extra_pkgs:
            return self._image, []

        tag = _derived_image_tag(self._image, extra_pkgs)

        if _image_exists_locally(tag):
            self.logger.info(
                "docker_image_cache_hit: tag=%s packages=%s", tag, extra_pkgs
            )
            log_action(
                self.logger,
                "docker_image_cache_hit",
                {"tag": tag, "packages": extra_pkgs},
                {"built": False},
            )
            return tag, extra_pkgs

        self.logger.info(
            "docker_image_build_start: tag=%s packages=%s", tag, extra_pkgs
        )
        log_action(
            self.logger,
            "docker_image_build_start",
            {"base_image": self._image, "tag": tag, "packages": extra_pkgs},
            {},
        )

        success, build_stderr = _build_derived_image(self._image, extra_pkgs, tag)

        log_action(
            self.logger,
            "docker_image_build_done",
            {"tag": tag, "packages": extra_pkgs},
            {"success": success, "build_stderr": build_stderr},
        )

        if not success:
            self.logger.warning(
                "docker_image_build_failed: tag=%s stderr=%s", tag, build_stderr[:500]
            )
            # Fall back to base image — the script may still work if some
            # packages are optional or already present despite the error.
            return self._image, extra_pkgs

        return tag, extra_pkgs

    def _build_cmd(self, image: str) -> list[str]:
        return [
            "docker", "run",
            "--rm",                        # auto-remove container on exit
            "--interactive",               # accept stdin
            "--network", self._network,    # network isolation for experiment
            "--memory", self._memory_limit,
            "--cpus", "1",
            "--stop-timeout", str(self._timeout + 5),  # hard kill after grace period
            image,
            "python", "-",                 # read code from stdin
        ]

    def _execute_sync(self, code: str, image: str) -> dict[str, Any]:
        """Blocking execution; called from a thread-pool executor."""
        cmd = self._build_cmd(image)
        print(
            f"[DockerExecutor] Running in {image} "
            f"(network={self._network}, mem={self._memory_limit}, "
            f"timeout={self._timeout}s)",
            flush=True,
        )
        try:
            proc = subprocess.run(
                cmd,
                input=code.encode("utf-8"),
                capture_output=True,
                timeout=self._timeout,
            )
            return {
                "stdout": proc.stdout.decode("utf-8", errors="replace"),
                "stderr": proc.stderr.decode("utf-8", errors="replace"),
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {self._timeout}s",
                "returncode": 1,
            }
        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": (
                    "Docker executable not found. "
                    "Is Docker installed and on PATH?"
                ),
                "returncode": 127,
            }
        except Exception as exc:
            return {
                "stdout": "",
                "stderr": f"Docker execution error: {exc!r}",
                "returncode": 1,
            }

    def _check_docker(self) -> bool:
        """Return True if ``docker version`` exits 0."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False
