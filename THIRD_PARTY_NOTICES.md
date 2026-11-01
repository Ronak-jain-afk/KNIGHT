# Third-Party Notices

Knight is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
This file lists third-party components used by the project, their copyright
holders, and their licenses. It is provided for attribution and compliance; it
does not modify the Apache-2.0 license for Knight's own source code.

## Vendored browser assets

The following files are committed to this repository under
`app/web/vendor/` and are redistributed with Knight.

### marked 15.0.7

- File: `app/web/vendor/marked.min.js`
- License: MIT
- Copyright (c) 2011-2025, Christopher Jeffrey
- Upstream: <https://github.com/markedjs/marked>

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### DOMPurify 3.2.4

- File: `app/web/vendor/purify.min.js`
- License: Apache-2.0 OR MPL-2.0 (used here under Apache-2.0)
- Copyright (c) Cure53 and other contributors
- Upstream: <https://github.com/cure53/DOMPurify>

The file retains its upstream license header:

```text
@license DOMPurify 3.2.4 | (c) Cure53 and other contributors |
Released under the Apache license 2.0 and Mozilla Public License 2.0 |
github.com/cure53/DOMPurify/blob/3.2.4/LICENSE
```

## Runtime components

These components are not vendored in the repository; they are installed or
downloaded by the user at runtime. Their licenses apply to their respective
files and model weights.

| Component | License | Notes |
|---|---|---|
| llama.cpp build b11980 | MIT | Downloaded by `scripts/download_llama.py`; binaries are not committed. |
| Qwen3.5-4B GGUF | Apache-2.0 | Model weights downloaded at runtime; not committed. |
| Whistle (Cactus Compute) | Apache-2.0 | On-device ASR model downloaded at runtime; not committed. |
| BAAI/bge-small-en-v1.5 | MIT | Embedding model downloaded by `fastembed`; not committed. |

If you redistribute a bundle that includes `vendor/llama/`, that bundle also
contains NVIDIA CUDA redistributable libraries (`libcublas`, `libcudart`, and
related files). Those libraries are governed by NVIDIA's own license terms and
are not covered by Knight's Apache-2.0 license. The repository's `.gitignore`
excludes `vendor/llama/` for this reason.

## Python dependencies

The direct runtime dependencies declared in `pyproject.toml` are:

| Package | License |
|---|---|
| fastapi | MIT |
| uvicorn | BSD-3-Clause |
| pydantic-settings | MIT |
| httpx | BSD-3-Clause |
| langchain | MIT |
| langchain-openai | MIT |
| langgraph | MIT |
| langgraph-checkpoint-sqlite | MIT |
| aiosqlite | MIT |
| ddgs | MIT |
| trafilatura | Apache-2.0 |
| fastembed | Apache-2.0 |
| huggingface-hub | Apache-2.0 |
| numpy | BSD-3-Clause (with bundled components under 0BSD, MIT, Zlib, and CC0-1.0) |
| python-multipart | Apache-2.0 |

Optional and development dependencies:

| Package | License |
|---|---|
| cactus-needle | Apache-2.0 |
| pytest | MIT |
| pytest-asyncio | Apache-2.0 |
| respx | BSD-3-Clause |
| ruff | MIT |
| playwright | Apache-2.0 |

Transitive dependencies are pinned in `uv.lock`. Each installed package ships
its own license metadata; consult the package distribution or its upstream
repository for the exact copyright notice and license text. When distributing
a built artifact, include the license texts required by those packages.

## Model and dataset attribution

Knight does not redistribute model weights. Model cards and licenses for the
default and optional models are available from their publishers:

- Qwen3.5-4B: <https://huggingface.co/Qwen>
- Whistle: <https://huggingface.co/Cactus-Compute/whistle>
- BAAI/bge-small-en-v1.5: <https://huggingface.co/BAAI/bge-small-en-v1.5>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
