# AI-Assisted Development

CorridorKey ships two files designed to give AI coding assistants deep
context about the project:

- **`AGENTS.md`** — a structured project guide at the repo root.
- **`docs/LLM_HANDOVER.md`** — a detailed architecture walkthrough and
  AI directive reference.

Together these files cover the codebase layout, dataflow rules, dev
commands, code style, and common pitfalls. Most AI tools can consume
them directly — the sections below explain what each file provides and
how tools discover them.

---

## Context Sources

### `AGENTS.md`

`AGENTS.md` sits at the repository root and follows the open
[AGENTS.md standard](https://agents.md). It gives any AI assistant a
compact overview of the project: architecture summary, key file map,
build and test commands, code style settings, platform caveats, and
prohibited actions.

Because the format is an open standard, multiple AI coding tools —
including GitHub Copilot, Windsurf, and Kiro — read it natively
without extra configuration. Dropping into the repo and opening a
session is often enough to get useful context.

!!! tip "Read the source"
    The full file is at the repo root:
    [`AGENTS.md`](../AGENTS.md).
    Refer to it directly rather than relying on summaries here.

### `LLM_HANDOVER.md`

`LLM_HANDOVER.md` lives in the `docs/` directory and provides a much
deeper technical handover. It covers the GreenFormer architecture in
detail, critical dataflow properties (color space and gamma math),
inference pipeline internals, and AI-specific directives for working
with the codebase.

If `AGENTS.md` is the quick-reference card, `LLM_HANDOVER.md` is the
full briefing document. Point your assistant at it when you need help
with inference code, compositing math, or EXR pipeline work.

!!! tip "Read the source"
    The full handover document is at
    [`docs/LLM_HANDOVER.md`](LLM_HANDOVER.md).
    It is the authoritative deep-dive — this page only summarises
    what it contains.

---

## Quick Start

Get a working dev environment and point your AI assistant at the
project context — this works with any tool.

```bash
git clone https://github.com/nikopueringer/CorridorKey.git
cd CorridorKey
uv sync --group dev    # installs all dependencies + dev tools
```

Once the repo is cloned, open `AGENTS.md` in your AI assistant as
the first step. It gives the assistant the project layout, key
rules, and common pitfalls in one read. For deeper architecture
context, also point it at `docs/LLM_HANDOVER.md`.

Core dev commands to keep handy:

```bash
uv run pytest                # run all tests
uv run ruff check            # check for lint errors
uv run ruff format --check   # check formatting (no changes)
```

---

## Tool Configuration

Each AI coding assistant has its own way of loading project context.
Pick your tool below for CorridorKey-specific setup instructions.

=== "Kiro"

    Kiro uses **steering files** stored in `.kiro/steering/*.md` to
    provide persistent project context. Each file is a Markdown
    document that Kiro loads according to one of three inclusion
    modes:

    | Mode | Behaviour |
    |------|-----------|
    | **Always-on** (default) | Loaded at the start of every session automatically. |
    | **Conditional** | Loaded only when the active file matches a `fileMatch` glob pattern (e.g., `*.py`). |
    | **Manual** | User provides the file explicitly via `#` in the chat prompt. |

    To give Kiro the full CorridorKey context, create a steering file
    that references both `AGENTS.md` and `LLM_HANDOVER.md`:

    ```markdown title=".kiro/steering/corridorkey-context.md"
    # CorridorKey Project Context

    This steering file gives Kiro persistent context about the
    CorridorKey codebase.

    ## Primary References

    - Read `AGENTS.md` at the repo root for the project overview,
      key file map, build commands, code style, and prohibited
      actions.
    - Read `docs/LLM_HANDOVER.md` for the deep architecture
      walkthrough, dataflow rules, and AI-specific directives.

    ## Key Rules

    - Tensor range is strictly [0.0, 1.0] float.
    - Never use pow(x, 2.2) for gamma — use piecewise sRGB
      transfer functions in `color_utils.py`.
    - Do not modify files in `gvm_core/` or
      `VideoMaMaInferenceModule/`.
    ```

=== "Claude Code"

    Claude Code loads a **`CLAUDE.md`** file from the repository root
    automatically at the start of every session. This is the primary
    way to give Claude Code persistent project context.

    Create a `CLAUDE.md` that points Claude Code at the existing
    context files:

    ```markdown title="CLAUDE.md"
    # CorridorKey — Claude Code Context

    Read these files for full project context:

    - `AGENTS.md` — project overview, key file map, build/test
      commands, code style, prohibited actions.
    - `docs/LLM_HANDOVER.md` — deep architecture walkthrough,
      dataflow rules, inference pipeline, AI directives.

    Key rules:
    - Tensors are [0.0, 1.0] float. Foreground is sRGB, alpha is
      linear.
    - Use piecewise sRGB transfer functions, never pow(x, 2.2).
    - Do not modify gvm_core/ or VideoMaMaInferenceModule/.
    ```

    Claude Code reads `CLAUDE.md` once at session start, so any
    updates require restarting the session to take effect.

=== "Cursor"

    Cursor uses **rule files** stored in `.cursor/rules/*.md` to
    inject project context into the assistant. Each rule file
    supports a frontmatter block that controls when it activates:

    | Mode | Frontmatter | Behaviour |
    |------|-------------|-----------|
    | **Always-on** | `alwaysApply: true` | Loaded in every chat and Cmd-K session. |
    | **Glob-based** | `globs: ["*.py"]` | Loaded when the active file matches the pattern. |
    | **Manual** | `manual: true` | User includes it explicitly via `@rules`. |
    | **Model-decision** | `agentRequested: true` | The model decides whether to load it based on the task description. |

    Example rule file for CorridorKey:

    ```markdown title=".cursor/rules/corridorkey.md"
    ---
    description: CorridorKey project context and coding rules
    alwaysApply: true
    ---

    # CorridorKey Context

    Read `AGENTS.md` at the repo root for the project overview,
    key file map, and build commands.

    Read `docs/LLM_HANDOVER.md` for the deep architecture
    walkthrough and dataflow rules.

    Key rules:
    - Tensors are [0.0, 1.0] float. Foreground sRGB, alpha linear.
    - Use piecewise sRGB functions, never pow(x, 2.2).
    - Do not modify gvm_core/ or VideoMaMaInferenceModule/.
    ```

=== "GitHub Copilot"

    GitHub Copilot supports project-level instructions via a
    **`.github/copilot-instructions.md`** file. This file is
    automatically included in Copilot Chat requests to provide
    project-specific guidance.

    Copilot also reads **`AGENTS.md`** natively, so CorridorKey's
    existing `AGENTS.md` already provides baseline context without
    any extra configuration. The instructions file is useful for
    adding Copilot-specific guidance beyond what `AGENTS.md` covers.

    ```markdown title=".github/copilot-instructions.md"
    # CorridorKey — Copilot Instructions

    This project already has an `AGENTS.md` at the repo root that
    Copilot reads automatically. For deeper context, also refer to
    `docs/LLM_HANDOVER.md`.

    Key rules:
    - Tensors are [0.0, 1.0] float. Foreground sRGB, alpha linear.
    - Use piecewise sRGB functions in color_utils.py, never
      pow(x, 2.2).
    - Do not modify gvm_core/ or VideoMaMaInferenceModule/.
    ```

=== "Windsurf"

    Windsurf uses **`.windsurf/rules/`** for project-level context
    files. Rules placed in this directory are loaded automatically
    during coding sessions.

    Windsurf also reads **`AGENTS.md`** files natively with
    directory-based auto-scoping — it picks up `AGENTS.md` at the
    repo root and applies its content as project-wide context. This
    means CorridorKey's existing `AGENTS.md` works out of the box.

    For additional Windsurf-specific rules, create a file in the
    rules directory:

    ```markdown title=".windsurf/rules/corridorkey.md"
    # CorridorKey Context

    AGENTS.md at the repo root is loaded automatically. For the
    deep architecture walkthrough, also read
    docs/LLM_HANDOVER.md.

    Key rules:
    - Tensors are [0.0, 1.0] float. Foreground sRGB, alpha linear.
    - Use piecewise sRGB functions, never pow(x, 2.2).
    - Do not modify gvm_core/ or VideoMaMaInferenceModule/.
    ```

=== "Gemini CLI"

    Gemini CLI uses a **`GEMINI.md`** file at the repository root
    for project-level context. It supports hierarchical context
    loading across three levels:

    | Level | Location | Scope |
    |-------|----------|-------|
    | **Global** | `~/.gemini/GEMINI.md` | Applied to all projects on the machine. |
    | **Project** | `GEMINI.md` (repo root) | Applied to the current project. |
    | **Subdirectory** | `GEMINI.md` in any subdirectory | Applied when working within that directory. |

    Gemini CLI merges context from all three levels, with more
    specific files taking precedence. For CorridorKey, a project-level
    file is sufficient:

    ```markdown title="GEMINI.md"
    # CorridorKey — Gemini CLI Context

    Read these files for full project context:

    - `AGENTS.md` — project overview, key file map, build/test
      commands, code style, prohibited actions.
    - `docs/LLM_HANDOVER.md` — deep architecture walkthrough,
      dataflow rules, inference pipeline, AI directives.

    Key rules:
    - Tensors are [0.0, 1.0] float. Foreground sRGB, alpha linear.
    - Use piecewise sRGB functions, never pow(x, 2.2).
    - Do not modify gvm_core/ or VideoMaMaInferenceModule/.
    ```

---

## Community Contributions

PRs adding configuration guides for AI tools not yet covered here
are welcome. See the [Contributing](contributing.md) page for the
PR workflow and submission guidelines.
