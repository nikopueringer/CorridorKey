This project uses **[uv](https://docs.astral.sh/uv/)** to manage Python and all
dependencies. uv is a fast, modern replacement for pip that automatically
handles Python versions, virtual environments, and package installation in a
single step. You do **not** need to install Python yourself — uv does it for
you.

Install uv if you don't already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

!!! tip
    On Windows the automated `.bat` installers handle uv installation for you.
    If you open a new terminal after installing uv and see `'uv' is not
    recognized`, close and reopen the terminal so the updated PATH takes effect.
