**Download the CorridorKey checkpoint (~300 MB):**

[Download CorridorKey_v1.0.pth from Hugging Face](https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth){ .md-button }

Place the file inside `CorridorKeyModule/checkpoints/` and rename it to
**`CorridorKey.pth`** so the final path is:

```
CorridorKeyModule/checkpoints/CorridorKey.pth
```

!!! warning
    The engine will not start without this checkpoint. Make sure the filename
    is exactly `CorridorKey.pth` (not `CorridorKey_v1.0.pth`).
