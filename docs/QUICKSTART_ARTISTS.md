# CorridorKey Quickstart for Artists

**No coding experience required.** This guide will get you from zero to pulling your first key in under 30 minutes.

---

## What You'll Need

- A computer with an **NVIDIA GPU** that has **24 GB of VRAM** or more (RTX 3090, 4090, 5090, or similar)
- **Green screen footage** (video file or image sequence)
- An internet connection (for the one-time download)

Don't know how much VRAM your GPU has? Open Task Manager (Windows) or `nvidia-smi` (Terminal), look for your GPU, and check the "Dedicated GPU memory" value.

---

## Step 1: Install Prerequisites

CorridorKey uses **[uv](https://docs.astral.sh/uv/)** to manage Python and all dependencies automatically — you do **not** need to install Python yourself.

**Windows users:** You can skip this step entirely. The installer in Step 3 handles everything, including installing uv.

**Mac / Linux users:** Open a terminal and run:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then close and reopen your terminal so the `uv` command is available.

---

## Step 2: Download CorridorKey

### Option A: Download as ZIP (Easiest)
1. Go to [github.com/nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey)
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Unzip it somewhere easy to find (like your Desktop or Documents folder)

### Option B: Clone with Git (If You Have Git)
```
git clone https://github.com/nikopueringer/CorridorKey.git
```

---

## Step 3: Run the Installer

### Windows
1. Open the CorridorKey folder
2. Double-click **`Install_CorridorKey_Windows.bat`**
3. Wait for it to finish (it will download dependencies and the AI model — about 300 MB)
4. When you see "Setup Complete!", you're ready

### Mac / Linux
Open a terminal, navigate to the CorridorKey folder, and run:
```bash
uv sync
```
This downloads Python (if needed) and installs all dependencies automatically.

Then download the model file (~300 MB):
```bash
mkdir -p CorridorKeyModule/checkpoints
curl -L -o CorridorKeyModule/checkpoints/CorridorKey.pth \
  https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth
```

---

## Step 4: Prepare Your Footage

You have two options for getting alpha hints (the rough mask that tells CorridorKey where your subject is):

### Option A: Make Your Own Alpha Hint (Recommended for Most Users)

If you already have a compositing tool (After Effects, Resolve, Nuke), pull a rough key:

1. **In After Effects:** Apply Keylight or the built-in chroma keyer. Export the matte as a PNG sequence. It doesn't need to be perfect — rough is fine!
2. **In DaVinci Resolve:** Use the Qualifier tool to pull a quick key. Export the alpha channel.
3. **In Nuke:** Use a simple Keyer node. Export the matte.
4. **Using AI Roto:** Tools like Runway, SAM, or Rotobrush can generate a rough mask.

The mask should be:
- **White** where your subject is
- **Black** where the green screen is
- It's OK if the edges are rough, blurry, or slightly wrong — CorridorKey will fix them

### Option B: Let CorridorKey Generate One (Requires More VRAM)

If you installed the optional GVM or VideoMaMa modules, the wizard can generate alpha hints automatically. See the [Optional Alpha Generators](#optional-alpha-generators) section below.

---

## Step 5: Organize Your Shot Folder

Create a folder structure like this:

```
MyShot/
├── Input/           ← Put your green screen frames here
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
└── AlphaHint/       ← Put your rough mask frames here
    ├── frame_001.png
    ├── frame_002.png
    └── ...
```

**Or**, if you have a video file, just put it in a folder. The wizard will organize it for you.

**Important:** The Input and AlphaHint folders must have the **same number of frames**.

---

## Step 6: Run CorridorKey

### Windows
Drag your shot folder (or video file) onto **`CorridorKey_DRAG_CLIPS_HERE_local.bat`**

> Don't double-click the .bat file directly — you need to drag something onto it.

### Mac / Linux
```bash
./CorridorKey_DRAG_CLIPS_HERE_local.sh /path/to/MyShot
```

### The Wizard Prompts

The wizard will ask you a few questions. Here's what they mean in plain language:

**"Is the input sequence Linear or sRGB?"**
- If your frames came from a video file (MP4, MOV) → choose **sRGB** (press `s`)
- If your frames are from a 3D render or are EXR files → choose **Linear** (press `l`)
- **When in doubt, choose sRGB** — it's correct for most footage

**"Enter Despill Strength (0-10)"**
- This removes the green tint that bleeds onto your subject's edges
- **10** = maximum green removal (default, usually best)
- **0** = no green removal (if you want to handle it yourself in your comp)

**"Enable Auto-Despeckle?"**
- This removes tiny floating specks (like tracking markers) from the matte
- **Y** (yes) is almost always correct
- If prompted for size, **400** is a good default

**"Enter Refiner Strength"**
- Just press Enter to accept the default (1.0)
- This is an experimental setting — leave it alone unless you're told otherwise

---

## Step 7: Use Your Output

After processing, you'll find a new `Output` folder inside your shot:

```
MyShot/
└── Output/
    ├── Processed/    ← Drop this into your editor for instant results
    ├── Matte/        ← The alpha channel by itself
    ├── FG/           ← The foreground color by itself
    └── Comp/         ← Preview images (composite over checkerboard)
```

### For Quick Results (After Effects, Premiere, Resolve)

Import the **`Processed/*.exr`** sequence. This is a ready-to-use RGBA file — the foreground is already cleanly separated with a proper alpha channel. Drop it on your timeline over any background.

### For Maximum Quality (Nuke, Fusion, Advanced Compositing)

Import **`FG/*.exr`** and **`Matte/*.exr`** separately. This gives you full control:

1. **Important:** The FG pass needs a color space conversion (sRGB → Linear) before you combine it with the matte. In Nuke, add a Colorspace node. In Fusion, use a Color Space Transform.
2. Set the matte as your alpha channel
3. Premultiply the FG by the alpha
4. Composite over your background plate

### For Quick Checking

Open any image in the **`Comp/`** folder — these are PNG previews showing your key over a checkerboard pattern. Great for quickly scanning through frames to spot problems.

---

## Optional Alpha Generators

If you don't want to create your own alpha hints, CorridorKey includes two AI-powered generators. Both require significant GPU power.

### GVM (Fully Automatic)

- **What it does:** Automatically generates a rough matte — no mask input needed
- **Best for:** People and organic subjects
- **VRAM required:** ~80 GB (typically cloud-only)
- **Install (Windows):** Run `Install_GVM_Windows.bat` (Warning: ~80 GB download)
- **Install (Mac/Linux):**
  ```bash
  uv run hf download geyongtao/gvm --local-dir gvm_core/weights
  ```

### VideoMaMa (Semi-Automatic)

- **What it does:** Generates a refined matte from a rough binary mask you provide
- **Best for:** Any subject, with user control
- **VRAM required:** ~24 GB+
- **Install (Windows):** Run `Install_VideoMaMa_Windows.bat`
- **Install (Mac/Linux):**
  ```bash
  uv run hf download SammyLim/VideoMaMa --local-dir VideoMaMaInferenceModule/checkpoints
  ```
- **Usage:** Place a rough black-and-white mask in a `VideoMamaMaskHint/` folder inside your shot. This mask can be very rough — just a binary silhouette.

---

## Tips for Better Results

1. **Your alpha hint doesn't need to be perfect.** CorridorKey was trained on rough, blurry masks. It's designed to fill in the fine detail (hair, motion blur, translucency) from a coarse hint.

2. **Slightly eroded masks work better than expanded ones.** The model excels at adding detail that your mask missed, but it's less effective at removing detail that shouldn't be there.

3. **Use a good green screen.** CorridorKey is powerful, but garbage in = garbage out. Evenly lit green screens with minimal spill will give the best results.

4. **Check the Comp folder first.** Before importing EXR sequences into your editor, scan through the Comp PNGs to make sure the key looks good.

5. **Use a dedicated GPU if possible.** Running CorridorKey on the same GPU that drives your displays can cause out-of-memory errors. If you have two GPUs, use the secondary one.

---

## Common Problems

| Problem | Solution |
|---|---|
| "No target folder provided" | You need to drag a folder onto the .bat file, not double-click it |
| "CUDA out of memory" | Your GPU doesn't have enough VRAM (need 24GB+), or other programs are using it |
| Output looks dark/wrong in Premiere | Make sure your project is set to work in Linear color space for EXR files |
| Green edges on subject | Increase despill strength to 10 |
| Specks/dots in the matte | Enable auto-despeckle, increase the size threshold |
| Frame count mismatch error | Your Input and AlphaHint must have the same number of frames |

For more detailed troubleshooting, see the full [Troubleshooting Guide](TROUBLESHOOTING.md).

---

## Getting Help

- Join the **Corridor Creates Discord**: https://discord.gg/zvwUrdWXJm
- File issues on **GitHub**: https://github.com/nikopueringer/CorridorKey/issues
- The project creator recommends using a smart IDE like **Antigravity** (free from Google) if you need help with any Python-related setup
