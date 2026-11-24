# Image Trainer for Xbox

A real neural-network trainer that runs on an Xbox Series S or X. You give it pictures from your phone, it learns from them on the Xbox's graphics chip, and you ask it to draw new pictures of the same kind. Nothing is simulated: the numbers on the chart are the real training error, and the pictures are really drawn by the network you are training.

The Xbox shows a controller-friendly screen on the TV. Your phone (or any browser on the same Wi-Fi) shows a full control page.

## First start, in three steps

1. Open the app on the Xbox. The **Phone** screen shows a web address and a square code.
2. On your phone, on the same Wi-Fi, scan the code with the camera or type the address into the browser.
3. On the phone page: **Pictures** > **Add pictures**, then **Train** > **Start**. The first drawings appear after a few seconds.

Without pictures the brain practises on four built-in shapes. That is fine for seeing how training works, but add your own to teach it something real. Pictures of one kind of thing (flowers, faces, logos) teach it much faster than a mix.

## The phone page

| Tab | What it is for |
| --- | --- |
| **Train** | Start, pause and stop. Live drawings, the error chart, speed, and what the numbers mean. |
| **Pictures** | Add as many pictures as you like (up to **100,000**, as long as the Xbox has room), or a zip file of them. Each picture is fitted to 128 by 128 pixels on your phone before it is sent, so big photos are fine and nothing is lost to the old 16 MB limit. **Enlarge small pictures** keeps pixel art sharp (see below). Select and remove pictures, or remove them all. |
| **Create** | Make new pictures from what the brain has learned: 1 to 16 at a time, with a choice of drawing steps (a one-pass brain has a **Variety** slider instead, see below). |
| **Saves** | Snapshots are save games for the brain. Save, load, rename and delete them. The Xbox also saves by itself while training. Download the brain as a file to keep on your phone, or add a snapshot file back in. |
| **Settings** | Quick setups (Quick test, Balanced, Detailed, Pixel art), learning speed, how it changes over time, warm-up, pictures per lesson, picture size, random flips/zoom/colour changes, pixel art options, a smoothed copy of the brain (EMA), how often to save, graphics chip or processor, and more. Every setting explains itself, and **Reset** puts the recommended values back. **Start over with a new brain** (small, medium or large; normal, wave or one-pass) is also here, and **About this Xbox** shows the version, address and what the app is calculating on. |

## Pixel art

Small pictures with few colours (sprites, icons, 16/32/64 pixel pixel art) need care. The brain's pictures are stored at 128 by 128 pixels, and the old way of enlarging blended neighbouring pixels, which turned crisp edges into soft blur and made up colours that were never in your pictures. Now:

- **Enlarge small pictures** (Pictures tab, Settings tab and the TV Settings screen). **Automatic** (the default) copies pixels when a small picture with few colours is enlarged, so every pixel of the stored picture is an exact copy of one pixel of yours, and blends the pixels of everything else (photos). **Keep pixels sharp** copies pixels for every picture. **Smooth** is the old behaviour. The choice applies to pictures added from now on. A picture counts as pixel art when it uses at most 256 colours and at most one colour per four pixels.
- **Picture size** should match your art: 64 px for 64 by 64 pixel art, 32 px for 32 by 32. A 64 pixel sprite is stored as 128 by 128 (every pixel twice in each direction) and trained at 64, which gives back the exact original pixels. Turn **Zoom in at random** and **Shift colours slightly** off for pixel art, because they resample or change the colours.
- **Crisp colours** (Off, 8, 16, 32 or 64 colours). The live drawings and Create pictures are moved pixel by pixel to the nearest of this many colours, chosen from your pictures (up to 64 of them, spread over the whole set). If your pictures use fewer colours than the number chosen, exactly their colours are used. It is only a finish: it does not change how the brain learns, and switching it on or off needs no new training.
- The **Pixel art** quick setup sets picture size 64, sharp pixels, 32 crisp colours, no zoom, no colour shift and no lesson limit.

## Wave brains (faster training)

When you start over with a new brain you can pick its **type**: **Normal** or **Wave (faster)**.

- A **normal** brain works on the pixels of your pictures, as before.
- A **wave** brain first turns every picture into its *wave form* (one level of the Haar transform): the picture becomes a half-size copy of itself (the averages) plus three more half-size layers that hold the small details (left-right, up-down and diagonal differences). A 64 by 64 colour picture becomes 12 layers of 32 by 32 numbers. Nothing is lost, because the step can be undone exactly, and noise and the distances between pictures stay the same. The brain learns and draws in that form, and the app turns the result back into an ordinary picture when it shows it.
- Because the layers are half as wide and half as tall, a lesson needs about **4 times less calculation**, so a wave brain should train noticeably faster on the same Xbox. At 128 pixels it can also study many pictures per lesson (a normal brain is limited to 2 there). Faster does not promise better: whether a wave brain draws as well as a normal one after the same number of lessons is something to try and compare on your own pictures.
- A wave brain needs an even picture size (every size on the Settings page is even; an odd size is rounded down by one).
- The two types cannot be swapped. A saved brain keeps its type, and loading it makes the brain that type. Everything saved before wave brains existed is a normal brain and keeps working. Pick the type under **Start over with a new brain** (phone: Settings; TV: Settings > Type of new brain).
- Before a wave brain is trusted with the graphics chip, the app checks the chip's answers against the processor's, just as it does for a normal brain at start-up. If the check fails, that brain trains on the processor.

## One-pass brains (an experiment)

When you start over with a new brain, the third **type** is **One-pass (experiment)**. It is a different kind of brain, built to draw a picture in a single go instead of cleaning up static step by step.

- **How it works.** A one-pass brain has two halves. The *encoder* shrinks a picture three times (each time to half the width and height, while it looks at more and more detail per spot) until only a small **code** is left: a 32 pixel picture becomes a 4 by 4 grid of 8 numbers, 128 numbers in all. The *decoder* grows the code back into a picture through the same three levels. While it learns, the code is smudged with a little random noise, and the **Code tidiness** setting pulls the codes towards plain random numbers. That is what makes it possible, afterwards, to draw **brand-new pictures**: feed the decoder random numbers and it draws, in one pass, no steps. (This is the idea of a variational autoencoder, with a U-Net-like shape.)
- **Sizes.** Small, Medium and Large have about **1.07 million, 2.91 million and 7.83 million** dials (the other types have 25,000 to 250,000). More dials can hold more detail, and are slower. The picture size must be a multiple of 8, which every size on the Settings page is.
- **What to expect.** Pictures drawn from random codes in one pass usually come out **softer** than a step-by-step brain draws them, and the error number is a different measure from the other types' error, so do not compare the two numbers. The point of this type is to compare, on your own pictures: how fast it learns, and what it draws, against a normal brain.
- **Learning speed.** This type likes a gentler one. When you start over with a one-pass brain the app lowers the learning speed to 0.0003 if it was higher, and puts the normal 0.001 back when you go back to another type (it never raises a speed you set yourself).
- **Create.** There are no drawing steps. The **Variety** slider says how wild the random codes are: 0 draws the same plain picture every time, 1 is the normal spread (the spread the brain learned), 2 is stranger and some pictures may look broken.
- **Settings.** In Advanced settings, **Code tidiness** (0.004 is standard) sets how strongly codes are pulled towards plain random numbers. Too low: copies of your pictures come out sharp, but pictures made from random codes look like noise. Too high: everything is blurry. It fades in over the first 2,000 lessons.
- **Saves.** A saved brain remembers that it is one-pass, and loading it makes the brain one-pass again. Apps from before this type existed refuse such files with a clear message (their dial count does not fit) and keep working.
- **Not in this version.** The "experts" switch (mixture of experts) and bigger sizes come later, after this first type has been tried on a real Xbox.
- The graphics chip's answers are checked against the processor's before this type is trusted with the chip, as for the other types.

## The TV screen and controller

| Button | What it does |
| --- | --- |
| **LB / RB** | Switch between Train, Settings, Saves, Pictures and Phone |
| **A** | The main action of the screen (Start/Pause, Apply, Load, Enlarge) |
| **X** | Make pictures (Train, Pictures) or Delete (Saves) |
| **Y** | Save now |
| **B** | Back / cancel |
| **D-pad or left stick** | Move around and change settings |
| **View** | Shows the Phone address and code, and goes back to training |

## Where things are saved

Everything (pictures, snapshots, settings) is kept in the app's own storage on the Xbox, so it is still there the next time you open the app and after an update. It is deleted if you uninstall the app. Download the brain from the **Saves** tab if you want a copy on your phone.

Each picture takes about 48 KB of storage, so 100,000 pictures take about 4.9 GB. The app keeps about half a gigabyte of the Xbox's storage free: when less than that is left, new pictures are refused with a clear message instead of filling the console up. Pictures you remove give their space back the next time the app starts.

## Good to know (honest limits)

- The network is small on purpose (about 25,000, 80,000 or 250,000 dials; 1 to 8 million for the one-pass type) so it can learn on a console in minutes. Expect simple, soft pictures that capture colours, shapes and the general look of your pictures, not photo-realism.
- Keep the app open in front on the TV while it trains. If you switch to another app, the Xbox pauses it: the app saves first and carries on when you come back.
- The phone page has no password. Anyone on the same Wi-Fi who knows the address can open it, so use it on a network you trust.
- Training on the graphics chip is much faster than on the processor. If the graphics chip is not available the app says so and uses the processor.
- Very large collections (thousands of pictures) take a while to send. Keep the phone screen on until the upload says it is done.
- Every stored picture is checked when the app starts, so a very big set makes the start slower (on a fast disk about 5 seconds for 100,000 pictures; the Xbox's storage may be slower). The loading screen says so.
- The app asks the Xbox how much memory it may use (the activity log says it on the first line starting with [Memory]) and lets the pictures of one lesson use up to a third of it, at most 1.5 GB. With the usual small allowance it stays at the careful 224 MB.
- The Xbox gives apps more memory and graphics power when it treats them as games. In Developer Mode, open the Device Portal, go to Preference Settings and turn on "Treat UWP apps as games by default", then start the app again.

## If something goes wrong

- **The page does not open:** the phone and the Xbox must be on the same Wi-Fi, and the app must be open on the TV. The page tries again by itself.
- **The page says it cannot reach the Xbox:** wake the Xbox, open the app, wait a few seconds.
- **The chart stays flat for a long time:** it has learned what it can at these settings. Try more pictures, a larger brain or a lower learning speed.
- **Everything looks wrong after a crash or a bad experiment:** open **Saves** and load an earlier snapshot.
- **Activity log:** at the bottom of the Settings tab, **About this Xbox** has a button that downloads the activity log (including any crash notes). Send it to whoever helps you.

## For developers

- Built by GitHub Actions (`.github/workflows/build-xbox.yml`, windows-2022, Visual Studio 2022, C++20, UWP / Xbox). The workflow copies the files below into `xbox-uwp/src`, compiles the GPU shaders with `fxc`, packs them into the app, and produces a signed `.msix`.
- `FMBrain.*` is the GPU and CPU network. `Trainer.*` is the training loop. `Dataset.*` keeps pictures (one append-only file with a small index and a cache in memory; see the comments at the top of `Dataset.h`). `Checkpoint.*` saves and loads snapshots. `AppCore.*` ties it together and is the only thing the web page and the TV screen talk to.
- `HttpServer.*`, `WebApi.*`, `WebHost.*` are the phone page's server. `WebUiAssets.h` is the phone page as a byte array (generated from `web/` by `web/build_web.py`). `TvUi.*` decides what the TV shows; `GamepadUI.*` and `Main.cpp` draw it and read the controller. `WebBridge.*` holds the few Windows-only helpers.
- The phone page uses the fonts Bricolage Grotesque and Lora under the SIL Open Font License; see `THIRD_PARTY_FONTS.txt`.
- Everything except the Windows-only files is compiled and tested natively on Linux (address and thread sanitizers, a real browser against the real server). The graphics code (`FMBrain.cpp` with `FMCompute.hlsl`) is also run in tests on a software graphics chip that executes the very same shader text and checks it against the processor path and against the rules of the D3D12 API (buffer states, barriers, ranges). The Windows-only files and the real shader compilation are only done by the GitHub workflow.
