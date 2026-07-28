# Media capture guide

Everything the README needs, and exactly how to make it. The architecture
diagram is already in the repo (`docs/assets/architecture-*.svg`, generated, no
capture needed). The three items below need the real robot — capture them once,
drop the files in `docs/assets/`, and uncomment the matching block in the README
(each drop-in point is marked with an `HTML comment` there).

Keep files small — a README that takes 10 s to load costs you the stars the
visuals were meant to win. Target: hero GIF < 5 MB, screenshot < 400 KB.

---

## 1. Hero GIF → `docs/assets/hero.gif`

The single most important asset. It should answer "what is this?" in five
seconds, before anyone reads a word.

**Shot list (5–8 s, no cuts):** PiDog standing → a spoken command lands
("Nox, sit!") → the dog sits and wags its tail. One continuous take, eye-level,
the whole dog in frame, plain background, good light.

**Record** on a phone (1080p is plenty), then convert — the two-pass palette
route keeps it sharp and small:

```bash
# trim to the good 6 seconds first (adjust -ss start / -t duration)
ffmpeg -ss 0 -t 6 -i raw.mov -vf "fps=15,scale=640:-1:flags=lanczos,palettegen" palette.png
ffmpeg -ss 0 -t 6 -i raw.mov -i palette.png -vf "fps=15,scale=640:-1:flags=lanczos,paletteuse" hero.gif
```

If it's still over ~5 MB, drop `fps` to 12 or `scale` to 480.

---

## 2. Demo video → YouTube (thumbnail `docs/assets/demo-thumb.jpg`)

30–60 s, hosted on YouTube (GitHub won't embed a video file inline — a linked
thumbnail is the standard pattern, already wired in the README).

**Shot list:** voice command → movement · autonomous patrol avoiding an obstacle
· a face being recognized (name overlay). Narrate or add captions so it reads
without sound.

Grab a thumbnail frame from the video:

```bash
ffmpeg -ss 3 -i demo.mp4 -vframes 1 -q:v 3 docs/assets/demo-thumb.jpg
```

Then set the real video id in the README (`https://youtu.be/YOUR_VIDEO_ID`).

---

## 3. Telegram control screenshot → `docs/assets/telegram-control.png`

Shows the "control your robot from your phone" angle — high-signal, takes two
minutes.

**What to capture:** a real Telegram exchange with the bot — you send a command
("sit and wag"), the dog's photo or a status reply comes back. Crop to the
conversation. **Scrub anything private** before committing: your phone number,
the bot token, other chats, real names in the header.

```bash
# optional: shrink/optimise a PNG screenshot
ffmpeg -i telegram-raw.png -vf "scale=360:-1" docs/assets/telegram-control.png
```

---

## Checklist before committing

- [ ] `hero.gif` under ~5 MB, plays in a loop, dog fully in frame
- [ ] `demo-thumb.jpg` set + real YouTube id in the README
- [ ] `telegram-control.png` — no phone number, token, or private chats visible
- [ ] uncomment the matching README blocks (hero, "See It in Action")
- [ ] check both light and dark GitHub themes render the page cleanly
