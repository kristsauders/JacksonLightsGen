# Building a sequence for a brand-new song

Written while building **BLUEY DANCE MODE! (OFFICIAL TRAP REMIX) - ZEESLOW**
(2026-09-17) and extended while building **Ariana Grande - December (OFFICIAL
TRAP REMIX)** (2026-09-17); both are worked examples throughout. The point of this file is
that the next new song should take a fraction of the time, and should not repeat
the mistakes listed at the bottom.

The generator is a rule system: it applies the same rules to every song, which
is why fixes generalise. This runbook is the other half of the job — the
per-song **judgement**, and the places where the rules need to be told what a
human can hear.

---

## 0. Get the file name exactly right

```bash
.venv/bin/python3 -c "
from pathlib import Path
for p in sorted(Path('/Volumes/XLIGHTS/XLights 2026').glob('*.mp3')):
    print(repr(p.stem))"
```

The song name is the mp3 stem and it is used verbatim everywhere (folder names,
per-song config keys, `(vN).xsq`). Titles with `!`, `(`, `)` and `-` are fine,
but quote them in shell commands. macOS also leaves `._Name.mp3` AppleDouble
sidecars on this volume — ignore them; never pass one as a song name.

## 1. Analysis (slow, once per song)

```bash
.venv/bin/python3 custom_tools/run_analysis.py "<song>"      # ~2.5 min, demucs
.venv/bin/python3 custom_tools/detect_cues.py  "<song>"      # seconds
```

`run_analysis` writes `.generation/<song>/analysis.json` (beats, downbeats,
sections, spectrum) and the four demucs stems under `stems/<song>/`.
`detect_cues` writes `cues.json` (drops, cutouts, risers, arpeggios, accents).

**Review the stems before trusting anything downstream.** Measure, don't listen:

```bash
# per stem: duration, rms, peak, onset count, <200Hz share, >2kHz share, silence
```
Bluey came out: bass 70% below 200 Hz and only 8% above 2 kHz (clean, no
bleed); drums 25% low / 53% high with 335 onsets (normal trap kit); vocals RMS
0.016 and 75% silent — i.e. an instrumental remix with chops, so vocal-driven
passes will do very little. That last fact is worth knowing *before* wondering
why nothing follows the vocal.

## 1b. Read the song before you design anything

Added on the second song (Ariana Grande - December). Search the web for the
track's meaning, lyrics and production notes *before* choosing colours or
effects, because the concept should come from the song rather than from the
palette list.

For "December" (from *Christmas & Chill*, 2015, here as a trap remix) that
turned up: jingle bells opening into trap-R&B, and a lyric about keeping
someone warm "through the wintertime" -- warmth as light against the cold. That
is a show concept in one line: **cold outside, warm inside**, icy palettes on
the verses and warm ones on the choruses, and deliberately NOT primary-red
Christmas, because the track is intimate rather than festive.

Write the concept down in a sentence before step 5. If you cannot, the palette
and shader choices will end up arbitrary again.

Two techniques that worked on "December" and are worth reusing:

* **Give each section ROLE its own look, and repeat it.** Verses ran icy `ice`
  + Rain on Glass, choruses warm `ember` + Falling Embers, builds and
  breakdowns `glacier` + Frost Growth. Repetition by role is what makes a show
  feel composed rather than shuffled — a verse reads as a verse.
* **Let the concept pick the contrast.** "Cold outside, warm inside" gives you
  a reason for every palette, and it means the breakdown (where the song strips
  down) can literally go cold again. The `smoulder` + Stained Glass climax then
  lands as the warmest point of the song.

Check what the stems say about the vocal before promising anything vocal-led.
"December" has a real vocal (RMS 0.047, 504 onsets, 34% silent) and got 26 sung
phrase accents — but only 13 melody notes, because the remix chops the vocal so
pyin holds pitch for just 8% of frames. That is a property of the source, not a
bug: median-filtering the pitch track and bridging gaps moved it from 21 usable
runs to 23. Measure before "fixing" it.

## 2. Check the beat grid

Median beat spacing should match the reported tempo, and the outliers tell you
where the tracker struggled:

- Bluey: 129.2 BPM, beat 464 ms, jitter sd 16 ms, with three gaps of 372/580/580 ms.
- All three outliers sat inside the near-empty breakdown at 48–52 s. That is the
  tracker having nothing to lock to, not a tempo change — harmless.

## 3. Map the song bar by bar, then decide the sections yourself

This is the single highest-value step, and the structure detector will not do it
for you. It is good at finding *that* something changes and poor at *when*.

```bash
.venv/bin/python3 custom_tools/song_map.py "<song>"
```

Written on the second song, after hand-rolling the same probes for the first.
It prints the beat statistics and tracker wobbles, one line per bar (energy,
bass share, hat share, with `<<< step up/down` markers), the detected sections
with the energy change across each boundary and its distance to the nearest
beat and bar line, the drop candidates with measured hush and lift, and a
paste-ready list of bar lines whose energy step is big enough to read as a
section change. It writes nothing — the decisions stay yours.

Bluey's map immediately showed the real arrangement: 8 full bars, a 2-bar
breakdown, a 6-bar drop, 2-bar breakdown, 8-bar drop, 2-bar near-empty break, a
12-bar half-energy middle, an 8-bar drop, a 6-bar **climax** (the loudest,
bassiest bars in the track), a 2-bar dip, a 2-bar final stab, then silence.

Treat the suggested boundaries as a starting point, not an answer: a step in
energy is a boundary candidate, but a section can also change character without
changing level (a verse into a pre-chorus), and two adjacent bars of the same
energy can still belong to different sections.

Compare candidate grids by the energy change across each boundary:

- detected boundaries: mean novelty **0.331**, only 3 of 8 within 30 ms of a beat
- hand-read boundaries: every one on a bar line, and the two weakest detected
  boundaries (43.58 s novelty 0.12, 59.21 s novelty 0.03) turned out to be inside
  one continuous 12-bar middle section

Write the result into `SONG_SECTIONS` in `generate_song.py`. The generator now
syncs it into `analysis.json` for you (keeping the detector's original once, as
`analysis.detected.json`), because the generator reads `SONG_SECTIONS` while the
verify tools read `analysis.json` — mistake #6, now fixed in code rather than by
remembering to do it twice.

A hand-planned grid is also treated as authoritative: sections are no longer
re-split at energy step-downs, sub-2-bar sections are not merged away, and every
planned boundary gets a look change even when the levels barely move. That last
one came from "December", whose 85.34s build differs from the verse before it by
0.02 energy — the marking guard's energy test skipped it and `verify_sections`
reported the boundary as passing with nothing changing.

## 3b. HAND-MAP THE SECTIONS. Always. This is not optional.

The single highest-value half-hour you can spend on a new song. The structure
detector is good at finding THAT something changed and **bad at finding when**,
and on a uniform song it invents boundaries that do not exist at all. Measured
across the 2026 library:

| song | detector said | reality |
|---|---|---|
| Aperture | 19 sections, **every one ~16.0s** | uniform grid; bars 65-148 never change |
| 12 to 12 | 20 sections | 4 real events; five 1-second fragments AFTER the audio ends |
| This Version of You | **100 sections**, median 1.2s, 94 called "transition" | one long crescendo |
| The Last Goodbye | 52 sections, median 2.0s | ~14 real turns |
| Gooey | 57 sections, median 1.2s | 5 real turns |

A look change at an invented boundary cannot be "on time" -- there is no event
for it to be on. Krists hears this as "the looks change a few beats early".

### The tool

    python custom_tools/section_plan.py "<song>"

It prints, per **four-bar phrase**, only the bar lines where total energy, bass
or hats actually turn -- plus the drop cues and the detector's own verdict.
(Four bars, not one: a hi-hat that drops out every other bar is texture, and a
bar-by-bar reading of Corners of the Earth reported forty "hats IN/OUT" turns,
none of them structural.) Use `song_map.py` when you want to study one passage
bar by bar; use `section_plan.py` to plan the grid.

### How to read it, as a songwriter

* **bass IN** is the drop. **bass OUT** is the breakdown. These are the two
  strongest boundaries in almost every song here, and they are unambiguous --
  0.13 -> 1.82 on World Is Mine at 95.55s.
* **hats IN/OUT** marks the lift *inside* an otherwise flat section. Aperture
  is one long 0.55-energy plateau; the hats entering at 74.43 and leaving at
  276.91 are the only interior structure it has, and they matter.
* **energy alone** moves last and least. Do not lead with it.
* **A long run with no change point is ONE section**, however many the detector
  claims. Corners of the Earth holds from 165.74 to 251.56 -- that is a single
  86-second climax, not five choruses.
* **Cross-check the drop cues.** A boundary within ~0.5s of a cue is the same
  event; take the bar line, not the cue time.
* Songs that start already running get no intro (Too Slow's bar 0 has bass
  1.32). Songs that stop dead get a boundary where they stop.
* Search the web for the track when the structure is ambiguous -- an official
  breakdown of verse/chorus/bridge positions resolves a judgement call fast.
  It is worth doing for anything with lyrics.

### Writing it down

Every boundary is a **bar line time from the map**, pasted into `SONG_SECTIONS`
with a label. Labels are free text and only need to be consistent within the
song -- the generator no longer keys behaviour off them (see below).

Put the evidence in a comment above the entry: which bars turned and why. Six
months from now the reason for overriding the detector must still be legible.

### Why labels stopped mattering

The detector reuses one name for most of a song. Three passes keyed off that
name and all three misbehaved:

* `_g_section_arc` held any "build"/"break"/"transition" at 75% of the previous
  section -- with 94 consecutive "transition" labels that cascaded to the floor
  and pinned **18 consecutive sections at 42%**, about a minute of dark.
* `escalation` counted repeats of the label, so it hit maximum at section three
  and stayed pinned -- the opposite of a build.
* melody detection took "the chorus sections" literally: most of Aperture, or
  an arbitrary 5 of Gooey's 57.

All three now measure instead (a dip is quieter-than-before-and-followed-by-
louder; escalation buckets the energy; melody takes the loudest third). **If
you add a pass, do not branch on `section.label`.**

### What good looks like

After hand-mapping, verify:

* boundaries **100% on a beat** (they are bar lines, so they should be);
* **most boundaries carry a look change** -- Aperture went 7/20 -> 15/17.
  Boundaries inside drop staging are expected to have none; the blackout is
  the change.

## 4. Name the drops and the important moments

`cues.json` offers far more candidates than a show should stage, ranked by local
band energy, which is not the same as "the moment the song is built around".
Bluey's detector staged 20, 34, 43, 75, 86 and 103 s — it spent a stage on 43 s
(mid-chorus) and missed the actual climax at 89.2 s by 3.2 s.

Read the drops off the bar map and set `SONG_DROPS`. Keep the list short; five
drops in a 112-second song is already a lot.

## 5. Choose the colours and the shaders deliberately

Per-song knobs, all in `generate_song.py`:

| knob | what it decides |
|---|---|
| `SONG_PALETTE_FAMILY` | the song's palette rotation; **any length**, not just three |
| `SONG_SECTION_LOOK` | per section start time: pin `palette` and/or `shader` |
| `SONG_SECTIONS` | the section grid (start time, label) |
| `SONG_DROPS` | which moments get pre-drop staging |
| `SONG_AMBIENT` | ambient windows; `[]` means "never goes ambient" |
| `PULSE_MODE_OVERRIDE`, `GROOVE_SONGS` | pulse feel; groove = 4-beat sphere cycle |

Choose against the song's identity, not at random. For Bluey: `chrome` (the
show's blue) as the identity, `cinema` (teal + orange = Bluey and Bingo) on the
first drop, `candy` and `resin` for the cartoon-bright later drops, `glacier` on
every build, and the built-in gold-and-blue `climax` palette on the loudest six
bars. Both builds deliberately reuse **one** shader (`frost_growth`, which
creeps outward) so a build reads as a build rather than as another change.

Shader vocabulary: 50 registered, 31 palette-driven, 12 centred. Prefer
palette-driven and non-centred for anything shared across canvases — a centred
pattern repeated across the house reads as ten copies of one picture.

## 6. Generate, then read the log before rendering

```bash
.venv/bin/python3 custom_tools/generate_song.py "<song>"
```

Lines worth reading every time: `Sections:`, `Palette family`, `Instrument
voices`, `Ambient passages:`, `Drops:`, `Section shaders:`, `Sphere looks`,
`Pre-drop implosions`, `placements across`. Each one is a decision you can
check against the bar map.

## 7. Render and verify

```bash
.venv/bin/python3 custom_tools/verify_render.py    "<song> (vN)"
.venv/bin/python3 custom_tools/verify_sections.py  "<song> (vN)"
.venv/bin/python3 custom_tools/verify_drops.py     "<song> (vN)"
.venv/bin/python3 custom_tools/verify_alignment.py "<song> (vN)"
```

Bluey v4 landed at: sections **100% on the beat** (median 0 ms off), 8 of 10
boundaries marked with a real change, drops **12/12 on time and none early**,
staged drops rising 2.67× / 2.09× / 1.69× / 1.63×, render VERDICT OK, 68% lit.

Note `verify_drops` evaluates every cue *candidate*, not just the staged ones,
so "no hush / no lift" on unstaged candidates is expected noise.

## 8. Review it as a lighting designer, not as a verifier

```bash
.venv/bin/python3 custom_tools/design_review.py "<song>" [version]
```

The verify tools answer "did the rules run". This one answers "is it a good
show", and it is the step most likely to change what you build. Run it after
every new sequence.

What it measures, and what each number is for:

| section | reads | what a bad number means |
|---|---|---|
| CONTRAST | mean/p10/p90 brightness, dynamic range, time near black, correlation with the music | a narrow range (under ~3x) means the house never really rests; near-black under a few percent means no silence to punctuate with |
| ARC | per section: music energy vs rendered light, lit%, colour dominance, movement | light that does not track the music's shape, or a peak in the wrong place |
| TENSION | each staged drop: level 2s before vs peak after | under ~1.4x the drop "barely registers"; over 2x reads as a real hit |
| BALANCE | lit% and mean per prop family | everything at 90%+ means nothing is ever featured |
| WHO RESTS | which families sit out each section | "NOTHING -- every group is lit" is the flag |
| REPETITION | sphere looks, effect vocabulary | an `On`-dominated vocabulary reads as things switching on rather than moving |

### Principles this is checking, and the levers for each

1. **The peak belongs where the song's drama is.** Default: the loudest
   section. Often that is right; sometimes it is exactly wrong -- on Bluey the
   drama is the FIRST drop, because a children's theme turning into heavy trap
   is the surprise the show is built on. Lever: `SONG_CLIMAX` names the
   section(s) that peak; `_g_section_arc` gives them 100% and holds everything
   else below 85%.
2. **Contrast is built from the floor, not the ceiling.** Dim the BEDS of
   quieter sections rather than capping the hits. `_g_section_arc` only scales
   effects longer than 0.6s and never touches drop staging, so hits keep their
   punch while the floor drops away.
3. **A build has to fall.** Measured flaw: one Bluey build got brighter as the
   music fell 57%. Sections labelled `build`/`break` are now held below the
   section before them.
4. **Subtraction is what makes a moment.** If every prop family is lit all the
   time, nothing is ever featured. Look for "NOTHING -- every group is lit".
5. **A section should read as a colour.** Colour dominance (top-2 hue share)
   above ~70% reads as an identity; 37% is a wash of everything.
6. **Movement should be paced, not constant.** One section measuring 2-3x the
   movement of its neighbours reads as frantic rather than exciting.

### The four levers built from the first design review (Bluey, 2026-09-17)

All are generic guards and run for every song:

| guard | what it does | measured effect on Bluey |
|---|---|---|
| `_g_section_arc` | every section takes a brightness ceiling from its rank in THIS song; builds/breaks are held below the section before them; only beds over 0.6s are scaled, never hits or drop staging | dynamic range 2.31x -> 10.25x; the build that got brighter as the music fell now falls |
| `_g_rest_rotation` | one prop family sits out per loud section, rotating; masked with an opaque `Off` (dropping its own effects is not enough — the house groups relight it), and it rejoins for staged drops | families went from a flat 90-97% lit to 52-95%, so something is finally featured |
| `_g_section_colour` | beds are retinted to the section's pinned palette, and a bed crossing a boundary is SPLIT so the old palette does not run on | colour dominance rose in seven sections (intro 55% -> 76%) |
| `_g_movement_over_flash` | a short flat `On` on a prop with real extent becomes a gesture that draws the prop's shape; one-node props keep the flash | `On` fell from 57% of placements to 25%, SingleStrand became the lead effect — much closer to Krists's own 6/30/21 profile |

`SONG_CLIMAX` names which section is the peak, because the loudest section is
not always the dramatic one.

### The hard-won lesson: contrast devices MULTIPLY

Applying the arc, the rest rotation and the colour fix together at their
individual best settings produced a dead house — mean brightness 38.7 against a
~50%-lit human norm, with a 22-second verse at 10% of the house lit. Each
device was reasonable; three of them landing on the same quiet bars was not.

The rules that came out of it, and the numbers to aim for:

* **Rest is a LOUD-section device.** Only where the arc has not already pulled
  the section down (loudness >= 0.55), and one family at a time.
* **The arc needs a floor.** 0.58 of full, not 0.45 — below that the quiet
  sections stop reading at all once anything else subtracts.
* **Splitting a bed at a boundary beats truncating it**, but the tail must be a
  DECAY (2 seconds, fading) rather than a continuation, or the loud section's
  beds flood the quiet one and halve the contrast again. Big canvases are the
  exception: they carry their texture through, because they are few.
* Targets: **lit fraction ~50%**, **dynamic range 6-10x**, **near-black 15-25%
  with no stretch beyond ~4s**, **every staged drop >= 2.5x**.

Expect two or three rounds of overshoot when tuning these. Measure after each
one; the review tool is cheap and guessing is not.

### The second hard-won lesson: a rest must END when it ends (2026-09-17)

Found on the 20-song rebuild, not on a new song — which is why this check now
belongs in every review. Five songs had a section under 25% of the house lit
*inside* the music (the ones after the audio stops are correct, that is
`_g_dark_after_audio`). The worst was Aperture's 153.8s chorus at **16% lit**,
under an arc ceiling of 97%. Two guards disagreeing.

The cause was not the arc and not the colour trim. `_g_rest_rotation` selects
effects by their START time — correct — and then **deleted** them. On the big
canvases the bed is one long span: resting Arches/Tunnel for the 138.1s section
deleted the Tunnel's shader, which ran to 170s, so the *next* section lost its
bed too and the arches sat dark for 32 continuous seconds.

Two fixes, both in `_g_rest_rotation`:

1. **Trim, don't drop.** An effect that outlives the rested section has its
   start moved to the section end. A rest must end when it ends.
2. **Only rest a family big enough to miss.** The rotation treated every family
   as an equal turn, but on the 2026 layout they are Arches/Tunnel 28.3% of all
   channels, Matrices 25.5%, Roof/Gutters 22.7%, Flakes/Spinners 14.0%,
   Canes/Trees 7.5%, **Verticals 2.0%, Floods 0.0%** (nine floods, 27 channels).
   Four of Aperture's twelve rests were spent on Floods and Verticals, where
   nothing visibly sat out. Families under 4% of the house are skipped.

**Add to the review, and run it whenever a section looks dark:** per-element
*coverage* — the fraction of a window that has any effect on that element —
for a dark window beside a comparable bright one. Brightness alone, effect
counts and effect-name histograms all looked normal here (1979 vs 2031 active
effects, near-identical vocabularies); coverage showed `Tunnel` at 3.7% against
100% and named the culprit in one line. Counts tell you something was placed;
coverage tells you whether anything is actually lit.

The generalisation, worth applying to any new guard: **a guard scoped to one
section must not perform an action whose effect outlives the section.**

### Effect-vocabulary levers (added 2026-09-17, from direct feedback)

| guard / knob | what it does | why |
|---|---|---|
| sphere look pools | glitter appears only in the TOP energy tier, rain only in the bottom; colour/flash/travel looks are weighted by repetition | "they do the sparkle glitter effect too often... should do more color changes and flashes and movement". Before: 17 of 19 looks were textures. After: glitter 1, colour_beat 4, flash_halves 5 |
| `_g_arch_leap` | arch chases alternate direction with the bar, a third carry 2-3 dots, every fourth bar takes the section accent colour | "they look like they are leaping because a chase goes from left to right on the beat... multiple chases, or chases changing colors" |
| `_g_matrix_beat_shaders` | a matrix shader bed longer than two bars is cut into two-bar pieces that REVERSE the shader speed (its drift direction) and step the zoom | "I'd like the shaders to maybe move more and change direction on the beat or change zoom direction on the beat" |
| `_g_matrix_bars` + `matrix_bar_*` keys | a big bar sweeps each matrix on the bar line, one sweep per two bars, in energetic sections only, ONE orientation per section | "we can use things like a big bar going across side to side, or up down, as rhythm". Bars were retired on matrices because the ORIENTATION flipped mid-phrase; fixing the orientation per section is what makes them usable again |

Two gotchas found while building these, both of which cost a rebuild:

* **Check which pass actually produces the thing you want to change.** The
  arches' centre-bursts were not coming from the old Bars substitution at all --
  `_g_movement_over_flash` was mapping every unmatched prop to `chase_mid_out`,
  which is From Middle. Fixing the mapping restored the leap; the dedicated
  guard then had almost nothing left to convert (2 of 173).
* **`section_meta` only carries what someone put there.** The arch colour
  variation read `section_meta[si]["accent"]`, which did not exist, so that
  branch silently did nothing until the accent was stored alongside the palette.

Record the flaws you find and what you changed, so the next song starts from a
better default rather than rediscovering the same faults.

---

### Making a gesture actually READ (2026-09-17, from review of the rebuilt library)

Four faults found together, all of the same shape: the effect was in the file
and could not be seen. Counting placements proves nothing — check that the
gesture is long enough, unmasked, on the right element, and phase-locked.

1. **Two beats is the floor for travelling motion.** "They're not set up to
   take one full beat like the verticals do." On Aperture (129.2 BPM) the
   vertical chases that read run a median 0.93s = two beats; the arch chases
   that did not read ran 0.42s. Krists's own sequences: arch chases median
   1.00s. Anything under two beats is a flicker.
2. **A group gesture must clear the model-level effects beneath it.** A chase
   on the `Arches` GROUP spanning all four arches was blanked 32% of the time
   by 0.23s chases on the individual arches — a model-level effect beats a
   group-level one at any layer. Place the span, then remove the blinks.
3. **Span the row, don't repeat on each prop.** 187 of 195 arch chases sat on a
   single arch. A chase confined to one arch cannot leap across the row. Use
   the group with `B_CHOICE_BufferStyle=Single Line` (the group as ONE strand);
   Krists puts 4,703 of his arch chases on the `Arches` group.
4. **Name tests silently exclude props.** Both matrix rhythm passes selected
   with `"matrix" in name.lower()`, so `Garage Roof` — an 18,576-channel
   matrix, the second-biggest flat canvas in the show — got none of the beat
   work while its eight neighbours did: "the garage roof matrix often acts very
   differently from the other matrices." Use one shared predicate
   (`is_matrix_canvas()`), never an inline name test.

**Cohesion needs ONE clock.** Shader pieces were cut from each effect's own
start time, so canvases whose beds began a beat apart flipped direction a beat
apart forever: "each matrix does a different pulsing speed so it doesn't look
good." Cut on an absolute grid derived from the song's first downbeat, and
index each piece by its position on that grid, so "which way is this piece
drifting" is a property of the bar and identical on every canvas.

**Beat-locked shader motion.** "There is a green shader that seems to be
changing with the beat. It's actually slightly off the beat but if it was right
on the beat it would be a perfect example." That shader had a FIXED
`E_SLIDER_Shader_Speed=209` — a constant drift near the tempo that walks out of
phase. The fix is a value curve on the speed with a spike on each of our own
beat times. Format (copied from his sequences, verified against them):
`E_VALUECURVE_Shader_Speed=Active=TRUE|Id=ID_VALUECURVE_Shader_Speed|Type=Custom|Min=-1000.00|Max=1000.00|RV=TRUE|Values=...`
— the range is **-1000..1000, so 0.50 is a DEAD STOP**; rest a little above it
and spike on the beat. Zoom and Offset_X/Y curves use Min/Max -100..100.

**Measurement trap: one file can own the histogram.** Pooling every reference
sequence to find "what does he put on the matrices" returned `Fill` as 40% of
everything — 21,600 of those effects were in ONE file,
`No Music - Lights In Winter.xsq`, a static idle loop. Always count per file
and look at the distribution across files before believing a vocabulary claim.

### Buffer style IS the effect (2026-09-17, the arches, three attempts)

"The arches still aren't leaping at all. I see some slow drifts but no rhythmic
leaps left right left right. Something is still totally wrong, I don't know that
it's just a render mode, I think it's just the wrong effect entirely." He was
right, and it took three tries because I kept changing the wrong variable:

1. `SingleStrand` chase, `Per Preview`, on ONE arch — a chase confined to a
   single arch cannot cross the row at all.
2. `SingleStrand` chase, `Single Line`, on the `Arches` group — `Single Line`
   strings every PIXEL of all four arches into one strand (~1,200 of them), so
   the dot crawls through them and reads as a smear.
3. **`Bars`, `As Pixel`, on the `Arches` group** — `As Pixel` makes each MODEL
   one pixel, so there are exactly four, and the bar steps arch → arch → arch
   → arch with nothing in between. One cycle across a 4-beat bar = one arch per
   beat. This is what Krists uses: 90 of the 107 `Bars` on his arches are
   `As Pixel`.

The lesson beyond the arches: **on a GROUP, the buffer style decides what the
gesture IS**, not merely how it looks. Ask "how many pixels does this group
have in this buffer?" before choosing the effect. And a chase keeps its
head/tail fade even on four pixels, which is why it reads as a drift where a
Bars block reads as a step.

### Name collisions in this file fail silently and weirdly

Two in one day, both inside nested functions in `main()`:

* a leftover `_rest` in a rewritten function resolved to an unrelated list in
  the enclosing scope and failed as `TypeError: unsupported format string
  passed to list.__format__`, not `NameError`;
* a module-level `_inactive_models()` helper was shadowed by `main`'s own local
  `_inactive_models` SET, failing as `'set' object is not callable`.

`main()` is thousands of lines with dozens of nested guards sharing one scope.
Before adding a helper, grep the name. Prefer an un-prefixed module-level name
for anything called from inside `main`.

### Beat grids drift — snap to the DETECTED downbeats

Piece boundaries computed as `first_downbeat + N * (beat_ms * 4)` use one
median beat length for the whole song. Over a 312s track (~160 bars) that walks
away from the music: measured, shader changes landed within 80ms of a real
downbeat only **21%** of the time, median 392ms off — which is what Krists
noticed as "I thought we had all matrices changing right on the kick drum beat
of a new bar, what happened to that?". Snapping every edge to the nearest
ACTUAL downbeat took it to 65% / 45ms. Never multiply a median beat length
across a whole song when the detected downbeats are available.

### Exclude switched-off props explicitly

Models with `Active="0"` in `xlights_rgbeffects.xml` render nothing, so effects
on them are silently discarded — but they still consume the pass that placed
them AND they corrupt every measurement that reads the .xsq back. The three
retired Front Window Matrices held 264 placements each, and made "some matrices
have a shader while others are blank" measure as 234 of 240 moments when the
true figure for the live canvases was 47. **Filter the dead props in the
measuring script too, or the number you report will be wrong.**

### A majority vote is the wrong rule for cohesion

"Fill the canvases that are missing what the MAJORITY are showing" cannot fix
the case that actually happens: one canvas lit and five blank is a majority of
zero, so nothing gets filled and the house looks broken. Use "any canvas is
showing one" and exclude deliberate darkness (blackout windows, rests)
explicitly, rather than inferring intent from a vote.

### GREP FOR AN EXISTING OWNER BEFORE WRITING A NEW PASS (2026-09-18)

The single most expensive mistake of this session. Krists asked for leaping
arches; a `Leaping arches` pass already existed, with exactly the three shapes
he described (sweep = one bar across all four, pairs = two bars across two
arches, solo = each arch independently) rotating by section. I did not look,
wrote a second arch pass, and the two then placed on the same props at the same
time: "one arch leaps, one spins, one two leap, the directions are random, it
makes no sense."

**Before adding a pass for a prop family, grep for that family's name in
`generate_song.py` and read every site that already places on it.** Then fix the
existing owner rather than adding a competitor. The clean-up guard that removes
competing effects is a second-best fallback, not a substitute.

### Every arch failure was ONE fact about the prop

`Arch N` is `LayerSizes=100,100,100` -- three concentric rows of 100 nodes,
stored one row after another. Almost every buffer style walks those rows one
after another instead of together, and the symptom changes just enough each
time to look like a new bug:

* `Per Preview` / `Single Line` -> runs the length of row 1, then row 2, then
  row 3. Krists saw this twice, described first as "a smear" and later, exactly
  right, as "each row chasing very very very fast, one after another" and as "a
  weird inner ring, middle ring, outer ring and back movement".
* `As Pixel` + `Bars`, `Per Model Default` + `Bars` -> **solid fill**, because
  `BarCount=1` in xLights means "divide the buffer into ONE bar". Measured as
  100% of the arch lit, position pinned at 0.50.
* `Per Model Horizontal Per Strand` -> travels ACROSS the three rows: the ring
  movement, retired.
* **`Per Model Vertical Per Strand` -> correct.** Each row gets its own line and
  the chase travels along all three together, so one bar crosses the arch.
  Verified: 10% of the arch lit, all three rows at an identical position,
  sweeping 0.92 -> 0.03 in a beat, then the next arch.

**Read the model definition in `xlights_rgbeffects.xml` before choosing an
effect** -- `LayerSizes`, `NodesPerArch`, `NumStrings`, node order.

### Verify at the granularity of the thing you claim to have built

Three false "it's fixed" reports came from measurements that could not
distinguish the two cases:

1. mean brightness per model -- cannot tell "the whole arch flashed" from "a dot
   crossed it"; both make Arch 2 bright.
2. centroid of lit nodes per model -- cannot tell one bar crossing from three
   rows crossing in turn.
3. the row check's own verdict line called 100%-lit "BAR across all 3 rows"
   because the three rows agreed on position 0.50. **Agreement between rows is
   necessary and nowhere near sufficient; a bar is also NARROW.**

Write the check so it can FAIL for the specific thing being claimed, and print
the raw numbers next to the verdict so a wrong verdict is visible.

### Cohesion is enforced by REPLACEMENT, not by merging

Two attempts to sync the matrices merged settings while leaving each canvas its
own effects, and both failed: a majority vote over the shader at each effect's
start cannot converge (at 155.90s the vote was 3-3 and both shaders survived),
and nothing touched the mismatched SPANS -- which is what "different movements
at different speeds" actually is. What worked: pick the canvas with the most
shader time, DELETE every other canvas's shader effects, and copy the
template's pieces verbatim. Identical inputs render identically. Measured after:
416 of 416 moments with all six canvases lit together and identical.

### Check the family table when one prop "does its own thing"

`Garage Roof` is an 18,576-channel flat canvas. Teaching the rhythm passes to
treat it as a matrix (via a shared `is_matrix_canvas()`) fixed its movement but
NOT `_prop_families`, where it stayed filed under Roof/Gutters -- so when the
rest rotation rested "Matrices" it masked five canvases and left the Garage Roof
firing nine bar sweeps through the silence: "the garage roof flashes at 4:15
which is weird, while all the others are quiet." A shared predicate is only
shared where it is actually used; grep for the other classifications too.

## Mistakes made this time — check these first next time

1. **`detect_cues.py` crashed** on the bass-arpeggio detector: `clean_pcs`
   replaced a `None` pitch class with `pcs[k % len(pcs)]`, which is the same
   `None`, then indexed `NOTE_NAMES`. Fixed to carry the previous resolved note
   forward. It had never fired before because it needs an ambiguous note inside
   an otherwise repeating bass figure.
2. **Per-section pins keyed by section index drifted.** The generator splits
   sections at energy step-downs and merges sub-2-bar slivers *after* the plan is
   applied (11 → 12 → 10), so every pin from the second section on landed a
   section late — the climax got `resin` instead of `climax`. Key pins by
   **start time**.
3. **2-bar sections were silently merged away.** A hand-planned 2-bar break is
   exactly at the `min_section_s` threshold, so float noise decided; 48.37–52.08 s
   (3.7151 s vs a 3.7152 s minimum) was absorbed. Hand-planned grids now skip
   both the step-down split and the sliver merge.
4. **Pinning the shared hero shader was not enough.** Six other passes roll
   their own shader (drop-cycle look, section and drop markers, bare-canvas
   fill, ambient hold, each canvas's own roll), so pinned sections still carried
   Corduroy, Ink Bloom, Bokeh Drift, Ribbon Braid. Enforced in one late guard
   (`_g_pinned_section_shaders`) instead, which is also the only place it can be
   verified completely.
5. **The ambient detector claimed 30 seconds that had a beat.** It judges on
   absolute energy, so a half-energy passage with bass at 0.48–0.81 and hats
   running was called ambient, and 2,705 rhythmic hits were cleared. `SONG_AMBIENT`
   with `[]` fixed it. Check the bar map against any ambient claim.
6. **Verifiers read `analysis.json`, the generator read `SONG_SECTIONS`.** Until
   the plan was written into both, `verify_sections` was grading boundaries that
   no longer existed. Write the grid to both places.
7. **The synthetic pre-drop hush was too short.** `dark_start` is
   `max(cutout_end, cutout_start + 260 ms)`, so a window ending 80 ms before the
   drop gave 80 ms of black and every staged drop reported "no hush". Ending it
   450 ms out gives about a beat of darkness.

8. **The show kept running after the music stopped.** "December" goes silent at
   168.1s of a 172.1s file and the house then sat at mean 95-104 against 45-55
   during the song — brighter after the song than during it. The existing
   final-dropout logic cannot catch this: it needs 8 seconds of quiet tail to
   fire and there were only 4. Bluey had it too (audio ends 108.6s, house at
   76-80 until 112.1s). Now handled absolutely by `_g_dark_after_audio`, which
   runs LAST and lets nothing outlive the audio by more than 250ms. **Always
   check the tail of a new render against where the audio actually ends.**
9. **Per-section pins only bind where you bind them.** A pinned shader needs the
   late guard (`_g_pinned_section_shaders`); a pinned palette binds in the
   palette block. If a new kind of pin is added, ask which passes can override
   it before believing the log.

Two things that LOOK like a mis-pinned shader and are not — check `X_Src` and
`X_AmbientSlowed` before chasing either:

* an **ambient hold** (`X_AmbientSlowed=1`) is deliberately exempt from the pin,
  because its slowed speed is the point. On "December" one appeared 2.6s before
  the breakdown, so the tail of the preceding chorus ran the ambient shader.
* a **boundary marker** is placed at the snapped boundary, which can be ~170ms
  before the planned time, so it lands in the previous section when you bucket
  effects by start time. On "December" that made the climax look like it carried
  a stray Frost Growth when it was the next section's marker.

Also: `verify_sections` measures the **aggregate house colour and level**, so a
boundary can change both palette and shader and still report `colour 0.002`.
"December"'s 85.34s verse-to-build boundary does change (ice + Rain on Glass to
velvet + Velvet Nap) but moves the mix almost not at all — a real change that
will read subtly. Confirm with a per-section pin probe before believing
"NOT MARKED".

## Measurement traps that will waste an hour if forgotten

- `.fseq` pixels are stored **B, R, G** — decode `raw[:, [1, 2, 0]]` before any
  hue maths, or every colour reading is 120° out.
- Read `.fseq` at **raw** channel offsets (`frame[first-1:last]`); prove any new
  probe with a one-prop file where every other prop must read exactly 0.
- **Layer 0 renders on top**, and an effect's dark pixels are transparent — put
  an `Off` on a lower layer to make a prop its own canvas.
- Props belong to ~16 nested groups; expand membership recursively before
  asking "what is lighting this prop".
- Refresh `channel_map` (and re-render) after any layout change, or pixel
  measurements silently read the wrong prop.

## 9. A DJ MIX as one sequence (the medley path, Jackson Lights 2026)

A 28:49 continuous mix of fifteen songs cannot be generated as one "song": the
generator assumes one tempo (dozens of passes use a single median beat), and the
mix ramps tempo between songs. So each song is cut out with +-10 s of padding,
sequenced on its own terms, and stitched back.

```bash
# 1. whole-mix beat + downbeat grid (madmom joint DBN, follows tempo ramps)
.venv/bin/python3 custom_tools/medley_identify.py "<mix>" beats
# 2. identify songs: Shazam every 10 s + Whisper lyrics; ask the owner for
#    the ones Shazam cannot name (remixes, instrumentals, unreleased edits)
uv run --python 3.12 --with shazamio python custom_tools/medley_identify.py "<mix>" shazam
uv run --with openai-whisper python custom_tools/medley_identify.py "<mix>" lyrics
# 3. cut points inside each crossfade, on a downbeat -> segments.json
#    (name, artist, start, end, audio_start, audio_end, song=".jl26/<name>")
# 4. per-song analysis + cues on each padded slice, then the mix grid:
.venv/bin/python3 custom_tools/medley_prep.py "<mix>" beats
# 5. read each song's structure IN MIX TIME, write the design file
#    custom_tools/medleys/<slug>.py (sections, looks, palettes, shader pool,
#    drops, climax, ambient, caps, reveal, outro), then put it on the beat:
.venv/bin/python3 custom_tools/refine_boundaries.py "<mix>" ["NN "]   # look first
.venv/bin/python3 custom_tools/refine_boundaries.py "<mix>" ["NN "] --write
.venv/bin/python3 custom_tools/medley_prep.py "<mix>" design
# 6. generate every song (each writes its own xsq; ~10 s - 3 min each)
.venv/bin/python3 custom_tools/generate_song.py ".jl26/<NN name>" --out ".../xsq/<NN name>.xsq"
# 7. stitch, render (~20 min for 29 min of show), measure
.venv/bin/python3 custom_tools/merge_medley.py "<mix>" --out "<show>/<mix> (vN).xsq"
/Applications/xLights.app/Contents/MacOS/xLights -s "<show>" -r "<show>/<mix> (vN).xsq"
.venv/bin/python3 custom_tools/measure_medley.py "<mix>" "<mix> (vN)"
```

To check one song's change without a 20-minute render, render that song's
own segment xsq (copied into the show folder) -- about 2 minutes -- and read
it at song time (mix time minus `audio_start`).

What the design file controls, per song (all times in MIX time, converted by
medley_prep): `sections`, `section_look` (palette + shader per section start),
`palettes` (the song's own colours), `palette_family`, `shader_pool` (the
shaders random picks may use -- split so NO shader is used by two songs),
`drops`, `climax` (name the section START; prep puts it 1 s inside), `ambient`,
`caps` (brightness range over a window: the warm-up at 10-20 %, the organ at
30-50 %), `reveal` (exact hits, break, drop, two contrasting colours split by
side of the house), `outro` (`no_flash_from`, `fade`), `groove`/`pulse_mode`.

Lessons, each measured on this mix:

- **Drops and ambient come ONLY from the design in a medley.** Left to the
  detectors, the drumless Nils Frahm opener got a staged whole-house white hit
  at 0:01.9. An omitted `drops` is now an empty list.
- **Hand-set drops are not re-snapped in a medley.** The generator's kick snap
  (first strong hit up to 2-3.5 s later) moved 6 of 21 verified drops a beat or
  two late -- the launch reveal by +1.16 s. It was right on 4 (Bluey x2, greedy,
  Light Of Day), so those were corrected in the design instead. Check a drop by
  looking at the low band at 0.1 s resolution around it (a sparkline per drop
  is enough), not by trusting any one detector.
- **Find a precise hit pattern on the drum stem, not the full mix.** The six
  reveal hits of Be With You (2:58.160 ... 2:59.288, 0.225 s apart) and the
  syncopated drop at 2:59.732 only resolved on a demucs `--two-stems drums`
  render of a 20 s slice; every full-mix detector smeared them.
- **The bar maps used to drift.** song_map/section_plan laid a uniform grid at
  4 x median beat; madmom's 10 ms frames quantize that median (480 ms for a
  486 ms beat) and by bar 40 the grid was more than a beat off. Both now read
  bar lines off the tracked downbeats (`bar_lines()`); medley_prep's tempo is
  the mean of the gaps within 8 % of the median (a line fit read Be With You's
  slow organ intro as 113.5 BPM).
- **The tracked downbeat phase is NOT reliable through a mix**, so do not snap
  boundaries to downbeats blindly: inside greedy the nearest "downbeat" sat two
  beats either side of the real turns. `refine_boundaries.py` tests every beat
  within +-1.3 s and moves a time only when one beat is decisively the turn
  (linear level difference over a bar each side). A log ratio over two beats
  was tried first and preferred the end of a SILENCE over the drop after it.
- **Keep the per-song beat grid from the whole mix.** The per-slice analysis
  takes beats from librosa's single global tempo and downbeats from madmom,
  independently -- on a DJ slice librosa read Bluey at 172 BPM.
- **A medley's tempo also changes INSIDE a song**: the last bars of Aperture
  stretch from 1.87 s to 2.04 s as it hands over to Light Of Day.
- **Nested song names need a stems symlink** (`stems/.jl26 -> .jl26/stems`):
  the separator writes beside the audio, the consumers read `stems/<song>`.
- **Shader pools narrow random picks only**; pinned looks are validated against
  every registered shader (`ALL_SHADER_KEYS`), or one song's pool rejects
  another song's pin.
- **A guard that removes effects and then adds some must remove IN PLACE.**
  `add()` appends to the generator's own list (a closure), not to the guard's
  `effects` parameter. The reveal guard did `effects = drop_effects(...)` and
  then placed 570 hits, the break and the drop hit -- the log counted them, the
  render measured **0% lit on all six hits**. Use `effects[:] = drop_effects(...)`.
  (A scan of every `_g_` guard found no other instance.)
- **End-of-song behaviour fires on the LAST song of a mix.** The final dropout
  saw the last drum hit at 27:09 and cleared 1,856 effects, so the strings
  outro ran on 18 dim outlines; dark-after-audio then cut the fade short. A
  song with a designed `outro` now ends the way the design says.
- **Never put a look change on the first frame of a fade.** It measured as a
  dip (lit 20% -> 4% -> 9%) while the new beds faded in.
- **Measure the render, not the log.** The log said "570 placements" for the
  reveal; only the .fseq showed they were gone. `custom_tools/measure_medley.py`
  checks every promise of the design (per-song lit/level/movement, each
  brightness cap, every reveal hit with its colours, outro flashes and the fade)
  and streams the multi-GB render through `fseq_stream.py`. A 29-minute render
  takes ~20 minutes and ~5-6 GB of xLights memory.
- **Pick each section's shader by MEASURED brightness.** A shader's own
  content sets how bright a section reads far more than the brightness curve:
  on the first full render nine songs had a chorus or drop dimmer than the
  verse or break beside it, and the launch chorus sat at 0.37 on a dark-sky
  shader while an ordinary chorus reached 0.54. `measure_medley.py` prints
  every section's display mean with its shader and flags a chorus/drop that
  reads dimmer than the song's quiet sections (`!`). Brightness also depends
  on the section, not only the shader -- re-measure after re-pinning.
- **A whole-house hit on the big canvases must not be a flash.** `add()`
  turns a flat hit on any prop too big to flash into a travelling burst, so
  the reveal's six hits lit the Mega Tree 12%. Six full-canvas swaps in 1.1 s
  would be the strobe that rule exists for; instead the big canvases FILL a
  sixth per hit (Fill, fixed position, held) and land the drop as a swell
  (fade-out >= 0.8 s, which the rule allows). Measured: 45 -> 94% of the
  house lit across the six hits, 95.5% on the drop, gold and blue both present.
- **"Movement but not flashing" needs movement that never switches off.**
  Static washes read as frozen (strings outro: display 0.06, 25% lit). One
  continuous Butterfly per non-canvas prop, never restarted, at speed 25 (10
  barely moves), swelling with the music, plus the section shader filling any
  gap on the canvases (only the canvases -- on the small Orbs the shader
  pulsed like a flash). Measured: 70-85% lit, 0 prop flashes, smooth fade.
- **Switched-off props are not dark props.** 50 models in this layout are
  Active="0"; counting them made a fully lit house measure as half dark.
- **Hand-choreographed scenes for the moments a show is built around.** The
  passes make good songs; they cannot make a teaser, an organ, a frenzy or a
  photo. A design `scenes` entry owns its window outright (cleared first, like
  the reveal) and places named gestures: `teaser` (one gesture per synth
  swell, full brightness, ~1-20% of the house lit), `organ` (verticals as
  organ pipes re-registered on each chord, grayscale outlines, a rhythmic
  build), `frenzy` (six colour-pair hits, a white drop, half-house strobing),
  `works` (a whole chorus bouncing, leaping, blooming, popping, spinning) and
  `horizon` (a swaying high-contrast photo, no flashing, swell then fade).
  Swells come from a 1-2 s smoothed RMS (peaks with prominence > 0.06);
  chords from chroma novelty between 1 s windows.
- **"Small share lit" needs checking per buffer.** Bars' highlight paints a
  white edge on every band, black ones too (a "thin" slit lit 55-70% of a
  matrix); a twinkle on a Per Preview group lights whole canvas cells (35% of
  the house at Count 3). Highlight off; twinkle per model.
- **A single-line prop's buffer is one row along the strand**: Fill Up is all
  or nothing on a vertical. Fill Right rises from node 1, which on these
  top-down `Dir=R` strips is the ground.
- **Thin is not rhythmic.** A one-dot chase per beat left the verticals,
  arches and roofline averaging 0.04-0.09 -- fewer props pulsing (26/84) than
  the generated chorus it replaced (52/84). A level-meter fill on the
  verticals, wide-band leaps (`E_SLIDER_Color_Mix1` 35-50) and sphere
  fill/drain brought it to 62/84. Measure per-beat swing per prop.
- **Use the proven arch chase for anything directional on the arches.** Fill
  on the three-row arches went opposite ways on two arches with the same
  `Dir`; the wiring-compensated SingleStrand chase is consistent.
- **Sync to the music's own events, at the right resolution.** Chord changes
  from 1 s chroma windows found half of the organ's changes, and a 0.6 s glide
  then landed each one most of a second late. 0.4 s windows, each change
  snapped to the nearest harmonic onset, and a 0.12 s snap: the pipes now
  change 35 ms after the chord (median). Bow strokes are the small swells of
  the strings' 300-3000 Hz envelope (a stroke every 0.55-0.75 s).
- **Drive a shader's motion by POSITION, not by its clock.** Reversing the
  shader speed curve on every bow stroke did make it swing per stroke, but the
  clock integrates, so stroke-to-stroke error drifted the image to one side
  within 15 s. xLights accepts a value curve on a shader's own float input
  (`E_VALUECURVE_SHADERXYZZY_<name>`, range = the slider's x100 range), so a
  custom shader can expose the gesture (`bow`, -1..1) and the curve sets it
  exactly -- no drift, and holding the curve holds the image still.
- **Custom shaders for a scene live in SHOW/Shaders and are built in the scene**
  (`shader_settings(...)` inline), never added to SHADER_PICKS, so no ordinary
  song can pick them at random. Colour inputs take the palette in order.
- **Following every musical event is not the same as following the music.**
  Organ pipes re-registered on each chord change (every 1-1.5 s) and a sway
  reversing on each bow stroke (every 0.6 s) were both accurate and both read
  as "too fast and messy". What Krists wanted for a slow, sustained passage
  was STILLNESS that BREATHES: steady outlines swelling once per bar, still
  patterned colour swelling with a 2 s envelope of the music, and motion only
  as slow abstract texture on the big canvases. Sync the brightness to the
  music; keep the motion slower than the music.
- **A big-canvas hit reads better as a pattern than a slab.** Six flat colour
  hits became hard two-colour stripes (static Bars, 0 cycles, 4 repeats), the
  orientation turning each hit, the sixth a rainbow.
- **Tune a shader's colour balance by measuring it.** Dawn Nebula's gold went
  10% -> 55% -> ~33% with two threshold changes; fbm values bunch near 0.5,
  so small threshold shifts swing coverage a lot.
- **The owner's ear beats every detector on WHERE things happen.** Three of my
  first guesses were wrong (organ entry, the reveal, a guitar build read as a
  drop); ask about the two or three moments the show is built around, then
  measure them exactly.
