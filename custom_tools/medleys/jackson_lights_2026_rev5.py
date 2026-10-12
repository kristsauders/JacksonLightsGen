"""Jackson Lights 2026 rev5 -- 25:21.6, 16 songs. Times are MIX time in
"Jackson Lights 2026 rev5.wav" (the LANDR-mastered file plays; same timing).

rev5 = rev3 with ENIGMA, Every Night, Liquid Spirit, Movin' To The Sun, Glad I
Found You and What Love Can Do removed, Happy Holidays moved after Never Forget
You, MGMT's Kids added before Light Of Day, and a new Kids -> Light Of Day
transition. Measured against rev3 (chroma, 2 s steps):

    0 - 730.5      identical (Ode .. NUEVAYoL, with the countdown)
    Santa          rev3 - 435.51, constant
    Never Forget   rev3 - 436.1 -> -437.95: its drumless intro ~1.7 s shorter,
                   then ~0.7% faster -- copied through the measured map
    Happy Holidays rev3 - 127.25 (drifting 0.2 s); its quiet vocal intro now runs
                   30 s (923.5 - 956.0) against rev3's 10.5
    Sleigh Ride    rev3 - 334.68;  Deck + Shokoufeh rev3 - 335.33 (a 0.65 s edit
                   at the Deck joint)
    Kids           new, 1286.34 - 1356.86
    Light Of Day   rev3 - 625.30 through the climax; the strings stop at 1461.44
                   (rev3 2086.83), so the finale is rev3 - 625.40 -- each part a
                   CONSTANT offset, so the spinners still freeze in place

Every song that was already done is COPIED from rev3's finished per-song
sequences through those maps (segments.json `time_map`), so the visuals and their
timing are exactly what was approved. Generated here: Kids, and Happy Holidays'
longer intro (923.5 - 945.5, in its rev3 intro look; rev3's own visuals take over
from there).
"""

DESIGN: dict[str, dict] = {
    # =====================================================================
    # 11  Happy Holidays -- ONLY its new long intro is used from this generation
    # (Bing Crosby's vocal over pads, no beat) -- the rev3 intro look, ambient.
    # =====================================================================
    '11 Happy Holidays': {'palettes': {'fireside': ['#A4123F', '#D4A017', '#FFF3D6'],
                                       'tinsel': ['#D4A017', '#FFF3D6', '#2E6B3F']},
                          'palette_family': ['fireside', 'tinsel'],
                          'shader_pool': ['velvet_nap', 'oil_film', 'marble_vein'],
                          'sections': [(923.50, 'intro'), (954.05, 'build'), (956.03, 'chorus')],
                          'section_look': {923.50: {'palette': 'tinsel', 'shader': 'shader_velvet_nap'},
                                           954.05: {'palette': 'tinsel', 'shader': 'shader_oil_film'},
                                           956.03: {'palette': 'fireside', 'shader': 'shader_oil_film'}},
                          'ambient': [(923.50, 954.05)],
                          'drops': [956.03]},

    # =====================================================================
    # 15  MGMT - Kids   21:26.3 - 22:36.9   123 BPM
    # "added Kids by MGMT near the end to help with the build up since it's such
    # a high energy song". Shokoufeh stops dead at 21:26.3 -> four quiet bars of
    # the synth -> the band and THE riff at 21:34.3; 8-bar phrases; the hats climb
    # from 22:21.1; Light Of Day's build takes over at 22:36.9.
    # Neon psychedelia (the video), on shaders freed by the songs that left: Heart
    # Lattice (the geometric pulse Krists liked in What Love Can Do, full + soft),
    # Ink Bloom, Interference. The second half gets the works (0.9: Light Of Day
    # stays the peak).
    # =====================================================================
    '15 Kids': {'palettes': {'kids': ['#FF2E97', '#00E5FF', '#FFE600'],
                             'acid': ['#7CFF00', '#FF00D4', '#3A0CA3']},
                'palette_family': ['kids', 'acid'],
                'shader_pool': ['heart_lattice', 'heart_full', 'ink_bloom', 'interference'],
                'extra_shaders': ['Heart Lattice',
                                  {'file': 'Heart Lattice', 'key': 'heart_full',
                                   'overrides': {'fill': 0.85, 'scale': 8.0, 'lines': 0.7}}],
                'sections': [(1286.34, 'intro'),
                             (1294.30, 'chorus'),
                             (1309.92, 'verse'),
                             (1325.53, 'chorus'),
                             (1341.14, 'build')],
                'section_look': {1286.34: {'palette': 'acid', 'shader': 'shader_ink_bloom'},
                                 1294.30: {'palette': 'kids', 'shader': 'shader_heart_full'},
                                 1309.92: {'palette': 'acid', 'shader': 'shader_ink_bloom'},
                                 1325.53: {'palette': 'kids', 'shader': 'shader_heart_full'},
                                 1341.14: {'palette': 'acid', 'shader': 'shader_interference'}},
                'ambient': [(1286.34, 1294.30)],
                'drops': [1294.30, 1325.53],
                'climax': [1325.53],
                'scenes': [{'kind': 'works', 'window': [1325.53, 1356.86], 'level': 0.9,
                            'colors': ['#FF2E97', '#00E5FF', '#FFE600', '#7CFF00']}]},
}
