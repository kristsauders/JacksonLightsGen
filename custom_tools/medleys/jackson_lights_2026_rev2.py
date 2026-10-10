"""Jackson Lights 2026 rev2 -- the remixed launch-party medley, 30:35, 17 songs.

Times are MIX time in "Jackson Lights 2026 rev2.wav".

Songs 01-04 (Ode, Be With You, Bluey, 12 to 12 to 6:48.68) are the same audio at
the same times as the first mix, so their per-song sequences are COPIED from it
unchanged (segments.json marks them `reused`); they need no design here.

The rule for every cut (Krists): "for transitions keep the previous look so it
doesn't give it away" -- each song starts where it is FULLY in, after the
outgoing one has gone, never at the start of the crossfade.

Returning songs keep their earlier design, mapped onto this mix by the measured
audio alignment; new songs were identified with Shazam (+ Whisper lyrics) and
hand-mapped from the bar map. Shaders stay unique per song: the three songs
that left (greedy, the Santa trap, Glad I Found You) freed nine, and Shokoufeh
gets a custom one (Blossom Drift), registered for that song only.
"""

DESIGN: dict[str, dict] = {

    # =====================================================================
    # 05  The 'Last Christmas' vocal blend   6:48.7 - 8:06.1   (new place)
    # The old mix's frost passage (it sat inside Deck the Halls), now its own song
    # leading into Aperture. Frost and mistletoe; Aperture's build under the vocal
    # keeps THIS look until Aperture's drop -- the incoming song is not given away.
    # =====================================================================
    '05 Last Christmas': {'palettes': {'frost': ['#BDE3FF', '#FFFFFF', '#C8102E'],
                  'mistletoe': ['#C8102E', '#FFFFFF', '#0B6623']},
     'palette_family': ['frost', 'mistletoe'],
     'shader_pool': ['frost_growth', 'crackle_glaze'],
     'sections': [(408.68, 'verse'),
                  (424.17, 'verse'),
                  (439.65, 'chorus'),
                  (455.14, 'break'),
                  (466.75, 'build')],
     'section_look': {408.68: {'palette': 'frost', 'shader': 'shader_frost_growth'},
                      424.17: {'palette': 'mistletoe', 'shader': 'shader_crackle_glaze'},
                      439.65: {'palette': 'frost', 'shader': 'shader_frost_growth'},
                      455.14: {'palette': 'mistletoe', 'shader': 'shader_crackle_glaze'},
                      466.75: {'palette': 'frost', 'shader': 'shader_frost_growth'}},
     'drops': []},

    # =====================================================================
    # 06  Harry Styles - Aperture        8:06.1 - 10:17.5    extended edit
    # Same colours/shaders as before. Drops: 'We belong together' 8:06.1, 'Let it
    # all out' 8:37.1, the added third chorus 9:15.7 (climax); fades to 10:11.7.
    # =====================================================================
    '06 Aperture': {'palettes': {'aperture': ['#FFF4E0', '#1E90FF', '#C0C8D0'],
                  'lens': ['#1E90FF', '#FFFFFF', '#7FB2FF']},
     'palette_family': ['aperture', 'lens'],
     'shader_pool': ['mobius_spiral', 'spacels', 'woven_bands'],
     'sections': [(486.1, 'chorus'),
                  (509.33, 'verse'),
                  (517.07, 'chorus'),
                  (540.3, 'break'),
                  (553.85, 'build'),
                  (555.74, 'chorus'),
                  (611.74, 'outro')],
     'section_look': {486.1: {'palette': 'aperture', 'shader': 'shader_mobius_spiral'},
                      509.33: {'palette': 'lens', 'shader': 'shader_spacels'},
                      517.07: {'palette': 'aperture', 'shader': 'shader_mobius_spiral'},
                      540.3: {'palette': 'lens', 'shader': 'shader_woven_bands'},
                      553.85: {'palette': 'lens', 'shader': 'shader_woven_bands'},
                      555.74: {'palette': 'aperture', 'shader': 'shader_woven_bands'},
                      611.74: {'palette': 'lens', 'shader': 'shader_woven_bands'}},
     'drops': [486.1, 517.07, 555.74],
     'climax': [555.74]},

    # =====================================================================
    # 07  Gigi D'Agostino - Star         10:17.5 - 11:05.4   new bridge
    # 1999 Italo-dance from L'Amour Toujours: dreamy, starry. Indigo, cyan, white.
    # =====================================================================
    '07 Star': {'palettes': {'stardust': ['#2A1B7A', '#FFFFFF', '#3BD6FF'],
                  'nova': ['#3BD6FF', '#B388FF', '#FFFFFF']},
     'palette_family': ['stardust', 'nova'],
     'shader_pool': ['caustic_pool', 'ribbon_braid', 'wind_wheat'],
     'sections': [(617.5, 'verse'), (630.83, 'chorus'), (647.97, 'chorus'), (661.3, 'outro')],
     'section_look': {617.5: {'palette': 'nova', 'shader': 'shader_ribbon_braid'},
                      630.83: {'palette': 'stardust', 'shader': 'shader_caustic_pool'},
                      647.97: {'palette': 'nova', 'shader': 'shader_wind_wheat'},
                      661.3: {'palette': 'stardust', 'shader': 'shader_ribbon_braid'}},
     'drops': [630.83]},

    # =====================================================================
    # 08  Bad Bunny - NUEVAYoL   11:05.4 - 12:12.7   identical audio, old design -44.5 s
    # =====================================================================
    '08 NUEVAYoL': {'palettes': {'boricua': ['#E4002B', '#0050F0', '#FFFFFF'],
                  'nuevayol': ['#FFC72C', '#E4002B', '#0050F0']},
     'palette_family': ['boricua', 'nuevayol'],
     'shader_pool': ['circuit_trace', 'herringbone', 'fracture'],
     'sections': [(665.4, 'intro'), (679.59, 'chorus'), (697.39, 'verse'), (712.64, 'verse')],
     'section_look': {665.4: {'palette': 'nuevayol', 'shader': 'shader_herringbone'},
                      679.59: {'palette': 'boricua', 'shader': 'shader_fracture'},
                      697.39: {'palette': 'nuevayol', 'shader': 'shader_circuit_trace'},
                      712.64: {'palette': 'boricua', 'shader': 'shader_fracture'}},
     'drops': [679.59],
     'climax': [679.59],
     'pulse_mode': 'groove'},

    # =====================================================================
    # 09  CloZee & LSDREAM - ENIGMA   12:12.7 - 14:21.3   identical, old design -47.6 s
    # =====================================================================
    '09 ENIGMA': {'palettes': {'enigma': ['#7CFF00', '#5B0E91', '#00FFC6'],
                  'ritual': ['#5B0E91', '#B6FF3B', '#FF7A00']},
     'palette_family': ['enigma', 'ritual'],
     'shader_pool': ['murmuration', 'ink_bloom', 'interference', 'hex_vortex'],
     'sections': [(732.66, 'intro'),
                  (759.8, 'chorus'),
                  (776.51, 'break'),
                  (779.6, 'chorus'),
                  (800.06, 'break'),
                  (814.94, 'build'),
                  (819.2, 'chorus'),
                  (834.66, 'verse'),
                  (856.94, 'outro')],
     'section_look': {732.66: {'palette': 'ritual', 'shader': 'shader_murmuration'},
                      759.8: {'palette': 'enigma', 'shader': 'shader_hex_vortex'},
                      776.51: {'palette': 'ritual', 'shader': 'shader_interference'},
                      779.6: {'palette': 'enigma', 'shader': 'shader_ink_bloom'},
                      800.06: {'palette': 'ritual', 'shader': 'shader_murmuration'},
                      814.94: {'palette': 'ritual', 'shader': 'shader_interference'},
                      819.2: {'palette': 'enigma', 'shader': 'shader_hex_vortex'},
                      834.66: {'palette': 'ritual', 'shader': 'shader_ink_bloom'},
                      856.94: {'palette': 'enigma', 'shader': 'shader_interference'}},
     'ambient': [(732.66, 757.99)],
     'drops': [759.8, 779.6, 819.2],
     'climax': [819.2]},

    # =====================================================================
    # 10  Spectacle - Every Night   14:21.3 - 17:49.2   re-edited (offset drifts -45 -> -50 s):
    # old design mapped through the measured alignment, then refined to rev2's beats
    # =====================================================================
    '10 Every Night': {'palettes': {'nocturne': ['#2B1B8F', '#8A2BE2', '#00D4FF'],
                  'afterglow': ['#00D4FF', '#6A00FF', '#FF5EDB']},
     'palette_family': ['nocturne', 'afterglow'],
     'shader_pool': ['drift_cells', 'contour_drift', 'tide_lines', 'pulse_rings'],
     'sections': [(861.28, 'verse'),
                  (893.30, 'chorus'),
                  (917.77, 'break'),
                  (927.59, 'break'),
                  (941.09, 'build'),
                  (952.69, 'verse'),
                  (972.98, 'break'),
                  (982.16, 'chorus'),
                  (1000.5, 'verse'),
                  (1019.48, 'chorus'),
                  (1056.19, 'outro')],
     'section_look': {861.28: {'palette': 'nocturne', 'shader': 'shader_pulse_rings'},
                      893.30: {'palette': 'afterglow', 'shader': 'shader_drift_cells'},
                      917.77: {'palette': 'nocturne', 'shader': 'shader_tide_lines'},
                      927.59: {'palette': 'nocturne', 'shader': 'shader_contour_drift'},
                      941.09: {'palette': 'afterglow', 'shader': 'shader_tide_lines'},
                      952.69: {'palette': 'nocturne', 'shader': 'shader_pulse_rings'},
                      972.98: {'palette': 'nocturne', 'shader': 'shader_contour_drift'},
                      982.16: {'palette': 'afterglow', 'shader': 'shader_drift_cells'},
                      1000.5: {'palette': 'nocturne', 'shader': 'shader_tide_lines'},
                      1019.48: {'palette': 'afterglow', 'shader': 'shader_drift_cells'},
                      1056.19: {'palette': 'nocturne', 'shader': 'shader_contour_drift'}},
     'ambient': [(927.59, 941.09)],
     'drops': [982.16],
     'climax': [1019.48]},

    # =====================================================================
    # 11  Happy Holidays (Beef Wellington)   17:49.2 - 19:23.1   old design mapped (~-49.7 s)
    # =====================================================================
    '11 Happy Holidays': {'palettes': {'fireside': ['#A4123F', '#D4A017', '#FFF3D6'],
                  'tinsel': ['#D4A017', '#FFF3D6', '#2E6B3F']},
     'palette_family': ['fireside', 'tinsel'],
     'shader_pool': ['velvet_nap', 'oil_film', 'marble_vein'],
     'sections': [(1069.18, 'intro'),
                  (1077.71, 'build'),
                  (1079.69, 'chorus'),
                  (1099.97, 'verse'),
                  (1121.59, 'chorus'),
                  (1139.95, 'break'),
                  (1142.59, 'chorus'),
                  (1158.3, 'outro')],
     'section_look': {1069.18: {'palette': 'tinsel', 'shader': 'shader_velvet_nap'},
                      1077.71: {'palette': 'tinsel', 'shader': 'shader_oil_film'},
                      1079.69: {'palette': 'fireside', 'shader': 'shader_oil_film'},
                      1099.97: {'palette': 'tinsel', 'shader': 'shader_marble_vein'},
                      1121.59: {'palette': 'fireside', 'shader': 'shader_oil_film'},
                      1139.95: {'palette': 'tinsel', 'shader': 'shader_velvet_nap'},
                      1142.59: {'palette': 'fireside', 'shader': 'shader_oil_film'},
                      1158.3: {'palette': 'tinsel', 'shader': 'shader_velvet_nap'}},
     'ambient': [(1069.18, 1077.71)],
     'drops': [1079.69],
     'climax': [1142.59]},

    # =====================================================================
    # 12  Santa Claus Is Coming to Town   19:23.1 - 20:24.7   old design -49.9 s;
    # its second drop now opens Deck the Halls (song 15)
    # =====================================================================
    '12 Santa Claus Is Coming to Town': {'palettes': {'candycane': ['#FF0A3C', '#FFFFFF', '#FF7A9A'],
                  'peppermint': ['#FFFFFF', '#FF0A3C', '#00B37A']},
     'palette_family': ['candycane', 'peppermint'],
     'shader_pool': ['kaleido_bloom', 'target_circle', 'innerdimensionalmatrix'],
     'sections': [(1163.09, 'verse'),
                  (1178.37, 'build'),
                  (1189.88, 'break'),
                  (1193.71, 'chorus'),
                  (1209.08, 'chorus'),
                  (1222.5, 'break')],
     'section_look': {1163.09: {'palette': 'peppermint', 'shader': 'shader_target_circle'},
                      1178.37: {'palette': 'peppermint', 'shader': 'shader_kaleido_bloom'},
                      1189.88: {'palette': 'candycane', 'shader': 'shader_target_circle'},
                      1193.71: {'palette': 'candycane',
                                'shader': 'shader_innerdimensionalmatrix'},
                      1209.08: {'palette': 'peppermint', 'shader': 'shader_kaleido_bloom'},
                      1222.5: {'palette': 'candycane', 'shader': 'shader_target_circle'}},
     'drops': [1193.71],
     'climax': [1193.71]},

    # =====================================================================
    # 13  Zara Larsson & MNEK - Never Forget You (guaracha edit)  20:24.7 - 22:38.3
    # ~145 BPM Latin-house: coral, turquoise, sunshine and neon. Drops after a
    # one-bar silence at 21:18.85 and 22:11.8 (the peak).
    # =====================================================================
    '13 Never Forget You': {'palettes': {'guaracha': ['#FF5A36', '#00D1C1', '#FFD23F'],
                  'neon': ['#FF2BD6', '#00D1C1', '#FFFFFF']},
     'palette_family': ['guaracha', 'neon'],
     'shader_pool': ['iso_blocks', 'liquid_chrome', 'chromatophore'],
     'sections': [(1224.73, 'intro'),
                  (1252.37, 'verse'),
                  (1278.85, 'chorus'),
                  (1306.99, 'verse'),
                  (1331.81, 'chorus')],
     'section_look': {1224.73: {'palette': 'neon', 'shader': 'shader_chromatophore'},
                      1252.37: {'palette': 'guaracha', 'shader': 'shader_liquid_chrome'},
                      1278.85: {'palette': 'guaracha', 'shader': 'shader_iso_blocks'},
                      1306.99: {'palette': 'neon', 'shader': 'shader_liquid_chrome'},
                      1331.81: {'palette': 'neon', 'shader': 'shader_iso_blocks'}},
     'drops': [1278.85, 1331.81],
     'climax': [1331.81],
     'pulse_mode': 'groove'},

    # =====================================================================
    # 14  The Ronettes - Sleigh Ride (dance edit)   22:38.3 - 23:46.5
    # Spector's 1963 jingles and hooves over a beat: sweater-knit shaders (argyle,
    # tartan, cable knit) in sleigh red/gold and evergreen. Snare roll -> drop 23:04.5.
    # =====================================================================
    # cut time: the tracker hears 182 BPM, the felt pulse is 91
    '14 Sleigh Ride': {'half_time': True,
                       'palettes': {'sleigh': ['#E3001B', '#FFFFFF', '#FFC82E'],
                  'evergreen': ['#0E8A3E', '#FFFFFF', '#E3001B']},
     'palette_family': ['sleigh', 'evergreen'],
     'shader_pool': ['argyle_march', 'tartan_shift', 'cable_knit'],
     'sections': [(1358.27, 'intro'),
                  (1371.42, 'build'),
                  (1384.51, 'chorus'),
                  (1405.49, 'chorus')],
     'section_look': {1358.27: {'palette': 'evergreen', 'shader': 'shader_cable_knit'},
                      1371.42: {'palette': 'sleigh', 'shader': 'shader_tartan_shift'},
                      1384.51: {'palette': 'sleigh', 'shader': 'shader_argyle_march'},
                      1405.49: {'palette': 'evergreen', 'shader': 'shader_tartan_shift'}},
     'ambient': [(1358.27, 1371.42)],
     'drops': [1384.51],
     'climax': [1384.51]},

    # =====================================================================
    # 15  Deck the Halls (J3, eXecutive Decision, Kevin Aleksander)  23:46.5 - 25:53.0
    # Opens on the old Santa Claus second drop (a 5.6 s reprise), then Deck's intro
    # 23:52.1, drops 24:24.0 and 24:39.0 (climax), fades from 25:41. The Last
    # Christmas passage that followed it in the old mix is now song 05.
    # =====================================================================
    '15 Deck the Halls': {'palettes': {'holly': ['#0B6623', '#FFD700', '#C8102E'],
                  'garland': ['#C8102E', '#0B6623', '#FFF5E1']},
     'palette_family': ['holly', 'garland'],
     # v1 measured the drops/choruses at 0.12-0.26 -- no brighter than the intro:
     # ember_swarm is sparse points. Garland Twist (written for this song) fills
     # the canvas with braided ribbons for the hits; ember_swarm keeps the builds.
     'shader_pool': ['garland_twist', 'ember_swarm', 'lava_lamp'],
     'extra_shaders': ['Garland Twist'],
     'sections': [(1426.49, 'chorus'),
                  (1432.13, 'intro'),
                  (1454.63, 'build'),
                  (1458.37, 'build'),
                  (1463.99, 'chorus'),
                  (1477.12, 'break'),
                  (1478.97, 'chorus'),
                  (1509.01, 'chorus'),
                  (1541.05, 'outro')],
     'section_look': {1426.49: {'palette': 'holly', 'shader': 'shader_garland_twist'},
                      1432.13: {'palette': 'garland', 'shader': 'shader_lava_lamp'},
                      1454.63: {'palette': 'garland', 'shader': 'shader_ember_swarm'},
                      1458.37: {'palette': 'holly', 'shader': 'shader_lava_lamp'},
                      1463.99: {'palette': 'holly', 'shader': 'shader_garland_twist'},
                      1477.12: {'palette': 'garland', 'shader': 'shader_lava_lamp'},
                      1478.97: {'palette': 'holly', 'shader': 'shader_garland_twist'},
                      1509.01: {'palette': 'garland', 'shader': 'shader_garland_twist'},
                      1541.05: {'palette': 'garland', 'shader': 'shader_lava_lamp'}},
     'drops': [1426.49, 1463.99, 1478.97],
     'climax': [1478.97]},

    # =====================================================================
    # 16  Bonobo - Shokoufeh ('blossom')   25:53.0 - 26:58.1
    # From Distance In Static (2026). A custom Blossom Drift shader: petals drifting
    # and turning over a slow wash, in pinks over teal / dusk violet.
    # =====================================================================
    '16 Shokoufeh': {'palettes': {'blossom': ['#FF7FA8', '#FFD6E7', '#2EC4B6'],
                  'dusk': ['#FF7FA8', '#FFC27A', '#6B2D9C']},
     'palette_family': ['blossom', 'dusk'],
     # v1 measured the last chorus (0.14) dimmer than the verses (0.18-0.21) at
     # full energy: the choruses get the same petals, denser and bigger, on a
     # brighter wash ("blossom_bloom"). The 4 s break before the last chorus is
     # under the 4 s min dwell, so its span merges INTO the chorus and lends it
     # its shader: the break carries the chorus look (its curve keeps it dark).
     'shader_pool': ['blossom_drift', 'blossom_bloom'],
     'extra_shaders': ['Blossom Drift',
                       {'file': 'Blossom Drift', 'key': 'blossom_bloom',
                        'overrides': {'density': 0.85, 'size': 0.55, 'wash': 0.7}}],
     'sections': [(1552.99, 'intro'),
                  (1557.05, 'verse'),
                  (1569.25, 'chorus'),
                  (1585.53, 'verse'),
                  (1589.59, 'chorus'),
                  (1601.8, 'break'),
                  (1605.87, 'chorus')],
     'section_look': {1552.99: {'palette': 'dusk', 'shader': 'shader_blossom_drift'},
                      1557.05: {'palette': 'blossom', 'shader': 'shader_blossom_drift'},
                      1569.25: {'palette': 'dusk', 'shader': 'shader_blossom_bloom'},
                      1585.53: {'palette': 'blossom', 'shader': 'shader_blossom_drift'},
                      1589.59: {'palette': 'dusk', 'shader': 'shader_blossom_bloom'},
                      1601.8: {'palette': 'dusk', 'shader': 'shader_blossom_bloom'},
                      1605.87: {'palette': 'blossom', 'shader': 'shader_blossom_bloom'}},
     'drops': [1569.25]},

    # =====================================================================
    # 17  ODESZA - Light Of Day   26:58.1 - 30:35.0   identical, old design +111.2 s:
    # the final chorus (works) and the strings finale (horizon) are the same scenes,
    # so the ending is the same. A new quiet intro 26:58.1-27:22.4 precedes the old start.
    # =====================================================================
    '17 Light Of Day': {'palettes': {'daybreak': ['#FFC857', '#FF9F80', '#87CEEB'],
                  'firstlight': ['#87CEEB', '#FFF1C1', '#FFB38A']},
     'palette_family': ['daybreak', 'firstlight'],
     'shader_pool': ['aurora_curtain', 'falling_embers', 'rain_on_glass', 'ripple_grid'],
     'sections': [(1618.08, 'intro'),
                  (1642.37, 'verse'),
                  (1675.53, 'build'),
                  (1682.19, 'break'),
                  (1687.76, 'chorus'),
                  (1724.42, 'chorus'),
                  (1739.99, 'outro')],
     'section_look': {1618.08: {'palette': 'firstlight', 'shader': 'shader_rain_on_glass'},
                      1642.37: {'palette': 'firstlight', 'shader': 'shader_rain_on_glass'},
                      1675.53: {'palette': 'firstlight', 'shader': 'shader_ripple_grid'},
                      1682.19: {'palette': 'daybreak', 'shader': 'shader_rain_on_glass'},
                      1687.76: {'palette': 'daybreak', 'shader': 'shader_ripple_grid'},
                      1724.42: {'palette': 'daybreak', 'shader': 'shader_ripple_grid'},
                      1739.99: {'palette': 'daybreak', 'shader': 'shader_aurora_curtain'}},
     'ambient': [(1618.08, 1642.37), (1739.99, 1835.0)],
     'drops': [1687.76],
     'climax': [1687.76],
     'outro': {'no_flash_from': 1739.99, 'fade': [1778.10, 1800.55]},
     'scenes': [{'kind': 'works', 'window': [1687.76, 1739.99]},
                {'kind': 'horizon',
                 'window': [1739.99, 1835.0],
                 'peak': 1775.36,
                 'fade': [1778.10, 1800.55],
                 'still_from': 1777.41,
                 'strokes': [1745.46,
                             1746.57,
                             1747.2,
                             1748.11,
                             1749.23,
                             1749.89,
                             1750.51,
                             1751.07,
                             1751.63,
                             1752.25,
                             1752.86,
                             1753.70,
                             1754.50,
                             1755.03,
                             1755.61,
                             1756.19,
                             1756.87,
                             1757.60,
                             1758.10,
                             1758.61,
                             1759.36,
                             1759.87,
                             1760.59,
                             1761.20,
                             1761.73,
                             1762.69,
                             1763.18,
                             1764.01,
                             1764.72,
                             1765.47,
                             1766.09,
                             1766.86,
                             1767.77,
                             1768.48,
                             1769.62,
                             1770.20,
                             1770.84,
                             1771.88,
                             1772.88,
                             1773.99,
                             1774.58,
                             1775.16,
                             1775.69,
                             1776.44]}]},
}
