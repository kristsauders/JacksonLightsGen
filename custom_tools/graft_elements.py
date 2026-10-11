"""Change only PART of an approved sequence, taking it from a fresh generation.

    python custom_tools/graft_elements.py <base.xsq> <donor.xsq> <out.xsq>
        [--rows "<model>" ...]                      whole rows (every effect)
        [--window A,B [--window-models "<m>|<m>"]]  every effect STARTING in
                                                    [A, B) seconds (xsq time)

Regenerating an approved song re-rolls passes that depend on anything that
moved in the generator since (Be With You's regenerated final guitar hit
gained a one-beat blackout), so: start from the approved file and replace
only what was asked -- named models' rows, or a time window (optionally only
on some models). Effects of the base that straddle a window edge are trimmed
to it; the donor's are trimmed the same way. Palette and EffectDB references
are positional, so the donor's are remapped into the base's tables.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("donor"); ap.add_argument("out")
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--window", action="append", default=[])
    ap.add_argument("--window-models", action="append", default=[])
    a = ap.parse_args()
    base = ET.parse(a.base); b = base.getroot(); d = ET.parse(a.donor).getroot()
    bp, bdb = b.find("ColorPalettes"), b.find("EffectDB")
    pal = [(c.text or "") for c in bp]; edb = [(c.text or "") for c in bdb]
    pix = {s: i for i, s in enumerate(pal)}; eix = {s: i for i, s in enumerate(edb)}
    dpal = [(c.text or "") for c in d.find("ColorPalettes")]
    dedb = [(c.text or "") for c in d.find("EffectDB")]

    def gp(s):
        if s not in pix:
            pix[s] = len(pal); pal.append(s); ET.SubElement(bp, "ColorPalette").text = s
        return pix[s]

    def ge(s):
        if s not in eix:
            eix[s] = len(edb); edb.append(s); ET.SubElement(bdb, "Effect").text = s
        return eix[s]

    def remap(ef):
        ef = ET.fromstring(ET.tostring(ef))
        if ef.get("ref") is not None:
            ef.set("ref", str(ge(dedb[int(ef.get("ref"))])))
        if ef.get("palette") is not None:
            ef.set("palette", str(gp(dpal[int(ef.get("palette"))])))
        return ef

    bee = b.find("ElementEffects")
    bel = {e.get("name"): e for e in bee}
    for nm in a.rows:
        de = next((e for e in d.find("ElementEffects") if e.get("name") == nm), None)
        if de is None:
            # the donor placed nothing on this row: the graft EMPTIES it
            if nm in bel:
                for L in bel[nm].findall("EffectLayer"):
                    for ef in list(L.findall("Effect")):
                        L.remove(ef)
            print(f"  row {nm}: emptied (nothing on it in the donor)")
            continue
        de = ET.fromstring(ET.tostring(de))
        for ef in list(de.iter("Effect")):
            ef.attrib.update(remap(ef).attrib)
        if nm in bel:
            old = bel[nm]; bee.insert(list(bee).index(old), de); bee.remove(old)
        else:
            bee.append(de)
        bel[nm] = de
        print(f"  row {nm}: {sum(1 for _ in de.iter('Effect'))} effects")

    for wi, w in enumerate(a.window):
        w0, w1 = (int(round(float(x) * 1000)) for x in w.split(","))
        only = None
        if wi < len(a.window_models) and a.window_models[wi]:
            only = set(a.window_models[wi].split("|"))
        n_out = n_in = 0
        for de in d.find("ElementEffects"):
            nm = de.get("name")
            if only is not None and nm not in only and nm.split("/")[0] not in only:
                continue
            be = bel.get(nm)
            if be is None:
                be = ET.SubElement(bee, "Element", dict(de.attrib)); bel[nm] = be
            blays = be.findall("EffectLayer")
            # base: drop what starts inside, trim what straddles the edges
            for L in blays:
                for ef in list(L.findall("Effect")):
                    s, e = int(ef.get("startTime")), int(ef.get("endTime"))
                    if w0 <= s < w1:
                        L.remove(ef); n_out += 1
                    elif s < w0 < e:
                        ef.set("endTime", str(w0))
            for li, L in enumerate(de.findall("EffectLayer")):
                while len(blays) <= li:
                    blays.append(ET.SubElement(be, "EffectLayer"))
                for ef in L.findall("Effect"):
                    s, e = int(ef.get("startTime")), int(ef.get("endTime"))
                    if not (w0 <= s < w1):
                        continue
                    nef = remap(ef)
                    if e > w1:
                        nef.set("endTime", str(w1))
                    # keep the layer's effects in time order
                    kids = blays[li].findall("Effect")
                    pos = next((i for i, k in enumerate(kids) if int(k.get("startTime")) > s), len(kids))
                    blays[li].insert(pos, nef); n_in += 1
            # base effects that start at/after w1 but overlap a trimmed donor end: none, by trimming
        print(f"  window {w0/1000:.2f}-{w1/1000:.2f}s{' on ' + str(len(only)) + ' models' if only else ''}: "
              f"{n_out} base effects replaced by {n_in} donor effects")
    base.write(a.out, encoding="UTF-8", xml_declaration=True)


if __name__ == "__main__":
    main()
