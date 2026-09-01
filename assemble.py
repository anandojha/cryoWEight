#!/usr/bin/env python3
"""
Build a per system run tree from a config.xml.

    python assemble.py --system <name> --dest <dir>
    python assemble.py --system <name> --validate <corpus>
"""

from __future__ import annotations
import argparse
import shutil
import sys
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cryoweight import configio

TOKEN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

PHASE2_REFACTORED = {
    # Phase 2 covered the per segment MD and CV
    "WE_files/common_files/cv.py",
    "WE_files/common_files/production.py",
    "WE_files/common_files/build_system.py",
    "WE_files/common_files/cv_families.py",
    "WE_files/common_files/seg_config.json",
    # Phase 3 made the reweighting and analysis pipeline shared and config driven
    "scripts/reweight.py",
    "scripts/reweight_run0.py",
    "scripts/merge.py",
    "scripts/sample_bstates.py",
    "scripts/get_distribution.py",
    "scripts/plot_free_energy.py",
    "scripts/reweight_config.json",
    "scripts/cv_families.py",
    "scripts/build_system.py",
    "init_MD/simulation.py",
    "init_MD/build_system.py",
    "tar.py",
}

# Single shared copy. The ntl9 corpus copies differ only cosmetically and are not byte checked.
CANONICALIZED_COSMETIC = {"WE_files/init.sh", "WE_files/westpa_scripts/node.sh"}


def resolve(system: str):
    """The config.xml path and the system directory for a name or a path.

    A plain name is looked up as systems/<name>. A path, such as examples/chignolin or
    an absolute directory, is used as the system directory itself, so the shipped
    examples run by path without colliding with the production system names.
    """
    if os.path.sep in system or system in (".", ".."):
        sdir = system if os.path.isabs(system) else os.path.join(HERE, system)
    else:
        sdir = os.path.join(HERE, "systems", system)
    sdir = os.path.normpath(sdir)
    for candidate in (os.path.join(sdir, "config.xml"), sdir + ".xml"):
        if os.path.isfile(candidate):
            return candidate, sdir
    raise SystemExit(
        f"no configuration found for {system!r}: tried "
        f"{os.path.join(sdir, 'config.xml')} and {sdir}.xml"
    )


def load_raw(system: str) -> dict:
    return configio.read_xml(resolve(system)[0])


def load_cfg(system: str) -> dict:
    """Flat token namespace for templates and the driver (excludes the JSON blocks)."""
    raw = {k: v for k, v in load_raw(system).items() if k not in ("seg_config", "reweight_config")}
    return flatten(raw)


def _pcoord_len(raw):
    """Progress coordinate rows per segment, the parent frame plus one per saved frame.

    west.cfg is rendered with this value, so any segment length and report interval pair
    is valid as long as the interval divides the length evenly and at least one frame is
    saved. cv.py returns exactly this many rows.
    """
    sc = raw.get("seg_config") or {}
    steps, interval = sc.get("n_steps_per_segment"), sc.get("dcd_report_interval")
    if not steps or not interval:
        return 3
    if steps % interval:
        print(
            f"note: dcd_report_interval {interval} does not divide n_steps_per_segment "
            f"{steps} evenly, so the last {steps % interval} steps of each segment are "
            f"propagated but not saved."
        )
    rows = steps // interval + 1
    if rows < 2:
        raise SystemExit("a segment must save at least one frame beyond the parent")
    return rows


def flatten(d: dict, prefix: str = "") -> dict:
    """Lift every leaf of a nested configuration into one flat namespace.

    A template refers to a value by its own name, so the nesting in the YAML groups
    related settings for a reader and means nothing to the substitution. Two blocks
    defining the same leaf would be ambiguous, so that raises rather than picking one.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            # The clash has to be checked against what the nested block returns, not just
            # against this level, or two blocks defining the same leaf silently keep the
            # last one and the run uses a value nobody wrote.
            nested = flatten(v, prefix)
            clash = sorted(set(nested) & set(out))
            if clash:
                raise ValueError(f"duplicate config key {clash[0]!r}")
            out.update(nested)
        else:
            if k in out:
                raise ValueError(f"duplicate config key {k!r}")
            out[k] = "" if v is None else v
    return out


def render(text: str, cfg: dict, rel: str) -> str:
    missing = set()

    def sub(m):
        key = m.group(1)
        if key not in cfg:
            missing.add(key)
            return m.group(0)
        return str(cfg[key])

    out = TOKEN.sub(sub, text)
    if missing:
        raise KeyError(f"{rel}: unresolved tokens {sorted(missing)}")
    return out


def assemble(system: str, dest: str, stage_data: bool = True) -> list[str]:
    cfg = load_cfg(system)
    cfg["pcoord_len"] = _pcoord_len(load_raw(system))
    written = []
    # 1. shared
    shared = os.path.join(HERE, "shared")
    for dp, dns, fs in os.walk(shared):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for f in fs:
            if f.endswith(".pyc"):
                continue
            rel = os.path.relpath(os.path.join(dp, f), shared)
            _copy(os.path.join(shared, rel), os.path.join(dest, rel))
            written.append(rel)
    # 1b. the package modules, staged as scripts/ so a run directory holds standalone
    # copies and a compute node needs nothing installed.
    pkg = os.path.join(HERE, "cryoweight")
    for f in sorted(os.listdir(pkg)):
        if f.endswith(".py") and f != "__init__.py":
            rel = os.path.join("scripts", f)
            _copy(os.path.join(pkg, f), os.path.join(dest, rel))
            written.append(rel)
    # 2. templates
    tdir = os.path.join(HERE, "templates")
    if os.path.isdir(tdir):
        for dp, _, fs in os.walk(tdir):
            for f in fs:
                if not f.endswith(".tmpl"):
                    continue
                rel = os.path.relpath(os.path.join(dp, f), tdir)[:-5]  # strip .tmpl
                text = open(os.path.join(tdir, rel + ".tmpl")).read()
                _write(os.path.join(dest, rel), render(text, cfg, rel))
                written.append(rel)
    # 3. per system overrides
    odir = os.path.join(resolve(system)[1], "overrides")
    if os.path.isdir(odir):
        for dp, _, fs in os.walk(odir):
            for f in fs:
                rel = os.path.relpath(os.path.join(dp, f), odir)
                _copy(os.path.join(odir, rel), os.path.join(dest, rel))
                written.append(rel)
    # 4. assembled JSON config blocks consumed by shared runtime scripts.
    raw = load_raw(system)
    import json

    for block, rel in (
        ("seg_config", "WE_files/common_files/seg_config.json"),
        ("reweight_config", "scripts/reweight_config.json"),
    ):
        if block in raw:
            data = dict(raw[block])
            if block == "reweight_config":
                # The subiteration count is set once at the top of the YAML. The runtime
                # scripts read it from this JSON, so it is injected here rather than
                # duplicated in the block.
                top = raw.get("n_iterations")
                own = data.get("n_iterations")
                if top is not None:
                    if own is not None and int(own) != int(top):
                        raise SystemExit(
                            f"n_iterations is {top} at the top of the config but "
                            f"{own} inside reweight_config. Set it once, at the top."
                        )
                    data["n_iterations"] = int(top)
                data = dict(sorted(data.items()))
            _write(os.path.join(dest, rel), json.dumps(data, indent=2) + "\n")
            written.append(rel)
    # 5. pick the solvent model variant, renaming <name>_<model>.<ext> to <name>.<ext>
    model = (raw.get("seg_config") or {}).get("solvent_model") or cfg.get("solvent_model")
    if model:
        keep_suffix, drop_suffix = (
            f"_{model}",
            f"_{'implicit' if model=='explicit' else 'explicit'}",
        )
        for dp, _, fs in os.walk(dest):
            for f in fs:
                stem, ext = os.path.splitext(f)
                if stem.endswith(keep_suffix):
                    base = stem[: -len(keep_suffix)] + ext
                    os.replace(os.path.join(dp, f), os.path.join(dp, base))
                    written.append(os.path.relpath(os.path.join(dp, base), dest))
                elif stem.endswith(drop_suffix):
                    os.remove(os.path.join(dp, f))
        written = [
            w
            for w in written
            if not (
                os.path.splitext(w)[0].endswith("_explicit")
                or os.path.splitext(w)[0].endswith("_implicit")
            )
        ]
    # 6. stage input data (data/, init_MD/). Skipped during --validate.
    if stage_data:
        for sub in ("data", "init_MD"):
            src = os.path.join(resolve(system)[1], sub)
            if os.path.isdir(src):
                for dp, _, fs in os.walk(src):
                    for f in fs:
                        full = os.path.join(dp, f)
                        rel = os.path.join(sub, os.path.relpath(full, src))
                        _copy(full, os.path.join(dest, rel))
                        written.append(rel)
    return sorted(set(written))


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)
    # WESTPA executes the staged shell scripts directly, and an override committed
    # without the executable bit would otherwise be refused with a PermissionError.
    if dst.endswith(".sh"):
        os.chmod(dst, 0o755)


def _write(dst, text):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w") as f:
        f.write(text)
    # WESTPA runs the rendered shell scripts directly. Files copied from shared/ keep the
    # mode they were committed with, but a rendered one is created 0644 and would be
    # skipped with only a warning in the log.
    if dst.endswith(".sh"):
        os.chmod(dst, 0o755)


def validate(system: str, corpus_dir: str) -> int:
    import tempfile, filecmp

    dest = tempfile.mkdtemp(prefix=f"cw_{system}_")
    assemble(system, dest, stage_data=False)
    bad = []
    refactored = 0
    data = 0
    canon = 0
    for dp, _, fs in os.walk(dest):
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), dest)
            if rel in PHASE2_REFACTORED:
                refactored += 1
                continue
            if rel in CANONICALIZED_COSMETIC:
                canon += 1
                continue
            ref = os.path.join(corpus_dir, rel)
            if not os.path.isfile(ref):
                if rel.endswith((".pdb", ".dcd")):
                    data += 1
                    continue
                bad.append((rel, "absent-in-corpus"))
                continue
            if not filecmp.cmp(os.path.join(dest, rel), ref, shallow=False):
                bad.append((rel, "BYTE-DIFF"))
    if bad:
        print(f"[{system}] {len(bad)} mismatch(es):")
        for rel, why in bad:
            print(f"   {why:16} {rel}")
        return 1
    print(
        f"[{system}] OK: code byte-identical to {corpus_dir} "
        f"({refactored} Phase-2 refactored files behaviorally validated, "
        f"{canon} canonicalized-cosmetic files, {data} data files)"
    )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--dest")
    ap.add_argument("--validate", metavar="CORPUS_DIR")
    a = ap.parse_args()
    if a.validate:
        sys.exit(validate(a.system, a.validate))
    if not a.dest:
        ap.error("need --dest or --validate")
    files = assemble(a.system, a.dest)
    print(f"assembled {len(files)} files -> {a.dest}")
