"""WESTPA per-segment progress coordinate. Reads seg_config.json, writes cv.dat."""
import json
import numpy as np
import mdtraj
import cv_families

cfg = json.load(open("seg_config.json"))

reference = mdtraj.load(cfg["cv_reference_pdb"])
parent = mdtraj.load("parent.xml", top="bstate.pdb")
seg = mdtraj.load("seg.dcd", top="bstate.pdb")

cv_data = cv_families.compute(parent, seg, reference, cfg)
np.savetxt("cv.dat", cv_data)
