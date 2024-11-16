"""
Get combined energy spectrum of all off-axis positions.
Requires duneutil, setup script in directory.
"""
import os, subprocess
from array import array

import numpy as np
import ROOT

IN_DIR="/pnfs/dune/persistent/users/chasnip/NDCAF_OnAxisHadd/FHC/*"
BINS = [
    0.,   0.5,  0.54, 0.58, 0.62, 0.66, 0.7,  0.74, 0.78, 0.82,
    0.86, 0.9,  0.94, 0.98, 1.02, 1.06, 1.1,  1.14, 1.18, 1.22,
    1.26, 1.3,  1.34, 1.38, 1.42, 1.46, 1.5,  1.54, 1.58, 1.62,
    1.66, 1.7,  1.74, 1.78, 1.82, 1.86, 1.9,  1.94, 1.98, 2.02,
    2.1,  2.18, 2.26, 2.34, 2.42, 2.5,  2.58, 2.66, 2.74, 2.82,
    2.9,  2.98, 3.06, 3.16, 3.26, 3.36, 3.46, 3.56, 3.66, 3.76,
    3.86, 3.96, 4.06, 4.5,  5.,   6.,   10.,  120.
]

res = subprocess.run(["pnfs2xrootd", IN_DIR], stdout=subprocess.PIPE)
fnames = res.stdout.decode("utf-8").rstrip("\n").split(" ")

Ev_all_hist, Ev_bins = np.histogram([], bins=BINS)
Ev_oa_hists = {}
for i_fname, fname in enumerate(fnames):
    f = ROOT.TFile.Open(fname, "READ")
    t_caf = f.Get("cafTree")
    t_summary = f.Get("FileSummaryTree")
    h1_280kA = f.Get("FileExposure_280")

    if float(h1_280kA.Integral()) != 0:
        oa = "280kA"
    else:
        for e in t_summary:
            oa = str(int(e.det_x))
            break
    if oa not in Ev_oa_hists:
        hist, _ = np.histogram([], bins=BINS)
        Ev_oa_hists[oa] = hist
    Ev_oa_hist = Ev_oa_hists[oa]

    print(i_fname, oa)

    for e in t_caf:
        bin_num = np.digitize(float(e.Ev), bins=Ev_bins) - 1
        Ev_all_hist[bin_num] += 1
        Ev_oa_hist[bin_num] += 1

np.save("allCAF_Ev_oaall_hist.npy", Ev_all_hist)
np.save("allCAF_Ev_oaall_bins.npy", Ev_bins)
for oa, hist in Ev_oa_hists.items():
    np.save(f"allCAF_Ev_oa{oa}_hist.npy", hist)
    np.save(f"allCAF_Ev_oa{oa}_bins.npy", Ev_bins)
