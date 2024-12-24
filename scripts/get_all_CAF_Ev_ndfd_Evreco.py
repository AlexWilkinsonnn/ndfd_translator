"""
Get combined energy spectrum of all off-axis positions.
Requires duneutil, setup script in directory.
"""
import os, subprocess
from array import array

import numpy as np
import ROOT

IN_DIR="/pnfs/dune/persistent/users/awilkins/ND_CAF_fdrecopreds/chasnip-NDCAF_OnAxisHadd-FHC_fdrecopreds_muresim_nolognorm_nocausalnearmask_seed38/*"
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

Ev_all_hist, bins = np.histogram([], bins=BINS)
Ev_reco_all_hist, _ = np.histogram([], bins=BINS)
pred_fd_numu_nu_E_all_hist, _ = np.histogram([], bins=BINS)
for i_fname, fname in enumerate(fnames):
    f = ROOT.TFile.Open(fname, "READ")
    t_caf = f.Get("cafTree")
    t_fdrecofriend = f.Get("FDRecoNumuPredFriend")

    print(i_fname)
    if i_fname > 2:
        break

    for e in t_caf:
        bin_num = np.digitize(float(e.Ev), bins=bins) - 1
        Ev_all_hist[bin_num] += 1
        bin_num = np.digitize(float(e.Ev_reco), bins=bins) - 1
        Ev_reco_all_hist[bin_num] += 1
    for e in t_fdrecofriend:
        bin_num = np.digitize(float(e.pred_fd_numu_nu_E), bins=bins) - 1
        pred_fd_numu_nu_E_all_hist[bin_num] += 1

np.save("allCAF_Ev_oaall_hist.npy", Ev_all_hist)
np.save("allCAF_Ev_oaall_bins.npy", bins)
np.save("allCAF_Ev_reco_oaall_hist.npy", Ev_reco_all_hist)
np.save("allCAF_Ev_reco_oaall_bins.npy", bins)
np.save("allCAF_pred_fd_numu_nu_E_oaall_hist.npy", pred_fd_numu_nu_E_all_hist)
np.save("allCAF_pred_fd_numu_nu_E_oaall_bins.npy", bins)
