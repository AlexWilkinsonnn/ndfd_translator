"""
Reads in npz with truth and predictions for the paired data test dataset, writes these to a new
TTree. This will be used to make a resolution matrix so that an appropriate MC correction can be
applied in the analysis.
"""
import argparse
from array import array

import ROOT
import numpy as np

def main(args):
    # Prep test predictions
    data = np.load(args.test_preds)
    test_preds = data["pred_fd_numu_nuE"]
    test_trues = data["true_fd_numu_nuE"]

    # Prep TTrees
    # in_f = ROOT.TFile.Open(args.outfile, "RECREATE")
    in_f = ROOT.TFile.Open(args.infile, "UPDATE")
    t_test = ROOT.TTree("FDModelTestPreds", "FDModelTestPreds")

    global b_pred_fd_numu_nu_E
    b_pred_fd_numu_nu_E = array("f", [0])
    t_test.Branch("pred_fd_numu_nu_E", b_pred_fd_numu_nu_E, "pred_fd_numu_nu_E/F")
    global b_true_fd_numu_nu_E
    b_true_fd_numu_nu_E = array("f", [0])
    t_test.Branch("true_fd_numu_nu_E", b_true_fd_numu_nu_E, "true_fd_numu_nu_E/F")


    # Loop test set preds and add to new TTree
    for pred, true in zip(test_preds, test_trues):
        b_pred_fd_numu_nu_E[0] = pred
        b_true_fd_numu_nu_E[0] = true
        t_test.Fill()

    print("Done!")

    in_f.Write()
    in_f.Close()

def parse_arguments():
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "outfile", type=str, help="output ROOT file for test data predictions"
    # )
    parser.add_argument(
        "infile", type=str, help="input ND CAF file for test data predictions"
    )
    parser.add_argument(
        "test_preds", type=str, help="input npz file with test set predictions and truth"
    )

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_arguments())
