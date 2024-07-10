"""
Reads in ND reco from CAF, runs Radi's train gpt model to predict FD reco, writes out result to
friend tree in the ND CAF file.
NOTE: This is currently hardcoded for the model architecture used for the FHC numu-numu training.
NOTE: Would be faster with larger batch size but it is too cumbersome to catch failed predictions
when working with batches
"""
import argparse, time, random
from array import array

import ROOT
import numpy as np

import torch

from model import GPT

# This the order we assume the input/output tensors to the model correspond to.
# These variables are mainly for reference.
ND_RECO_VARS = [
    'eRecoP', 'eRecoN', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
    'Ev_reco',
    'Elep_reco',
    'theta_reco',
    'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
    'fd_x_vert', 'fd_y_vert', 'fd_z_vert'
]
FD_RECO_CVN_VARS = [ 'fd_numu_score', 'fd_nue_score', 'fd_nc_score', 'fd_nutau_score' ]
FD_RECO_E_VARS = [ 'fd_nue_lep_E', 'fd_numu_lep_E', 'fd_numu_nu_E', 'fd_nue_nu_E' ]

FD_VTX_MINMAX = [ (-310.0, 310.0), (-550.0, 550.0), (50.0, 1244.0) ]

def main(args):
    # Prep model
    model = get_model(args.model_weights)

    # Prep vertices
    fd_vertices = np.load(args.vertices_file) if args.vertices_file is not None else None

    # Prep TTrees
    in_f = ROOT.TFile.Open(args.infile, "UPDATE")
    t_caf = in_f.Get("cafTree")
    t_pred = ROOT.TTree("FDRecoNumuPredFriend", "FDRecoNumuPredFriend")

    global b_numu_score
    b_numu_score = array("f", [0])
    t_pred.Branch("pred_fd_numu_score", b_numu_score, "pred_fd_numu_score/F")
    global b_nue_score
    b_nue_score = array("f", [0])
    t_pred.Branch("pred_fd_nue_score", b_nue_score, "pred_fd_nue_score/F")
    global b_nc_score
    b_nc_score = array("f", [0])
    t_pred.Branch("pred_fd_nc_score", b_nc_score, "pred_fd_nc_score/F")
    global b_nutau_score
    b_nutau_score = array("f", [0])
    t_pred.Branch("pred_fd_nutau_score", b_nutau_score, "pred_fd_nutau_score/F")
    global b_nue_lep_E
    b_nue_lep_E = array("f", [0])
    t_pred.Branch("pred_fd_nue_lep_E", b_nue_lep_E, "pred_fd_nue_lep_E/F")
    global b_numu_lep_E
    b_numu_lep_E = array("f", [0])
    t_pred.Branch("pred_fd_numu_lep_E", b_numu_lep_E, "pred_fd_numu_lep_E/F")
    global b_nue_nu_E
    b_nue_nu_E = array("f", [0])
    t_pred.Branch("pred_fd_nue_nu_E", b_nue_nu_E, "pred_fd_nue_nu_E/F")
    global b_numu_nu_E
    b_numu_nu_E = array("f", [0])
    t_pred.Branch("pred_fd_numu_nu_E", b_numu_nu_E, "pred_fd_numu_nu_E/F")
    global b_vtx_x
    b_vtx_x = array("f", [0])
    t_pred.Branch("pred_fd_vtx_x", b_vtx_x, "pred_fd_vtx_x/F")
    global b_vtx_y
    b_vtx_y = array("f", [0])
    t_pred.Branch("pred_fd_vtx_y", b_vtx_y, "pred_fd_vtx_y/F")
    global b_vtx_z
    b_vtx_z = array("f", [0])
    t_pred.Branch("pred_fd_vtx_z", b_vtx_z, "pred_fd_vtx_z/F")

    # Loop CAF tree to make FD preds
    nd_recos = []
    t_0 = time.time()
    for i_ev, ev in enumerate(t_caf):
        fd_vtx = gen_fd_vtx(fd_vertices)
        nd_recos.append(torch.tensor([[
            ev.eRecoP, ev.eRecoN, ev.eRecoPip, ev.eRecoPim, ev.eRecoPi0, ev.eRecoOther,
            ev.Ev_reco,
            ev.Elep_reco,
            ev.theta_reco,
            ev.reco_numu, ev.reco_nc, ev.reco_nue, ev.reco_lepton_pdg,
            fd_vtx[0], fd_vtx[1], fd_vtx[2]
        ]]))
        pred_fd_cvn, pred_fd_E = make_fd_preds(model, nd_recos)
        write_fd_preds_to_branches(pred_fd_cvn, pred_fd_E, fd_vtx)
        t_pred.Fill()
        nd_recos = []
        if (i_ev + 1) % 1000 == 0:
            print(
                "{} / {} ({:.2f}s)".format(
                    t_pred.GetEntries(), t_caf.GetEntries(), time.time() - t_0
                )
            )
            t_0 = time.time()

    print("Done!")

    t_caf.AddFriend("FDRecoNumuPredFriend")
    in_f.Write()
    in_f.Close()

""" helpers """

def get_model(model_weights):
    conf = GPT.get_default_config()
    conf.model_type = 'gpt-mini'
    conf.block_size = len(ND_RECO_VARS) + len(FD_RECO_CVN_VARS) + len(FD_RECO_E_VARS) + 1
    conf.scores_size = len(FD_RECO_CVN_VARS)
    conf.far_reco_size = len(FD_RECO_E_VARS)
    model = GPT(conf)
    model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
    model.eval()
    return model

# XXX Not using this anymore, can just apply selection cuts afterwards
def passes_sel_cuts(mu_contained, mu_tracker, mu_ecal, reco_numu, Ehad_veto):
    """
    Model was trained only on data that passes these cuts. Predictions are only good for events
    with the same cuts. The model is also unstable and sometimes crashed if the events dont have
    these cuts applied.
    """
    return (mu_contained or mu_tracker or mu_ecal) and reco_numu and Ehad_veto < 30

def gen_fd_vtx(fd_vertices):
    if fd_vertices is None:
        return tuple(
            FD_VTX_MINMAX[i][0] + random.random() * (FD_VTX_MINMAX[i][1] - FD_VTX_MINMAX[i][0])
            for i in range(3)
        )
    else:
        return tuple(fd_vertices[np.random.randint(0, len(fd_vertices))])

# NOTE assumes batch size is 1
def make_fd_preds(model, nd_recos):
    in_batch = torch.cat(nd_recos)
    with torch.no_grad():
        try:
            pred_batch = model.generate(in_batch).numpy()
        except ValueError:
            # Model is unstable and can fail for some inputs, just move on when this hapens
            return None, None
        pred_fd = pred_batch[0, len(ND_RECO_VARS):]
        pred_fd_cvn = pred_fd[:len(FD_RECO_CVN_VARS)]
        pred_fd_E = pred_fd[len(FD_RECO_CVN_VARS):]
    return pred_fd_cvn, pred_fd_E

def write_fd_preds_to_branches(pred_fd_cvn=None, pred_fd_E=None, fd_vtx=None):
    """
    Uses all b_* variables as globals.
    """
    if pred_fd_cvn is not None:
        b_numu_score[0] = pred_fd_cvn[0]
        b_nue_score[0] = pred_fd_cvn[1]
        b_nc_score[0] = pred_fd_cvn[2]
        b_nutau_score[0] = pred_fd_cvn[3]
    else:
        b_numu_score[0] = -999.0
        b_nue_score[0] = -999.0
        b_nc_score[0] = -999.0
        b_nutau_score[0] = -999.0
    if pred_fd_E is not None:
        b_nue_lep_E[0] = pred_fd_E[0]
        b_numu_lep_E[0] = pred_fd_E[1]
        b_nue_nu_E[0] = pred_fd_E[2]
        b_numu_nu_E[0] = pred_fd_E[3]
    else:
        b_nue_lep_E[0] = -999.0
        b_numu_lep_E[0] = -999.0
        b_nue_nu_E[0] = -999.0
        b_numu_nu_E[0] = -999.0
    if fd_vtx is not None:
        b_vtx_x[0] = fd_vtx[0]
        b_vtx_y[0] = fd_vtx[1]
        b_vtx_z[0] = fd_vtx[2]
    else:
        b_vtx_x[0] = -9999.0
        b_vtx_y[0] = -9999.0
        b_vtx_z[0] = -9999.0

""" end helpers """

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "infile", type=str, help="input ND CAF file for friend FD pred tree to be added to"
    )
    parser.add_argument("model_weights", type=str)
    # Potentially needed because the geometric efficiency FD throws did not generate a
    # smooth uniform distribution of vertices
    parser.add_argument(
        "--vertices_file", type=str, default=None,
        help="numpy array of shape (N, 3) that contains vertices to randomly draw from"
    )

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_arguments())
