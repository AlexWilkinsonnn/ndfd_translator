"""
Reads in ND reco from CAF, runs Radi's train gpt model to predict FD reco, writes out result to
friend tree in the ND CAF file.
NOTE: This is currently hardcoded for the model architecture used for the FHC numu-numu training.
"""
import argparse, os, time

import ROOT
import numpy as np

import torch

from model import GPT

ND_RECO_VARS = [
    'eRecoP', 'eRecoN', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
    'Ev_reco',
    'Elep_reco',
    'theta_reco',
    'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg'
]
FD_RECO_CVN_VARS = [ 'fd_numu_score', 'fd_nue_score', 'fd_nc_score', 'fd_nutau_score' ]
FD_RECO_E_VARS = [ 'fd_nue_lep_E', 'fd_numu_lep_E', 'fd_numu_nu_E', 'fd_nue_nu_E' ]

def main(args):
    conf = GPT.get_default_config()
    conf.model_type = 'gpt-mini'
    conf.block_size = len(ND_RECO_VARS) + len(FD_RECO_CVN_VARS) + len(FD_RECO_E_VARS) + 1
    conf.scores_size = len(FD_RECO_CVN_VARS)
    conf.far_reco_size = len(FD_RECO_E_VARS)
    model = GPT(conf)
    model.load_state_dict(torch.load(args.model_weights, map_location=torch.device('cpu')))
    model.eval()

    in_f = ROOT.TFile.Open(args.infile, "READ")
    in_t = in_f.Get("cafTree")

    nd_recos = []
    for i_ev, ev in enumerate(in_t):
        nd_recos.append(torch.tensor([[
            ev.eRecoP, ev.eRecoN, ev.eRecoPip, ev.eRecoPi0, ev.eRecoOther,
            ev.Ev_reco,
            ev.Elep_reco,
            ev.theta_reco,
            ev.reco_numu, ev.reco_nc, ev.reco_nue, ev.reco_lepton_pdg
        ]]))
        if (i_ev + 1) % args.batch_size == 0:
            in_batch = torch.cat(nd_recos)
            nd_recos = []
            s = time.time()
            pred_batch = model.generate(in_batch).numpy()
            e = time.time()
            print(e-s)
            # print(in_batch)
            # print(in_batch.shape)
            # print("-->")
            # print(pred_batch)
            # print(pred_batch.shape)
            break


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("infile", type=str, help="input ND CAF file")
    parser.add_argument(
        "outfile", type=str, help="output ND CAF file that will include pred fd reco friend tree"
    )
    parser.add_argument("model_weights", type=str)

    parser.add_argument("--batch_size", type=int, default=4, help="batch size for inference")

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_arguments())
