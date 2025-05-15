import argparse
from evaluation import nerf_eval, dtu_eval
from pathlib import Path
from pyhocon import ConfigFactory


parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True) # 'nerf' or 'dtu'
parser.add_argument("--est_mesh", required=True, default="//hdd2/planb/nerf_synthetic/ship/ours_nogs/clean/epoch_05000_iteration_000500000_checkpoint.ply")
parser.add_argument("--gt_pcd", required=True, default="//hdd2/planb/nerf_synthetic/gt_mesh/ship_pcd.ply")
parser.add_argument("--scene", required=False, default=83)

args = parser.parse_args()

if args.data == 'nerf':
    nerf_eval.eval(args.est_mesh, args.gt_pcd)
elif args.data == 'dtu':
    dtu_eval.eval(args.est_mesh, args.gt_pcd, args.scene)