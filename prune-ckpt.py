#usage: prune-ckpt.py [-h] [--ckpt CKPT] [--half] [--emaonly]
#This script will crated `archive` file in the directory of this script. You will ned to rename it manualy to .ckpt file.
#This is to avoid the error of PyTorch that incorectly names the directories inside the .ckpt file.
#Also with aditional commands you can now set what kind of data you want to prune next to optimizers.
#--half will crate flout16 version
#--emaonly will erase MA data. Leavin only EMA.
import os
import torch
import argparse
import glob


parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('--ckpt', type=str, default=None, help='Path to model ckpt')
parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
parser.add_argument('--emaonly', action="store_true", help='Keep only ema data')
args = parser.parse_args()
ckpt = args.ckpt

def prune_it(p):
    print(f"prunin' in path: {p}")
    size_initial = os.path.getsize(p)
    nsd = dict()
    sd = torch.load(p, map_location="cpu")
    print(sd.keys())
    for k in sd.keys():
        if k != "optimizer_states":
            nsd[k] = sd[k]
    else:
        print(f"removing optimizer states for path {p}")
    if "global_step" in sd:
        print(f"This is global step {sd['global_step']}.")
    if args.emaonly:
        sd = nsd["state_dict"].copy()
        # infer ema keys
        ema_keys = {k: "model_ema." + k[6:].replace(".", "") for k in sd.keys() if k.startswith("model.")}
        new_sd = dict()

        for k in sd:
            if k in ema_keys:
                new_sd[k] = sd[ema_keys[k]] if not args.half else sd[ema_keys[k]].half()
            elif not k.startswith("model_ema") or k in ["model_ema.num_updates", "model_ema.decay"]:
                new_sd[k] = sd[k] if not args.half else sd[k].half()

        assert len(new_sd) == len(sd) - len(ema_keys)
        nsd["state_dict"] = new_sd
    else:
        sd = nsd['state_dict'].copy()
        new_sd = dict()
        for k in sd:
            new_sd[k] = sd[k] if not args.half else sd[k].half()
        nsd['state_dict'] = new_sd

    fn = f"archive"
    print(f"saving pruned checkpoint at: {fn}")
    torch.save(nsd, fn)
    newsize = os.path.getsize(fn)
    MSG = f"New ckpt size: {newsize*1e-9:.2f} GB. " + \
          f"Saved {(size_initial - newsize)*1e-9:.2f} GB by removing optimizer states"
    if args.emaonly:
        MSG += " and non-EMA weights"
    print(MSG)


if __name__ == "__main__":
    prune_it(ckpt)
