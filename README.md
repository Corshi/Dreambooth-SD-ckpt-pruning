# ckpt_prune - quickfix and options

This script was adapted from:
https://github.com/harubaru/waifu-diffusion/blob/e4736c11f580197a8c55c69ac3be14e7b0de4588/scripts/prune.py

Usage: prune-ckpt.py [-h] [--ckpt CKPT] [--half] [--emaonly]
`--ckpt` will crate a copy of the ckpt with only erased optimizers
Optional:
`--half` will crate flout16 version
`--emaonly` will erase MA data. Leavin only EMA.

This script will crated `archive` file in the directory of this script. You will ned to rename it manualy to .ckpt file.
This is to avoid the error of PyTorch that incorectly names the directories inside the .ckpt file.
Also with aditional commands you can now set what kind of data you want to prune next to optimizers.
