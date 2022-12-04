# ckpt_prune - quickfix and options


This script works with merged models and dreambooth models. For some reason it does not work with models that ware naively trained.
Edit to the code is necessary to accommodate that and I don’t know enough to make that change.


This script was adapted from:

https://github.com/harubaru/waifu-diffusion/blob/e4736c11f580197a8c55c69ac3be14e7b0de4588/scripts/prune.py


***Usage***: prune-ckpt.py [-h] [--ckpt CKPT] [--half] [--emaonly]

`--ckpt` will crate a copy of the ckpt with only erased optimizers

 
*Optional*:

`--half` will crate float16 version

`--emaonly` will erase MA data. Leavin only EMA.


**This script will crated `archive` file in the directory of this script.**

**You will need to rename it manually to .ckpt file.**

This is to avoid the error of PyTorch that incorrectly names the directories inside the .ckpt file.

Also with additional commands you can now set what kind of data you want to prune next to optimizers.
