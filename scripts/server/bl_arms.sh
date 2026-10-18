#!/usr/bin/env bash
# Shared definition of the OFFICIAL-baseline ladder, used by both the batch probe
# (scripts/server/probe_batches.sh) and the serial campaign
# (scripts/server/run_official_bl.sh) so the two can never disagree about arms,
# flags or the sample budget.
#
# Every arm is trained from scratch on RealSR Train (390 pairs) with one shared recipe
# (AdamW 1e-4, L1, EMA, cosine, warmup 500) and one shared sample budget, so the ladder
# compares architectures rather than optimisers or budgets. `unet_ctrl` is OUR OWN
# architecture under that same recipe, which is what makes "ours vs the baselines"
# apples-to-apples (our champion uses Muon 5e-3 + wavelet loss + pretraining, a
# different recipe on purpose - it is reported as a separate row).
#
# Order = cheap first, so the table fills early. RLFN is deliberately absent: its
# official repo is not reachable from this box, and a re-implementation whose param
# count does not match the paper's 2.11M would repeat the mistake this ladder exists
# to fix. SRCNN likewise (24.7k official; our variant row is labelled as a variant).
#
# vdsr_off IS included: the HR-input failure was not VDSR's model but a variable-shadowing
# bug of our own in sr_pixel_tiled (the accumulator is named `out`; assigning the model
# result to `out` emptied every later slice). With that fixed the gate passes for it too.
# ARMS_OVERRIDE + PROBE_SUFFIX let the probe re-run for a single arm without clobbering
# the measured table for the rest.
ARMS="${ARMS_OVERRIDE:-rdn_off edsr_off rcan_off vdsr_off srformer_off swinir_off unet_ctrl hat_off}"
BUDGET=1280000                      # samples = batch x steps (our champion's budget)
MEM_CAP_GB=40                       # above ~40G this card slows super-linearly (docs/INFRA.md)

flags_for(){
  case "$1" in
    unet_ctrl) echo "--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --attn-levels 2,3 --native-lr 0 --residual 1" ;;
    # VDSR (as in the reference implementation) refines a bicubic-upsampled image and
    # predicts the image itself: HR in, --residual 0. All the others upsample internally.
    vdsr_off)  echo "--backbone vdsr_off --net-input hr --residual 0" ;;
    *)         echo "--backbone $1 --net-input lr" ;;
  esac
}

batch_candidates(){   # largest -> smallest; the probe keeps the best THROUGHPUT, not the biggest
  case "$1" in
    unet_ctrl) echo "256 128 64" ;;
    *)         echo "64 32 16 8 4" ;;
  esac
}

probe_secs(){         # short-run steps: enough for >=2 marks (marks are every 50 steps)
  echo 150
}
