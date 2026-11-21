#!/usr/bin/env bash
# Lab-machine performance check
#
#   bash tools/vm_check.sh            # everything: build, tests, all benchmarks, deep-learning models
#   bash tools/vm_check.sh --no-dl    # skip the PyTorch install and the deep-learning checks
#   bash tools/vm_check.sh --no-tum   # skip the TUM download (~0.5 GB) and the real-data runs
#
# Run it in a fresh terminal, outside conda. Everything goes to ~/vm_check/ and
# ~/vm_check.tar.gz; report.md is the one-page summary to send back.

set -o pipefail
DL=1; TUM=1
for a in "$@"; do
  case "$a" in
    --no-dl) DL=0 ;;
    --no-tum) TUM=0 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HOME/vm_check"
REPORT="$OUT/report.md"
DATA="$HOME/workspace/datasets"
mkdir -p "$OUT" "$DATA"
: > "$REPORT"

say() { echo -e "$*" | tee -a "$REPORT"; }

# run <name> <command...>: logs to $OUT/<name>.log, records exit code and wall time
run() {
  local name="$1"; shift
  echo ">>> $name"
  local t0=$SECONDS
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  say "- **$name**: $([ $rc -eq 0 ] && echo OK || echo "FAILED (exit $rc)"), $((SECONDS - t0)) s  (log: $name.log)"
  return $rc
}

# attach <title> <file>: copy a summary table into the report
attach() {
  [ -f "$2" ] && { say "\n#### $1\n"; cat "$2" >> "$REPORT"; say ""; }
}

# ----------------------------------------------------------------------------- #
say "# Lab-machine check — $(date '+%Y-%m-%d %H:%M')\n"
say "## Machine\n"
say "- host: $(hostname), $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r)"
say "- CPU: $(lscpu | sed -n 's/^Model name:[ ]*//p' | head -1), $(nproc) cores"
say "- RAM: $(free -h | awk '/^Mem:/{print $2" total, "$7" available"}')"
say "- disk free in \$HOME: $(df -h "$HOME" | awk 'NR==2{print $4}')"
say "- python: $(python3 --version 2>&1)"
if command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1; then
  GPU=1
  say "- GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
else
  GPU=0
  say "- GPU: none visible (nvidia-smi missing or no device)"
fi
if command -v glxinfo > /dev/null; then
  say "- OpenGL: $(glxinfo -B 2>/dev/null | sed -n 's/^OpenGL renderer string: //p')"
else
  say "- OpenGL: glxinfo not installed (sudo apt install mesa-utils) — matters only for Gazebo cameras"
fi
say "- display: ${DISPLAY:-none (RViz/Gazebo need a remote desktop)}"

# ----------------------------------------------------------------------------- #
say "\n## Build + unit tests\n"
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  say "- **ROS 2 Jazzy not found in /opt/ros/jazzy** — stopping here."
  exit 1
fi
source /opt/ros/jazzy/setup.bash
say "- ROS: $ROS_DISTRO"
python3 -c "import scipy, matplotlib, cv2" 2> /dev/null \
  || say "- missing Python packages: sudo apt install -y python3-scipy python3-matplotlib python3-opencv"

build() {  # build <workspace>: colcon build + test, then source it
  local ws="$REPO/$1"
  run "build_$1" bash -c "cd '$ws' && colcon build --symlink-install" || return 1
  run "test_$1" bash -c "cd '$ws' && colcon test --event-handlers console_direct+ && colcon test-result --verbose"
  source "$ws/install/setup.bash"
}
build ros2_sensor_motion_ws   # W1 (underlay of W2 and W4)
build ros2_localization_ws    # W2
build ros2_planning_ws        # W4
build ros2_visual_slam_ws     # W3

# ----------------------------------------------------------------------------- #
say "\n## Offline benchmarks (CPU)\n"
run w1_benchmark ros2 run sensor_motion_lab offline_benchmark --out "$OUT/w1"
run w2_train ros2 run localization_lab train_pose_regressor
run w2_benchmark ros2 run localization_lab localization_benchmark --out "$OUT/w2"
run w3_benchmark_synthetic ros2 run visual_slam vslam_benchmark --configs orb,sift,orb@thumbnail --out "$OUT/w3_synth"
run w4_benchmark ros2 run planning_lab planning_benchmark --out "$OUT/w4"
attach "W2 localization" "$OUT/w2/summary.md"
attach "W3 visual SLAM (synthetic)" "$OUT/w3_synth/summary.md"
attach "W4 planning" "$OUT/w4/summary.md"

# ----------------------------------------------------------------------------- #
say "\n## Live ROS 2 (60 s each, no RViz)\n"
live() {  # live <name> <launch args...>: run headless, keep the last diagnostics lines
  local name="$1"; shift
  # timeout's exit code 124 = it ran the full 60 s, which is what we want
  run "$name" bash -c 'timeout --signal=INT 60 ros2 launch "$@" rviz:=false; rc=$?; [ $rc -eq 124 ] || [ $rc -eq 0 ]' _ "$@"
  say "\n\`\`\`plaintext"
  grep -E "processed_fps|ate_rmse|error_now|success=|ms_per" "$OUT/$name.log" | tail -4 | cut -c1-300 >> "$REPORT"
  say "\`\`\`"
}
live live_w1 sensor_motion_lab sim.launch.py
live live_w2_mcl localization_lab localization.launch.py
live live_w3_orb visual_slam vslam.launch.py
live live_w4_dstar planning_lab planning.launch.py

# ----------------------------------------------------------------------------- #
if [ $TUM -eq 1 ]; then
  say "\n## Real data (TUM RGB-D)\n"
  for seq in rgbd_dataset_freiburg1_xyz rgbd_dataset_freiburg1_room; do
    [ -d "$DATA/$seq" ] || run "download_$seq" bash -c \
      "cd '$DATA' && wget -q https://cvg.cit.tum.de/rgbd/dataset/freiburg1/$seq.tgz && tar xzf $seq.tgz && rm $seq.tgz"
  done
  run w3_benchmark_tum ros2 run visual_slam vslam_benchmark --source tum \
    --path "$DATA/rgbd_dataset_freiburg1_xyz" --configs orb,sift --out "$OUT/w3_tum"
  attach "W3 visual SLAM (TUM fr1_xyz, 640x480 @ 30 Hz)" "$OUT/w3_tum/summary.md"
fi

# ----------------------------------------------------------------------------- #
if [ $DL -eq 1 ]; then
  say "\n## Deep learning (W3)\n"
  DEV=cpu; TORCH_INDEX=(--index-url https://download.pytorch.org/whl/cpu)
  if [ $GPU -eq 1 ]; then DEV=cuda; TORCH_INDEX=(); fi
  PIP=(pip install --user --break-system-packages "numpy<2")
  run pip_torch "${PIP[@]}" torch torchvision "${TORCH_INDEX[@]}"
  run pip_models "${PIP[@]}" "transformers>=4.45" git+https://github.com/cvg/LightGlue.git
  say "- torch: $(python3 -c 'import torch; print(torch.__version__, "cuda" if torch.cuda.is_available() else "cpu only")' 2>&1 | tail -1)"
  run dl_check ros2 run visual_slam dl_check --device $DEV --out "$OUT/dl_check.md"
  attach "dl_check ($DEV)" "$OUT/dl_check.md"
  run w3_dl_synthetic ros2 run visual_slam vslam_benchmark --device $DEV --max-frames 300 \
    --configs orb,superpoint,superpoint+lightglue,orb@dinov2 --out "$OUT/w3_dl_synth"
  attach "W3 learned components (synthetic, 300 frames)" "$OUT/w3_dl_synth/summary.md"
  if [ $TUM -eq 1 ]; then
    run w3_dl_tum ros2 run visual_slam vslam_benchmark --source tum --device $DEV --max-frames 300 \
      --path "$DATA/rgbd_dataset_freiburg1_xyz" --configs orb,superpoint+lightglue --out "$OUT/w3_dl_tum"
    attach "W3 learned features (TUM fr1_xyz, 300 frames)" "$OUT/w3_dl_tum/summary.md"
    run w3_dl_loops ros2 run visual_slam vslam_benchmark --source tum --device $DEV \
      --path "$DATA/rgbd_dataset_freiburg1_room" --configs orb,orb@thumbnail,orb@dinov2 --out "$OUT/w3_dl_loops"
    attach "W3 learned loop detection (TUM fr1_room)" "$OUT/w3_dl_loops/summary.md"
    run w3_dl_depth ros2 run visual_slam vslam_benchmark --source tum --device $DEV --max-frames 150 \
      --path "$DATA/rgbd_dataset_freiburg1_xyz" --configs orb --depth learned --out "$OUT/w3_dl_depth"
    attach "W3 learned depth (TUM fr1_xyz, 150 frames)" "$OUT/w3_dl_depth/summary.md"
  fi
fi

# ----------------------------------------------------------------------------- #
tar czf "$HOME/vm_check.tar.gz" -C "$HOME" vm_check
say "\nDone. Send back: $REPORT (and $HOME/vm_check.tar.gz if anything FAILED)."
