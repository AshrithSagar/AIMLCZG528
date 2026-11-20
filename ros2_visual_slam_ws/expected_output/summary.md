# VSLAM benchmark — synthetic_room_loop

Machine: 2 CPUs, OpenCV 5.0.0, torch None, CUDA no
Input: 660 frames, 320x240 @ 15 Hz, depth: sensor

Configuration = `features[+matcher][@place_recognition]` (defaults: brute-force matching, proximity loop search).

| configuration        | device | fps                                                                                                                           | real-time? | extract ms | match ms | PnP ms | loop+PGO ms | ATE VO cm | ATE SLAM cm | RPE cm/s | KFs | loops | lost |
| -------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- | -------- | ------ | ----------- | --------- | ----------- | -------- | --- | ----- | ---- |
| orb                  | cpu    | 82.4                                                                                                                          | yes        | 3.1        | 6.7      | 1.0    | 0.3         | 7.1       | 4.9         | 4.0      | 37  | 2     | 0    |
| sift                 | cpu    | 49.5                                                                                                                          | yes        | 15.5       | 2.9      | 1.0    | 0.2         | 2.6       | 1.8         | 1.6      | 36  | 2     | 0    |
| orb@thumbnail        | cpu    | 82.6                                                                                                                          | yes        | 3.3        | 6.9      | 1.1    | 0.2         | 7.2       | 4.7         | 4.0      | 37  | 1     | 0    |
| akaze                | —      | skipped: cv2.AKAZE_create is not available in OpenCV 5.0.0                                                                    |            |            |          |        |             |           |             |          |     |       |      |
| superpoint           | —      | skipped: 'torch' is not installed: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu (CPU build) |            |            |          |        |             |           |             |          |     |       |      |
| superpoint+lightglue | —      | skipped: 'torch' is not installed: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu (CPU build) |            |            |          |        |             |           |             |          |     |       |      |
| orb@dinov2           | —      | skipped: 'torch' is not installed: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu (CPU build) |            |            |          |        |             |           |             |          |     |       |      |

Real-time = the pipeline keeps up with the camera rate on this machine.
If a learned component is not real-time on CPU but is with `--device cuda`, that is the case for a GPU.

## Loop closure: place recognition

Ground truth: same place = true poses within 1 m and 30°. Recall@1 counts queries whose best candidate (before the geometric check) is a true revisit; false loops = accepted but wrong.

| configuration | queries | with a true revisit | recall@1 | loops accepted | false loops | ms / query |
| ------------- | ------- | ------------------- | -------- | -------------- | ----------- | ---------- |
| orb           | 33      | 2                   | 1.00     | 2              | 0           | 1.5        |
| sift          | 32      | 2                   | 1.00     | 2              | 0           | 0.6        |
| orb@thumbnail | 35      | 4                   | 0.25     | 1              | 0           | 0.2        |
