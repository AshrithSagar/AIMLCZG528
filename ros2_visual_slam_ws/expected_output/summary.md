# VSLAM benchmark — synthetic_room_loop

Machine: 2 CPUs, OpenCV 5.0.0, torch None, CUDA no
Input: 660 frames, 320x240 @ 15 Hz

| front-end  | device | fps                                                                 | real-time? | extract ms | match ms | PnP ms | loop+PGO ms | ATE VO cm | ATE SLAM cm | RPE cm/s | KFs | loops | lost |
| ---------- | ------ | ------------------------------------------------------------------- | ---------- | ---------- | -------- | ------ | ----------- | --------- | ----------- | -------- | --- | ----- | ---- |
| orb        | cpu    | 100.9                                                               | yes        | 2.5        | 5.7      | 0.8    | 0.3         | 7.1       | 4.9         | 4.0      | 37  | 2     | 0    |
| sift       | cpu    | 68.9                                                                | yes        | 11.0       | 2.2      | 0.7    | 0.1         | 2.6       | 1.8         | 1.6      | 36  | 2     | 0    |
| akaze      | —      | skipped: cv2.AKAZE_create is not available in OpenCV 5.0.0          |            |            |          |        |             |           |             |          |     |       |      |
| superpoint | —      | skipped: feature_type 'superpoint' needs PyTorch: pip install torch |            |            |          |        |             |           |             |          |     |       |      |

Real-time = the pipeline keeps up with the camera rate on this machine.
If a learned front-end is not real-time on CPU but is with `--device cuda`, that is the case for a GPU.
