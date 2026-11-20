# W4 path-planning benchmark

Machine: 2 CPUs (Linux-6.18.44-fc-v726-x86_64-with-glibc2.39)

## Static map (30 random queries; reference = A\* + shortcut)

| planner        | success | length / reference (median, max) | min clearance m (median) | turning deg (median) | ms (mean) | work (nodes / evaluations) |
| -------------- | ------- | -------------------------------- | ------------------------ | -------------------- | --------- | -------------------------- |
| astar          | 100 %   | 1.070, 1.160                     | 0.45                     | 293                  | 4.0       | 414                        |
| dstar_lite     | 100 %   | 1.070, 1.160                     | 0.45                     | 203                  | 63.9      | 379                        |
| ga             | 100 %   | 1.000, 1.351                     | 0.40                     | 15                   | 679.1     | 4543                       |
| ga_seeded      | 100 %   | 1.000, 1.043                     | 0.35                     | 18                   | 647.1     | 4297                       |
| astar+shortcut | 100 %   | 1.000, 1.000                     | 0.28                     | 17                   | 3.9       | 414                        |

## Moving obstacles (3 people, 4 start/goal pairs, constant-velocity prediction 1.5 s)

| strategy    | success | collisions (robot moving) | walked into while stopped | time to goal s | replans | ms / replan (mean, max) |
| ----------- | ------- | ------------------------- | ------------------------- | -------------- | ------- | ----------------------- |
| static_plan | 100 %   | 3                         | 2                         | 20.0           | 0       | 0.0, 0                  |
| astar       | 100 %   | 0                         | 1                         | 29.6           | 49      | 3.1, 9                  |
| dstar_lite  | 100 %   | 0                         | 1                         | 29.4           | 43      | 123.4, 415              |
| ga_seeded   | 100 %   | 2                         | 1                         | 34.2           | 76      | 306.5, 1074             |
