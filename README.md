# Frame Playground


## Transition Criterion
Target seq `A` and target seq `B`, both with `n` frames.

- Flow
  - Predict optical flow fields between subsequent frames of `A`, yielding `n-1` flow fields (`Flo_A`)
  - Predict optical flow fields between subsequent frames of `B`, yielding `n-1` flow fields (`Flo_B`)
  - Slide `Flo_B` across `Flo_A` and for each combination of window overlap...
    - For each overlapping pair of flow fields (`Flo_A[i]` and `Flo_B[i]`)
      - Calculate the average angular difference between `Flo_A[i]` and `Flo_B[i]`
      - Calculate the average flow endpoint error (EPE) between `Flo_A[i]` and `Flo_B[i]`
      - Add the two errors with some heuristic
    - Smooth the error graphs to avoid spikes
    - Find and describe the local minima for that overlap
  - Select the overlap with the best minimum error


- Mask RCNN
  - Predict masks for each frame of `A` and `B`
  - Slide mask frames from `A` over mask frames from `B`
  - Calculate IoU, for each combination
  - Smooth graphs of IoU over frames
  - Identify overlap with the best overlap behavior

- Bounding box
  - Predict bounding boxes for each frame of `A` and `B`
  - Calculate trajectories ("flow") of bounding boxes across frames
  - Smooth over frames
  - Select overlaps with similar trajectories, similar to flow version

- Edges
  - Extract edges for each frame of `A` and `B`, yielding binary images
  - Dilate edges slightly (to encourage overlap of slightly offset objects)
  - Slide frames of `A` over frames of `B`
  - Sum overlap of edges for each pair of frames