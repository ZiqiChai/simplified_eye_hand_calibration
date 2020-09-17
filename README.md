# A simplified eye hand calibration repo (cmake project)

## Usage

- eye in hand calibration
- eye to hand calibration

## Input

- image with chessboard
- robot pose in txt file (xyz(mm),rpy(deg))

## Output

- camera intrinsic parameters
- eye hand transformations

## dependency

- OpenCV
- Eigen

## how to use

```bash
mkdir build
cd build
cmake ..
make
```

put your images and robot_pose_txt file under `build/data` folder.

- images should named like `numbers.png`, and index from 0.
- robot_pose_txt file should name with `pose.txt`.
- you can use tools under scripts folder to re-format your input data.

for eye to hand calibration:

```bash
./eye_to_hand_calibration
```

for eye in hand calibration:

```bash
./eye_in_hand_calibration
```

results are in the same folder.