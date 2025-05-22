import numpy as np

ISAID_palette = np.array([
    [0, 0, 0],  # 0 - 背景
    [63, 0, 0],  # 1 - ship
    [63, 63, 0],  # 2 - storage_tank
    [255, 63, 0],  # 3 - baseball_diamond
    [127, 63, 0],  # 4 - tennis_court
    [191, 63, 0],  # 5 - basketball_court
    [0, 63, 0],  # 6 - Ground_Track_Field
    [63, 127, 0],  # 7 - bridge
    [127, 127, 0],  # 8 - large_vehicle
    [127, 0, 0],  # 9 - small_vehicle
    [191, 0, 0],  # 10 - helicopter
    [255, 0, 0],  # 11 - swimming_pool
    [127, 191, 0],  # 12 - Roundabout
    [191, 127, 0],  # 13 - Soccer_ball_field
    [255, 127, 0],  # 14 - plane
    [155, 100, 0]  # 15 - harbor
], dtype=np.uint8)

Potsdam_palette = np.array([
    [255, 255, 255],  # 0  # surface
    [  0,   0, 255],  # 1  # building
    [  0, 255, 255],  # 2  # low vegetation
    [  0, 255,   0],  # 3  # tree
    [255, 255,   0],  # 4  # car
    [255,   0,   0]   # 5  # clutter/background red
], dtype=np.uint8)

Deepglobe_palette = np.array([
    [0, 255, 255],  # 0  -> Yellow
    [255, 255, 0],  # 1  -> Cyan
    [255, 0, 255],  # 2  -> Magenta
    [0, 255, 0],  # 3  -> Green
    [0, 0, 255],  # 4  -> Red
    [255, 255, 255],  # 5  -> White
    [0, 0, 0]  # 6  -> Black
], dtype=np.uint8)