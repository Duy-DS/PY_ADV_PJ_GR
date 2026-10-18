import math


# Tính khoảng cách Euclidean giữa hai điểm trong không gian 2D
def euclidean_distance_2d(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
