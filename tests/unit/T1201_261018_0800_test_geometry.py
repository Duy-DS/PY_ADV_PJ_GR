import pytest

from src.utils.geometry import euclidean_distance_2d


@pytest.mark.parametrize(
    ("point1", "point2", "expected"),
    [
        ((0, 0), (0, 0), 0),
        ((0, 0), (3, 0), 3),
        ((0, 0), (0, 4), 4),
        ((0, 0), (3, 4), 5),
        ((-1, -2), (2, 2), 5),
        ((1.5, 2.5), (4.5, 6.5), 5),
    ],
)
def test_euclidean_distance_2d_with_numeric_points(point1, point2, expected):
    assert euclidean_distance_2d(point1, point2) == pytest.approx(expected)
