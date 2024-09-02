from typing import List, Tuple
import numpy as np

ORIGIN = (0, 0, 0)

def generate_cube_coords(radius: float, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Generate eight coordinates for a cube.

        Args:
            radius (float): radius of the cube
            center (Tuple[float, float, float]): center of the cube
        
        Returns:
            List[Tuple[float, float, float]]: list of coordinates (x, y, z)
        """
        x, y, z = center
        return [
            (x + radius, y + radius, z + radius),
            (x + radius, y + radius, z - radius),
            (x + radius, y - radius, z + radius),
            (x + radius, y - radius, z - radius),
            (x - radius, y + radius, z + radius),
            (x - radius, y + radius, z - radius),
            (x - radius, y - radius, z + radius),
            (x - radius, y - radius, z - radius),
        ]

def generate_tilted_cube_coords(radius: float, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
    """Generate eight coordinates for a cube that is tilted 45 degrees on the z-axis.

    Args:
        radius (float): radius of the cube
        center (Tuple[float, float, float]): center of the cube
    
    Returns:
        List[Tuple[float, float, float]]: list of coordinates (x, y, z)
    """

    # Hard-coded to rotation of 45 degres on the z-axis
    rot_z = np.matrix([[0.7071, -0.7071, 0.0],
                        [0.7071, 0.7071, 0.0],
                        [0.0, 0.0, 1.0]])

    cube_coords = generate_cube_coords(radius, ORIGIN)
    tilted_cube_coords = []

    # Convert each coord tuple into a np.ndarray and perform matrix rotation with rot_z
    # First rotate, then translate to specified center
    for cube_coord in cube_coords:
        coord = np.asarray(cube_coord)
        rot_coord = tuple(np.squeeze(np.asarray(rot_z @ coord)))
        tilted_cube_coords.append(tuple(map(lambda x, y: x + y, center, rot_coord)))
    return tilted_cube_coords