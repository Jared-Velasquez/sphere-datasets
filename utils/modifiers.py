from typing import List, Tuple
import numpy as np

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.modifiers import (
      RangeMeasurementModel,
      add_landmark_at_position
)


def add_landmarks_to_fg(fg: FactorGraphData, coords: List[Tuple[float, float, float]], range_model: RangeMeasurementModel) -> FactorGraphData:
        """Adds a list of landmarks, where each landmark position is indicated by a tuple (x, y, z)

        Args:
            fg (FactorGraphData): factor graph that will be modified
            coords (List[Tuple[float, float, float]]): list of coordinates (x, y, z)
            range_model (RangeMeasurementModel): Sensor model for range measurements
        
        Returns:
            FactorGraphData: modified factor graph
        """
        new_fg = fg
        for coord in coords:
            new_fg = add_landmark_at_position(new_fg, np.array(coord), range_model)
        return new_fg