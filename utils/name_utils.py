DEFAULT_PYFG_FILENAME = "synthetic_dataset"

def name_pyfg_file(inter_robot_enabled: bool = False, num_robots: int = None, noise: float = None, meas_prob: float = None, num_landmarks: int = None, add_extension: bool = True) -> str:
    name = ""
    if inter_robot_enabled:
        name += "inter_robot_"
    if num_robots is not None:
        name += f"{num_robots}_robots_"
    if noise is not None:
        name += f"{noise}_stddev_"
    if meas_prob is not None:
        name += f"{meas_prob}_meas_prob_"
    if num_landmarks is not None and num_landmarks > 0:
        name += f"{num_landmarks}_landmarks_"
    if len(name) != 0 and name[-1] == "_":
        name = name[:-1]
    if len(name) == 0:
        name = DEFAULT_PYFG_FILENAME
    if add_extension:
        name += ".pyfg"

    return name