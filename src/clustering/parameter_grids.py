from itertools import product


def generate_parameter_grid(
    parameter_dict
):

    keys = parameter_dict.keys()

    values = parameter_dict.values()

    combinations = product(*values)

    return [
        dict(zip(keys, combo))
        for combo in combinations
    ]