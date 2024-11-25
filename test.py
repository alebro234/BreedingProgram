from collections import defaultdict
import numpy as np


def group_and_process(dict_list, group_keys, avg=True):
    """
    Groups dictionaries by specified keys and either averages or lists other keys.

    Args:
        dict_list (list): List of dictionaries to be grouped.
        group_keys (list): Keys to group by.
        avg (bool): If True, computes averages for non-grouping keys; 
                    if False, collects them into lists.

    Returns:
        list: A list of grouped dictionaries with processed non-grouping keys.
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for d in dict_list:
        # Create a tuple of grouping values
        group_values = tuple(d[key] for key in group_keys)
        for key, value in d.items():
            if key in group_keys:
                # Set grouping keys
                grouped[group_values][key] = value
            else:
                # Collect values for averaging or listing
                grouped[group_values][key].append(value)

    # Build the result with processed values for non-grouping keys
    result = []
    for group_values, combined_dict in grouped.items():
        processed_dict = {key: value for key, value in combined_dict.items()}
        for key, values in combined_dict.items():
            if key not in group_keys:
                if avg:
                    processed_dict[key] = np.mean(values)  # Compute average
                else:
                    processed_dict[key] = values  # Collect into a list
        result.append(processed_dict)

    return result


dict1 = {
    "pop_size": 100,
    "n_genes": 25,
    "sel": "tournament",
    "cross": "1p",
    "mut": "flip",
    "ps": 0.2,
    "pc": 0.3,
    "pm": 0.08,
    "T0": 0,
    "alpha": 0,
    "max_gen": 350
}

dict2 = {
    "pop_size": 100,
    "n_genes": 25,
    "sel": "tournament",
    "cross": "1p",
    "mut": "flip",
    "ps": 0.3,
    "pc": 0.4,
    "pm": 0.1,
    "T0": 0,
    "alpha": 0,
    "max_gen": 350
}

dict3 = {
    "pop_size": 50,
    "n_genes": 20,
    "sel": "tournament",
    "cross": "1p",
    "mut": "flip",
    "ps": 0.1,
    "pc": 0.2,
    "pm": 0.05,
    "T0": 1,
    "alpha": 0.5,
    "max_gen": 100
}

dict_list = [dict1, dict2, dict3]

dict_list = group_and_process(
    dict_list, ["pop_size", "n_genes", "sel", "cross", "mut"], avg=True)

dict_list = group_and_process(dict_list, ["sel", "cross", "mut"], avg=False)

for d in dict_list:
    print(d)
