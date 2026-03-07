from typing import List


def spaced_combinations(k: int, total_elements: int, min_spacing: int, max_spacing: int) -> List[int]:
    if min_spacing < 1:
        raise RuntimeError("min_spacing must be positive.")

    if max_spacing < min_spacing:
        raise RuntimeError("max_spacing must be greater or equal to min_spacing.")

    first_possible_index_for_first_pick = min_spacing - 1
    last_possible_index_for_first_pick = min(total_elements - min_spacing * k, max_spacing - 1)

    stack = []
    for first_pick in range(first_possible_index_for_first_pick, last_possible_index_for_first_pick + 1):
        stack.append([first_pick])

    complete = []
    while True:
        try:
            picks = stack.pop()
        except IndexError:
            break
        
        if len(picks) == k:
            complete.append(picks)
            continue

        lower_bound = picks[-1] + min_spacing
        upper_bound = min(picks[-1] + max_spacing, total_elements - min_spacing * (k - len(picks)))

        for pick in range(lower_bound, upper_bound + 1):
            new_picks = picks + [pick]
            remaining_space = total_elements - 1 - new_picks[-1]
            if remaining_space < min_spacing - 1:
                break
            elif remaining_space > max_spacing - 1 and len(new_picks) == k:
                continue
            else:
                stack.append(new_picks)

    return complete
