
import numpy as np

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns a tuple of two sorted coordinate lists:
    group and liberties.
    """

    board = np.array(board)
    n = board.shape[0]

    color = board[row, col]

    visited = set()
    liberties = set()
    group = []

    stack = [(row, col)]
    visited.add((row, col))

    # Four possible directions
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    while stack:
        r, c = stack.pop()
        group.append((r, c))

        for dr, dc in directions:
            nr = r + dr
            nc = c + dc

            # Check board boundaries
            if 0 <= nr < n and 0 <= nc < n:

                # Empty intersection = liberty
                if board[nr, nc] == 0:
                    liberties.add((nr, nc))

                # Same-colour connected stone
                elif board[nr, nc] == color:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))

    return sorted(group), sorted(liberties)
