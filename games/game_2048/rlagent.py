from typing import Literal
import numpy as np

class RandomAgent:
    def get_action(self, board: list[list[int]]) -> Literal["w", "s", "a", "d"]:
        """
        Get the next action for the agent to take.

        Parameters:
            board (list): game board
        Returns:
            (str): action to be taken by the agent
        """
        idx = np.random.randint(0, 4)
        return ["w", "s", "a", "d"][idx]
