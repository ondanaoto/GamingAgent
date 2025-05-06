from typing import Literal

class RLAgent:

    def get_action(self, board: list[list[int]]) -> Literal["w", "s", "a", "d"]:
        """
        Get the next action for the agent to take.

        Parameters:
            board (list): game board
        Returns:
            (str): action to be taken by the agent
        """
        pass
