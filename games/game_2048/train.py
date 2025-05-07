import time
from copy import deepcopy

from .game import newGame, display, winCheck
from .logic import move, checkGameStatus, fillTwoOrFour
from .rlagent import RandomAgent


def main():
    board = newGame("light", 2048, (500, 500))
    status = checkGameStatus(board, 2048)
    agent = RandomAgent()
    while status == "PLAY":
        direction = agent.get_action(board)
        new_board = move(direction, deepcopy(board))
        if new_board != board:
            board = fillTwoOrFour(new_board)

            # Update game status
            status = checkGameStatus(board, 2048)

        display(board, "light", (500, 500))
        time.sleep(0.5)


if __name__ == "__main__":
    main()
