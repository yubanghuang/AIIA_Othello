import numpy as np

WEIGHT_MATRIX = np.array([
    [100, -25, 10, 10, -25, 100],
    [-25, -50, 5, 5, -50, -25],
    [10, 5, 5, 5, 5, 10],         
    [10, 5, 5, 5, 5, 10],         
    [-25, -50, 5, 5, -50, -25], 
    [100, -25, 10, 10, -25, 100],
])

def weighted_evaluate_move(board, color):
        evaluation = 0
        for i in range(6):
            for j in range(6):
                if board[i][j] == color:
                    evaluation += color * WEIGHT_MATRIX[i][j]
                elif board[i][j] == -color:
                    evaluation -= WEIGHT_MATRIX[i][j]
        return evaluation
