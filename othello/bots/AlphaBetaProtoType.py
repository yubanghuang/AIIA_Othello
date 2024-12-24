
import numpy as np
from othello.OthelloUtil import getValidMoves, isEndGame, executeMove

class AlphaBetaBotProtoType():
    def __init__(
        self, 
        search_depth=3,
        evaluate_move=None,
        *args, **kargs
        ): 
        
        self.color=None
        self.search_depth=search_depth
        self.evaluateMove = self.evaluateMove if evaluate_move is None else evaluate_move
        
    def getAction(self, game, color):
        self.color = color
        best_move, best_score = self.findBestMove(game, color, self.search_depth)
        return best_move

    def opponent(self, color):
        return -color
    
    def evaluateMove(self, board, color): 
        return np.sum(board == color)
    
    def findBestMove(
            self, 
            game, 
            current_color, 
            search_depth,
            alpha=-np.inf, 
            beta=np.inf,
            ):

        end_game = isEndGame(game)
        if search_depth <= 0 or end_game is not None:
            return None, self.evaluateMove(game, current_color)

        valid_moves = getValidMoves(game, current_color)
        if valid_moves.size == 0:  
            return None, self.evaluateMove(game, current_color)
        best_score = -np.inf if current_color == self.color else np.inf
        best_move = None

        for move in valid_moves:
            game_copy = game.copy()
            executeMove(game_copy, current_color, move)
            _, score = self.findBestMove(game_copy, self.opponent(current_color), search_depth - 1)
        
            if current_color == self.color:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)

            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)

            if alpha >= beta:
                break
        return best_move, best_score