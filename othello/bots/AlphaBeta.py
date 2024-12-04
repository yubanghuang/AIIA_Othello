import numpy as np
from othello.OthelloUtil import getValidMoves, isEndGame, executeMove

class BOT():
    def __init__(self, search_depth=3, *args, **kargs): 
        self.search_depth=search_depth
    

    def getAction(self, game, color):
        self.color = color
        best_move, best_score = self.findBestMove(game, color, self.search_depth)
        return best_move


    def opponent(self, color):
        return -color
    
    def calculateScore(self, board, color): 
        return np.sum(board == color)
    
    def findBestMove(self, game, current_color, search_depth):

        end_game = isEndGame(game)
        if search_depth <= 0 or end_game is not None:
            return None, self.calculateScore(game, current_color)

        valid_moves = getValidMoves(game, current_color)

        if valid_moves.size == 0:  
            return None, self.calculateScore(game, current_color)

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
            
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move, best_score