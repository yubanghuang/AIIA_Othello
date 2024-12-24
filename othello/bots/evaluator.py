import numpy as np
from othello.OthelloUtil import getValidMoves, executeMove
# Base class for evaluating moves
class BaseEvaluator:
    def __init__(self, bot=None):
        self.bot = bot

    def evaluate(self, board, color):
        raise NotImplementedError("Subclasses should implement the 'evaluate' method.")

class DefaultEvaluator(BaseEvaluator):
    def evaluate(self, board, color):
        current_discs = np.sum(board == color)
        opponent_discs = np.sum(board == -color)
        
        current_moves = len(getValidMoves(board, color))
        opponent_moves = len(getValidMoves(board, -color))
        
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_control = sum([1 if board[x, y] == color else 0 for x, y in corners])
        opponent_corner_control = sum([1 if board[x, y] == -color else 0 for x, y in corners])
        corner_score = 3 * (corner_control - opponent_corner_control)
        
        return (current_discs + current_moves - opponent_discs - opponent_moves + corner_score)

WEIGHT_MATRIX = np.array([
    [150, -30, 20, 20, -30, 150],
    [-30, -50,  5,  5, -50, -30],
    [ 20,   5,  1,  1,   5,  20],
    [ 20,   5,  1,  1,   5,  20],
    [-30, -50,  5,  5, -50, -30],
    [150, -30, 20, 20, -30, 150]
])

class WeightMatrixEvaluator(BaseEvaluator):
    def __init__(self, bot=None, weight_matrix=None):
        super().__init__(bot)
        self.weight_matrix = weight_matrix if weight_matrix is not None else WEIGHT_MATRIX
        
    def evaluate(self, board, color):
        return self._calculate_weighted_score(board, color) - self._calculate_weighted_score(board, -color)
    
    def _calculate_weighted_score(self, board, color):
        weighted_score = 0
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] == color:
                    weighted_score += self.weight_matrix[i, j]
        return weighted_score

class MaxMovesEvaluator(BaseEvaluator):
    def evaluate(self, board, color):
        return len(getValidMoves(board, color))


EARLY_MATRIX = np.array([
    [120, -20,  20,  20, -20, 120],
    [-20, -40,   5,   5, -40, -20],
    [ 20,   5,   1,   1,   5,  20],
    [ 20,   5,   1,   1,   5,  20],
    [-20, -40,   5,   5, -40, -20],
    [120, -20,  20,  20, -20, 120],
])

MID_MATRIX = np.array([
    [100, -30,  10,  10, -30, 100],
    [-30, -50,   2,   2, -50, -30],
    [ 10,   2,   0,   0,   2,  10],
    [ 10,   2,   0,   0,   2,  10],
    [-30, -50,   2,   2, -50, -30],
    [100, -30,  10,  10, -30, 100],
])

LATE_MATRIX = np.array([
    [200,  10,  20,  20,  10, 200],
    [ 10,  30,  10,  10,  30,  10],
    [ 20,  10,   5,   5,  10,  20],
    [ 20,  10,   5,   5,  10,  20],
    [ 10,  30,  10,  10,  30,  10],
    [200,  10,  20,  20,  10, 200],
])


class DynamicWeightMatrixEvaluator(BaseEvaluator):
    def __init__(self, bot, early_matrix=None, mid_matrix=None, late_matrix=None):
        super().__init__(bot)
        self.early_matrix = early_matrix if early_matrix is not None else EARLY_MATRIX
        self.mid_matrix = mid_matrix if mid_matrix is not None else MID_MATRIX
        self.late_matrix = late_matrix if late_matrix is not None else LATE_MATRIX
    
    def evaluate(self, board, color):
        total_pieces = np.count_nonzero(board)
        total_slots = board.size
        # Determine phase
        if total_pieces < total_slots * 0.3:  # Early game
            weight_matrix = self.early_matrix
        elif total_pieces < total_slots * 0.7:  # Mid game
            weight_matrix = self.mid_matrix
        else:  # Late game
            weight_matrix = self.late_matrix
        
        board_values = board * weight_matrix
        return np.sum(board_values[board == color]) - np.sum(board_values[board == -color])

    
    
    

def countStableDiscs(board, color):
        """
        Count the number of stable discs for the given color.
        Stable discs are those that cannot be flipped.
        """
        stable_discs = 0
        rows, cols = board.shape
        
        for r in range(rows):
            for c in range(cols):
                if board[r, c] == color and isStable(board, r, c):
                    stable_discs += 1
        
        return stable_discs

def isStable(board, r, c):
    """
    Check if the disc at position (r, c) is stable.
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in directions:
        x, y = r, c
        while 0 <= x < board.shape[0] and 0 <= y < board.shape[1]:
            if board[x, y] == 0:
                return False
            x += dr
            y += dc
    return True

class StrategyEvaluator(BaseEvaluator):
    def __init__(
        self, 
        bot=None,
        amount_moves_weight=2.0,
        coin_parity_weight=1.0,
        mobility_weight=2.0,
        corner_occupancy_weight=5,
        stability_weight=3.0,
        edge_occupancy_weight=2.5,
        corner_proximity_penalty=1.5,
        diagonal_corner_penalty=1.0 
    ):
        super().__init__(bot)
        self.amount_moves_weight = amount_moves_weight 
        self.coin_parity_weight = coin_parity_weight
        self.mobility_weight = mobility_weight
        self.corner_occupancy_weight = corner_occupancy_weight
        self.stability_weight = stability_weight
        self.edge_occupancy_weight = edge_occupancy_weight
        self.corner_proximity_penalty = corner_proximity_penalty
        self.diagonal_corner_penalty = diagonal_corner_penalty
        
        self.corners = [(0, 0), (0, 5), (5, 0), (5, 5)]
        self.edges = [(0, i) for i in range(1, 5)] + \
                     [(5, i) for i in range(1, 5)] + \
                     [(i, 0) for i in range(1, 5)] + \
                     [(i, 5) for i in range(1, 5)]
        
        self.corner_adjacent_edges = {
            (0, 0): [(0, 1), (1, 0), (1, 1)],
            (0, 5): [(0, 4), (1, 5), (1, 4)],
            (5, 0): [(4, 0), (5, 1), (4, 1)],
            (5, 5): [(5, 4), (4, 5), (4, 4)]
        }
        
        self.corners_diagonal = {
            (0, 0): [(1, 1)],
            (0, 5): [(1, 4)], 
            (5, 0): [(4, 1)],  
            (5, 5): [(4, 4)]   
        }

    def evaluate(self, board, color):
        amount_moves = self._calculate_amount_moves(board, color)
        coin_parity = self._calculate_coin_parity(board, color)
        mobility = self._calculate_mobility(board, color)
        corner_occupancy = self._calculate_corner_occupancy(board, color)
        stability = self._calculate_stability(board, color)
        edge_occupancy = self._calculate_edge_occupancy(board, color)
        

        corner_penalty = self._calculate_corner_proximity_penalty(board, color)
        
        diagonal_corner_penalty = self._calculate_diagonal_corner_penalty(board, color)
        
        evaluation_score =  (
            self.amount_moves_weight * amount_moves +
            self.coin_parity_weight * coin_parity + 
            self.mobility_weight * mobility + 
            self.corner_occupancy_weight * corner_occupancy + 
            self.stability_weight * stability + 
            self.edge_occupancy_weight * edge_occupancy + 
            corner_penalty + diagonal_corner_penalty
        )
        
        return evaluation_score

    def _calculate_coin_parity(self, board, color):
        current_discs = np.sum(board == color)
        opponent_discs = np.sum(board == -color)
        return current_discs - opponent_discs

    def _calculate_mobility(self, board, color):
        current_moves = len(getValidMoves(board, color))
        opponent_moves = len(getValidMoves(board, -color))
        return current_moves - opponent_moves

    def _calculate_corner_occupancy(self, board, color):
        current_corner_count = sum([1 for corner in self.corners if board[corner] == color])
        opponent_corner_count = sum([1 for corner in self.corners if board[corner] == -color])
        return current_corner_count - opponent_corner_count

    def _calculate_edge_occupancy(self, board, color):
        current_edge_count = sum([1 for pos in self.edges if board[pos] == color])
        opponent_edge_count = sum([1 for pos in self.edges if board[pos] == -color])
        return current_edge_count - opponent_edge_count

    def _calculate_amount_moves(self, board, color):
        current_amount_moves = len(getValidMoves(board, color))
        opponent_amount_moves = len(getValidMoves(board, -color))
        return current_amount_moves - opponent_amount_moves

    def _calculate_stability(self, board, color):
        current_stable_count = 0
        opponent_stable_count = 0
        rows, cols = board.shape
        
        def isStableDisk(r, c):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Horizontal and Vertical
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
            for dr, dc in directions:
                x, y = r, c
                while 0 <= x + dr < rows and 0 <= y + dc < cols:
                    x += dr
                    y += dc
                    if board[x, y] == 0:
                        return False
                    if board[x, y] == -color:
                        break
                    if board[x, y] == color:
                        return True
            return False

        for r in range(rows):
            for c in range(cols):
                if isStableDisk(r, c):
                    if board[r, c] == color:
                        current_stable_count += 1
                    elif board[r, c] == -color:
                        opponent_stable_count += 1
        return current_stable_count - opponent_stable_count

    def _calculate_corner_proximity_penalty(self, board, color):
        """
        Penalize edges near corners occupied by the opponent.
        """
        penalty = 0
        for corner in self.corners:
            # Check if the opponent occupies the corner
            if board[corner] == -color:
                # If the opponent occupies this corner, apply penalty to adjacent edge positions
                for edge in self.corner_adjacent_edges[corner]:
                    if board[edge] == color:
                        penalty -= self.corner_proximity_penalty  # Penalize the player's piece
        return penalty

    def _calculate_diagonal_corner_penalty(self, board, color):
        """
        Apply penalty to pieces adjacent to diagonal corners occupied by the opponent.
        """
        penalty = 0

        # Check for diagonal corners and apply penalty to adjacent positions
        for corner, diagonal_positions in self.corners_diagonal.items():
            for r, c in diagonal_positions:
                # If the current player occupies any of the diagonal positions, apply the penalty
                if board[r, c] == color:
                    penalty -= self.diagonal_corner_penalty
        return penalty