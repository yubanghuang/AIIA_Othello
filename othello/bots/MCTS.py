import numpy as np
import random
from othello.OthelloUtil import getValidMoves, isEndGame, executeMove

class MCTSNode:
    def __init__(self, state, parent=None, move=None, color=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.color = color
        self.visits = 0
        self.wins = 0
        self.children = []

    def uct_value(self, exploration_weight=1.0):
        if self.visits == 0:
            return float('inf')  # Unvisited node has the highest priority
        return self.wins / self.visits + exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits)


class MCTS_BOT():
    def __init__(self, 
                 search_depth=3,
                 evaluate_move=None,
                 mcts_simulations=1000,
                 *args, **kargs):
        self.color = None
        self.search_depth = search_depth
        self.evaluateMove = self.evaluateMove if evaluate_move is None else evaluate_move
        self.mcts_simulations = mcts_simulations
    
    def getAction(self, game, color):
        self.color = color
        best_move = self.mcts(game, self.color)
        return best_move

    def opponent(self, color):
        return -color
    
    def evaluateMove(self, board, color): 
        return np.sum(board == color)
    
    def select(self, node):
        while node.children:  # Traverse until we find a leaf node
            node = max(node.children, key=lambda child: child.uct_value())
        return node
    
    def expand(self, node):
        valid_moves = getValidMoves(node.state, node.color)

        # If there are no valid moves, the current player has to pass their turn.
        if len(valid_moves) == 0:
            # In Othello, if a player has no valid moves, they must pass.
            # We pass the turn to the opponent and expand the node accordingly.
            # We create a child node that represents the opponent's turn.
            new_state = node.state.copy()
            child_node = MCTSNode(new_state, parent=node, move=None, color=self.opponent(node.color))
            node.children.append(child_node)
            return child_node  # Return this new node (representing the opponent's turn)

        # If there are valid moves, expand the node as usual.
        for move in valid_moves:
            new_state = node.state.copy()
            executeMove(new_state, node.color, move)
            child_node = MCTSNode(new_state, parent=node, move=move, color=self.opponent(node.color))
            node.children.append(child_node)

        # Return the first child for expansion, this can be improved for more dynamic selection.
        return node.children[0]
    
    def simulate(self, node):
        game_copy = node.state.copy()
        current_color = node.color
        while not isEndGame(game_copy):  # Play random game
            valid_moves = getValidMoves(game_copy, current_color)
            if len(valid_moves) == 0:  # No valid moves
                break
            move = random.choice(valid_moves)
            executeMove(game_copy, current_color, move)
            current_color = self.opponent(current_color)
        return self.evaluateMove(game_copy, node.color)
    
    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.wins += result  # Update win count based on result
            node = node.parent
    
    def mcts(self, game, color):
        root = MCTSNode(game, color=color)
        
        for _ in range(self.mcts_simulations):
            node = self.select(root)  # Select the best node
            if not isEndGame(node.state):  # If not terminal, expand it
                self.expand(node)
            result = self.simulate(node)  # Simulate a random game
            self.backpropagate(node, result)  # Update nodes with simulation result
        
        # Select the move with the most visits (best move)
        best_move = max(root.children, key=lambda child: child.visits).move
        return best_move
