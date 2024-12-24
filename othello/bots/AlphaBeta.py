import random
import numpy as np
import multiprocessing
import threading
from typing import Optional, Literal
from othello.OthelloUtil import getValidMoves, isEndGame, executeMove
from othello.bots.evaluator import BaseEvaluator, DefaultEvaluator
class AlphaBetaBot():
    def __init__(
        self, 
        max_search_depth: int = 3,
        evaluator: Optional[BaseEvaluator] = None,
        paralle_method: Literal['sequential', 'multiprocess'] = 'multiprocess',
        cpu_use: int = 1,
        precompute: bool = False,
        precompute_search_depth: int = 3,
        *args, **kargs
    ):
        self.color = None
        self.search_depth = max_search_depth - 3
        self.max_search_depth = max_search_depth
        # Check if evaluator is a class and instantiate it if needed
        if evaluator is not None:
            # Check if it's already an instance, otherwise instantiate
            if isinstance(evaluator, type) and issubclass(evaluator, BaseEvaluator):
                self.evaluator = evaluator(self)  # Instantiate the class
            else:
                self.evaluator = evaluator  # Already an instance
        else:
            self.evaluator = DefaultEvaluator(self)
        
        self.paralle_method = paralle_method
        self.cpu_use = min(cpu_use, max(1, multiprocessing.cpu_count() - 4)) if cpu_use > 0 else 1
        
        self.precompute = precompute
        self.precompute_search_depth = max(max_search_depth, precompute_search_depth)
        self.precomputed_move = None  # Store precomputed move
        self.lock = threading.Lock()  # Mutex Lock for thread safety
        self.precompute_thread_active = False  # Flag to check if precompute is in progress
        self.precomputed_results = {}  # Store partial results
        
        self.corners = []
        self.args = args
        self.kargs = kargs

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def getAction(self, game, color):
        """
        Calculate and execute the best move while predicting the opponent's move.
        """
        self.color = color
        size = game.shape[0]
        self.corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
        
        if self.search_depth < self.max_search_depth:
            self.search_depth += 1
            
        # Step 1: Calculate best move
        best_move, _ = self.findBestMove(game, color, self.search_depth)

        # Step 2: Execute best move
        executeMove(game, color, best_move)
        
        if self.precompute and not self.precompute_thread_active:
            # Step 3: Precompute opponent's next move asynchronously
            precompute_thread = threading.Thread(
                target=self.precomputeNextMove,
                args=(game.copy(), self.opponent(color), self.precompute_search_depth)
            )
            precompute_thread.start()
            self.precompute_thread_active = True

        # Return the move that the bot executed
        return best_move

    def opponent(self, color):
        return -color

    def findBestMove(self, game, current_color, search_depth, alpha=-np.inf, beta=np.inf):
        """
        Check for precomputed move before performing the normal search.
        If part of the computation is done, use it.
        """
        # Check if we have any precomputed results to utilize
        if self.precompute and self.precomputed_move is not None:
            # If there are partial results, continue from where we left off
            if search_depth > self.precompute_search_depth:
                # Use the available precomputed move and evaluate
                return self.precomputed_move, self.evaluator.evaluate(game, current_color)
        
        if self.paralle_method == 'sequential':
            return self.findBestMoveSequential(game, current_color, search_depth, alpha, beta)
        elif self.paralle_method == 'multiprocess':
            return self.findBestMoveMultiprocess(game, current_color, search_depth, alpha, beta)
        else:
            raise ValueError(f"Invalid paralle_method: {self.paralle_method}.")

    def findBestMoveSequential(self, game, current_color, search_depth, alpha=-np.inf, beta=np.inf):
        """
        Sequential version of the Alpha-Beta pruning algorithm.
        Save partial results to improve subsequent searches.
        """
        end_game = isEndGame(game)
        if search_depth <= 0 or end_game is not None:
            return None, self.evaluator.evaluate(game, current_color)

        valid_moves = getValidMoves(game, current_color)
        if valid_moves.size == 0:
            return None, self.evaluator.evaluate(game, current_color)


        best_score = -np.inf if current_color == self.color else np.inf
        best_move = None

        # Store intermediate results for partial pruning
        if search_depth not in self.precomputed_results:
            self.precomputed_results[search_depth] = {}
        
        valid_moves = self.prioritizedSort(valid_moves)
        # random.shuffle(valid_moves)
        
        for move in valid_moves:
            game_copy = game.copy()
            executeMove(game_copy, current_color, move)
            _, score = self.findBestMoveSequential(game_copy, self.opponent(current_color), search_depth - 1, alpha, beta)

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

            # Save partial results
            self.precomputed_results[search_depth][tuple(move)] = score

        return best_move, best_score
    
    def findBestMoveMultiprocess(self, game, current_color, search_depth, alpha=-np.inf, beta=np.inf):
        """ 
        Multiprocess version of the Alpha-Beta pruning algorithm.
        """
        end_game = isEndGame(game)
        if search_depth <= 0 or end_game is not None:
            return None, self.evaluator.evaluate(game, current_color)

        valid_moves = getValidMoves(game, current_color)
        if valid_moves.size == 0:
            return None, self.evaluator.evaluate(game, current_color)
        
        with multiprocessing.Pool(processes=self.cpu_use) as pool:
            results = pool.starmap(
                self.evaluateMoveForMultiprocess,
                [(game, current_color, move, search_depth, alpha, beta) for move in valid_moves]
            )

        best_score = -np.inf if current_color == self.color else np.inf
        best_move = None

        for move, score in results:
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
    
    def evaluateMoveForMultiprocess(self, game, current_color, move, search_depth, alpha, beta):
        game_copy = game.copy()
        executeMove(game_copy, current_color, move)
        _, score = self.findBestMoveSequential(game_copy, self.opponent(current_color), search_depth - 1, alpha, beta)
        return move, score

    def precomputeNextMove(self, game, color, search_depth):
        """
        Precompute the opponent's best move, storing partial results.
        """
        precomputed_move, _ = self.findBestMove(game, color, search_depth)
        with self.lock:
            self.precomputed_move = precomputed_move
        self.precompute_thread_active = False  # Mark precompute as complete

    def getPrecomputedMove(self):
        """
        Get the precomputed next move.
        """
        return self.precomputed_move
    
    def prioritizedSort(self, valid_moves):
        """
        Prioritize corner moves first by sorting the valid moves.
        This places corner positions at the start of the list.
        """
        valid_moves = [tuple(move) for move in valid_moves]

        corner_moves = [move for move in valid_moves if move in self.corners]
        non_corner_moves = [move for move in valid_moves if move not in self.corners]


        return corner_moves + non_corner_moves
