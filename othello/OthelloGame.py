import numpy as np
from othello.OthelloUtil import getValidMoves, executeMove, isValidMove, isEndGame
import time

# 需要連上平台比賽不需要使用此code
# 可使用此code進行兩個bot對弈

BLACK = 1
WHITE = -1
PLAYER_TO_STR = {BLACK: 'B', WHITE:'W', 0: '_'}
class OthelloGame(np.ndarray):
    
    def __new__(cls, n, *args, **kwargs):
        obj = super().__new__(cls, shape=(n, n), dtype='int')
        return obj

    def __init__(self, n, time_limit=180):
        self.n=n
        self.current_player=BLACK
        # clear board
        self[np.where(self!=0)]=0 
        # n/2 = center of board
        self[int(n/2)][int(n/2)]=WHITE
        self[int(n/2)-1][int(n/2)-1]=WHITE
        self[int(n/2)-1][int(n/2)]=BLACK
        self[int(n/2)][int(n/2)-1]=BLACK
        
        """ custom member variables """
        self.player_remaining_time={BLACK: time_limit, WHITE: time_limit}
        self.player_cost_time = {BLACK: 0, WHITE: 0}
        self.timer=0
        
    def move(self, position):
        if isValidMove(self, self.current_player, position):
            executeMove(self, self.current_player, position)
            y, x = position.copy()
            print(PLAYER_TO_STR[self.current_player] + ' Place at ' + '(' + str(y+1) + ',' + str(chr(x+ord('A'))) + ')')
        else:
            raise Exception('invalid move')
    
    def play(self, black, white, verbose=True):
        while isEndGame(self) == None:
            if verbose:
                print('{:#^30}'.format( ' Player '+ PLAYER_TO_STR[self.current_player] + ' ' ))
                self.showBoard()
            if len(getValidMoves(self, self.current_player))==0:
                if verbose:
                    print('no valid move, next player')
                self.current_player=-self.current_player
                continue
            
            self.startMoveTimer()
            if self.current_player==WHITE:
                position=white.getAction(self.clone(), self.current_player)
            else:
                position=black.getAction(self.clone(), self.current_player)
                
            try:
                self.move(position)
            except:
                if verbose:
                    print('invalid move', end='\n\n')
                continue
            self.stopMoveTimer()
            
            print()
            self.current_player=-self.current_player
        
        if verbose:
            print('---------- Result ----------', end='\n\n')
            self.showBoard()
            print()

            winner = isEndGame(self)
            if isEndGame(self) == WHITE:
                winner = 'W'
            elif isEndGame(self) == BLACK:
                winner = 'B'
            else:
                winner = 'None'
            print('Winner:', winner)
            self.displayAllPlayerCostTime()
            self.displayScores()
        return isEndGame(self)
    
    def showBoard(self):
        corner_offset_format='{:^'+str(len(str(self.n))+1)+'}'
        print(corner_offset_format.format(''), end='')
        for i in range(self.n):
            print('{:^3}'.format( chr(ord('A')+i) ), end='')
        print()
        # print(corner_offset_format.format(''), end='')
        # for i in range(self.n):
        #     print('{:^3}'.format('-'), end='')
        # print()
        for i in range(self.n):
            print(corner_offset_format.format(i+1), end='')
            for j in range(self.n):
                if isValidMove(self, self.current_player, (i,j)):
                    print('{:^3}'.format('∎'), end='')
                else:
                    print('{:^3}'.format(PLAYER_TO_STR[self[i][j]]), end='')
            print()
    def clone(self):
        new=self.copy()
        new.n=self.n
        new.current_player=self.current_player
        return new


    """ custom method """
    
    def resetBoard(self):
        n = self.n
        self.current_player=BLACK
        self[np.where(self!=0)]=0 
        self[int(n/2)][int(n/2)]=WHITE
        self[int(n/2)-1][int(n/2)-1]=WHITE
        self[int(n/2)-1][int(n/2)]=BLACK
        self[int(n/2)][int(n/2)-1]=BLACK

    def startMoveTimer(self):
        if self.timer == 0 :
            self.timer = 1
            self.start_time = time.time()

    def stopMoveTimer(self):
        if self.timer == 1:
            self.timer = 0
            time_cost = time.time() - self.start_time
            self.calculatePlayerCostTime(time_cost)
    
    def calculatePlayerCostTime(self, time_cost):
        self.player_cost_time[self.current_player] += time_cost
        self.player_remaining_time[self.current_player] -= time_cost
        remaining_time = self.player_remaining_time[self.current_player]
        print(f"Player {PLAYER_TO_STR[self.current_player]} spent {time_cost:.2f}s on move.")
        print(f"Player {PLAYER_TO_STR[self.current_player]} Remaining Time: {remaining_time:.2f}s.")
        
    def displayAllPlayerCostTime(self):
        
        black_time = self.player_cost_time[BLACK]
        print(f"Player {PLAYER_TO_STR[BLACK]} Total Time: {black_time:.2f}s")
        print(f"Player {PLAYER_TO_STR[BLACK]} Remaining Time: {self.player_remaining_time[BLACK]:.2f}s.")
        
        white_time = self.player_cost_time[WHITE]
        print(f"Player {PLAYER_TO_STR[WHITE]} Total Time: {white_time:.2f}s")
        print(f"Player {PLAYER_TO_STR[WHITE]} Remaining Time: {self.player_remaining_time[WHITE]:.2f}s.")
    
    def displayScores(self):
        print(f"Player {PLAYER_TO_STR[BLACK]} Scores:{np.sum(self == BLACK)}")
        print(f"Player {PLAYER_TO_STR[WHITE]} Scores:{np.sum(self == WHITE)}")
    