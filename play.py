from AIGamePlatform import Othello

from othello.bots.AlphaBeta import AlphaBetaBot
from othello.bots.evaluator import WeightMatrixEvaluator, StrategyEvaluator, CombinedEvaluator
from othello.Human import Human

app=Othello() # 和平台建立WebSocket連線

evaluator_1 = WeightMatrixEvaluator
evaluator_2 = StrategyEvaluator

my_bot=AlphaBetaBot(
        max_search_depth=6,
        evaluator=evaluator_1,
        paralle_method='multiprocess', 
        cpu_use=6, 
        precompute=False,
        precompute_search_depth=6)


competition_id='test_robot_6x6_1'
@app.competition(competition_id=competition_id) # 競賽ID
def _callback_(board, color): # 當需要走步會收到盤面及我方棋種
    return my_bot.getAction(board, color) # bot回傳落子座標
