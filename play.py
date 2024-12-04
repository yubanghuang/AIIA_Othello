from AIGamePlatform import Othello
from othello.bots.AlphaBeta import AlphaBeta_BOT


app=Othello() # 和平台建立WebSocket連線
player=AlphaBeta_BOT(search_depth=4)

competition_id='test_robot_6x6_1'
@app.competition(competition_id=competition_id) # 競賽ID
def _callback_(board, color): # 當需要走步會收到盤面及我方棋種
    return player.getAction(board, color) # bot回傳落子座標

