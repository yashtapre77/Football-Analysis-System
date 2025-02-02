import sys
sys.path.append("../")
from utils import *

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):

        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = 999999
        assigned_player = -1 

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance(ball_position,(player_bbox[0],player_bbox[-1]))
            distance_right = measure_distance(ball_position,(player_bbox[2],player_bbox[-1]))
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id
 
        return assigned_player