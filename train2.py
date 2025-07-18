import os
import sys
import random
import numpy as np
import RocketSim as rsim
import rlviser_py as rlviser
from rlgym_ppo import Learner
from rlgym.api import StateMutator
from gym.vector import SyncVectorEnv
from RocketSim import Arena, GameMode
from typing import List, Dict, Any, Tuple
from rlgym_ppo.util import RLGymV2GymWrapper
from rlgym.rocket_league import common_values
from torch.utils.tensorboard import SummaryWriter
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.api import RewardFunction, AgentID, RLGym, Renderer
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardProvider





from rlgym.rocket_league.common_values import (
    CAR_MAX_SPEED,
    BALL_MAX_SPEED,
    BACK_NET_Y,
    SIDE_WALL_X,
    CEILING_Z,
    CAR_MAX_ANG_VEL,
    BOOST_LOCATIONS
)

from rlgym.rocket_league.done_conditions import (
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
    AnyCondition
)

from rlgym.rocket_league.state_mutators import (
    MutatorSequence,
    FixedTeamSizeMutator,
    KickoffMutator
)

from rlbot.utils.game_state_util import (
    GameState as RLBotGameState,
    BallState,
    CarState,
    Physics,
    Vector3,
    Rotator
)




class SafeScoreboardProvider(ScoreboardProvider):
    def set_state(self, agents, state, shared_info):
        try:
            return super().set_state(agents, state, shared_info)
        except ValueError:
            return shared_info

    def step(self, agents, state, shared_info):
        try:
            result = super().step(agents, state, shared_info)
        except (ValueError, TypeError):
            result = shared_info
        return result





class RLViserRenderer(Renderer[GameState]):
    def __init__(self, tick_rate: float = 120/8):
        rlviser.set_boost_pad_locations(BOOST_LOCATIONS)
        self.tick_rate = tick_rate
        self.packet_id = 0

    def render(self, state, shared_info):
        boost_states = [t == 0 for t in state.boost_pad_timers]
        ball = rsim.BallState()
        ball.pos = rsim.Vec(*state.ball.position)
        ball.vel = rsim.Vec(*state.ball.linear_velocity)
        ball.ang_vel = rsim.Vec(*state.ball.angular_velocity)
        ball.rot_mat = rsim.RotMat(*state.ball.rotation_mtx.T.flatten())

        cars = []
        for i, car in enumerate(state.cars.values(), start=1):
            cs = rsim.CarState()
            phy = car.physics
            cs.pos = rsim.Vec(*phy.position)
            cs.vel = rsim.Vec(*phy.linear_velocity)
            cs.ang_vel = rsim.Vec(*phy.angular_velocity)
            cs.rot_mat = rsim.RotMat(*phy.rotation_mtx.T.flatten())
            cs.boost = car.boost_amount
            cars.append((i, car.team_num, rsim.CarConfig(car.hitbox_type), cs))

        self.packet_id += 1
        rlviser.render(
            tick_count=self.packet_id,
            tick_rate=self.tick_rate,
            game_mode=rsim.GameMode.SOCCAR,
            boost_pad_states=boost_states,
            ball=ball,
            cars=cars
        )

    def close(self):
        rlviser.quit()

        
class FlightTowardsBallReward(RewardFunction[AgentID, GameState, float]):

    def __init__(
        self,
        height_thresh: float = 300.0,
        min_air_ticks: int = 5
    ):
        super().__init__()
        self.height_thresh = height_thresh
        self.min_air_ticks = min_air_ticks
        self.air_counters: Dict[AgentID, int] = {}

    def reset(self, agents, initial_state, shared_info):
        self.air_counters = {a: 0 for a in agents}

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated,
        is_truncated,
        shared_info
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        ball_pos = state.ball.position

        for a in agents:
            car = state.cars[a]
            phy = car.physics
            z = phy.position[2]

            if not car.on_ground:
                self.air_counters[a] = self.air_counters.get(a, 0) + 1
            else:
                self.air_counters[a] = 0

            if self.air_counters[a] >= self.min_air_ticks and z > self.height_thresh:
                diff = ball_pos - phy.position
                dist = np.linalg.norm(diff)
                if dist > 0:
                    dir_to_ball = diff / dist
                    speed_proj = float(np.dot(phy.linear_velocity, dir_to_ball))
                    rewards[a] = max(0.0, speed_proj / CAR_MAX_SPEED)
                else:
                    rewards[a] = 0.0
            else:
                rewards[a] = 0.0

        return rewards





class BallGoalImpulseReward(RewardFunction[AgentID, GameState, float]):

    def __init__(self):
        super().__init__()
        self.prev_vel: np.ndarray = None
        self._touch = TouchReward()

    def reset(self, agents, initial_state, shared_info):
 
        self.prev_vel = initial_state.ball.linear_velocity.copy()
        self._touch.reset(agents, initial_state, shared_info)

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated,
        is_truncated,
        shared_info
    ) -> Dict[AgentID, float]:
        curr_vel = state.ball.linear_velocity
        delta_v = curr_vel - self.prev_vel

        touches = self._touch.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        rewards: Dict[AgentID, float] = {a: 0.0 for a in agents}

        for a, touched in touches.items():
            if touched > 0:
                goal_y = -BACK_NET_Y if state.cars[a].is_orange else BACK_NET_Y
                dir_goal = np.array([0.0, goal_y, 0.0]) - state.ball.position
                norm = np.linalg.norm(dir_goal)
                if norm > 0:
                    dir_goal /= norm

                impulse = float(np.dot(delta_v, dir_goal))
                rewards[a] = impulse / BALL_MAX_SPEED

        self.prev_vel = curr_vel.copy()
        return rewards


class AerialTouchReward(RewardFunction[AgentID, GameState, float]):

    def __init__(self, height_threshold: float = 600.0):
        super().__init__()
        self.height_threshold = height_threshold
        self._touch = TouchReward()

    def reset(self, agents, initial_state, shared_info):
        self._touch.reset(agents, initial_state, shared_info)

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated,
        is_truncated,
        shared_info
    ) -> Dict[AgentID, float]:

        base = self._touch.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            z = car.physics.position[2]
            if not car.on_ground and z >= self.height_threshold and base.get(a, 0.0) > 0.0:
                rewards[a] = float(base[a])
            else:
                rewards[a] = 0.0
        return rewards



class FirstArrivalReward(RewardFunction[AgentID, GameState, float]):

    def __init__(self, first_reward: float = 3.0, second_penalty: float = -1.0):
        super().__init__()
        self._touch = TouchReward()
        self.first: AgentID = None
        self.second: AgentID = None
        self.first_reward = first_reward
        self.second_penalty = second_penalty

    def reset(self, agents, initial_state, shared_info):
        self._touch.reset(agents, initial_state, shared_info)
        self.first = None
        self.second = None

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated, is_truncated, shared_info) -> Dict[AgentID, float]:
        base = self._touch.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        rewards: Dict[AgentID, float] = {a: 0.0 for a in agents}
        for a in agents:
            if base.get(a, 0.0) > 0:
                if self.first is None:
                    self.first = a
                    rewards[a] = float(self.first_reward)
                elif self.second is None and a != self.first:
                    self.second = a
                    rewards[a] = float(self.second_penalty)
        return rewards


class RoofContactPenalty(RewardFunction[AgentID, GameState, float]):

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, *args) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            phy = state.cars[a].physics
            up_vec = phy.rotation_mtx[:, 2]
            rewards[a] = 1.0 if (up_vec[2] < 0 and state.cars[a].on_ground) else 0.0
        return rewards


class ConcedeGoalReward(RewardFunction[AgentID, GameState, float]):

    def reset(self, agents, initial_state, shared_info):
        sb = shared_info.get("scoreboard")
        if sb:
            self.prev_blue = sb.blue_score
            self.prev_orange = sb.orange_score
        else:
            self.prev_blue = 0
            self.prev_orange = 0

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated, is_truncated, shared_info) -> Dict[AgentID, float]:
        sb = shared_info.get("scoreboard")
        if sb:
            curr_blue = sb.blue_score
            curr_orange = sb.orange_score
        else:
            curr_blue = self.prev_blue
            curr_orange = self.prev_orange
        d_blue = curr_blue - self.prev_blue
        d_orange = curr_orange - self.prev_orange
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            if (car.is_orange and d_blue > 0) or (not car.is_orange and d_orange > 0):
                rewards[a] = 1.0
            else:
                rewards[a] = 0.0
        self.prev_blue = curr_blue
        self.prev_orange = curr_orange
        return rewards

class IdlePenalty(RewardFunction[AgentID, GameState, float]):

    def __init__(self, threshold_speed: float = 200.0, idle_ticks_threshold: int = 120):

        super().__init__()
        self.threshold = threshold_speed
        self.idle_ticks_threshold = idle_ticks_threshold
        self.counter: Dict[AgentID, int] = {}

    def reset(self, agents, initial_state, shared_info):
        self.counter = {a: 0 for a in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState, *args) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            phy = state.cars[a].physics
            speed = np.linalg.norm(phy.linear_velocity)
            if speed < self.threshold:
                self.counter[a] = self.counter.get(a, 0) + 1
            else:
                self.counter[a] = 0
            rewards[a] = 1.0 if self.counter[a] > self.idle_ticks_threshold else 0.0
        return rewards


class BackwardDrivingPenalty(RewardFunction[AgentID, GameState, float]):

    def __init__(
        self,
        threshold_seconds: float = 2.0,
        tick_rate: float = 15.0  
    ):
        super().__init__()
        self.threshold_ticks = int(threshold_seconds * tick_rate)
        self.counter: Dict[AgentID, int] = {}

    def reset(self, agents, initial_state, shared_info):
        for a in agents:
            self.counter[a] = 0

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated,
        is_truncated,
        shared_info
    ) -> Dict[AgentID, float]:
        rewards = {}
        for a in agents:
            car = state.cars[a]
            v = car.physics.linear_velocity
            forward = car.physics.rotation_mtx[:, 0]
            proj = float(np.dot(v, forward))
            if proj < 0:
                self.counter[a] = self.counter.get(a, 0) + 1
            else:
                self.counter[a] = 0

            rewards[a] = 1.0 if self.counter[a] > self.threshold_ticks else 0.0
        return rewards



class OffensivePositionReward(RewardFunction[AgentID, GameState, float]):

    def __init__(self, scale: float = 3000.0, weight_along: float = 0.3):
        super().__init__()
        self.scale = scale
        self.weight_along = weight_along

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated, is_truncated, shared_info) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        ball_pos = state.ball.position
        for a in agents:
            car = state.cars[a]

            goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
            goal_pos = np.array([0.0, goal_y, 0.0])

            if (car.is_orange and ball_pos[1] < 0) or (not car.is_orange and ball_pos[1] > 0):

                vec_ball_goal = goal_pos - ball_pos
                target_pos = ball_pos + self.weight_along * vec_ball_goal
                dist = np.linalg.norm(state.cars[a].physics.position - target_pos)
                val = 1.0 - dist / self.scale
                rewards[a] = float(max(0.0, val))
            else:
                rewards[a] = 0.0
        return rewards

class DefensivePositionReward(RewardFunction[AgentID, GameState, float]):

    def __init__(self, scale: float = 3000.0, weight_along: float = 0.3):

        super().__init__()
        self.scale = scale
        self.weight_along = weight_along

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated, is_truncated, shared_info) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        ball_pos = state.ball.position
        for a in agents:
            car = state.cars[a]
            
            goal_y = BACK_NET_Y if car.is_orange else -BACK_NET_Y
            goal_pos = np.array([0.0, goal_y, 0.0])

            if (car.is_orange and ball_pos[1] > 0) or (not car.is_orange and ball_pos[1] < 0):

                vec_ball_goal = goal_pos - ball_pos
                target_pos = ball_pos + self.weight_along * vec_ball_goal
                dist = np.linalg.norm(state.cars[a].physics.position - target_pos)
                val = 1.0 - dist / self.scale
                rewards[a] = float(max(0.0, val))
            else:
                rewards[a] = 0.0
        return rewards

class LowBoostPenalty(RewardFunction[AgentID, GameState, float]):

    def __init__(self, threshold: float = 20.0, ball_dist_thresh: float = 3000.0):
        super().__init__()
        self.threshold = threshold
        self.ball_dist_thresh = ball_dist_thresh

    def reset(self, agents, initial_state, shared_info): 
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, *args) -> Dict[AgentID, float]:
        rewards = {}
        ball_pos = state.ball.position
        for a in agents:
            car = state.cars[a]
            dist = np.linalg.norm(ball_pos - car.physics.position)
            if dist > self.ball_dist_thresh and car.boost_amount < self.threshold:
                rewards[a] = 1.0
            else:
                rewards[a] = 0.0
        return rewards


class InterceptReward(RewardFunction[AgentID, GameState, float]):

    def reset(self, agents, initial_state, shared_info):
        pass 

    def predict_intercept(self, state: GameState) -> Any:

        pos = state.ball.position
        vel = state.ball.linear_velocity
        if abs(vel[1]) < 1e-3:
            return pos.copy(), float('inf')

        t = -pos[1] / vel[1]
        intercept = pos + vel * t
        return intercept, t

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated,
        is_truncated,
        shared_info
    ) -> Dict[AgentID, float]:
        intercept_pt, intercept_t = self.predict_intercept(state)
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car_pos = state.cars[a].physics.position
            dist = np.linalg.norm(car_pos - intercept_pt)

            if 0 < intercept_t < 3.0:
    
                rewards[a] = max(0.0, 1.0 - dist / 3000.0) * 0.5
            else:
                rewards[a] = 0.0
        return rewards


class OffBallPenalty(RewardFunction[AgentID, GameState, float]):

    def __init__(self,
                 dist_thresh: float = 4000.0,
                 idle_ticks_threshold: int = 120):

        super().__init__()
        self.dist_thresh = dist_thresh
        self.idle_ticks_threshold = idle_ticks_threshold
        self.counter: Dict[AgentID, int] = {}

    def reset(self,
              agents: List[AgentID],
              initial_state: GameState,
              shared_info: Dict[str, Any]):

        self.counter = {a: 0 for a in agents}

    def get_rewards(self,
                    agents: List[AgentID],
                    state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        ball_pos = state.ball.position
        for a in agents:
            car_pos = state.cars[a].physics.position
            dist = np.linalg.norm(ball_pos - car_pos)
            if dist > self.dist_thresh:
                self.counter[a] = self.counter.get(a, 0) + 1
            else:
                self.counter[a] = 0


            rewards[a] = 1.0 if self.counter[a] > self.idle_ticks_threshold else 0.0

        return rewards



class HighGoalReward(RewardFunction[AgentID, GameState, float]):

    def __init__(self, height_threshold: float = 320.0, reward_amount: float = 100.0):
        super().__init__()
        self.height_threshold = height_threshold
        self.reward_amount = reward_amount
        self.prev_blue = 0
        self.prev_orange = 0

    def reset(self, agents, initial_state, shared_info):
        sb = shared_info.get("scoreboard")
        if sb:
            self.prev_blue = sb.blue_score
            self.prev_orange = sb.orange_score

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated, is_truncated,
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        sb = shared_info.get("scoreboard")
        curr_blue = sb.blue_score if sb else self.prev_blue
        curr_orange = sb.orange_score if sb else self.prev_orange
        d_blue = curr_blue - self.prev_blue
        d_orange = curr_orange - self.prev_orange

        ball_z = state.ball.position[2]
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car = state.cars[a]
            if ((not car.is_orange and d_blue > 0) or (car.is_orange and d_orange > 0)) \
               and ball_z >= self.height_threshold:
                rewards[a] = float(self.reward_amount)
            else:
                rewards[a] = 0.0

        self.prev_blue = curr_blue
        self.prev_orange = curr_orange
        return rewards



class SingleLowBoostPenalty(RewardFunction[AgentID, GameState, float]):

    def __init__(self, threshold: float = 12.0, penalty: float = -1.0):
        super().__init__()
        self.threshold = threshold
        self.penalty_amount = penalty
        self._above_threshold: Dict[AgentID, bool] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]):

        self._above_threshold = {a: True for a in agents}

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            boost = state.cars[a].boost_amount

            if self._above_threshold.get(a, True) and boost < self.threshold:
                rewards[a] = self.penalty_amount
                self._above_threshold[a] = False
            else:
                rewards[a] = 0.0

                if not self._above_threshold.get(a, True) and boost >= self.threshold:
                    self._above_threshold[a] = True
        return rewards


class BoostSqrtReward(RewardFunction[AgentID, GameState, float]):

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]):
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            b = state.cars[a].boost_amount

            rewards[a] = float(np.sqrt(b / 100.0))
        return rewards



class SupersonicReward(RewardFunction[AgentID, GameState, float]):

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]):

        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            vel = state.cars[a].physics.linear_velocity
            speed = float(np.linalg.norm(vel))

            rewards[a] = 1.0 if speed >= CAR_MAX_SPEED else 0.0
        return rewards



class RandomStartMutator(StateMutator[GameState]):

    def __init__(self, kickoff_prob: float = 0.3, margin: float = 500.0):
        self.kickoff = KickoffMutator()
        self.kickoff_prob = kickoff_prob
        self.margin = margin

    def apply(self, state: GameState, shared_info: Dict[int, Any]) -> None:
        if random.random() < self.kickoff_prob:
            self.kickoff.apply(state, shared_info)
            return

        # --- Randomize ball ---
        state.ball.position = np.array([
            random.uniform(-SIDE_WALL_X, SIDE_WALL_X),
            random.uniform(-BACK_NET_Y, BACK_NET_Y),
            random.uniform(100, 1000)
        ], dtype=np.float32)

        dir2center = -state.ball.position[:2]
        dir2center /= (np.linalg.norm(dir2center) + 1e-6)
        speed = random.uniform(500, 2000)
        state.ball.linear_velocity = np.array([
            dir2center[0] * speed,
            dir2center[1] * speed,
            random.uniform(-500, 500)
        ], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

       
        car_radius = 118.0  
        min_x, max_x = -SIDE_WALL_X + car_radius + self.margin, SIDE_WALL_X - car_radius - self.margin
        min_y, max_y = -BACK_NET_Y + car_radius + self.margin, BACK_NET_Y - car_radius - self.margin

    
        for car in state.cars.values():

            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            if car.is_orange:
                y = abs(y)
            else:
                y = -abs(y)

            x = max(min_x, min(max_x, x))
            y = max(min_y, min(max_y, y))
            car.physics.position = np.array([x, y, random.uniform(17,1200)], dtype=np.float32)


            to_ball = state.ball.position[:2] - np.array([x, y], dtype=np.float32)
            yaw = np.arctan2(to_ball[1], to_ball[0])
            c, s = np.cos(yaw), np.sin(yaw)
            car.physics.rotation_mtx = np.array([
                [ c, -s, 0],
                [ s,  c, 0],
                [ 0,  0, 1]
            ], dtype=np.float32)


            dir3 = np.random.normal(size=3).astype(np.float32)
            dir3[2] = 0
            dir3 /= (np.linalg.norm(dir3) + 1e-6)
            car.physics.linear_velocity = dir3 * random.uniform(0, CAR_MAX_SPEED)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.boost_amount = random.uniform(33.0, 100.0)



class BoostPickupReward(RewardFunction[AgentID, GameState, float]):

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self._last_boost: Dict[AgentID, float] = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]):

        self._last_boost = {a: initial_state.cars[a].boost_amount for a in agents}

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any]
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        for a in agents:
            car: Car = state.cars[a]
            boost_now = car.boost_amount
            boost_prev = self._last_boost.get(a, boost_now)
            delta = max(boost_now - boost_prev, 0.0)
            rewards[a] = (delta / 100) * self.weight
            self._last_boost[a] = boost_now
        return rewards

def build_rlgym_v2_env() -> RLGymV2GymWrapper:

    action_parser = RepeatAction(LookupTableAction(), repeats=8)
    term_cond = GoalCondition()
    trunc_cond = AnyCondition(NoTouchTimeoutCondition(60), TimeoutCondition(300))

    reward_fn = CombinedReward(
        
        (BackwardDrivingPenalty(threshold_seconds=2.0, tick_rate=15.0), -2),
        (RoofContactPenalty(), -2),
        (IdlePenalty(200.0, 120), -3),
        (SingleLowBoostPenalty(threshold=12.0, penalty=-1.0), 15),
        (LowBoostPenalty(threshold=35.0, ball_dist_thresh=2200.0), -2),
        (ConcedeGoalReward(), -210.0),
        (OffBallPenalty(dist_thresh=4500.0, idle_ticks_threshold=120), -1.0),


        
        
        (DefensivePositionReward(3000.0, 0.1), 0.0051),
        (OffensivePositionReward(3000.0, 0.1), 0.0051),
        
        (HighGoalReward(height_threshold=320.0, reward_amount=150.0), 1),

        
        (FirstArrivalReward(4.0, -2.5), 8.0),
        (GoalReward(), 450.0),
        (BoostSqrtReward(), 0.6),
        (SupersonicReward(), 0.5),
        (TouchReward(), 1.0), 
        (BallGoalImpulseReward(), 30.0),
        (BoostPickupReward(weight=5.0), 2.0),


        (AerialTouchReward(height_threshold=270.0), 25.0),
        (FlightTowardsBallReward(height_thresh=250, min_air_ticks=100), 10.0),
    
        
    )

    obs = DefaultObs(
        zero_padding=None,
        pos_coef=np.array([1/SIDE_WALL_X, 1/BACK_NET_Y, 1/CEILING_Z]),
        ang_coef=1/np.pi,
        lin_vel_coef=1/CAR_MAX_SPEED,
        ang_vel_coef=1/CAR_MAX_ANG_VEL,
        boost_coef=1/100
    )

    mutator = MutatorSequence(
        FixedTeamSizeMutator(1, 1),
        RandomStartMutator(kickoff_prob=0.5)
    )
    base_env = RLGym(
        state_mutator=mutator,
        obs_builder=obs,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=term_cond,
        truncation_cond=trunc_cond,
        transition_engine=RocketSimEngine(),
        shared_info_provider=SafeScoreboardProvider(),
        renderer=RLViserRenderer()
    )
    return RLGymV2GymWrapper(base_env)


if __name__ == "__main__":


    n_proc = 48 
    min_inference_size = max(1, int(round(n_proc * 0.9)))


    checkpoint_root = "salvataggi"
    checkpoint_load_folder = None  

    if os.path.isdir(checkpoint_root):
        subdirs = [
            d for d in os.listdir(checkpoint_root)
            if os.path.isdir(os.path.join(checkpoint_root, d)) and d.isdigit()
        ]
        if subdirs:
            latest_num = max(int(d) for d in subdirs)
            latest = str(latest_num)
            checkpoint_load_folder = os.path.join(checkpoint_root, latest)
            print(f"✔️ Caricherò il checkpoint da: {checkpoint_load_folder}")
        else:
            print("ℹ️ Nessun checkpoint trovato, partirò da zero.")
    else:
        print("ℹ️ La cartella checkpoint non esiste, partirò da zero.")


    
    learner = Learner(
        build_rlgym_v2_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        render=True,
        render_delay=0.03,
        metrics_logger=None,
        device="cuda",
        ppo_batch_size=200_000,
        policy_layer_sizes=[2048, 2048, 1024, 1024],
        critic_layer_sizes=[2048, 2048, 1024, 1024],
        ts_per_iteration=200_000,
        exp_buffer_size=500_000,
        ppo_minibatch_size=20_000,
        ppo_ent_coef=0.006,
        policy_lr=2.1e-4,
        critic_lr=1e-4,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=10_000_000,
        timestep_limit=100_000_000_000,
        checkpoints_save_folder=checkpoint_root,
        checkpoint_load_folder=checkpoint_load_folder,

        add_unix_timestamp=False,
        log_to_wandb=False
    )
    learner.learn()
