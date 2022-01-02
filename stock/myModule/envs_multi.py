import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class TradingEnv_multi(gym.Env):
  """
  multi -stock  trading environment.

  State: [# of stock owned, current stock prices, cash in hand]
    - array of length n_stock * 2 + 1
    - price is discretized (to integer) to reduce state space
    - use close price for each stock
    - cash in hand is evaluated at each step based on action performed
  Action: sell (0), hold (1), and buy (2)
    - when selling, sell all the shares
    - when buying, buy as many as cash in hand allows
    - if buying multiple stock, equally distribute cash in hand and then utilize the balance
  """

  # 인스턴스시 자동 생성(초기 변수) : 초기화 과정 :
  def __init__(self, train_data, init_invest):

    ##########    인스턴스 속성(instance attributes)
    # n_stock : 주식 종목수(3), n_step : 주식 데이터 수(3516 : 주식 거래일)

    self.stock_price_history = np.around(train_data)                # (3,3526) round up to integer to reduce state space

    self.n_stock, self.n_step = self.stock_price_history.shape      # (3, 3526)

    self.init_invest = init_invest
    self.cur_step = None       # 현재 단계
    self.stock_owned = None    # 보유한 해당 주식수
    self.stock_price = None    # 해당 주식 가격
    self.cash_in_hand = None   # 보유 현금

    ##### Action 공간(주식 매매 행동(매수/매도/보유) 공간)  : gym.space.Discrete 모듈 활용

    self.action_space = spaces.Discrete(3**self.n_stock)        # 주식종목수 ** (매수/매도/보유) => 27가지

    ##### Observation 공간(관찰 공간): give estimates in order to sample and build scaler

    stock_max_price = self.stock_price_history.max(axis=1)                  # (3,1) :  axis=1(y축 열 공간에서 해당 종목 최고 가격
    # stock_max_price = stock_price.max()

    stock_range = [[0, init_invest * 2 // mx] for mx in stock_max_price]    # (3,2) : 주식보유 수량 범위 [0 ~ 초기투자금*2 //종목 최고가]   [[0, ?],[0.?],[0, ?]]
    # stock_range = [0, init_invest * 2 // stock_max_price]

    price_range = [[0, mx] for mx in stock_max_price]                     # (3,2) 주가 범위 [0 ~ 최고가]   [[0, ?],[0.?],[0, ?]]
    # price_range = [0, stock_max_price]

    cash_in_hand_range = [[0, init_invest * 2]]                          # (1,2) 보유 현금 범위 [ 0 ~ 초기투자금 *2 ] (1,2) [[0, 40000]]

    self.observation_space = spaces.MultiDiscrete(stock_range + price_range + cash_in_hand_range)    # (3,2) + (3,2) + (1,2) = (7,2) multiDiscrete [ 0, 341] [] [] [] [] [] [0, 400000] ]

    # seed and start
    self._seed()
    self._reset()


  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  # 초기화
  def _reset(self):

    self.cur_step = 0
    self.stock_owned = [0] * self.n_stock                           # (3,1)   [[0][0][0]]
    # self.stock_owned = 0

    self.stock_price = self.stock_price_history[:, self.cur_step]   # (3,1) cur_step 거래일(다음날) 따른 주식 가격 (3,1)
    self.cash_in_hand = self.init_invest                            # 20000

    state = self._get_state()                                          # (7,2) 배열 생성 리턴 self.stock_owned * self.stock_price) + self.cash_in_hand

    return state

  # action에따라 다음 프로세스(매매 거래)에 따라 next_state, reward 산출
  def _step(self, action):

    assert self.action_space.contains(action)

    # state에 따른 value 산정(주식수*주식가격+보유현금)
    prev_val = self._get_val()

    self.cur_step += 1

    self.stock_price = self.stock_price_history[:, self.cur_step]        # update price 다음날 주식 가격
    # self.stock_price = self.stock_price_history[self.cur_step]  # update price 다음날 주식 가격

    # 주식 거래 처리(action에 따라) => 보유 주식과 보유 현금 state 가 변함 => next_state
    self._trade(action)

    cur_val = self._get_val()

    reward = cur_val - prev_val

    done = self.cur_step == self.n_step - 1

    info = {'cur_val': cur_val}

    next_state = self._get_state()

    return next_state, reward, done, info

  # state 생성
  def _get_state(self):

    obs = []

    obs.extend(self.stock_owned)
    obs.extend(list(self.stock_price))
    obs.append(self.cash_in_hand)

    return obs

  # state에 따른 value 산정(주식수*주식가격+보유현금)
  def _get_val(self):

    value = np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand
    return value

  # 주식 거래 처리(action에 따라) => 보유 주식과 보유 현금 state 가 변함 => next_state
  def _trade(self, action):

    # all combo to sell(0), hold(1), or buy(2) stocks
    # lst = list(map(list, itertools.product([0, 1], repeat=n))

    # action:0 => [0,0,0] action:8 => [0,2,2] action:19 => [2,0,1] action:26 => [2,2,2]
    action_combo = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    action_vec = action_combo[action]      # [0,2,1] => 종목1 매도, 종목2 매수, 종목3 보유
    # print("action:", action)
    # print("action_vec:", action_vec)

    #### one pass to get sell/buy index
    sell_index = []
    buy_index = []

    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)

    #### two passes: sell first, then buy; might be naive in real-world settings
    if sell_index:
      for i in sell_index:
        self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.stock_owned[i] = 0

    if buy_index:
      can_buy = True
      while can_buy:
        for i in buy_index:
          if self.cash_in_hand > self.stock_price[i]:
            self.stock_owned[i] += 1                      # buy one share
            self.cash_in_hand -= self.stock_price[i]
          else:
            can_buy = False