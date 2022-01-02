from collections import deque
import random
import numpy as np
from stock.myModule.auto_trade_model import mlp



class DQNAgent(object):
  """ A simple Deep Q agent """

  # 초기화 과정 : 인스턴스시 자동 생성(초기 변수)
  def __init__(self, state_size, action_size):

    self.input_data_column_cnt = len(state_size)      #(3,2) => 3
    self.action_size = action_size                    # 3

    self.memory = deque(maxlen=2000)          # deque : list 보다 메모리 저장 속도가 2배 빠름, stack 과 que 기능을 모두 지원하는모듈(앞과 뒤에서 출입 가능)

    self.gamma = 0.95                         # discount rate
    self.epsilon = 1.0                        # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995

    self.model = mlp(state_size, action_size)      # mlp((3,2),  3)

  # state(1,3)  :[ [stock_owned, stock_price, cash_in hand  ]  에서 최적의 행동 추출(인덱스 0, 1, 2) 산출
  def act(self, state):

    # 초기에는 데이터가 없어 신뢰성이 떨어져 랜덤하게 action 선택
    if np.random.rand() <= self.epsilon:            # np.random.rand(0~1사이) <= 1, 0.995

      return random.randrange(self.action_size)     # random.randrange(start, stop[, step])    randrange(3) : 0~3

    ###########################
    # 자료가 축적된 후반부 진행과정(매번 epsilon 0.05씩 감소 )에서 나타남   np.random.rand) > self.epsilon:
    # state에서 최적의 행동 추출(인덱스) : # state(1, 3)에 따라 3개 action별 최고의 값 산출(1, 3)

    act_values = self.model.predict(state)    # (1,) <=  state(1,3)  : state(1,3)[ [stock_owned, stock_price, cash_in hand]  ]을 통해 [0:sell, 1:hold, 2:buy] 각 인덱스의 최고의 value 값

    fited_action = np.argmax(act_values[0])  # (1) 0: sell, 1: hold, 2:buy => 가장 유사한 (높은) 값의 의 인덱스(action) 리턴

    return fited_action


  #### target(가중치)를 통해 y_train을 구하고 학습하는 과정
  # for문으로 처리하기에는 너무 속도가 느려 배열로 처리하기 위함(for문 보다 배열 처리가 훨씬 빠르다)
  # 에피소드(100회) 처리중 memory 저장 횟수가 32회(에피소드 32회 동일) batch를 초과하면 처리 속도를 높이기 위해 batch_size(32)로 처리

  def replay(self, batch_size=32):

    """ vectorized implementation; 30x speed up compared with for loop """

    # memory에서  32개의 랜덤 샘플을 뽑아 내라   # sample (memory, 32 )
    minibatch = random.sample(self.memory, batch_size)        # (32,5)   memory : [ state(1,3), action(1), reward(1), next_state(1,3), done(1: T or False)] 의 32개(minibatch)

    states = np.array([tup[0][0] for tup in minibatch])       # (32,3)     [stock_owned, stock_price, cash_in hand]
    actions = np.array([tup[1] for tup in minibatch])         # (32,1)
    rewards = np.array([tup[2] for tup in minibatch])         # (32,1)
    next_states = np.array([tup[3][0] for tup in minibatch])   #(32,3)
    done = np.array([tup[4] for tup in minibatch])            #(32,1)  True or false

    ### target(가중치)를 통해 학습 레벨 값 (y_train)  을 구하는 과정
    #  next_state에서 최고의 추출 : # next_states(32,3)에 따라  3개 action별 최고의 값 산출(1, 3)

    # Q(s', a)
    next_predict = self.model.predict(next_states)            # (32, 3) <=  next_states(32, 3)에 따라,  3개 action (0,1,2)각각의 최고의 값 산출
    ref = np.max(next_predict, axis=1)                       # (32,)   매일 3개 action(0,1,2)별 최고의 값 산출

   # reward 2D(1259,1) to 1D(1259,)로 전환
    rewards = np.reshape(rewards, (np.product(rewards.shape),))    # (32,1 ) => (32,):ref(최고값)에 가중치 더함 => target) : 최대값에 discount rate 곱하고 rewards 더함(32,1)
    target = rewards + self.gamma * ref                            # (32,)

    #유일한 차이  target = rewards + self.gamma * ref                  # (32,)  ref(최고값)에 가중치 더함 => target) : 최대값에 discount rate 곱하고 rewards 더함(32,1)

    # end state target is reward itself (no lookahead)
    target[done] = rewards[done]

    # Q(s, a)
    predict = self.model.predict(states)     # (32, 3) : states(32, 3)에 따라  3개 action (0,1,2)각각의 최고의 값 산출

    #### 변경 target을 actions 컬럼에 치환 : make the agent to approximately map the current state to future discounted reward
    # state에서의 predict 값에서, 가중치(target)의 인덱스에 해당하는 action(3개중)의 값을 찾아서 가중치(target)값으로 치환
    # make the agent to approximately map the current state to future discounted reward
    # predict[range(32), actions[]]

    predict[range(batch_size), actions] = target     # (32,3)  predict (32,3)에서 target(32,)의 값의 해당 actions의 인데스위치에 target 값으로 치환

    #########  학습 :   Fit

    self.model.fit(states, predict, epochs=1, verbose=0)    # states(32,3), 조정된 target값으로 치환된 predict(32,3)로 학습 <= model.fit(x_train, y_train..) 동일

    # epsilon 가변치(무한 전달 가변치 0에 도달하지 못함) 설정
    if self.epsilon > self.epsilon_min:    # 1> 0.01  0.05
      self.epsilon *= self.epsilon_decay     # eps = eps*0.995

  # action 이후 데이터 저장 (state, action, reward, next_state, done)  (1,5)
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)