from collections import deque
import random
import numpy as np
from deepStock.myModule.auto_trade_model import mlp



class DQNAgent_multi(object):
  """ A simple Deep Q agent """

  # 초기화 과정 : 인스턴스시 자동 생성(초기 변수)
  def __init__(self, state_size, action_size):

    self.state_size = state_size
    self.action_size = action_size

    self.memory = deque(maxlen=2000)    # deque : list 보다 메모리 저장 속도가 2배 빠름, stack 과 que 기능을 모두 지원하는모듈(앞과 뒤에서 출입 가능)
    self.gamma = 0.95                         # discount rate
    self.epsilon = 1.0                        # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995

    # 입력((7,2), 27)
    self.model = mlp(state_size, action_size)

  # state에서 최적의 행동 추출(인덱스) 전달
  def act(self, state):

    # 초기에는 데이터가 없어 신뢰성이 떨어져 랜덤하게 action 선택
    if np.random.rand() <= self.epsilon:            # np.random.rand(0~1사이) <= 1, 0.995

      return random.randrange(self.action_size)     # random.randrange(start, stop[, step])    randrange(27) : 0~27

    ###########################
    # 자료가 축적된 후반부 진행과정(매번 epsilon 0.05씩 감소 )에서 나타남   np.random.rand) > self.epsilon:
    # state에서 최적의 행동 추출(인덱스) : # state(1, 3)에 따라 3개 action별 최고의 값 산출(1, 3)

    # 후반부 진행과정(매번 epsilon 0.05씩 감소 )에서 나타남   np.random.rand) > self.epsilon:
    # state에서 최적의 행동 추출(인덱스) : # state(3,2)에 따라 3개 action별 최고의 값 산출(1, 3)

    act_values = self.model.predict(state)

    fited_action = np.argmax(act_values[0])         # 가장 높은 값의 의 인덱스(action) 리턴

    return fited_action


  #### target(가중치)를 통해 y_train을 구하고 학습하는 과정
  # for문으로 처리하기에는 너무 속도가 느려 배열로 처리하기 위함(for문 보다 배열 처리가 훨씬 빠르다)
  # 에피소드(100회) 처리중 memory 저장 횟수가 32회(에피소드 32회 동일) batch를 초과하면 처리 속도를 높이기 위해 batch_size(32)로 처리

  def replay(self, batch_size=32):

    """ vectorized implementation; 30x speed up compared with for loop """

    # memory에서  32개의 랜덤 샘플을 뽑아 내라   # sample (memory, 32 )
    minibatch = random.sample(self.memory, batch_size)    # memory : [ state(7,2), action(1), reward(1), next_state(7,2), done(1: T or False)] 의 32개(minibatch)

    # print("minibatch:", minibatch)     # (32, memory )

    states = np.array([tup[0][0] for tup in minibatch])       # (32,7)
    actions = np.array([tup[1] for tup in minibatch])         # (32,1)
    rewards = np.array([tup[2] for tup in minibatch])         # (32,1)
    next_states = np.array([tup[3][0] for tup in minibatch])   #(32,7)
    done = np.array([tup[4] for tup in minibatch])            #(32,1)  True or false

    # print("rewards-shape:", rewards.shape)

    ### target(가중치)를 통해 (y_train)  레벨 값 을 구하는 과정
    ## next_state에서 최고의 추출 : # next_states(32,7)에 따라 27개 action별 최고의 값 산출(1, 27)

    # Q(s', a)
    next_predict = self.model.predict(next_states)    # (32, 27) next_states(32, 7)에 따라 27개 action별 최고의 값 산출=>
    ref = np.amax(next_predict, axis=1)              # (32,)  ref1 각 행 값의 최대값 산출    : 각 베치(0~32)에서 최고의 값 선택,  action = np.argmax(predict_0[0])

    # rewards => target
    target = rewards + self.gamma * ref           # (32,)  ref(최고값)에 가중치 더함 => target) : 최대값에 discount rate 곱하고 rewards 더함(32,1)
    # print("ref-shape:", ref, "target_shape:", target.shape, "target:", target)

    # end state target is reward itself (no lookahead)
    target[done] = rewards[done]
    # print("target-done:", target)

    # Q(s, a)
    predict = self.model.predict(states)     # (32, 27) : states(32, 7)에 따라 27개 action별 최고의 값 산출

    #### 변경 target을 actions 컬럼에 치환 : make the agent to approximately map the current state to future discounted reward
    # state에서의 predict 값에서, 가중치(target)의 인덱스에 해당하는 action(3개중)의 값을 찾아서 가중치(target)값으로 치환
    # make the agent to approximately map the current state to future discounted reward
    # predict[range(32), actions[]]
    # 변경 target을 actions 컬럼에 치환 : make the agent to approximately map the current state to future discounted reward
    # state에서의 predict 값에서, 가중치(target)의 인덱스에 해당하는 action(0~27중)의 값을 찾아서 가중치(target)값으로 치환

    predict[range(batch_size), actions] = target

    #########  학습 : Fit
    # states(32,3), 조정된 target값으로 치환된 predict(32,3)로 학습 <= model.fit(x_train, y_train..) 동일

    self.model.fit(states, predict, epochs=1, verbose=0)

    # epsilon 가변치(무한 전달 가변치 0에 도달하지 못함) 설정
    if self.epsilon > self.epsilon_min:    # 1> 0.01  0.05
      self.epsilon *= self.epsilon_decay     # eps = eps*0.995

  # action 이후 데이터 저장 (state, action, reward, next_state, done)
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)