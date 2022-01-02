from django.shortcuts import render, redirect
from django.http import HttpResponse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

import os
import json
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from stock.myModule.envs import TradingEnv
from stock.myModule.envs_multi import TradingEnv_multi
from stock.myModule.agent import DQNAgent
from stock.myModule.agent_multi import DQNAgent_multi

from datetime import datetime, date

import time

from tensorflow.keras.models import load_model

# selenium을 통해 웹브라우저 접근시
from selenium import webdriver
import requests

# BeautifulSoup4 운용
from bs4 import BeautifulSoup

# 파일 복사
import shutil

# stock 리스트(시작)
def stockhome(request):

    return render(request, 'stock/stockhome.html')


#########################################  단일 종목 주가 예측 ## ###################################################
##### 단일 종목에 대한 주가 예측모델(LSTM) + 매매추천 : Q-Learning(강화학습),  Standardizati  #######################
#####################################################################################################################

def stockmodel(request):

    if request.method == "POST":

        ##################################   Mystock (주가 예측)  ##############################################
        # 기계 [Tensorflow] LSTM RNN을 이용하여 아마존 주가 예측하기
        # 학습을 통한 하나의 주식에 대한 다음날 주식 예측  +  qlearn(강화학습)을 통해 자동 주식 매매

        # 하이퍼파라미터
        seq_length = 28  # 1개 시퀀스의 길이(시계열데이터 입력 개수: 학습을 위한 데이터 갯수 0~28일 => 29번째 날를 위한 시퀀스)

        # 입력 파일 지정하기
        item = request.POST["item"]
        trade = request.POST.get("trade")
        option = request.POST.get("option")
        update = request.POST.get("update")

        # 준비: 디렉토리 생성
        # maybe_make_dir('stock/refStage/weights')
        # maybe_make_dir('stock/refStage/portfolio')

        downloadfile = "C:/Users/USER/Downloads/" + item + ".csv"
        stock_file_name = "C:/Users/USER/PycharmProjects/futureworld/stock/refStage/stock/" + item + ".csv"

        # 준비: 날짜 reference : The strftime() : datetime 데이터를 문자열로 전환 :  method returns a string representing date and time using date, time or datetime object.
        # now = datetime.now()                                    # current date and time
        # timestamp = now.strftime('%Y%m%d%H%M')                  # datatime을 문자열로 전환

        ##################################### 1. 옵션 : 최신 주식 데이터 업데이트   #############################
        if update == "update":

            # 기존 파일 제거
            if os.path.exists(downloadfile):
                os.remove(downloadfile)
            if os.path.exists(stock_file_name):
                os.remove(stock_file_name)

            # base_date = datetime(2015, 1, 1).date()
            my_string = "2015-01-01"
            start_day = 1420070400            # 2015.1.1 기준
            base_date = datetime.strptime(my_string, "%Y-%m-%d")
            base_date = base_date.date()
            print("base_date:", base_date)

            today = datetime.now().date()
            print("today:", today)

            # Difference between two dates
            n_date = today - base_date
            print("날짜 기간:", n_date.days)

            ref_1=start_day+n_date.days*86400
            print(ref_1)


            browser = webdriver.Chrome("C:/Users/USER/PycharmProjects/futureworld/static/resource/chromedriver.exe")
            browser.implicitly_wait(3)

            url="https://finance.yahoo.com/quote/"+item+"/history?period1="+str(start_day)+"&period2="+str(ref_1)+"&interval=1d&filter=history&frequency=1d"
            browser.get(url)
            print(url)

            download = browser.find_element_by_css_selector('#Col1-1-HistoricalDataTable-Proxy section div:nth-child(1) div:nth-child(2) span:nth-child(2) a')
            download.click()

            time.sleep(15)                              # 5초 대기

            browser.quit()

            redirect('stock/stockmodel.html')


        ####### 2. 주식 데이터 분석(정규화, X-train, Y-train 추출)

        ## 파일 복사
        shutil.copyfile(downloadfile, stock_file_name)

        encoding = 'euc-kr'                                                                         # 문자 인코딩
        names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        ################## csv 파일을 pandas dataframe으로 불러들임
        raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding)                # Pandas 이용 csv파일 로딩(dataframe)

        raw_dataframe.info()                                                                        # 데이터 정보 출력

        ## dataframe에서 데이터 불필요 부분 정리
        del raw_dataframe['Date']                           # axis=1 세로 column축 date 열 시간열을 제거(dataframe 재생성하지 않음)
        # raw_dataframe.drop('Date', axis=1, inplace=True)

        print("raw_dataframe-type:", type(raw_dataframe))

        ######################################## ndarray로 데이터 변환(dataframe => nparray) 데이터 형태 변환
        stock_info = raw_dataframe.values[1:]            # (1259,6) 금액&거래량 문자열을 부동소수점형으로 변환한다(0줄 레벨 제외)
        print("stock_info:", type(stock_info))

        stock_info = stock_info.astype(np.float)      #  => 정수를 실수형으로 전환

        # 데이터 정규화 : 가격과 거래량 수치의 차이가 많아나서 각각 별도로 정규화한다
        price = stock_info[:, :-1]                              # (1259, 5),  volumn(마지막 열) 제외한 데이터
        norm_price = min_max_scaling(price)                     # 가격형태 데이터 정규화 처리(5개 열)

        # 거래량형태 데이터를 정규화한다
        # ['Open','High','Low','Close','Adj Close','Volume']에서 마지막 'Volume'만 취함
        # [:,-1]이 아닌 [:,-1:]이므로 주의하자! 스칼라가 아닌 벡터값 산출해야만 쉽게 병합 가능(차원 통일 해야)

        volume = stock_info[:, -1:]                             # (1259, 1)   volumn 데이터만 가져옴(1개 열)
        norm_volume = min_max_scaling(volume)                   # 거래량형태 데이터 정규화 처리

        ## 새로운 데이터(dafaframe) 생성 :  행은 그대로 두고 열을 우측에 붙여 합친다  axis 0(동일한 컬럼),1(상이한 컬럼)
        ## axis : {0/’index’, 1/’columns’}, default 0


        stock_data = np.concatenate((norm_price, norm_volume), axis=1)          # (1259, 6) : column을 합친다(1259,5) +(1259,1)
        # stock_data = pd.concat([norm_price, norm_volume], axis=1)             # pasdas numpy 차이

        print("stock_data.shape", stock_data.shape)
        # stock_data = np.around(stock_data)                          # (1259, 6)

        cPrice = stock_data[:, [-3]]                            # (1268,1) 정규화된 주식종가(close행 -3)이다
        closePrice = stock_info[:, [-3]]                        # (1268,1) 아래 Qlearning을 위함 비정규화된 주식 종가 (close행 -3)이다

        dataX = []                                              # 리스트 입력으로 사용될 Sequence Data
        dataY = []                                              # 리스트 출력(타켓)으로 사용

        for i in range(0, len(stock_data) - seq_length):
            _x = stock_data[i: i + seq_length]                  # [ [ 0:28] [1:29].......[?~?+28]
            _y = cPrice[i + seq_length]                         # 다음 나타날 주가(정답)  cPrice[28], cPrice[29].....cPrice[?+28]

            dataX.append(_x)                                    # 리스트 (1231, 28, 6) dataX 리스트에 추가 => 3차원 리스트 생성
            dataY.append(_y)                                    # 리스트 (1231, 1) dataY 리스트에 추가

        print("dataX.shape", np.array(dataX).shape)
        print("dataY.shape", np.array(dataY).shape)

        # 학습용/테스트용 데이터 생성(전체 90%를 학습용 데이터로 사용, 나머지(10%)를 테스트용 데이터로 사용)
        train_size = int(len(dataY) * 0.9)
        test_size = len(dataY) - train_size

        # 데이터를 잘라 학습용 데이터,  테스트용 데이터 생성(리스트를 nparray로 생성)
        x_train = np.array(dataX[0:train_size])                 # (1168, 28, 6)
        y_train = np.array(dataY[0:train_size])                 # (1168, 1)
        x_test = np.array(dataX[train_size:len(dataX)])         # (125, 28, 6)
        y_test = np.array(dataY[train_size:len(dataY)])         # (125, 1)



        ######### 3. 주가 예측 모델
        start_time = datetime.now()                     # 시작시간을 기록한다
        print('학습을 시작합니다...')

        ## model 이용하는 경우(LSTM방식 이용)
        input_data_column_cnt = 6  # 입력데이터의 컬럼 개수(Variable 개수)
        output_data_column_cnt = 1  # 결과데이터의 컬럼 개수
        epoch_num = 100  # 에폭 횟수(학습용 전체데이터를 몇 회 반복해서 학습할 것인가 입력)
        batch_size_num = 100

        model_file = "stock/refStage/weights/" + item + "_predict.h5"

        #######  주식 예측 모델이 존재 여부
        if not os.path.exists(model_file):

            model = Sequential()
            model.add(LSTM(seq_length, return_sequences=True, input_shape=(seq_length, input_data_column_cnt)))     # 28개 유닉스 지정하고 입력값은 (28,6)개
            model.add(LSTM(64, return_sequences=False))                                                             # 64개 유닉스 지정(조정해서 최적치 구할수)
            model.add(Dense(output_data_column_cnt, activation='linear'))                                           # 1개의 결과 산출
            model.compile(loss='mse', optimizer='rmsprop')

            # 모델 상태 보기
            model.summary()

            # 모델 학습하기
            model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size_num, epochs=epoch_num)

            end_time = datetime.now()                               # 종료시간을 기록한다
            elapsed_time = end_time - start_time                    # 경과시간을 구한다
            print('elapsed_time per epoch:', elapsed_time / epoch_num)

            ## 주가 예상가격 모델 가중치 저장
            model.save('stock/refStage/weights/'+ item + '_predict.h5')

        else:

            model = load_model(model_file)


        ###################  주가 가격 예상(TensorFlow 플레이스 홀더 확인)

        # sequence length만큼의 가장 최근 데이터를 슬라이싱한다

        recent_data = np.array([stock_data[len(stock_data) - seq_length:]])        #(1, 28, 6)    (1258,6)에서 마지막 28일간의 주식데이터를 추출
        print("recent_data.shape", recent_data.shape)

        # 내일 종가 예측
        next_predict = model.predict(recent_data)           # (1,1) : recent_data(1, 28, 6) 최근 28일 데이터의 예상치는 하나의 가격[ [ ] ]

        # 금액데이터를 역정규화
        next_predict = reverse_min_max_scaling(price, next_predict)

        resp1 = next_predict[0][0]                              # 가장 확율이 높은 예상가격
        resp0 = closePrice[len(closePrice)-1][0]                # 전날 최종가격
        rate = (resp1 - resp0) / resp0 * 100                    # 상승율

        # 결과물 처리 : 소수점 2자리에서 반올림
        # resp1 = round(resp1, 2)                    # 소숫점 2자리에서 라운드
        resp1 = "%.2f" % resp1                      # 문자열로 (소숫점2자리)표현
        resp0 = round(resp0, 2)
        rate = round(rate, 2)                        # rate = "%.2f" % rate

        ####################  옵션2: 예상가격과 y_test(실제 값) 비교 그래프###################################

        if option == "plot":

            prediction = model.predict(x_test)          # (124,1) : x_test(124, 28, 6) 최근 28일 데이터의 예상치는 124일의 예상 가격
            plot(y_test, prediction, price)             # 실제가격(y_test (124,1))과 예상가격의 그래프 작성

        ####################  옵션2: 관련 뉴스 검색 ###################################
        # article_dict = {}
        #
        # if option == "news":
        #     url = "https://news.google.com/search?q="+item+"%20when%3A1d%20-market&hl=en-US&gl=US&ceid=US%3Aen"
        #     resp = requests.get(url)
        #
        #     soup_1 = BeautifulSoup(resp.text, "html.parser")
        #     news_list = soup_1.select("main.HKt8rc div.lBwEZb div.xrnccd h3 a")
        #
        #     for news_item in news_list:
        #
        #         article_item = news_item.string
        #         article_item = article_item.replace(",", "")
        #         # article_list.append(article_item)
        #
        #         item_href = news_item.attrs['href']
        #         item_href = item_href.replace(".", "")
        #         # href_list.append(item_href)
        #
        #         article_dict[article_item] = item_href

        ################################################################################################################
        ####  옵션 한종목에 대한 매매 추천 :  Q-Learning(강화학습),    Standardization, Sequence Model  #####
        ####  (멀티 종목에 대한 분석은 아래 Qlearn_trade 메소드에 정리)
        ################################################################################################################

        recommended_action = "매매방법 미지정"

        if trade == 'action':
            ## 최종가격(비정규화) 주식 데이터 추출(학습용, 테스트용)
            train_data = closePrice[:train_size]                #(1139,1)  : 비정규화된 값
            test_data = closePrice[train_size:]                 #(125,1)

            # 참고자료 : The strftime() : datetime 데이터를 문자열로 전환  method returns a string representing date and time using date, time or datetime object.
            now = datetime.now()                                # current date and time
            timestamp = now.strftime('%Y%m%d%H%M')              # datatime을 문자열로 전환

            #############      Env(주식시장) 인스턴스 생성:  ##############
            # (#1.reset(state)  #2.agent(트레이딩로봇)의 action에 대한 거래  #3.action 이후 state, next_state, reward 메모리 구축/업데이트)
            initial_invest = 20000

            # default로서 train_data(학습용) 주식 데이터를 사용

            env = TradingEnv(train_data, initial_invest)        # train_data (1139,1) 과 초기 자금을 통해 주식 env(환경) 인스턴스 생성

            # Standization 정규화 과정 인스턴스 :   Takes a env and returns a scaler for its observation space
            scaler = get_scaler(env)

            ############     Agent(트레이딩 로봇) 인스턴스 생성:  ###################
            # #1. action(매수/매도/보유)결정   #2. 예상가격(target, y_test) 산정 및 학습(model.fit)

            state_size = env.observation_space.shape            # (3,2) : [stock_range[], price_range[], cash_in_hand_range[]]
            action_size = env.action_space.n                    # 3 ==(0,1,2)       <주의> .n  없으면 에러

            # state(3,2)와 action_size(3)를 통해 action(3)중 최적을 구하는 모델을 구현 하는 인스턴스
            agent = DQNAgent(state_size, action_size)


            weights_file = 'stock/refStage/weights/' + item + '_trading.h5'

            #######  trading 예측 모델(가중치)이 존재 여부 : 학습된 모델 가중치가 있다면 로드(load trained weights)
            ####  test_data 데이터(가장 최근 주직가격 포함) 를 통해 최종 매매 추천을 구함
            if os.path.exists(weights_file):

                # remake the env with test data
                env = TradingEnv(test_data, initial_invest)             # test_data(125,1)

                # Standization 정규화 과정 인스턴스 :   Takes a env and returns a scaler for its observation space
                scaler = get_scaler(env)

                agent.load(weights_file)            # # model.load_weights(weights_file)

            ############  100회의 에피소드(s->a->r,s1  : s=s1) 반복 => 모델 가중치 저장

            episode = 100
            batch_size = 32
            portfolio_value = []
            actions = []

            for e in range(episode):                            # 0 ~ 100

                # 초기화 state 생성
                state = env._reset()                            # : state(1,3)  :[ [stock_owned, stock_price, cash_in hand]  ]
                state = np.array(state)

                # state 표준화(standardization)
                state = scaler.transform(state)

                for date in range(env.n_step - 1):              # 0 ~ 1267   n_step(주식 데이터수) : 1268

                    # state 입력데이터를 모델 예측에 활용하여 action 산출
                    action = agent.act(state)

                    # action에따라  next_state, reward 산출
                    next_state, reward, done, info = env._step(action)

                    # next_state 표준화
                    next_state = scaler.transform(next_state)

                    #################  옵션 : 모델 학습 Train  #################################
                    if trade == 'train' or not os.path.exists(weights_file):

                        #  매 에피소드 마다 memory에 저장
                        agent.remember(state, action, reward, next_state, done)

                        ################## 모델 학습   (model.fit)###################
                        # 에피소드(100회) 처리중 memory 저장 횟수가 32회(에피소드 32회 동일) 이상이면 처리 속도를 높이기 위해
                        # 메모리에서 랜덤 샘플을(32개) 만들어 처리
                        # for문 보다 배열 처리가 훨씬 빠르다

                        if len(agent.memory) > batch_size:          # len(memory) > 32(초기 32일 이후 33일째 에피소드)

                            # 로봇(agent)에서 학습 처리(model.fit)  => 여기에서 모델에 학습

                            agent.replay(batch_size)



                    state = next_state  # (1,3)  거래후 상태가 다음날에는 초기 상태로

                    ################# 하나의 에피소드(n_step 주식 데이터수 1268) 종료시 결과물 생산 ##################
                    # 각 에피소드당 n_step(주식 데이터수)만큼 학습하였다면(done="true") 최종적 주식 가치, 매매 종류 추천 저장
                    if done:
                        print("episode: {}/{}, episode end value: {}/action:{}".format(
                            e + 1, episode, info['cur_val'], action))

                        portfolio_value.append(info['cur_val'])        # 각 에피소드가 끝날 때의 포트폴리오 가치를 기록합니다.

                        actions.append(action)                          # 에피소드가 끝날 때(마지막날)의 매매방법을 기록합니다.


                ################### 에피소드 10회 당 agent (가중치) 저장 => model.save_weight  ("/weights/ibm/train-dqn.h5")
                if (not os.path.exists(weights_file) or trade == 'train') and (e + 1) % 10 == 0:

                    agent.save(weights_file)

            # 각 에피소드당 포드포리오(자산가치) 저장 => 텍스트가 아닌 (리스트)데이터를 저장할때 => pickle 모듈 이용
            # with open('stock/refStage/portfolio/' + item + '-{}.p'.format(trade), 'wb') as fp:
            #     pickle.dump(portfolio_value, fp)

            #### 매매 추천(에피소드당 최종일의 매매추천중 최대를 추천)
            a, b, c = actions.count(0), actions.count(1), actions.count(2)
            action_list = np.array([a,b,c])

            if np.argmax(action_list) > 1:
                recommended_action = "매수"
            elif np.argmax(action_list) < 1:
                recommended_action = "매도"
            else:
                recommended_action = "Holding"


        resp = {
            "item": item,
            "resp0": resp0,
            "resp1": resp1,
            "rate": rate,
            "action": recommended_action,
            # "article_list": article_list,
            # "href_list": href_list,
            # "article_dict": article_dict,
        }
        print(resp)
        ## dictionary => json 문자열로 전환 하여 ajax로 전달  => json.dumps()
        ## nd array(numpy)에서는 json화에 에러가 발생하여 이를 해결하기 위해 cls = NumpyEncoder 클래스 추가

        # resp = json.dumps(resp, cls=NumpyEncoder)

        return render(request, 'stock/stockresult.html', resp)
        # return HttpResponse(resp)

    else:

        # return render(request, 'stock/stockmodel.html')
        print('check')
        return render(request, 'stock/stockmodel.html')



# 관련 메소드
##########################  mystock 관련 ###################################################################

# state 생성 : 2D로 구조로 작업(1,3)
def get_state(stock_owned, stock_price, cash_in_hand):

    obs = []

    obs.append(stock_owned)
    obs.append(stock_price)
    obs.append(cash_in_hand)

    state = []
    state.append(obs)      #(1,3)  => [ [stock_owned, stock_price, cash_in_hand] ]

    return state

####################################################################  데렉토리 생성
def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

####################################################################### 정규화 모듈
# Normalization (Min-Max scaling) : x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)       # 1e-7은 0으로 나누는 오류 예방차원

### 정규화 복원
# De-standization : 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

################################################################# Standization (표준화) :
# ###sklearn의 StandardScaler메소드를 이용한 env(주식시장)에서의 정규화
# 평균값 0, 표준 편차 1의 값으로 변환시킴. (표준화란 값이 평균으로 부터 얼마만큼의 표준 편차가 있는지를 측정하는 것)
def get_scaler(env):
    """ Takes a env and returns a scaler for its observation space """

    low = [0] * (env.n_stock * 2 + 1)                   # [0][0][0]
    high = []

    max_price = env.stock_price_history.max()
    min_price = env.stock_price_history.min()
    max_cash = env.init_invest * 3                      # 3 is a magic number...
    max_stock_owned = max_cash // min_price

    high.append(max_stock_owned)
    high.append(max_price)
    high.append(max_cash)

    # 표준화 과정
    scaler = StandardScaler()
    scaler.fit([low, high])                     # (3,2)   [[0,122][0,120][0,28000]]

    return scaler

def get_scaler_multi(env):
  """ Takes a env and returns a scaler for its observation space
   차이점 : shape 차원에 대한 처리과정(axis=1)
   """

  low = [0] * (env.n_stock * 2 + 1)    #[0][0][0]
  high = []
  max_price = env.stock_price_history.max(axis=1)
  min_price = env.stock_price_history.min(axis=1)
  max_cash = env.init_invest * 3                            # 3 is a magic number...
  max_stock_owned = max_cash // min_price

  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)

  high.append(max_cash)

  scaler = StandardScaler()
  scaler.fit([low, high])

  return scaler

#######################################################   그래프그리기
def plot(y_test, prediction, price):

    ## plt . plot ( [x축 값] , [ y축 값 ] )  =>   plt . show()
    ### fig = plt . figure ( )
    ### ax1 = fig.add_subplot(2, 1, 1)     # (nRows, nColumns, axes number to plot)
    ### ax2 = fig.add_subplot(2, 1, 1)




    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)

    y_test = reverse_min_max_scaling(price, y_test)
    ax.plot(y_test, label="Mid-Price")

    prediction = reverse_min_max_scaling(price, prediction)
    ax.plot(prediction, label='Prediction')

    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.legend()

    plt.show()
    a=1

    return a

# Numpy 타입의 json화를 위해 아래 클래스 추가
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):                   #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



#########################################  멀티 종목 Auto-Trading ###################################################
##### Multi 종목에 대한 Auto-Trading/매매추천 : Q-Learning(강화학습),  Standardization, Sequence Model  #############
#####################################################################################################################
def stocktrade(request):

    if request.method == "POST":

        # 하이퍼파라미터
        # seq_length = 28  # 1개 시퀀스의 길이(시계열데이터 입력 개수: 학습을 위한 데이터 갯수 0~28일 => 29번째 날를 위한 시퀀스)

        # 입력 주식 종목 데이터 처리
        items = request.POST["items"]
        option = request.POST["option"]

        items = items.split(',')
        n_stock = len(items)

        my_stock = {}
        for idx, item in enumerate(items):
            my_stock[idx] = item

        print(my_stock)

        # 준비: 디렉토리 생성
        maybe_make_dir('stock/refStage/weights')
        maybe_make_dir('stock/refStage/portfolio')

        # 준비: 날짜 reference : The strftime() : datetime 데이터를 문자열로 전환 :  method returns a string representing date and time using date, time or datetime object.
        now = datetime.now()                                    # current date and time
        timestamp = now.strftime('%Y%m%d%H%M')                  # datatime을 문자열로 전환

        ######################### 옵션1:  최신 주식 데이터 업데이트   #############################
        if option == "update":

            downloadfile = {}
            for idx, item in enumerate(items):
                downloadfile[idx] = "C:/Users/ilee/Downloads/" + my_stock[idx] + ".csv"

                # 기존 파일 제거
                if os.path.exists(downloadfile[idx]):
                    os.remove(downloadfile[idx])

            browser = webdriver.Chrome("C:/Users/USER/PycharmProjects/futureworld/static/resource/chromedriver.exe")
            browser.implicitly_wait(3)

            url = "https://finance.yahoo.com/quote/"+my_stock[0]+"/history?p="+my_stock[0]
            browser.get(url)

            redirect('stock/stocktrade.html')

        ################## 2. 주식 데이터 분석(정규화, X-train, Y-train 추출)
        stock_file_name = {}
        for idx, item in enumerate(items):
            stock_file_name[idx] = "stock/refStage/stock/" + my_stock[idx] + ".csv"                           # 주가데이터 파일


        # encoding = 'euc-kr'                                     # 문자 인코딩
        # names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        col = 'Close'
        stock_data = {}
        for idx, item in enumerate(items):
            stock_data[idx] = pd.read_csv(stock_file_name[idx], usecols=[col])                # Pandas 이용 csv파일 로딩(dataframe)

        # item1.info()
        print("stock-data-shape:", np.array(stock_data).shape)

        ####  주가 데이터 np.array로 생성   # (3, 4526)  3개 주식, 4526일 최종가각 데이터

        ## dataframe 데이터를 nparray로 전환  =>    np_raw_dataframe = raw_dataframe.as_matrix()
        # stock_info = item1.values[1:].astype(np.float)    # (1259,6) 금액&거래량 문자열을 부동소수점형으로 변환한다(0줄 레벨 제외)

        # stock_info = np.array([item1[col].values, item2[col].values, item3[col].values])

        stock_info = []
        for idx, item in enumerate(items):
            # stock_info = np.array([item1[col].values[::-1], item2[col].values[::-1], item3[col].values[::-1]])     # [::-1] => 데이터 역순
            stock_info.append(stock_data[idx][col].values[::-1])

        stock_info = np.array(stock_info)

        # data = np.around(stock_info)                                    # (3, 4526)  3개 주식, 4526일 최종가각 데이터
        print("stock_info-shape:", np.array(stock_info).shape)

        ### 1. 주식 데이터(학습용, 테스트용) 구분/ 호출
        train_data = stock_info[:, :1168]                                # (3, 1168)
        test_data = stock_info[:, 1168:]


        ## 2. Q-Learning을 통한 강화학습 => (s->a->r,s1  : s=s1) 반복 => 모델 가중치 저장  Standardization, Sequence Model

        #############      Env(주식시장) 인스턴스 생성:  ##############
        # (#1.reset(state)  #2.agent(트레이딩로봇)의 action에 대한 거래  #3.action 이후 state, next_state, reward 메모리 구축/업데이트)

        initial_invest = 20000

        # default 데이터로 학습용 주식 데이터를 사용
        env = TradingEnv_multi(train_data, initial_invest)            # train_data (3, 3526), 초기 자금을 통해 주식 env(환경) 인스턴스 생성

        # Standization 정규화 과정 인스턴스 :   Takes a env and returns a scaler for its observation space
        scaler = get_scaler_multi(env)

        ############     Agent(트레이딩 로봇) 인스턴스 생성:  ###################
        # action(매수/매도/보유)결정   #2. 예상가격(target, y_test) 산정 및 학습(model.fit)
        # state 사이즈(7,2) +  action 사이즈(27) 입력
        # 한 종목과의 차이 state (3,2) : [stock_range[], price_range[], cash_in_hand_range[]]과 차이 비교

        state_size = env.observation_space.shape         # (7,2) : 보유주식수(3,2) + 주식가격(3,2) + 보유현금(1, 2)  : ([0, ]
        action_size = env.action_space.n                # 27  <주의> .n  없으면 에러

        # state(7,2)와 action_size(27)를 통해 action(27)중 최적을 구하는 모델을 구현 하는 인스턴스
        # (차이 :  state(3,2)와 action_size(3)를 통해 action(3))

        agent = DQNAgent_multi(state_size, action_size)

        w_file = []
        for idx, item in enumerate(items):
            w_file.append(item)

        stock_name = '_'.join(w_file)
        weights_file = 'stock/refStage/weights/'+ stock_name +'_trading.h5'

        #######  trading 예측 모델(가중치)이 존재 여부 : 학습된 모델 가중치가 있다면 로드(load trained weights)
        ##       test 데이터(가장 최근 주직가격 포함) 를 통해 최종 매매 추천을 구함

        if os.path.exists(weights_file):
            # remake the env with test data
            env = TradingEnv_multi(test_data, initial_invest)                        # (125,1)

            # Standization 정규화 과정 인스턴스 :   Takes a env and returns a scaler for its observation space
            scaler = get_scaler_multi(env)

            agent.load(weights_file)  # # model.load_weights(weights_file)


        ############  100회의 에피소드(s->a->r,s1  : s=s1) 반복 => 모델 가중치 저장

        episode = 100
        batch_size = 32
        portfolio_value = []
        actions = []

        for e in range(episode):  # 0 ~ 100

            # 초기화 state 생성
            state = env._reset()  # : state(1,3)  :[ [stock_owned, stock_price, cash_in hand]  ]

            state = np.array(state)

            # state 표준화(standardization)
            state = scaler.transform([state])
            # state = scaler.transform(state)

            for date in range(env.n_step - 1):  # 0 ~ 1267   n_step(주식 데이터수) : 1268

                # state를 통해 action을 구하는
                action = agent.act(state)

                # action에따라  next_state, reward 산출
                next_state, reward, done, info = env._step(action)

                # next_state 표준화
                next_state = scaler.transform([next_state])
                # next_state = scaler.transform(next_state)


                #################  학습 Train  #################################

                if option == 'train' or not os.path.exists(weights_file):

                    #  매 에피소드 마다 memory에 저장
                    agent.remember(state, action, reward, next_state, done)

                    ################## 모델 학습   (model.fit)###################
                    # 에피소드(100회) 처리중 memory 저장 횟수가 32회(에피소드 32회 동일) 이상이면 처리 속도를 높이기 위해
                    # 메모리에서 랜덤 샘플을(32개) 만들어 처리
                    # for문 보다 배열 처리가 훨씬 빠르다

                    if len(agent.memory) > batch_size:  # len(memory) > 32(초기 32일 이후 33일째 에피소드)

                        # 로봇(agent)에서 학습 처리(model.fit)
                        agent.replay(batch_size)

                    # state = next_state  # (1,3)  거래후 상태가 다음날에는 초기 상태로

                ################# 하나의 에피소드(n_step 주식 데이터수 1268) 종료시 ##################
                # 에피소드당 n_step(주식 데이터수)만큼 학습하였다면(done="true") 최종적 주식 가치 표시
                if done:
                    print("episode: {}/{}, episode end value: {}/action:{}".format(
                        e + 1, episode, info['cur_val'], action))

                    portfolio_value.append(info['cur_val'])  # 에피소드가 끝날 때의 포트폴리오 가치를 기록합니다.

                    actions.append(action)                      # 에피소드가 끝날 때(마지막날)의 매매방법을 기록합니다.

            ################### 에피소드 10회 당 agent (가중치) 저장 ("/weights/ibm/train-dqn.h5")
            if (not os.path.exists(weights_file) or option == 'train') and (e + 1) % 10 == 0:         # weights를 중간중간에 저장합니다.

                agent.save(weights_file)

        # 각 에피소드당 포드포리오(자산가치) 저장 => 텍스트가 아닌 (리스트)데이터를 저장할때 => pickle 모듈 이용
        with open('stock/refStage/portfolio/'+ stock_name +'.p', 'wb') as fp:
            pickle.dump(portfolio_value, fp)

        #### 자산가치 Plot 생성

        prediction = portfolio_value

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(1, 1, 1)


        x1 = [0, 100]
        y1 = [initial_invest, initial_invest]

        ax.plot(x1, y1, label="Initial invest")
        ax.plot(prediction, label='Value')

        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        head_title = "auto-trading : " + stock_name
        ax.set_title(head_title)

        ax.legend()

        plt.show()

        #### 매매 추천
        # a, b, c = actions.count(0), actions.count(1), actions.count(2)
        # action_list = np.array([a, b, c])

        # if np.argmax(action_list) > 1:
        #     recommended_action = "매수"
        # elif np.argmax(action_list) < 1:
        #     recommended_action = "매도"
        # else:
        #     recommended_action = "Holding"
        #
        # resp = {
        #     "action": recommended_action,
        # }

        ## dictionary => json 문자열로 전환 하여 ajax로 전달  => json.dumps()
        ## nd array(numpy)에서는 json화에 에러가 발생하여 이를 해결하기 위해 cls = NumpyEncoder 클래스 추가

        # resp = json.dumps(resp, cls=NumpyEncoder)

        resp = "test"
        return HttpResponse(resp)

    else:
        return render(request, "stock/stocktrade.html")


