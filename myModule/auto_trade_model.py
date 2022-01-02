from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# state_size((3,2) 또는 (7,2))를 통해 => 최적의 action(3 또는 27)을 구하는 모델
def mlp(state_size, action_size):

    # n_obs, n_action, n_hidden_layer = 1, n_neuron_per_layer = 32, activation = 'relu', loss = 'mse'
    """ A multi-layer perceptron """

    n_hidden_layer = 1          # 은닉층 갯수
    n_neuron_per_layer = 32     # 계층별(layer) 신경망 갯수(unites) => 레이어 속성으로 output shape와 연관
    activation = 'relu'        # rulu : 주로 은닉층에 사용되는 활성화 함수
    loss = 'mse'               # compile시 손실함수
    optimizer = Adam()          # 모델의 업데이트 방법

    print("state_size:", state_size)

    input_data_column_cnt = state_size[0]        # 입력데이터 컬럼 갯수(3 or 7)
    output_data_column_cnt = action_size                 # 출력데이터 컬럼 갯수 (3 or 27)


    #######    Keras Sequential Model로 작성
    model = Sequential()

    # 입력층(input layer)   # imput_shape=(n_input,) or input_dim = n_input
    model.add(Dense(n_neuron_per_layer, input_shape=(input_data_column_cnt,), activation=activation))

    # 은닉층(hidden layer)
    for _ in range(n_hidden_layer):
        model.add(Dense(n_neuron_per_layer, activation=activation))

    # 출력층(output layer)
    model.add(Dense(output_data_column_cnt, activation='linear'))

    # model compiling
    model.compile(loss=loss, optimizer=Adam())

    # 모델 요약: 다수의 출력 레이어 경우 각자의 개별 출력 shape 대신 multiple로 표현(크기 제한에 따름)
    print(model.summary())

    return model