import numpy as np

#dataset preparation

input_data = open('read_it.txt', 'r').read()
contents = input_data.lower()

chars_set = set()

for char in contents:
    chars_set.add(char)

chars = list(chars_set)

vocabulary_size = len(chars)  # vocabulary size =27
char_to_index = {}
index_to_chars = {}

for i in range(vocabulary_size):
    index_to_chars[i] = chars[i]  # (0 : '\n')

for i, ch in index_to_chars.items():
    char_to_index[ch] = i  # ('\n' : 0)


#hyperparameters
number_of_neuron = 100
T_steps = 25
weight_sd = .2

learning_rate = .01

#weight parameters initialize randomly
W_r = np.random.randn(number_of_neuron, vocabulary_size) * weight_sd + .1
W_C = np.random.randn(number_of_neuron, vocabulary_size) * weight_sd + .1
W_u = np.random.randn(number_of_neuron, vocabulary_size) * weight_sd + .1
W_y = np.random.randn(vocabulary_size, number_of_neuron) * weight_sd + .1

U_r = np.random.randn(number_of_neuron, number_of_neuron) * weight_sd + .1
U_C = np.random.rand(number_of_neuron, number_of_neuron) * weight_sd + .1
U_u = np.random.rand(number_of_neuron, number_of_neuron) * weight_sd + .1

#biases initialize with zero
b_r = np.zeros((number_of_neuron, 1), dtype=float)
b_C = np.zeros((number_of_neuron, 1), dtype=float)
b_u = np.zeros((number_of_neuron, 1), dtype=float)
b_y = np.zeros((vocabulary_size, 1), dtype=float)



parameters = {'W_r': W_r, 'W_C': W_C, 'W_u': W_u, 'W_y': W_y, 'U_r': U_r, 'U_C': U_C, 'U_u': U_u}
biases = {'b_r': b_r, 'b_C': b_C, 'b_u': b_u, 'b_y': b_y}


def sigmoid(p):
    #return 1 / (1 + np.exp(-1 * x))

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i][j] > 0:
                p[i][j] = 1. / (1. + np.exp(-p[i][j]))
            elif p[i][j] <= 0:
                p[i][j] = np.exp(p[i][j]) / (1 + np.exp(p[i][j]))
            else:
                raise ValueError


    return p



def dsigmoid(x):
    return x * (1 - x)


def TANH(p):
    #print(type((np.exp(z) - np.exp(-1 * z)) / (np.exp(z) + np.exp(-1 * z))))
    #return (np.exp(z) - np.exp(-1 * z)) / (np.exp(z) + np.exp(-1 * z))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):

            if p[i][j] > 0:
                p[i][j] = (1 - np.exp(-2 * p[i][j])) / (1 + np.exp(-2 * p[i][j]))
            elif p[i][j] <= 0:
                p[i][j] = (np.exp(2 * p[i][j]) - 1) / (np.exp(2 * p[i][j]) + 1)
            else:
                raise ValueError

    return p


def dtanh(z):
    return 1 - z * z


def softmax(x):
    e_x = np.exp(x - np.max(x))

    return e_x / np.sum(e_x, axis=0)


def clip(parameter, max_value, min_value):

    for p in parameter.values():
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):

                #if i == 0 and j == 0:
                    #print(p[i][j])
                if p[i][j] > max_value:
                    p[i][j] = max_value
                elif p[i][j] < min_value:
                    p[i][j] = min_value

    return parameter


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1 if x > 0 else 0



def gru_forward(x, y, C_previous):

    reset_gate, update_gate, cell_state, C_bar, input_gate = {}, {}, {}, {}, {}
    probability, target = {}, {}

    cell_state[-1] = np.copy(C_previous)
    input_gate[0] = np.zeros((vocabulary_size, 1), dtype=float)

    loss = 0
    input_len = len(x)



    for t in range(input_len):

        input_gate[t] = np.zeros((vocabulary_size, 1), dtype=float)
        input_gate[t][x[t]] = 1

        reset_gate[t] = sigmoid(W_r.dot(input_gate[t]) + U_r.dot(cell_state[t-1]) + b_r)
        update_gate[t] = sigmoid(W_u.dot(input_gate[t]) + U_u.dot(cell_state[t-1]) + b_u)
        C_bar[t] = TANH(W_C.dot(input_gate[t]) + U_C.dot(np.multiply(reset_gate[t], cell_state[t-1])) + b_C)
        cell_state[t] = update_gate[t] * cell_state[t-1] + (1 - update_gate[t]) * C_bar[t]

        probability[t] = softmax(W_y.dot(cell_state[t]) + b_y)

        target[t] = np.zeros_like(input_gate[t])
        target[t][y[t]] = 1

        for i in range(vocabulary_size):
            loss += -target[t][i, 0] * np.log(probability[t][i, 0])

        cache = (input_gate, cell_state, C_bar, reset_gate, update_gate, probability)
        return cache, loss



def gru_backward(y, cache):

    input_gate, cell_state, C_bar, reset_gate, update_gate, probability = cache

    # print('dh  ', dh_next.shape)
    dC_next = np.zeros_like(cell_state[0])
    input_len = len(input_gate)

    gradients = {}
    d_biases = {}

    global parameters, biases

    for parameters_name in parameters.keys():
        gradients['d_' + parameters_name] = np.zeros_like(parameters[parameters_name])

    for bias_name in biases.keys():
        d_biases['d_' + bias_name] = np.zeros_like(biases[bias_name])

    for t in range(input_len - 1, -1, -1):

        d_y = np.copy(probability[t])
        d_y[y[t]] -= 1

        gradients['d_W_y'] += d_y.dot(cell_state[t-1].T)
        d_biases['d_b_y'] += d_y

        d_cell = parameters['W_y'].T.dot(d_y) + dC_next
        dC_bar = d_cell * (1 - update_gate[t])

        gradients['d_W_C'] += (dC_bar * dtanh(C_bar[t])).dot(input_gate[t].T)
        gradients['d_U_C'] += (dC_bar * dtanh(C_bar[t])).dot(np.multiply(reset_gate[t], cell_state[t-1]).T)
        d_biases['d_b_C'] += dC_bar * dtanh(cell_state[t])

        d_r = (parameters['U_C'].T.dot(dC_bar * dtanh(cell_state[t]))) * cell_state[t-1]

        gradients['d_W_r'] += (d_r * dsigmoid(reset_gate[t])).dot(input_gate[t].T)
        gradients['d_U_r'] += (d_r * dsigmoid(reset_gate[t])).dot(cell_state[t-1].T)
        d_biases['d_b_r'] += d_r * dsigmoid(reset_gate[t])

        d_u = d_cell * (cell_state[t-1] - C_bar[t])

        gradients['d_W_u'] += (d_u * dsigmoid(update_gate[t])).dot(input_gate[t].T)
        gradients['d_U_u'] += (d_u * dsigmoid(update_gate[t])).dot(cell_state[t-1].T)
        d_biases['d_b_u'] += d_u * dsigmoid(update_gate[t])

        d_cell_update = parameters['U_C'].T.dot(d_u * dsigmoid(update_gate[t]))
        d_cell_reset = parameters['U_r'].T.dot(d_r * dsigmoid(reset_gate[t]))
        d_cell_cell = gradients['d_W_C'] * update_gate[t]
        d_cell_C_bar = (parameters['U_C'].T.dot(dC_bar * dtanh(cell_state[t]))) * reset_gate[t]

        dC_next = d_cell_update + d_cell_reset + d_cell_cell + d_cell_C_bar


    return gradients, d_biases, cell_state[input_len - 1]




def update_parameter_adagrad(m_parameters, m_biases, gradients, d_biases):

    global parameters, biases
    epsilon = 1.0e-8

    for p, m, g in zip([parameters['W_r'], parameters['W_C'], parameters['W_u'], parameters['W_y'], parameters['U_r'], parameters['U_C'], parameters['U_u']],
                       [m_parameters['m_W_r'], m_parameters['m_W_C'], m_parameters['m_W_u'], m_parameters['m_W_y'],
                        m_parameters['m_U_r'], m_parameters['m_U_C'], m_parameters['m_U_u']],
                       [gradients['d_W_r'], gradients['d_W_C'], gradients['d_W_u'], gradients['d_W_y'],
                        gradients['d_U_r'], gradients['d_U_C'], gradients['d_U_u']]):
        m += g * g
        p += -learning_rate * g / np.sqrt(m + epsilon)

    for b_p, b_m, b_d in zip([biases['b_r'], biases['b_C'], biases['b_u'], biases['b_y']],
                             [m_biases['m_b_r'], m_biases['m_b_C'], m_biases['m_b_u'], m_biases['m_b_y'],
                              ],
                             [d_biases['d_b_r'], d_biases['d_b_C'], d_biases['d_b_u'], d_biases['d_b_y']
                              ]):
        b_m += b_d * b_d
        b_p += -learning_rate * b_d / np.sqrt(b_m + epsilon)

    return m_parameters, m_biases


def update_parameter_rmsprop(m_parameters, m_biases, gradients, d_biases):

    global parameters, biases
    epsilon = 1.0e-8
    decay_rate = .9

    # m_Why = decay_rate * m_Why + (1 - decay_rate) * (dWhy ** 2)
    # Why -= .01 * dWhy / (np.sqrt(m_Why) + epsilon)

    for p, m, g in zip([parameters['W_r'], parameters['W_C'], parameters['W_u'], parameters['W_y'], parameters['U_r'], parameters['U_C'], parameters['U_u']],
                       [m_parameters['m_W_r'], m_parameters['m_W_C'], m_parameters['m_W_u'], m_parameters['m_W_y'],
                        m_parameters['m_U_r'], m_parameters['m_U_C'], m_parameters['m_U_u']],
                       [gradients['d_W_r'], gradients['d_W_C'], gradients['d_W_u'], gradients['d_W_y'],
                        gradients['d_U_r'], gradients['d_U_C'], gradients['d_U_u']]):

        m += decay_rate * m + (1 - decay_rate) * (g ** 2)
        p -= -learning_rate * g / (np.sqrt(m) + epsilon)

    for b_p, b_m, b_d in zip([biases['b_r'], biases['b_C'], biases['b_u'], biases['b_y']],
                             [m_biases['m_b_r'], m_biases['m_b_C'], m_biases['m_b_u'], m_biases['m_b_y'],
                              ],
                             [d_biases['d_b_r'], d_biases['d_b_C'], d_biases['d_b_u'], d_biases['d_b_y']
                              ]):

        b_m -= decay_rate * b_m + (1 - decay_rate) * (b_d ** 2)
        b_p -= learning_rate * b_d / (np.sqrt(b_m) + epsilon)

    return m_parameters, m_biases



def sample(C_previous, position, n):

    x = np.zeros((vocabulary_size, 1), dtype=float)
    x[position] = 1

    indices = []
    counter = 0

    #extract weights and biases
    global parameters, biases

    W_r, W_C, W_u, W_y, U_r, U_C, U_u = parameters['W_r'], parameters['W_C'], parameters['W_u'], parameters['W_y'], parameters['U_r'], parameters['U_C'], parameters['U_u']
    b_r, b_C, b_u, b_y = biases['b_r'], biases['b_C'], biases['b_u'], biases['b_y']

    #print(parameters , "\n")
    #print(biases , "\n")
    while(counter != n):

        reset_gate = sigmoid(W_r.dot(x) + U_r.dot(C_previous) + b_r)
        update_gate = sigmoid(W_u.dot(x) + U_u.dot(C_previous) + b_u)
        C_bar = TANH(W_C.dot(x) + U_C.dot(np.multiply(reset_gate, C_previous)) + b_C)
        cell_state = update_gate * C_previous + (1 - update_gate) * C_bar

        probability = softmax(W_y.dot(cell_state) + b_y)

        #print(W_o)
        index = np.random.choice(list(range(vocabulary_size)), p=probability.ravel())

        x = np.zeros((vocabulary_size, 1), dtype=float)
        x[index] = 1
        C_previous = cell_state

        indices.append(index)
        counter += 1


    return indices



def gru_forward_backward(x, y, C_previous, m_parameters, m_biases):

    cache, loss = gru_forward(x, y, C_previous)
    gradients, d_biases, cell_state = gru_backward(y, cache)

    gradients = clip(gradients, 5, -5)
    d_biases = clip(d_biases, 5, -5)
    # print('yes')

    m_parameters, m_biases = update_parameter_adagrad(m_parameters, m_biases, gradients, d_biases)
    #m_parameters, m_biases = update_parameter_rmsprop(gradients, d_biases, m_parameters, m_biases)

    return loss, cell_state, gradients, d_biases




def traning_dataset(number_of_iteration):

    m_parameters = {}  # memory parameter for momentum
    m_biases = {}

    global parameters, biases

    for parameters_name in parameters.keys():
        m_parameters['m_' + parameters_name] = np.zeros_like(parameters[parameters_name])

    for bias_name in biases.keys():
        m_biases['m_' + bias_name] = np.zeros_like(biases[bias_name])

    loss = -np.log(1.0 / vocabulary_size) * T_steps   #initial  loss

    pointer = 0
    C_previous = np.zeros((number_of_neuron, 1), dtype=float)

    for iter in range(number_of_iteration):

        if pointer + T_steps + 1 >= len(input_data) or iter == 0:
            h_previous = np.zeros((number_of_neuron, 1), dtype=float)
            C_previous = np.zeros((number_of_neuron, 1), dtype=float)
            pointer = 0

        x = [char_to_index[ch] for ch in contents[pointer:pointer + T_steps]]
        y = [char_to_index[ch] for ch in contents[pointer + 1:pointer + T_steps + 1]]

        current_loss, C_previous, gradients, d_biases = gru_forward_backward(x, y, C_previous, m_parameters, m_biases)

        loss = loss * .999 + current_loss * .001  # smooth loss

        if iter % 200 == 0:
            print('iteration: ', iter)
            print('loss: ', loss)
            print('\n')
            steps = 200
            sampled_indices = sample(C_previous, x[0], steps)

            txt = ''.join(index_to_chars[index] for index in sampled_indices)
            print(txt)
            print('\n')

        pointer += T_steps

    return C_previous




def generate_chars(C_previous, x):

    inputs = np.zeros((vocabulary_size, 1), dtype=float)
    inputs[x[0]] = 1  # one hot vector x for 1st character

    counter = 0
    index = x[0]
    indices = [index]

    # extract weights and biases
    W_r, W_C, W_u, W_y, U_r, U_C, u_u = parameters['W_r'], parameters['W_C'], parameters['W_u'], parameters['W_y'],\
                                        parameters['U_r'], parameters['U_C'], parameters['U_u']
    b_r, b_C, b_u, b_y = biases['b_r'], biases['b_C'], biases['b_u'], biases['b_y']

    while (counter < 15 and index != char_to_index['\n']):

        reset_gate = sigmoid(W_r.dot(x) + U_r.dot(C_previous) + b_r)
        update_gate = sigmoid(W_u.dot(x) + U_u.dot(C_previous) + b_u)
        C_bar = TANH(W_C.dot(x) + U_C.dot(np.multiply(reset_gate, C_previous)) + b_C)
        cell_state = update_gate * C_previous + (1 - update_gate) * C_bar

        probability = softmax(W_y.dot(cell_state) + b_y)

        if (counter + 1 < len(x)):
            index = x[counter + 1]

        else:
            index = np.random.choice(list(range(vocabulary_size)), p=probability.ravel())

        indices.append(index)
        # print(index)

        inputs = np.zeros((vocabulary_size, 1))
        inputs[index] = 1

        C_previous = cell_state

        counter += 1

    return indices


if __name__ == '__main__':

    C_0 = traning_dataset(99999991)

    indices = []

    '''now give a few characters to see 
    what word the model predicts'''

    for i in range(10):

        input_text = input("enter a chunk of sequence : ")
        x = [char_to_index[ch] for ch in input_text]

        indices = generate_chars(C_0, x)

        txt = ''.join(index_to_chars[index] for index in indices)
        print(txt)
        print('\n')
