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
z_size = 2 * number_of_neuron + vocabulary_size
learning_rate = .01

#weight parameters initialize randomly
W_f = np.random.randn(number_of_neuron, z_size) * weight_sd + .1      #forgate gate equation
W_i = np.random.randn(number_of_neuron, z_size) * weight_sd + .1      #update / input gate
W_C = np.random.randn(number_of_neuron, z_size) * weight_sd + .1
W_o = np.random.randn(number_of_neuron, z_size) * weight_sd + .1      #output gate
W_y = np.random.randn(vocabulary_size, number_of_neuron) * weight_sd + .1

#weight biases initialize with zeros
b_f = np.zeros((number_of_neuron, 1), dtype=float)
b_i = np.zeros((number_of_neuron, 1), dtype=float)
b_C = np.zeros((number_of_neuron, 1), dtype=float)
b_o = np.zeros((number_of_neuron, 1), dtype=float)
b_y = np.zeros((vocabulary_size, 1), dtype=float)

#store them in dictionary
parameters = {'W_f': W_f, 'W_i': W_i, 'W_C': W_C, 'W_o': W_o, 'W_y': W_y}
biases = {'b_f': b_f, 'b_i': b_i, 'b_C': b_C, 'b_o': b_o, 'b_y': b_y}


def sigmoid(p):
    #return 1 / (1 + np.exp(-1 * x))

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i][j] > 0:
                return 1. / (1. + np.exp(-p[i][j]))
            elif p[i][j] <= 0:
                np.exp(p[i][j]) / (1 + np.exp(p[i][j]))
            else:
                raise ValueError


    return p



def dsigmoid(x):
    return x * (1 - x)


def TANH(z):
    #print(type((np.exp(z) - np.exp(-1 * z)) / (np.exp(z) + np.exp(-1 * z))))
    return (np.exp(z) - np.exp(-1 * z)) / (np.exp(z) + np.exp(-1 * z))


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


def relu(x):
    return x * (x > 0)


def drelu(x):
    return 1 if x > 0 else 0



def lstm_forward(x, y, h_previous, C_previous):

    input_state, z_state, forgate_gate, input_gate, C_bar, cell_state, output_gate = {}, {}, {}, {}, {}, {}, {}
    hidden_state, probability, target = {}, {}, {}

    hidden_state[-1] = np.copy(h_previous)
    cell_state[-1] = np.copy(C_previous)
    input_state[0] = np.zeros((vocabulary_size, 1), dtype=float)

    loss = 0
    input_len = len(x)

    for t in range(input_len):

        input_state[t] = np.zeros((vocabulary_size, 1), dtype=float)
        input_state[t][x[t]] = 1

        z_state[t] = np.row_stack((cell_state[t - 1], hidden_state[t - 1]))
        z_state[t] = np.row_stack((z_state[t], input_state[t]))

        forgate_gate[t] = sigmoid(W_f.dot(z_state[t]) + b_f)
        input_gate[t] = sigmoid(W_i.dot(z_state[t]) + b_i)
        C_bar[t] = np.tanh(W_C.dot(z_state[t]) + b_C)           #TANH(W_C.dot(z_state[t]) + b_C)

        cell_state[t] = forgate_gate[t] * cell_state[t - 1] + input_gate[t] * C_bar[t]

        z_state[t] = np.row_stack((cell_state[t], hidden_state[t - 1]))
        z_state[t] = np.row_stack((z_state[t], input_state[t]))

        output_gate[t] = sigmoid(W_o.dot(z_state[t]) + b_o)
        hidden_state[t] = output_gate[t] * np.tanh(cell_state[t])              #TANH(cell_state[t])

        probability[t] = softmax(W_y.dot(hidden_state[t]) + b_y)

        target[t] = np.zeros_like(input_state[t])
        target[t][y[t]] = 1

        for iter in range(vocabulary_size):
            loss += -target[t][iter, 0] * np.log(probability[t][iter, 0])

    cache = (input_state, z_state, forgate_gate, input_gate, C_bar, cell_state, output_gate, hidden_state,probability)  #cache for backpropagation
    return cache, loss




def lstm_backward(y, cache):

    #retreive elements from cache
    input_state, z_state, forgate_gate, input_gate, C_bar, cell_state, output_gate, hidden_state, probability = cache

    dh_next = np.zeros_like(hidden_state[0])
    #print('dh  ', dh_next.shape)
    dC_next = np.zeros_like(cell_state[0])
    input_len = len(input_state)

    gradients = {}
    d_biases = {}

    global parameters, biases

    for parameters_name in parameters.keys():
        gradients['d_' + parameters_name] = np.zeros_like(parameters[parameters_name])

    for bias_name in biases.keys():
        d_biases['d_' + bias_name] = np.zeros_like(biases[bias_name])

    for t in range(input_len - 1, -1, -1):

        d_y = np.copy(probability[t])
        #print('dy  ', d_y.shape)
        #print(y[t])
        d_y[y[t]] -= 1

        gradients['d_W_y'] += d_y.dot(hidden_state[t].T)
        d_biases['d_b_y'] += d_y

        d_hidden = parameters['W_y'].T.dot(d_y) + dh_next

        d_out = d_hidden * np.tanh(cell_state[t])
        d_out = dsigmoid(output_gate[t]) * d_out
        gradients['d_W_o'] += d_out.dot(z_state[t].T)
        d_biases['d_b_o'] += d_out

        d_Cell = d_hidden * output_gate[t] * dtanh(np.tanh(cell_state[t]))
        d_Cell += dC_next
        dC_bar = d_Cell * i
        dC_bar *= dtanh(C_bar[t])

        gradients['d_W_C'] += dC_bar.dot(z_state[t].T)
        d_biases['d_b_C'] += dC_bar

        d_i = d_Cell * C_bar[t]
        d_i *= dsigmoid(input_gate[t])

        gradients['d_W_i'] += d_i.dot(z_state[t].T)
        d_biases['d_b_i'] += d_i

        d_f = d_Cell * cell_state[t - 1]
        d_f *= dsigmoid(input_gate[t])

        gradients['d_W_f'] += d_f.dot(z_state[t].T)
        d_biases['d_b_f'] += d_i

        dz = parameters['W_f'].T.dot(biases['b_f']) + parameters['W_i'].T.dot(biases['b_i']) + \
            parameters['W_C'].T.dot(biases['b_C']) + parameters['W_o'].T.dot(biases['b_o'])

        dh_next = dz[: number_of_neuron, :]
        dC_next = forgate_gate[t] * d_Cell


    return gradients, d_biases, hidden_state[input_len - 1], cell_state[input_len - 1]




def update_parameter_adagrad(m_parameters, m_biases, gradients, d_biases):

    global parameters, biases
    epsilon = 1.0e-8

    for p, m, g in zip([parameters['W_f'], parameters['W_i'], parameters['W_C'], parameters['W_o'], parameters['W_y']],
                         [m_parameters['m_W_f'], m_parameters['m_W_i'], m_parameters['m_W_C'], m_parameters['m_W_o'], m_parameters['m_W_y']],
                         [gradients['d_W_f'], gradients['d_W_i'], gradients['d_W_C'], gradients['d_W_o'], gradients['d_W_y']]):
        m += g * g
        p += -learning_rate * g / np.sqrt(m + epsilon)

    for b_p, b_m, b_d in zip([biases['b_f'], biases['b_i'], biases['b_C'], biases['b_o'], biases['b_y']],
                             [m_biases['m_b_f'], m_biases['m_b_i'], m_biases['m_b_C'], m_biases['m_b_o'], m_biases['m_b_y']],
                             [d_biases['d_b_f'], d_biases['d_b_i'], d_biases['d_b_C'], d_biases['d_b_o'], d_biases['d_b_y']]):
        b_m += b_d * b_d
        b_p += -learning_rate * b_d / np.sqrt(b_m + epsilon)

    return m_parameters, m_biases



def lstm_forward_backward(x, y, h_previous, C_previous, m_parameters, m_biases):

    cache, loss = lstm_forward(x, y, h_previous, C_previous)
    gradients, d_biases, hidden_state, cell_state = lstm_backward(y, cache)

    gradients = clip(gradients, 5, -5)
    d_biases = clip(d_biases, 5, -5)
    #print('yes')

    m_parameters, m_biases = update_parameter_adagrad(m_parameters, m_biases, gradients, d_biases)

    return loss, hidden_state, cell_state, gradients, d_biases



def sample(h_previous, C_previous, position, n):

    x = np.zeros((vocabulary_size, 1), dtype=float)
    x[position] = 1

    indices = []
    counter = 0

    #extract weights and biases
    W_f, W_i, W_C, W_o, W_y = parameters['W_f'], parameters['W_i'], parameters['W_C'], parameters['W_o'], parameters['W_y']
    b_f, b_i, b_C, b_o, b_y = biases['b_f'], biases['b_i'], biases['b_C'], biases['b_o'], biases['b_y']

    #print(parameters , "\n")
    #print(biases , "\n")
    while(counter != n):

        z = np.row_stack((C_previous, h_previous))
        z = np.row_stack((z, x))
        forgate_gate = sigmoid(W_f.dot(z) + b_f)
        input_gate = sigmoid(W_i.dot(z) + b_i)
        C_bar = np.tanh(W_C.dot(z) + b_C)

        cell_state = forgate_gate * C_previous + input_gate * C_bar

        z = np.row_stack((cell_state, h_previous))
        z = np.row_stack((z, x))

        output_gate = sigmoid(W_o.dot(z) + b_o)
        hidden_state = output_gate * np.tanh(cell_state)
        #print(hidden_state)
        probability = softmax(W_y.dot(hidden_state) + b_y)
        #print(W_o)
        index = np.random.choice(list(range(vocabulary_size)), p=probability.ravel())

        x = np.zeros((vocabulary_size, 1), dtype=float)
        x[index] = 1
        h_previous = hidden_state
        C_previous = cell_state

        indices.append(index)
        counter += 1


    return indices



def traning_dataset(number_of_iteration):

    m_parameters = {}  # memory parameter for momentum
    m_biases = {}

    global parameters, biases

    for parameters_name in parameters.keys():
        m_parameters['m_' + parameters_name] = np.zeros_like(parameters[parameters_name])

    for bias_name in biases.keys():
        m_biases['m_' + bias_name] = np.zeros_like(biases[bias_name])

    loss = -np.log(1.0 / vocabulary_size) * T_steps

    pointer = 0
    h_previous = np.zeros((number_of_neuron, 1), dtype=float)
    C_previous = np.zeros((number_of_neuron,1), dtype=float)

    for iter in range(number_of_iteration):

        if pointer + T_steps + 1 >= len(input_data) or iter == 0:
            h_previous = np.zeros((number_of_neuron, 1), dtype=float)
            C_previous = np.zeros((number_of_neuron, 1), dtype=float)
            pointer = 0

        x = [char_to_index[ch] for ch in contents[pointer:pointer + T_steps]]
        y = [char_to_index[ch] for ch in contents[pointer + 1:pointer + T_steps + 1]]

        current_loss, h_previous, C_previous, gradients, d_biases = lstm_forward_backward(x, y, h_previous, C_previous,
                                                                               m_parameters, m_biases)

        loss = loss * .999 + current_loss * .001  # smooth loss

        if iter % 200 == 0:

            print('iteration: ', iter)
            print('loss: ', loss)
            print('\n')
            steps = 200
            sampled_indices = sample(h_previous, C_previous, x[0], steps)

            txt = ''.join(index_to_chars[index] for index in sampled_indices)
            print(txt)
            print('\n')

        pointer += T_steps


    return h_previous, C_previous



def generate_chars(h_previous, C_previous, x):

    inputs = np.zeros((vocabulary_size, 1), dtype=float)
    inputs[x[0]] = 1  # one hot vector x for 1st character

    counter = 0
    index = x[0]
    indices = [index]

    # extract weights and biases
    W_f, W_i, W_C, W_o, W_y = parameters['W_f'], parameters['W_i'], parameters['W_C'], parameters['W_o'], parameters[
        'W_y']
    b_f, b_i, b_C, b_o, b_y = biases['b_f'], biases['b_i'], biases['b_C'], biases['b_o'], biases['b_y']

    while (counter < 15 and index != char_to_index['\n']):

        z = np.row_stack((C_previous, h_previous))
        z = np.row_stack((z, x))

        forgate_gate = sigmoid(W_f.dot(z) + b_f)
        input_gate = sigmoid(W_i.dot(z) + b_i)
        C_bar = np.tanh(W_C.dot(z) + b_C)

        cell_state = forgate_gate * C_previous + input_gate * C_bar

        z = np.row_stack((cell_state, h_previous))
        z = np.row_stack((z, x))

        output_gate = sigmoid(W_o.dot(z) + b_o)
        hidden_state = output_gate * np.tanh(cell_state)

        probability = softmax(W_y.dot(hidden_state) + b_y)

        if (counter + 1 < len(x)):
            index = x[counter + 1]

        else:
            index = np.random.choice(list(range(vocabulary_size)), p=probability.ravel())

        indices.append(index)
        # print(index)

        inputs = np.zeros((vocabulary_size, 1))
        inputs[index] = 1

        h_previous = hidden_state
        C_previous = cell_state

        counter += 1

    return indices



if __name__ == '__main__':

    h_0, C_0 = traning_dataset(99999991)

    indices = []

    '''now give a few characters to see 
    what word the model predicts'''

    for i in range(10):

        input_text = input("enter a chunk of sequence : ")
        x = [char_to_index[ch] for ch in input_text]

        indices = generate_chars(h_0, C_0, x)

        txt = ''.join(index_to_chars[index] for index in indices)
        print(txt)
        print('\n')
