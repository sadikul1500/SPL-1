#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:01:07 2019
@author: sadikul
"""


import numpy as np


def clip(gradients, min_value, max_value):
    dWhh, dWxh, dWhy, dbh, dby = gradients['dWhh'], gradients['dWxh'], gradients['dWhy'], gradients['dbh'], gradients[
        'dby']


    for gradient in [dWhh, dWxh, dWhy, dbh, dby]:

        for i in range(gradient.shape[0]):
            for j in range(gradient.shape[1]):
                if gradient[i][j] > max_value:

                    gradient[i][j] = max_value
                elif gradient[i][j] < min_value:
                    gradient[i][j] = min_value

    gradients = {'dWhh': dWhh, 'dWxh': dWxh, 'dWhy': dWhy, 'dbh': dbh, 'dby': dby}

    return gradients


def TANH(z):
    return (np.exp(z) - np.exp(-1 * z)) / (np.exp(z) + np.exp(-1 * z))


def softmax(x):
    e_x = np.exp(x - np.max(x))

    return e_x / np.sum(e_x, axis=0)



def rnn_forward(x, y, h_previous, parameters):
    # retrieve parameters
    Wxh = parameters['Wxh']
    Whh = parameters['Whh']
    Why = parameters['Why']
    by = parameters['by']
    bh = parameters['bh']


    input_one_hot, hidden_state, output_one_hot,expected_one_hot = {}, {}, {}, {}
    hidden_state[-1] = np.copy(h_previous)
    input_one_hot[0] = np.zeros((vocabulary_size, 1), dtype=float)

    loss = 0

    for t in range(len(x)):
        input_one_hot[t] = np.zeros((vocabulary_size, 1), dtype=float)
        input_one_hot[t][x[t]] = 1

        expected_one_hot[t] = np.zeros((vocabulary_size, 1),dtype=float)
        expected_one_hot[t][y[t]] = 1

        hidden_state[t] = TANH(Wxh.dot(input_one_hot[t]) + Whh.dot(hidden_state[t - 1]) + bh)
        output_one_hot[t] = softmax((Why.dot(hidden_state[t]) + by))


        for i in range(vocabulary_size):
            loss += -expected_one_hot[t][i, 0] * np.log(output_one_hot[t][i, 0])


    cache = (input_one_hot, hidden_state, output_one_hot)
    return loss, cache


def rnn_backward(y, parameters, cache):
    # retrieve elements from cache
    input_one_hot, hidden_state, output_one_hot = cache

    dh_next = np.zeros_like(hidden_state[0])

    gradients = {}
    parameters_names = ["Whh", "Wxh", "bh", "Why", "by"]

    for parameters_name in parameters_names:
        gradients['d' + parameters_name] = np.zeros_like(parameters[parameters_name])


    for t in range(len(input_one_hot) -1, -1, -1):
        dy = np.copy(output_one_hot[t])
        dy[y[t]] -= 1           #dl / dy
        gradients['dWhy'] += dy.dot(hidden_state[t].T)   #dl / dWhy
        gradients['dby'] += dy             #dl / dby

        dh = parameters['Why'].T.dot(dy) + dh_next  #dl / dh[t]
        d_tanh = (1 - hidden_state[t] ** 2) * dh

        gradients['dWhh'] += d_tanh.dot(hidden_state[t - 1].T)        #dl / dWhh
        gradients['dWxh'] += d_tanh.dot(input_one_hot[t].T)           #dl / dWxh
        gradients['dbh'] += d_tanh                                    #dl / dbh
        dh_next = parameters['Whh'].T.dot(d_tanh)

    h_previous = hidden_state[len(input_one_hot) - 1]

    return gradients, h_previous


def update_parameters_rmsprop(parameters, gradients, m_parameters):
    # uses rmsprop formula
    decay_rate = .9
    epsilon = 1e-8

    # retieve parameters from 'parameters'

    Why = parameters['Why']
    Wxh = parameters['Wxh']
    Whh = parameters['Whh']
    by = parameters['by']
    bh = parameters['bh']

    # retrieve parameters from 'gradients'
    dWhy = gradients['dWhy']
    dWhh = gradients['dWhh']
    dWxh = gradients['dWxh']
    dby = gradients['dby']
    dbh = gradients['dbh']

    m_Why = m_parameters['m_Why']
    m_Wxh = m_parameters['m_Wxh']
    m_Whh = m_parameters['m_Whh']
    m_by = m_parameters['m_by']
    m_bh = m_parameters['m_bh']

    # now update


    m_Why = decay_rate * m_Why + (1 - decay_rate) * (dWhy ** 2)
    Why -= .01 * dWhy / (np.sqrt(m_Why) + epsilon)

    m_Whh = decay_rate * m_Whh + (1 - decay_rate) * (dWhh ** 2)
    Whh -= .01 * dWhh / (np.sqrt(m_Whh) + epsilon)

    m_Wxh = decay_rate * m_Wxh + (1 - decay_rate) * (dWxh ** 2)
    Wxh -= .01 * dWxh / (np.sqrt(m_Wxh) + epsilon)

    m_by = decay_rate * m_by + (1 - decay_rate) * (dby ** 2)
    by -= .01 * dby / (np.sqrt(m_by) + epsilon)

    m_bh = decay_rate * m_bh + (1 - decay_rate) * (dbh ** 2)
    bh -= .01 * dbh / (np.sqrt(m_bh) + epsilon)

    parameters = {'Whh': Whh, 'Wxh': Wxh, 'Why': Why, 'bh': bh, 'by': by}

    return parameters, m_parameters



def update_parameters_adagrad(parameters, gradients, m_parameters):

    #uses adagrad formula

    epsilon = 1e-8

    #retieve parameters from 'parameters'

    Why = parameters['Why']
    Wxh = parameters['Wxh']
    Whh = parameters['Whh']
    by = parameters['by']
    bh = parameters['bh']

    #retrieve parameters from 'gradients'
    dWhy = gradients['dWhy']
    dWhh = gradients['dWhh']
    dWxh = gradients['dWxh']
    dby = gradients['dby']
    dbh = gradients['dbh']

    m_Why = m_parameters['m_Why']
    m_Wxh = m_parameters['m_Wxh']
    m_Whh = m_parameters['m_Whh']
    m_by = m_parameters['m_by']
    m_bh = m_parameters['m_bh']

    #now update

    m_Why += dWhy ** 2   #dWhy * dWhy
    Why -= learning_rate * dWhy / np.sqrt(m_Why + epsilon)

    m_Whh += dWhh * dWhh
    Whh -= learning_rate * dWhh / np.sqrt(m_Whh + epsilon)


    m_Wxh += dWxh * dWxh
    Wxh -= learning_rate * dWxh / np.sqrt(m_Wxh + epsilon)

    m_by += dby * dby
    by -= learning_rate * dby / np.sqrt(m_by + epsilon)

    m_bh += dbh * dbh
    bh -= learning_rate * dbh / np.sqrt(m_bh + epsilon)

    parameters = {'Whh': Whh, 'Wxh': Wxh, 'Why': Why, 'bh': bh, 'by': by}

    return parameters, m_parameters


def sample(parameters, h_previous, position, n):

    Whh, Wxh, Why, bh, by = parameters['Whh'], parameters['Wxh'], parameters['Why'], parameters['bh'], parameters['by']
    n_h, n_x = Wxh.shape
    vocabulary_size = by.shape[0]

    x = np.zeros((n_x, 1), dtype=float)
    x[position] = 1     # one hot vector x for 1st character

    indices = []       # empty list that contains the list of indices of chars to generate
    counter = 0

    while (counter != n ):

        h = TANH(Whh.dot(h_previous) + Wxh.dot(x) + bh)
        z = Why.dot(h) + by
        y = softmax(z)
        #print(y)

        #np.random.seed(counter + 0)
        #print(type(y), "  ", y.shape," ", y.ravel().shape)
        index = np.random.choice(list(range(vocabulary_size)), p=y.ravel())



        indices.append(index)
        #print(index)

        x = np.zeros((n_x, 1))
        x[index] = 1
        h_previous = h

        counter += 1


    return indices



def rnn_forward_backward(x, y, h_previous, parameters, m_parameters):
    loss, cache = rnn_forward(x, y, h_previous, parameters)

    gradients, h = rnn_backward(y, parameters, cache)

    gradients = clip(gradients, -5, 5)

    #parameters, m_parameters = update_parameters_rmsprop(parameters, gradients, m_parameters)

    parameters, m_parameters = update_parameters_adagrad(parameters, gradients, m_parameters)

    return loss, gradients, h, parameters


def print_sample(index, index_to_chars):
    text = ''.join(index_to_chars[ix] for ix in index)
    print(text)
    print('\n')



def gradient_check(input, y, h_previous, parameters, gradients):

    delta = 1e-5
    # retrieve parameters
    Wxh = parameters['Wxh']
    Whh = parameters['Whh']
    Why = parameters['Why']
    by = parameters['by']
    bh = parameters['bh']

    # retrieve gradients
    dWxh = gradients['dWxh']
    dWhh = gradients['dWhh']
    dWhy = gradients['dWhy']
    dby = gradients['dby']
    dbh = gradients['dbh']

    for x, dx in zip([Wxh, Whh, Why, by, bh], [dWxh, dWhh, dWhy, dby, dbh]):
        for i in range(5):  #check a times if realative error is less than the tolerence value, here a = 5

            position = int(np.random.uniform(0, x.size))
            temp = x.flat[position]
            x.flat[position] = temp + delta
            y1, _ = rnn_forward(input, y, h_previous, parameters)
            x.flat[position] = temp - delta
            y2, _ = rnn_forward(input, y, h_previous, parameters)

            x.flat[position] = temp
            gradient_rnn = dx.flat[position]
            gradient_numerical = (y1 - y2) / (2 * delta)   #gradient = del_y / del_x
            relative_error = abs(gradient_numerical - gradient_rnn) / abs(gradient_numerical + gradient_rnn)

            print("relative error: ", relative_error, '\n')



def train_dataset(data, index_to_chars, char_to_index, number_of_iteration, num_of_neuron, learning_rate, vocabulary_size,
          ):


    n_x = vocabulary_size
    n_h = num_of_neuron
    number_of_rnn_unit = 30

    Wxh = np.random.randn(num_of_neuron, vocabulary_size) * learning_rate   # input vector  to hidden layer
    Whh = np.random.randn(num_of_neuron, num_of_neuron) * learning_rate   # hidden layer to hidden layer
    Why = np.random.randn(vocabulary_size, num_of_neuron) * learning_rate   # hidden layer to output vector
    bh = np.zeros((num_of_neuron, 1), dtype=float)    # hidden layer bias value
    by = np.zeros((vocabulary_size, 1), dtype=float)  # output layer bias value

    parameters = {'Wxh': Wxh, 'Whh': Whh, 'Why': Why, 'bh': bh, 'by': by}

    # memory variables for update_parameters
    m_Wxh = np.zeros_like(Wxh)
    m_Whh = np.zeros_like(Whh)
    m_Why = np.zeros_like(Why)
    m_bh = np.zeros_like(bh)
    m_by = np.zeros_like(by)

    m_parameters = {'m_Wxh': m_Wxh, 'm_Whh': m_Whh, 'm_Why': m_Why, 'm_bh': m_bh, 'm_by': m_by}

    loss = -np.log(1.0 / n_x) * number_of_rnn_unit  # initial loss

    p = 0
    h_previous = np.zeros((n_h, 1), dtype=float)

    #send dataset to train number_of_iteration times

    for iter in range(number_of_iteration):

        if p+number_of_rnn_unit+1 >= len(data) or iter == 0:
            h_previous = np.zeros((n_h, 1), dtype=float)
            p = 0

        x = [char_to_index[ch] for ch in data[p:p + number_of_rnn_unit]]
        y = [char_to_index[ch] for ch in data[p + 1:p + number_of_rnn_unit + 1]]

        current_loss, gradients, h_previous, parameters = rnn_forward_backward(x, y, h_previous, parameters, m_parameters)

        loss = loss * .999 + current_loss * .001  # smooth loss

        if iter % 1000 == 0:

            print('iteration: ', iter)
            print('loss: ', loss)
            print('\n')
            if iter != 0:
                gradient_check(x, y, h_previous, parameters, gradients)
            steps = 200
            sampled_indices = sample(parameters, h_previous, x[0], steps)
            print_sample(sampled_indices, index_to_chars)

        p += number_of_rnn_unit

    return parameters, h_previous


def generate_chars(parameters, h_previous, x):

    Whh, Wxh, Why, bh, by = parameters['Whh'], parameters['Wxh'], parameters['Why'], parameters['bh'], parameters['by']
    n_h, n_x = Wxh.shape
    vocabulary_size = by.shape[0]

    inputs = np.zeros((n_x, 1), dtype=float)
    inputs[x[0]] = 1     # one hot vector x for 1st character

    counter = 0
    index = x[0]
    indices = [index]

    while (counter < 10 and index != char_to_index['\n'] ):

        h = TANH(Whh.dot(h_previous) + Wxh.dot(inputs) + bh)
        z = Why.dot(h) + by
        y = softmax(z)

        if(counter + 1 < len(x)):
            index = x[counter + 1]

        else:
            index = np.random.choice(list(range(vocabulary_size)), p=y.ravel())


        indices.append(index)
        #print(index)

        inputs = np.zeros((n_x, 1))
        inputs[index] = 1
        h_previous = h

        counter += 1


    return indices



if __name__ == '__main__':

    input_data = open('read_it.txt', 'r').read()
    name = input_data.lower()

    chars = []
    chars.append('\n')

    for letter in range(97, 123):
        chars.append(chr(letter))

    vocabulary_size = len(chars)   #vocabulary size =27
    char_to_index = {}
    index_to_chars = {}

    for i in range(vocabulary_size):
        index_to_chars[i] = chars[i]  # (0 : '\n')

    for i, ch in index_to_chars.items():
        char_to_index[ch] = i        # ('\n' : 0)


    num_of_neuron = 100
    learning_rate = .01

    parameters, h_0 = train_dataset(name, index_to_chars, char_to_index, 90001, num_of_neuron, learning_rate,
                       vocabulary_size)

    indices = []

    '''now give a few characters to see 
    what word the model predicts'''
    for i in range(10):
       input_text = input("enter a chunk of sequence : ")
       x = [char_to_index[ch] for ch in input_text]

       indices = generate_chars(parameters, h_0, x)
       print_sample(indices, index_to_chars)
