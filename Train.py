from helper_functions import *
from Model import *
import time
import pickle
import os


def train_network():
    """
        Trains the network in 10-fold cross validation manner
    """

    length = 1024  # length of the signal

    try:  # create directory to save models
        os.makedirs('models')
    except:
        pass

    try:  # create directory to save training history
        os.makedirs('History')
    except:
        pass

        # 10-fold cross validation
    for foldname in range(10):
        print('----------------')
        print('Training Fold {}'.format(foldname + 1))
        print('----------------')
        # loading training data
        dt = pickle.load(open(os.path.join('./data', 'train{}.p'.format(foldname)), 'rb'))
        X_train = dt['X_train']
        Y_train = dt['Y_train']
        # loading validation data
        dt = pickle.load(open(os.path.join('./data', 'val{}.p'.format(foldname)), 'rb'))
        X_val = dt['X_val']
        Y_val = dt['Y_val']

        # loading metadata
        dt = pickle.load(open(os.path.join('./data', 'meta{}.p'.format(foldname)), 'rb'))
        max_ppg = dt['max_ppg']
        min_ppg = dt['min_ppg']
        max_abp = dt['max_abp']
        min_abp = dt['min_abp']

        Y_train = prepareLabel(Y_train)  # prepare labels for training deep supervision

        Y_val = prepareLabel(Y_val)  # prepare labels for training deep supervision

        mdl = MultiResUNetDS(length)  # create approximation network

        # loss = mae, with deep supervision weights
        mdl.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'],
                     loss_weights=[1., 0.9, 0.8, 0.7, 0.6])

        checkpoint_ = ModelCheckpoint(os.path.join('models', '{}_fold{}.h5'.format('MLSU_Net', foldname)),
                                       verbose=1, monitor='val_out_loss', save_best_only=True, mode='auto')
        # train network for 100 epochs
        history = mdl.fit(
            X_train,
            {'out': Y_train['out'], 'level1': Y_train['level1'], 'level2': Y_train['level2'],
             'level3': Y_train['level3'], 'level4': Y_train['level4']},
            epochs=1,
            batch_size=32,
            validation_data=(X_val, {'out': Y_val['out'], 'level1': Y_val['level1'], 'level2': Y_val['level2'],
                                     'level3': Y_val['level3'], 'level4': Y_val['level4']}),
            callbacks=[checkpoint_],
            verbose=1
        )
        history_dict = {
            'loss': history.history['loss'],
            'out_loss': history.history['out_loss'],
            'out_mean_squared_error': history.history['out_mean_squared_error'],
            'val_loss': history.history['val_loss'],
            'val_out_loss': history.history['val_out_loss'],
            'val_out_mean_squared_error': history.history['val_out_mean_squared_error']
        }
        # save training history
        pickle.dump(history_dict,
                    open('History/{}_fold{}.p'.format('MLSU_Net', foldname), 'wb'))


        mdl = None  # garbage collection

        time.sleep(90)


def main():
    train_network()  # train the models for 10-fold


if __name__ == '__main__':
    main()
