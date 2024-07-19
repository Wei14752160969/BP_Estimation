from helper_functions import *
from models import UNetDS64, MultiResUNet1D
import os


def predict_test_data():
    """
        Computes the outputs for test data
        and saves them in order to avoid recomputing
    """

    length = 1024               # length of signal

    dt = pickle.load(open(os.path.join(r'data', 'test_phase.p'),'rb'))  # loading test data
    X_test = dt['X_test']
    Y_test = dt['Y_test']


    mdl = MLSU_net(length)
    mdl.load_weights(os.path.join('models', 'MLSU_net_fold0.h5'))       # loading weights

    Y_test_pred = mdl.predict(X_test, verbose=1)                        # predicting abp waveform

    pickle.dump(Y_test_pred, open('test_output.p', 'wb'))               # saving the predictions


def main():
    predict_test_data()     # predicts and stores the outputs of test data to avoid recomputing

if __name__ == '__main__':
    main()