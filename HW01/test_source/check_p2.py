import argparse
from pathlib import Path
import numpy as np
from checker_utils import catch_exceptions, send_msg
from code.augmented_logistic_regression import AugmentedLogisticRegression as LR

EVAL     = True
DATA_DIR = Path('data')
TEST_DIR = Path('solution-data')
        
def create_test_data():
    #Set RNG Seed
    seed = int(input("Enter a random seed: "))
    np.random.seed(seed)
    
    #Load data sets
    train_X = np.load(str(DATA_DIR / 'q2_train_X.npy'))
    train_y = np.load(str(DATA_DIR / 'q2_train_y.npy'))
    test_X = np.load(str(DATA_DIR / 'q2_test_X.npy'))
    test_y = np.load(str(DATA_DIR / 'q2_test_y.npy'))

    #Select a random subset of training data 
    ind = np.random.permutation(np.arange(train_y.shape[0]))
    train_X = train_X[ind[:100],:]
    train_y = train_y[ind[:100]]
    
    #Save data set
    np.save(str(TEST_DIR / 'q2_train_X.npy'),train_X )
    np.save(str(TEST_DIR / 'q2_train_y.npy'),train_y)
    np.save(str(TEST_DIR / 'q2_test_X.npy'),test_X )
    np.save(str(TEST_DIR / 'q2_test_y.npy'),test_y)        
    
    #Creste a random parameter vector
    w_init = np.random.rand(test_X.shape[1])/5.0
    c_init = np.random.rand(test_X.shape[1])/5.0
    b_init = np.random.rand()
    
    #Save parameter set
    np.save(str(TEST_DIR / 'q2_w_init.npy'),w_init)
    np.save(str(TEST_DIR / 'q2_c_init.npy'),c_init)
    np.save(str(TEST_DIR / 'q2_b_init.npy'),b_init)   
        
        
def load_data():
    #Load data sets
    train_X = np.load(str(TEST_DIR / 'q2_train_X.npy'))
    train_y = np.load(str(TEST_DIR / 'q2_train_y.npy'))
    test_X = np.load(str(TEST_DIR / 'q2_test_X.npy'))
    test_y = np.load(str(TEST_DIR / 'q2_test_y.npy'))

    #Load parameters
    w_init = np.load(str(TEST_DIR / 'q2_w_init.npy'))
    c_init = np.load(str(TEST_DIR / 'q2_c_init.npy'))
    b_init = np.load(str(TEST_DIR / 'q2_b_init.npy'))
        
    return train_X, train_y, test_X, test_y, w_init, c_init, b_init

@catch_exceptions
def check_set_get():
    """[Q2] Check the correctness of set() get() loop"""
    train_X, train_y, test_X, test_y, w, c, b = load_data()
    
    lr = LR(1e-6)
    lr.set_params(w,c,b)
    w2,c2,b2 = lr.get_params()
 
    msg = []
    if not np.allclose(w, w2, rtol=1e-10, atol=1e-10):
        msg.append('w set-get loop is incorrect')
    if not np.allclose(c, c2, rtol=1e-10, atol=1e-10):
        msg.append('c set-get loop is incorrect')            
    if not np.allclose(b, b2, rtol=1e-10, atol=1e-10):
        msg.append('b set-get loop is incorrect')
    send_msg('pass' if msg == [] else ', '.join(msg))   


@catch_exceptions
def check_objective():
    """[Q2] Check the correctness of objective()"""
    train_X, train_y, test_X, test_y, w, c, b = load_data()
    wcb = np.hstack((w,c,[b]))
    
    lr = LR(1e-6)
    ans = lr.objective(wcb, train_X, train_y)
    if(EVAL):    
        sol = np.load(str(TEST_DIR / 'q2_obj.npy'))
        send_msg('pass' if np.allclose(ans, sol, rtol=1e-04, atol=1e-04) else 'incorrect objective')
    else:
        np.save(str(TEST_DIR / 'q2_obj.npy'), ans)
        send_msg('pass')

@catch_exceptions
def check_gradient():
    """[Q2] Check the correctness of objective_grad()"""
    train_X, train_y, test_X, test_y, w, c, b = load_data()
    wcb = np.hstack((w,c,[b]))
    
    lr = LR(1e-6)
    ans = lr.objective_grad(wcb, train_X, train_y)
    if(EVAL): 
        sol = np.load(str(TEST_DIR / 'q2_grad.npy'))
        send_msg('pass' if np.allclose(ans, sol, rtol=1e-04, atol=1e-04) else 'incorrect gradient')
    else:
        np.save(str(TEST_DIR / 'q2_grad.npy'), ans)
        send_msg('pass')

@catch_exceptions
def check_fit():
    """[Q2] Check the correctness of fit()"""
    train_X, train_y, test_X, test_y, w, c, b = load_data()
    lr = LR(1e-6)
    lr.fit(train_X, train_y)
    lr_w, lr_c, lr_b = lr.get_params()
    
    if(EVAL): 
        lr_sol_w = np.load(str(TEST_DIR / 'q2_w.npy'))
        lr_sol_c = np.load(str(TEST_DIR / 'q2_c.npy'))
        lr_sol_b = np.load(str(TEST_DIR / 'q2_b.npy'))
        
        msg = []
        if not np.allclose(lr_w, lr_sol_w, rtol=1e-5, atol=1e-3):
            msg.append('w returned by fit is incorrect')
        if not np.allclose(lr_c, lr_sol_c, rtol=1e-5, atol=1e-3):
            msg.append('c returned by fit is incorrect')            
        if not np.allclose(lr_b, lr_sol_b, rtol=1e-5, atol=1e-3):
            msg.append('b returned by fit is incorrect')
        send_msg('pass' if msg == [] else ', '.join(msg))
    else:
        np.save(str(TEST_DIR / 'q2_w.npy'),lr_w)
        np.save(str(TEST_DIR / 'q2_c.npy'),lr_c)
        np.save(str(TEST_DIR / 'q2_b.npy'),lr_b)
        send_msg('pass')        


@catch_exceptions
def check_predict():
    """[Q2] Check the correctness of predict()"""
    train_X, train_y, test_X, test_y, w, c, b = load_data()

    lr = LR(1e-6)
    lr.set_params(w, c, b)
    lr_ans = lr.predict(test_X)

    if(EVAL):     
        lr_sol = np.load(str(TEST_DIR / 'q2_predict.npy'))
        send_msg('pass' if np.allclose(lr_ans, lr_sol) else 'incorrect predict')
    else:
        np.save(str(TEST_DIR / 'q2_predict.npy'), lr_ans)
        send_msg('pass')


def main():
    global EVAL
    
    np.random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target')
    parser.add_argument('-m', '--mode')
    args = parser.parse_args()
    target = args.target
    mode   = args.mode
    
    #If in eval mode, run a test
    #Else run setup for all tests
    if(mode=="eval"):
        EVAL=True
        #Run a test
        if target == 'objective':
            check_objective()
        elif target == 'gradient':
            check_gradient()
        elif target == 'fit':
            check_fit()
        elif target == 'predict':
            check_predict()
        elif target == 'setget':    
            check_set_get()
        elif target is None:
            check_set_get()
            check_objective()
            check_gradient()
            check_fit()
            check_predict()
            
    elif(mode=="setup"):
        EVAL=False
        import os
        if not os.path.exists(TEST_DIR):
            os.makedirs(TEST_DIR)
        create_test_data()
        
        check_set_get()
        check_objective()
        check_gradient()
        check_fit()
        check_predict()

if __name__ == '__main__':
    main()
