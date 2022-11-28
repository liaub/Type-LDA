import random

import skmultiflow
from pandas import DataFrame
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import AGRAWALGenerator, LEDGenerator, SEAGenerator, HyperplaneGenerator

def generateAbruptDriftStream(max_samples, first_func, second_func, random_state, drift_pos, window_len):

    resultList = []


    stream = skmultiflow.data.ConceptDriftStream(stream=AGRAWALGenerator(balance_classes=False, classification_function=first_func,
                                                                perturbation=0.0, random_state=random_state),
                                        drift_stream=AGRAWALGenerator(balance_classes=False, classification_function=second_func,
                                                                      perturbation=0.0, random_state=random_state), position=drift_pos,
                                        width=1, random_state=None, alpha=0.0)

    naive_bayes = NaiveBayes()
    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = max_samples
    while n_samples < max_samples and stream.has_more_samples():
        iter_max_samples = max_samples/window_len
        iter_n_samples = 0
        correct_cnt = 0
        # Train the estimator with the samples provided by the data stream
        while iter_n_samples < iter_max_samples and stream.has_more_samples():
            X, y = stream.next_sample()
            y_pred = naive_bayes.predict(X)
            if y[0] == y_pred[0]:
                correct_cnt += 1
            naive_bayes.partial_fit(X, y)
            iter_n_samples = iter_n_samples+1
            n_samples += 1

        if n_samples==drift_pos+15:
            resultList.append([correct_cnt / iter_n_samples,1])
        else:
            resultList.append([correct_cnt / iter_n_samples,0])
        file_name = "./input/Data/drift-50-50-4800/abrupt/AGRAWALGenerator_"+str(first_func)+\
                    "_"+str(second_func)+"_"+str(random_state)+"_"+str(drift_pos)+".csv"
        DataFrame(resultList).to_csv(file_name)
        print("abrupt processing has been completed:", file_name)




def generateAbruptDriftStream_plus(max_samples, first_func, second_func, first_random_state, second_random_state,
                                   all_random_state, drift_pos, window_len):
    resultList = []

    stream = skmultiflow.data.ConceptDriftStream(
        stream=SEAGenerator(classification_function=first_func, random_state=first_random_state,
                            balance_classes=False, noise_percentage=0.28),
        drift_stream=SEAGenerator(classification_function=second_func, random_state=second_random_state,
                                  balance_classes=False, noise_percentage=0.28), position=drift_pos,
        width=1, random_state=all_random_state, alpha=0.0)

    naive_bayes = NaiveBayes()
    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = max_samples

    while n_samples < max_samples and stream.has_more_samples():
        iter_max_samples = max_samples / window_len
        iter_n_samples = 0
        correct_cnt = 0
        # Train the estimator with the samples provided by the data stream
        while iter_n_samples < iter_max_samples and stream.has_more_samples():
            X, y = stream.next_sample()
            y_pred = naive_bayes.predict(X)
            if y[0] == y_pred[0]:
                correct_cnt += 1
            naive_bayes.partial_fit(X, y)
            iter_n_samples = iter_n_samples + 1
            n_samples += 1

        #Mark drift position
        if n_samples==drift_pos+15:
            resultList.append([correct_cnt / iter_n_samples,1])
        else:
            resultList.append([correct_cnt / iter_n_samples,0])

        # resultList.append(correct_cnt / iter_n_samples)
        file_name = "./input/Data/drift-50-50-4800/abrupt/SEAGenerator_" + str(first_func) + \
                    "_" + str(second_func) + "_" + str(first_random_state) + "_" + \
                    str(second_random_state) + "_" + str(all_random_state) \
                    + "_" + str(drift_pos) + ".csv"
        DataFrame(resultList).to_csv(file_name)
        print("abrupt processing has been completed:", file_name)

def generateGradualDriftStream(max_samples, first_func, second_func,
                              first_random_state, second_random_state,
                              all_random_state, drift_pos, window_len):

    resultList = []

    stream = skmultiflow.data.ConceptDriftStream(stream=SEAGenerator(classification_function = first_func, random_state = first_random_state,
                                                                     balance_classes = False, noise_percentage = 0.28),
                                        drift_stream=SEAGenerator(classification_function = second_func, random_state = second_random_state,
                                                                     balance_classes = False, noise_percentage = 0.28), position=drift_pos,
                                        width=1000, random_state=all_random_state, alpha=0.0)

    naive_bayes = NaiveBayes()
    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = max_samples

    while n_samples < max_samples and stream.has_more_samples():
        iter_max_samples = max_samples/window_len
        iter_n_samples = 0
        correct_cnt = 0
        # Train the estimator with the samples provided by the data stream
        while iter_n_samples < iter_max_samples and stream.has_more_samples():
            X, y = stream.next_sample()
            y_pred = naive_bayes.predict(X)
            if y[0] == y_pred[0]:
                correct_cnt += 1
            naive_bayes.partial_fit(X, y)
            iter_n_samples = iter_n_samples+1
            n_samples += 1

        #Mark drift position
        if n_samples==drift_pos+15:
            resultList.append([correct_cnt / iter_n_samples,1])
        else:
            resultList.append([correct_cnt / iter_n_samples,0])
        # resultList.append(correct_cnt / iter_n_samples)
        file_name = "./input/Data/drift-50-50-4800/gradual/SEAGenerator_"+str(first_func)+\
                    "_"+str(second_func)+"_"+str(first_random_state)+"_"+\
                    str(second_random_state)+"_"+str(all_random_state)\
                    +"_" + str(drift_pos)+ ".csv"
        DataFrame(resultList).to_csv(file_name)
        print("gradual processing has been completed:", file_name)

def generateIncrementalDriftStream(max_samples, random_state, first_mag_change, second_mag_change,
                                   first_sig, sec_sig, drift_pos, window_len):

    resultList = []

    stream = skmultiflow.data.ConceptDriftStream(stream=HyperplaneGenerator(random_state=random_state, n_features=10,
                                                                     n_drift_features=10, mag_change=first_mag_change,
                                                                     noise_percentage=0.05, sigma_percentage=first_sig),
                                        drift_stream=HyperplaneGenerator(random_state=random_state, n_features=10,
                                                                         n_drift_features=10, mag_change=second_mag_change,
                                                                         noise_percentage=0.05, sigma_percentage=sec_sig),
                                                 position=drift_pos, width=1000, random_state=None, alpha=0.0)

    naive_bayes = NaiveBayes()
    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = max_samples

    while n_samples < max_samples and stream.has_more_samples():
        iter_max_samples = max_samples/window_len
        iter_n_samples = 0
        correct_cnt = 0
        # Train the estimator with the samples provided by the data stream
        while iter_n_samples < iter_max_samples and stream.has_more_samples():
            X, y = stream.next_sample()
            y_pred = naive_bayes.predict(X)
            if y[0] == y_pred[0]:
                correct_cnt += 1
            naive_bayes.partial_fit(X, y)
            iter_n_samples = iter_n_samples+1
            n_samples += 1
        # Mark drift position
        if n_samples == drift_pos + 15:
            resultList.append([correct_cnt / iter_n_samples, 1])
        else:
            resultList.append([correct_cnt / iter_n_samples, 0])
        # resultList.append(correct_cnt / iter_n_samples)
        file_name = "./input/Data/drift-50-50-4800/incremental/HyperplaneGenerator_"+str(random_state)+\
                    "_"+str(first_mag_change)+"_"+str(second_mag_change)+"_"\
                    +str(first_sig)+"_"+str(sec_sig)+"_"+str(drift_pos)+".csv"
        DataFrame(resultList).to_csv(file_name)
        print("incremental processing has been completed:", file_name)

def generateNormalStream(max_samples, GeneratorType, random_state, window_len):

    resultList = []

    if GeneratorType == 0:
        stream = AGRAWALGenerator(classification_function=0, random_state=random_state,
                                                   balance_classes=False, perturbation=0.0)

        naive_bayes = NaiveBayes()
        # Setup variables to control loop and track performance
        n_samples = 0
        # samples_of_this_type = max_samples/4

        while n_samples < max_samples and stream.has_more_samples():
            iter_max_samples = max_samples / window_len
            iter_n_samples = 0
            correct_cnt = 0
            # Train the estimator with the samples provided by the data stream
            while iter_n_samples < iter_max_samples and stream.has_more_samples():
                X, y = stream.next_sample()
                y_pred = naive_bayes.predict(X)
                if y[0] == y_pred[0]:
                    correct_cnt += 1
                naive_bayes.partial_fit(X, y)
                iter_n_samples = iter_n_samples + 1
                n_samples += 1

            resultList.append(correct_cnt / iter_n_samples)
            file_name = "./input/Data/drift-50-50-4800/normal/AGRAWALGenerator"+str(random_state)+".csv"
            DataFrame(resultList).to_csv(file_name)
            print("Normal processing has been completed:", file_name)

    if GeneratorType == 1:
        stream = HyperplaneGenerator(random_state=random_state, n_features=10, n_drift_features=2,
                                     mag_change=0.0, noise_percentage=0.05, sigma_percentage=0.1)

        naive_bayes = NaiveBayes()
        # Setup variables to control loop and track performance
        n_samples = 0
        # samples_of_this_type = max_samples / 4

        while n_samples < max_samples and stream.has_more_samples():
            iter_max_samples = max_samples / window_len
            iter_n_samples = 0
            correct_cnt = 0
            # Train the estimator with the samples provided by the data stream
            while iter_n_samples < iter_max_samples and stream.has_more_samples():
                X, y = stream.next_sample()
                y_pred = naive_bayes.predict(X)
                if y[0] == y_pred[0]:
                    correct_cnt += 1
                naive_bayes.partial_fit(X, y)
                iter_n_samples = iter_n_samples + 1
                n_samples += 1

            resultList.append(correct_cnt / iter_n_samples)
            file_name = "./input/Data/drift-50-50-4800/normal/HyperplaneGenerator"+str(random_state)+".csv"
            DataFrame(resultList).to_csv(file_name)
            print("Normal processing has been completed:", file_name)

    if GeneratorType == 2:
        stream = SEAGenerator(classification_function=0, random_state=random_state,
                              balance_classes=False, noise_percentage=0.0)

        naive_bayes = NaiveBayes()
        # Setup variables to control loop and track performance
        n_samples = 0
        # samples_of_this_type = max_samples / 4

        while n_samples < max_samples and stream.has_more_samples():
            iter_max_samples = max_samples / window_len
            iter_n_samples = 0
            correct_cnt = 0
            # Train the estimator with the samples provided by the data stream
            while iter_n_samples < iter_max_samples and stream.has_more_samples():
                X, y = stream.next_sample()
                y_pred = naive_bayes.predict(X)
                if y[0] == y_pred[0]:
                    correct_cnt += 1
                naive_bayes.partial_fit(X, y)
                iter_n_samples = iter_n_samples + 1
                n_samples += 1

            resultList.append(correct_cnt / iter_n_samples)
            file_name = "./input/Data/drift-50-50-4800/normal/SEAGenerator"+str(random_state)+".csv"
            DataFrame(resultList).to_csv(file_name)
            print("Normal processing has been completed:", file_name)

    if GeneratorType == 3:
        stream = LEDGenerator(random_state=random_state, noise_percentage=0.0, has_noise=False)

        naive_bayes = NaiveBayes()
        # Setup variables to control loop and track performance
        n_samples = 0
        # samples_of_this_type = max_samples / 4

        while n_samples < max_samples and stream.has_more_samples():
            iter_max_samples = max_samples / window_len
            iter_n_samples = 0
            correct_cnt = 0
            # Train the estimator with the samples provided by the data stream
            while iter_n_samples < iter_max_samples and stream.has_more_samples():
                X, y = stream.next_sample()
                y_pred = naive_bayes.predict(X)
                if y[0] == y_pred[0]:
                    correct_cnt += 1
                naive_bayes.partial_fit(X, y)
                iter_n_samples = iter_n_samples + 1
                n_samples += 1

            resultList.append(correct_cnt / iter_n_samples)
            file_name = "./input/Data/drift-50-50-4800/normal/LEDGenerator"+str(random_state)+".csv"
            DataFrame(resultList).to_csv(file_name)
            print("Normal processing has been completed:", file_name)


if __name__ == "__main__":

    window_len = 51
    max_samples = 765

    #Position range[5,95]
    position=[]
    for r in  range(50):
        if r%1==0:
            if r>=2 and r <50:
                position.append(r)
    print(len(position))
    #Abrupt Drift (10*9*48)(5*2*48) 4800
    for i in range(10):
        first_func = i
        for j in range(10):
            second_func = j
            if first_func != second_func:
                for pos in range(len(position)):
                    random_state=0
                    drift_pos=position[pos]*15
                    generateAbruptDriftStream(max_samples, first_func, second_func, random_state, drift_pos, window_len)

    pair = [(2, 1), (2, 3), (1, 2), (3, 2), (0, 3)]#（48*2*5）
    for i in pair:
        first_func = i[0]
        second_func = i[1]
        for first_random_state in range(100, 101):#before(100, 110)
            for second_random_state in range(100, 102):#before(100, 107)
                for random_state in range(len(position)):
                    drift_pos = position[random_state]*15
                    all_random_state = None
                    # drift_pos = drift_pos_list[random.randint(1, 5) - 1]
                    generateAbruptDriftStream_plus(max_samples, first_func, second_func, first_random_state,
                                                   second_random_state, all_random_state, drift_pos, window_len)

    #Gradual Drift （48*2*2*5*5）4800
    pair = [(2, 1), (2, 3), (1, 2), (3, 2), (0, 3)]
    for i in pair:
        first_func = i[0]
        second_func = i[1]
        for first_random_state in range(100, 105):#(100, 112)
            for second_random_state in range(100, 102):#(100, 112)
                for random_state in range(len(position)):
                    all_random_state = None
                    drift_pos = position[random_state]*15
                    generateGradualDriftStream(max_samples, first_func, second_func, first_random_state,
                                               second_random_state, all_random_state, drift_pos, window_len)
                    all_random_state = 112
                    drift_pos = position[random_state]*15
                    generateGradualDriftStream(max_samples, first_func, second_func, first_random_state,
                                               second_random_state, all_random_state, drift_pos, window_len)

   # Incremental Drift(48*10*10) 4800
    for i in range(1):
        random_state = i
        for j in range(10):
            first_mag_change = j/10
            first_sig = j/50
            for k in range(10):
                second_mag_change = k/10
                second_sig = k/50
                for pos in range(len(position)):
                    drift_pos = position[pos]*15
                    generateIncrementalDriftStream(max_samples, random_state, first_mag_change, second_mag_change,
                                                   first_sig, second_sig, drift_pos, window_len)

    # Normal（4*1200）4800
    for GeneratorType in range(0,5):
        for random_state in range(1,1201):#before (0,350)
            generateNormalStream(max_samples, GeneratorType, random_state, window_len)
