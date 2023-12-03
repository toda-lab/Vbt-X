import random
import time
from sklearn.tree import DecisionTreeClassifier
import csv
from utils.XORSampler import XORSampler
from utils.SearchTree import Tree2SMT
import logging


class BlackBoxModel:
    def __init__(self, data_range, predict_func, feature_list):
        self.no_attr = len(data_range)
        self.data_range = data_range # e.g., [[1, 2], [3, 4]]
        self.predict_func = predict_func
        self.feature_list = feature_list

    def predict(self, inputs):
        outputs = self.predict_func(inputs)
        return outputs


class Tester:
    def __init__(self, black_box_model: "BlackBoxModel", protected_list, no_train_data_sample, vbtx_ver="improved", show_logging=False):
        self.black_box_model = black_box_model
        self.tree2smt = Tree2SMT(feature_names=self.black_box_model.feature_list[:-1],
                                 class_name=self.black_box_model.feature_list[-1], protected_att=protected_list, vbtx_ver=vbtx_ver)
        self.train_data = list()
        self.disc_data = list()
        self.test_data = list()
        self.protected_list = [self.black_box_model.feature_list[i] for i in protected_list]
        self.no_train_data_sample = no_train_data_sample
        self.vbtx_ver = vbtx_ver
        self.no_test = 0
        self.no_disc = 0
        if show_logging:
            logging.basicConfig(format="",level=logging.INFO)
        else:
            logging.basicConfig(level=logging.CRITICAL + 1)

    def create_train_data(self, num):
        self.train_data = list()
        black_model = self.black_box_model
        data_range = black_model.data_range
        for _ in range(num):
            temp = list()
            for i in range(black_model.no_attr):
                temp.append(random.randint(data_range[i][0], data_range[i][1]))
            temp.append(int(black_model.predict([temp])))
            self.train_data.append(temp)

    def train_approximate_DT(self):
        X = [item[:-1] for item in self.train_data]
        Y = [item[-1] for item in self.train_data]
        clf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                    random_state=None)
        return clf.fit(X, Y)

    def check_disc(self, testdata):
        """Execute test cases against black box model

        For each test cases, check whether the test case are discriminatory instances of black box model (self.black_box_model),
        if so, add the test case to the set of discriminatory instances (self.disc_data),
        if not, add the failing parts to the training data (self.train_data)

        Args:
            testdata (list): the set of test cases

        Returns:
            int: the number of test cases that are added to training data (self.train_data)
        """
        no_test = len(testdata) // 2
        X = [item[:-1] for item in testdata]
        Y = [int(item[-1]) for item in testdata]
        real_Y = self.black_box_model.predict(X)
        count = 0
        train_data_count = 0
        for i in range(0, no_test * 2, 2):
            equal1 = True if Y[i] == real_Y[i] else False
            equal2 = True if Y[i + 1] == real_Y[i + 1] else False
            if not equal1:
                self.train_data.append(X[i] + [real_Y[i]])
                train_data_count += 1
            if not equal2:
                self.train_data.append(X[i + 1] + [real_Y[i + 1]])
                train_data_count += 1

            if equal1 and equal2:
                self.disc_data.append(testdata[i])
                self.disc_data.append(testdata[i + 1])
                self.no_disc += 1
                count += 1
        return train_data_count

    def test(self, deadline=None, max_test_data=None, label=("res", 0)):
        """run a fairness test

        perform a fairness testing for black box model (self.black_box_model) against protected attributes (self.protected_list)

        Args:
            deadline (int): the runtime in seconds
            max_test_data (int): when the number of test cases reaches this specified value, the test is terminated
            label (tuple): related to the filename of the results

        Returns:
            the detected discriminatory instances and generated test cases will be saved to ./DiscData and ./TestData
        """
        start_time = time.time()
        restart_flag = True
        self.no_test = 0
        no_new_train_count = 0
        loop = 0

        # main loop of VBT-X
        logging.info(f"Starting fairness test -- {label[0]}")
        while True:
            loop += 1
            if (deadline is not None) and (time.time() - start_time >= deadline):
                break
            if (max_test_data is not None) and (self.no_test >= max_test_data):
                break

            # Step1: make an approximation model DT of black box model
            if restart_flag:
                self.create_train_data(self.no_train_data_sample)
                restart_flag = False
            DT = self.train_approximate_DT()

            # Step2: Construct an SMT formula from DT
            smt_str = self.tree2smt.dt_to_smt(DT)
            param_xor = self.tree2smt.get_parm_xor()

            # Step3: Generate test cases by SMT solver and hashing-based sampling
            if self.vbtx_ver == "improved":
                sampler = XORSampler(smt_str=smt_str, param_xor=param_xor, max_loop=1000, max_path=50, no_of_xor=5,
                                     need_only_one_sol=False, need_change_s=True, need_blocking=False,
                                     class_list=[self.black_box_model.feature_list[-1]], protected_list=self.protected_list)
            elif self.vbtx_ver == "improveds10":
                sampler = XORSampler(smt_str=smt_str, param_xor=param_xor, max_loop=1000, max_path=50, no_of_xor=10,
                                     need_only_one_sol=False, need_change_s=False, need_blocking=False,
                                     class_list=[self.black_box_model.feature_list[-1]], protected_list=self.protected_list)
            elif self.vbtx_ver == "naive":
                sampler = XORSampler(smt_str=smt_str, param_xor=param_xor, vbtx_ver="naive", max_loop=1000, max_path=50, no_of_xor=10,
                                     need_only_one_sol=False, need_change_s=False, need_blocking=False,
                                     class_list=[self.black_box_model.feature_list[-1]], protected_list=self.protected_list)
            else:
                print("No such version of vbt-x")
                exit()
            satFlag, test_data = sampler.sample()

            if satFlag:
                # if at least one test cases is found
                # Step4: Execute test cases against black box model,
                # and Step5: Update the training dataset
                self.no_test += len(test_data) // 2
                self.test_data += test_data
                if self.check_disc(test_data) == 0:
                    no_new_train_count += 1
                    if no_new_train_count >= 5:
                        restart_flag = True
                        no_new_train_count = 0
                else:
                    no_new_train_count = 0
            else:
                # if no test cases can be found from the decision tree, then restart the loop
                restart_flag = True
                logging.info(f"Restarting due to not finding any test cases in this loop")
            logging.info(f"Loop {loop}: #Disc={len(self.disc_data)//2}, #Test={self.no_test}")

        # save the results of detected discriminatory instances and generated test cases
        logging.info(f"The fairness test is completed")
        logging.info(f"Saving the generated test cases to TestData/{label[0]}-{label[1]}.csv")
        with open(f'TestData/{label[0]}-{label[1]}.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(self.test_data)
        logging.info(f"Saving the detected discriminatory instances to DiscData/{label[0]}-{label[1]}.csv")
        with open(f'DiscData/{label[0]}-{label[1]}.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(self.disc_data)
        logging.info(f"Finished")
