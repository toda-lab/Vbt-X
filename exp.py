from vbtx import *
import multiprocessing
from joblib import load
import pandas as pd
import sys


def exp(dataset_name, model_name, protected_pair, vbtx_ver, deadline, repeat, show_logging):
    protected_name = protected_pair[0]
    protected_param = protected_pair[1]

    # create black box model
    dataset_csv = "GermanCredit" if dataset_name == "Credit" else dataset_name
    data_range = []
    df = pd.read_csv(f"./Datasets/{dataset_csv}.csv")
    number_attr = df.shape[1] - 1
    for i in range(0, number_attr):
        min_ = df.iloc[:, i].min()
        max_ = df.iloc[:, i].max()
        data_range += [[min_, max_]]

    MODEL_ = load(f"FairnessTestCases/{model_name}{dataset_name}.joblib")
    def predict_func(inputs):
        inputs = [[int(item) for item in row] for row in inputs]
        return MODEL_.predict(inputs)
    black_model = BlackBoxModel(data_range, predict_func, feature_list=df.columns.tolist())

    # perform fairness testing
    for _ in range(repeat):
        tester = Tester(black_model, [protected_param], no_train_data_sample=5000, vbtx_ver=vbtx_ver, show_logging=show_logging)
        tester.test(deadline=deadline, label=(f"{vbtx_ver}-{model_name}-{dataset_name}-{protected_name}-{deadline}", _))


def para_exp_main(deadline=1200, repeat=31):
    check_models = ["LogReg",  "NB", "RanForest", "DecTree"]
    dataset_names = ["Adult", "Credit", "Bank"]
    vbtx_ver_list = ["naive", "improveds10", "improved"]
    paras = list()
    for vbtx_ver in vbtx_ver_list:
        for dataset_name in dataset_names:
            protected_list = list()
            if dataset_name == "Adult":
                protected_list = [("sex", 8), ("race", 7), ("age", 0)]
            elif dataset_name == "Bank":
                protected_list = [("age", 0)]
            elif dataset_name == "Credit":
                protected_list = [("sex", 8), ("age", 12)]
            for protected_pair in protected_list:
                for model_name in check_models:
                    paras.append((dataset_name, model_name, protected_pair, vbtx_ver, deadline, repeat, False))
    pool = multiprocessing.Pool(processes=6)
    pool.starmap(exp, paras)
    pool.close()
    pool.join()

def print_usage():
    print("Usage: python exp.py dataset protected_attr model vbtx_ver [runtime] [loop_times]")
    print("The possible values for each parameter are listed below:")
    print("- dataset and protected_attr pairs: (Adult,sex), (Adult,race), (Adult,age), (Credit,sex), (Credit,age) (Bank,age)")
    print("- models: LogReg, NB, RanForest, DecTree")
    print("- vbtx_ver: naive, improveds10, improved")

if __name__ == "__main__":
    # dataset and protected attribute pairs: ("Adult","sex"), ("Adult","race"), ("Adult","age"), ("Credit","sex"), ("Credit","age"), ("Bank","age")
    # models: "LogReg", "NB", "RanForest", "DecTree"
    # vbt versions: "naive", "improveds10", "improved"
    if len(sys.argv) == 2 and sys.argv[1] == "all":
        para_exp_main(1200, 31)
    if len(sys.argv) not in [5, 6, 7]:
        print_usage()
        exit()
    else:
        attr_index = -1
        dataset_name = sys.argv[1]
        protected_attr = sys.argv[2]
        if dataset_name == "Adult":
            if protected_attr == "sex":
                attr_index = 8
            elif protected_attr == "race":
                attr_index = 7
            elif protected_attr == "age":
                attr_index = 0
            else:
                print(f"no protected attribute called {protected_attr}")
                print_usage()
                exit()
        elif dataset_name == "Bank":
            if protected_attr == "age":
                attr_index = 0
            else:
                print(f"no protected attribute called {protected_attr}")
                print_usage()
                exit()
        elif dataset_name == "Credit":
            if protected_attr == "sex":
                attr_index = 8
            elif protected_attr == "age":
                attr_index = 12
            else:
                print(f"no protected attribute called {protected_attr}")
                print_usage()
                exit()
    deadline = int(sys.argv[5]) if 5 < len(sys.argv) else 1200
    repeat = int(sys.argv[6]) if 6 < len(sys.argv) else 31
    exp(dataset_name=sys.argv[1], model_name=sys.argv[3], protected_pair=(sys.argv[2], 8), vbtx_ver=sys.argv[4],
        deadline=deadline, repeat=repeat, show_logging=True)
