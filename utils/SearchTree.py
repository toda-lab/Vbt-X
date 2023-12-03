import numpy as np
import copy


class LogicalFormula:
    signs_to_rev = {'<': '>=', '>': '<=', '<=': '>', '>=': '<', '=': '!='}

    def __init__(self, sign, num, var):
        self.sign     = sign
        self.var     = var
        self.num      = num
        if sign in self.signs_to_rev:
            self.sign_rev = self.signs_to_rev[sign]
        else:
            print("can't find reverse")

    def __eq__(self, other):
        if self.num == other.num and (self.sign == other.sign or self.sign == other.sign_rev):
            return True
        return False

    def __str__(self):
        return self.sign + ' ' + str(self.num)


class Tree2SMT:
    def __init__(self, feature_names=[], class_name="", protected_att=[8], no_of_tree=2, vbtx_ver="improved"):
        self.declare_smt = ""
        self.fairness_constraints = ""
        self.check_smt = "(check-sat)\n(get-model)"
        self.feature_names = feature_names
        self.class_name = class_name
        self.protected_att = protected_att
        self.no_of_tree = no_of_tree
        self.vbtx_ver = vbtx_ver

        # Prepare for XORSampler
        self.not_equal_list = list()
        not_equal_list_temp = [feature_names[index] for index in protected_att] + [class_name]
        for item in not_equal_list_temp:
            for index in ['','0','1']:
                self.not_equal_list.append(item + index)
        self.no_of_new_var = 0
        self.new_var_list = list()
        self.old_var_list = feature_names + [class_name]
        self.dict_old_to_new = dict()
        self.smt2_content = {"old": "", "tree": "", "fairness": "", "new": "", "xor": [], "blocking_loop1": [], "blocking_loop2": "", "check": self.check_smt}
        self.param_xor = {"new_var_list": self.new_var_list, "old_var_list": self.old_var_list, "dict_old_to_new": self.dict_old_to_new, "smt2_content": self.smt2_content, "not_equal_list": self.not_equal_list}

        # Create declare part in smt file
        for i in range(no_of_tree):
            self.declare_smt += f";{i}th attribute\n"
            for feature in self.feature_names:
                self.declare_smt += f"(declare-fun {feature}{i} () Int)\n"
            self.declare_smt += f"(declare-fun {self.class_name}{i} () Int)\n"
        self.smt2_content["old"] = self.declare_smt

        # Create fairness constraints in smt file
        for index, feature in enumerate(self.feature_names):
            temp = ""
            for i in range(self.no_of_tree):
                temp += f" {feature}{i}"
            if index in protected_att:
                self.fairness_constraints += f"(assert (not(= {temp})))\n"
            else:
                self.fairness_constraints += f"(assert (= {temp}))\n"
        temp = ""
        for i in range(self.no_of_tree):
            temp += f" {self.class_name}{i}"
        self.fairness_constraints += f"(assert (not(= {temp})))\n"
        self.smt2_content["fairness"] = "\n" + self.fairness_constraints

    def record_node_basic(self, signal, var_name, number, new_var_log):
            if var_name not in self.dict_old_to_new:
                self.dict_old_to_new[var_name] = list()
            new_log = signal + var_name + number

            exist_flag = True
            for var in self.dict_old_to_new[var_name]:
                if new_log == new_var_log[var]:
                    exist_flag = False
                    break

            if exist_flag: # not exist
                # create new var
                self.no_of_new_var += 1
                new_var = 'x' + str(self.no_of_new_var)
                self.new_var_list.append(new_var)
                self.dict_old_to_new[var_name].append(new_var)
                new_var_log[new_var] = new_log
                self.smt2_content['new'] += f"(declare-fun {new_var} () Bool)\n"
                self.smt2_content['new'] += f"(assert (= {new_var} ({signal} {var_name} {number})))\n"

    def record_node(self, signal, var_name, number, new_var_log):
        if var_name in self.not_equal_list:
            if var_name + "0" not in self.dict_old_to_new:
                self.dict_old_to_new[var_name+"0"] = list()
                self.dict_old_to_new[var_name+"1"] = list()
        else:
            if var_name not in self.dict_old_to_new:
                self.dict_old_to_new[var_name] = list()
            new_log = LogicalFormula(sign=signal, num=number, var=var_name)

            exist_flag = True
            for var in self.dict_old_to_new[var_name]:
                if new_log == new_var_log[var]:
                    exist_flag = False
                    break

            if exist_flag: # not exist
                # create new var
                self.no_of_new_var += 1
                new_var = 'x' + str(self.no_of_new_var)
                self.new_var_list.append(new_var)
                self.dict_old_to_new[var_name].append(new_var)
                new_var_log[new_var] = new_log
                self.smt2_content['new'] += f"(declare-fun {new_var} () Bool)\n"
                self.smt2_content['new'] += f"(assert (= {new_var} ({signal} {var_name+'0'} {number})))\n"

    def reset_smt(self):
        self.no_of_new_var = 0
        self.new_var_list.clear()
        self.dict_old_to_new.clear()
        self.smt2_content["tree"] = ""
        self.smt2_content["new"] = ""

    def get_parm_xor(self):
        return copy.deepcopy(self.param_xor)

    def dt_to_smt(self, DT):
        # traverse the given decision tree (DT), to construct an SMT formula
        tree_ = DT.tree_
        feature = tree_.feature
        all_paths = []

        def recurse(node, path):
            if feature[node] != -2: # find a node
                index = feature[node]
                threshold = tree_.threshold[node]
                attr_name = self.feature_names[index]
                if tree_.children_left[node] != -1:
                    path.append(["<=", attr_name, str(int(threshold))])
                    recurse(tree_.children_left[node], path)
                if tree_.children_right[node] != -1:
                    path.append([">", attr_name, str(int(threshold))])
                    recurse(tree_.children_right[node], path)
            else: # find leaf
                pre_res = np.argmax(tree_.value[node][0])
                all_paths.append(path + [["=", self.class_name, str(pre_res)]])
            if len(path) != 0:
                path.pop()

        recurse(0, [])
        self.reset_smt()
        res = ""
        res += self.declare_smt
        dt_constraints = ""
        for i in range(self.no_of_tree):
            dt_constraints += f";-------------{i}th-number tree constraint-------------\n"
            for path in all_paths:
                path_str = ""
                for items in path[:-1]:
                    temp = copy.deepcopy(items)
                    temp[1] = temp[1] + str(i)
                    path_str += " (" + " ".join(temp) + ")"
                temp = copy.deepcopy(path[-1])
                temp[1] = temp[1] + str(i)
                res_str = " ".join(temp)
                dt_constraints += f"(assert (=> (and{path_str}) ({res_str})))\n"
        self.smt2_content["tree"] += "\n" + dt_constraints
        res += "\n" + dt_constraints
        res += "\n" + self.fairness_constraints
        res += "\n" + self.check_smt

        new_var_list = dict()
        if self.vbtx_ver == "naive":
            for index in ["0", "1"]:
                for path in all_paths:
                    for items in path[:-1]:
                        self.record_node_basic(items[0], items[1]+index, items[2], new_var_list)
        else:
            for path in all_paths:
                for items in path[:-1]:
                    self.record_node(items[0], items[1], items[2], new_var_list)
        return res