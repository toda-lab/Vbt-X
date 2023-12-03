import random
import math
from z3 import Solver


class XORSampler:
    def __init__(self, smt_str, param_xor, vbtx_ver="improved", no_of_xor=5, p=.5, max_path=100, max_loop=1000, need_only_one_sol=True, need_blocking=True, need_change_s=True, class_list=["Class"], protected_list=["sex"]):
        self.smt_str = smt_str
        self.no_of_xor = no_of_xor
        self.p = p
        self.max_path = max_path
        self.max_loop = max_loop
        self.need_only_one_sol = need_only_one_sol
        self.need_blocking = need_blocking
        self.need_change_s = need_change_s
        self.class_list = class_list
        self.vbtx_ver = vbtx_ver

        not_equal_list = protected_list + class_list
        self.not_equal_list = list()
        for ch in not_equal_list:
            for index in ['', '0', '1']:
                self.not_equal_list.append(ch+index)
        self.protected_list = list()
        for ch in protected_list:
            for index in ['0', '1']:
                self.protected_list.append(ch+index)
        self.blocking_str = ""
        self.res = dict()
        self.samples = list()

        self.smt2_content = param_xor["smt2_content"]
        self.new_var_list = param_xor["new_var_list"]
        self.old_var_list = param_xor["old_var_list"]
        self.dict_old_to_new = param_xor["dict_old_to_new"]

    def create_input_string(self, in_loop_1=True):
        # update the self.smt_str
        smt_str = ""
        smt_str += self.smt2_content["old"]
        smt_str += self.smt2_content["tree"]
        smt_str += self.smt2_content["fairness"]
        smt_str += self.smt2_content["new"]

        if self.vbtx_ver != "naive":
            smt_str += "(assert (> %s %s))\n" % (self.protected_list[0], self.protected_list[1])

        for lines in self.smt2_content["xor"]:
            smt_str += "%s" % lines
        for lines in self.smt2_content["blocking_loop1"]:
            smt_str += "%s" % lines
        if not in_loop_1:
            smt_str += "%s" % self.smt2_content["blocking_loop2"]
        smt_str += self.smt2_content["check"]
        self.smt_str = smt_str

    def analysis_z3Output(self, in_loop_1=True):
        # find a solution of self.smt_str
        solver = Solver()
        solver.from_string(self.smt_str)
        if "unsat" == str(solver.check()):
            # there is no solution
            return False
        elif in_loop_1:
            # save the found solution to self.res
            model = solver.model()
            for item in model:
                self.res[str(item)] = str(model[item])
        return True  # sat

    def have_sol(self):
        self.create_input_string()
        if self.analysis_z3Output():
            self.blocking_str = ""
            for var in self.new_var_list:
                self.blocking_str += " (= " + var + " " + self.res[var] + ")"
            return True
        else:
            return False

    def generate_XOR(self):
        # generate XOR clauses
        self.smt2_content["xor"].clear()
        var_list = self.dict_old_to_new
        for i in range(0, self.no_of_xor):
            xor_str = ""
            if self.vbtx_ver == "naive":
                for var in self.new_var_list:
                    if random.random() > self.p:
                        xor_str += " " + var
            else:
                for var_od in var_list:
                    if var_od not in self.not_equal_list:
                        if random.random() > self.p:
                            no_x = random.randint(0, len(var_list[var_od]) - 1)
                            var = var_list[var_od][no_x]
                            xor_str += " " + var
            if random.random() > 0.5:
                xor_str = xor_str + " true"
            if xor_str != "":
                self.smt2_content["xor"].append("(assert (xor%s))\n" % xor_str)

    def have_another_sol(self):
        self.smt2_content["blocking_loop2"] = "(assert (not (and%s)))\n" % self.blocking_str
        self.create_input_string(in_loop_1=False)
        self.smt2_content["blocking_loop2"] = ""
        return self.analysis_z3Output(in_loop_1=False)

    def generate_simple_ins(self):
        res1 = list()
        res2 = list()
        for ovar in self.old_var_list:
            res1.append(int(self.res[ovar+'0']))
            res2.append(int(self.res[ovar+'1']))
        self.samples.append(res1)
        self.samples.append(res2)

    def add_blocking(self):
        self.smt2_content["blocking_loop1"].append("(assert (not (and%s)))\n" % self.blocking_str)
    
    def clear_data(self):
        self.res = dict()
        self.blocking_str = ""
    
    def sample(self):
        """test cases generation through hashing-based sampling

        The goal here is to sample the solutions of the given SMT formula (i.e., self.smt_str) by hashing-based sampling.
        The generated samples is the test cases we need.
        """
        # check if self.smt_str has any solution
        solver = Solver()
        solver.from_string(self.smt_str)
        if "unsat" == str(solver.check()):
            # if self.smt_str does not have any solution, there is no need for sampling
            return False, []

        satFlag = False
        i = 0
        no_of_path = 0
        if self.need_change_s:
            times = 0
            change_s = True
        else:
            change_s = False

        while i < self.max_loop and no_of_path < self.max_path:
            i += 1
            # randomly generate XOR Clauses
            self.generate_XOR()
            # sample a solution
            satFlag = self.have_sol()

            if satFlag:
                # sat: found a solution
                if change_s:
                    times += 1
                    if times == 5:
                        times = 0
                        self.no_of_xor += 20
                    continue
                if self.need_only_one_sol:
                    # check whether a solution is unique or not
                    if not self.have_another_sol():
                        # the solution is unique
                        pass
                    else:
                        # have found another solution
                        # do not save this solution as a sample
                        continue
                # add the solution to the set of samples (i.e., self.samples)
                self.generate_simple_ins()
                no_of_path += 1
                if self.need_blocking:
                    self.add_blocking()
            else:
                # unsat: there is no solution found
                if change_s:
                    self.no_of_xor = math.floor(self.no_of_xor*0.5)
                    change_s = False
                continue

        self.clear_data()
        return satFlag, self.samples
