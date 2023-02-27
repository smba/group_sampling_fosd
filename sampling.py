#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import z3
import time 
from z3 import *
import random
import matplotlib.pyplot as plt
import pandas as pd

z3.set_option('auto_config', False)
z3.set_option('smt.phase_selection',5)

class GroupSampler:
    
    def __init__(self, fm_file: str):
        
        # get mapping
        with open(fm_file) as f:
            lines = f.readlines()
                    
            lines = filter(lambda l: l.startswith("c"), lines)
            lines = map(lambda l: l.split(" "), lines)
            lines = {l[2].strip(): int(l[1]) for l in lines}
            self.mapping = lines
            self.features = list(self.mapping.keys())
            
            
        self.solver = z3.Solver()
        
        self.solver.from_file(fm_file)
        self.solver.push()
        
    def get_features(self):
        return np.array(sorted(self.features))
        
    def __fid(self, fname):
        return self.mapping[fname]

    def __add_group_constraints(self, group):
        # add AND(o1, o2, ...)
        self.solver.add(
              z3.And([z3.Bool(f"k!{self.__fid(o)}") for o in group])
        )
        
        # add NOT(AND(p1, p2, p3, ...))
        self.solver.add(
            z3.Not(z3.And([
                z3.Bool(o) for o in group    
            ]))      
        )
        
        # -> add AND(XOR(o1, p1) for each literal]
        self.solver.add(z3.And(
            [z3.Xor(z3.Bool(f"k!{self.__fid(o)}"), z3.Bool(o)) for o in group]
        ))
        
    def sample(self, group, size: int = 1, diversity=True):
        
        self.__add_group_constraints(group)
        self.solver.push()
  
        solutions = []
        duplicate_constraints = []
        
        for i in range(size):
            # add constraints for this round:
            # previous solutions
            for sol in duplicate_constraints:
                self.solver.add(sol)
            
            # distance constraints solutions
            if diversity:
                distance = np.random.randint(len(self.mapping))
                self.solver.add([
                    z3.Sum([z3.If(z3.Bool(f"k!{self.__fid(o)}"), 1, 0) for o in self.get_features()]) < distance
                ])

            if self.solver.check() == z3.sat:

                model = self.solver.model()
                features = self.get_features()
                
                # get current solution
                solution = [
                    bool(model.evaluate(z3.Bool(f"k!{self.__fid(o)}"), model_completion=True)) for o in features
                ]
                solutions.append(solution)

                # duplicate constraint for all non-group options
                noptions = set(self.features) - set(group)
                ncopy = z3.Or([
                    z3.Bool(f"k!{self.__fid(o)}") != model.evaluate(z3.Bool(f"k!{self.__fid(o)}"), model_completion=True) for o in noptions 
                ])
                duplicate_constraints.append(ncopy)

            self.solver.pop()
            self.solver.push()
                
        if len(solutions) > 0:
            solutions = np.vstack(solutions)
            solutions = pd.DataFrame(solutions, columns=self.get_features())
            solutions = solutions.drop(columns=group)
            
            enabled, disabled = solutions.copy(), solutions.copy()
            enabled[group] = True
            disabled[group] = False
            
            
            return enabled, disabled
        else:
            return [], []
            
            
if __name__ == "__main__":
    print(2)

    sampler = GroupSampler("bb.dimacs")
    
    options = sampler.get_features()
    group = options[19:79]
    
    sample = sampler.sample(group, 60)
    print(sample.shape)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].pcolormesh(sample, cmap="Blues_r")
    ax[1].plot(sample.sum(axis=0).values, color="black", linewidth=1)
    
    plt.show()
    
    """
        #print(group)
        # create scope point
        s.push()
        
        # create group constraints
        enabled = z3.And([z3.Bool(f"k!{o}") == True for o in group])
        s.add(enabled)
        print(i)
        if s.check() == z3.sat:
            m = s.model()
     
            # identical options
            copy_options = set(options) - set(group)
            copy = [o == v for o, v in [(z3.Bool(f"k!{mapping[o]}"), m.evaluate(z3.Bool(f"k!{mapping[o]}"), model_completion=True)) for o in copy_options]]
            
            # differing
            non_copy = [o != v for o, v in [(z3.Bool(f"k!{mapping[o]}"), m.evaluate(z3.Bool(f"k!{mapping[o]}"), model_completion=True)) for o in group]]
            
            s.pop()# discard enabled constraints
            s.push() # new scope
            
            s.add(copy)
            s.add(non_copy)
            
            if s.check() != sat:
                print("companion satisfiable")
            else:
                print("companion is unsat")
                
            s.pop()
            
        else:
            print("core unsat", group)
            # drop scope      
            s.pop()
            
        """    
