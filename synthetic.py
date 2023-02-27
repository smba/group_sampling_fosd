#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import itertools
import json
import matplotlib.pyplot as plt
import random

from sampling import GroupSampler

def flatten(data):
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data

class OracleGenerator:
    def __init__(self, dimacs_file, seed: int = 1):
        
        np.random.seed(seed)
        
        self.n_options = self.__partial_parse(dimacs_file)
        self.sampler = GroupSampler(dimacs_file)
        
    def generate(
            self,
            term_frequency: float,
            interaction_p: float,
            feature_hierarchy: float,
        ):

        # sample options which can be used at all
        inf_options = set(list(range(1, self.n_options + 1)))

        terms = set([])
        
        # How many terms to generate
        # term frequency 0.01 - 2
        n_terms = int(term_frequency * self.n_options)
        
        # sample degrees of terms
        degrees = sorted(
            np.random.geometric(interaction_p, size=n_terms)    
        )

        terms = set([])
        
        degree_limit = len(list(filter(lambda x: x == 1, degrees)))
        
        for degree in degrees:
            new = self.suggest_new_term(degree, terms, feature_hierarchy, inf_options, degree_limit)
            if new is not None:
                terms.add(new)
                
            inf_options = inf_options - set(list(new))
        
        # fill terms with actual influences
        influences = self.__generate_influences(terms)
        
        # make json
        result = {
            "n_options": self.n_options,
            "feature_hierarchy": feature_hierarchy,
            "interaction_p": interaction_p,
            "term_frequency": feature_hierarchy,
            "terms": influences
        }
        
        return result

    def __partial_parse(self, dimacs_file):
        
        with open(dimacs_file) as dmf:
            lines = dmf.readlines()
            lines = list(filter(lambda l: l.startswith("c "), lines))
            lines = [l.split(" ")[1:] for l in lines]
            lines = [[int(l[0]), l[1].strip()] for l in lines]
            self.mapping = dict(lines)
            
        return len(lines)
            

    def __generate_influences(self, terms, base: float = 1000, a = 0.6):

        terms = [[int(o) for o in term] for term in terms]
        terms = [[self.mapping[o] for o in term] for term in terms]
        influences = np.random.power(a, size=len(terms)) * base
        influences *= np.random.choice([-1, 1], size=len(terms))
        influences = np.sort(influences)[::-1]

        result = [
            {
                "options": list(terms[i]),
                "influence": round(influences[i], 3)
            } for i in range(len(terms))
        ]
        
        return(result)
            
    
    def __validate_term(self, options):
        options = [self.mapping[o] for o in options]
        en, dis = self.sampler.sample(options, size=1, diversity=False)

        return len(en) > 0


    def suggest_new_term(self, degree, terms, hierarchy, options, degree_limit):

        term_options = []
        for term in terms:
            term_options += list(term)
        term_options = set(term_options)
        
        while True:
            
            if degree == 1:
                term = tuple(np.random.choice(list(options), size=1))
            else:
                if np.random.random() < hierarchy: # reuse
                    #print("bottom-up interaction")    
                    new_term = set([]) 
                    
                    # select random term from terms
                    while len(new_term) < degree:
                        term = list(terms)[np.random.choice(len(terms), size=1)[0]]
                        for i in term:
                            if len(new_term) < degree:
                                new_term.add(i)
                        
                    new_term = sorted(list(new_term))
                    term = tuple(new_term)
                    
                else: # do not reuse
                    new_term = np.random.choice(list(options), size=degree, replace=False)
                    term = tuple(new_term)
            
            # terminate if term is valid (can be sampled)
            if self.__validate_term(term):
                return term
                
class Oracle:
    
    def __init__(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
            self.terms = data["terms"]
    
    def perfs(self, configs):
        perfs = []
        for i in range(configs.shape[0]):
            perf = self.perf(configs.iloc[i])
            perfs.append(perf)
        
        return np.array(perfs)
    
    def perf(self, config: pd.DataFrame):
        perf = 0.0
        for term in self.terms:
            options = term["options"]
            if np.all(config[options]):
                perf += term["influence"]
                
        return perf

if __name__ == "__main__":
    d = OracleGenerator("dimacs.dimacs", seed=2)
    terms = d.generate(
        term_frequency = 0.05,  
        interaction_p = 0.6, 
        feature_hierarchy = 1.0
    )
    
    print(terms)
    
    with open("file.json", "w+") as f:
        txt = json.dumps(terms, indent=4)
        f.write(txt)
        
    #sampler = GroupSampler("dimacs.dimacs")
    #en, dis = sampler.sample(["EJECT"], size=3)

    
    #orac = Oracle("file.json")
    #en = orac.perfs(en)
    #dis = orac.perfs(dis)
    #plt.hist(en - dis)
