# -*- coding: utf-8 -*-
"""
Genetic algorithm (GA) used as a baseline optimizer.

Each generation: evaluate the population with EVAL, keep the best individual (elitism),
build children by uniform crossover from the top-10 parents, then apply gene-wise mutation
(each layer is re-drawn uniformly within its range with probability `mutation`).
`iterate(callback)` calls callback(generation, population, fitness) after each evaluation;
returning True stops the search (used for the solver-validated protocol in run_GA_passK.py).

@author: Gibaek Kim
"""
import numpy as np

class GA:
    def __init__(self, X, X_max, X_min, Y_max, Y_min, t_gain, t_lw, t_wave, t_Ein, t_Eout, EVAL, xfile, DNN = False, total = True, iteration = 100, n_particles = 50, mutation = 0.04, mutation_mode = 'gene', local_step = 3):
        
        self.DNN = DNN
        self.xfile = xfile
        self.total = total
        self.t_gain = t_gain
        self.t_lw = t_lw
        self.t_wave = t_wave
        self.t_Ein = t_Ein
        self.t_Eout = t_Eout

        
        self.iteration = iteration
        self.n_particles = n_particles
        self.mutation = mutation
        # mutation_mode: 'gene'       = after crossover, each gene (layer) is re-drawn uniformly in [min, max] with probability `mutation` (default, used in the paper)
        #                'gene_local' = each gene is shifted by +-1..local_step A with probability `mutation` (clipped to range)
        #                'individual' = the whole child is replaced by a random individual
        self.mutation_mode = mutation_mode
        self.local_step = local_step
        
        self.X = X
        
        self.X_max = X_max
        self.X_min = X_min

        self.Y_max = Y_max
        self.Y_min = Y_min
        
        self.EVAL = EVAL
        self.point = np.zeros(iteration)
        
        self.point_gain = np.zeros(iteration)
        self.point_wave = np.zeros(iteration)
        self.point_LW = np.zeros(iteration)
        self.point_Ein = np.zeros(iteration)
        self.point_Eout = np.zeros(iteration)
        self.str_var = np.zeros(iteration)
        
        
        self.local_point = np.zeros(n_particles)
        self.local_index = np.zeros(n_particles)
        
        self.peak_point = np.zeros(n_particles)
        self.LW_point = np.zeros(n_particles)
        self.wave_point = np.zeros(n_particles)
        self.Ein_point = np.zeros(n_particles)
        self.Eout_point = np.zeros(n_particles)
        
        
        if self.total:
            self.LocalBest_X = np.zeros([iteration,24])
            self.next_X = np.zeros([n_particles,24])
            
            self.LocalBest_X = np.zeros([iteration,24])
            self.next_X = np.zeros([n_particles,24])
        else:
            self.LocalBest_X = np.zeros([iteration,8])
            self.next_X = np.zeros([n_particles,8])
            
            self.LocalBest_X = np.zeros([iteration,8])
            self.next_X = np.zeros([n_particles,8])
            
            
    def iterate(self, callback=None):
        for ii in range(self.iteration):
            
            # ## for comparision
            # np.random.seed(ii)
            
            # print(self.local_point)
            if self.DNN:
                self.local_point, self.peak_point, self.LW_point,self.wave_point, self.Ein_point, self.Eout_point = self.EVAL(self.X, self.X_max, self.X_min, self.Y_max, self.Y_min, xfile = self.xfile,
                                                                             t_gain = self.t_gain, t_LW = self.t_lw,
                                                                             t_wave = self.t_wave, t_Ein = self.t_Ein,
                                                                             t_Eout = self.t_Eout)
            else:
                for jj in range(self.n_particles):
                    self.local_point[jj], self.peak_point[jj], self.LW_point[jj],self.wave_point[jj], self.Ein_point[jj], self.Eout_point[jj] = self.EVAL(self.X[jj,:], t_gain = self.t_gain,
                                                                                             t_LW = self.t_lw, t_wave = self.t_wave,
                                                                                             t_Ein = self.t_Ein,t_Eout = self.t_Eout)
                    
            # Evaluate points and sort it
            self.local_index = np.argsort(self.local_point)[::-1]
            # convert to int type
            # self.local_index = np.asarray(self.lacal_index, dtype = int)
            
            self.LocalBest_X[ii,:] = self.X[self.local_index[0],:]
            self.point[ii] = max(self.local_point)
            self.point_gain[ii] = self.peak_point[np.argmax(self.local_point)]
            self.point_wave[ii] = self.wave_point[np.argmax(self.local_point)]
            self.point_LW[ii] = self.LW_point[np.argmax(self.local_point)]
            self.point_Ein[ii] = self.Ein_point[np.argmax(self.local_point)]
            self.point_Eout[ii] = self.Eout_point[np.argmax(self.local_point)]
            self.str_var[ii] = np.mean(self.X.std(axis = 0))

            # optional hook: callback(gen_index, X, surrogate_points) -> True to stop (X stays = evaluated population)
            if callback is not None and callback(ii, self.X, self.local_point):
                self.n_done = ii + 1
                break
            
            # Generate next generation
            n_gene = 24 if self.total else 8
            for kk in range(self.n_particles):

                if kk == 0:                                  # elitism: the best individual is kept unchanged (no mutation)
                    self.next_X[kk,:] = self.X[self.local_index[0],:]
                    continue

                if self.mutation_mode == 'individual':       # 'individual' mode
                    if np.random.rand() < self.mutation:
                        for jj in range(n_gene):
                            self.next_X[kk,jj] = np.random.randint(self.X_min[jj],self.X_max[jj]+1)
                        continue

                # uniform crossover: each layer is copied from a random parent among the top 10
                for jj in range(n_gene):
                    self.next_X[kk,jj] = self.X[self.local_index[np.random.randint(0,10)],jj]

                # gene-wise mutation
                if self.mutation_mode == 'gene':
                    for jj in range(n_gene):
                        if np.random.rand() < self.mutation:
                            self.next_X[kk,jj] = np.random.randint(self.X_min[jj],self.X_max[jj]+1)
                elif self.mutation_mode == 'gene_local':
                    for jj in range(n_gene):
                        if np.random.rand() < self.mutation:
                            step = np.random.choice([-1,1])*np.random.randint(1,self.local_step+1)
                            self.next_X[kk,jj] = np.clip(self.next_X[kk,jj]+step,self.X_min[jj],self.X_max[jj])

            self.X = self.next_X
            self.Gbest = self.LocalBest_X[np.argmax(self.point),:]
            if ii % 50 == 0:
                print(f"Generation: {ii} Gbest: {max(self.point[:ii+1]):.3f}")
                    