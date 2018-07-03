# -*- coding: utf-8 -*-
import numpy as np
import copy
import os
import time
import pandas
import matplotlib.pyplot as plt

class Optimization(object):
    def __init__(self, wind_farm_design, flow_field, wind_farm_constraint, 
                 site_condition, AEP,
                 save_path=None):
        self.wind_farm_design = wind_farm_constraint
        self.flow_field = flow_field
        self.wind_farm_constraint = wind_farm_constraint
        self.site_condition = site_condition
        self.AEP = AEP
        self.save_path = save_path
        
        self.complete_layout_original = wind_farm_design.complete_layout
        self.num_wt = wind_farm_design.num_wt
        
        self.obj_original = self.cal_objective(self.complete_layout_original)
        self.constraint_original_bool = self.cal_constraint_bool(
                self.complete_layout_original)
        self.constraint_original_float = self.cal_constraint_float(
                self.complete_layout_original)
        
        self.complete_layout_optimized = []
        self.obj_optimized = []
        self.evolution_hist = []
        self.maximal_evaluation = []
        self.cpu_time = []
        
        self.i_run = -1
        self.show_info_in_run = True
    
    def cal_objective(self, complete_layout):
        self.flow_field.change_layout(complete_layout)
        self.flow_field.cal_flow_field()
        AEPs = self.AEP.cal_AEP(self.flow_field)
        return AEPs[1]
    
    def cal_constraint_bool(self, complete_layout):
        return self.wind_farm_constraint.check_constraints_bool(
                complete_layout)
        
    def cal_constraint_float(self, complete_layout):
        return self.wind_farm_constraint.check_constraints_float(
                complete_layout)
        
    def random_move_one_wt(self, complete_layout, maximal_step_size):   
        i_wt = np.random.randint(self.num_wt)
        move_size = np.random.random() * maximal_step_size
        move_theta = np.random.random() * np.pi * 2
        
        x_new =  complete_layout[i_wt, 0] + move_size*np.sin(move_theta)
        y_new =  complete_layout[i_wt, 1] + move_size*np.cos(move_theta)  
        
        layout_new = copy.copy(complete_layout)
        layout_new[i_wt, :2] = [x_new, y_new]
        
        return layout_new
        
    def random_search_run(self, maximal_evaluations=1000, 
                          maximal_step_size=1000):
        start = time.clock()
        
        self.i_run += 1
        current_run = self.i_run
        
        evolution_hist = np.zeros((maximal_evaluations, 1))
        complete_layout_current = self.complete_layout_original
        obj_current = self.obj_original
        constraint_current_bool = self.constraint_original_bool
        
        if not constraint_current_bool:
            raise ValueError('The original wind farm design is not feasible!')
            
        evolution_hist[0, 0] = obj_current
        
        for i_step in range(1, maximal_evaluations):
            # Find a feasible new layout by randomly move one turbine
            feasible_flag = False
            while not feasible_flag:
                complete_layout_new = self.random_move_one_wt(
                        complete_layout_current, maximal_step_size)
                feasible_flag = self.cal_constraint_bool(
                        complete_layout_new)
            
            obj_new = self.cal_objective(complete_layout_new)
            
            if obj_new > obj_current:
                complete_layout_current = complete_layout_new
                obj_current = obj_new
            
            evolution_hist[i_step, 0] = obj_current
            
            if self.show_info_in_run:
                print(
                 'Step {0}/{1}: new_obj={2:.2f}; best_obj={3:.2f};'.format(
                         i_step, maximal_evaluations, 
                         obj_new, obj_current) + 
                 ' original_obj={0:.2f}; imp = {1:.2f} %'.format(
                         self.obj_original, 
                         (obj_current/self.obj_original-1)*100))
            
        cpu_time = time.clock()-start
        print('\n')
        print('Random search run - {0} done'.format(current_run))
        print('Total cpu time for this run: {0} seconds'.format(cpu_time))
        
        self.complete_layout_optimized.append(complete_layout_current)
        self.obj_optimized.append(obj_current)
        self.evolution_hist.append(evolution_hist)
        self.cpu_time.append(cpu_time)
        self.maximal_evaluation.append(maximal_evaluations)      
        self.plot_and_save_results(current_run) 
    
    
    def plot_and_save_results(self, current_run, show_plot=True):
        if self.save_path is None:
            current_path = './Opt_run_{0}/'.format(current_run)
        else:
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)
            Opt_run_path = '/Opt_run_{0}/'.format(current_run)
            current_path = self.save_path +Opt_run_path
            
        if not os.path.isdir(current_path):
            os.mkdir(current_path)

        ####################################################################
        # Saving results
        with open(current_path+'general_info.txt', 'w') as f:
            f.write('Run num: {0}\n'.format(current_run) +
                    'Num of wts: {0}\n'.format(self.num_wt)  +    
                    'Original obj: {:.2f}\n'.format(self.obj_original) +
                    'Optimized obj: {:.2f}\n'.format(
                            self.obj_optimized[current_run]) +
                    'Maximal evalutions: {0}\n'.format(
                            self.maximal_evaluation[current_run]) +
                    'Total cpu time: {0} second\n'.format(
                            self.cpu_time[current_run]))
        
        layout_optimized = self.complete_layout_optimized[current_run]
        
        
        np.savetxt(current_path+'original_layout.txt', 
                   self.complete_layout_original)
        np.savetxt(current_path+'optimized_layout.txt', 
                   layout_optimized)
        
        df = pandas.DataFrame(self.complete_layout_original[:, :3])
        df.to_csv(current_path+'original_layout_WAsP.txt', 
                  sep='\t', header=False)   
        
        df = pandas.DataFrame(layout_optimized[:, :3])
        df.to_csv(current_path+'optimized_layout_WAsP.txt', 
                  sep='\t', header=False)  
        
        np.savetxt(current_path+'evolution_history.txt', 
                   self.evolution_hist[current_run])   
        
       
        
        #######################################################################
        # plot Original and optimized WF
        labels = ['Original', 'Optimized']
        layouts = [self.complete_layout_original, 
                   layout_optimized]
        objs = [self.obj_original, self.obj_optimized[current_run]]   # net AEP of WF
        x_min = self.wind_farm_constraint.x_min
        x_max = self.wind_farm_constraint.x_max
        y_min = self.wind_farm_constraint.y_min
        y_max = self.wind_farm_constraint.y_max
        for i in range(2):
            label = labels[i]

            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            im = ax.imshow(np.flipud(
                    self.site_condition.mean_wind_speed.T),
                    extent = [x_min, x_max, y_min, y_max])
            cbar = fig.colorbar(im, shrink=0.5, aspect=5)
            cbar.set_label('mean wind speed (m/s)')
            ax.plot(layouts[i][:, 0], layouts[i][:, 1], 'r.')
            
#            ax.plot(np.array([x_min, x_max, x_max, x_min, x_min]), 
#                     np.array([y_min, y_min, y_max, y_max, y_min]), 'k-')
            buffer_size = 100
            plt.xlim(x_min-buffer_size, x_max+buffer_size)
            plt.ylim(y_min-buffer_size, y_max+buffer_size)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(label + ' wind farm\n'
                          + 'AEP = {:.2f} GWh'.format(objs[i]/1000.))
            plt.tight_layout()
            if show_plot:
                plt.show()
            
            fig.savefig(current_path + label + ' wind farm.pdf')
            
        
        fig = plt.figure()
        plt.plot(self.evolution_hist[current_run][:, 0]/1000.)
        plt.xlabel('Evaluation (-)')
        plt.ylabel('AEP (GWh)')
        plt.title('Evolution history of the random search run\n')
        plt.tight_layout()
        if show_plot:
            plt.show()    
        fig.savefig(current_path + 
                        'Evolution history of the optimization search run.pdf')
                
        
        