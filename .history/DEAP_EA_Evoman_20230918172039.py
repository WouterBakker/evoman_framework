
#General packages
import numpy as np
import pandas as pd
import random as random

# imports framework
import sys, os
from evoman.environment import Environment

from demo_controller import player_controller


#DEAP

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

toolbox = base.Toolbox()


###### Initialization parameters #########

experiment_name = "CompareMuLambdaAlgorithms"
single_run_name = "test"
n_hidden_neurons = 10

####Booleans
train_single = True
default_fitness = True
cambrian = False #changes to random enemy every x turns, for y turns

#No video
Headless = False
## In case of testing the best solutions:
test_the_best = True
#Test most recent version, otherwise test version experiment_run
most_recent = True
#Test x times
test_times = 10


#Lower/upper bound for output value
low = -1
high = 1


###Parameters
generations = 50

#Starting population size
npop = 50

#Tournament size
tourn_size = 2

#Mu: population size, (in case of VarOr)-Lambda: number of offspring
mu = 10
lambda_ = 50 #Evaluates all the offspring, then selects mu individuals

#probability of two individuals mating, mutating
p_mate = 0.55
p_mutate_sel = 0.30 #Chance of being selected for mutation
p_mutate = 1/(1.5) #Chance of a genome being mutated

#Enemy number = [1:8]
#test single enemy
enemy = 2
#Test multiple enemies
enemies = [1,2,3]


#Number of times to repeat the algorithm (10 times for this assignment)
rep = 10


#Convenience variables
experiment_run = 1

single_run_name += str(experiment_run)
experiment_name += str(experiment_run)


########################### 


#Directory setup

while train_single == True and os.path.exists("single_run/" + single_run_name + "_best.txt"):
    
    experiment_run += 1
    single_run_name = single_run_name[:-len([experiment_run-1])] + str(experiment_run)

    if test_the_best == True:
        if most_recent == True:
            print("Finding most recent best solution...")
            
            if not os.path.exists("single_run/" + single_run_name + "_best.txt"):
                single_run_name = single_run_name[:-len([experiment_run-1])] + str(experiment_run-1)
                print("Testing " + single_run_name)
        break
        
    else:
        print(single_run_name + " taken...")
        print("...trying new path: " + single_run_name)


if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    
if not os.path.exists("single_run"):
    os.makedirs("single_run")
    
    
if train_single == True and not os.path.exists("single_run/data_parameters.csv"):
    df_para = pd.DataFrame(columns = ["experiment", "generations", "mu", "lambda", "p_mate", "p_mutate", "enemy", "pop_start", "neurons"])
    df_para.to_csv("single_run/data_parameters.csv", index = False)
    print("Parameter file created")
    
else:
    df_para = pd.read_csv("single_run/data_parameters.csv")
    
parameters = [single_run_name, generations, mu, lambda_, p_mate, p_mutate, enemy, npop, n_hidden_neurons]
parameters = pd.DataFrame(parameters).T
parameters.columns = df_para.columns


#To watch the game
if Headless == True:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    
# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name= experiment_name,
                  #enemies=n_enemy,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  logs="off",
                  savelogs = "no",
                  randomini = "no",
                  multiplemode = "no")
          
        
######## Fitness calculation

# Simulates a game for individual x
def simulation(env,individual):
    fitness,health_player,health_enemy,game_time = env.play(pcont= np.array(individual))
    
    #Default
    #fitness = 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - numpy.log(self.get_time())
    #fitness = 0.9*(100 - health_enemy) + 0.1 * health_player - np.log(game_time)

    
    if default_fitness == False:
    #print("contrib1:" + str(0.1 * health_player - 0.9 * health_enemy))
    #print("contrib2:" + str(((health_player == 0) - (health_enemy == 0)) * np.log(game_time)))
        fitness = 0.1 * health_player - 0.9 * health_enemy + (health_player == 0) * np.log(game_time) + (health_enemy == 0) * (np.log(game_time) + 15)
    
    return fitness



# Runs a simulated game for each individual, calculates fitness
def evaluate(individual):
    return [simulation(env,individual)]

#################
#Algorithms

def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__,
                    cambrian = False):

    
    
    assert lambda_ >= mu, "lambda must be greater or equal to mu."
    

    # Evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
        

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'rolling_avg'] + (stats.fields if stats else []) 

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(population), poep=0, **record)
    if verbose:
        print(logbook.stream)
        
        
    
    fixed_enemy = [env.enemyn]
    enemies = list(range(1,9))
    enemies.remove(fixed_enemy[0])
    i, j = 0,1
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        
        
        if cambrian:
            i+=1
            #Every i generations, changes the environment to play a random enemy
            if i % 30 == 0:
                
                random_enemy = random.sample(enemies, 1)
                env.update_parameter('enemies', random_enemy)
                j+=1
                print("Random enemy selected: " + str(random_enemy))
            
    
            
            #After playing j-2 times, resets to the old enemy
            if j % 8 == 0:
                env.update_parameter('enemies', fixed_enemy)
                random_enemy = fixed_enemy
                i,j = 0,1
                print("Enemy reset")
                
            if j > 1:
                j+=1
        
            
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        #offspring = algorithms.varAnd(population,   toolbox, cxpb, mutpb)

        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
            
            
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)
        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        rolling_average = np.mean(logbook.select("avg")[-10:])
        logbook.record(gen=gen, nevals=len(offspring), rolling_avg = rolling_average, **record)
        if verbose:
            print(logbook.stream)
            
    print("Algorithm finished")
    return population, logbook

  
#Individual size. ----Note: 20 sensors in total
# Number of weights for multilayer with x hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

#DEAP init

#Initial population generation
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("indices", np.random.uniform, low = low, high = high, size = n_vars)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)


# make a random number of individuals by calling toolbox.population()
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n = npop)

#Add evaluate function to the DEAP toolbox, calculates fitness
toolbox.register("evaluate", evaluate)
#Mate/crossover parameters
toolbox.register("mate", tools.cxUniform, indpb = 0.5)
#Mutation: generate variation in individual's 'DNA' with probability indpb
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
#toolbox.register("mutate", tools.mutUniformInt, low = low, up = high, indpb = p_mutate)
#Selection: determines how best individuals are selected
toolbox.register("select", tools.selTournament, tournsize = tourn_size)



def main():

    #Variables
    log = tools.Logbook()
    df = df_alg1 = df_alg2 = pd.DataFrame()
    pop = toolbox.population()
    
    #Stats
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("mu", len)
    env.update_parameter("enemies", [enemy])
    

    
    #Test best algorithm
    if test_the_best:
        #print("Testing the best")
        for times in range(test_times):
            #If you want to watch the game being played
            if Headless == False:
                env.update_parameter('speed','normal')
            
            #Evaluates the best solution of the single training algorithm
            if train_single == True:
                env.update_parameter("logs", "on")
                best_sol = np.loadtxt("single_run/" + single_run_name +"_best.txt")
                print( '\n Test the Best Saved Solution \n')
                evaluate(best_sol)  
                env.play()
            
            #Evaluates best solutions of the algorithm outcomes generated for the assignment
            else:
                for i in enemies:
                    env.update_parameter("enemies", [i])
                    print(i)
                    for j in range(rep):
                        best_sol = np.loadtxt(experiment_name+"/" +"E"+str(i)+"rep"+str(j)+ "best.txt")
                        print("\n Testing solution Enemy " + str(i) + ", rep" + str(j))
                        evaluate(best_sol)
                        
                        #To write: output health
            
            print("Playerlife: " + str(env.get_playerlife()) + "\nEnemylife: " + str(env.get_enemylife()))
            
        sys.exit()
    
    #print("Parameters:\nmu=" + str(mu) +"\nlambda=" + str(lambda_) +"\np_mate=" + str(p_mate) + "\np_mutate=" + str(p_mutate) + "\ngenerations=" + str(generations) + "\nTournament size=" + str(tourn_size))    
    print(f"Parameters:\n mu={mu} \n lambda={lambda_} \n p_mate={p_mate} \n p_mutate={p_mutate} \n generations = {generations} \n Tournament size={tourn_size} \n Default fitness={default_fitness}")
    
    #Single run of an algorithm
    if train_single:

        pop, log = eaMuCommaLambda(pop, toolbox, mu, lambda_, p_mate, p_mutate_sel, generations, stats = stats, 
                                  halloffame=hof, verbose = True,
                                  cambrian= cambrian)
        
        #Save fittest individua.
        np.savetxt("single_run/"+single_run_name+"_best.txt", hof[:])
        
        #Save logged stats
        df = pd.DataFrame.from_dict(log)
        df.to_csv("single_run/" + single_run_name+ '_data.csv', index = False, header=True)
        
        
        
    #For two different algorithms:
    #Run three different enemies
    #Run 10 times for each enemy 
    #--> both algorithms in different .csv   
    else:
        for i in enemies:
            env.update_parameter("enemies", [i])
            print("Enemy "+ str(i))
            
            #Runs algorithm 1 and 2 x times
            for j in range(rep):
                
                print("Rep "+str(j+1))
                
                #First algorithm
                print("Algorithm 1")
                pop = toolbox.population()
                pop, log = algorithms.eaMuCommaLambda(pop, toolbox, mu, lambda_, p_mate, p_mutate, generations, stats = stats, 
                                          halloffame=hof, verbose = True)
                
                #Saves best solution to txt
                np.savetxt(experiment_name+"/ALG1E"+str(i)+"rep"+str(j+1)+ "best.txt",hof[:])
                #Saves stats to csv
                df_alg1 = pd.concat([df_alg1, pd.DataFrame.from_dict(log)])
                df_alg1.to_csv(experiment_name+'/alg1_data.csv', index = False, header=True)
                
                #Second algorithm
                print("Algorithm 2")
                pop = toolbox.population()
    
                pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu, lambda_, p_mate, p_mutate, generations, stats = stats, 
                                          halloffame=hof, verbose = True)
                #Saves best solution to txt
                np.savetxt(experiment_name+"/ALG2E"+str(i)+"rep"+str(j+1)+ "best.txt",hof[:])
                #Saves stats to csv
                df_alg2 = pd.concat([df_alg2, pd.DataFrame.from_dict(log)])
                df_alg2.to_csv(experiment_name+'alg2_data.csv', index = False, header=True)


            
            
    
        repen = list(np.repeat(np.array(enemies), rep*(generations+1)))
        repc = list(np.repeat(np.array(list(range(1, rep+1))*len(enemies)), (generations+1)))
    
        
        df_alg1 = df_alg1.assign(enemy = repen)
        df_alg1 = df_alg1.assign(rep = repc)
        df_alg2 = df_alg2.assign(enemy = repen)
        df_alg2 = df_alg2.assign(rep = repc)
        
        
        df_alg1.to_csv(experiment_name+ '/alg1_data.csv', index = False, header=True)
        df_alg2.to_csv(r'alg2_data.csv', index = False, header=True)
        
    
    df2 = pd.concat([df_para, parameters], ignore_index=True)
    df2.to_csv("single_run/data_parameters.csv", index = False)
    
    print(single_run_name + " finished")

    return pop, stats, hof


if __name__ == "__main__":
    main()





