from pm4py.objects.log.importer.csv import factory as csv_importer
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.objects.log.importer.xes import factory as xes_import_factory

# mining algorithms
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

# visualization
from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualization

#http://pm4py.pads.rwth-aachen.de/documentation/conformance-checking/evaluation-log-model/
from pm4py.evaluation.replay_fitness import factory as replay_factory
from pm4py.evaluation.precision import factory as precision_factory
from pm4py.evaluation.generalization import factory as generalization_factory
from pm4py.evaluation.simplicity import factory as simplicity_factory
from pm4py.evaluation import factory as evaluation_factory

def read_as_log_xes(filepath):
    return xes_import_factory.apply(filepath)

def read_as_log_csv(filepath):
    event_stream = csv_importer.import_event_stream(filepath)
    return conversion_factory.apply(event_stream)

def a_miner(log, visualize=True):
    net, initial_marking, final_marking = alpha_miner.apply(log)
    if visualize:
        gviz = pn_vis_factory.apply(net, initial_marking, final_marking)
        pn_vis_factory.view(gviz)
    return net, initial_marking, final_marking
        
def ind_miner(log, visualize=True):
    net, initial_marking, final_marking = inductive_miner.apply(log)
    if visualize:
        tree = inductive_miner.apply_tree(log)
        gviz = pt_visualizer.apply(tree)
        pt_visualizer.view(gviz)
                
        print("\nPetri Net:")
        gviz1 = pn_vis_factory.apply(net, initial_marking, final_marking)
        pn_vis_factory.view(gviz1)
    return net, initial_marking, final_marking
        
def heu_miner(log, visualize=True):
    net = heuristics_miner.apply_heu(log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.99})
    net1, initial_marking, final_marking = heuristics_miner.apply(log, parameters={"dependency_thresh": 0.99})
    if visualize:
        gviz = hn_visualizer.apply(net)
        hn_visualizer.view(gviz)
        
        print("\nPetri Net:")
        gviz1 = pn_vis_factory.apply(net1, initial_marking, final_marking)
        pn_vis_factory.view(gviz1)
    return net1, initial_marking, final_marking
        
def dfg_miner(log, visualize=True):
    dfg = dfg_discovery.apply(log)
    #dfg1 = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)
    if visualize:
        gviz = dfg_visualization.apply(dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY)
        dfg_visualization.view(gviz)
        
        #gviz1 = dfg_visualization.apply(dfg1, log=log, variant=dfg_visualization.Variants.PERFORMANCE)
        #dfg_visualization.view(gviz1)

def fitness_metric(log, net, initial_marking, final_marking):
    fitness = replay_factory.apply(log, net, initial_marking, final_marking)
    print("Fitness = ", fitness)
    
def precision_metric(log, net, initial_marking, final_marking):
    precision = precision_factory.apply(log, net, initial_marking, final_marking)
    print("Precision = ", precision)

def generalization_metric(log, net, initial_marking, final_marking):
    generalization = generalization_factory.apply(log, net, initial_marking, final_marking)
    print("Generalization = ", generalization)
    
def simplicity_metric(log, net, initial_marking, final_marking):
    simplicity = precision_factory.apply(log, net, initial_marking, final_marking)
    print("Simplicity = ", simplicity)
    
def evaluation_metric(log, net, initial_marking, final_marking):
    evaluation = evaluation_factory.apply(log, net, initial_marking, final_marking)
    print("Evaluation = ", evaluation)