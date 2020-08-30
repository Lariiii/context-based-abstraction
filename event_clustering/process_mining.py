from pm4py.objects.log.importer.csv import factory as csv_importer
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.objects.log.importer.xes import factory as xes_import_factory

from pm4py.objects.log.importer.csv import factory as csv_importer
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.objects.log.importer.xes import factory as xes_import_factory

from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.visualization.petrinet import visualizer as pn_visualizer

# helper function to read xes logs
def read_as_log_xes(filepath):
    return xes_import_factory.apply(filepath)

# helper function to read csv logs
def read_as_log_csv(filepath):
    event_stream = csv_importer.import_event_stream(filepath)
    return conversion_factory.apply(event_stream)
        
# this fucntion visualized a given net as petri net and stores it under the given path
def visualize_as_petri_net(net, initial_marking, final_marking, path=''):
    if len(path) > 0:    
        parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "svg"}
        gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters)
        pn_visualizer.save(gviz, path)   
    else:
        gviz = pn_vis_factory.apply(net, initial_marking, final_marking)
        pn_vis_factory.view(gviz)
    return net, initial_marking, final_marking