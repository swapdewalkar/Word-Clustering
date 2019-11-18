#!/usr/bin/env python
import sys
import numpy as np
import time
from optparse import OptionParser
import logging

def normalize(A):
    column_sums = A.sum(axis=0)
    new_matrix = A / column_sums[np.newaxis, :]
    return new_matrix

def inflate(A, inflate_factor):
    return normalize(np.power(A, inflate_factor))

def expand(A, expand_factor):
    return np.linalg.matrix_power(A, expand_factor)

def add_diag(A, mult_factor):
    return A + mult_factor * np.identity(A.shape[0])

def get_clusters(A):
    clusters = []
    for i, r in enumerate((A>0).tolist()):
        if r[i]:
            clusters.append(A[i,:]>0)

    clust_map  ={}
    for cn , c in enumerate(clusters):
        for x in  [ i for i, x in enumerate(c) if x ]:
            clust_map[cn] = clust_map.get(cn, [])  + [x]
    return clust_map

def draw(G, A, cluster_map,labels):
    import networkx as nx
    import matplotlib.pyplot as plt

    clust_map = {}
    for k, vals in cluster_map.items():
        for v in vals:
            clust_map[v] = k
	#~ G=nx.cubical_graph()
    colors = []
    for i in range(len(G.nodes())):
        colors.append(clust_map.get(i))
        #print clust_map.get(i)
	
	#~ print "swapnil",colors

    pos = nx.spring_layout(G)

    from matplotlib.pylab import matshow, show, cm
    plt.figure(2)
    nl=G.nodes()
    el=[]
    ecolors=[]
    col=0;
    for c in cluster_map:
		n11=cluster_map[c]
		n22=cluster_map[c]
		#~ print "n11",n11
		#~ print "n22",n22
		for n1 in n11:
			for n2 in n22:
				if n1!=n2:
					#~ print "swap",n1,n2
					el.append((n1,n2))
					ecolors.append(c)	
		col=col+1
		
    #~ print el
    nx.draw_networkx_nodes(G, pos, nodelist=nl, node_size = 200, node_color =colors )
    nx.draw_networkx_edges(G,pos, edgelist=el, alpha=0.5, edge_color =ecolors )
    #~ G.add_path([0,1,2])
    nx.draw_networkx_labels(G,pos,labels,font_size=16)
    matshow(A, fignum=1, cmap=cm.gray)
    plt.show()
    show()
    
def stop(M, i):

    if i%5==4:
        m = np.max( M**2 - M) - np.min( M**2 - M)
        if m==0:
            logging.info("Stop at iteration %s" % i)
            return True

    return False


def mcl(M, expand_factor = 2, inflate_factor = 2, max_loop = 10 , mult_factor = 1):
    M = add_diag(M, mult_factor)
    M = normalize(M)

    for i in range(max_loop):
        logging.info("loop %s" % i)
        M = inflate(M, inflate_factor)
        M = expand(M, expand_factor)
        if stop(M, i): break

    clusters = get_clusters(M)
    return M, clusters

def networkx_mcl(G, expand_factor = 2, inflate_factor = 2, max_loop = 10 , mult_factor = 1):
    import networkx as nx
    A = nx.adjacency_matrix(G)
    return mcl(np.array(A.todense()), expand_factor, inflate_factor, max_loop, mult_factor)

def print_info(options):
    print("-" * 60)
    print("MARKOV CLUSTERING:")
    print("-" * 60)
    print("  expand_factor: %s" % options.expand_factor)
    print("  inflate_factor: %s" % options.inflate_factor)
    print("  mult factor: %s" % options.mult_factor)
    print("  max loops: %s\n" % options.max_loop)

def get_options():
    usage = "usage: %prog [options] <input_matrix>"
    parser = OptionParser(usage)
    parser.add_option("-e", "--expand_factor",
                      dest="expand_factor",
                      default=2,
                      type=int,
                      help="expand factor (default: %default)")
    parser.add_option("-i", "--inflate_factor",
                      dest="inflate_factor",
                      default=2,
                      type=float,
                      help="inflate factor (default: %default)")
    parser.add_option("-m", "--mult_factor",
                      dest="mult_factor",
                      default=2,
                      type=float,
                      help="multiply factor (default: %default)")
    parser.add_option("-l", "--max_loops",
                      dest="max_loop",
                      default=60,
                      type=int,
                      help="max loops (default: %default)")
    parser.add_option("-o", "--output", metavar="FILE", 
                      help="output (default: stdout)")

    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=True,
                      help="verbose (default: %default)")
    parser.add_option("-d", "--draw-graph",
                      action="store_true", dest="draw", default=False,
                      help="show graph with networkx (default: %default)")
    

    (options, args) = parser.parse_args()

    try:
        filename = args[0]
    except:
        raise Exception('input', 'missing input filename')


    return options, filename

def get_graph(csv_filename):
    import networkx as nx

    M = []
    for r in open(csv_filename):
        r = r.strip().split(",")
        M.append(list(map(lambda x: float(x.strip()), r)))

    G = nx.from_numpy_matrix(np.matrix(M))
    return np.array(M), G

def clusters_to_output(clusters, options):
    if options.output and len(options.output)>0:
        f = open(options.output, 'w')
        for k, v in clusters.items():
            f.write("%s|%s\n" % (k, ", ".join(map(str, v)) ))
        f.close()
    else:
        print("Clusters:")
        for k, v in clusters.items():
            print('{}, {}'.format(k, v))

if __name__ == '__main__':

    options, filename = get_options()
    print_info(options)
    M, G = get_graph(filename)

    print(" number of nodes: %s\n" % M.shape[0])

    print("{}: {}".format(time.time(), "evaluating clusters..."))
    M, clusters = networkx_mcl(G, expand_factor = options.expand_factor,
                               inflate_factor = options.inflate_factor,
                               max_loop = options.max_loop,
                               mult_factor = options.mult_factor)
    print("{}: {}".format(time.time(), "done\n"))

    clusters_to_output(clusters, options)

    if options.draw:
        print("{}: {}".format(time.time(), "drawing..."))
        draw(G, M, clusters)
        print("{}: {}".format(time.time(), "done"))
