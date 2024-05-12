import time
import random
import math
import collections
from itertools import chain, combinations
import copy

def N(G,v):
    neighbor = []
    for u in G[v]:
        neighbor.append(u)
    return neighbor

def bfs(graph, root): 

    visited, queue = set(), collections.deque([root])
    visited.add(root)

    while queue:

        vertex = queue.popleft()
        
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited

def get_min_deg(G):
    min = math.inf
    for v in G.keys():
        if min > len(G[v]):
            min = len(G[v])
    return min

def get_min_deg_vertex(G):
    min = math.inf
    ver = None
    for v in G.keys():
        if min > len(G[v]):
            min = len(G[v])
            ver = v
    return ver

def get_max_deg_vertex(G):
    max = -1
    ver = None
    for v in G.keys():
        if max < len(G[v]):
            max = len(G[v])
            ver = v
    return ver


def is_connected(G):
    u = min(G.keys())
    visited = bfs(G,u)
    if set(G.keys())==set(visited):
        return True
    return False


def getEdges(G):
    E = set()
    for v in G.keys():
        for u in G[v]:
            if (u,v) not in E:
                E.add((v,u))
    return E

def powerset(fullset):
  listsub = list(fullset)
  subsets = []
  for i in range(2**len(listsub)):
    subset = []
    for k in range(len(listsub)):            
      if i & 1<<k:
        subset.append(listsub[k])
    subsets.append(subset)        
  return subsets

def get_edge_connectivity(G):
    edgeSets = powerset(getEdges(G))
    min = math.inf
    for edges in edgeSets:
        H = copy.deepcopy(G)
        
        for e in edges:
            # print(e[0], e[1], H[e[0]])
            H[e[0]].remove(e[1])
            H[e[1]].remove(e[0])
        if is_connected(H) == False:
            if len(edges) < min:
                min = len(edges)
        # print(G)
    return min   

def get_vertex_connectivity(G):
    vertexSets = powerset(set(G.keys()))
    min = math.inf
    current = G.keys()
    for vertices in vertexSets:
        H = copy.deepcopy(G)
        if len(vertices) != len(G.keys()):
            for v in vertices:
                del H[v]
                for u in H.keys():
                    if v in H[u]:
                        H[u].remove(v)

            if is_connected(H) == False:
                if len(vertices) < min:
                    min = len(vertices)
                    current = vertices
        # print(G)
    return min   

def permutation(lst):

    if len(lst) == 0:
        return []
 
    if len(lst) == 1:
        return [lst]
 
 
    l = [] 
    for i in range(len(lst)):
       m = lst[i]
 
       remLst = lst[:i] + lst[i+1:]
 
       for p in permutation(remLst):
           l.append([m] + p)
    return l


          
def getEdgesLatex(G):
    E = set()
    for v in G.keys():
        for u in G[v]:
            if (u,v) not in E:
                E.add("("+str(v)+")"+"--"+"("+str(u)+")")
    for i in E:
        print(i,end = " ")
    # return E

def getmaxdeg(G):
    max = 0
    for v in G.keys():
        if len(G[v])>max:
            max = len(G[v])
    return max

    
    
    
    
def randomGraph(s,n):
    # E =  random.randint(s, n*(n-1)/2)
    E =  random.randint(s, (n*(n-1)/2))
    G  =  dict()
    for i in range(n):
        G[i+1] = []
    for i in range(E):
        a = random.randint(1,n)
        b = random.randint(1,n)
        while a == b or b in G[a] or a in G[b]:
            a = random.randint(1,n)
            b = random.randint(1,n)
        G[a] = G.get(a, []) + [b]
        G[b] = G.get(b, []) + [a]
    # print(G)
    return G



def getDirEdges(G):
    E = set()
    for v in G.keys():
        for u in G[v]:
            if (v,u) not in E:
                E.add((v,u))
    return E

petersen = {
    0: [5,6,9],
    1: [3,4,6],
    2: [4,5,7],
    3: [1,5,8],
    4: [1,2,9],
    5: [2,3,0],
    6: [1,7,0],
    7: [2,6,8],
    8: [3,7,9],
    9: [4,8,0]
}

# print(getDirEdges(petersen))

def faceUpperbound(G):
    E = len(list(getEdges(G)))
    V = len(list(G.keys()))
    print(E)
    print(V)
    return min(math.floor((2*E)/3), E-V)

print(faceUpperbound(petersen))

def constructILP(G):
    X = []
    f_ = faceUpperbound(G)
    for i in range(f_):
        X.append(0)
    X = tuple(X)
    
    A = getDirEdges(G)
    c = []
    for i in range(f_):
        r = []
        for a in A:
            r.append(1)
        c.append(tuple(r))
    c = tuple(c)
    
    p = {}
    for v in list(G.keys()):
        d = {}
        for u in N(G,v):
            l = []
            for w in N(G,v):
                if u != w:
                    l.append(0)
            d[u] = l
        p[v] = d    
    print(p)    

# constructILP(petersen)     
# def ILPbruteforce(G):

def constructILP_random(G):
    X = []
    f_ = faceUpperbound(G)
    for i in range(f_):
        X.append(0)

    p = {}
    for v in list(G.keys()):
        d = {}
        for u in N(G,v):
            l = []
            for w in N(G,v):
                if u != w:
                    l.append([w,0])
            r = random.randint(0,len(l)-1)
            # print(r)
            l[r] = (l[r][0], 1)
            d[u] = l
        p[v] = d    
    print(p)   
    
    A = getDirEdges(G)
    c = []
    for i in range(f_):
        r = []
        for a in A:
            r.append(1)
        c.append(r)
    
    for v in list(p.keys()):
        for u in list(p[v].keys()):
            for k in p[v][u]:
                pass
            
    
    
    
constructILP_random(petersen)  


def faceTracing(G, rotation):
    D = list(getDirEdges(G))
    faces = 0 
    unused = D
    arc = unused[0]
    start = arc
    while len(unused) > 0:
        unused.remove(arc)
        if arc[0] != rotation[arc[1]][-1]:
            arc = (arc[1], rotation[arc[1]][rotation[arc[1]].index(arc[0]) + 1])
        else:
            arc = (arc[1], rotation[arc[1]][0])
        
        if arc == start:
            faces+=1
            if len(unused) <= 0:
                break
            arc = unused[0]
            start = arc
    # return faces
    print(faces)
            
        
faceTracing(petersen, petersen)
K4 = {
    0: [1,3,2],
    1: [2,3,0],
    2: [0,3,1],
    3: [2,0,1]
}
faceTracing(K4, K4)
    # for arc in unused:
        
        
def faceTracing(Graph, rotation_system):
    D = list(getDirEdges(Graph))
    faces = 0 
    unused = D
    arc = unused[0]
    start = arc
    while len(unused) > 0:
        unused.remove(arc)
        arc = (arc[1], rotation_system[arc[1]][(rotation_system[arc[1]].index(arc[0]) + 1)%len(rotation_system[arc[1]])])
        if arc == start:
            faces+=1
            if len(unused) <= 0:
                break
            arc = unused[0]
            start = arc
    return faces
    