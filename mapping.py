def get_node_map_for_dataset(dataset):

    if dataset == 'AIDS':
        return get_aids_node_map()
    elif dataset == 'BZR':
        return get_bzr_node_map()

def get_bzr_node_map():
    activities = """0	O
    1	C
    2	N
    3	F
    4	Cl
    5	S
    6	Br
    7	Si
    8	Na
    9	I
    10	Hg
    11	B
    12	K
    13	P
    14	Au
    15	Cr
    16	Sn
    17	Ca
    18	Cd
    19	Zn
    20	V
    21	As
    22	Li
    23	Cu
    24	Co
    25	Ag
    26	Se
    27	Pt
    28	Al
    29	Bi
    30	Sb
    31	Ba
    32	Fe
    33	H
    34	Ti
    35	Tl
    36	Sr
    37	In
    38	Dy
    39	Ni
    40	Be
    41	Mg
    42	Nd
    43	Pd
    44	Mn
    45	Zr
    46	Pb
    47	Yb
    48	Mo
    49	Ge
    50	Ru
    51	Eu
    52	Sc
    53	Gd"""

    node_map = {i.split('\t')[0].strip() : i.split('\t')[1].strip() for i in activities.split("\n")}

    return node_map


def get_aids_node_map():
    activities = """0	C  
1	O  
2	N  
3	Cl 
4	F  
5	S  
6	Se 
7	P  
8	Na 
9	I  
10	Co 
11	Br 
12	Li 
13	Si 
14	Mg 
15	Cu 
16	As 
17	B  
18	Pt 
19	Ru 
20	K  
21	Pd 
22	Au 
23	Te 
24	W  
25	Rh 
26	Zn 
27	Bi 
28	Pb 
29	Ge 
30	Sb 
31	Sn 
32	Ga 
33	Hg 
34	Ho 
35	Tl 
36	Ni 
37	Tb"""

    node_map = {i.split('\t')[0].strip() : i.split('\t')[1].strip() for i in activities.split("\n")}

    return node_map
