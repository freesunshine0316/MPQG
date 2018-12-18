
import re
import sys
import os

fout = open(sys.argv[1]+'.1best','w')
fref = open(sys.argv[1]+'.ref','w')
for i,line in enumerate(open(sys.argv[1],'rU')):
    line = line.replace(' </s>','').strip().split()
    if len(line) == 0:
        continue
    if line[0] == 'Truth:':
        print >>fref, ' '.join(line[1:])
    elif line[0] == 'Hyp-0:':
        print >>fout, ' '.join(line[1:-1])
fout.close()
fref.close()

os.system('/home/lsong10/ws/exp.graph_to_seq/mosesdecoder/scripts/generic/multi-bleu.perl %s.ref < %s.1best' %(sys.argv[1],sys.argv[1]))
