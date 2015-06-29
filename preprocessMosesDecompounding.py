__author__ = 'lqrz'
import codecs
import sys

if __name__ == '__main__':

    # mosesResults = 'MT/mosesCompound_full_results' #TODO:hardcoded
    # outFile = 'MT/pastedResults_mod' #TODO:hardcoded

    if len(sys.argv)==3:
        mosesResults = sys.argv[1]
        outFile = sys.argv[2]
    elif len(sys.argv)>1:
        print 'Error in params.'
        exit()

    fMosesOut = codecs.open(mosesResults, 'r', encoding='utf-8')
    fout = codecs.open(outFile, 'w', encoding='utf-8')

    for l in fMosesOut:
        compound = l.strip().split()[0]
        if len(l.strip().split())>3:
            split1 = l.strip().split()[1]
            split2 = l.strip().split()[2].lower()
            idx = compound.find(split2)
            rest = compound[idx:]
            split = '\t'.join([split1,rest])
        elif len(l.strip().split())==3:
            split = l.strip().split()[1]+'\t'+l.strip().split()[2]
        elif len(l.strip().split())==2:
            split = l.strip().split()[1:][0]+'\t'+''
        else:
            split = l.strip()

        fout.write(compound+'\t'+split+'\n')

    fout.close()