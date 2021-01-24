import zlib
import sys

def printdict(vocab_file,end=None,start=0,min_=None,max_=None):
    '''start:index of term from which you wanna print.
        end: index of term to which you wanna print
        min_ and max_ lexographical comparisions
        '''
    with open(vocab_file, 'r+b') as f:
        for oneterm in zlib.decompress(f.read()).decode('utf-8').split('=')[0].split('?')[start:end]:
            l=oneterm.split('%')
            if min_ and l[0]<min_: continue
            if max_ and l[0]>max_: break
            print(l[0]+' : '+l[1]+' : '+l[3])


if __name__=='__main__':
    printdict(sys.argv[1])