import numpy as _np

def to_gray(img):
	return (img[...,0] + img[...,1] + img[...,2]) / 3

relu = _np.vectorize(lambda x : max(0,x))

def binary(n):
    if n > 1:
        return f'{binary(n//2)}{n%2}'
    else: return f'{n%2}'

def generate_patterns():
    paths = _np.zeros(256)
    for n in range(256):
        b = f'{binary(n):0>8}'
        for i in range(8):
            if ( b[(i+1) % 8] == '0' ) and ( b[i] == '1' ):
                paths[n] += 1
    bundle = _np.zeros(256)
    for n in range(256):
        b = f'{binary(n):0>8}'
        if b[7] == b[6] == b[0] == '1':
            bundle[n] = 1
        elif b[6] == b[5] == b[4] == '1':
            bundle[n] = 1
        elif b[4] == b[3] == b[2] == '1':
            bundle[n] = 1
        elif b[2] == b[1] == b[0] == '1':
            bundle[n] = 1

    end_patterns = []
    for n in range(256): 
        if paths[n] == 1: end_patterns.append(n)
    branch_patterns = []
    for n in range(256): 
        if paths[n] > 2: branch_patterns.append(n)

    return paths.astype('int'), bundle.astype('int'), _np.array(end_patterns), _np.array(branch_patterns)

def generate_dl():
    dl = _np.zeros(256)
    for n in range(256):
        b = f'{binary(n):0>8}'
        L = D = R = U = False
        if b[0] == '1':
            L = True
            dl[n] += 0.5
        if b[2] == '1':
            D = True
            dl[n] += 0.5
        if b[4] == '1':
            R = True
            dl[n] += 0.5
        if b[6] == '1':
            U = True
            dl[n] += 0.5
        if (b[1] == '1') and not (L or D):
            dl[n] += (2**(1/2)/2)
        if (b[3] == '1') and not (D or R):
            dl[n] += (2**(1/2)/2)
        if (b[5] == '1') and not (R or U):
            dl[n] += (2**(1/2)/2)
        if (b[7] == '1') and not (L or U):
            dl[n] += (2**(1/2)/2)

    return dl

def label0(img):
    reprint = _np.copy(img).astype('int')
    coords = _np.where(reprint==1)
    for n in range(int(_np.sum(reprint))):
        reprint[coords[0][n],coords[1][n]] = n+1
    return reprint