from sys import argv
with open(argv[1], "r") as f:
    sc_to_loss = {}
    for l in f.readlines():
        if l == "" or l.startswith('{') or l.startswith('}'):
            continue
        lin = l.strip().split(":")
        fn = lin[0][1:-1]
        
        if lin[1].endswith(','):
            loss = float(lin[1][:-1])
        else:
            loss = float(lin[1])
        
        s, c = fn[:-4].split("_")
        s = int(s.replace("style", ""))
        c = int(c.replace("content", ""))
        sc_to_loss[(s,c)] = loss

sc_keys = list(sc_to_loss.keys())
sc_keys = sorted(sc_keys, key=lambda x : x[1])
sc_keys = sorted(sc_keys, key=lambda x : x[0])


with open(argv[1] + ".txt", 'w') as f:
    for k in sc_keys:
        loss = sc_to_loss[k]
        f.write(f'"style{k[0]}_content{k[1]}.png": {loss}\n')