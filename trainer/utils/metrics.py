from sklearn import metrics 

def get_pos_accuracy(TRUE,PRED, outfile=None):
    acc = metrics.f1_score(TRUE,PRED, average='micro')
    details = metrics.classification_report(TRUE,PRED, digits=3)
    print(details)

    if outfile is not None:
        fp = open(outfile,"w")
        fp.writelines(details + "\n")
        fp.close()
    return acc




