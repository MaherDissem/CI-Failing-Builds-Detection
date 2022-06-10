
tn, fn, fp, tp = 956, 369, 53, 210

prec = tp/(tp+fp)
recall = tp/(tp+fn)

f1=2*(recall*prec)/(recall+prec)
auc=(1+tp/(tp+fn)-fp/(fp+tn))/2

print(f'f1={f1}, auc={auc}')