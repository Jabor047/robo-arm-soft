import utilities
import numpy as np

Tx = 5511
n_freq = 101
Ty = 1375

ons, negatives, backgrounds = utilities.load_trigger_raw_audio()
print('loaded negative, ons, and backgrounds')

X = []
Y = []
i = 0
j = 0
back_indicies = np.arange(len(backgrounds))
while i <= 2:
    print('working on trigger example {}'.format(i))
    random_indices = np.random.choice(back_indicies, size=len(backgrounds))
    random_backs = [backgrounds[i] for i in random_indices]
    for back in random_backs:
        x, y = utilities.create_trigger_dataset(back, ons, negatives, i, j, Ty)
        if x.shape[1] != 5511:
            continue
        else:
            print('Saving the spectogram for {}{}'.format(i, j))
            X.append(x)
            Y.append(y)
        j += 1
    i += 1

print('Saving')
np.save('preprocessing/X_triggerword_1.npy', X)
np.save('preprocessing/Y_triggerword_1.npy', Y)

del X
del Y

print('Doing the second bunch')
k = 3
j = 0
X1 = []
Y1 = []
while k <= 5:
    print('working on trigger example {}'.format(k))
    random_indices = np.random.choice(back_indicies, size=len(backgrounds))
    random_backs = [backgrounds[i] for i in random_indices]
    for back in random_backs:
        x, y = utilities.create_trigger_dataset(back, ons, negatives, k, j, Ty)
        if x.shape[1] != 5511:
            continue
        else:
            print('Saving the spectogram for {}{}'.format(k, j))
            X1.append(x)
            Y1.append(y)
        j += 1
    k += 1

print('Saving')
np.save('preprocessing/X_triggerword_2.npy', X1)
np.save('preprocessing/Y_triggerword_2.npy', Y1)
