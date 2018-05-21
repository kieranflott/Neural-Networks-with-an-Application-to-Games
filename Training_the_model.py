# train_model.py

import numpy as np
from alexnet import alexnet
WIDTH = 200
HEIGHT = 300
LR = 1e-3
EPOCHS = 1
MODEL_NAME = 'Self-Driving-Car-{}-{}-{}.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

hm_data = 22
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        train_data = np.load('training_data-{}.npy'.format(i))

        train = train_data[:-550]
        test = train_data[-550:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)



# tensorboard --logdir=foo:C:/Users/kieran/Desktop/Project/Program/log

model.save(MODEL_NAME)




