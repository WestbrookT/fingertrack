from model import model_v1, model_v2
from ibench import create_data_generator, create_dataset
from data import parse_datasets

import random, time

random.seed(time.time())
model = model_v2('model_v2_4.h5')
paths, points = parse_datasets(keep_prob=0, max_amount=20000)
#datagen = create_data_generator(paths, points, 100, 32)


#model = model_v1('model_v1.h5')

gepochs = 30
for i in range(gepochs):
    print('On {} out of {}'.format(i, gepochs))
    X, y = create_dataset(paths, points, 50, 8192)

    model.fit(X, y, 64, epochs=20, validation_split=.1)

    model.save('model_v2_5.h5')

#datagen = create_data_generator(paths, points, 100, 64)

#model.fit_generator(datagen, 1024, 20)


model.save('model_v2_5.h5')