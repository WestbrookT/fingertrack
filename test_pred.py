from model import model_v1, model_v2
from ibench import create_data_generator, create_dataset
from data import parse_datasets
paths, points = parse_datasets(keep_prob=.001, max_amount=20)
#datagen = create_data_generator(paths, points, 100, 32)


#model = model_v1('model_v1.h5')
model = model_v2('model_v2.h5')
X, y = create_dataset(paths, points, 150, 32)

#model.fit(X, y, 10, epochs=1, validation_split=.1)

print(model.predict(X, batch_size=32), y)

#model.save('model_v1.h5')