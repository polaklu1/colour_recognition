from tensorflow.keras.callbacks import ModelCheckpoint
from keras_adabound import AdaBound


## DATASET MAIN
dataset_path = r'PATH TO DATASET'
labels_path = r'PATH TO LABELS'
train_data_npy = r'TRAIN DATA NPY FILE'
train_labels_npy = r'TRAIN LABELS NPY FILE'
val_data_npy = r'VAL DATA NPY FILE'
val_labels_npy = r'VAL LABELS NPY FILE'

best_network = "bestNetwork.hdf5"

output_size = 9
classes_dict = {
    'black': 0,
    'silver-grey': 1,
    'white': 2,
    'red': 3,
    'blue': 4,
    'brown': 5,
    'green': 6,
    'yellow': 7,
    'orange': 8
}

## NETWORK
img_width = 128
img_height = 128
img_channels = 3

pretrained = False
loss = "categorical_crossentropy"
metrics = ['accuracy']
optimizer = AdaBound(lr=1e-4, final_lr=0.01)
save_model_callback = ModelCheckpoint(best_network, monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
size_mod = 0.1
batch_size = 16
# stepsPerEpoch = int(450*sizeMod)
steps_per_epoch = 400
epochs = 100

validation_batch_size = 16
validation_steps = int(20 * size_mod)
validation_split = 0.2

