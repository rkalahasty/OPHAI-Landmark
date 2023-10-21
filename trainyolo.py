from models.yolov2 import YoloV2, get_gens, train
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
  if epoch < 150: 
    return 1e-4
  elif epoch < 300:
    return 1e-5
  else:
    return 1e-6

lr_callback = LearningRateScheduler(lr_schedule)

def trainYOLO(train_paths, test_paths, img):
    model = YoloV2(input_shape=(img, img, 3),
                   GRID_H=8, GRID_W=8)
    
    batch_size = 8
    grid_size = (8,8)
    box_size = 10 # pixel size of bounding box 
    anchors_count = 5
    class_count = 2
    max_annot = 10
    
    num_train = len(train_paths)
    num_val = math.ceil(num_train * 0.2)

    train_paths_subset = train_paths[:-num_val]
    val_paths = train_paths[-num_val:]

    train_gen, val_gen, test_gen = get_gens(img, 
                                            train_paths_subset,
                                            test_paths, 
                                            batch_size,
                                            val_size=0.1,
                                            grid_size=grid_size,
                                            box_size=box_size,
                                            anchors_count=anchors_count,
                                            class_count=class_count,
                                            max_annot=max_annot)

    train_len = len(train_paths_subset)//batch_size  
    val_len = len(val_paths)//batch_size

    
    history = train(epochs=400,
                    model=model,
                    train_dataset=train_gen,
                    val_dataset=val_gen,  
                    steps_per_epoch_train=train_len,
                    steps_per_epoch_val=val_len,
                    train_name='yolov2',
                    callbacks=[lr_callback])
    print(history)