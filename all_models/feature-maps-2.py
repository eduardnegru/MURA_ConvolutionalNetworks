from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing import image
import pandas as pd
import numpy as np
import os
import random
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

# Dataset directory
PATH = '/kaggle/input/mura/'
tensorboard = TensorBoard(log_dir="/kaggle/working/logs/feature_maps_2")
class ModelTraining:
        
        def __init__(self):
            self.train_gen, self.train_images, self.train_studies = self.get_set(setname='train')
            self.valid_gen, self.valid_images, self.valid_studies = self.get_set(setname='valid')
            self.step_size_train = self.train_gen.n // self.train_gen.batch_size
            self.step_size_valid = self.valid_gen.n // self.valid_gen.batch_size
    
        def get_set(self, setname='train'):
            # Get images dataframe
            image_df = pd.read_csv(PATH + 'MURA-v1.1/' + setname + '_image_paths.csv',
                                 header=None)
            image_paths = image_df.values.flatten()
            
            # Get studies dataframe
            study_df = pd.read_csv(PATH + 'MURA-v1.1/' + setname + '_labeled_studies.csv',
                                 header=None)
            

            # Map study name to label
            studies = {}
            for v in study_df.values:
                studies[v[0]] = v[1]
            
            # Compute the label of each image from its respective study
            labels = []
            for img in image_paths:
                study = os.path.dirname(img) + '/'
                labels.append(studies[study])
            

            images = {}
            #Map image name to label
            for (index, image_name) in enumerate(image_paths):
                images[image_name] = labels[index]
            
            # Add labels to image dataframe
            image_df[1] = labels
            image_df.columns = ['x', 'y']
            
            
            # Build image generator
            datagen = image.ImageDataGenerator(rescale=1./255.)
            generator = datagen.flow_from_dataframe(dataframe=image_df,
                                                  directory=PATH,
                                                  x_col="x",
                                                  y_col="y",
                                                  seed=42,
                                                  shuffle=True,
                                                  color_mode='grayscale',
                                                  class_mode="other",
                                                  target_size=(227, 227))
            
            return generator, images, studies
        
        def model_load(self, path):
            try:
                model = load_model(path)
            except:
                model = None    
            return model
        
        def model_build(self):
            model = Sequential()
            model.add(Conv2D(80, kernel_size=(7, 7), strides=2,padding='SAME', activation='relu', input_shape=(227,227,1)))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Conv2D(140, (5, 5), strides=2, activation='relu', padding='SAME'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Conv2D(265, (3, 3), strides=1, activation='relu', padding='SAME'))
            model.add(Conv2D(395, (3, 3), strides=1, activation='relu', padding='SAME'))
            model.add(Conv2D(265, (3, 3), strides=1, activation='relu', padding='SAME'))
            model.add(Conv2D(265, (3, 3), strides=1, activation='relu', padding='SAME'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(Dropout(0.5))
            model.add(Dense(2048, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))
            self.model = model
            return model
        
        def model_train(self, save_path):
            
            # Compile model for training, BCE loss 
            self.model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy'])
            history = self.model.fit_generator(generator=self.train_gen,
                            steps_per_epoch=self.step_size_train,
                            validation_data=self.valid_gen,
                            validation_steps=self.step_size_valid,
                            epochs=10,
                            callbacks=[tensorboard])

            self.model.save(save_path)
        
            return history
        
        def accuracy_get(self, model, generator, step_size, labeledImages, isStudy = False, labeledStudies = None):

            if isStudy is False:
                my_predictions =  model.predict_generator(generator, steps=step_size).flatten()
                test_predictions = list(labeledImages.values())
            else:
                image_predictions =  model.predict_generator(generator, steps=step_size).flatten()
                predictions_per_study = {}
                index = 0
                print(image_predictions.shape)
                print(len(labeledImages))

                for image_path in labeledImages:
                    label = round(image_predictions[index])
                    study = "_".join(image_path.split("_")[:-1])
                    if study not in predictions_per_study:
                        predictions_per_study[study] = {}
                    if label not in predictions_per_study[study]:
                        predictions_per_study[study][label] = 0
                    
                    predictions_per_study[study][label] += 1
                    index += 1
                    if(index == image_predictions.size):
                        break
                my_predictions = []
                
                for study, votes in predictions_per_study.items():
                    vote = random.choice([0, 1])
                    
                    negativeVotes = 0 if 0 not in votes else votes[0]
                    positiveVotes = 0 if 1 not in votes else votes[1]
                    
                    if positiveVotes > negativeVotes:
                        vote = 1
                    elif positiveVotes < negativeVotes:
                        vote = 0
                    
                    my_predictions.append(vote)
                
                test_predictions = list(labeledStudies.values())


            true_positive = 0
            true_negative = 0

            for (index, my_prediction) in enumerate(my_predictions):

                if round(my_prediction) == int(test_predictions[index]):
                    true_positive += 1
                else:
                    true_negative += 1
            
            accuracy = true_positive / (true_positive + true_negative)
            return accuracy
        
        def model_test(self, model):
            images_test_accuracy = self.accuracy_get(model, self.valid_gen, self.step_size_valid, self.valid_images)
            images_train_accuracy = self.accuracy_get(model, self.train_gen, self.step_size_train, self.train_images)
            studies_test_accuracy = self.accuracy_get(model, self.valid_gen, self.step_size_valid, self.valid_images, True, self.valid_studies)
            studies_train_accuracy = self.accuracy_get(model, self.train_gen, self.step_size_train, self.train_images, True, self.train_studies)
            print("Accuracy " + str(studies_test_accuracy))
            print("Accuracy " + str(studies_train_accuracy))
            print("Accuracy " + str(images_train_accuracy))
            print("Accuracy " + str(images_test_accuracy))
            
                

        def training_loss_plot(self, image_path_save):
            loss_values = history.history['loss']
            epochs = range(1, len(loss_values)+1)
            plt.plot(epochs, loss_values, label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            #/kaggle/working/first_training_loss_plot
            plt.savefig(image_path_save)    
    
def main():
    modelTraining = ModelTraining()
    modelTraining.model_build()
    modelTraining.model_train("/kaggle/working/feature_maps2")
    
    
    # Evaluate the model
    # model = modelTraining.model_load("/kaggle/input/model20epochs/first_model_10_epochs")
    # modelTraining.model_test(model)

if __name__ == "__main__":
    main()