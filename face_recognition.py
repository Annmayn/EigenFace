
import numpy as np
import pandas as pd
import cv2
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import isfile, join, exists

class EigenFace:
    """
    The PCA based algorithm for face recognition
    """ 
    
    def __init__(self, image_x=200, image_y=200, root_dir=""):
        """
        Creates an instance of EigenFace class
        
        Parameters:
        -------------
        image_x :  integer
            The size/length of image row.
        
        image_y :  integer
            The size/length of image column.
        
        root_dir :  string, optional
            Location of directory where the program will save all the necessary model parameters.
            Uses the current directory by default.
        """
        
        if not exists('etc'):
            os.makedirs('etc')
        self.image_x = image_x
        self.image_y = image_y
        self.label_csv_path = os.path.join(root_dir, 'etc', 'data.csv')
        self.trained_weight_path = os.path.join(root_dir, 'etc', 'weight.txt')
        self.eigen_vector_path = os.path.join(root_dir, 'etc', 'eigen_vector.txt')
        self.average_face_path = os.path.join(root_dir, 'etc', 'average_face.txt')
        self.train_model_output_path = os.path.join(root_dir, 'etc', 'y_train.txt')

        self.eigen_vector = None 
        self.y_train = None
        self.avg_face = None
        self.weights = None


    def generateLabels(self, dataset_path):
        """
        Creates a pandas DataFrame with 2 columns: the location of the image
        and the corresponding name of the associated face in the image
        
        Parameters
        --------
        dataset_path : string
            Path to the folder consisting of the images
            
        Returns
        --------
            pandas DataFrame with 2 columns: path to the image and the name
            of the user associated with the image
        
        Notes
        --------
            The dataset_path must be a directory of directories with each 
            directory consisting the images of a single user with the name 
            of that directory set to the name of that particular user
        """
        
        cnt=0
        self.label_df = pd.DataFrame()
        self.label_map = {}

        for dirr in listdir(dataset_path):
            data_path = join(dataset_path,dirr)
            only_images = [f for f in listdir(data_path) if isfile(join(data_path,f))]
          
            for image_name in only_images:
                image_path = join(data_path, image_name)
                self.label_df = self.label_df.append([[image_path, dirr]])

            cnt += 1
        return self.label_df


    def readLabels(self, label_df=None):
        """
        Reads the pandas DataFrame consisting of image path and the output(the 
        name of the user associated with the image) and separates them into 2 
        numpy arrays with one consisting of image paths and the other 
        consisting of the output
        
        Parameters
        --------
            label_df : pandas.DataFrame, optional
            
        Returns
        --------
            X : numpy n-d array, string
                Image paths
            
            y : numpy n-d array, string
                Output associated with the image path
        """
        
        if label_df is None:
            label_df = pd.read_csv(self.label_csv_path)
        X = label_df.iloc[:, 0].values
        y = label_df.iloc[:, 1].values
        return X,y

    def trainModel(self, X, y, num_of_eigen):
        """
        Trains the model using EigenFace
        
        Parameters
        --------
            X : list, string
                Array of strings consisting of image path.
                
            y : list, string
                Output associated with image corresponding to X.
                
            num_of_eigen : int
                Number of eigen vectors to use when building model.
        """
        
        self.y_train = y
        all_image = np.zeros((X.shape[0], self.image_x*self.image_y))

        for i, image_path in enumerate(X):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image.resize(1,self.image_x * self.image_y)
            all_image[i,:] = image

            #Calculate average face
            self.avg_face = all_image.sum(axis=0)/all_image.shape[0]

            #adjusted face dataset
            self.avg_face.resize(1,self.image_x*self.image_y)
            adjusted_face = np.subtract(all_image,self.avg_face)

            #covariance matrix calculation
            cov_matrix = np.dot(adjusted_face, adjusted_face.T)

            #calcluate eigen vectors of the covariance matrix
            temp_eigen = np.linalg.eig(cov_matrix)

            #Find all eigen vectors
            df_eigen = pd.DataFrame(temp_eigen[1])
            df_eigen['eigen_value'] = temp_eigen[0]
            df_eigen.sort_values(by='eigen_value', ascending=False, inplace=True)
            selected_eigen = df_eigen.iloc[:,:num_of_eigen]
          
            #Compute eigenvector for actual covariance matrix of the image
            self.eigen_vector = (adjusted_face.T).dot(selected_eigen)

            #weight calculation
            self.weights = np.dot(self.eigen_vector.T, adjusted_face.T)

    def saveModel(self, label_df=None):
        """
        Saves the current model in the 'root_dir'.
        
        Parameters
        --------
            label_df : pandas.DataFrame, optional
                 If provided, saves the pandas DataFrame consisting 
                 of the image paths and corresponding output, alongside 
                 the model.
        """

        joblib.dump(self.eigen_vector, self.eigen_vector_path)
        joblib.dump(self.avg_face, self.average_face_path)
        joblib.dump(self.weights, self.trained_weight_path)
        joblib.dump(self.y_train, self.train_model_output_path)  
        with open(self.label_csv_path, "w+") as f:
            f.write(label_df.to_csv())

    def loadModel(self):
        """
        Loads the saved model
        """

        self.eigen_vector = joblib.load(self.eigen_vector_path)
        self.avg_face = joblib.load(self.average_face_path)
        self.weights = joblib.load(self.trained_weight_path)
        self.y_train = joblib.load(self.train_model_output_path)

    def fit(self,X,y,mode='train',num_of_eigen=20):
        if mode=='train':
            self.trainModel(X,y,num_of_eigen)
        elif mode=='load':
            self.loadModel()

    def image_predict(self,test_img,threshold=3e14,n_neighbors=1):
        """
        Makes prediction for single image.

        Parameters
        --------
            img : list, ndarray
                test image array
            

            threshold : float, default=3e14
                Defines the threshold beyond which the prediction will be discarded

            n_neighbors : integer, default=1
                Defines the number of neighbors to consider during prediction. 
                Uses KNN algorithm for n>1. 

        Note
        --------
            Use n_neighbors only when there are a lot of images of a single user
            in the training set.
        """

#        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_img.resize(1, self.image_x*self.image_y)
        adjusted_face = test_img - self.avg_face
        test_weight = np.dot(self.eigen_vector.T, adjusted_face.T)
        diff_weight = self.weights - test_weight
        sum_of_squared_errors = np.sum(diff_weight*diff_weight, axis=0)

        if n_neighbors==1:
            name = self.y_train[np.argmin(sum_of_squared_errors)]
            potential_match = None
          
        else:
            tmp_df = pd.DataFrame()
            tmp_df['name'] = self.y_train[list(range(len(sum_of_squared_errors)))]
            tmp_df['sse'] = sum_of_squared_errors
            tmp_df.sort_values(by='sse', inplace=True)
            tmp_df.iloc[:,:n_neighbors]
            tmp_df.groupby('name_index').count()
            
            name = tmp_df.iloc[0,0]
            #return atmost 3 potential id if the first one doesn't match
            potential_match = tmp_df.iloc[0,:3]
          
        # print('sse: ', min(sum_of_squared_errors))
        y_pred = name
#        if min(sum_of_squared_errors)<threshold:
#            y_pred = name
#        else:
#            y_pred = 'nan'

        return y_pred

    def predict(self,X,y,threshold=3e14,n_neighbors=1):
        """
        Makes prediction for provided test images.

        Parameters
        --------
            X : list, string
                List of string that contain path to the image.
            
            y : list, string
                List of string that contains name of user associated 
                with the image represented by X.

            threshold : float, default=3e14
                Defines the threshold beyond which the prediction will be discarded

            n_neighbors : integer, default=1
                Defines the number of neighbors to consider during prediction. 
                Uses KNN algorithm for n>1. 

        Note
        --------
            Use n_neighbors only when there are a lot of images of a single user
            in the training set.
        """

        y_pred = []
        y_act = []
        y_comp = []
        sse_min = []
        sse_max = []

        for i,img_path in enumerate(X):
            test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            test_img.resize(1, self.image_x*self.image_y)
            adjusted_face = test_img - self.avg_face
            test_weight = np.dot(self.eigen_vector.T, adjusted_face.T)
            diff_weight = self.weights - test_weight
            sum_of_squared_errors = np.sum(diff_weight*diff_weight, axis=0)

            if n_neighbors==1:
                name = self.y_train[np.argmin(sum_of_squared_errors)]
                potential_match = None
              
            else:
                tmp_df = pd.DataFrame()
                tmp_df['name'] = self.y_train[list(range(len(sum_of_squared_errors)))]
                tmp_df['sse'] = sum_of_squared_errors
                tmp_df.sort_values(by='sse', inplace=True)
                tmp_df.iloc[:,:n_neighbors]
                tmp_df.groupby('name_index').count()
                
                name = tmp_df.iloc[0,0]
                #return atmost 3 potential id if the first one doesn't match
                potential_match = tmp_df.iloc[0,:3]
              
            if min(sum_of_squared_errors)<threshold:
                y_pred.append(name)
            else:
                y_pred.append('nan')

        return y_pred

          
