

# num_of_training_eg = 4
train_ratio = 0.4
image_per_eg = 10
num_of_eigen = 30
image_size_x = 112
image_size_y = 92



# #split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, stratify=y)
# X_train, X_test, y_train, y_test = split_train_test(train_size=num_of_training_eg, image_size=image_per_eg)

"""---

# Training
"""

def save_model():
  joblib.dump(eigen_vector, eigen_vector_path)
  joblib.dump(avg_face, average_face_path)
  joblib.dump(weights, trained_weight_path)
  joblib.dump(y, train_model_output_path)  
  #save the label_map info to drive
  with open(label_csv_path, "w+") as f:
      f.write(label_df.to_csv())
  joblib.dump(label_map, self.label_map_path)

train_model(X_train,y_train, image_x=image_size_x, image_y=image_size_y, num_eigen=num_of_eigen)

"""---

# Testing
"""

def predict(X,y, mode='knn', n_neighbors=1, threshold=3e14):
  
  
  y_pred = []
  y_act = []
  y_comp = []
  
  sse_min = []
  sse_max = []
  
  if mode=='knn':
    for i,img_path in enumerate(X):
      test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      test_img.resize(1, image_size_x*image_size_y)
      adjusted_face = test_img - avg_face
      test_weight = np.dot(eigen_vector.T, adjusted_face.T)
      diff_weight = trained_weights - test_weight
      sum_of_squared_errors = np.sum(diff_weight*diff_weight, axis=0)

      if n_neighbors==1:
        name_index = y_train[np.argmin(sum_of_squared_errors)]
        potential_match = None
        
      else:
        tmp_df = pd.DataFrame()
        tmp_df['name_index'] = y_train[list(range(len(sum_of_squared_errors)))]
        tmp_df['sse'] = sum_of_squared_errors
        tmp_df.sort_values(by='sse', inplace=True)
        tmp_df.iloc[:,:n_neighbors]
        tmp_df.groupby('name_index').count()
        
        name_index = tmp_df.iloc[0,0]
        #return atmost 3 potential id if the first one doesn't match
        potential_match = tmp_df.iloc[0,:3]
        
      if min(sum_of_squared_errors)<threshold:
        y_pred.append(label_map[name_index])
      else:
        y_pred.append('nan')
        
      y_act.append(label_map[y[i]])

      y_comp.append('True' if y_pred[-1]==y_act[-1] else 'False')


      #only for testing/analysis
      sse_min.append(min(sum_of_squared_errors))
      sse_max.append(max(sum_of_squared_errors))


#     print('Matrix dimensions:')
#     print('X: ',X.shape)
#     print('y:',y.shape)
#     print('eigen vector:',eigen_vector.shape)
#     print('average face:',avg_face.shape)
#     print('weight:',trained_weights.shape)

    df_result = pd.DataFrame()
    df_result['input'] = X
    df_result['predicted output'] = y_pred
    df_result['actual output'] = y_act
    df_result['Correct'] = y_comp
    df_result['min SSE'] = sse_min
    df_result['max SSE'] = sse_max

    y_comp_base = ['True' for i in y_comp]
    tf_cm = confusion_matrix(y_comp, y_comp_base)


    return df_result,tf_cm

df, cm = predict(X_test, y_test, n_neighbors=1, threshold=3e14)
print('Confusion matrix:\n ',cm)

"""---

# Analysis
"""

#10-fold cross validation

label_df = pd.read_csv(label_csv_path, index_col=0)
# label_df.sort_values(by='0', inplace=True)
label_df.sample(frac=1)
X = label_df.iloc[:, 0].values
y = label_df.iloc[:, 1].values
image_size_x = 112
image_size_y = 92

result=[]

X_train = []
X_test = []
y_train = []
y_test = []

ind=0
while ind<10:
  x_ind=[i*10+ind for i in range(40)]
  ind=ind+1
  X_test = X[x_ind]
  y_test = y[x_ind]
  X_train = np.delete(X, x_ind)
  y_train = np.delete(X, x_ind)
  

  train_model(X_train,y_train, image_x=image_size_x, image_y=image_size_y, num_eigen=30)
  df,cm = predict(X_test, y_test)
  if cm.shape == (1,1):
    acc = cm[0,0]/cm[0,0]*100
  else:
    acc = cm[1,1]/(cm[1,1]+cm[0,1])*100
  result.append(acc)

res_df = pd.DataFrame()
res_df['Fold'] = list(range(1,1+len(result)))
res_df['Accuracy'] = result
res_df

res_df['Accuracy'].describe()

#5-fold cross validation

label_df = pd.read_csv(label_csv_path, index_col=0)
# label_df.sort_values(by='0', inplace=True)
label_df.sample(frac=1)
X = label_df.iloc[:, 0].values
y = label_df.iloc[:, 1].values

result1=[]

X_train = []
X_test = []
y_train = []
y_test = []

ind=0
while ind<9:
  x_ind=np.array([i*10+ind for i in range(40)])
  x_ind=np.append(x_ind, [i*10+ind+1 for i in range(40)])
  ind=ind+1
  X_test = X[x_ind]
  y_test = y[x_ind]
  X_train = np.delete(X, x_ind)
  y_train = np.delete(X, x_ind)

#     train_model(X_train,y_train, image_x=100, image_y=100, num_eigen=20)
  df, cm = predict(X_test, y_test)
  if cm.shape == (1,1):
    acc = cm[0,0]/cm[0,0]*100
  else:
    acc = cm[1,1]/(cm[1,1]+cm[0,1])*100
  result1.append(acc)

res1_df = pd.DataFrame()
res1_df['Fold'] = list(range(1,1+len(result1)))
res1_df['Accuracy'] = result1
res1_df

res1_df['Accuracy'].describe()

"""---

# ROUGH
"""

weight = joblib.load(trained_weight_path)
weight.shape

df

a=np.random.randint(5, size=(5,4))
b=np.random.randint(5, size=(4,3))
a

a=a*-1
a

abs(a)

np.subtract(tm, t)

r = pd.DataFrame([[1,3,5,7]])
r.columns = tmp.columns
tmp.append(r)

a = pd.DataFrame([[1,2,3,4,5],['a','b','c','d','e']])
a

csv_path = "gdrive/My Drive/Major project/data.csv"
with open(csv_path, 'w') as f:
  f.write(a.to_csv())

a = np.zeros((400,40000))
a.shape

b = np.ones((1,40000))
b.shape

a[0,:] = b

c = np.zeros((5,4))
d = np.array([1,2,3,4]).resize(1,4)
c[1,:] = d
c

a=np.array([2,5,9,6,1,8,3,7,0])
a.shape

a.resize(3,3)
a.resize(1,9)
a

a=np.random.randint(10, size=(4,4))
b=np.random.randint(10, size=(4,1))
print(a)
print(b)

np.subtract(a,b)

a=np.array([1,2,34,0,5,4])

np.argmin(a)

a=np.random.randint(10,size=(4,3))

print(a)
print(a*a)

eigen_vector = joblib.load(eigen_vector_path)
avg_face = joblib.load(average_face_path)
trained_weights = joblib.load(trained_weight_path)

trained_weights.shape

trained_weights[:5, :10]

#optimal value for the threshold seems to be ~8e16
print(min(sse_min), '\t', max(sse_min), '\n', min(sse_max), '\t', max(sse_max))

a=np.array([11,10,13,11,10,12,11,9,9,11,10,9])

from collections import Counter
# Counter(a).most_common()[0][0]
b=Counter(a)
sorted(a, key= lambda x: (b[x],x), reverse=True)

