import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
import mlflow 
from map_class import mapper
from cleaning_class import cleaning
from valid_class import valid
from onehot_class import onehot_1
from ordinal_class import label_1
from outlier_class import outlier_m
from scaler_class import scaler
from FNN_class import DepressionClassifier
import pickle

d_t = pd.read_csv("C:/Users/Hp/OneDrive/Desktop/mental_excel/train.csv")
d_ts = pd.read_csv("C:/Users/Hp/OneDrive/Desktop/mental_excel/test.csv")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "FNN_MODEL"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
with mlflow.start_run(run_name="Z-scaler", experiment_id=experiment.experiment_id):




        #mapping worng values with original value
        map_1 = mapper(d_t)
        d_tr = map_1.mop()
        #test
        map_2 = mapper(d_ts)
        d_tx = map_2.mop()


        #cleaning dataset
        clean = cleaning(d_tr)
        dt = clean.data()
        #test
        clean_1 = cleaning(d_tx)
        df = clean_1.data()

        #keeping valid data 
        val_1 = valid(dt)
        dx = val_1.val()
        #test
        val_2 = valid(df)
        dg = val_2.val()

        #add new column
        x_1 = dx[['Depression']]

        #removed unwanted column
        dx = dx.drop(columns=['CGPA', 'Degree', 'id', 'Name'], errors='ignore')
        dx = dx.drop(columns = ['Depression'])
        #test
        dg = dg.drop(columns=['CGPA', 'Degree', 'id', 'Name'], errors='ignore')


        #changing ogject to float
        dx['Sleep Duration'] = dx['Sleep Duration'].astype(float)
        #test
        dg['Sleep Duration'] = dg['Sleep Duration'].astype(float)


        #one_hot encoding
        cat0 = ['Gender', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
                  'Family History of Mental Illness']
        #test
        cat_test = ['Gender', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
            'Family History of Mental Illness']

        hot_1 = onehot_1(dx,cat0)
        one_df = hot_1.hot()
        #test
        hot_11 = onehot_1(dg,cat_test)
        one_dg = hot_11.hot()

        #label encoding
        cat1 = ['City','Working Professional or Student','Profession']
        #test
        cat_test_1 = ['City','Working Professional or Student','Profession']

        label_2 = label_1(one_df,cat1)
        label_df = label_2.lab()
        #test
        label_22 = label_1(one_dg,cat_test_1)
        label_dg = label_22.lab()



        #example why 0 and 1 are outliers
        #plt.hist(label_df['Depression'], bins=30, color='skyblue', edgecolor='black')
        #example why 0 and 1 are not outliers
        #plt.hist(label_df['Gender_Female'], bins=30, color='skyblue', edgecolor='black')	

        #outlier
        outliers_m = outlier_m(label_df)
        outlier = outliers_m.out()
        #test
        outliers_t = outlier_m(label_dg)
        outlier_t = outliers_t.out() 

        #copy of original tabel
        label_df_copy = label_df.copy()
        #test
        label_dg_copy = label_dg.copy()

        #converting outlier to z-scaler
        z_scaler = scaler(outlier,label_df_copy)
        scaled_df = z_scaler.zscaler()
        #test
        z_scaler_t = scaler(outlier_t,label_dg_copy)
        scaled_dg = z_scaler_t.zscaler()

        #reseting the index
        scaled_df = scaled_df.reset_index(drop=True)
        x_1 = x_1.reset_index(drop=True)

        #adding depression column

        scaled_df_n = pd.concat([scaled_df,x_1],axis = 1)

        #model tuning
        input_dim = 20  # Example: 20 input features
        hidden_dims = [64, 32]
        model = DepressionClassifier(input_dim, hidden_dims)

        criterion = nn.BCELoss()  # For binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        best_val_loss = float('inf')
        best_model_state = None


        #model training

        X = scaled_df_n.drop(columns=['Depression'])
        y = scaled_df_n['Depression']

        #balancing the inbalance data set 

        smote_enn = SMOTEENN(random_state=42)
        X_res, y_res = smote_enn.fit_resample(X, y)

        X_temp, X_test, y_temp, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)   # 20% test set
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 20% of 80% = 16% val

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Train on train tensors
        epochs = 20
        for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
    
                # Forward pass
                outputs = model(X_train_tensor)
                ll1_lambda = 0.001
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = criterion(outputs, y_train_tensor) + ll1_lambda * l1_norm

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                          val_pred = model(X_val_tensor)
                          val_loss = criterion(val_pred, y_val_tensor)
    
  
                mlflow.log_metric("train_loss",round(loss.item(),4), step=epoch)
                mlflow.log_metric("val_loss", round(val_loss.item(),4), step=epoch)



                

        if val_loss.item() < best_val_loss:
              best_val_loss = val_loss.item()
              best_model_state = model.state_dict()

         # Predict on test tensors
        model.load_state_dict(best_model_state)#loading best weights 
        model.eval() # seting model to evalution mode
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            y_pred_labels = (y_pred > 0.5).float().view(-1)
            y_test_flat = y_test_tensor.view(-1)



    
        acc = accuracy_score(y_test_flat.numpy(), y_pred_labels.numpy())
        per = precision_score(y_test_flat, y_pred_labels)
        re = recall_score(y_test_flat, y_pred_labels)
        f1 = f1_score(y_test_flat, y_pred_labels)
        name = "Z_SCALER"
        mlflow.log_param("Scaler" , name)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_param("learning_rate", 0.0005)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("hidden_layers", hidden_dims)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("Accuracy", round(acc, 2))
        mlflow.log_metric("Percision", round(per, 2))
        mlflow.log_metric("recall", round(re, 2))
        mlflow.log_metric("f1_score", round(f1, 2))

        with open("C:/Users/Hp/OneDrive/Desktop/Depressionmodel1.pkl", "wb") as f:
           pickle.dump(model, f)