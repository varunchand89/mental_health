import pandas as pd
from sklearn.preprocessing import StandardScaler


class scaler:
    def __init__(self,outlier,label_df_copy):
        self.label_df_copy = label_df_copy
        self.outlier = outlier
    def zscaler(self):
        scaler = StandardScaler()
        #scaler_1 = scaler.fit_transform(self.label_df_copy[list(self.outlier)])
        if (self.label_df_copy.shape[0]) == 140426:
                scaler_1 = scaler.fit_transform(self.label_df_copy[list(self.outlier)])
                scaler_d = pd.DataFrame(scaler_1,columns =list(self.outlier),index=self.label_df_copy.index)
        else:
                scaler_1 = scaler.fit_transform(self.label_df_copy[list(self.outlier)[0:5]])
                scaler_d = pd.DataFrame(scaler_1,columns =list(self.outlier)[0:5],index = self.label_df_copy.index)
        for a in list(self.outlier):
            self.label_df_copy = self.label_df_copy.drop(columns = [a])
        scaler_df = pd.concat([scaler_d,self.label_df_copy],axis = 1)
        return scaler_df