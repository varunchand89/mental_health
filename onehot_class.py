from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class onehot_1:
    def __init__(self,dx,cat0):
       self.dx = dx
       self.cat0 = cat0
    def hot(self):
        encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        one_hot = encoder.fit_transform(self.dx[self.cat0])
        one_d = pd.DataFrame(one_hot,columns = encoder.get_feature_names_out(self.cat0),index=self.dx.index)
        one_dx = pd.concat([one_d,self.dx],axis= 1)
        one_dx = one_dx.drop((self.cat0),axis=1)
        return one_dx