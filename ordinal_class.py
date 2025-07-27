from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


class label_1:
    def __init__(self,one_df,cat1):
        self.one_df = one_df
        self.cat1 = cat1
    def lab(self):
        encoder_l = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        label_1 = encoder_l.fit_transform(self.one_df[self.cat1])
        label_df = pd.DataFrame(label_1,columns = ['City_l','Working Professional or Student_l','Profession_l'] ,index=self.one_df.index)
        label_d = pd.concat([label_df,self.one_df],axis = 1)
        label_d = label_d.drop(columns=label_d.select_dtypes(include='object').columns)
        return label_d