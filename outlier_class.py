class outlier_m:
    def __init__(self,label_df):
        self.label_df = label_df
    def out(self):
        outliers_dit = {}
        for x in self.label_df.columns:
            q1 = self.label_df[x].quantile(0.25)
            q3 = self.label_df[x].quantile(0.75)
            iqr = q3 - q1 
            lower_b = q1 - iqr*1.5
            upper_b = q3 + iqr*1.5
            outliers = [val for val in self.label_df[x] if val < lower_b or upper_b < val ]
           
            if outliers:
               
               outliers_dit[x] = outliers
        return outliers_dit.keys()