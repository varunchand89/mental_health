class cleaning:
    def __init__(self,d_tr):
        self.d_tr = d_tr
    def data(self):
        self.d_tr.loc[self.d_tr["Working Professional or Student"] != "Student", "Academic Pressure"] = (self.d_tr.loc[self.d_tr["Working Professional or Student"] != "Student","Academic Pressure"].fillna(0))
        self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student", "Academic Pressure"] = (self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student",
                                                                                       "Academic Pressure"].fillna(self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student","Academic Pressure"].mean().round()))
        self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student", "Work Pressure"] = (self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student",
                                                                                       "Work Pressure"].fillna(0))
        self.d_tr.loc[self.d_tr["Working Professional or Student"] != "Student", "Work Pressure"] = (self.d_tr.loc[self.d_tr["Working Professional or Student"] != "Student",
                                                                                       "Work Pressure"].fillna(self.d_tr.loc[self.d_tr["Working Professional or Student"] != "Student","Work Pressure"].mean().round()))
        self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student", "Study Satisfaction"] = (self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student",
                                                                                       "Study Satisfaction"].fillna(self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student", "Study Satisfaction"]).mean().round())
        self.d_tr.loc[self.d_tr["Working Professional or Student"] != "Student", "Study Satisfaction"] = (self.d_tr.loc[self.d_tr["Working Professional or Student"] != "Student",
                                                                                       "Study Satisfaction"].fillna(0))
        self.d_tr["CGPA"] = self.d_tr["CGPA"].fillna(self.d_tr["CGPA"].mean())
        mask = self.d_tr["Working Professional or Student"] != "Student"
        mean_value = self.d_tr.loc[mask, "Job Satisfaction"].mean().round()
        self.d_tr.loc[mask, "Job Satisfaction"] = self.d_tr.loc[mask, "Job Satisfaction"].fillna(mean_value)
        self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student", "Job Satisfaction"] = (self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student",
                                                                                       "Job Satisfaction"].fillna(0))
        self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student", "Profession"] = (self.d_tr.loc[self.d_tr["Working Professional or Student"] == "Student",
                                                                                       "Profession"].fillna("Student"))
        self.d_tr.loc[self.d_tr["Working Professional or Student"] != "Student", "Profession"] = (self.d_tr.loc[self.d_tr["Working Professional or Student"] != "Student",
                                                                                       "Profession"].fillna("Teacher"))
        self.d_tr.loc[self.d_tr["Degree"] == "Class 12", "Profession"] = (self.d_tr.loc[self.d_tr["Degree"] == "Class 12", "Profession"].fillna("Content Writer"))
        self.d_tr.loc[:, 'Dietary Habits'] = self.d_tr['Dietary Habits'].fillna("Moderate")
        self.d_tr.loc[:,'Degree'] = self.d_tr.loc[:,'Degree'].fillna("Class 12")
        self.d_tr["Financial Stress"] = self.d_tr["Financial Stress"].fillna(2.0).astype(int)
        return self.d_tr