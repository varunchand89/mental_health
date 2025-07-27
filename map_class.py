d_mapping = {
    'More Healthy' : 'Healthy',
    'Less than Healthy' : 'Moderate',
    'No Healthy' : 'Unhealthy',
    'Less Healthy' : 'Moderate'}
d_mapping_1 = {
    'BEd' : 'B.Ed',
    'BPharm':'B.Pharm',
    'LL B.Ed':'B.Ed',
    'MTech' : 'M.Tech',
    'B BA' : 'BBA',
    'B.B.Arch' : 'B.Arch',
    'M_Tech' : 'M.Tech',
    'B B.Com' : 'B.Com'}
sleep_map = {
    'Less than 5 hours':'4.9','7-8 hours':'7.5','More than 8 hours':'8.1',
    '5-6 hours':'5.5','3-4 hours':'3.5','6-7 hours':'6.5','4-5 hours':'4.5',
    '4-6 hours':'5','2-3 hours':'2.5','1-6 hours':'3.5','6-8 hours':'7.5',
    '10-11 hours':'10.5','9-11 hours':'10','8-9 hours':'8.5','1-2 hours':'1.5',
    '9-6 hours':'7.5','1-3 hours':'2','8 hours':'8','10-6 hours':'8','than 5 hours':'5.1',
    '3-6 hours':'4.5','9-5':'7','9-5 hours':'7','20-21 hours':'20.5','6 hours':'6','9-10 hours':'9.5'}
valid_values = ['Healthy', 'Unhealthy', 'Moderate']
valid_degree = ['Class 12','B.Ed','B.Arch','B.Com','B.Pharm','BCA','M.Ed','MCA','BBA','BSc','MSc','LLM','M.Pharm','M.Tech','B.Tech','LLB','BHM','MBA',
               'BA','ME','MD','MHM','BE','PhD','M.Com','MBBS','MA','M.Arch','B.Sc','BEd','LLBA','M.S']
no_valid_name =['M.Tech','K.Pharm','UX/UI Designer','A.Ed','18','BE','R.Com','M.Com']
valid_city = [
    'Kalyan', 'Patna', 'Vasai-Virar', 'Kolkata', 'Ahmedabad', 'Meerut',
    'Ludhiana', 'Pune', 'Rajkot', 'Visakhapatnam', 'Srinagar', 'Mumbai',
    'Indore', 'Agra', 'Surat', 'Varanasi', 'Vadodara', 'Hyderabad',
    'Kanpur', 'Jaipur', 'Thane', 'Lucknow', 'Nagpur', 'Bangalore',
    'Chennai', 'Ghaziabad', 'Delhi', 'Bhopal', 'Faridabad', 'Nashik',
    'Gurgaon', 'Morena']
valid_professions = [
    'Teacher', 'Student', 'Content Writer', 'Architect', 'Consultant',
    'HR Manager', 'Pharmacist', 'Doctor', 'Business Analyst',
    'Entrepreneur', 'Chemist', 'Chef', 'Educational Consultant',
    'Data Scientist', 'Researcher', 'Lawyer', 'Customer Support',
    'Marketing Manager', 'Pilot', 'Travel Consultant', 'Plumber',
    'Sales Executive', 'Manager', 'Judge', 'Electrician',
    'Financial Analyst', 'Software Engineer', 'Civil Engineer',
    'UX/UI Designer', 'Digital Marketer', 'Accountant',
    'Finanancial Analyst', 'Mechanical Engineer', 'Graphic Designer',
    'Research Analyst', 'Investment Banker','Analyst']
no_valid_sleep = ['No','Sleep_Duration','Unhealthy','Moderate','Indore','Work_Study_Hours','Pune','45','40-45 hours','55-66 hours','35-36 hours',
                 '49 hours','45-48 hours','Meerut','Vivan','60-65 hours','8-89 hours','Have_you_ever_had_suicidal_thoughts','50-75 hours','']


class mapper:
    def __init__(self,d_t):
        self.d_t = d_t
    def mop(self):
        self.d_t['Dietary Habits'] = self.d_t['Dietary Habits'].replace(d_mapping)
        self.d_t.loc[:,'Degree'] = self.d_t['Degree'].replace(d_mapping_1)
        self.d_t['Sleep Duration'] = self.d_t['Sleep Duration'].replace(sleep_map)
        return self.d_t