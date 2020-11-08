import requests



url = 'http://localhost:5000/predict_api'

r = requests.post(url,json={'Year':2013-2014,'State_Code':35,'District_name':'Nicobars','District_code':603, 'Season':'Rabi','Crop_code':606, 'Crop_category':'Vegetables', 'Area':11})


print(r.json())