from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
import pandas as pd

# Create your views here.

from sklearn.externals import joblib

reloadModel=joblib.load('./models/RFModelforMPG.pkl')


def index(request):
    context={'a':'HelloWorld'}
    return render(request,'index.html',context)
def predictMPG(request):
    print (request)
    if request.method == 'POST':
        temp={}
        temp['cylinders']=request.POST.get('cylinderVal')
        temp['displacement']=request.POST.get('dispVal')
        temp['horsepower']=request.POST.get('hrsPwrVal')
        temp['weight']=request.POST.get('weightVal')
        temp['acceleration']=request.POST.get('accVal')
        temp['model_year']=request.POST.get('modelVal')
        temp['origin']=request.POST.get('originVal')
        temp2=temp.copy()
        temp2['model year']=temp['model_year']
        print (temp.keys(),temp2.keys())
        # del temp2['model_year']

    testDtaa=pd.DataFrame({'x':temp2}).transpose()
    scoreval=reloadModel.predict(testDtaa)[0]
    context={'scoreval':scoreval,'temp':temp}
    return render(request,'index.html',context)

    