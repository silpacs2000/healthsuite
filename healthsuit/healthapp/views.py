from django.shortcuts import redirect, render
from django.contrib import messages
from healthapp.models import Users,doctors,contact
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from ctgan import CTGAN
from sklearn.metrics import accuracy_score 
import joblib

# Create your views here.
def index(request):
    if 'EmailID' in request.session:
        current_user=request.session['EmailID']
        user=Users.objects.get(EmailID=current_user)
        return render(request,"index.html",{'current_user':current_user,'user':user})
    return render(request,"index.html")
def user_registration(request):
    if request.method=='POST':
        name=request.POST['name']
        email=request.POST['email']
        passw=request.POST['password']
        cpass=request.POST['cpassword']
        address=request.POST['address']
        phone=request.POST['phone']
        place=request.POST['place']
        emailexists=Users.objects.filter(EmailID=email)
        if emailexists:
            messages.error(request,'Email ID already registered')
        elif passw!=cpass:
            messages.error(request,'Password not match')
        else:
            Users.objects.create(Name=name,EmailID=email,Password=passw,Address=address,PhoneNo=phone,Place=place)
            return redirect('/')
    return render(request,"register.html")
def user_login(request):
    if request.method=='POST':
        email=request.POST['email']
        passw=request.POST['password']
        user=Users.objects.filter(EmailID=email,Password=passw)
        if user:
            request.session['EmailID']=email
            return redirect('/')
        else:
            messages.error(request,'Invalid Credentials')
    return render(request,"login.html")
def user_logout(request):
    del request.session['EmailID']
    return redirect('/')
def heart_prediction(request):
    if 'EmailID' in request.session:
        current_user=request.session['EmailID']
        user=Users.objects.get(EmailID=current_user)
        df = pd.read_csv('static/csv/framingham.csv')
        df.dropna(axis = 0, inplace = True) 
        df.drop(columns=['education'], inplace=True)
        X = df.drop(columns=['TenYearCHD'])
        y = df['TenYearCHD'] 
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21)
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        # Prompt the user for input
        if request.method=='POST':
            name=request.POST['name']
            age = float(request.POST['age'])
            sex = float(request.POST['gender'])
            # Repeat this for other features...

            # Create a DataFrame with user input
            user_input = pd.DataFrame({
                'male': [sex],  # Adjust feature name to match the one used in training
                'age': [age],
            # Add other features here...
 
            'currentSmoker': [float(request.POST['smoker'])],
            'cigsPerDay': [float(request.POST['cigerete'])],
            'BPMeds': [float(request.POST['bp'])],
            'prevalentStroke': [float(request.POST['stroke'])],
            'prevalentHyp': [float(request.POST['hyper'])],
            'diabetes': [float(request.POST['diabetes'])],
            'totChol': [float(request.POST['cholestrol'])],
            'sysBP': [float(request.POST['sysbp'])],
            'diaBP': [float(request.POST['diabp'])],
            'BMI': [float(request.POST['bmi'])],
            'heartRate': [float(request.POST['heartrate'])],
            'glucose': [float(request.POST['glucose'])]
            })

            # Use the trained model to predict
            prediction = logreg.predict(user_input)

            # Display the prediction result
            if prediction[0] == 1:
                messages.success(request," is predicted to have a 10-year risk of coronary heart disease (CHD).")
                messages.success(request,"\nYou can consult a doctor")
            else:
                messages.success(request,"The user is predicted to not have a 10-year risk of coronary heart disease (CHD).")
        return render(request,"heartprediction.html",{'current_user':current_user,'user':user})
def myprofile(request):
    if 'EmailID' in request.session:
        current_user=request.session['EmailID']
        user=Users.objects.get(EmailID=current_user)
        return render(request,"profile.html",{'current_user':current_user,'user':user})
def liver_prediction(request):
    if 'EmailID' in request.session:
        current_user=request.session['EmailID']
        user=Users.objects.get(EmailID=current_user)
        df=pd.read_csv('static/csv/liver_dataset.csv')
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Apply label encoding
        df['Gender of the patient'] = label_encoder.fit_transform(df['Gender of the patient'])
        # Replace missing values with mean
        df.fillna(df.median(), inplace=True)
        X = df.drop(columns=['Result'])
        y = df['Result'] 
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        score = logreg.score(X_test, y_test)
        print("Prediction score is:",score) 
        if request.method=="POST":
            ##### Assume you have already trained your model and stored it in the variable log_reg

            # Accept user input for each feature
            age = float(request.POST['age'])
            gender = float(request.POST['gender'])
            total_bilirubin = float(request.POST['totalbilirubin'])
            direct_bilirubin = float(request.POST['directbilirubin'])
            alkaline_phosphotase = float(request.POST['alkaline'])
            aminotransferase = float(request.POST['Alamine'])
            aspartate_aminotransferase = float(request.POST['Sgot'])
            total_proteins = float(request.POST['Protiens'])
            albumin = float(request.POST['Albumin'])
            ag_ratio_albumin_and_globulin_ratio = float(request.POST['Globulin'])

            # Change input data into a NumPy array
            input_data_as_numpy_array = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                       aminotransferase, aspartate_aminotransferase, total_proteins, albumin,
                                       ag_ratio_albumin_and_globulin_ratio]])

            # Predicting the result
            prediction = logreg.predict(input_data_as_numpy_array)
            print("Prediction:", prediction)

            if prediction == 1:
                messages.success(request,"Your liver Condition is better.")
            else:
                messages.success(request,"Your liver Condition is not good.")
        return render(request,"liverprediction.html",{'current_user':current_user,'user':user})
def lung_prediction(request):
    if 'EmailID' in request.session:
        current_user=request.session['EmailID']
        user=Users.objects.get(EmailID=current_user)
        # Load the logistic regression model from the HDF5 file
        loaded_model = joblib.load('static/h5/lung_cancer_model.h5')
        # Assuming you have loaded the model into loaded_model variable using joblib or pickle
        if request.method=='POST':
            # Take input from the user
            age = float(request.POST['age'])
            smokes = float(request.POST['smokes'])
            Carcinoembryonic_antigen = float(request.POST['antigen'])
            Alcohol = float(request.POST['alcohol'])

            # Convert input data into a NumPy array
            input_data = np.array([[age, smokes, Carcinoembryonic_antigen, Alcohol]])

            # Make prediction using the loaded model
            prediction = loaded_model.predict(input_data)

            # Print prediction
            if prediction == 1:
                messages.success(request,"The person doesn't have lung cancer.")
                messages.success(request,prediction)
                return render(request,"lungprediction.html",{'current_user':current_user,'user':user,'prediction':prediction})
            else:
                messages.success(request,"The person has lung cancer.")
                messages.success(request,prediction)
                return render(request,"lungprediction.html",{'current_user':current_user,'user':user,'prediction':prediction})
        
        return render(request,"lungprediction.html",{'current_user':current_user,'user':user})
def departments(request):
    if 'EmailID' in request.session:
        current_user=request.session['EmailID']
        user=Users.objects.get(EmailID=current_user)
        return render(request,"departments.html",{'current_user':current_user,'user':user})
def doctorslist(request,department):
    doc=doctors.objects.filter(Department=department)
    return render(request,"doctors.html",{'doc':doc})
def updateprofile(request,id):
    user=Users.objects.get(id=id)
    if request.method=='POST':
        name=request.POST['name']
        passw=request.POST['password']
        cpass=request.POST['cpassword']
        address=request.POST['address']
        phone=request.POST['phone']
        place=request.POST['place']
        if passw!=cpass:
            messages.error(request,'Password not match')
        else:
            user.Name=name
            user.Password=passw
            user.Address=address
            user.PhoneNo=phone
            user.Place=place
            user.save()
    return render(request,"profileedit.html",{'user':user})
def contactus(request):
    if request.method=='POST':
        name=request.POST['name']
        email=request.POST['email']
        msg=request.POST['message']
        contact.objects.create(Name=name,Email=email,Message=msg)
        return redirect('/')
    return render(request,"contact.html")