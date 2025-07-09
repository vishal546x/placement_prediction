from flask import Flask,request,render_template
import pickle
import pandas as pd

app=Flask(__name__)

with open('place.pkl','rb') as file:
    model=pickle.load(file)
cols_to_train=['CGPA', 'Internships', 'Projects',
       'Workshops/Certifications', 'AptitudeTestScore', 'SoftSkillsRating',
       'ExtracurricularActivities', 'PlacementTraining', 'SSC_Marks',
       'HSC_Marks']
f3=['AptitudeTestScore','HSC_Marks','SSC_Marks','SoftSkillsRating','CGPA']

def new_feature(li):
    b=1
    for i in li:
        b*=float(request.form[i])
    return b

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/pred", methods=["POST"])
def pred():
    input=[]

    for i in cols_to_train:
        input.append(float(request.form[i]))

    input.append(new_feature(f3))
    finals=cols_to_train+['f3']
    input_df=pd.DataFrame([input],columns=finals)
    
    ypred=model.predict(input_df)
    result = "PLACED 🏆" if ypred[0] == 1 else "NOT PLACED ❌"
    return render_template("predi.html",prediction=result)


@app.route("/predi")
def predi():
    return render_template("predi.html")
    
if __name__=='__main__':
    app.run(debug=True)