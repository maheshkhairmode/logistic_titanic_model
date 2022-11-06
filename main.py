from distutils.command.config import config
from flask import Flask,jsonify,render_template,request
from models.utils import TitanicSurvival
import config

app =Flask(__name__)

@app.route("/")
def hello_flask():
    return render_template("index.html")

@app.route("/predict_survival",methods=["POST","GET"])
def titanic_surv():
    if request.method=="POST":
        
        PassengerId=int(request.form.get("PassengerId"))
        Pclass=int(request.form.get("Pclass"))
        Gender=request.form.get("Gender")
        Age=int(request.form.get("Age"))
        SibSp=int(request.form.get("SibSp"))
        Parch=int(request.form.get("Parch"))
        Fare=float(request.form.get("Fare"))
        Embarked=request.form.get("Embarked")

        pred_survival=TitanicSurvival(PassengerId,Pclass,Gender,Age,SibSp,Parch,Fare,Embarked)
        survival=pred_survival.Survival_prediction()
        print("preson survival  status in titanic",survival)

        return render_template("index.html",prediction=survival)


if __name__=="__main__":

    app.run(host="0.0.0.0",port=config.PORT_NUMBER,debug=True)

