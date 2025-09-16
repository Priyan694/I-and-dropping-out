import joblib
import pandas as pd

model = joblib.load("model.joblib")

print("Expected features:", model.feature_names_in_)


# Format: [attendance, avg_marks, backlogs, assignment%, payment_status, dept, year, rollno]



model = joblib.load("model.joblib")


new_student = pd.DataFrame([{
    "Dept": "CSE",
    "Year": "3",
    "Attendance%": 5,
    "Avg_Marks": 10,
    "Assignments_Submitted%": 0,
    "Backlogs": 5,
    "payment_status": "due",
    "risk_score": 1.0
}])

# new_student = pd.DataFrame([{
#     "Dept": "CSE",
#     "Year": "3",
#     "Attendance%": 50,
#     "Avg_Marks": 70,
#     "Assignments_Submitted%": 80,
#     "Backlogs": 1,
#     "payment_status": "paid",
#     "risk_score": 0.2
# }])


prob = model.predict_proba(new_student)[:,1][0] 
print("Dropout probability:", prob)


threshold = 0.2
prediction = "Yes" if prob >= threshold else "No"
print("Dropout Prediction:", prediction)



