import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.joblib")

print("Expected features:", model.feature_names_in_)

# New student data (only academic features now)
# new_student = pd.DataFrame([{
#     "Attendance%": 35,
#     "Avg_Marks": 40,
#     "Assignments_Submitted%": 20,
#     "Backlogs": 3
# }])

# new_student = pd.DataFrame([{
#     "Attendance%": 74,
#     "Avg_Marks": 100,
#     "Assignments_Submitted%": 100,
#     "Backlogs": 0
# }])

# Prediction
# prob = model.predict_proba(new_student)[:, 1][0]
# print("Dropout probability:", prob*100, "%")

# Threshold (tweak as needed)
threshold = 0.5
# prediction = "Yes" if prob >= threshold else "No"
# print("Dropout Prediction:", prediction)





students = [
    ["Student_ID", "Name", "Dept", "Year", "Attendance%", "Avg_Marks", "Assignments_Submitted%", "Backlogs"],
    ["S1001", "Student_1001", "IT", 4, 82, 65, 24, 0],
    ["S1002", "Student_1002", "ECE", 1, 41, 47, 96, 3],
    ["S1003", "Student_1003", "ECE", 4, 64, 35, 21, 3],
    ["S1004", "Student_1004", "CIVIL", 1, 34, 91, 36, 6],
    ["S1005", "Student_1005", "ECE", 2, 71, 50, 36, 0],
    ["S1006", "Student_1006", "CSE", 1, 81, 87, 100, 3],
    ["S1007", "Student_1007", "ECE", 3, 78, 30, 48, 5],
    ["S1008", "Student_1008", "ECE", 2, 45, 53, 40, 0],
    ["S1009", "Student_1009", "IT", 2, 67, 86, 97, 3],
    ["S1010", "Student_1010", "MECH", 4, 92, 31, 35, 3],
    ["S1011", "Student_1011", "IT", 2, 86, 62, 74, 4],
    ["S1012", "Student_1012", "IT", 4, 66, 40, 66, 0],
    ["S1013", "Student_1013", "ECE", 1, 85, 64, 75, 6],
    ["S1014", "Student_1014", "CIVIL", 4, 33, 56, 26, 1],
    ["S1015", "Student_1015", "ECE", 1, 93, 36, 98, 1],
    ["S1016", "Student_1016", "CIVIL", 2, 71, 55, 99, 3],
    ["S1017", "Student_1017", "IT", 1, 39, 38, 88, 5],
    ["S1018", "Student_1018", "CSE", 1, 94, 67, 34, 3],
    ["S1019", "Student_1019", "CSE", 4, 48, 75, 69, 6],
    ["S1020", "Student_1020", "IT", 1, 53, 58, 24, 2],
    ["S1021", "Student_1021", "MECH", 3, 75, 59, 60, 3],
    ["S1022", "Student_1022", "CSE", 1, 54, 50, 53, 0],
    ["S1023", "Student_1023", "CIVIL", 1, 72, 48, 34, 0],
    ["S1024", "Student_1024", "ECE", 4, 68, 79, 72, 5],
    ["S1025", "Student_1025", "IT", 2, 91, 48, 70, 2],
    ["S1026", "Student_1026", "CSE", 4, 96, 89, 76, 5],
    ["S1027", "Student_1027", "MECH", 4, 95, 82, 25, 6],
    ["S1028", "Student_1028", "CSE", 2, 42, 46, 78, 6],
    ["S1029", "Student_1029", "CIVIL", 2, 49, 94, 77, 3],
    ["S1030", "Student_1030", "IT", 3, 36, 40, 61, 6],
    ["S1031", "Student_1031", "CIVIL", 4, 66, 47, 22, 6],
    ["S1032", "Student_1032", "ECE", 2, 92, 80, 81, 1],
    ["S1033", "Student_1033", "IT", 2, 50, 93, 42, 2],
    ["S1034", "Student_1034", "MECH", 1, 34, 47, 61, 2],
    ["S1035", "Student_1035", "IT", 1, 42, 40, 57, 0],
    ["S1036", "Student_1036", "MECH", 4, 41, 44, 93, 2],
    ["S1037", "Student_1037", "CSE", 3, 87, 42, 70, 5],
    ["S1038", "Student_1038", "IT", 1, 38, 81, 28, 6],
    ["S1039", "Student_1039", "ECE", 4, 56, 49, 24, 0],
    ["S1040", "Student_1040", "IT", 3, 88, 42, 47, 2],
    ["S1041", "Student_1041", "CIVIL", 3, 49, 48, 54, 4],
    ["S1042", "Student_1042", "IT", 2, 60, 34, 75, 2],
    ["S1043", "Student_1043", "IT", 4, 86, 60, 26, 0],
    ["S1044", "Student_1044", "MECH", 3, 92, 80, 24, 0],
    ["S1045", "Student_1045", "IT", 2, 91, 88, 55, 0],
    ["S1046", "Student_1046", "CSE", 4, 67, 67, 59, 6],
    ["S1047", "Student_1047", "ECE", 3, 78, 58, 45, 5],
    ["S1048", "Student_1048", "ECE", 2, 59, 31, 53, 1],
    ["S1049", "Student_1049", "MECH", 3, 66, 44, 89, 0],
    ["S1050", "Student_1050", "IT", 3, 64, 36, 41, 2]
]


for row in students[1:]:
    student_data = pd.DataFrame([{
        "Attendance%": row[4],
        "Avg_Marks": row[5],
        "Assignments_Submitted%": row[6],
        "Backlogs": row[7]
    }])
    
    prob = model.predict_proba(student_data)[:, 1][0]
    prediction = "Yes" if prob >= threshold else "No"
    print(f"Student: {row[1]}, Dropout Probability: {prob*100:.2f}%, Prediction: {prediction}")



