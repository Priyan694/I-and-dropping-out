async function predict() {
  const student = {
    Dept: document.getElementById("dept").value,
    Year: parseInt(document.getElementById("year").value),
    "Attendance%": parseFloat(document.getElementById("attendance").value),
    Avg_Marks: parseFloat(document.getElementById("marks").value),
    "Assignments_Submitted%": parseFloat(document.getElementById("assignments").value),
    Backlogs: parseInt(document.getElementById("backlogs").value),
    payment_status: parseInt(document.getElementById("payment").value),
    risk_score: parseFloat(document.getElementById("risk").value)
  };

  const res = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(student)
  });

  const data = await res.json();
  document.getElementById("result").innerText =
    `Dropout Prediction: ${data.dropout_prediction} (Prob: ${data.dropout_probability.toFixed(2)})`;
}
