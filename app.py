from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def home():
    file_path = "attendance/attendance.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        data = df.to_dict(orient="records")
    else:
        data = []
    return render_template("index.html", records=data)

if __name__ == "__main__":
    app.run()
