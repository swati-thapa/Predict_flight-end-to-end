from flask import Flask, request, render_template
import pickle
from flask_cors import cross_origin

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Total_Stops = int(request.form['Total Stops'])
        Total_Dur = float(request.form['Total Duration'])
        Additional_Info = int(request.form['Additional_Info'])
        if (Additional_Info == '1 Long layover'):
            Additional_Info = 0
        elif (Additional_Info == '1 Short layover'):
            Additional_Info = 1
        elif (Additional_Info == '2 Long layover'):
            Additional_Info = 2
        elif (Additional_Info == 'Business class'):
            Additional_Info = 3
        elif (Additional_Info == 'Change airports'):
            Additional_Info = 4
        elif (Additional_Info == 'In-flight meal not included'):
            Additional_Info = 5
        elif (Additional_Info == 'No Info'):
            Additional_Info = 6
        elif (Additional_Info == 'No check-in baggage included'):
            Additional_Info = 7
        elif (Additional_Info == 'No info'):
            Additional_Info = 8
        else:
            Additional_Info = 9

        Airline = request.form['Airline']
        if (Airline == 'Air-Asia'):
            Airline = 0
        elif (Airline == 'Air India'):
            Airline = 1
        elif (Airline == 'GoAir'):
            Airline = 2
        elif (Airline == 'IndiGo'):
            Airline = 3
        elif (Airline == 'Jet Airways'):
            Airline = 4
        elif (Airline == 'Jet Airways Business'):
            Airline = 5
        elif (Airline == 'Multiple carriers'):
            Airline = 6
        elif (Airline == 'Multiple carriers Premium economy'):
            Airline = 7
        elif (Airline == 'SpiceJet'):
            Airline = 8
        elif (Airline == 'Trujet'):
            Airline = 9
        elif (Airline == 'Vistara'):
            Airline = 10
        else:
            Airline = 11

        Source = request.form['To']
        if (Source == 'Bangalore'):
            Source = 0
        elif (Source == 'Chennai'):
            Source = 1
        elif (Source == 'Delhi'):
            Source = 2
        elif (Source == 'Kolkata'):
            Source= 3
        else:
            Source = 4

        Destination = request.form['From']
        if (Destination == 'Bangalore'):
            Destination = 0
        elif (Destination == 'Cochin'):
            Destination = 1
        elif (Destination == 'Delhi'):
            Destination = 2
        elif (Destination == 'Hyderabad'):
            Destination = 3
        elif (Destination == 'Kolkata'):
            Destination = 4
        else:
            Destination = 5

        Departure_Flag = request.form['Departure Flag']
        if (Departure_Flag == 'Non Peak'):
            Departure_Flag = 0
        else:
            Departure_fFag = 1

        LowCostTag = request.form['Airline']
        if (Airline == 'Air-Asia'):
            Airline = 1
        elif (Airline == 'Air India'):
            Airline = 0
        elif (Airline == 'GoAir'):
            Airline = 1
        elif (Airline == 'IndiGo'):
            Airline = 1
        elif (Airline == 'Jet Airways'):
            Airline = 0
        elif (Airline == 'Jet Airways Business'):
            Airline = 0
        elif (Airline == 'Multiple carriers'):
            Airline = 0
        elif (Airline == 'Multiple carriers Premium economy'):
            Airline = 0
        elif (Airline == 'SpiceJet'):
            Airline = 1
        elif (Airline == 'Trujet'):
            Airline = 1
        elif (Airline == 'Vistara'):
            Airline = 1
        else:
            Airline = 0

        PeakdayFlag= request.form['Weekday']
        if (PeakdayFlag == 'Friday' or PeakdayFlag == 'Saturday' or PeakdayFlag== 'Sunday'):
            PeakdayFlag = 1
        else:
            PeakdayFlag = 0

        prediction = model.predict([[Additional_Info, Airline, Departure_Flag,Source, Destination, Total_Dur, Total_Stops, LowCostTag, PeakdayFlag]])
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_price='Predicted flight ticket price is Rs {}'.format(output))
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
