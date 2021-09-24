from flask import Flask
from flask_restful import Api, Resource
from flask_cors import CORS
from main import send_data

app = Flask(__name__)
CORS(app)
api = Api(app)

data_array = []
id = 0


class DrivingInfo(Resource):
    def get(self):
        return data_array

    def post(self):
        global id
        date, start_time, \
        end_time, time_of_drive, \
        avg_speed, max_speed, \
        avg_acc, max_acc, \
        all_maneuvers, breaking_len, \
        accelerating_len, normal_breaking, \
        aggresive_breaking, normal_accelerating, \
        aggresive_accelerating, percent_of_aggresive_breakins, \
        percent_of_normal_breakings, percent_of_aggresive_accelerating, \
        percent_of_normal_accelerating = send_data()
        id += 1
        date = str(date)
        start_time = str(start_time)
        end_time = str(end_time)
        time_of_drive = str(time_of_drive)
        data_array.append({"id": id, "date": date, "start_time": start_time, "end_time": end_time,
                           "time_of_drive": time_of_drive, "avg_speed": avg_speed, "max_speed": max_speed,
                           "avg_acc": avg_acc, "max_acc": max_acc, "all_maneuvers": all_maneuvers,
                           "breaking_len": breaking_len,
                           "accelerating_len": accelerating_len, "normal_breaking": normal_breaking,
                           "aggressive_breaking": aggresive_breaking,
                           "normal_accelerating": normal_accelerating, "aggresive_accelerating": aggresive_accelerating,
                           "percent_of_aggresive_breakings": percent_of_aggresive_breakins,
                           "percent_of_normal_breakings": percent_of_normal_breakings,
                           "percent_of_aggresive_accelerating": percent_of_aggresive_accelerating,
                           "percent_of_normal_accelerating": percent_of_normal_accelerating})
        return data_array


api.add_resource(DrivingInfo, "/info")


if __name__ == '__main__':
    app.run(debug=True)
