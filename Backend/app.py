from flask import Flask, request
import pandas as pd
from flask_cors import CORS
from flask import jsonify

df = pd.read_csv("test.csv")


# def recommend_hotels(pref1, pref2, pref3):

#     preferred_clusters = [pref1, pref2, pref3]
#     weights = {preferred_clusters[0]: 0.5,
#                preferred_clusters[1]: 0.3, preferred_clusters[2]: 0.2}
#     top_hotels = {}

#     topNhotels = 5
#     for cluster in preferred_clusters:
#         cluster_df = df[df['assigned_cluster'] == cluster].sort_values(
#             'confidence_score', ascending=False)
#         top_hotels[cluster] = list(cluster_df['Hotel_Name'][:topNhotels])
#         print("top hotels are", top_hotels)

#     recommended_hotels = []
#     for i in range(topNhotels):
#         for cluster in preferred_clusters:
#             if len(top_hotels[cluster]) > i:
#                 hotel = top_hotels[cluster][i]
#                 confidence_score = df[df['Hotel_Name'] ==
#                                       hotel]['confidence_score'].values[0]
#                 hotel_ratings = df[df['Hotel_Name']
#                                    == hotel]['Average_Score'].values[0]
#                 recommended_hotels.append(
#                     (hotel, confidence_score * weights[cluster], hotel_ratings))

#     recommended_hotels.sort(key=lambda x: x[1], reverse=True)
#     print("recommended hotels are", recommended_hotels)
#     return [hotel for hotel in recommended_hotels[:topNhotels]]

def recommend_hotels(pref1, pref2, pref3):

    # for two nulls
    if pref1 != 'Null' and pref2 == 'Null' and pref3 == 'Null':
        preferred_clusters = [pref1]
        weights = {preferred_clusters[0]: 1}
        top_hotels = {}
        for cluster in preferred_clusters:
            cluster_df = df[df['assigned_cluster'] == cluster].sort_values(
                'confidence_score', ascending=False)
            top_hotels[cluster] = list(cluster_df['Hotel_Name'][:5])

        recommended_hotels = []
        for i in range(5):
            hotel = top_hotels[cluster][i]
            confidence_score = df[df['Hotel_Name'] ==
                                  hotel]['confidence_score'].values[0]
            hotel_ratings = df[df['Hotel_Name'] ==
                               hotel]['Average_Score'].values[0]
            recommended_hotels.append(
                (hotel, round(confidence_score * weights[cluster], 1), hotel_ratings))

    # for one null
    elif pref1 != 'Null' and pref2 != 'Null' and pref3 == 'Null':
        preferred_clusters = [pref1, pref2]
        weights = {preferred_clusters[0]: 0.6, preferred_clusters[1]: 0.4}
        top_hotels = {}

        for cluster in preferred_clusters:
            cluster_df = df[df['assigned_cluster'] == cluster].sort_values(
                'confidence_score', ascending=False)
            top_hotels[cluster] = list(cluster_df['Hotel_Name'][:5])

        recommended_hotels = []
        for i in range(5):
            hotel = top_hotels[cluster][i]
            confidence_score = df[df['Hotel_Name'] ==
                                  hotel]['confidence_score'].values[0]
            hotel_ratings = df[df['Hotel_Name'] ==
                               hotel]['Average_Score'].values[0]
            recommended_hotels.append(
                (hotel, round(confidence_score * weights[cluster], 1), hotel_ratings))

    else:
        preferred_clusters = [pref1, pref2, pref3]
        weights = {
            preferred_clusters[0]: 0.5, preferred_clusters[1]: 0.3, preferred_clusters[2]: 0.2}
        top_hotels = {}

        for cluster in preferred_clusters:
            cluster_df = df[df['assigned_cluster'] == cluster].sort_values(
                'confidence_score', ascending=False)
            top_hotels[cluster] = list(cluster_df['Hotel_Name'][:5])

        recommended_hotels = []
        for i in range(5):
            for cluster in preferred_clusters:
                if len(top_hotels[cluster]) > i:
                    hotel = top_hotels[cluster][i]
                    confidence_score = df[df['Hotel_Name']
                                          == hotel]['confidence_score'].values[0]
                    hotel_ratings = df[df['Hotel_Name'] ==
                                       hotel]['Average_Score'].values[0]
                    recommended_hotels.append(
                        (hotel, round(confidence_score * weights[cluster], 1), hotel_ratings))

    recommended_hotels.sort(key=lambda x: x[1], reverse=True)
    return jsonify([hotel for hotel in recommended_hotels[:5]])


app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict')
def predict():
    print(request.args)
    pref1 = request.args.get("pref1", "Null")
    pref2 = request.args.get("pref2", "Null")
    pref3 = request.args.get("pref3", "Null")

    predictions = recommend_hotels(pref1, pref2, pref3)

    return predictions


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
