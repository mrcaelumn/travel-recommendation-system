import pickle
import pandas as pd
from google_images_download import google_images_download
import re
from urllib.parse import quote
import glob
from ipywidgets import HBox, VBox
from ipywidgets import Layout, widgets
from IPython.display import display, IFrame, HTML

class get_recomendation:
    def __init__(self) -> None:
        weight_path = "hybrid_model/weight/"
        # load the model from a file
        with open(weight_path + 'knn_weights.pkl', 'rb') as f:
            self.knn_model = pickle.load(f)
        
        # load the model weights from a file
        with open(weight_path + 'kmeans_weights.pkl', 'rb') as f:
            self.kmeans_model = pickle.load(f)
        
        self.reviews_data = pd.read_csv('hybrid_model/data/final_data_train.csv')
        self.reviews_data_kmeans = pd.read_csv('hybrid_model/data/final_data_reviews_kmeans.csv')
        self.reviews_data_kmeans['cluster'] = self.kmeans_model.predict(self.reviews_data_kmeans[['review_star','sentiment_score', 'state', 'city']])
        self.df_train_knn_final = self.reviews_data[['user_id','business_id', 'review_star']].groupby(['user_id', 'business_id'])['review_star'].mean().unstack().fillna(0)
        self.business_data = pd.read_csv('hybrid_model/data/business_restaurant_clean.csv')
    
    def get_image(self, name):
        name = re.sub(' ','_',name)
        response = google_images_download.googleimagesdownload()
        args_list = ["keywords", "keywords_from_file", "prefix_keywords", "suffix_keywords",
                "limit", "format", "color", "color_type", "usage_rights", "size",
                "exact_size", "aspect_ratio", "type", "time", "time_range", "delay", "url", "single_image",
                "output_directory", "image_directory", "no_directory", "proxy", "similar_images", "specific_site",
                "print_urls", "print_size", "print_paths", "metadata", "extract_metadata", "socket_timeout",
                "thumbnail", "language", "prefix", "chromedriver", "related_images", "safe_search", "no_numbering",
                "offset", "no_download"]
        args = {}
        for i in args_list:
            args[i]= None
        args["keywords"] = name
        args['limit'] = 1
        params = response.build_url_parameters(args)
        url = 'https://www.google.com/search?q=' + quote(name) + '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' + params + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg'
        try:
            response.download(args)
            for filename in glob.glob("downloads/{name}/*jpg".format(name=name))+glob.glob("downloads/{name}/*png".format(name=name)):
                return filename
        except:
            for filename in glob.glob("downloads/*jpg"):
                return filename
    
    def get_kmeans_recc(self,user_id):
        reviews_data_kmeans = self.reviews_data_kmeans.loc[self.reviews_data_kmeans['user_id'] == user_id]
        # print(reviews_data_kmeans)
        
        kmeans_rec = self.kmeans_model.predict(reviews_data_kmeans[['review_star','sentiment_score', 'state', 'city']])
        kmeans_rec = list(dict.fromkeys(kmeans_rec))
        # print(kmeans_rec)
        data = self.reviews_data_kmeans[self.reviews_data_kmeans['cluster'].isin(kmeans_rec)]

        data = data[['review_star', 'business_id']]
        grouped_data = data.groupby(['business_id']).agg({'review_star': 'mean'})
        sorted_data = grouped_data.sort_values(by='review_star', ascending=False)
        kmeans_recs = sorted_data.head(100).to_dict()['review_star']

        return kmeans_recs
    
    def get_knn_recc(self,user_id):
        self.reviews_data_knn = pd.read_csv('hybrid_model/data/final_data_reviews_knn.csv')
        # Create empty dataframe with 0 values
        input_dict = self.reviews_data_knn.to_dict()
        reviews_data_user = self.reviews_data.loc[self.reviews_data['user_id'] == user_id]
        reviews_data_user = reviews_data_user[["business_id", "review_star"]].groupby('business_id')['review_star'].mean().reset_index()
        # print(reviews_data_user)

        # print(input_df)
        # Create a dictionary to store the mapping of business_id to review_star
        mapping = dict(zip(reviews_data_user['business_id'], reviews_data_user['review_star']))

        # Map the ratings from table A to the corresponding columns in table B
        for col in self.reviews_data_knn .columns:
            if col in mapping:
                # print("insert: ", col, mapping[col])
                input_dict[col] = mapping[col]
            else:
                input_dict[col] = 0

        # print(input_dict)
        # convert dictionary to dataframe
        input_df = pd.DataFrame(input_dict, index=[0])

        # print(input_df)
        

        # Get the K nearest neighbors of the user
        _, indices = self.knn_model.kneighbors(input_df, n_neighbors=100)
        # print(indices)
        # Get the top recommended restaurants from the neighbors
        recommended_restaurants = []
        for neighbor_index in indices[0]:
            neighbor_ratings = self.df_train_knn_final.iloc[neighbor_index]
            recommended_restaurants += list(neighbor_ratings[neighbor_ratings > 1].index)

        recommended_restaurants = list(set(recommended_restaurants))
        # print(recommended_restaurants)
        # Filter the reviews dataframe to only include the business IDs in the business_ids array
        # print(self.df_train_knn_final)
        filtered_reviews = self.reviews_data[self.reviews_data['business_id'].isin(recommended_restaurants)]

        # Group the filtered reviews dataframe by business_id and calculate the mean review_star for each group
        review_stars = filtered_reviews.groupby('business_id')['review_star'].mean().reset_index()
        sorted_data = review_stars.sort_values('review_star', ascending=False)
        # print(sorted_data)
        sorted_data.set_index('business_id', inplace=True)
        knn_recs = sorted_data.head(100).to_dict()['review_star']

        return knn_recs

    def recc(self, user_id, top_n = 10):
        # Assuming you have the KNN and K-means recommendation dictionaries stored as knn_recs and kmeans_recs respectively

        user = self.reviews_data.sample(n = 1, replace = True)
        user_id = user[['user_id']].values.tolist()[0][0]
        # print(user_id)
        kmeans_recs = self.get_kmeans_recc(user_id)
        knn_recs = self.get_knn_recc(user_id)
        
        

        # Combine the two dictionaries into a single dictionary with the restaurant names as keys and the weighted average scores as values
        combined_recs = []
        for business_id in set(knn_recs.keys()) | set(kmeans_recs.keys()):
            knn_score = knn_recs.get(business_id, 0)
            kmeans_score = kmeans_recs.get(business_id, 0)
            weighted_avg = (knn_score + kmeans_score) / 2
            combined_recs.append({
                'business_id': business_id, 
                'review_star': weighted_avg, 
            })
            
        # Deduplicate the recommendations and sort by the weighted average score in descending order
        sorted_recs = sorted(combined_recs, key=lambda x: x['review_star'], reverse=True)

        # create an empty list to store the results
        results = []

        # loop through the ratings array
        # print(sorted_recs)
        for rating in sorted_recs:
            # look up the location and name for the business in the business dataframe
            row = self.business_data.loc[self.business_data['business_id'] == rating['business_id']]
            address = row['address'].iloc[0]
            name = row['name'].iloc[0]
            city = row['city'].iloc[0]
            state = row['state'].iloc[0]
            latitude = row['latitude'].iloc[0]
            longitude = row['longitude'].iloc[0]
            categories = row['categories'].iloc[0]
            stars = row['stars'].iloc[0]
            hours = row['hours'].iloc[0]
            
            # add the business_id, rating_star, location, and name to the results list
            results.append({
                'business_id': rating['business_id'], 
                'review_star': rating['review_star'], 
                'name': name,
                'address': address,
                'city': city, 
                'state': state, 
                'latitude': latitude, 
                'longitude': longitude, 
                'categories': categories,
                'stars': stars,
                'hours': hours,
                })


        # display the result
        # print(results)
        # Create a dictionary to hold the final recommendations for each meal
        final_recs = {"breakfast": [], "lunch": [], "dinner": []}

        # Iterate over the sorted recommendations and assign them to the appropriate meal, up to a maximum of 2 recommendations per meal
        for rest in results:
            if len(final_recs["breakfast"]) < 2:
                final_recs["breakfast"].append(rest)
            elif len(final_recs["lunch"]) < 2:
                final_recs["lunch"].append(rest)
            elif len(final_recs["dinner"]) < 2:
                final_recs["dinner"].append(rest)
            else:
                break
        print(final_recs)

        return final_recs
    
    def final_output(days, final):
        time = ['breakfast', 'lunch', 'dinner']
        fields = ['NAME', 'CATEGORY', 'LOCATION', 'PRICE', 'RATING']
        recommendations = ['Recommendation 1:', 'Recommendation 2:']

        box_layout = Layout(justify_content='space-between',
                            display='flex',
                            flex_flow='row', 
                            align_items='stretch',
                        )
        column_layout = Layout(justify_content='space-between',
                            width='75%',
                            display='flex',
                            flex_flow='column', 
                        )
        tab = []
        for i in range(days):

            start_idx = i*4
            end_idx = (i+1)*4
            images = final['image'][start_idx:end_idx]
                
                
            final_images = []
            for i in images:
                image = "etl/attractions.png"
                if i is not None:
                    image = i
                final_images.append(image)

            images = [open(i, "rb").read() for i in final_images]
    
        


            name = [re.sub('_',' ',i).capitalize() for i in final['name'][start_idx:end_idx]]

            category = [re.sub('_',' ',i).capitalize() for i in final['category'][start_idx:end_idx]]
            location = ["("+str(i[0])+","+str(i[1])+")" for i in final['location'][start_idx:end_idx]]
            price = [str(i) for i in final['price'][start_idx:end_idx]]
            rating = [str(i) for i in final['rating'][start_idx:end_idx]]
            tab.append(VBox(children=
                            [HBox(children=
                                [VBox(children=
                                        [widgets.HTML(value = f"<b><font color='orange'>{time[0]}</b>"),
                                        widgets.HTML(value = f"<b><font color='purple'>{recommendations[0]}</b>"),
                                        widgets.Image(value=images[0], format='jpg', width=300, height=400), 
                                        widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[0]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[0]}</b>", disabled=True),
                                        widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[0]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[0]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[0]}</b>", disabled=True)
                                        ], layout=column_layout), 
                                    VBox(children=
                                        [widgets.HTML(value = f"<b><font color='orange'>{time[1]}</b>"), 
                                        widgets.HTML(value = f"<b><font color='purple'>{recommendations[0]}</b>"),
                                        widgets.Image(value=images[2], format='jpg', width=300, height=400), 
                                        widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[2]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[2]}</b>", disabled=True),
                                        widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[2]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[2]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[2]}</b>", disabled=True)
                                        ], layout=column_layout)
                                ], layout=box_layout),

                            HBox(children=
                                [VBox(children=
                                        [widgets.HTML(value = f"<b><font color='purple'>{recommendations[1]}</b>"),
                                        widgets.Image(value=images[1], format='jpg', width=300, height=400), 
                                        widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[1]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[1]}</b>", disabled=True),
                                        widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[1]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[1]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[1]}</b>", disabled=True)
                                        ], layout=column_layout), 
                                    VBox(children=
                                        [widgets.HTML(value = f"<b><font color='purple'>{recommendations[1]}</b>"),
                                        widgets.Image(value=images[3], format='jpg', width=300, height=400), 
                                        widgets.HTML(description=fields[0], value=f"<b><font color='black'>{name[3]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[1], value=f"<b><font color='black'>{category[3]}</b>", disabled=True),
                                        widgets.HTML(description=fields[2], value=f"<b><font color='black'>{location[3]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[3], value=f"<b><font color='black'>{price[3]}</b>", disabled=True), 
                                        widgets.HTML(description=fields[4], value=f"<b><font color='black'>{rating[3]}</b>", disabled=True)
                                        ], layout=column_layout),
                                ], layout=box_layout)

                            ]))

        tab_recc = widgets.Tab(children=tab)
        for i in range(len(tab_recc.children)):
            tab_recc.set_title(i, str('Day '+ str(i+1)))
        return tab_recc