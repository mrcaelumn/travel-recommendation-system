import pandas as pd
import ipywidgets as w
import numpy as np, glob
from urllib.parse import quote
from bing_image_downloader import downloader
from utils import Util
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pickle

def get_rating(x):
    val = x / 5
    if x >= 0 and x <= val:
        return 1
    elif x > val and x <= 2*val:
        return 2
    elif x > 2*val and x <= 3*val:
        return 3
    elif x > 3*val and x <= 4*val:
        return 4
    else:
        return 5
    

def amenities_rating(amenities_pref, newh_df):
    pa_df = pd.DataFrame(amenities_pref,columns=["amenities_pref"])
    newa_df = pd.merge(newh_df, pa_df, left_on='amenities', right_on='amenities_pref')
    
    ameni_comb = newa_df.groupby('id')['amenities'].apply(list).reset_index(name='amenities')
    
    amenities_len = ameni_comb.assign(ameni_len=ameni_comb['amenities'].str.len()).sort_values('ameni_len', ascending=False)

    ameni_df = pd.merge(newh_df, amenities_len, on='id').sort_values('ameni_len', ascending=False)

    ameni_df['rating'] = ameni_df['ameni_len'].apply(get_rating)
    ameni_df.rename(columns = {'amenities_x':'amenities'}, inplace = True)
    # print(ameni_df.head())
    return ameni_df[['id', 'amenities', 'rating']]


def model_train(usr_rating, train=True):
    ## Adding new user info to original dataset for training
    utils = Util()
    u_id_df = utils.read_newline_json("etl/u_id_df/")

    uid_count = u_id_df['user_id'].nunique()

    usrid_df = usr_rating.merge(u_id_df[['id', 'att_id']], on='id', how='inner').assign(usr_id=uid_count)

    usrid_final_df = usrid_df[['usr_id', 'att_id', 'rating']].rename(columns={'usr_id': 'user_id', 'rating': 'user_rating'})

    org_df = u_id_df[['user_id', 'att_id', 'user_rating']]

    usrid_s1, usrid_s2 = usrid_final_df.sample(frac=0.1), usrid_final_df.sample(frac=0.9)

    comb_df = pd.concat([org_df, usrid_s1])

    # Create sparse user-movie matrix
    user_ids = list(np.sort(comb_df.user_id.unique()))
    hotel_ids = list(np.sort(comb_df.att_id.unique()))
    ratings = list(comb_df.user_rating)
    row_indices = [user_ids.index(i) for i in comb_df.user_id]
    col_indices = [hotel_ids.index(i) for i in comb_df.att_id]
    user_hotel_matrix = csr_matrix((ratings, (row_indices, col_indices)), shape=(len(user_ids), len(hotel_ids)))

    # Initialize ALS model
    model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=10)
    
    if train:
        # Fit the model to the user-movie matrix
        model.fit(user_hotel_matrix)

        # Save the model to a file
        with open('als_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        

    return user_hotel_matrix, usrid_s2, hotel_ids


def get_user_data(usr_rating):
    ## Adding new user info to original dataset for training
    utils = Util()
    u_id_df = utils.read_newline_json("etl/u_id_df/")

    uid_count = u_id_df['user_id'].nunique()
    
    usrid_df = usr_rating.merge(u_id_df[['id', 'att_id']], on='id')
    usrid_df['usr_id'] = uid_count
    usrid_final_df = usrid_df[['usr_id', 'att_id', 'rating']]
    
    usrid_s1 = usrid_final_df.sample(frac=0.1, random_state=42)
    usrid_s2 = usrid_final_df.drop(usrid_s1.index)
    
    # Create sparse user-movie matrix
    user_ids = list(np.sort(usrid_final_df.usr_id.unique()))
    hotel_ids = list(np.sort(usrid_final_df.att_id.unique()))
    ratings = list(usrid_final_df.rating)
    row_indices = [user_ids.index(i) for i in usrid_final_df.usr_id]
    col_indices = [hotel_ids.index(i) for i in usrid_final_df.att_id]
    user_hotel_matrix = csr_matrix((ratings, (row_indices, col_indices)), shape=(len(user_ids), len(hotel_ids)))

    return user_hotel_matrix, usrid_s2


def get_hotel_recc(user_matrix, usrid_s2, hotel_ids):
    with open('als_model.pkl', 'rb') as f:
        als_model = pickle.load(f)

    user = usrid_s2['usr_id'].unique()
    # print(user[0])
    recomm_ids, recomm_preds = als_model.recommend(user[0], user_matrix, N=50)
    
    # print("hotel_ids: ", hotel_ids)
    # print("recomm_ids: ", recomm_ids)
    # print("recomm_preds: ", recomm_preds)

    item_ids = [hotel_ids[i] for i in recomm_ids]
    # scores = [i[1] for i in recomm]


    # Create a new DataFrame with the item IDs and scores as columns
    get_attid = pd.DataFrame({'item_id': item_ids, 'prediction': recomm_preds})
    print(get_attid.head())

    utils = Util()
    u_id_df = utils.read_newline_json("etl/u_id_df/")
    print(u_id_df.head())
    u_tempdf = pd.merge(u_id_df[['id', 'att_id']], get_attid, left_on=['att_id'], right_on=['item_id'], how='inner')[['id', 'prediction']]

    return u_tempdf

def get_image(name):
    # print(name)
    util = Util()
    name = name.replace("_", " ")

    dir_path = "downloads"
    try:
        cont = util.check_string(name)
        if cont:
            name = util.remove_special_characters(name)

        downloader.download(name, limit=1,  output_dir=dir_path, adult_filter_off=True, force_replace=False, timeout=60)
        # print("download: ", resp)
        for filename in glob.glob("downloads/{name}/*jpg".format(name=name)) + glob.glob("downloads/{name}/*png".format(name=name)):
            return filename
    except Exception as e:
        # print(e)
        # for filename in glob.glob("downloads/*jpg"):
        #     return filename
        return "etl/hotels.png"
        
def get_hotel_output(days, final):
    fields = ['NAME', 'PRICE', 'RATING', 'EXPERIENCE','LOCATION', 'ADDRESS', "AMENITIES"]
    recommendations = ['Recommendation']

    box_layout = w.Layout(justify_content='space-between',
                        display='flex',
                        flex_flow='row', 
                        align_items='stretch',
                       )
    column_layout = w.Layout(justify_content='space-between',
                        width='75%',
                        display='flex',
                        flex_flow='column', 
                       )
    tab = []
    
    for i in range(len(final['name'])):

        file_image = "etl/hotels.png"

        if final['image'][i] is not None:
            # print("Not None")
            file_image = final['image'][i]

        image = open(file_image, "rb").read()
        name = final['name'][i]
        price= final['price'][i]
        rating= final['rating'][i]
        experience= final['experience'][i]
        loc=final['location'][i]
        # print(loc)
        address=final['address'][i]
        amenities=final['amenities'][i]
        tab.append(w.VBox(children=
                        [
                         w.Image(value=image, format='jpg', width=300, height=400),
                         w.HTML(description=fields[0], value=f"<b><font color='black'>{name}</b>", disabled=True),
                         w.HTML(description=fields[1], value=f"<b><font color='black'>{price}</b>", disabled=True),
                         w.HTML(description=fields[2], value=f"<b><font color='black'>{rating}</b>", disabled=True), 
                         w.HTML(description=fields[3], value=f"<b><font color='black'>{experience}</b>", disabled=True), 
                         w.HTML(description=fields[4], value=f"<b><font color='black'>{loc}</b>", disabled=True),
                         w.HTML(description=fields[5], value=f"<b><a color='black' href='https://maps.google.com/?q={loc}'>{address}</b>", disabled=True)
                        ], layout=column_layout))

    tab_recc = w.Tab(children=tab)
    for i in range(len(tab_recc.children)):
        tab_recc.set_title(i, str('Hotel '+ str(i+1)))
    return tab_recc
