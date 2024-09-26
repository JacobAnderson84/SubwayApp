import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("subway.csv")

def reset_app():
    st.rerun()

#Normalization with min-max scaling
scaler = MinMaxScaler()

#Minmaxscaler for calories column
calories = df["Calories"]
calories_reshaped = np.array(calories).reshape(-1, 1)
calories_scaled = scaler.fit_transform(calories_reshaped)
df["Calories Scaled"] = calories_scaled

#min-max for total fat
total_fat = df["Total Fat (g)"]
total_fat_reshaped = np.array(total_fat).reshape(-1, 1)
total_fat_scaled = scaler.fit_transform((total_fat_reshaped))
df["Total Fat Scaled"] = total_fat_scaled

#min-max for saturated fat
saturated_fat = df["Saturated Fat (g)"]
saturated_fat_reshaped = np.array(saturated_fat).reshape(-1, 1)
saturated_fat_scaled = scaler.fit_transform((saturated_fat_reshaped))
df["Saturated Fat Scaled"] = saturated_fat_scaled

#min-max trans fat
trans_fat = df["Trans Fat (g)"]
trans_fat_reshaped = np.array(trans_fat).reshape(-1, 1)
trans_fat_scaled = scaler.fit_transform(trans_fat_reshaped)
df["Trans Fat Scaled"] = trans_fat_scaled

#min-max cholesterol
cholesterol = df["Cholesterol (mg)"]
cholesterol_reshaped = np.array(cholesterol).reshape(-1, 1)
cholesterol_scaled = scaler.fit_transform(cholesterol_reshaped)
df["Cholesterol Scaled"] = cholesterol_scaled

#min-max sodium
sodium = df["Sodium (mg)"]
sodium_reshaped = np.array(sodium).reshape(-1, 1)
sodium_scaled = scaler.fit_transform(sodium_reshaped)
df["Sodium Scaled"] = sodium_scaled

#min-max carbs
carbs = df["Carbs (g)"]
carbs_reshaped = np.array(carbs).reshape(-1, 1)
carbs_scaled = scaler.fit_transform(carbs_reshaped)
df["Carbs Scaled"] = carbs_scaled

#mix max Dietary Fiber
dietary_fiber = df["Dietary Fiber (g)"]
dietary_fiber_reshaped = np.array(dietary_fiber).reshape(-1, 1)
dietary_fiber_scaled = scaler.fit_transform(dietary_fiber_reshaped)
df["Dietary Fiber Scaled"] = dietary_fiber_scaled

#min max sugar
sugar = df["Sugars (g)"]
sugar_reshaped = np.array(sugar).reshape(-1, 1)
sugar_scaled = scaler.fit_transform(sugar_reshaped)
df["Sugars Scaled"] = sugar_scaled

#min max protien
protein = df["Protein (g)"]
protein_reshaped = np.array(protein).reshape(-1, 1)
protein_scaled = scaler.fit_transform(protein_reshaped)
df["Protein Scaled"] = protein_scaled

#min max weight watcher points
weight_watcher_points = df["Weight Watchers Pnts"]
weight_watcher_points_reshaped = np.array(weight_watcher_points).reshape(-1, 1)
weight_watcher_points_scaled = scaler.fit_transform(weight_watcher_points_reshaped)
df["Weight Watchers Pnts Scaled"] = weight_watcher_points_scaled

#st.dataframe(df, height=500)



#FINDING INERTIA
# def optomise_k_means(data, max_k):
#     means = []
#     intertias = []
#
#     for k in range(1, max_k):
#         kmeans = KMeans(n_clusters=k)
#         kmeans.fit(data)
#
#         means.append(k)
#         intertias.append(kmeans.inertia_)
#     #generate Elbow plot
#     fig, ax = plt.subplots(figsize=(10,5))
#     ax.plot(means, intertias, 'bo-', marker='o')
#     ax.set_title('Elboy method for optomising K')
#     ax.set_xlabel('number of clusters')
#     ax.set_ylabel('Inertia')
#     ax.grid(True)
#
#     st.pyplot(fig)
#
# optomise_k_means(df[["Calories Scaled", "Sodium Scaled"]], 10)
# optomise_k_means(df[["Trans Fat Scaled", "Sodium Scaled"]], 10)
st.title("Subway Sandwich Recommendation App")
st.write("With this app you can select one of your favorite subway sandwiches and two nutrional values that you are tracking. The application will take your choices and give you the top 3 sandwiches most like your selection.")

st.write("Below is a bar chart showing the amount of items per Category")
#bar chart to look at different cateogries of items in the data
category_counts = df["Category"].value_counts()
st.bar_chart(category_counts)

feature_options = ['Calories', 'Total Fat (g)', 'Saturated Fat (g)', 'Trans Fat (g)', 'Cholesterol (mg)', 'Sodium (mg)', 'Carbs (g)',
                   'Dietary Fiber (g)', 'Sugars (g)', 'Protein (g)', 'Weight Watchers Pnts']
#when taking user selection need to map the actual column names
feature_mapping = {
    'Calories': 'Calories Scaled',
    'Total Fat (g)': 'Total Fat Scaled',
    'Saturated Fat (g)': 'Saturated Fat Scaled',
    'Trans Fat (g)': 'Trans Fat Scaled',
    'Cholesterol (mg)': 'Cholesterol Scaled',
    'Sodium (mg)': 'Sodium Scaled',
    'Carbs (g)': 'Carbs Scaled',
    'Dietary Fiber (g)': 'Dietary Fiber Scaled',
    'Sugars (g)': 'Sugars Scaled',
    'Protein (g)': 'Protein Scaled',
    'Weight Watchers Pnts': 'Weight Watchers Pnts Scaled'
}
st.write("Using this form, make your selection and press \"Go\"")
st.write("You will be presented with a scatter chart showing all the items and how they are clustered into groups "
         "using K-means algorithm. Two histogram charts will show a spread of the items for different nutritional value "
         "ranges. At the bottom are the recommended sandwiches and a Reset button to start over. Have fun!!")
#form for user input
with st.form("Sandwich Selection"):
    #select a sandwich they like
    selected_item = st.selectbox('Select an item you like:', df['Item'])
    #selecting nutrional features
    selected_features = st.multiselect('Select 2 nutritional features', list(feature_mapping.keys()),
                                       default=['Calories', 'Carbs (g)'])

    submitted = st.form_submit_button("Go")
    print(selected_features)
    print(len(selected_features))

if submitted:
    if len(selected_features) != 2:
        st.write("Please select 2 features")
    else:
        #get the selected nutrional values column names
        selected_columns = [feature_mapping[feature] for feature in selected_features]
        print(f'Selected columns:', selected_columns)
        st.write(selected_columns)
        #create a data frame of the selected columns
        df_selected = df[selected_columns]
        #Kmeans and assigning the labels back onto the main Dataframe
        kmeans = KMeans(n_clusters=6, random_state=84)
        kmeans.fit(df_selected)
        df['Cluster'] = kmeans.labels_

        # get selected columns names for data variable
        col1_name = df_selected.columns[0]
        col2_name = df_selected.columns[1]

        #scatter plot
        data = df[[col1_name, col2_name, 'Cluster']]
        print("This is the DATA *******************************")
        print(data)
        st.scatter_chart(
            data,
            x=col1_name,
            y=col2_name,
            color='Cluster'
        )
        st.dataframe(df, height=500)
        #histograms plots of the chosen nutrinoal values
        #plot for the first selected feature
        st.write(f"Histogram of {selected_features[0]} values")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_features[0]])
        ax.set_title((f"Distribution of {selected_features[0]} values"))
        st.pyplot(fig)

        st.write(f"Histogram of {selected_features[1]} values")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_features[1]])
        ax.set_title((f"Distribution of {selected_features[1]} values"))
        st.pyplot(fig)

        #find the selected_item cluster
        selected_item_cluster = df.loc[df['Item'] == selected_item, 'Cluster'].values[0]

        #create list of items in that cluster
        similar_items = df[df['Cluster'] == selected_item_cluster]
        #remove selected_item from the similar_items list
        similar_items = similar_items[similar_items['Item'] != selected_item]
        print(similar_items)

        #value of the selected_item based on the selected_columns
        selected_item_data = df[df['Item'] == selected_item][selected_columns].values.flatten()
        #values of similar_items based on the selected_columns
        similar_items_data = similar_items[selected_columns]
        #calculate distance between selcted_item and similar_items based on the selected columns
        similar_items['Distance'] = similar_items_data[selected_columns].apply(lambda row:((row - selected_item_data) ** 2).sum() ** 0.5, axis=1)
        print(similar_items)
        #sort distnace to find the most similar items from similar_items to the selected_item
        recommended_items = similar_items.sort_values(by='Distance').head(3)
        st.write(f"These are the three most similar items to {selected_item}:")
        st.write(recommended_items[['Category', 'Item']])


if st.button("Reset"):
    reset_app()


    # #get cluster of the selected item
    # selected_item_cluster = df[df['Item'] == selected_item]['Cluster'].values[0]
    #
    # #filter items from the same cluster
    # similar_items = df[df['Cluster'] == selected_item_cluster]
    #
    # #remove selected item from the reccomendations
    # similar_items = similar_items[similar_items['Item'] != selected_item]
    #
    # #calculate the distance between the selected item and others in the same cluster
    # selected_item_data = df[df['Item'] == selected_item][selected_columns].values.flatten()
    # similar_item_selected = similar_items[selected_columns]
    # similar_items['Distance'] = similar_item_selected[selected_columns].apply(lambda row:((row - selected_item_data) ** 2).sum() ** 0.5, axis=1)
    #
    # #sort distance to find the most similar items
    # recommended_items = similar_items.sort_values(by='Distance').head(3)
    #
    # #display sandwiches
    # st.write(f"Here are the three most similar items to {selected_item}:")
    # st.write(recommended_items[['Cat,'Item']])
