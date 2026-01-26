import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# load data
@st.cache_data
def load_data():
    # Load data
    file_path = "/home/user/Desktop/komendra/EduPro/EduPro Online Platform.xlsx"
    users_df = pd.read_excel(file_path, sheet_name="Users")
    courses_df = pd.read_excel(file_path, sheet_name="Courses")
    transactions_df = pd.read_excel(file_path, sheet_name="Transactions")
    teachers_df = pd.read_excel(file_path, sheet_name="Teachers")

    # Build learner features
    learner_base = (
        transactions_df
        .groupby("UserID")
        .agg(
            total_transactions=("TransactionID", "count")
        )
        .reset_index()
    )

    courses_per_learner = (
        transactions_df
        .groupby("UserID")["CourseID"]
        .nunique()
        .reset_index(name="total_courses_enrolled")
    )

    learner_features = learner_base.merge(
        courses_per_learner,
        on="UserID",
        how="left"
    )

    transaction_frequency = (
        transactions_df
        .groupby("UserID")["TransactionID"]
        .count()
        .reset_index(name="enrollment_frequency")
    )

    learner_features = learner_features.merge(
        transaction_frequency,
        on="UserID",
        how="left"
    )

    # Preferred category
    txn_courses = transactions_df.merge(
        courses_df[["CourseID", "CourseCategory"]],
        on="CourseID",
        how="left"
    )

    preferred_category = (
        txn_courses
        .groupby("UserID")["CourseCategory"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index(name="preferred_course_category")
    )

    learner_features = learner_features.merge(
        preferred_category,
        on="UserID",
        how="left"
    )

    # Preferred level
    txn_courses_level = transactions_df.merge(
        courses_df[["CourseID", "CourseLevel"]],
        on="CourseID",
        how="left"
    )

    preferred_level = (
        txn_courses_level
        .groupby("UserID")["CourseLevel"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index(name="preferred_course_level")
    )

    learner_features = learner_features.merge(
        preferred_level,
        on="UserID",
        how="left"
    )

    # Avg spending
    avg_spending = (
        transactions_df
        .groupby("UserID")["Amount"]
        .mean()
        .reset_index(name="avg_spending")
    )

    learner_features = learner_features.merge(
        avg_spending,
        on="UserID",
        how="left"
    )

    # Category diversity
    category_diversity = (
        txn_courses
        .groupby("UserID")["CourseCategory"]
        .nunique()
        .reset_index(name="category_diversity_score")
    )

    learner_features = learner_features.merge(
        category_diversity,
        on="UserID",
        how="left"
    )

    # Learning depth index
    txn_with_level = transactions_df.merge(
        courses_df[["CourseID", "CourseLevel"]],
        on="CourseID",
        how="left"
    )

    level_counts = (
        txn_with_level
        .pivot_table(
            index="UserID",
            columns="CourseLevel",
            values="CourseID",
            aggfunc="count",
            fill_value=0
        )
        .reset_index()
    )

    for col in ["Beginner", "Intermediate", "Advanced"]:
        if col not in level_counts.columns:
            level_counts[col] = 0

    level_counts["learning_depth_index"] = (
        (level_counts["Intermediate"] + level_counts["Advanced"]) /
        level_counts["Beginner"].replace(0, np.nan)
    )

    learning_depth = level_counts[["UserID", "learning_depth_index"]]

    learner_features = learner_features.merge(
        learning_depth,
        on="UserID",
        how="left"
    )

    # Handle missing values
    learner_features["learning_depth_index"] = learner_features["learning_depth_index"].fillna(learner_features["learning_depth_index"].max() + 1)

    # Preprocessing for clustering
    numerical_features = [
        "total_courses_enrolled",
        "enrollment_frequency",
        "avg_spending",
        "category_diversity_score",
        "learning_depth_index"
    ]

    categorical_features = [
        "preferred_course_category",
        "preferred_course_level"
    ]

    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(learner_features[numerical_features])
    scaled_numerical_df = pd.DataFrame(
        scaled_numerical,
        columns=numerical_features,
        index=learner_features.index
    )

    categorical_encoded_df = pd.get_dummies(learner_features[categorical_features], drop_first=False)

    final_feature_matrix = pd.concat([scaled_numerical_df, categorical_encoded_df], axis=1)

    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(final_feature_matrix)
    learner_features["cluster"] = cluster_labels

    # Cluster course popularity
    txn_with_cluster = transactions_df.merge(
        learner_features[["UserID", "cluster"]],
        on="UserID",
        how="left"
    )

    cluster_course_popularity = (
        txn_with_cluster
        .groupby(["cluster", "CourseID"])
        .size()
        .reset_index(name="enrollment_count")
    )

    cluster_course_popularity = cluster_course_popularity.merge(
        courses_df[["CourseID", "CourseName", "CourseCategory", "CourseLevel", "CourseRating"]],
        on="CourseID",
        how="left"
    )

    return learner_features, cluster_course_popularity

def recommend_courses(user_id, top_n=5):
    if user_id not in learner_features["UserID"].values:
        return f"UserID {user_id} not found in learner data."
    
    user_cluster = learner_features.loc[
        learner_features["UserID"] == user_id, "cluster"
    ].values[0]
    
    user_category = learner_features.loc[
        learner_features["UserID"] == user_id, "preferred_course_category"
    ].values[0]
    
    user_level = learner_features.loc[
        learner_features["UserID"] == user_id, "preferred_course_level"
    ].values[0]
    
    cluster_courses = cluster_course_popularity[
        cluster_course_popularity["cluster"] == user_cluster
    ]
    
    filtered_courses = cluster_courses[
        (cluster_courses["CourseCategory"] == user_category) &
        (cluster_courses["CourseLevel"] == user_level)
    ]
    
    if filtered_courses.empty:
        filtered_courses = cluster_courses
    
    ranked_courses = filtered_courses.sort_values(
        by=["enrollment_count", "CourseRating"],
        ascending=False
    )
    
    return ranked_courses[
        ["CourseID", "CourseName", "CourseCategory", "CourseLevel", "CourseRating"]
    ].head(top_n)

learner_features, cluster_course_popularity = load_data()


# app title and sidebar
st.title("EduPro Course Recommendation System")
st.sidebar.header("User Input Features")

page=st.sidebar.radio(
    "Go to",
    [
        "Learner Profile Explorer",
        "Cluster Overview",
        "Personalized Recommendations",
        "Cluster Comparison"
    ]

)

# Module 1: Learner Profile Explorer
if page == "Learner Profile Explorer":
    st.header("Learner Profile Explorer")

    user_id = st.selectbox(
        "Select Learner ID",
        learner_features['UserID'].unique()
    )

    learner=learner_features[learner_features['UserID']==user_id]
    st.subheader("Learner Details")
    st.dataframe(learner)


# Module 2: Cluster Overview
elif page == "Cluster Overview":
    st.header("Cluster Overview")

    cluster_summary = (
        learner_features
        .groupby('cluster')
        .mean(numeric_only=True)
        .round(2)

    )

    st.dataframe(cluster_summary)


# Module 3: Personalized Recommendations
elif page == "Personalized Recommendations":
    st.header("personalized course recommendations")

    user_id=st.selectbox(
        "select UserID",
        learner_features['UserID'].unique()
    )

    top_n=st.slider("Number of recommendations",1,10,5)

    if st.button("Get Recommendations"):
        recs=recommend_courses(user_id,top_n)
        if isinstance(recs, str):
            st.error(recs)
        else:
            st.dataframe(recs)



# Module 4: Cluster Comparison

elif page == "Cluster Comparison":
    st.header("cluster comparison")
    feature=st.selectbox(
        "select feature to compare",
        ['total_courses_enrolled',
         'enrollment_frequency',
         'avg_spending',
         'category_diversity_score',
         'learning_depth_index']
    )        
    comparison=(
        learner_features
        .groupby('cluster')[feature]
        .mean()
        .reset_index()
    )

    st.bar_chart(
        comparison.set_index('cluster')
    )