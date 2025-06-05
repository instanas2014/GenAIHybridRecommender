import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import HybridPersonalizationSystem as HPS

# Configure page
st.set_page_config(
    page_title=" Gen AI AUgment Deep CF Personalized Recommender",
    page_icon="🎬",
    layout="wide"
)

# Initialize the system
@st.cache_resource
def load_recommendation_system():
    return HPS.HybridPersonalizationSystem.load_model("./movie_recommender_model")

system = load_recommendation_system()

# App Header
st.title("🎬 Personalized Recommendation System")
st.markdown("Get personalized movie recommendations based on your preferences and context!")

# Sidebar for user inputs
with st.sidebar:
    st.header("🎯 Your Preferences")
    
    # User ID input
    user_id = st.number_input("User ID", min_value=1, max_value=1000, value=1, step=1)
    
    st.subheader("Recent Activity")
    genres = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    recent_activity = st.multiselect(
        "What genres have you been watching recently?",
        genres,
        default=['Action', 'Sci-Fi']
    )
    
    st.subheader("Context Information")
    time_context = st.selectbox(
        "When are you planning to watch?",
        ['morning', 'afternoon', 'evening', 'late_night'],
        index=2
    )
    
    device = st.selectbox(
        "What device will you use?",
        ['desktop', 'mobile', 'tablet', 'tv'],
        index=1
    )
    
    duration_pref = st.selectbox(
        "How much time do you have?",
        ['15min', '30min', '60min', '90min', '120min', '180min+'],
        index=1
    )
    
    # Number of recommendations
    top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    
    # Generate recommendations button
    generate_btn = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if generate_btn or 'recommendations' not in st.session_state:
        # Prepare user context
        user_context = {
            'recent_activity': recent_activity,
            'time_context': time_context,
            'session_info': {
                'device': device,
                'duration': duration_pref
            }
        }
        
        # Get recommendations
        with st.spinner("🔍 Finding perfect recommendations for you..."):
            recommendations = system.get_personalized_recommendations(
                user_id, 
                context=user_context, 
                top_k=top_k
            )
            st.session_state.recommendations = recommendations
            st.session_state.user_context = user_context
    
    # Display recommendations
    if 'recommendations' in st.session_state:
        recommendations = st.session_state.recommendations

        st.subheader(f"🎯 Top {len(recommendations)} Recommendations for User {user_id}")
        
        # Create recommendation cards
        for i, rec in enumerate(recommendations, 1):
            with st.container():
                st.markdown(f"### {i}. {rec['title']}")
                
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    st.markdown(f"**Genres:** {rec['genres']}")
                
                with col_b:
                    st.metric("Rating", f"{rec['rating']}")
                
                with col_c:
                    st.metric("Deep Score", f"{rec['deep_score']:.3f}")
                    st.metric("LLM Adjusted Score", f"{rec['adjusted_score']:.3f}")
                    st.metric("Final Score", f"{rec['final_score']:.3f}")
       
                # Progress bar for recommendation score
                st.progress(rec['final_score'])
                
                st.markdown("---")

with col2:
    if 'recommendations' in st.session_state:
        st.subheader("📊 Recommendation Analytics")
        
        recommendations = st.session_state.recommendations
        
        # Score distribution chart
        scores = [rec['final_score'] for rec in recommendations]
        titles = [rec['title'] for rec in recommendations]
        
        fig_scores = go.Figure(data=[
            go.Bar(x=titles, y=scores, 
                  marker_color='lightblue',
                  text=[f"{score:.3f}" for score in scores],
                  textposition='auto')
        ])
        fig_scores.update_layout(
            title="Recommendation Scores",
            xaxis_title="Movies",
            yaxis_title="Score",
            height=400,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Genre distribution
        all_genres = []
        for rec in recommendations:
            all_genres.extend(rec['genres'])
        
        from collections import Counter
        genre_counts = Counter(all_genres)
        
        fig_genres = px.pie(
            values=list(genre_counts.values()),
            names=list(genre_counts.keys()),
            title="Genre Distribution in Recommendations"
        )
        st.plotly_chart(fig_genres, use_container_width=True)
        
        # Context summary
        st.subheader("🔍 Your Context")
        user_context = st.session_state.user_context
        
        st.info(f"""
        **Recent Activity:** {', '.join(user_context['recent_activity']) if user_context['recent_activity'] else 'None'}
        
        **Time:** {user_context['time_context'].title()}
        
        **Device:** {user_context['session_info']['device'].title()}
        
        **Duration:** {user_context['session_info']['duration']}
        """)

# Footer with system info
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.metric("Total Movies in Database", "10")

with col_footer2:
    if 'recommendations' in st.session_state:
        avg_score = np.mean([rec['final_score'] for rec in st.session_state.recommendations])
        st.metric("Average Recommendation Score", f"{avg_score:.3f}")

with col_footer3:
    st.metric("User ID", user_id)

# Additional features section
with st.expander("🔧 System Information"):
    st.markdown("""
    ### How the Recommendation System Works:
    
    1. **User Context Analysis**: The system analyzes your recent viewing activity, preferred time, device, and available duration.
    
    2. **Content Matching**: Movies are scored based on how well they match your preferences and context.
    
    3. **Personalization**: Recommendations are tailored to your specific situation (e.g., shorter movies for mobile viewing).
    
    4. **Ranking**: Results are ranked by recommendation score, showing the most relevant content first.
    
    ### Features:
    - 📱 Device-aware recommendations
    - ⏰ Time-based suggestions  
    - 🎭 Genre preference matching
    - ⭐ Quality-based scoring
    - 📊 Visual analytics
    
    *Note: This is a demo version. Replace the MockRecommendationSystem with your actual recommendation engine.*
    """)
    st.markdown(f"""
                ### Oriignal Recommendation JSON: ###
                {recommendations}
                """)

# OPTIONAL - to simplify loading henc not adding it
    
# # Export recommendations
# if 'recommendations' in st.session_state:
#     st.subheader("💾 Export Recommendations")
    
#     # Convert to DataFrame for export
#     df_recommendations = pd.DataFrame(st.session_state.recommendations)
    
#     col_export1, col_export2 = st.columns(2)
    
#     with col_export1:
#         csv = df_recommendations.to_csv(index=False)
#         st.download_button(
#             label="📄 Download as CSV",
#             data=csv,
#             file_name=f"recommendations_user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             mime="text/csv"
#         )
    
#     with col_export2:
#         json_data = df_recommendations.to_json(orient='records', indent=2)
#         st.download_button(
#             label="📋 Download as JSON",
#             data=json_data,
#             file_name=f"recommendations_user_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#             mime="application/json"
#         )