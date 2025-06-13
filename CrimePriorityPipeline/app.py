import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import time
import joblib
import nltk

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Set page configuration
st.set_page_config(page_title="Case Prioritization Dashboard",
                   page_icon="🚨",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Main title
st.title("Case Prioritization Dashboard")
st.markdown("### Prioritize crime cases based on urgency and severity")


# Load existing data if available, otherwise generate new data
@st.cache_data
def load_data():
    try:
        # Try to load from Excel if it exists
        if os.path.exists('crime_cases_prioritized.xlsx'):
            df = pd.read_excel('crime_cases_prioritized.xlsx')
            return df
        else:
            # Run the prioritization script to generate data
            import subprocess
            subprocess.run(['python', 'generate_top_cases.py'])
            if os.path.exists('crime_cases_prioritized.xlsx'):
                df = pd.read_excel('crime_cases_prioritized.xlsx')
                return df
            else:
                st.error("Failed to generate or load the data!")
                return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Function to enhance case descriptions with more detailed information
def enhance_descriptions(df):
    """Add more details to case descriptions where they're too short"""

    # Add contextual information based on crime type
    context_by_crime = {
        'Theft':
        "Officer notes indicate a pattern of similar thefts in the area. Victim reports no suspicious individuals in the vicinity before the incident. No evidence of forced entry or tools used in commission of the crime. Investigators canvassed area for witnesses with no success so far.",
        'Burglary':
        "Property was secured at the time of the break-in. Fingerprint evidence was collected at the scene. No security camera footage available. No signs of other properties in the vicinity being targeted. Police have increased patrols in the neighborhood.",
        'Assault':
        "Medical report indicates the injuries are consistent with victim's statement. No prior history between victim and suspect. Several bystanders witnessed the incident. The area is known to have had similar incidents in the past month.",
        'Fraud':
        "Victim has already filed reports with credit bureaus. No other victims have come forward with similar complaints against the same suspect. Digital evidence is being analyzed by the cybercrime unit. Financial institution has opened their own investigation.",
        'Drug Offense':
        "Lab analysis confirmed substance type and purity. Area is known for increased drug activity. Suspect has no prior drug-related charges. Drug task force is monitoring the location for additional suspicious activity.",
        'Vandalism':
        "City maintenance has been notified for cleanup and repair. No political or gang-related motives apparent in the damage. Area has experienced an increase in similar incidents in recent weeks. No witnesses have come forward despite public request.",
        'Robbery':
        "Victim received medical evaluation at the scene. Description of suspect matches other recent robbery reports. Officers searched the area but were unable to locate suspect. Stolen items had no identifying features that would aid in recovery.",
        'Harassment':
        "Victim has documented previous incidents in a personal log. No restraining order currently in effect. Communications received through multiple channels. Mental health resources have been offered to the victim.",
        'Homicide':
        "Forensic team collected extensive evidence from the scene. Weapon has not been recovered. Medical examiner has not yet determined exact time of death. Family members have been notified and interviewed. Multiple detectives assigned to the case.",
        'Murder':
        "Chief detective has been assigned to lead investigation with full team resources. Extensive crime scene analysis underway with specialized forensics. Background checks on all known associates in progress. Media briefing scheduled. All departmental resources have been authorized for use in solving this case. Multiple units collaborating on evidence collection and witness interviews. Case marked for daily command staff review.",
        'Kidnapping':
        "AMBER alert was issued within 30 minutes. Vehicle description has been circulated to all patrol units. Family members report no recent threats or suspicious activities. Border patrol and neighboring jurisdictions have been notified.",
        'Other':
        "Case has been referred to the appropriate specialized unit. Evidence collected has been sent for processing. Follow-up investigation scheduled. Patrol officers advised to be on alert for similar incidents."
    }

    for i, row in df.iterrows():
        crime_type = row['CrimeType']
        current_desc = row['Description']

        # If description is too short, add more context
        if len(current_desc) < 200 and crime_type in context_by_crime:
            df.at[
                i,
                'Description'] = f"{current_desc} {context_by_crime[crime_type]}"

    return df


# Load and process the data
df = load_data()

if df is not None:
    # Convert dates to datetime format
    df['FilingDate'] = pd.to_datetime(df['FilingDate'])

    # Enhance descriptions
    df = enhance_descriptions(df)

    # Calculate days since filing - handle this more carefully
    current_date = datetime.now().date()
    df['days_since_filing'] = df['FilingDate'].apply(
        lambda x: (current_date - x.date()).days)

    # Sidebar with filters
    st.sidebar.header("Filters")

    # Crime type filter
    crime_types = sorted(df['CrimeType'].unique())
    selected_crimes = st.sidebar.multiselect("Select Crime Types",
                                             crime_types,
                                             default=crime_types)

    # Priority level filter
    priority_levels = sorted(df['priority_level'].unique())
    selected_priorities = st.sidebar.multiselect("Select Priority Levels",
                                                 priority_levels,
                                                 default=priority_levels)

    # Date range filter
    min_date = df['FilingDate'].min().date()
    max_date = df['FilingDate'].max().date()
    date_range = st.sidebar.date_input("Select Date Range",
                                       value=(min_date, max_date),
                                       min_value=min_date,
                                       max_value=max_date)

    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Apply filters
    filtered_df = df[(df['CrimeType'].isin(selected_crimes))
                     & (df['priority_level'].isin(selected_priorities)) &
                     (df['FilingDate'].dt.date >= start_date) &
                     (df['FilingDate'].dt.date <= end_date)]

    # Display number of filtered cases
    st.sidebar.metric("Filtered Cases", len(filtered_df))
    st.sidebar.metric(
        "High Priority Cases",
        len(filtered_df[filtered_df['priority_level'] == 'HIGH']))

    # Main dashboard
    tab1, tab2, tab3 = st.tabs(
        ["Dashboard Overview", "Top Priority Cases", "Case Explorer"])

    with tab1:
        st.header("Crime Case Prioritization Overview")

        # Row 1 - Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            high_priority_count = len(
                filtered_df[filtered_df['priority_level'] == 'HIGH'])
            high_priority_percentage = high_priority_count / len(
                filtered_df) * 100 if len(filtered_df) > 0 else 0
            st.metric("High Priority Cases", high_priority_count,
                      f"{high_priority_percentage:.1f}%")

        with col2:
            avg_priority_score = filtered_df['priority_percentage'].mean()
            st.metric("Average Priority Score", f"{avg_priority_score:.1f}%")

        with col3:
            avg_days_since = filtered_df['days_since_filing'].mean()
            st.metric("Average Case Age", f"{avg_days_since:.1f} days")

        with col4:
            most_common_crime = filtered_df['CrimeType'].value_counts(
            ).index[0] if not filtered_df.empty else "N/A"
            crime_count = filtered_df['CrimeType'].value_counts(
            ).iloc[0] if not filtered_df.empty else 0
            st.metric("Most Common Crime", most_common_crime,
                      f"{crime_count} cases")

        # Row 2 - Charts
        col1, col2 = st.columns(2)

        with col1:
            # Priority distribution by crime type
            st.subheader("Priority Distribution by Crime Type")
            crime_priority_counts = filtered_df.groupby(
                ['CrimeType', 'priority_level']).size().unstack().fillna(0)

            fig, ax = plt.subplots(figsize=(10, 6))
            crime_priority_counts.plot(kind='bar',
                                       stacked=True,
                                       ax=ax,
                                       color=['green', 'red'])
            plt.title('Case Priority Distribution by Crime Type')
            plt.xlabel('Crime Type')
            plt.ylabel('Number of Cases')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Priority Level')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            # Priority score distribution
            st.subheader("Priority Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(filtered_df['priority_percentage'],
                         bins=20,
                         kde=True,
                         ax=ax)
            plt.axvline(x=50,
                        color='red',
                        linestyle='--',
                        label='Priority Threshold (50%)')
            plt.title('Distribution of Priority Scores')
            plt.xlabel('Priority Score (%)')
            plt.ylabel('Number of Cases')
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Row 3 - More charts
        col1, col2 = st.columns(2)

        with col1:
            # Filing date trend
            st.subheader("Case Filing Trend")
            filing_counts = filtered_df.resample('ME', on='FilingDate').size()

            fig, ax = plt.subplots(figsize=(10, 6))
            filing_counts.plot(kind='line', marker='o', ax=ax)
            plt.title('Monthly Case Filing Trend')
            plt.xlabel('Month')
            plt.ylabel('Number of Cases')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            # Sentiment analysis
            st.subheader("Sentiment Analysis Overview")

            if 'sentiment_compound' in filtered_df.columns:
                # Group by priority level and calculate mean sentiment
                sentiment_by_priority = filtered_df.groupby('priority_level')[[
                    'sentiment_compound', 'sentiment_positive',
                    'sentiment_negative', 'sentiment_neutral'
                ]].mean()

                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_by_priority.plot(kind='bar', ax=ax)
                plt.title('Average Sentiment Scores by Priority Level')
                plt.xlabel('Priority Level')
                plt.ylabel('Average Score')
                plt.xticks(rotation=0)
                plt.legend(title='Sentiment Type')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Sentiment analysis data not available")

    with tab2:
        st.header("Top 10 Highest Priority Cases")

        # Get top 10 priority cases from filtered data
        top_cases = filtered_df.sort_values(by='priority_percentage',
                                            ascending=False).head(10)

        # Display each top case in an expander
        for i, case in enumerate(top_cases.itertuples(), 1):
            with st.expander(
                    f"Case {i}: {case.CaseID} - {case.CrimeType} - Priority Score: {case.priority_percentage:.1f}%"
            ):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown(
                        f"**Filed:** {case.FilingDate.strftime('%Y-%m-%d')} ({case.days_since_filing} days ago)"
                    )
                    st.markdown(f"**Location:** {case.Location}")
                    st.markdown(f"**Priority Level:** {case.priority_level}")

                    # If sentiment data is available
                    if hasattr(case, 'sentiment_compound'):
                        st.markdown("**Sentiment Analysis:**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(
                                f"Compound: {case.sentiment_compound:.3f}")
                            st.markdown(
                                f"Positive: {case.sentiment_positive:.3f}")
                        with col_b:
                            st.markdown(
                                f"Negative: {case.sentiment_negative:.3f}")
                            st.markdown(
                                f"Neutral: {case.sentiment_neutral:.3f}")

                with col2:
                    st.markdown("**Case Description:**")
                    st.markdown(case.Description)

                    # Priority factors
                    st.markdown("**Priority Factors:**")
                    if hasattr(case, 'severity_score'):
                        st.markdown(
                            f"• Crime Severity: {case.severity_score*10:.1f}/10"
                        )
                    if hasattr(case, 'recency_score'):
                        st.markdown(
                            f"• Case Recency: {case.recency_score*100:.1f}%")
                    if hasattr(case, 'keyword_matches'):
                        st.markdown(
                            f"• Priority Keywords: {case.keyword_matches} present"
                        )
                    if hasattr(case, 'sentiment_compound'):
                        st.markdown(
                            f"• Sentiment: {case.sentiment_compound:.2f} (negative: {case.sentiment_negative:.2f})"
                        )

    with tab3:
        st.header("Case Explorer")

        # Search functionality
        search_term = st.text_input("Search cases by keyword or case ID:")

        if search_term:
            search_results = df[
                df['CaseID'].str.contains(search_term, case=False)
                | df['Description'].str.contains(search_term, case=False)
                | df['CrimeType'].str.contains(search_term, case=False)
                | df['Location'].str.contains(search_term, case=False)]

            st.write(
                f"Found {len(search_results)} cases matching '{search_term}'")

            if not search_results.empty:
                st.dataframe(search_results[[
                    'CaseID', 'FilingDate', 'CrimeType', 'Location',
                    'priority_percentage', 'priority_level'
                ]])

        # Advanced case browser
        st.subheader("Browse All Cases")

        # Create a sortable and filterable dataframe
        st.dataframe(
            filtered_df[[
                'CaseID', 'FilingDate', 'CrimeType', 'Location',
                'priority_percentage', 'priority_level', 'Description'
            ]],
            column_config={
                "FilingDate":
                st.column_config.DateColumn("Filing Date"),
                "priority_percentage":
                st.column_config.ProgressColumn(
                    "Priority Score",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Description":
                st.column_config.TextColumn(
                    "Description",
                    width="large",
                ),
            },
            hide_index=True,
        )

        # Option to download the data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data as CSV",
                           csv,
                           "crime_cases_filtered.csv",
                           "text/csv",
                           key='download-csv')

else:
    st.error(
        "No data available. Please run the crime case prioritization script first."
    )

    if st.button("Generate Crime Case Data"):
        with st.spinner(
                "Generating crime case data and calculating priorities..."):
            # Execute the script to generate data
            import subprocess
            result = subprocess.run(['python', 'generate_top_cases.py'],
                                    capture_output=True,
                                    text=True)

            if result.returncode == 0:
                st.success(
                    "Data generated successfully! Please refresh the page.")
                st.text(result.stdout)
            else:
                st.error("Failed to generate data.")
                st.text(result.stderr)
