"""
Crime Case Prioritization System
--------------------------------
This script generates a synthetic dataset of crime cases and outputs 
the top 10 highest priority cases that need immediate attention.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import nltk
import random
from faker import Faker
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier

# Ensure all required NLTK downloads
print("Setting up NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize faker for generating realistic text
faker = Faker()
random.seed(42)
np.random.seed(42)
Faker.seed(42)

# Define crime types with realistic distributions including Murder cases
crime_types = {
    'Theft': 0.20,
    'Burglary': 0.12,
    'Assault': 0.12,
    'Fraud': 0.08,
    'Drug Offense': 0.08,
    'Vandalism': 0.05,
    'Robbery': 0.06,
    'Harassment': 0.05,
    'Homicide': 0.08,     # Increased from 0.02 to 0.08
    'Murder': 0.07,       # New category specifically for murder cases
    'Kidnapping': 0.01,
    'Other': 0.08
}

# Define locations with realistic distributions
locations = {
    'Downtown': 0.20,
    'North District': 0.15,
    'South District': 0.15,
    'East District': 0.12,
    'West District': 0.12,
    'Central Park': 0.08,
    'Industrial Area': 0.07,
    'Shopping Mall': 0.05,
    'University Campus': 0.04,
    'Transit Station': 0.02
}

# Crime severity weights (higher means more severe)
severity_weights = {
    'Murder': 11,     # Murder gets highest priority (above homicide)
    'Homicide': 10,
    'Kidnapping': 9,
    'Robbery': 8,
    'Assault': 7,
    'Burglary': 6,
    'Fraud': 5,
    'Drug Offense': 4, 
    'Theft': 3,
    'Harassment': 3,
    'Vandalism': 2,
    'Other': 1
}

# Keywords that indicate high priority
high_priority_keywords = [
    'weapon', 'gun', 'knife', 'armed', 'violent', 'injured', 'child', 
    'elderly', 'vulnerable', 'hospital', 'emergency', 'severe', 'death', 
    'threat', 'blood', 'trauma', 'victim', 'wounds', 'attack',
    'murder', 'killed', 'serial', 'premeditated', 'execution', 'brutal',
    'homicide', 'strangled', 'gunshot', 'stabbed', 'beaten', 'poisoned'
]

# Description templates for each crime type
description_templates = {
    'Murder': [
        "Premeditated killing with {weapon} resulting in victim's death. Evidence suggests careful planning.",
        "Serial pattern killing with signature {cause_of_death}. Multiple victims with similar characteristics.",
        "Contract killing execution-style with {cause_of_death}. Professional hit suspected.",
        "Victim brutally murdered with {weapon} during {crime_circumstance}. High level of violence indicated.",
        "Murder-suicide incident where perpetrator killed victim with {cause_of_death} before taking own life."
    ],
    'Theft': [
        "Victim reported {item} stolen from {location_detail}. No witnesses present.",
        "Suspect took victim's {item} when victim was not looking at {location_detail}.",
        "Unknown suspect stole {item} from victim's {location_detail}. No surveillance footage available.",
        "Victim's {item} was taken while they were {activity}. No suspects identified."
    ],
    'Burglary': [
        "Forced entry through {entry_point}. {items_stolen} were stolen. No alarm system triggered.",
        "Suspect broke into residence via {entry_point} and took {items_stolen}. No witnesses.",
        "Home invasion occurred while residents were away. {entry_point} was damaged. {items_stolen} missing.",
        "Business broken into overnight. {entry_point} showed signs of forced entry. {items_stolen} reported missing."
    ],
    'Assault': [
        "Victim sustained {injuries} after altercation with suspect at {location_detail}.",
        "Physical altercation between victim and known suspect resulted in {injuries}.",
        "Unprovoked attack by unknown suspect left victim with {injuries}. Witnesses described suspect as {suspect_description}.",
        "Domestic dispute escalated to physical violence resulting in {injuries} to victim."
    ],
    'Fraud': [
        "Victim reported unauthorized transactions totaling ${amount} on their credit card.",
        "Suspect used victim's identity to open {accounts} accounts, causing financial damage of ${amount}.",
        "Victim was deceived into transferring ${amount} to fraudulent investment scheme.",
        "Business reported accounting discrepancies amounting to ${amount}, suspected internal fraud."
    ],
    'Drug Offense': [
        "Suspect found in possession of {drug_type} during routine traffic stop.",
        "Anonymous tip led to discovery of {drug_type} manufacturing operation.",
        "Suspect observed selling {drug_type} near {location_detail}.",
        "Search warrant execution revealed {drug_type} and paraphernalia in suspect's residence."
    ],
    'Vandalism': [
        "Property damaged by graffiti depicting {graffiti_content}. Estimated repair cost: ${amount}.",
        "Vehicle's {vehicle_part} was deliberately damaged while parked at {location_detail}.",
        "Public {property_type} was destroyed, causing ${amount} in damages. No witnesses.",
        "Business storefront windows broken with {tool}. Surveillance camera captured incident."
    ],
    'Robbery': [
        "Armed suspect demanded money from store clerk, escaped with ${amount}.",
        "Victim was approached by {num_suspects} suspects who took {items_stolen} by force.",
        "Suspect threatened victim with {weapon} before taking {items_stolen}.",
        "Bank robbery involving suspect with {weapon}, escaped with undisclosed amount."
    ],
    'Harassment': [
        "Victim received {num_incidents} threatening messages from known suspect over {time_period}.",
        "Suspect repeatedly made unwanted contact with victim at their workplace.",
        "Victim reported ongoing harassment by neighbor including {harassment_actions}.",
        "Online harassment campaign targeted victim with {harassment_content}."
    ],
    'Homicide': [
        "Victim found deceased with {cause_of_death}. Crime scene indicates signs of struggle.",
        "Domestic dispute escalated resulting in fatal {cause_of_death} to victim.",
        "Multiple gunshot wounds led to victim's death in apparent targeted attack.",
        "Victim discovered deceased in {location_detail} with suspicious circumstances."
    ],
    'Kidnapping': [
        "Child taken by non-custodial parent across state lines, violation of custody agreement.",
        "Victim forced into vehicle by {num_suspects} unknown suspects at {location_detail}.",
        "Ransom demand received after business executive was abducted outside office.",
        "Brief abduction during carjacking incident, victim released unharmed."
    ],
    'Other': [
        "Suspicious activity reported near {location_detail}, investigation ongoing.",
        "Noise complaint escalated requiring police intervention at residential property.",
        "Trespassing incident at closed business premises after hours.",
        "Violation of restraining order when suspect approached protected person."
    ]
}

# Details for filling templates
items = ["wallet", "purse", "phone", "laptop", "bicycle", "jewelry", "cash", "credit cards", "identity documents", "vehicle"]
location_details = ["parking lot", "public restroom", "restaurant", "bus stop", "sidewalk", "store", "gym", "library", "office", "park bench"]
activities = ["shopping", "dining", "working", "walking", "exercising", "commuting", "distracted by phone", "talking with others"]
entry_points = ["rear window", "front door", "garage door", "basement window", "side entrance", "balcony door", "skylight", "pet door"]
items_stolen_groups = ["electronics and jewelry", "cash and electronics", "personal documents and valuables", "artwork and antiques", "electronics and cash", "jewelry and watches"]
injuries = ["facial bruising", "lacerations requiring stitches", "broken nose", "concussion", "minor cuts and bruises", "fractured ribs", "black eye", "defensive wounds on hands"]
suspect_descriptions = ["tall male in dark clothing", "medium-built female with distinctive tattoos", "young male in hooded sweatshirt", "older male with facial hair", "person wearing mask and gloves"]
fraud_amounts = [500, 1200, 2500, 5000, 7500, 10000, 15000, 25000, 50000, 100000]
account_types = ["credit card", "bank", "loan", "online shopping", "mobile phone", "utility", "rental", "investment"]
drug_types = ["marijuana", "cocaine", "methamphetamine", "heroin", "prescription pills", "synthetic drugs", "hallucinogens"]
graffiti_content = ["gang symbols", "political messages", "obscene images", "tags and signatures", "racial slurs", "anarchist symbols"]
vehicle_parts = ["windows", "tires", "hood", "doors", "mirrors", "paint", "headlights", "interior"]
property_types = ["park bench", "statue", "playground equipment", "bus shelter", "street sign", "trash bins", "public bathroom", "memorial"]
vandalism_tools = ["rocks", "baseball bat", "spray paint", "hammer", "crowbar", "skateboard", "brick", "tire iron"]
weapons = ["handgun", "knife", "baseball bat", "pepper spray", "taser", "broken bottle", "blunt object", "implied weapon in pocket"]
harassment_actions = ["noise disturbances", "verbal threats", "property damage", "stalking behavior", "false complaints to authorities", "spreading rumors"]
harassment_content = ["false accusations", "private information", "manipulated images", "threats of violence", "derogatory comments"]
causes_of_death = ["gunshot wounds", "stab wounds", "blunt force trauma", "strangulation", "poisoning", "severe beating"]
crime_circumstances = ["home invasion", "robbery gone wrong", "domestic dispute", "drug deal", "gang conflict", "hate crime"]

def generate_case_id(idx):
    """Generate unique case ID with format CR-YYYY-XXXXX"""
    year = np.random.randint(2018, 2023)
    return f"CR-{year}-{idx+10000}"

def generate_filing_date():
    """Generate a realistic filing date within the last 3 years"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3*365)  # 3 years back
    random_days = np.random.randint(0, (end_date - start_date).days)
    return start_date + timedelta(days=random_days)

def generate_crime_type():
    """Generate crime type based on defined distributions"""
    return np.random.choice(
        list(crime_types.keys()),
        p=list(crime_types.values())
    )

def generate_location():
    """Generate location based on defined distributions"""
    return np.random.choice(
        list(locations.keys()),
        p=list(locations.values())
    )

def generate_description(crime_type):
    """Generate a realistic description based on crime type"""
    # Choose a random template for the crime type
    template = random.choice(description_templates[crime_type])
    
    # Replace placeholders with random values depending on crime type
    if crime_type == 'Theft':
        return template.format(
            item=random.choice(items),
            location_detail=random.choice(location_details),
            activity=random.choice(activities)
        )
    elif crime_type == 'Burglary':
        return template.format(
            entry_point=random.choice(entry_points),
            items_stolen=random.choice(items_stolen_groups)
        )
    elif crime_type == 'Assault':
        return template.format(
            injuries=random.choice(injuries),
            location_detail=random.choice(location_details),
            suspect_description=random.choice(suspect_descriptions)
        )
    elif crime_type == 'Fraud':
        return template.format(
            amount=random.choice(fraud_amounts),
            accounts=random.choice(account_types)
        )
    elif crime_type == 'Drug Offense':
        return template.format(
            drug_type=random.choice(drug_types),
            location_detail=random.choice(location_details)
        )
    elif crime_type == 'Vandalism':
        return template.format(
            graffiti_content=random.choice(graffiti_content),
            amount=random.randint(100, 2000),
            vehicle_part=random.choice(vehicle_parts),
            location_detail=random.choice(location_details),
            property_type=random.choice(property_types),
            tool=random.choice(vandalism_tools)
        )
    elif crime_type == 'Robbery':
        return template.format(
            amount=random.randint(50, 2000),
            num_suspects=random.randint(1, 3),
            items_stolen=random.choice(items),
            weapon=random.choice(weapons)
        )
    elif crime_type == 'Harassment':
        return template.format(
            num_incidents=random.randint(3, 20),
            time_period=f"{random.randint(1, 6)} months",
            harassment_actions=random.choice(harassment_actions),
            harassment_content=random.choice(harassment_content)
        )
    elif crime_type == 'Murder':
        return template.format(
            weapon=random.choice(weapons),
            cause_of_death=random.choice(causes_of_death),
            crime_circumstance=random.choice(crime_circumstances)
        )
    elif crime_type == 'Homicide':
        return template.format(
            cause_of_death=random.choice(causes_of_death),
            location_detail=random.choice(location_details)
        )
    elif crime_type == 'Kidnapping':
        return template.format(
            num_suspects=random.randint(1, 3),
            location_detail=random.choice(location_details)
        )
    else:  # Other
        return template.format(
            location_detail=random.choice(location_details)
        )

def generate_cases(num_samples=1000):
    """Generate synthetic crime cases"""
    print(f"Generating {num_samples} synthetic crime cases...")
    data = []
    
    for i in range(num_samples):
        case_id = generate_case_id(i)
        filing_date = generate_filing_date()
        crime_type = generate_crime_type()
        location = generate_location()
        description = generate_description(crime_type)
        
        data.append({
            'CaseID': case_id,
            'FilingDate': filing_date,
            'CrimeType': crime_type,
            'Location': location,
            'Description': description
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} crime cases with {len(df['CrimeType'].unique())} different crime types")
    return df

def analyze_sentiment(df):
    """Analyze sentiment of case descriptions"""
    print("Analyzing sentiment of case descriptions...")
    sid = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis to each description
    sentiment_scores = []
    for description in df['Description']:
        scores = sid.polarity_scores(description)
        sentiment_scores.append(scores)
    
    # Add sentiment scores to dataframe
    df['sentiment_compound'] = [score['compound'] for score in sentiment_scores]
    df['sentiment_positive'] = [score['pos'] for score in sentiment_scores]
    df['sentiment_negative'] = [score['neg'] for score in sentiment_scores]
    df['sentiment_neutral'] = [score['neu'] for score in sentiment_scores]
    
    print(f"✓ Completed sentiment analysis: {len(df)} cases analyzed")
    return df

def calculate_priority_features(df):
    """Calculate features that contribute to priority"""
    print("Calculating priority features...")
    
    # 1. Crime Severity (based on crime type)
    df['severity_score'] = df['CrimeType'].map(lambda x: severity_weights.get(x, 1) / 11)  # Using full 0-11 scale
    
    # 2. Recency Score (more recent cases get higher priority)
    current_date = datetime.now().date()
    
    # Calculate days since filing manually without using dt accessor
    df['days_since_filing'] = df['FilingDate'].apply(lambda x: (current_date - x).days)
    max_days = 3 * 365  # 3 years
    df['recency_score'] = df['days_since_filing'].apply(lambda x: 1 - min(x / max_days, 1))
    df['recency_score'] = df['recency_score'].clip(0, 1)  # Ensure between 0 and 1
    
    # 3. Keyword Score (count of high priority keywords in description)
    df['keyword_matches'] = df['Description'].apply(
        lambda x: sum(1 for keyword in high_priority_keywords if keyword.lower() in x.lower())
    )
    df['keyword_score'] = df['keyword_matches'] / 5  # Normalize by dividing by 5
    df['keyword_score'] = df['keyword_score'].clip(0, 1)  # Cap at 1
    
    # 4. Sentiment Impact (negative sentiment may indicate more serious cases)
    df['sentiment_impact'] = (df['sentiment_negative'] * 0.7) + ((1 - df['sentiment_compound']) * 0.3)
    
    print(f"✓ Priority features calculated for {len(df)} cases")
    return df

def calculate_priority_score(df):
    """Calculate overall priority score"""
    print("Calculating final priority scores...")
    
    # Weighted combination of features
    df['priority_score'] = (
        (df['severity_score'] * 0.40) +  # Crime severity is most important
        (df['recency_score'] * 0.25) +   # Recency is very important
        (df['keyword_score'] * 0.20) +   # Keywords indicate urgency
        (df['sentiment_impact'] * 0.15)  # Sentiment provides additional context
    )
    
    # Scale to 0-100% for better interpretability
    df['priority_percentage'] = df['priority_score'] * 100
    
    # Binary priority (high/low) using 0.5 as threshold
    df['priority_level'] = df['priority_score'].apply(lambda x: 'HIGH' if x > 0.5 else 'LOW')
    
    # Sort by priority score in descending order
    df = df.sort_values(by='priority_score', ascending=False)
    
    print(f"✓ Priority scores calculated. Identified {len(df[df['priority_level'] == 'HIGH'])} high priority cases")
    return df

def generate_report(df, top_n=10):
    """Generate a report of the top N priority cases"""
    print(f"\n{'='*80}")
    print(f"                TOP {top_n} CASES REQUIRING IMMEDIATE ATTENTION")
    print(f"{'='*80}")
    
    top_cases = df.head(top_n)
    
    for i, case in enumerate(top_cases.itertuples(), 1):
        print(f"\nCASE {i} - PRIORITY SCORE: {case.priority_percentage:.1f}%")
        print(f"Case ID: {case.CaseID}")
        print(f"Filed: {case.FilingDate.strftime('%Y-%m-%d')} ({case.days_since_filing} days ago)")
        print(f"Crime Type: {case.CrimeType}")
        print(f"Location: {case.Location}")
        
        # Format description to be readable
        desc = case.Description
        if len(desc) > 150:
            desc = desc[:150] + "..."
        print(f"Description: {desc}")
        
        # Show key factors
        print("\nPriority Factors:")
        print(f"  • Crime Severity: {case.severity_score*10:.1f}/10")
        print(f"  • Case Recency: {case.recency_score*100:.1f}%")
        print(f"  • Priority Keywords: {case.keyword_matches} present")
        print(f"  • Sentiment: {case.sentiment_compound:.2f} (negative: {case.sentiment_negative:.2f})")
        print("-" * 80)
    
    # Summary statistics
    print("\nCASE PRIORITIZATION SUMMARY:")
    high_priority = df[df['priority_level'] == 'HIGH']
    print(f"• {len(high_priority)} high priority cases identified out of {len(df)} total cases")
    
    # Crime type distribution in high priority cases
    high_priority_crimes = high_priority['CrimeType'].value_counts()
    print("\nMost common high priority crime types:")
    for crime_type, count in high_priority_crimes.head(5).items():
        print(f"• {crime_type}: {count} cases ({count/len(high_priority)*100:.1f}% of high priority)")
    
    # Average days since filing for high priority cases
    avg_days_high = high_priority['days_since_filing'].mean()
    avg_days_low = df[df['priority_level'] == 'LOW']['days_since_filing'].mean()
    print(f"\nHigh priority cases are on average {avg_days_high:.1f} days old")
    print(f"Low priority cases are on average {avg_days_low:.1f} days old")
    
    print(f"\n{'='*80}")
    print(f"                         REPORT GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    
    # Save results to Excel
    try:
        df.to_excel('crime_cases_prioritized.xlsx', index=False)
        print("\nFull prioritized case list saved to 'crime_cases_prioritized.xlsx'")
    except Exception as e:
        df.to_csv('crime_cases_prioritized.csv', index=False)
        print("\nFull prioritized case list saved to 'crime_cases_prioritized.csv'")
    
    # Generate visualization
    try:
        # Plot crime type distribution in high priority cases
        plt.figure(figsize=(12, 6))
        ax = high_priority_crimes.head(7).plot(kind='bar')
        plt.title('Distribution of Crime Types in High Priority Cases')
        plt.xlabel('Crime Type')
        plt.ylabel('Number of High Priority Cases')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_file = 'high_priority_crime_distribution.png'
        plt.savefig(chart_file)
        print(f"A chart of high priority cases by crime type has been saved to '{chart_file}'")
    except Exception as e:
        print(f"Could not generate visualization: {e}")

def main():
    print("\n" + "="*80)
    print("                    CRIME CASE PRIORITIZATION SYSTEM")
    print("="*80 + "\n")
    
    # Step 1: Generate synthetic crime data
    df = generate_cases(num_samples=1000)
    
    # Step 2: Analyze sentiment of descriptions
    df = analyze_sentiment(df)
    
    # Step 3: Calculate priority features
    df = calculate_priority_features(df)
    
    # Step 4: Calculate final priority scores
    df = calculate_priority_score(df)
    
    # Step 5: Generate prioritization report
    generate_report(df, top_n=10)

if __name__ == "__main__":
    main()