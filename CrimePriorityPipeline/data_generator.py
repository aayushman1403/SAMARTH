import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os

class DataGenerator:
    """Class to generate synthetic crime data"""
    
    def __init__(self, num_samples=1000, random_state=42):
        self.num_samples = num_samples
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        self.fake = Faker()
        Faker.seed(random_state)
        
        # Define crime types with realistic distributions and increased murder cases
        self.crime_types = {
            'Theft': 0.20,
            'Burglary': 0.12,
            'Assault': 0.12,
            'Fraud': 0.08,
            'Drug Offense': 0.08,
            'Vandalism': 0.05,
            'Robbery': 0.06,
            'Harassment': 0.05,
            'Homicide': 0.08,     
            'Murder': 0.07,      
            'Kidnapping': 0.01,
            'Other': 0.08
        }
        
        # Define locations with realistic distributions
        self.locations = {
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
        
        # Define descriptions templates for each crime type to generate more realistic descriptions
        self.description_templates = {
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
            'Murder': [
                "Premeditated killing with {weapon} resulting in victim's death. Evidence suggests careful planning.",
                "Serial pattern killing with signature {cause_of_death}. Multiple victims with similar characteristics.",
                "Contract killing execution-style with {cause_of_death}. Professional hit suspected.",
                "Victim brutally murdered with {weapon} during {crime_circumstance}. High level of violence indicated.",
                "Murder-suicide incident where perpetrator killed victim with {cause_of_death} before taking own life."
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
        
        # Items that could be stolen
        self.items = ["wallet", "purse", "phone", "laptop", "bicycle", "jewelry", "cash", "credit cards", "identity documents", "vehicle"]
        
        # Location details
        self.location_details = ["parking lot", "public restroom", "restaurant", "bus stop", "sidewalk", "store", "gym", "library", "office", "park bench"]
        
        # Activities during theft
        self.activities = ["shopping", "dining", "working", "walking", "exercising", "commuting", "distracted by phone", "talking with others"]
        
        # Entry points for burglary
        self.entry_points = ["rear window", "front door", "garage door", "basement window", "side entrance", "balcony door", "skylight", "pet door"]
        
        # Items commonly stolen in burglaries
        self.items_stolen_groups = ["electronics and jewelry", "cash and electronics", "personal documents and valuables", "artwork and antiques", "electronics and cash", "jewelry and watches"]
        
        # Possible injuries from assault
        self.injuries = ["facial bruising", "lacerations requiring stitches", "broken nose", "concussion", "minor cuts and bruises", "fractured ribs", "black eye", "defensive wounds on hands"]
        
        # Suspect descriptions
        self.suspect_descriptions = ["tall male in dark clothing", "medium-built female with distinctive tattoos", "young male in hooded sweatshirt", "older male with facial hair", "person wearing mask and gloves"]
        
        # Fraud amounts
        self.fraud_amounts = [500, 1200, 2500, 5000, 7500, 10000, 15000, 25000, 50000, 100000]
        
        # Account types for fraud
        self.account_types = ["credit card", "bank", "loan", "online shopping", "mobile phone", "utility", "rental", "investment"]
        
        # Drug types
        self.drug_types = ["marijuana", "cocaine", "methamphetamine", "heroin", "prescription pills", "synthetic drugs", "hallucinogens"]
        
        # Graffiti content
        self.graffiti_content = ["gang symbols", "political messages", "obscene images", "tags and signatures", "racial slurs", "anarchist symbols"]
        
        # Vehicle parts
        self.vehicle_parts = ["windows", "tires", "hood", "doors", "mirrors", "paint", "headlights", "interior"]
        
        # Property types
        self.property_types = ["park bench", "statue", "playground equipment", "bus shelter", "street sign", "trash bins", "public bathroom", "memorial"]
        
        # Vandalism tools
        self.vandalism_tools = ["rocks", "baseball bat", "spray paint", "hammer", "crowbar", "skateboard", "brick", "tire iron"]
        
        # Robbery weapons
        self.weapons = ["handgun", "knife", "baseball bat", "pepper spray", "taser", "broken bottle", "blunt object", "implied weapon in pocket"]
        
        # Harassment details
        self.harassment_actions = ["noise disturbances", "verbal threats", "property damage", "stalking behavior", "false complaints to authorities", "spreading rumors"]
        self.harassment_content = ["false accusations", "private information", "manipulated images", "threats of violence", "derogatory comments"]
        
        # Causes of death
        self.causes_of_death = ["gunshot wounds", "stab wounds", "blunt force trauma", "strangulation", "poisoning", "severe beating"]
        
        # Murder circumstances
        self.crime_circumstances = ["home invasion", "robbery gone wrong", "domestic dispute", "drug deal", "gang conflict", "hate crime"]
        
    def _generate_case_id(self, idx):
        """Generate unique case ID with format CR-YYYY-XXXXX"""
        year = np.random.randint(2018, 2023)
        return f"CR-{year}-{idx+10000}"
    
    def _generate_filing_date(self):
        """Generate a realistic filing date within the last 3 years"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=3*365)  # 3 years back
        random_days = np.random.randint(0, (end_date - start_date).days)
        return start_date + timedelta(days=random_days)
    
    def _generate_crime_type(self):
        """Generate crime type based on defined distributions"""
        return np.random.choice(
            list(self.crime_types.keys()),
            p=list(self.crime_types.values())
        )
    
    def _generate_location(self):
        """Generate location based on defined distributions"""
        return np.random.choice(
            list(self.locations.keys()),
            p=list(self.locations.values())
        )
    
    def _generate_description(self, crime_type):
        """Generate a realistic description based on crime type"""
        # Choose a random template for the crime type
        template = random.choice(self.description_templates[crime_type])
        
        # Replace placeholders with random values depending on crime type
        if crime_type == 'Theft':
            return template.format(
                item=random.choice(self.items),
                location_detail=random.choice(self.location_details),
                activity=random.choice(self.activities)
            )
        elif crime_type == 'Burglary':
            return template.format(
                entry_point=random.choice(self.entry_points),
                items_stolen=random.choice(self.items_stolen_groups)
            )
        elif crime_type == 'Assault':
            return template.format(
                injuries=random.choice(self.injuries),
                location_detail=random.choice(self.location_details),
                suspect_description=random.choice(self.suspect_descriptions)
            )
        elif crime_type == 'Fraud':
            return template.format(
                amount=random.choice(self.fraud_amounts),
                accounts=random.choice(self.account_types)
            )
        elif crime_type == 'Drug Offense':
            return template.format(
                drug_type=random.choice(self.drug_types),
                location_detail=random.choice(self.location_details)
            )
        elif crime_type == 'Vandalism':
            return template.format(
                graffiti_content=random.choice(self.graffiti_content),
                amount=random.randint(100, 2000),
                vehicle_part=random.choice(self.vehicle_parts),
                location_detail=random.choice(self.location_details),
                property_type=random.choice(self.property_types),
                tool=random.choice(self.vandalism_tools)
            )
        elif crime_type == 'Robbery':
            return template.format(
                amount=random.randint(50, 2000),
                num_suspects=random.randint(1, 3),
                items_stolen=random.choice(self.items),
                weapon=random.choice(self.weapons)
            )
        elif crime_type == 'Harassment':
            return template.format(
                num_incidents=random.randint(3, 20),
                time_period=f"{random.randint(1, 6)} months",
                harassment_actions=random.choice(self.harassment_actions),
                harassment_content=random.choice(self.harassment_content)
            )
        elif crime_type == 'Homicide':
            return template.format(
                cause_of_death=random.choice(self.causes_of_death),
                location_detail=random.choice(self.location_details)
            )
        elif crime_type == 'Murder':
            return template.format(
                weapon=random.choice(self.weapons),
                cause_of_death=random.choice(self.causes_of_death),
                crime_circumstance=random.choice(self.crime_circumstances)
            )
        elif crime_type == 'Kidnapping':
            return template.format(
                num_suspects=random.randint(1, 3),
                location_detail=random.choice(self.location_details)
            )
        else:  # Other
            return template.format(
                location_detail=random.choice(self.location_details)
            )
    
    def _determine_priority(self, crime_type, filing_date, description):
        """
        Determine the priority of a case based on various factors:
        - Severity of crime type (violent crimes get higher priority)
        - Recency (more recent cases get higher priority)
        - Certain keywords in description
        """
        # Crime type severity (0-11 scale, with Murder having the highest priority)
        severity_map = {
            'Murder': 11,    # Murder gets highest priority
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
        # Normalize to 0-1 scale
        severity_score = min(severity_map.get(crime_type, 1) / 11, 1.0)
        
        # Recency score (0-1)
        days_since_filing = (datetime.now().date() - filing_date).days
        max_days = 3 * 365  # 3 years
        recency_score = 1 - min(days_since_filing / max_days, 1)
        
        # Description keyword score (0-1)
        high_priority_keywords = [
            'weapon', 'gun', 'knife', 'armed', 'violent', 'injured', 'child', 
            'elderly', 'vulnerable', 'hospital', 'emergency', 'severe', 'death', 
            'threat', 'blood', 'trauma', 'victim', 'wounds', 'attack',
            'murder', 'killed', 'serial', 'premeditated', 'execution', 'brutal',
            'homicide', 'strangled', 'gunshot', 'stabbed', 'beaten', 'poisoned'
        ]
        
        keyword_matches = sum(1 for keyword in high_priority_keywords if keyword.lower() in description.lower())
        keyword_score = min(keyword_matches / 5, 1)  # Cap at 1 with 5 or more matches
        
        # Calculate final priority score
        priority_score = 0.5 * severity_score + 0.3 * recency_score + 0.2 * keyword_score
        
        # Determine binary priority (threshold can be adjusted)
        return 1 if priority_score > 0.5 else 0
    
    def generate_dataset(self):
        """Generate the complete synthetic dataset"""
        data = []
        
        for i in range(self.num_samples):
            case_id = self._generate_case_id(i)
            filing_date = self._generate_filing_date()
            crime_type = self._generate_crime_type()
            location = self._generate_location()
            description = self._generate_description(crime_type)
            priority = self._determine_priority(crime_type, filing_date, description)
            
            data.append({
                'CaseID': case_id,
                'FilingDate': filing_date,
                'CrimeType': crime_type,
                'Location': location,
                'Description': description,
                'PriorityLabel': priority
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df
    
    def save_dataset(self, df, filename='crime_data_synthetic.xlsx'):
        """Save the generated dataset to an Excel file"""
        df.to_excel(filename, index=False)
        print(f"Dataset saved to {filename}")
        return filename

if __name__ == "__main__":
    # Test the data generator
    generator = DataGenerator(num_samples=1000)
    df = generator.generate_dataset()
    generator.save_dataset(df)
