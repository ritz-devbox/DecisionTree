"""
Script to generate a large training dataset for placement prediction
Creates realistic data with proper correlations and patterns

Usage: python generate_dataset.py
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of records to generate (change this to generate more/fewer records)
NUM_RECORDS = 1200

def generate_student_data(n_records):
    """Generate realistic student placement data"""
    data = []
    roll_start = 1234567
    
    for i in range(n_records):
        roll_no = roll_start + i
        
        # Aggregate: Normal distribution centered around 72% with std 12
        aggregate = np.random.normal(72, 12)
        aggregate = max(45, min(98, aggregate))  # Clamp between 45-98
        
        # Backlogs: 25% chance of having 1 backlog
        backlogs = 1 if np.random.random() < 0.25 else 0
        
        # 10th aggregate: Correlated with aggregate but slightly higher
        tenth_agg = aggregate + np.random.normal(8, 5)
        tenth_agg = max(50, min(98, tenth_agg))
        
        # 12th aggregate: Correlated with aggregate
        twelfth_agg = aggregate + np.random.normal(5, 5)
        twelfth_agg = max(50, min(98, twelfth_agg))
        
        # Workshops: Exponential distribution (most have 0-2, few have 4-5)
        workshops = int(np.random.exponential(1.5))
        workshops = min(5, max(0, workshops))
        
        # Languages: Exponential + 1 (most have 1-2, few have 4-5)
        languages = int(np.random.exponential(1.2) + 1)
        languages = min(5, max(1, languages))
        
        # Placement logic: Realistic patterns
        # High performers (agg >= 75) almost always placed
        # Medium-high (70-75, no backlogs, workshops) usually placed
        # Medium (65-70) sometimes placed
        # Low (< 65) rarely placed
        
        if aggregate >= 75:
            # High performers: 95% placed
            placed = "Placed" if np.random.random() < 0.95 else "Not Placed"
        elif aggregate >= 70 and backlogs == 0 and workshops >= 1:
            # Good candidates: 85% placed
            placed = "Placed" if np.random.random() < 0.85 else "Not Placed"
        elif aggregate >= 70 and backlogs == 0:
            # Good but no workshops: 70% placed
            placed = "Placed" if np.random.random() < 0.70 else "Not Placed"
        elif aggregate >= 65 and backlogs == 0:
            # Average performers: 50% placed
            placed = "Placed" if np.random.random() < 0.50 else "Not Placed"
        elif aggregate >= 65:
            # Average with backlogs: 25% placed
            placed = "Placed" if np.random.random() < 0.25 else "Not Placed"
        elif aggregate >= 60:
            # Below average: 15% placed
            placed = "Placed" if np.random.random() < 0.15 else "Not Placed"
        else:
            # Low performers: 5% placed
            placed = "Placed" if np.random.random() < 0.05 else "Not Placed"
        
        # Add some correlation: More workshops/languages helps placement
        if placed == "Not Placed" and workshops >= 3 and languages >= 3:
            # Strong extracurriculars can help
            if np.random.random() < 0.20:  # 20% chance to override
                placed = "Placed"
        
        data.append([
            roll_no,
            round(aggregate, 1),
            backlogs,
            round(tenth_agg, 1),
            round(twelfth_agg, 1),
            workshops,
            languages,
            placed
        ])
    
    return data

if __name__ == "__main__":
    # Generate the dataset
    print(f"Generating {NUM_RECORDS} student records...")
    data = generate_student_data(NUM_RECORDS)
    
    # Create DataFrame
    df = pd.DataFrame(
        data,
        columns=['RollNo', 'Aggregate', 'Backlogs', '10th_agg', '12th_agg', 
                 'Workshops', 'Languages', 'Target']
    )
    
    # Save to CSV
    df.to_csv('a.csv', index=False)
    
    # Print statistics
    print(f"\nGenerated {len(df)} records")
    print(f"\nDataset Statistics:")
    print(f"  - Placed: {len(df[df['Target'] == 'Placed'])} ({len(df[df['Target'] == 'Placed'])/len(df)*100:.1f}%)")
    print(f"  - Not Placed: {len(df[df['Target'] == 'Not Placed'])} ({len(df[df['Target'] == 'Not Placed'])/len(df)*100:.1f}%)")
    print(f"\nFeature Ranges:")
    print(f"  - Aggregate: {df['Aggregate'].min():.1f} - {df['Aggregate'].max():.1f} (mean: {df['Aggregate'].mean():.1f})")
    print(f"  - Backlogs: {df['Backlogs'].sum()} students have backlogs ({df['Backlogs'].mean()*100:.1f}%)")
    print(f"  - Workshops: {df['Workshops'].min()} - {df['Workshops'].max()} (mean: {df['Workshops'].mean():.2f})")
    print(f"  - Languages: {df['Languages'].min()} - {df['Languages'].max()} (mean: {df['Languages'].mean():.2f})")
    print(f"\nDataset saved to 'a.csv'")
