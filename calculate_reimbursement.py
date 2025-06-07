#!/usr/bin/env python3
import sys
import joblib
import pandas as pd

def engineer_features(trip_duration_days, miles_traveled, total_receipts_amount):
    """Engineer the 10 optimal features from the 3 original inputs"""
    features = {}
    
    # Original features
    features['trip_duration_days'] = trip_duration_days
    features['miles_traveled'] = miles_traveled
    features['total_receipts_amount'] = total_receipts_amount
    
    # Key interaction terms (most important discoveries)
    features['days_miles_interaction'] = trip_duration_days * miles_traveled
    features['days_receipts_interaction'] = trip_duration_days * total_receipts_amount
    
    # Efficiency metrics
    features['miles_per_day'] = miles_traveled / trip_duration_days
    features['spending_per_day'] = total_receipts_amount / trip_duration_days
    
    # Business rule discoveries from employee interviews
    features['lucky_rounding'] = int(((total_receipts_amount * 100) % 100) in [49, 99])
    features['low_receipt_penalty'] = int(total_receipts_amount < 50)
    features['vacation_trip'] = int(trip_duration_days >= 8)  # Actually a bonus!
    
    return features

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Parse command line arguments - handle both int and float values
        trip_duration_days = int(float(sys.argv[1]))  # Convert to float first, then int
        miles_traveled = float(sys.argv[2])           # Miles can be float
        total_receipts_amount = float(sys.argv[3])
        
        # Load the trained model
        model_data = joblib.load('acme_reimbursement_model.joblib')
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Engineer features
        features = engineer_features(trip_duration_days, miles_traveled, total_receipts_amount)
        
        # Create DataFrame with correct feature order
        X = pd.DataFrame([features])[feature_names]
        
        # Predict
        prediction = model.predict(X)[0]
        
        # Output just the reimbursement amount (rounded to 2 decimal places)
        print(f"{prediction:.2f}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 